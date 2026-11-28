import random
import unittest
from collections import Counter

import numpy as np

from eightfm.env import (
    ClassicEnv,
    action_features,
    state_features,
)
from eightfm.rules import (
    Action,
    has_open_alexey,
    score_action_delta,
    score_player,
)

try:
    import torch

    from eightfm.torch_env import (
        ACTION_SIZE,
        CARD_INDEX,
        BatchedClassicEnv,
        score_batched,
    )
except ImportError:  # pragma: no cover - torch is optional
    torch = None


def _action_index(action: Action) -> int:
    return CARD_INDEX[action.card_id] * 2 + (1 if action.is_closed else 0)


@unittest.skipIf(torch is None, "torch is not installed")
class BatchedScoringTests(unittest.TestCase):
    def test_score_batched_matches_score_player(self):
        rng = random.Random(7)
        from eightfm.model import CARD_IDS

        for player_count in (3, 4, 5, 6, 7):
            opens = []
            closed = []
            plays = []
            for _player in range(player_count):
                row = {card: 0 for card in CARD_IDS}
                history = []
                for _ in range(5):
                    card = rng.choice(list(CARD_IDS))
                    if rng.random() < 0.35:
                        history.append((None, True))
                        closed.append(1)
                    else:
                        row[card] += 1
                        history.append((card, False))
                        closed.append(0)
                plays.append(history)
                opens.append([row[card] for card in CARD_IDS])
            alexey_players = sum(1 for row in opens if row[CARD_INDEX["alaksiej"]] > 0)
            open_tensor = torch.tensor([opens], dtype=torch.float32)
            closed_tensor = torch.tensor(
                [[sum(1 for _, is_closed in history if is_closed) for history in plays]],
                dtype=torch.float32,
            )
            expected = [
                score_player(history, alexey_players) for history in plays
            ]
            actual = score_batched(
                open_tensor, closed_tensor, torch.tensor([float(alexey_players)])
            )[0].tolist()
            np.testing.assert_allclose(actual, expected, atol=1e-5)


@unittest.skipIf(torch is None, "torch is not installed")
class BatchedFeatureParityTests(unittest.TestCase):
    def _compare(self, env, batch, version):
        observations = env.observations()
        batched = batch.observation(version)
        for player, obs in enumerate(observations):
            expected_state = np.asarray(state_features(obs, version), dtype=np.float32)
            actual_state = batched.state_features[0, player].cpu().numpy()
            np.testing.assert_allclose(actual_state, expected_state, atol=2e-5, rtol=2e-5)

            expected_legal = {_action_index(action) for action in obs.legal}
            actual_legal = set(
                int(index)
                for index in batched.legal[0, player].cpu().nonzero(as_tuple=False).flatten()
            )
            self.assertEqual(actual_legal, expected_legal)

            for action in obs.legal:
                index = _action_index(action)
                expected_action = np.asarray(
                    action_features(obs, action, version), dtype=np.float32
                )
                actual_action = batched.action_features[0, player, index].cpu().numpy()
                np.testing.assert_allclose(
                    actual_action, expected_action, atol=2e-5, rtol=2e-5
                )
                expected_delta = score_action_delta(
                    obs.own_plays,
                    action,
                    own_has_alexey=has_open_alexey(obs.own_plays),
                    players_with_alexey=len(obs.alexey_players),
                )
                actual_delta = float(batched.delta[0, player, index])
                self.assertAlmostEqual(actual_delta, expected_delta, places=4)

    def test_features_match_on_random_positions(self):
        for player_count in (3, 4, 5, 6, 7):
            for version in (1, 2, 3, 4):
                for seed in (11, 23):
                    env = ClassicEnv(player_count, seed=seed)
                    batch = BatchedClassicEnv(player_count, 1, seed=seed)
                    batch.load_from_classic(env)
                    rng = random.Random(seed + 1000)
                    while not env.done:
                        self._compare(env, batch, version)
                        observations = env.observations()
                        actions = [rng.choice(obs.legal) for obs in observations]
                        indices = torch.tensor(
                            [[_action_index(action) for action in actions]],
                            dtype=torch.long,
                        )
                        batch.step(indices)
                        env.step(actions)


@unittest.skipIf(torch is None, "torch is not installed")
class BatchedDynamicsTests(unittest.TestCase):
    def test_step_matches_classic_env(self):
        for player_count in (3, 4, 5, 6, 7):
            for seed in (5, 17):
                env = ClassicEnv(player_count, seed=seed)
                batch = BatchedClassicEnv(player_count, 1, seed=seed)
                batch.load_from_classic(env)
                rng = random.Random(seed + 2000)
                steps = 0
                while not env.done:
                    observations = env.observations()
                    actions = [rng.choice(obs.legal) for obs in observations]
                    indices = torch.tensor(
                        [[_action_index(action) for action in actions]],
                        dtype=torch.long,
                    )
                    old_scores = list(env.scores)
                    rewards, done = batch.step(indices)
                    env.step(actions)
                    steps += 1
                    self.assertEqual(done, env.done)

                    np.testing.assert_allclose(
                        batch.scores[0].cpu().numpy(), np.asarray(env.scores), atol=1e-5
                    )
                    deltas = [
                        env.scores[index] - old_scores[index]
                        for index in range(player_count)
                    ]
                    mean_delta = sum(deltas) / player_count
                    expected_rewards = [
                        max(-1.0, min(1.0, 0.08 * (delta - mean_delta)))
                        for delta in deltas
                    ]
                    np.testing.assert_allclose(
                        rewards[0].cpu().numpy(),
                        np.asarray(expected_rewards),
                        atol=1e-5,
                    )
                    for player in range(player_count):
                        expected_hand = Counter(env.hands[player])
                        actual_hand = {
                            card: int(batch.hand[0, player, CARD_INDEX[card]])
                            for card in CARD_INDEX
                            if int(batch.hand[0, player, CARD_INDEX[card]]) > 0
                        }
                        self.assertEqual(
                            actual_hand,
                            {card: count for card, count in expected_hand.items() if count > 0},
                        )
                        expected_open = Counter(
                            card
                            for card, is_closed in env.played[player]
                            if not is_closed and card
                        )
                        actual_open = {
                            card: int(batch.open[0, player, CARD_INDEX[card]])
                            for card in CARD_INDEX
                            if int(batch.open[0, player, CARD_INDEX[card]]) > 0
                        }
                        self.assertEqual(
                            actual_open,
                            {card: count for card, count in expected_open.items() if count > 0},
                        )
                        expected_closed = sum(
                            1 for _, is_closed in env.played[player] if is_closed
                        )
                        self.assertEqual(
                            int(batch.closed[0, player]), expected_closed
                        )
                    self.assertEqual(
                        int(batch.closed_total[0]), env.closed_total
                    )
                    self.assertEqual(batch.round_number, env.round_number)
                    self.assertEqual(batch.turn_number, env.turn_number)
                self.assertEqual(steps, 10)
