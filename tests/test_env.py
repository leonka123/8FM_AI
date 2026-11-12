import random
import unittest
from collections import Counter

from eightfm.env import (
    ClassicEnv,
    _unseen_counts,
    action_feature_size,
    action_features,
    state_feature_size,
    state_features,
)
from eightfm.rules import Action, score_players


class EnvironmentTests(unittest.TestCase):
    def test_reset_deals_six_cards_to_every_player(self):
        for player_count in (3, 4, 5, 6, 7):
            env = ClassicEnv(player_count, seed=7)
            self.assertEqual(len(env.hands), player_count)
            self.assertTrue(all(len(hand) == 6 for hand in env.hands))
            self.assertEqual(sum(len(hand) for hand in env.hands), 6 * player_count)
            self.assertEqual((env.round_number, env.turn_number), (1, 1))
            self.assertFalse(env.done)

    def test_round_one_passes_hands_to_the_previous_seat(self):
        env = ClassicEnv(5, seed=3)
        original = [list(hand) for hand in env.hands]
        observations = env.observations()
        actions = [obs.legal[0] for obs in observations]
        env.step(actions)
        for player in range(5):
            receiver = (player - 1) % 5
            expected = Counter(original[player])
            expected[actions[player].card_id] -= 1
            self.assertEqual(Counter(env.hands[receiver]), expected)

    def test_round_two_passes_hands_to_the_next_seat(self):
        env = ClassicEnv(4, seed=11)
        rng = random.Random(5)
        while not (env.round_number == 2 and env.turn_number == 1):
            observations = env.observations()
            env.step([rng.choice(obs.legal) for obs in observations])
        original = [list(hand) for hand in env.hands]
        observations = env.observations()
        actions = [obs.legal[0] for obs in observations]
        env.step(actions)
        for player in range(4):
            receiver = (player + 1) % 4
            expected = Counter(original[player])
            expected[actions[player].card_id] -= 1
            self.assertEqual(Counter(env.hands[receiver]), expected)

    def test_full_game_has_two_rounds_and_consistent_scores(self):
        for player_count in (3, 4, 5, 6, 7):
            env = ClassicEnv(player_count, seed=100 + player_count)
            rng = random.Random(17)
            steps = 0
            while not env.done:
                observations = env.observations()
                env.step([rng.choice(obs.legal) for obs in observations])
                steps += 1
                if steps == 5:
                    self.assertEqual((env.round_number, env.turn_number), (2, 1))
                    self.assertTrue(all(len(hand) == 6 for hand in env.hands))
            self.assertEqual(steps, 10)
            self.assertEqual(env.scores, score_players(env.played, env.alexey_players))
            ranks = env.final_ranks()
            self.assertEqual(len(ranks), player_count)
            self.assertTrue(all(1 <= rank <= player_count for rank in ranks))
            rewards = env.terminal_rewards()
            self.assertEqual(len(rewards), player_count)
            self.assertTrue(all(reward == reward for reward in rewards))

    def test_same_seed_produces_the_same_game(self):
        first = ClassicEnv(6, seed=99)
        second = ClassicEnv(6, seed=99)
        self.assertEqual(first.hands, second.hands)
        rng = random.Random(1)
        while not first.done:
            first_observations = first.observations()
            second_observations = second.observations()
            self.assertEqual(
                [obs.legal for obs in first_observations],
                [obs.legal for obs in second_observations],
            )
            actions = [rng.choice(obs.legal) for obs in first_observations]
            first.step(actions)
            second.step(actions)
        self.assertEqual(first.scores, second.scores)

    def test_observation_features_have_stable_sizes(self):
        env = ClassicEnv(5, seed=2)
        observation = env.observations()[0]
        state = state_features(observation)
        self.assertEqual(len(state), state_feature_size())
        for action in observation.legal:
            features = action_features(observation, action)
            self.assertEqual(len(features), action_feature_size())
            self.assertEqual(features[14], 1.0 if action.is_closed else 0.0)

    def test_unseen_pool_does_not_include_opponents_hands(self):
        env = ClassicEnv(4, seed=8)
        observation = env.observations()[0]
        unseen = _unseen_counts(observation)
        own_hand = Counter(observation.hand)
        deck = dict(observation.deck_counts)
        self.assertEqual(sum(unseen.values()), 12 * 4 - len(observation.hand))
        for card_id, count in deck.items():
            self.assertEqual(unseen[card_id], count - own_hand[card_id])

    def test_illegal_open_marat_is_rejected(self):
        env = ClassicEnv(3, seed=4)
        env.hands[0] = ["marat", "akim"]
        env.played[0] = [("alaksiej", False)]
        env.alexey_players = {0}
        observations = env.observations()
        actions = [Action("marat", False)]
        actions.extend(obs.legal[0] for obs in observations[1:])
        with self.assertRaises(ValueError):
            env.step(actions)

    def test_terminal_reward_prefers_the_winner_and_success(self):
        env = ClassicEnv(3, seed=1)
        env.done = True
        env.scores = [70.0, 50.0, 40.0]
        rewards = env.terminal_rewards()
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])
        self.assertGreater(rewards[0], 1.5)
        without_success = env.terminal_rewards(success_threshold=100.0)
        self.assertGreater(rewards[0], without_success[0])


if __name__ == "__main__":
    unittest.main()
