import tempfile
import unittest
from pathlib import Path

import numpy as np

from eightfm.baselines import GreedyPolicy
from eightfm.env import (
    ClassicEnv,
    Play,
    action_feature_size,
    state_feature_size,
)
from eightfm.rl import (
    Adam,
    RLGreedyPolicy,
    _pad_observations,
    behaviour_clone,
    clone_network,
    collect_bc_dataset,
    create_network,
    forward_batch,
    load_rl,
    masked_softmax,
    save_rl,
)
from eightfm.rl_train import (
    _composite_score,
    _value_loss_and_gradient,
    collect_episode,
    head_to_head_report,
    observation_from_message,
    rank_checkpoints,
    rank_message_actions,
    rl_update,
    train_absolute,
    train_lu_luna,
)


class NetworkTests(unittest.TestCase):
    def test_forward_shapes_and_masked_probabilities(self):
        net = create_network(seed=3)
        env = ClassicEnv(5, seed=1)
        states, actions, mask, legal = _pad_observations(
            env.observations(), net.feature_version
        )
        self.assertEqual(states.shape, (5, state_feature_size(net.feature_version)))
        self.assertEqual(actions.shape[0], 5)
        self.assertEqual(actions.shape[2], action_feature_size(net.feature_version))
        self.assertEqual(mask.shape[0], 5)
        scores, values, _ = forward_batch(net, states, actions, mask)
        self.assertEqual(scores.shape, (5, mask.shape[1]))
        self.assertEqual(values.shape, (5,))
        probabilities = masked_softmax(scores, mask)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)
        self.assertTrue(np.all(probabilities[~mask] == 0.0))
        for row, legal_actions in zip(probabilities, legal):
            self.assertEqual(int(np.count_nonzero(row)), len(legal_actions))

    def test_save_and_load_roundtrip(self):
        net = create_network(seed=5)
        net.metadata["note"] = "roundtrip"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_rl(net, path)
            loaded = load_rl(path)
        np.testing.assert_allclose(net.W1, loaded.W1)
        np.testing.assert_allclose(net.wv, loaded.wv)
        self.assertEqual(loaded.metadata["note"], "roundtrip")

    def test_feature_version_layouts_roundtrip(self):
        for version in (1, 2):
            net = create_network(seed=version, feature_version=version)
            self.assertEqual(net.feature_version, version)
            self.assertEqual(
                net.W1.shape[0], state_feature_size(version)
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.json"
                save_rl(net, path)
                loaded = load_rl(path)
            self.assertEqual(loaded.feature_version, version)
            self.assertEqual(loaded.W1.shape, net.W1.shape)
            np.testing.assert_allclose(net.W1, loaded.W1)
        # The richer layout must expose strictly more inputs than v1.
        self.assertGreater(state_feature_size(2), state_feature_size(1))
        self.assertGreater(action_feature_size(2), action_feature_size(1))

    def test_absolute_preset_is_the_maximal_recipe(self):
        import inspect

        signature = inspect.signature(train_absolute)
        params = signature.parameters
        self.assertEqual(params["feature_version"].default, 2)
        self.assertEqual(params["updates"].default, 3000)
        self.assertEqual(params["games_per_update"].default, 48)
        self.assertEqual(params["hidden1"].default, 512)
        self.assertEqual(params["hidden2"].default, 256)
        self.assertEqual(params["action_hidden"].default, 256)
        net = create_network(seed=1, feature_version=2, hidden1=512, hidden2=256, action_hidden=256)
        self.assertEqual(net.W1.shape, (158, 512))
        self.assertEqual(net.W2.shape, (512, 256))

    def test_greedy_policy_returns_a_legal_action(self):
        net = create_network(seed=2)
        env = ClassicEnv(6, seed=9)
        policy = RLGreedyPolicy(net)
        for observation in env.observations():
            self.assertIn(policy(observation), observation.legal)

    def test_clone_network_is_independent(self):
        net = create_network(seed=4)
        clone = clone_network(net)
        clone.W1[0, 0] += 1.0
        self.assertNotEqual(net.W1[0, 0], clone.W1[0, 0])
        np.testing.assert_allclose(net.W2, clone.W2)


class EpisodeTests(unittest.TestCase):
    def test_self_play_episode_has_one_decision_per_seat_and_turn(self):
        net = create_network(seed=7)
        episode = collect_episode(
            net,
            4,
            seed=12,
            rng=np.random.default_rng(3),
        )
        self.assertEqual(episode.states.shape[0], 10 * 4)
        self.assertEqual(episode.old_log_probs.shape, (10 * 4,))
        self.assertEqual(len(episode.terminal_rewards), 4)
        self.assertEqual(len(episode.scores), 4)
        self.assertEqual(len(episode.ranks), 4)
        self.assertTrue(np.all(np.isfinite(episode.returns)))
        self.assertTrue(np.all(episode.mask[np.arange(len(episode.chosen)), episode.chosen]))
        self.assertEqual(episode.old_values.shape, (10 * 4,))
        self.assertTrue(np.all(np.isfinite(episode.old_values)))

    def test_league_episode_only_records_learner_seats(self):
        net = create_network(seed=8)
        episode = collect_episode(
            net,
            5,
            seed=13,
            rng=np.random.default_rng(4),
            opponent_policies=[
                None,
                GreedyPolicy(seed=1),
                GreedyPolicy(seed=2),
                GreedyPolicy(seed=3),
                GreedyPolicy(seed=4),
            ],
            rl_seats=[0],
        )
        self.assertEqual(episode.states.shape[0], 10)
        self.assertTrue(np.all(episode.player_ids == 0))
        self.assertEqual(len(episode.terminal_rewards), 1)
        self.assertEqual(len(episode.scores), 1)

    def test_ppo_update_returns_finite_metrics(self):
        net = create_network(seed=11)
        episodes = [
            collect_episode(net, 3, seed=20, rng=np.random.default_rng(1)),
            collect_episode(net, 3, seed=21, rng=np.random.default_rng(2)),
        ]
        optimizer = Adam(net.parameters, learning_rate=1e-3)
        metrics = rl_update(
            net,
            episodes,
            optimizer,
            batch_size=64,
            ppo_epochs=1,
            seed=5,
        )
        self.assertEqual(metrics["decisions"], 60)
        for key in ("policy_loss", "value_loss", "entropy", "grad_norm", "clip_fraction"):
            self.assertTrue(np.isfinite(metrics[key]), key)
        self.assertGreaterEqual(metrics["clip_fraction"], 0.0)
        self.assertLessEqual(metrics["clip_fraction"], 1.0)

    def test_ppo_update_supports_value_clip_huber_and_kl_stop(self):
        net = create_network(seed=21)
        episodes = [
            collect_episode(net, 4, seed=30, rng=np.random.default_rng(1)),
            collect_episode(net, 4, seed=31, rng=np.random.default_rng(2)),
        ]
        optimizer = Adam(net.parameters, learning_rate=1e-3)
        metrics = rl_update(
            net,
            episodes,
            optimizer,
            batch_size=64,
            ppo_epochs=4,
            seed=6,
            value_clip=0.2,
            huber_delta=1.0,
            target_kl=0.5,
        )
        for key in ("approx_kl", "value_clip_fraction", "ppo_epochs_run", "early_stopped"):
            self.assertTrue(np.isfinite(metrics[key]), key)
        self.assertGreaterEqual(metrics["ppo_epochs_run"], 1.0)
        self.assertLessEqual(metrics["ppo_epochs_run"], 4.0)
        self.assertGreaterEqual(metrics["value_clip_fraction"], 0.0)
        self.assertLessEqual(metrics["value_clip_fraction"], 1.0)

    def test_value_loss_gradient_matches_finite_difference(self):
        values = np.array([0.5, -0.2, 1.4], dtype=np.float32)
        returns = np.array([0.4, 0.1, 0.2], dtype=np.float32)
        old_values = np.array([0.5, -0.2, 0.3], dtype=np.float32)
        _loss, gradient = _value_loss_and_gradient(
            values, returns, old_values, huber_delta=None, value_clip=0.2
        )
        epsilon = 1e-3
        for index in range(len(values)):
            shifted = values.copy()
            shifted[index] += epsilon
            loss_plus, _ = _value_loss_and_gradient(
                shifted, returns, old_values, huber_delta=None, value_clip=0.2
            )
            shifted[index] -= 2 * epsilon
            loss_minus, _ = _value_loss_and_gradient(
                shifted, returns, old_values, huber_delta=None, value_clip=0.2
            )
            numeric = (float(np.mean(loss_plus)) - float(np.mean(loss_minus))) / (2 * epsilon)
            self.assertAlmostEqual(
                float(gradient[index]) / len(values), numeric, places=3
            )

    def test_behaviour_cloning_runs_on_a_tiny_dataset(self):
        net = create_network(seed=13)
        dataset = collect_bc_dataset(games=1, player_counts=(3,), seed=2)
        self.assertEqual(dataset.size, 30)
        metrics = behaviour_clone(net, dataset, epochs=1, batch_size=32, seed=3)
        self.assertTrue(np.isfinite(metrics["bc_loss"]))
        self.assertGreaterEqual(metrics["bc_accuracy"], 0.0)
        self.assertLessEqual(metrics["bc_accuracy"], 1.0)


class LuLunaTests(unittest.TestCase):
    def test_train_lu_luna_writes_artifacts_with_a_tiny_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "lu_luna.json"
            _net, metrics = train_lu_luna(
                updates=1,
                games_per_update=1,
                player_counts=(3,),
                warm_start_games=0,
                hidden1=16,
                hidden2=8,
                action_hidden=8,
                eval_every=9999,
                eval_games=1,
                selection_opponent_path=None,
                head_to_head_games=1,
                checkpoint_every=0,
                checkpoint_dir=None,
                verbose=False,
                output=output,
                latest_path=root / "latest.json",
                metrics_path=root / "metrics.json",
                plot_path=root / "plot.png",
            )
            self.assertTrue(output.exists())
            self.assertEqual(metrics["config"]["hidden1"], 16)
            self.assertEqual(metrics["config"]["value_clip"], 0.2)
            self.assertEqual(len(metrics["updates"]), 1)
            self.assertIn("approx_kl", metrics["updates"][0])
            self.assertGreaterEqual(metrics["best_update"], 1)

    def test_head_to_head_report_returns_win_rate_and_chance(self):
        net = create_network(seed=31)
        rival = create_network(seed=32)
        report = head_to_head_report(
            net, rival, player_counts=(3, 4), games=2, seed_base=1
        )
        self.assertGreaterEqual(report["win_rate"], 0.0)
        self.assertLessEqual(report["win_rate"], 1.0)
        self.assertAlmostEqual(report["chance"], (1 / 3 + 1 / 4) / 2)

    def test_composite_score_rewards_beating_the_champion(self):
        base = {
            "summary": {
                "supervised_win_rate": 0.75,
                "greedy_win_rate": 0.40,
                "combo_win_rate": 0.80,
            }
        }
        with_edge = {
            "summary": {
                **base["summary"],
                "head_to_head_win_rate": 0.30,
                "head_to_head_chance": 0.20,
            }
        }
        without = _composite_score(base)
        self.assertAlmostEqual(
            without, 0.45 * 0.75 + 0.30 * 0.40 + 0.25 * 0.80
        )
        self.assertGreater(_composite_score(with_edge), without)

    def test_rank_checkpoints_sorts_by_composite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed in (41, 42):
                path = root / f"model-{seed}.json"
                save_rl(create_network(seed), path)
                paths.append(path)
            ranked = rank_checkpoints(
                paths, player_counts=(3,), games=1, seed_base=1
            )
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0]["score"], ranked[1]["score"])
        self.assertIn("greedy_win_rate", ranked[0]["summary"])


class MessageTests(unittest.TestCase):
    def test_observation_from_real_message(self):
        message = Path(__file__).with_name("message-4734.txt").read_text(encoding="utf-8")
        observation, warnings = observation_from_message(message, "Леончик")
        self.assertEqual(observation.player_count, 5)
        self.assertEqual(observation.round_number, 2)
        self.assertEqual(observation.turn_number, 1)
        self.assertEqual(len(observation.hand), 6)
        self.assertGreater(len(observation.legal), 0)
        self.assertEqual(warnings, [])

    def test_observation_includes_round_one_history_at_round_two(self):
        message = Path(__file__).with_name("message-4734.txt").read_text(encoding="utf-8")
        observation, warnings = observation_from_message(message, "Леончик")
        self.assertEqual(observation.round_number, 2)
        self.assertEqual(observation.turn_number, 1)
        self.assertEqual(
            observation.own_plays,
            (
                Play(None, True),
                Play("sergay", False),
                Play("vanya", False),
                Play("detector", False),
                Play(None, True),
            ),
        )
        self.assertEqual(observation.closed_total, 9)
        self.assertEqual(warnings, [])

    def test_observation_includes_both_rounds(self):
        message = Path(__file__).with_name("message-4784-round2.txt").read_text(encoding="utf-8")
        observation, warnings = observation_from_message(message, "Ivan")
        self.assertEqual(observation.round_number, 2)
        self.assertEqual(observation.turn_number, 4)
        self.assertEqual(
            observation.own_plays,
            (
                Play("akim", False),
                Play("sergay", False),
                Play("akim", False),
                Play(None, True),
                Play(None, True),
                Play("vanya", False),
                Play("olympiad", False),
                Play("sergay", False),
            ),
        )
        self.assertEqual(observation.closed_total, 6)
        self.assertEqual(warnings, [])

    def test_rank_message_actions_is_sorted_and_normalised(self):
        message = Path(__file__).with_name("message-4734.txt").read_text(encoding="utf-8")
        net = create_network(seed=17)
        result = rank_message_actions(net, message, "Леончик")
        self.assertEqual(result["player_name"], "Леончик")
        self.assertEqual(result["warnings"], [])
        self.assertTrue(np.isfinite(result["value"]))
        candidates = result["candidates"]
        self.assertGreater(len(candidates), 0)
        self.assertAlmostEqual(sum(item["probability"] for item in candidates), 1.0, places=5)
        scores = [item["policy_score"] for item in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
