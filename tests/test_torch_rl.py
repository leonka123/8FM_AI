import tempfile
import unittest
from pathlib import Path

import numpy as np

from eightfm.env import action_feature_size, state_feature_size
from eightfm.rl import create_network, forward_batch, load_rl

try:
    import torch

    from eightfm.torch_env import ACTION_SIZE, BatchedClassicEnv
    from eightfm.torch_rl import (
        BatchedNetPolicy,
        TorchActorCritic,
        behaviour_clone_torch,
        collect_bc_dataset_torch,
        collect_torch_rollout,
        head_to_head_report_torch,
        to_rl_network,
        torch_rl_update,
        train_lu_astra,
        train_lu_sol_torch,
    )
except ImportError:  # pragma: no cover - torch is optional
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class TorchNetworkTests(unittest.TestCase):
    def test_export_matches_numpy_forward(self):
        rl = create_network(
            seed=3, hidden1=32, hidden2=16, action_hidden=16, feature_version=2
        )
        model = TorchActorCritic.from_rl_network(rl)
        exported = to_rl_network(model)
        for name in ("W1", "b1", "W2", "b2", "Wa", "ba", "Wp", "wv", "bp", "bv"):
            np.testing.assert_allclose(
                getattr(exported, name), getattr(rl, name), atol=1e-6
            )

        rng = np.random.default_rng(7)
        states = rng.normal(size=(5, state_feature_size(2))).astype(np.float32)
        actions = rng.normal(size=(5, ACTION_SIZE, action_feature_size(2))).astype(np.float32)
        mask = np.ones((5, ACTION_SIZE), dtype=bool)
        expected_scores, expected_values, _ = forward_batch(rl, states, actions, mask)
        with torch.no_grad():
            actual_scores, actual_values = model(
                torch.from_numpy(states), torch.from_numpy(actions)
            )
        np.testing.assert_allclose(
            actual_scores.numpy(), expected_scores, atol=1e-4, rtol=1e-4
        )
        np.testing.assert_allclose(
            actual_values.numpy(), expected_values, atol=1e-4, rtol=1e-4
        )


@unittest.skipIf(torch is None, "torch is not installed")
class TorchRolloutTests(unittest.TestCase):
    def test_self_play_rollout_shapes(self):
        env = BatchedClassicEnv(4, 3, seed=1)
        model = TorchActorCritic(
            state_feature_size(2), action_feature_size(2), hidden1=32, hidden2=16,
            action_hidden=16, feature_version=2,
        )
        seat_types = torch.zeros((3, 4), dtype=torch.long)
        generator = torch.Generator()
        generator.manual_seed(5)
        rollout = collect_torch_rollout(
            env, model, seat_types, {}, generator=generator
        )
        self.assertEqual(rollout.decisions, 3 * 4 * 10)
        self.assertEqual(
            rollout.states.shape,
            (rollout.decisions, state_feature_size(2)),
        )
        self.assertEqual(
            rollout.actions.shape,
            (rollout.decisions, ACTION_SIZE, action_feature_size(2)),
        )
        self.assertEqual(rollout.mask.shape, (rollout.decisions, ACTION_SIZE))
        for tensor in (rollout.returns, rollout.advantages, rollout.log_probs):
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertEqual(rollout.scores.numel(), 3 * 4)
        self.assertEqual(rollout.ranks.numel(), 3 * 4)

    def test_ppo_update_runs(self):
        env = BatchedClassicEnv(3, 4, seed=2)
        model = TorchActorCritic(
            state_feature_size(2), action_feature_size(2), hidden1=32, hidden2=16,
            action_hidden=16, feature_version=2,
        )
        seat_types = torch.zeros((4, 3), dtype=torch.long)
        generator = torch.Generator()
        generator.manual_seed(9)
        rollout = collect_torch_rollout(
            env, model, seat_types, {}, generator=generator
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = torch_rl_update(
            model,
            rollout,
            optimizer,
            batch_size=64,
            ppo_epochs=2,
            entropy_coef=0.01,
        )
        self.assertEqual(metrics["decisions"], rollout.decisions)
        self.assertTrue(np.isfinite(metrics["policy_loss"]))
        self.assertTrue(np.isfinite(metrics["value_loss"]))
        self.assertTrue(np.isfinite(metrics["entropy"]))

    def test_behaviour_cloning_runs(self):
        dataset = collect_bc_dataset_torch(
            games=6, player_counts=(3, 4), seed=3, batch_size=3, feature_version=2
        )
        self.assertGreater(dataset.size, 0)
        model = TorchActorCritic(
            state_feature_size(2), action_feature_size(2), hidden1=32, hidden2=16,
            action_hidden=16, feature_version=2,
        )
        metrics = behaviour_clone_torch(
            model, dataset, epochs=1, batch_size=64, learning_rate=1e-3
        )
        self.assertTrue(np.isfinite(metrics["bc_loss"]))
        self.assertGreaterEqual(metrics["bc_accuracy"], 0.0)
        self.assertLessEqual(metrics["bc_accuracy"], 1.0)


@unittest.skipIf(torch is None, "torch is not installed")
class TorchChampionTests(unittest.TestCase):
    def test_v2_opponent_can_play_against_a_v4_observation(self):
        opponent_net = create_network(seed=1, feature_version=2)
        opponent = TorchActorCritic.from_rl_network(opponent_net, seed=2)
        policy = BatchedNetPolicy(opponent)
        env = BatchedClassicEnv(5, 2, seed=3)
        observation = env.observation(4)
        actions = policy.act(observation, torch.Generator().manual_seed(0))
        self.assertEqual(actions.shape, (2, 5))
        self.assertTrue(
            bool(observation.legal.gather(-1, actions.unsqueeze(-1)).all())
        )

    def test_batched_head_to_head_is_fair_and_complete(self):
        candidate_net = create_network(
            seed=3, feature_version=4, hidden1=64, hidden2=32, action_hidden=32
        )
        opponent_net = create_network(
            seed=4, feature_version=2, hidden1=64, hidden2=32, action_hidden=32
        )
        candidate = TorchActorCritic.from_rl_network(candidate_net, seed=5)
        report = head_to_head_report_torch(
            candidate, opponent_net, player_counts=(3, 4), games=4
        )
        self.assertAlmostEqual(report["chance"], (1 / 3 + 1 / 4) / 2)
        self.assertGreaterEqual(report["win_rate"], 0.0)
        self.assertLessEqual(report["win_rate"], 1.0)
        self.assertGreaterEqual(report["mean_score"], 0.0)


class TorchTrainingSmokeTests(unittest.TestCase):
    def test_tiny_training_writes_loadable_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, metrics = train_lu_sol_torch(
                updates=1,
                games_per_update=8,
                player_counts=(3,),
                seed=1,
                warm_start_games=4,
                bc_epochs=1,
                eval_every=0,
                eval_games=1,
                head_to_head_games=1,
                selection_opponent_path=None,
                checkpoint_every=0,
                checkpoint_dir=None,
                output=root / "lu_sol.json",
                latest_path=root / "lu_sol-latest.json",
                metrics_path=root / "lu_sol-metrics.json",
                plot_path=None,
                verbose=False,
            )
            self.assertTrue((root / "lu_sol.json").exists())
            self.assertTrue((root / "lu_sol-latest.json").exists())
            loaded = load_rl(root / "lu_sol.json")
            self.assertEqual(loaded.feature_version, 2)
            self.assertEqual(loaded.W1.shape, (state_feature_size(2), 512))
            self.assertEqual(loaded.Wa.shape, (action_feature_size(2), 256))
            self.assertEqual(metrics["backend"], "torch")
            self.assertEqual(metrics["best_update"], 1)
            self.assertEqual(len(metrics["updates"]), 1)

    def test_astra_preset_defaults_are_upsized(self):
        import inspect

        signature = inspect.signature(train_lu_astra)
        self.assertEqual(signature.parameters["hidden1"].default, 2048)
        self.assertEqual(signature.parameters["hidden2"].default, 1024)
        self.assertEqual(signature.parameters["action_hidden"].default, 1024)
        self.assertEqual(signature.parameters["feature_version"].default, 4)
        self.assertEqual(signature.parameters["games_per_update"].default, 1024)
        self.assertEqual(signature.parameters["ppo_epochs"].default, 6)
        self.assertEqual(signature.parameters["champion_league_weight"].default, 0.35)
        self.assertEqual(signature.parameters["head_to_head_games"].default, 100)
        self.assertTrue(signature.parameters["goal_selection"].default)
        self.assertGreater(
            signature.parameters["score_weight"].default,
            signature.parameters["placement_weight"].default,
        )

    def test_tiny_astra_training_writes_loadable_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, metrics = train_lu_astra(
                updates=1,
                games_per_update=8,
                player_counts=(3,),
                seed=1,
                warm_start_games=4,
                bc_epochs=1,
                hidden1=64,
                hidden2=32,
                action_hidden=32,
                eval_every=0,
                eval_games=1,
                head_to_head_games=1,
                selection_opponent_path=None,
                checkpoint_every=0,
                checkpoint_dir=None,
                output=root / "lu_astra.json",
                latest_path=root / "lu_astra-latest.json",
                metrics_path=root / "lu_astra-metrics.json",
                plot_path=None,
                verbose=False,
            )
            self.assertTrue((root / "lu_astra.json").exists())
            self.assertTrue((root / "lu_astra-latest.json").exists())
            loaded = load_rl(root / "lu_astra.json")
            self.assertEqual(loaded.feature_version, 4)
            self.assertEqual(loaded.W1.shape, (state_feature_size(4), 64))
            self.assertEqual(loaded.Wa.shape, (action_feature_size(4), 32))
            self.assertEqual(metrics["backend"], "torch")
            self.assertEqual(len(metrics["updates"]), 1)
