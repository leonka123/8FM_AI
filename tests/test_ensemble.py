import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eightfm.baselines import GreedyPolicy, play_episode
from eightfm.ensemble import (
    EnsembleMember,
    EnsemblePolicy,
    _softmax,
    load_ensemble,
    members_from_tokens,
    resolve_member,
    save_ensemble,
)
from eightfm.env import ClassicEnv


class _OneHotScorer:
    """Test double that puts all mass on a single legal action."""

    def __init__(self, index: int, value: float = 50.0):
        self.index = index
        self.value = value

    def raw_scores(self, obs) -> np.ndarray:
        raw = np.zeros(len(obs.legal), dtype=np.float64)
        raw[self.index % len(obs.legal)] = self.value
        return raw


class SoftmaxTests(unittest.TestCase):
    def test_softmax_sums_to_one_and_is_shift_invariant(self):
        values = np.array([1.0, 2.0, 3.0])
        probabilities = _softmax(values)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=9)
        shifted = _softmax(values + 100.0)
        np.testing.assert_allclose(probabilities, shifted, atol=1e-12)

    def test_softmax_of_empty_is_empty(self):
        self.assertEqual(_softmax(np.array([])).size, 0)


class ResolveMemberTests(unittest.TestCase):
    def test_shorthands(self):
        self.assertEqual(resolve_member("lu_luna"), ("rl", "models/lu_luna.json"))
        self.assertEqual(resolve_member("classic-rl"), ("rl", "models/classic-rl.json"))
        self.assertEqual(
            resolve_member("supervised"), ("supervised", "models/classic-policy.json")
        )

    def test_kind_with_path(self):
        self.assertEqual(resolve_member("rl:models/x.json"), ("rl", "models/x.json"))

    def test_rejects_unknown(self):
        for token in ("telegram", "rl:", "magic:models/x.json"):
            with self.assertRaises(ValueError):
                resolve_member(token)


class MembersFromTokensTests(unittest.TestCase):
    def test_default_weights_follow_the_named_blend(self):
        members = members_from_tokens(["lu_luna", "classic-rl", "supervised"])
        self.assertEqual([m.weight for m in members], [0.5, 0.3, 0.2])
        self.assertEqual([m.temperature for m in members], [1.0, 1.0, 1.0])

    def test_explicit_weights_and_temperatures(self):
        members = members_from_tokens(
            ["lu_luna", "classic-rl"], weights=[0.9, 0.1], temperatures=[0.5, 2.0]
        )
        self.assertEqual([m.weight for m in members], [0.9, 0.1])
        self.assertEqual([m.temperature for m in members], [0.5, 2.0])

    def test_weight_count_must_match(self):
        with self.assertRaises(ValueError):
            members_from_tokens(["lu_luna", "classic-rl"], weights=[0.5])

    def test_rejects_negative_weight_and_bad_temperature(self):
        with self.assertRaises(ValueError):
            members_from_tokens(["lu_luna", "classic-rl"], weights=[-1.0, 2.0])
        with self.assertRaises(ValueError):
            members_from_tokens(["lu_luna", "classic-rl"], temperatures=[0.0, 1.0])

    def test_requires_at_least_one_member(self):
        with self.assertRaises(ValueError):
            members_from_tokens([])


class EnsembleBlendTests(unittest.TestCase):
    def setUp(self):
        self.obs = ClassicEnv(3, seed=1).observations()[0]
        self.assertTrue(len(self.obs.legal) >= 2)

    def _policy(self, weights, scorers):
        members = [
            EnsembleMember(name=f"m{i}", kind="rl", path="x", weight=w)
            for i, w in enumerate(weights)
        ]
        return EnsemblePolicy(members, seed=0, scorers=scorers)

    def test_blend_is_a_weighted_average_of_member_softmaxes(self):
        policy = self._policy([0.75, 0.25], [_OneHotScorer(0), _OneHotScorer(1)])
        legal, probs = policy.probabilities(self.obs)
        self.assertEqual(len(legal), len(self.obs.legal))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=9)
        self.assertAlmostEqual(float(probs[0]), 0.75, places=6)
        self.assertAlmostEqual(float(probs[1]), 0.25, places=6)

    def test_weights_decide_the_winner(self):
        lean_first = self._policy([0.75, 0.25], [_OneHotScorer(0), _OneHotScorer(1)])
        lean_second = self._policy([0.25, 0.75], [_OneHotScorer(0), _OneHotScorer(1)])
        legal, _ = lean_first.probabilities(self.obs)
        self.assertEqual(lean_first(self.obs), legal[0])
        self.assertEqual(lean_second(self.obs), legal[1])

    def test_temperature_softens_the_distribution(self):
        scorer = _OneHotScorer(1, value=5.0)
        sharp = EnsemblePolicy(
            [EnsembleMember("a", "rl", "x", 1.0, 0.5)], seed=0, scorers=[scorer]
        )
        soft = EnsemblePolicy(
            [EnsembleMember("a", "rl", "x", 1.0, 5.0)], seed=0, scorers=[scorer]
        )
        _, sharp_probs = sharp.probabilities(self.obs)
        _, soft_probs = soft.probabilities(self.obs)
        self.assertGreater(float(sharp_probs[1]), float(soft_probs[1]))
        self.assertLess(float(sharp_probs[0]), float(soft_probs[0]))


class EnsembleSerializationTests(unittest.TestCase):
    def test_save_and_load_roundtrip_without_loading_models(self):
        members = members_from_tokens(["lu_luna", "classic-rl", "supervised"])
        policy = EnsemblePolicy(members, seed=7, name="test-blend")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.json"
            save_ensemble(path, policy)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "eightfm-ensemble")
            restored = load_ensemble(path)
        self.assertEqual(restored.name, "test-blend")
        self.assertEqual(restored.seed, 7)
        self.assertEqual([m.name for m in restored.members], ["lu_luna", "classic-rl", "supervised"])
        self.assertEqual([m.weight for m in restored.members], [0.5, 0.3, 0.2])

    def test_load_rejects_foreign_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"format": "something-else"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_ensemble(path)


class EnsembleSimulationTests(unittest.TestCase):
    def test_ensemble_plays_a_full_game_in_the_simulator(self):
        members = [
            EnsembleMember(name="a", kind="rl", path="x", weight=0.5),
            EnsembleMember(name="b", kind="rl", path="y", weight=0.5),
        ]
        policy = EnsemblePolicy(members, seed=0, scorers=[_OneHotScorer(0), _OneHotScorer(0)])
        env = play_episode(3, [policy, GreedyPolicy(seed=1), GreedyPolicy(seed=2)], seed=11)
        self.assertTrue(env.done)
        self.assertEqual(len(env.scores), 3)


if __name__ == "__main__":
    unittest.main()
