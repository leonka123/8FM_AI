import unittest

from eightfm.baselines import RandomPolicy
from eightfm.model_bench import (
    RunningStats,
    benchmark_models,
    format_report,
    play_game,
)


class PlayGameTests(unittest.TestCase):
    def test_play_game_returns_one_score_and_rank_per_seat(self):
        policies = [RandomPolicy(seed=index) for index in range(3)]
        scores, ranks = play_game(3, seed=5, policies=policies)
        self.assertEqual(len(scores), 3)
        self.assertEqual(len(ranks), 3)
        self.assertEqual(min(ranks), 1)
        self.assertTrue(all(rank >= 1 for rank in ranks))

    def test_play_game_requires_one_policy_per_seat(self):
        with self.assertRaises(ValueError):
            play_game(3, seed=1, policies=[RandomPolicy(seed=0)])


class FormatReportTests(unittest.TestCase):
    @staticmethod
    def _stats(score: float, rank: int) -> dict:
        stats = RunningStats()
        stats.add(score, rank)
        return stats.to_dict()

    def test_format_report_uses_russian_plurals(self):
        models = {
            "rl": self._stats(55, 1),
            "supervised": self._stats(50, 2),
        }
        report = {
            "games_per_player_count": 1,
            "filler": "greedy",
            "seed_base": 1,
            "model": "model.json",
            "supervised_model": "supervised.json",
            "players": {
                "3": {
                    "games": 1,
                    "models": models,
                    "filler": self._stats(52, 2),
                    "head_to_head": {
                        "rl_wins": 1,
                        "supervised_wins": 0,
                        "ties": 0,
                    },
                }
            },
            "summary": {
                "games": 1,
                "models": models,
                "filler": self._stats(52, 2),
                "head_to_head": {
                    "rl_wins": 1,
                    "supervised_wins": 0,
                    "ties": 0,
                },
            },
        }
        text = format_report(report)
        self.assertIn("3 игрока", text)
        self.assertIn("1 партия на каждый размер стола", text)


class BenchmarkValidationTests(unittest.TestCase):
    def test_rejects_zero_games(self):
        with self.assertRaises(ValueError):
            benchmark_models(games=0)

    def test_rejects_unknown_filler(self):
        with self.assertRaises(ValueError):
            benchmark_models(filler="telegram")

    def test_rejects_unsupported_player_count(self):
        with self.assertRaises(ValueError):
            benchmark_models(player_counts=(2,))


if __name__ == "__main__":
    unittest.main()
