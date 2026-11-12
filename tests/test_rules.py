import unittest
from collections import Counter

from eightfm.rules import (
    ALEXEY_TOTALS,
    DECK_COUNTS,
    SUPPORTED_PLAYER_COUNTS,
    Action,
    deck_for_players,
    has_open_alexey,
    legal_actions,
    score_action_delta,
    score_player,
    score_players,
)


class DeckTests(unittest.TestCase):
    def test_deck_sizes_match_two_rounds_of_six_cards(self):
        for player_count in SUPPORTED_PLAYER_COUNTS:
            deck = deck_for_players(player_count)
            self.assertEqual(len(deck), 12 * player_count)
            self.assertEqual(Counter(deck), Counter(DECK_COUNTS[player_count]))

    def test_unsupported_player_count_is_rejected(self):
        with self.assertRaises(ValueError):
            deck_for_players(2)
        with self.assertRaises(ValueError):
            deck_for_players(8)


class ScoringTests(unittest.TestCase):
    def test_closed_card_is_two_points(self):
        self.assertEqual(score_player([(None, True)], 0), 2.0)

    def test_alexey_cumulative_curve(self):
        for copies, expected in enumerate(ALEXEY_TOTALS):
            plays = [("alaksiej", False)] * copies
            self.assertEqual(score_player(plays, 1 if copies else 0), float(expected))

    def test_olympiad_tiers(self):
        cases = {0: 0.0, 1: 2.0, 2: 4.0, 3: 12.0, 5: 20.0, 6: 36.0, 7: 42.0}
        for copies, expected in cases.items():
            plays = [("olympiad", False)] * copies
            self.assertEqual(score_player(plays, 0), expected)

    def test_sergay_scores_three_with_a_closed_card(self):
        self.assertEqual(score_player([("sergay", False)], 0), 1.0)
        self.assertEqual(score_player([(None, True), ("sergay", False)], 0), 5.0)
        # Own Olympiad adds one point per Sergay.
        self.assertEqual(
            score_player([("olympiad", False), ("sergay", False)], 0),
            2.0 + 2.0,
        )

    def test_akim_scales_with_own_sergay(self):
        self.assertEqual(score_player([("akim", False)], 0), 3.0)
        self.assertEqual(
            score_player([("sergay", False), ("akim", False)], 0),
            1.0 + 5.0,
        )

    def test_vanya_needs_a_closed_card_and_scales_with_akim(self):
        self.assertEqual(score_player([("vanya", False)], 0), 5.0)
        self.assertEqual(
            score_player([("akim", False), ("vanya", False)], 0),
            3.0 + 8.0,
        )

    def test_marat_exactly_one_scores_six(self):
        self.assertEqual(score_player([("marat", False)], 0), 6.0)
        self.assertEqual(score_player([("marat", False)] * 2, 0), 8.0)
        self.assertEqual(
            score_player([("olympiad", False), ("marat", False)], 0),
            2.0 + 7.0,
        )

    def test_amogus_pair_bonus(self):
        self.assertEqual(score_player([("amogus", False)], 0), 3.0)
        self.assertEqual(score_player([("amogus", False)] * 2, 0), 12.0)
        self.assertEqual(score_player([("amogus", False)] * 3, 0), 9.0)

    def test_detector_counts_global_open_alexeys(self):
        plays = [("detector", False)]
        self.assertEqual(score_player(plays, 0), 4.0)
        self.assertEqual(score_player(plays, 1), 5.0)
        self.assertEqual(score_player(plays, 3), 7.0)

    def test_leonidia_penalises_own_alexey(self):
        # Two Leonidias: each is 2 + types (1) = 3.
        self.assertEqual(score_player([("leonidia", False)] * 2, 0), 6.0)
        # One Alexey and one Leonidia: Alexey 1, Leonidia 2 + 2 types - 2 = 2.
        self.assertEqual(
            score_player([("alaksiej", False), ("leonidia", False)], 1),
            3.0,
        )

    def test_veeeep_single_and_multiple(self):
        self.assertEqual(score_player([("veeeep", False)], 0), 3.0)
        self.assertEqual(score_player([("veeeep", False)] * 2, 0), 2.0)
        self.assertEqual(
            score_player([("marat", False), ("veeeep", False)], 0),
            6.0 + 6.0,
        )
        self.assertEqual(
            score_player([("alaksiej", False), ("veeeep", False)], 1),
            1.0 + 1.0,
        )

    def test_jaman_closed_bonus_is_capped_at_five(self):
        plays = [(None, True)] * 5 + [("jaman", False)]
        # Five closed cards = 10 points, Jaman = 2 * 5 = 10.
        self.assertEqual(score_player(plays, 0), 20.0)

    def test_score_players_uses_global_alexey_count(self):
        plays = [[("alaksiej", False)], [("detector", False)]]
        self.assertEqual(score_players(plays, [0]), [1.0, 5.0])
        self.assertEqual(score_players(plays, [0, 1]), [1.0, 6.0])

    def test_score_action_delta_recomputes_the_whole_player(self):
        plays = [("akim", False)]
        delta = score_action_delta(
            plays,
            Action("sergay", False),
            own_has_alexey=False,
            players_with_alexey=0,
        )
        # Sergay adds 1 point itself and turns Akim from 3 into 5.
        self.assertEqual(delta, 3.0)


class LegalityTests(unittest.TestCase):
    def test_vanya_cannot_be_played_open_without_a_closed_card(self):
        legal = legal_actions(["vanya"], [])
        self.assertIn(Action("vanya", True), legal)
        self.assertNotIn(Action("vanya", False), legal)
        legal_after_closed = legal_actions(["vanya"], [(None, True)])
        self.assertIn(Action("vanya", False), legal_after_closed)

    def test_marat_and_detector_are_blocked_by_own_open_alexey(self):
        plays = [("alaksiej", False)]
        for card_id in ("marat", "detector"):
            legal = legal_actions([card_id], plays)
            self.assertIn(Action(card_id, True), legal)
            self.assertNotIn(Action(card_id, False), legal)

    def test_closed_copy_is_always_legal(self):
        for card_id in ("alaksiej", "marat", "detector", "vanya"):
            legal = legal_actions([card_id], [("alaksiej", False)])
            self.assertIn(Action(card_id, True), legal)

    def test_has_open_alexey_ignores_closed_alexey(self):
        self.assertTrue(has_open_alexey([("alaksiej", False)]))
        self.assertFalse(has_open_alexey([("alaksiej", True)]))
        self.assertFalse(has_open_alexey([(None, True)]))


if __name__ == "__main__":
    unittest.main()
