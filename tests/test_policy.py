import tempfile
import unittest
from collections import Counter
from pathlib import Path

from eightfm.policy import (
    PolicyAction,
    PolicyModel,
    PolicyState,
    action_feature_names,
    action_feature_size,
    iter_policy_examples,
    load_policy,
    policy_input_size,
    recommend_policy,
    save_policy,
    state_feature_names,
    state_feature_size,
    state_from_message,
)


def policy_game():
    """A tiny reconstructable game with one successful player."""

    def moves(cards, final_score):
        return [
            {"card_id": card_id, "is_closed": False, "score": score}
            for card_id, score in zip(cards, final_score)
        ]

    p0_round1 = ["alaksiej", "olympiad", "evgen", "jaman", "vanya"]
    p0_round2 = ["akim", "sergay", "veeeep", "olympiad", "alaksiej"]
    p1_cards = ["evgen"] * 5
    p2_cards = ["marat"] * 5
    players = {
        "0": {"moves": moves(p0_round1 + p0_round2, [2, 5, 8, 12, 17, 20, 24, 28, 32, 80])},
        "1": {"moves": moves(p1_cards + p1_cards, [1, 2, 3, 4, 5, 6, 7, 8, 9, 40])},
        "2": {"moves": moves(p2_cards + p2_cards, [1, 2, 3, 4, 5, 6, 7, 8, 9, 50])},
    }
    return {
        "id": 1,
        "status": "FINISHED",
        "metadata": {"deck": {"id": 0, "version": "0.2.1"}, "modifiers": []},
        "arrangement": ["0", "1", "2"],
        "players": players,
        "cards": [
            [["akim"], ["evgen"], ["marat"]],
            [["marat"], ["evgen"], ["marat"]],
        ],
    }


def tiny_policy_model():
    return PolicyModel(
        hidden_weights=[[0.0] * policy_input_size()],
        hidden_bias=[0.0],
        output_weights=[0.0],
        output_bias=0.0,
        validation_top1=0.0,
        validation_top1_success=0.0,
        validation_mrr=0.0,
        baseline_top1=0.0,
        baseline_top1_success=0.0,
        train_states=1,
        validation_states=1,
        train_games=1,
        validation_games=1,
        best_epoch=1,
        success_threshold=60.0,
    )


class PolicyTests(unittest.TestCase):
    def test_feature_layout_is_stable(self):
        self.assertEqual(state_feature_size(), 66)
        self.assertEqual(action_feature_size(), 19)
        self.assertEqual(policy_input_size(), 85)
        self.assertEqual(len(state_feature_names()), state_feature_size())
        self.assertEqual(len(action_feature_names()), action_feature_size())

    def test_reconstructs_hand_from_moves_and_saved_leftover(self):
        examples = list(iter_policy_examples({"games": [policy_game()]}))
        self.assertEqual(len(examples), 30)
        example = next(
            item
            for item in examples
            if item.state.player_id == "0"
            and item.state.round_number == 2
            and item.state.turn_number == 1
        )
        # Five future moves of round 2 plus the saved leftover.
        self.assertEqual(
            example.state.hand_counts,
            Counter({
                "akim": 1,
                "sergay": 1,
                "veeeep": 1,
                "olympiad": 1,
                "alaksiej": 1,
                "marat": 1,
            }),
        )
        self.assertEqual(example.state.unknown_hand, 0)
        self.assertEqual(example.state.chosen_card, "akim")
        self.assertEqual(example.actions[example.positive_index].card_id, "akim")
        self.assertTrue(example.state.success)
        self.assertGreater(example.state.weight, 3.0)

    def test_success_and_rank_increase_example_weight(self):
        examples = list(iter_policy_examples({"games": [policy_game()]}))
        weights = {
            player_id: next(
                item.state.weight
                for item in examples
                if item.state.player_id == player_id
            )
            for player_id in ("0", "1", "2")
        }
        self.assertGreater(weights["0"], weights["2"])
        self.assertGreater(weights["2"], weights["1"])

    def test_save_and_load_policy_roundtrip(self):
        model = tiny_policy_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            save_policy(model, path)
            restored = load_policy(path)
        self.assertEqual(restored.hidden_weights, model.hidden_weights)
        self.assertEqual(restored.hidden_bias, model.hidden_bias)
        self.assertEqual(restored.output_weights, model.output_weights)
        self.assertEqual(restored.output_bias, model.output_bias)
        self.assertEqual(restored.success_threshold, model.success_threshold)

    def test_state_from_4784_message_uses_visible_hand(self):
        message = Path(__file__).with_name("message-4784-round2.txt").read_text(encoding="utf-8")
        state, candidates, player_name = state_from_message(message, player_name="Ivan")
        self.assertEqual(player_name, "Ivan")
        self.assertEqual(state.player_count, 6)
        self.assertEqual((state.round_number, state.turn_number), (2, 4))
        self.assertEqual(
            state.hand_counts,
            {"veeeep": 1, "leonidia": 1, "alaksiej": 1},
        )
        # The generic closed option is added by recommend_policy, not parsed
        # from the message.
        self.assertEqual(len(candidates), 3)

    def test_recommend_policy_ranks_every_visible_card_and_closed_option(self):
        message = Path(__file__).with_name("message-4784-round2.txt").read_text(encoding="utf-8")
        result = recommend_policy(
            message,
            player_name="Ivan",
            model=tiny_policy_model(),
        )
        self.assertEqual(result.player_name, "Ivan")
        self.assertEqual(len(result.candidates), 4)
        self.assertEqual(sum(item.play == "закрыто" for item in result.candidates), 1)
        self.assertAlmostEqual(sum(item.probability for item in result.candidates), 1.0, places=4)
        self.assertTrue(all(len(item.feature_vector) == policy_input_size() for item in result.candidates))

    def test_closed_action_is_generic_and_distinct(self):
        state = PolicyState(
            player_count=5,
            round_number=1,
            turn_number=1,
            own_counts={},
            own_closed=0,
            table_counts={},
            table_closed=0,
            own_score=0.0,
            other_scores=(0.0, 0.0, 0.0, 0.0),
            hand_counts={"akim": 1},
            unknown_hand=1,
        )
        opened = PolicyAction(card_id="akim", is_closed=False)
        closed = PolicyAction(card_id=None, is_closed=True)
        model = tiny_policy_model()
        self.assertEqual(model.predict_action(state, opened), 0.0)
        self.assertEqual(model.predict_action(state, closed), 0.0)


if __name__ == "__main__":
    unittest.main()
