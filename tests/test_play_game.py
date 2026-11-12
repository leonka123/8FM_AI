import io
import unittest
from contextlib import redirect_stdout

from eightfm.play_game import (
    _format_play,
    parse_opponents,
    play_interactive_game,
    resolve_opponent,
)


class ParseOpponentsTests(unittest.TestCase):
    def test_accepts_two_known_bots(self):
        self.assertEqual(parse_opponents("greedy,combo"), ("greedy", "combo"))
        self.assertEqual(parse_opponents(" RL , random "), ("rl", "random"))
        self.assertEqual(parse_opponents("supervised,rl"), ("supervised", "rl"))

    def test_accepts_up_to_six_bots(self):
        tokens = parse_opponents("rl,supervised,greedy,combo,random,rl")
        self.assertEqual(len(tokens), 6)

    def test_accepts_shorthands_and_explicit_checkpoints(self):
        tokens = parse_opponents(
            "lu_luna,classic-rl,supervised,rl:models/lu_luna.json"
        )
        self.assertEqual(len(tokens), 4)

    def test_rejects_bad_bot_count(self):
        for value in ("greedy", "rl,rl,rl,rl,rl,rl,rl", ""):
            with self.assertRaises(ValueError):
                parse_opponents(value)

    def test_rejects_unknown_bot(self):
        with self.assertRaises(ValueError):
            parse_opponents("greedy,telegram")

    def test_accepts_ensemble_bot(self):
        self.assertEqual(resolve_opponent("ensemble"), ("ensemble", None))
        self.assertEqual(
            resolve_opponent("ensemble:models/x.json"), ("ensemble", "models/x.json")
        )
        self.assertEqual(parse_opponents("ensemble,supervised"), ("ensemble", "supervised"))

    def test_resolves_alias_and_checkpoint(self):
        self.assertEqual(resolve_opponent("lu_luna"), ("rl", "models/lu_luna.json"))
        self.assertEqual(
            resolve_opponent("classic-rl"), ("rl", "models/classic-rl.json")
        )
        self.assertEqual(resolve_opponent("rl:models/x.json"), ("rl", "models/x.json"))
        self.assertEqual(resolve_opponent("Greedy"), ("greedy", None))


class HiddenCardDisplayTests(unittest.TestCase):
    def test_closed_cards_are_masked_and_open_cards_are_not(self):
        self.assertEqual(_format_play("akim", True), "❌")
        self.assertEqual(_format_play("akim", False), "🍌")


class InteractiveGameTests(unittest.TestCase):
    def test_full_scripted_game_finishes_after_ten_moves(self):
        prompts = []
        answers = iter(["1"] * 10)

        def input_fn(prompt: str) -> str:
            prompts.append(prompt)
            return next(answers)

        output = io.StringIO()
        with redirect_stdout(output):
            code = play_interactive_game(
                seed=1,
                opponents=("greedy", "greedy"),
                input_fn=input_fn,
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(prompts), 10)
        text = output.getvalue()
        self.assertIn("8ФМ Classic", text)
        self.assertIn("Раунд 2/2", text)
        self.assertIn("Игра окончена", text)
        self.assertIn("Ваш счёт", text)
        self.assertIn("Цель >60", text)
        self.assertNotIn("Traceback", text)

    def test_four_player_game_with_three_opponents(self):
        answers = iter(["1"] * 10)

        output = io.StringIO()
        with redirect_stdout(output):
            code = play_interactive_game(
                seed=4,
                opponents=("greedy", "combo", "random"),
                input_fn=lambda _prompt: next(answers),
            )

        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("Соперники: Бот 1 (greedy), Бот 2 (combo), Бот 3 (random)", text)
        self.assertIn("Игра окончена", text)
        self.assertNotIn("Traceback", text)

    def test_invalid_input_is_reasked_without_consuming_a_move(self):
        prompts = []
        answers = iter(["abc", "0", "1"] + ["1"] * 9)

        def input_fn(prompt: str) -> str:
            prompts.append(prompt)
            return next(answers)

        output = io.StringIO()
        with redirect_stdout(output):
            code = play_interactive_game(
                seed=2,
                opponents=("greedy", "greedy"),
                input_fn=input_fn,
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(prompts), 12)
        text = output.getvalue()
        self.assertIn("Введите номер варианта", text)
        self.assertIn("Нужен номер от 1 до", text)
        self.assertIn("Игра окончена", text)

    def test_quit_aborts_the_game(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = play_interactive_game(
                seed=3,
                opponents=("greedy", "greedy"),
                input_fn=lambda _prompt: "q",
            )

        self.assertEqual(code, 1)
        self.assertIn("Игра прервана", output.getvalue())


if __name__ == "__main__":
    unittest.main()
