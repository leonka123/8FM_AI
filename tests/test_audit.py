import unittest
from pathlib import Path

from eightfm.audit import audit_data
from eightfm.agent import _parse_state, build_message_features, predict_card, predict_message
from eightfm.history import analyze_data
from eightfm.model import (
    CARD_IDS,
    ScoreModel,
    _archive_examples,
    feature_names,
    feature_size,
    make_features,
)


def dummy_score_model():
    return ScoreModel(
        hidden_weights=[[0.0] * feature_size()],
        hidden_bias=[0.0],
        output_weights=[0.0],
        output_bias=0.0,
        target_mean=4.25,
        target_scale=1.0,
        validation_rmse=0.0,
        validation_baseline_rmse=0.0,
        train_games=1,
        validation_games=1,
        training_records=1,
    )


def finished_classic_game(player_count=3, leftover_size=1, modifiers=None):
    players = {
        str(index): {"moves": [{"card_id": "evgen", "is_closed": False, "score": index} for _ in range(10)]}
        for index in range(player_count)
    }
    return {
        "status": "FINISHED",
        "metadata": {
            "deck": {"id": 0, "version": "0.2.1"},
            "modifiers": modifiers or [],
        },
        "players": players,
        "arrangement": [str(index) for index in range(player_count)],
        # cards[round][seat] is the card left after five moves, not a deal.
        "cards": [
            [["evgen"] * leftover_size for _ in range(player_count)],
            [["alaksiej"] * leftover_size for _ in range(player_count)],
        ],
    }


class AuditTests(unittest.TestCase):
    def test_counts_replayable_classic_game(self):
        report = audit_data({"schema_version": 2, "timestamp": 1, "users": {}, "games": [finished_classic_game()]})
        self.assertEqual(report.classic_candidate_finished, 1)
        self.assertEqual(report.classic_candidate_with_supported_player_count, 1)
        self.assertEqual(report.classic_candidate_with_clean_move_history, 1)
        self.assertEqual(report.classic_candidate_with_reconstructable_hands, 1)
        self.assertEqual(report.card_id_counts, {"alaksiej": 3, "evgen": 3})

    def test_flags_short_saved_hands(self):
        report = audit_data({"schema_version": 2, "games": [finished_classic_game(leftover_size=0)]})
        self.assertEqual(report.classic_candidate_with_reconstructable_hands, 0)
        self.assertTrue(any("reconstructable six-card hands" in warning for warning in report.warnings))

    def test_only_classic_and_supported_player_counts_are_replay_candidates(self):
        games = [
            finished_classic_game(player_count=2),
            finished_classic_game(player_count=7),
            finished_classic_game(modifiers=["meteors"]),
            {
                "status": "FINISHED",
                "metadata": {"deck": {"id": 3, "version": "0.1"}},
                "players": {},
            },
        ]
        report = audit_data({"schema_version": 2, "games": games})
        self.assertEqual(report.classic_candidate_finished, 3)
        # The third record is a 3-player game with a modifier, so it still
        # qualifies for the deck/player-count summary but not a vanilla run.
        self.assertEqual(report.classic_candidate_with_supported_player_count, 1)
        self.assertEqual(report.classic_candidate_with_reconstructable_hands, 1)
        self.assertEqual(report.classic_candidate_with_modifiers, 1)

    def test_rejects_invalid_root_games_type(self):
        with self.assertRaises(ValueError):
            audit_data({"games": {}})

    def test_history_analysis_is_descriptive_and_filters_modifiers(self):
        game = finished_classic_game()
        report = analyze_data({"games": [game, finished_classic_game(modifiers=["meteors"])]})
        self.assertEqual(report.eligible_games, 1)
        self.assertEqual(report.games_used, 1)
        self.assertEqual(report.move_records_used, 30)
        first_turn = next(stat for stat in report.action_stats if stat.players == 3 and stat.round == 1 and stat.turn == 1)
        self.assertEqual(first_turn.action, "evgen")
        self.assertEqual(first_turn.mean_recorded_score_delta, 1.0)
        self.assertTrue(any("do not identify the optimal" in note for note in report.notes))

    def test_feature_vector_has_stable_size_and_closed_action_is_distinct(self):
        opened = make_features(card_id="evgen", is_closed=False, players=5, round_number=2, turn_number=4)
        closed = make_features(card_id=None, is_closed=True, players=5, round_number=2, turn_number=4)
        self.assertEqual(len(opened), feature_size())
        self.assertEqual(len(closed), feature_size())
        self.assertNotEqual(opened, closed)
        self.assertEqual(len(CARD_IDS), 14)

    def test_training_features_do_not_leak_hidden_archive_cards(self):
        players = {}
        for player_id in ("a", "b", "c"):
            card_id = "alaksiej" if player_id == "a" else "olympiad"
            players[player_id] = {
                "moves": [
                    {"card_id": card_id, "is_closed": player_id == "a", "score": 2},
                    *[
                        {"card_id": "evgen", "is_closed": False, "score": 2 + turn}
                        for turn in range(1, 10)
                    ],
                ]
            }
        data = {"games": [{
            "id": 1,
            "status": "FINISHED",
            "metadata": {"deck": {"id": 0, "version": "0.2.1"}},
            "players": players,
        }]}
        features, _, _ = _archive_examples(data)
        # Turn two, player a: the archive knows the identity of their first
        # closed card, but a live observer does not. Counts start after the
        # action/context one-hot fields.
        own_alaksiej_index = 14 + 1 + 1 + 4 + 2 + 5
        table_alaksiej_index = own_alaksiej_index + len(CARD_IDS)
        self.assertEqual(features[3][own_alaksiej_index], 0.0)
        self.assertEqual(features[3][table_alaksiej_index], 0.0)

    def test_non_classic_message_is_rejected(self):
        message = """👾 ИГРА 8 ФИЗМАТОВ #1
⬆️ Раунд 1/2 | 🃏 Карта 1/5
🃏 8ФМ Turbo

Выберите свою карту!

▫ 🍌 АКИМ
Даёт 3 очка
"""
        with self.assertRaises(ValueError):
            predict_message(message, player_name="Ivan", model=dummy_score_model())

    def test_predict_message_reads_pasted_classic_message_and_does_not_submit(self):
        message = """👾 ИГРА 8 ФИЗМАТОВ #4784
⬆️ Раунд 1/2 | 🃏 Карта 1/5
🃏 8ФМ Classic

▫ 👤 Ivan (1509)
▫ 👤 Андрэй (1560)
▫ 👤 olegg (1430)

Выберите свою карту!

▫ 🤓 СЕРГЕЙ
Даёт 1 очко или 3 очка при хотя бы 1 карте взакрытую; Олимпиада = +1
▫ 🥰 ЯН-ПАВЕЛ
Даёт 1 очко при хотя бы 1 Жене; Женя/Сергей/Аким/Шевчик Максим = +1; первые 5 карт взакрытую = +2
▫ 🇰🇵 МАРАТ
Даёт 4 очка или 6 очков при ровно 1 Марате; Олимпиада = +1; Нельзя играть при Алексее Пшеничном
"""
        result = predict_message(message, player_name="Ivan", model=dummy_score_model())
        self.assertEqual((result.round_number, result.turn_number), (1, 1))
        self.assertEqual(result.player_name, "Ivan")
        self.assertEqual(len(result.candidates), 4)  # three open choices + generic face-down option
        self.assertEqual(sum(item.play == "закрыто" for item in result.candidates), 1)
        self.assertTrue(any("Score-модель" in warning for warning in result.warnings))

    def test_message_4734_builds_training_vector_and_ignores_confirmation(self):
        message = Path(__file__).with_name("message-4734.txt").read_text(encoding="utf-8")
        vector = build_message_features(
            message,
            player_name="Леончик",
            card_id="akim",
        )
        values = dict(zip(feature_names(), vector))
        self.assertEqual(len(vector), feature_size())
        self.assertEqual(values["action:akim"], 1.0)
        self.assertEqual(values["players:5"], 1.0)
        self.assertEqual(values["round:2"], 1.0)
        self.assertEqual(values["turn:1"], 1.0)
        # The five icons in each row are round 1 history. The model resets
        # card counts at the start of round 2, so all card-count features are
        # zero for the first turn of the new round.
        self.assertAlmostEqual(values["own_count:sergay"], 0.0)
        self.assertAlmostEqual(values["own_count:vanya"], 0.0)
        self.assertAlmostEqual(values["own_count:detector"], 0.0)
        self.assertAlmostEqual(values["own_closed"], 0.0)
        self.assertAlmostEqual(values["own_score"], 17 / 150)
        self.assertAlmostEqual(values["table_count:alaksiej"], 0.0)
        self.assertAlmostEqual(values["table_count:leonidia"], 0.0)

        # The confirmation line is not part of the pre-move state. Removing
        # it must not change the vector.
        without_confirmation = message.split("Вы сыграли карту", 1)[0]
        same_vector = build_message_features(
            without_confirmation,
            player_name="Леончик",
            card_id="akim",
        )
        self.assertEqual(vector, same_vector)

        with_comment = message.replace(
            "Вы сыграли карту 🍌 АКИМ",
            "Вы сыграли карту 🍌 АКИМ - этот текст он должен не видеть",
        )
        self.assertEqual(
            vector,
            build_message_features(with_comment, player_name="Леончик", card_id="akim"),
        )

        result = predict_message(message, player_name="Леончик", model=dummy_score_model())
        self.assertEqual(len(result.candidates), 7)  # six open cards + one face-down option
        self.assertTrue(all("Вы сыграли" not in item.name for item in result.candidates))

    def test_message_4784_uses_only_last_three_round_two_cards(self):
        message = Path(__file__).with_name("message-4784-round2.txt").read_text(encoding="utf-8")
        round_number, turn_number, rows, chosen = _parse_state(message, "Ivan")
        self.assertEqual((round_number, turn_number), (2, 4))
        self.assertEqual(chosen, "Ivan")
        self.assertEqual(len(rows), 6)
        # Eight icons per row: five from round 1 plus three from round 2.
        self.assertTrue(all(len(row["tokens"]) == 8 for row in rows))

        vector = build_message_features(message, player_name="Ivan", card_id="veeeep")
        values = dict(zip(feature_names(), vector))
        self.assertEqual(len(vector), feature_size())
        self.assertEqual(values["action:veeeep"], 1.0)
        self.assertEqual(values["players:6"], 1.0)
        self.assertEqual(values["round:2"], 1.0)
        self.assertEqual(values["turn:4"], 1.0)
        # Ivan's last three icons are 🤯 🏆 🤓.
        self.assertAlmostEqual(values["own_count:vanya"], 0.1)
        self.assertAlmostEqual(values["own_count:olympiad"], 0.1)
        self.assertAlmostEqual(values["own_count:sergay"], 0.1)
        self.assertAlmostEqual(values["own_count:akim"], 0.0)  # round 1 only
        self.assertAlmostEqual(values["own_closed"], 0.0)
        self.assertAlmostEqual(values["own_score"], 39 / 150)
        self.assertAlmostEqual(values["mean_other_score"], 36.6 / 150)
        # Other players' round-2 cards: two Leonidias, two Marats, one closed.
        self.assertAlmostEqual(values["table_count:leonidia"], 2 / 30)
        self.assertAlmostEqual(values["table_count:marat"], 2 / 30)
        self.assertAlmostEqual(values["table_closed"], 1 / 30)

        without_confirmation = message.split("Вы сыграли карту", 1)[0]
        self.assertEqual(
            vector,
            build_message_features(without_confirmation, player_name="Ivan", card_id="veeeep"),
        )

        result = predict_message(message, player_name="Ivan", model=dummy_score_model())
        self.assertEqual(result.player_name, "Ivan")
        self.assertEqual(len(result.candidates), 4)  # three open cards + one face-down option
        self.assertTrue(all("Вы сыграли" not in item.name for item in result.candidates))

    def test_only_current_round_cards_enter_features(self):
        message = """👾 ИГРА 8 ФИЗМАТОВ #1
⬇️ Раунд 1/2 | 🃏 Карта 2/5
🃏 8ФМ Classic

▫ 👤 A (0): 🤠 (2)
▫ 👤 B (0): 🐷 (3)
▫ 👤 C (0): ❌ (1)

Выберите свою карту!

▫ 🍌 АКИМ
Даёт 3 очка
"""
        vector = build_message_features(message, player_name="A", card_id="akim")
        values = dict(zip(feature_names(), vector))
        self.assertAlmostEqual(values["own_count:leonidia"], 0.1)
        self.assertAlmostEqual(values["table_count:alaksiej"], 1 / 30)
        self.assertAlmostEqual(values["own_closed"], 0.0)
        self.assertAlmostEqual(values["table_closed"], 1 / 30)

    def test_predict_card_returns_vector_and_model_prediction(self):
        message = Path(__file__).with_name("message-4734.txt").read_text(encoding="utf-8")
        model = ScoreModel(
            hidden_weights=[[0.0] * feature_size()],
            hidden_bias=[0.0],
            output_weights=[0.0],
            output_bias=0.0,
            target_mean=4.25,
            target_scale=1.0,
            validation_rmse=0.0,
            validation_baseline_rmse=0.0,
            train_games=1,
            validation_games=1,
            training_records=1,
        )
        vector, prediction = predict_card(
            message,
            player_name="Леончик",
            model=model,
            card_id="akim",
        )
        self.assertEqual(len(vector), feature_size())
        self.assertEqual(prediction, 4.25)


if __name__ == "__main__":
    unittest.main()
