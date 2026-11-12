import asyncio
import random
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from eightfm import telegram_assistant as ta
from eightfm.telegram_assistant import (
    MSK,
    AssistantConfig,
    ButtonTarget,
    PackError,
    TelegramAssistant,
    _msk,
    _preview,
    card_emoji,
    check_pack,
    choose_plan,
    find_button_target,
    is_choice_message,
    message_text,
    parse_deadline,
    plan_reply_time,
    rich_message_text,
)

SAMPLE = """👾 ИГРА 8 ФИЗМАТОВ #4797
⬆️ Раунд 1/2 | 🃏 Карта 1/5
🃏 8ФМ Classic
🛡 МАТЧ ЛИГ: Сезон #26, Лига 3, Матч #2/2
⌛️ Ход заканчивается: 11.11.2026 16:00:00 МСК

▫ 👤 Дима (1443)
▫ 👤 Ваня (1423)
▫ 👤 Леончик (1526)
▫ 👤 Андрэй (1560)
▫ 👤 olegg (1430)
▫ 👤 Margaret☀️ (1411)

Выберите свою карту!

▫ 🍌 АКИМ
Даёт 3 очка; Сергей = +2
▫ 🤫 ЖЕНЯ
Даёт 2 очка; Амогус = +3; Детектор Алексея = +1
▫ 🤠 ЛЕОНИДИЯ
Даёт 2 очка; различный вид карт (в том числе 🐷, 🤠 и взакрытую) = +1; Алексей Пшеничный = -2
▫ 🏆 ОЛИМПИАДА
Даёт 2/4/6 очков при хотя бы 1/3/6 Олимпиадах
▫ 🤯 ИВАН ГРЕЧИХА
Даёт 5 очков; Аким = +3; Нельзя играть без закрытой карты
▫ 🇰🇵 МАРАТ
Даёт 4 очка или 6 очков при ровно 1 Марате; Олимпиада = +1; Нельзя играть при Алексее Пшеничном
"""

OPEN_ROW = ["🍌", "🤫", "🤠", "🏆", "🤯", "🇰🇵"]
CLOSE_ROW = ["❌", "❌", "❌", "❌", "❌", "❌"]
BUTTONS = [OPEN_ROW, CLOSE_ROW]


def config(**overrides):
    base = {
        "min_delay_minutes": 40.0,
        "max_delay_minutes": 80.0,
        "deadline_margin_minutes": 10.0,
        "reply_delay_seconds": 0.0,
        "seed": 1,
    }
    base.update(overrides)
    return AssistantConfig(**base)


class PackCheckTests(unittest.TestCase):
    def test_sample_is_the_right_pack(self):
        result = check_pack(SAMPLE, config())
        self.assertTrue(result.ok, result.reason)

    def test_missing_pack_marker_is_rejected(self):
        result = check_pack(SAMPLE.replace("8ФМ Classic", "8ФМ Duo"), config())
        self.assertFalse(result.ok)

    def test_modifier_is_rejected(self):
        result = check_pack(SAMPLE + "\n🎲 Модификатор: двойные очки\n", config())
        self.assertFalse(result.ok)
        self.assertIn("модификатор", result.reason.lower())

    def test_custom_modifier_pattern(self):
        cfg = config(modifier_patterns=(r"(?i)особый\s+режим",))
        self.assertFalse(check_pack(SAMPLE + "\nОсобый режим включён\n", cfg).ok)
        self.assertTrue(check_pack(SAMPLE, cfg).ok)


class MessageShapeTests(unittest.TestCase):
    def test_detects_choice_section(self):
        self.assertTrue(is_choice_message(SAMPLE))
        self.assertFalse(is_choice_message("👾 ИГРА 8 ФИЗМАТОВ #1\nИгра окончена"))

    def test_played_message_is_not_a_choice(self):
        played = SAMPLE + "\nВы сыграли карту 🤠 ЛЕОНИДИЯ\n"
        self.assertFalse(is_choice_message(played))

    def test_parses_deadline_in_moscow_time(self):
        deadline = parse_deadline(SAMPLE)
        self.assertEqual(deadline, datetime(2026, 11, 11, 16, 0, 0, tzinfo=MSK))

    def test_missing_deadline_is_none(self):
        self.assertIsNone(parse_deadline("без дедлайна"))


class CardEmojiTests(unittest.TestCase):
    def test_known_cards(self):
        self.assertEqual(card_emoji("akim"), "🍌")
        self.assertEqual(card_emoji("marat"), "🇰🇵")
        self.assertEqual(card_emoji("leonidia"), "🤠")
        self.assertIsNone(card_emoji(None))


class ButtonMappingTests(unittest.TestCase):
    def test_open_button_maps_to_emoji_column(self):
        target = find_button_target(BUTTONS, "akim")
        self.assertEqual((target.open_row, target.open_col), (0, 0))
        self.assertEqual((target.close_row, target.close_col), (1, 0))

    def test_last_card_maps_to_last_column(self):
        target = find_button_target(BUTTONS, "marat")
        self.assertEqual((target.open_row, target.open_col), (0, 5))
        self.assertEqual((target.close_row, target.close_col), (1, 5))

    def test_unknown_card_has_no_button(self):
        self.assertIsNone(find_button_target(BUTTONS, None))

    def test_missing_close_row_is_rejected(self):
        self.assertIsNone(find_button_target([OPEN_ROW], "akim"))


class ReplyTimingTests(unittest.TestCase):
    def test_immediate_returns_now(self):
        now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
        result = plan_reply_time(
            now,
            deadline=None,
            rng=random.Random(0),
            min_delay_minutes=40,
            max_delay_minutes=80,
            deadline_margin_minutes=10,
            immediate=True,
        )
        self.assertEqual(result, now)

    def test_delay_is_within_window(self):
        now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
        for seed in range(20):
            result = plan_reply_time(
                now,
                deadline=None,
                rng=random.Random(seed),
                min_delay_minutes=40,
                max_delay_minutes=80,
                deadline_margin_minutes=10,
            )
            delta = (result - now).total_seconds() / 60
            self.assertGreaterEqual(delta, 40 - 1e-9)
            self.assertLessEqual(delta, 80 + 1e-9)

    def test_delay_is_capped_before_the_deadline(self):
        now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
        deadline = now + timedelta(minutes=30)
        result = plan_reply_time(
            now,
            deadline=deadline,
            rng=random.Random(0),
            min_delay_minutes=40,
            max_delay_minutes=80,
            deadline_margin_minutes=10,
        )
        self.assertLessEqual(result, deadline - timedelta(minutes=10))


class ChoosePlanTests(unittest.TestCase):
    def _fake_rank(self, card_id, is_closed):
        return {
            "round_number": 1,
            "turn_number": 1,
            "value": 0.0,
            "candidates": [
                {
                    "card_id": card_id,
                    "is_closed": is_closed,
                    "policy_score": 1.23,
                    "probability": 0.42,
                }
            ],
            "warnings": [],
        }

    def test_plan_picks_top_action_and_maps_button(self):
        with patch(
            "eightfm.rl_train.rank_message_actions",
            return_value=self._fake_rank("marat", False),
        ):
            plan = choose_plan(
                SAMPLE,
                net=object(),
                config=config(),
                message_id=4797,
                button_rows=BUTTONS,
                now=datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc),
                immediate=True,
            )
        self.assertEqual(plan.card_id, "marat")
        self.assertFalse(plan.is_closed)
        self.assertEqual((plan.row, plan.col), (0, 5))
        self.assertEqual(plan.message_id, 4797)

    def test_closed_action_uses_close_button(self):
        with patch(
            "eightfm.rl_train.rank_message_actions",
            return_value=self._fake_rank("akim", True),
        ):
            plan = choose_plan(
                SAMPLE,
                net=object(),
                config=config(),
                button_rows=BUTTONS,
                immediate=True,
            )
        self.assertTrue(plan.is_closed)
        self.assertEqual((plan.row, plan.col), (1, 0))

    def test_modifier_blocks_the_plan(self):
        with patch(
            "eightfm.rl_train.rank_message_actions",
            return_value=self._fake_rank("akim", False),
        ):
            with self.assertRaises(PackError):
                choose_plan(
                    SAMPLE + "\nМодификатор: тест\n",
                    net=object(),
                    config=config(),
                    button_rows=BUTTONS,
                )


class RichMessageTests(unittest.TestCase):
    def test_plain_message_wins_over_rich(self):
        message = type("M", (), {"message": "hello", "rich_message": {"_": "RichMessage"}})()
        self.assertEqual(message_text(message), "hello")

    def test_rich_message_tree_is_flattened(self):
        rich = {
            "_": "RichMessage",
            "blocks": [
                {
                    "_": "PageBlockParagraph",
                    "text": {
                        "_": "TextConcat",
                        "texts": [
                            {"_": "TextButton", "text": {"_": "TextPlain", "text": "👾"}, "type": {}},
                            {"_": "TextPlain", "text": " "},
                            {"_": "TextBold", "text": {"_": "TextPlain", "text": "ИГРА 8 ФИЗМАТОВ"}},
                            {"_": "TextPlain", "text": " "},
                            {"_": "TextButton", "text": {"_": "TextPlain", "text": "#4797"}, "type": {}},
                            {"_": "TextPlain", "text": "\n"},
                            {"_": "TextBold", "text": {"_": "TextPlain", "text": "⬆️ Раунд 1/2 | 🃏 Карта 3/5"}},
                            {"_": "TextPlain", "text": "\n🃏 "},
                            {"_": "TextBold", "text": {"_": "TextPlain", "text": "8ФМ Classic"}},
                            {"_": "TextPlain", "text": "\n⌛️ "},
                            {"_": "TextBold", "text": {"_": "TextPlain", "text": "Ход заканчивается"}},
                            {"_": "TextPlain", "text": ": "},
                            {
                                "_": "TextDate",
                                "text": {"_": "TextPlain", "text": "12.11.2026 16:00:00 МСК"},
                                "date": "2026-11-12T13:00:00+00:00",
                            },
                            {"_": "TextPlain", "text": "\n\n▫ "},
                            {"_": "TextButton", "text": {"_": "TextPlain", "text": "👤"}, "type": {}},
                            {"_": "TextPlain", "text": " "},
                            {
                                "_": "TextMentionName",
                                "text": {"_": "TextPlain", "text": "Леончик"},
                                "user_id": 1408445769,
                            },
                            {"_": "TextPlain", "text": " (1526): 🤠🧨 (10)\n\n"},
                            {"_": "TextBold", "text": {"_": "TextPlain", "text": "Выберите свою карту!"}},
                        ],
                    },
                },
            ],
        }
        message = type("M", (), {"message": "", "rich_message": rich})()
        text = message_text(message)
        self.assertIn("ИГРА 8 ФИЗМАТОВ", text)
        self.assertIn("8ФМ Classic", text)
        self.assertIn("Леончик (1526): 🤠🧨 (10)", text)
        self.assertIn("Выберите свою карту!", text)
        self.assertTrue(is_choice_message(text))
        self.assertTrue(check_pack(text, config()).ok)
        self.assertEqual(parse_deadline(text), datetime(2026, 11, 12, 16, 0, tzinfo=MSK))

    def test_telethon_style_objects(self):
        class TextPlain:
            def __init__(self, text):
                self.text = text

        class TextConcat:
            def __init__(self, texts):
                self.texts = texts

        class TextBold:
            def __init__(self, text):
                self.text = text

        class PageBlockParagraph:
            def __init__(self, text):
                self.text = text

        class RichMessage:
            def __init__(self, blocks):
                self.blocks = blocks

        node = RichMessage([
            PageBlockParagraph(TextConcat([
                TextPlain("a"),
                TextBold(TextPlain("b")),
            ])),
        ])
        self.assertEqual(rich_message_text(node), "ab")

    def test_empty_rich_message_is_empty(self):
        self.assertEqual(rich_message_text(None), "")
        self.assertEqual(rich_message_text({"_": "RichMessage", "blocks": []}), "")


class LogHelperTests(unittest.TestCase):
    def test_log_sink_receives_lines(self):
        lines = []
        ta.set_log_sink(lines.append)
        try:
            ta._log("привет из ассистента")
        finally:
            ta.set_log_sink(None)
        self.assertTrue(any("привет из ассистента" in line for line in lines))

    def test_msk_formats_in_moscow_time(self):
        moment = datetime(2026, 11, 11, 13, 5, 9, tzinfo=timezone.utc)
        self.assertEqual(_msk(moment), "11.11 16:05:09")

    def test_msk_handles_missing_moment(self):
        self.assertEqual(_msk(None), "?")

    def test_preview_is_flat_and_truncated(self):
        self.assertEqual(_preview("a\n b   c"), "a b c")
        self.assertTrue(_preview("x" * 200).endswith("…"))
        self.assertEqual(len(_preview("x" * 200)), 61)


class FakeMessage:
    def __init__(self, message_id, text, buttons=None, out=False, chat_id=1):
        self.id = message_id
        self.message = text
        self.buttons = buttons
        self.out = out
        self.chat_id = chat_id
        self.date = datetime(2026, 11, 11, 13, 0, tzinfo=timezone.utc)


class FakeFresh:
    def __init__(self, buttons, text=SAMPLE):
        self.buttons = buttons
        self.message = text
        self.rich_message = None
        self.clicked = []

    async def click(self, row, col):
        self.clicked.append((row, col))


class FakeClient:
    def __init__(self, fresh):
        self._fresh = fresh

    async def get_messages(self, chat_id, ids):
        return self._fresh


class StartupBehaviourTests(unittest.TestCase):
    """Startup answers a pending choice immediately, with no 40-minute delay."""

    def _plan(self):
        now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
        return ta.ChoicePlan(
            message_id=4797,
            card_id="akim",
            is_closed=False,
            label="🍌 akim (открыто)",
            target=ButtonTarget(0, 0, 1, 0),
            reply_at=now,
            deadline=None,
            policy_score=1.0,
            probability=0.9,
        )

    def test_new_choice_is_answered_immediately(self):
        assistant = TelegramAssistant(config(), object())
        message = FakeMessage(4797, SAMPLE, buttons=BUTTONS)
        fresh = FakeFresh([["🍌", "🤫"], ["❌", "❌"]])
        client = FakeClient(fresh)
        with patch.object(ta, "choose_plan", return_value=self._plan()):
            asyncio.run(assistant._on_new(client, message))
        self.assertEqual(fresh.clicked, [(0, 0)])
        self.assertIn(4797, assistant._handled)

    def test_new_choice_waits_configured_seconds(self):
        assistant = TelegramAssistant(config(reply_delay_seconds=20), object())
        message = FakeMessage(4797, SAMPLE, buttons=BUTTONS)
        fresh = FakeFresh([["🍌", "🤫"], ["❌", "❌"]])
        client = FakeClient(fresh)
        with patch.object(ta, "choose_plan", return_value=self._plan()), patch.object(
            ta.asyncio, "sleep", new_callable=AsyncMock
        ) as sleep:
            asyncio.run(assistant._on_new(client, message))
        sleep.assert_awaited_once_with(20.0)
        self.assertEqual(fresh.clicked, [(0, 0)])

    def test_reply_delay_defaults_to_20_seconds(self):
        self.assertEqual(AssistantConfig().reply_delay_seconds, 20.0)
        parsed = AssistantConfig.from_dict({"reply_delay_seconds": 5})
        self.assertEqual(parsed.reply_delay_seconds, 5.0)

    def test_stop_disconnects_the_client(self):
        assistant = TelegramAssistant(config(), object())

        class FakeClient:
            def __init__(self):
                self.disconnected = False

            async def disconnect(self):
                self.disconnected = True

        client = FakeClient()
        assistant.client = client
        asyncio.run(assistant.stop())
        self.assertTrue(client.disconnected)

    def test_dry_run_never_clicks(self):
        assistant = TelegramAssistant(config(dry_run=True), object())
        message = FakeMessage(4797, SAMPLE, buttons=BUTTONS)
        fresh = FakeFresh([["🍌", "🤫"], ["❌", "❌"]])
        client = FakeClient(fresh)
        with patch.object(ta, "choose_plan", return_value=self._plan()):
            asyncio.run(assistant._on_new(client, message))
        self.assertEqual(fresh.clicked, [])

    def test_handled_message_is_not_answered_twice(self):
        assistant = TelegramAssistant(config(), object())
        assistant._handled.add(4797)
        message = FakeMessage(4797, SAMPLE, buttons=BUTTONS)
        fresh = FakeFresh([["🍌", "🤫"], ["❌", "❌"]])
        client = FakeClient(fresh)
        with patch.object(ta, "choose_plan", return_value=self._plan()) as mocked:
            asyncio.run(assistant._on_new(client, message))
        mocked.assert_not_called()
        self.assertEqual(fresh.clicked, [])

    def test_missing_button_skips_the_press(self):
        assistant = TelegramAssistant(config(), object())
        message = FakeMessage(4797, SAMPLE, buttons=BUTTONS)
        # Keyboard no longer contains the chosen card's emoji.
        fresh = FakeFresh([["🏆"], ["❌"]])
        client = FakeClient(fresh)
        with patch.object(ta, "choose_plan", return_value=self._plan()):
            asyncio.run(assistant._on_new(client, message))
        self.assertEqual(fresh.clicked, [])

    def test_already_played_message_is_not_clicked(self):
        assistant = TelegramAssistant(config(), object())
        message = FakeMessage(4797, SAMPLE, buttons=BUTTONS)
        fresh = FakeFresh(
            [["🍌", "🤫"], ["❌", "❌"]],
            text=SAMPLE + "\nВы сыграли карту 🍌 АКИМ\n",
        )
        client = FakeClient(fresh)
        with patch.object(ta, "choose_plan", return_value=self._plan()):
            asyncio.run(assistant._on_new(client, message))
        self.assertEqual(fresh.clicked, [])


if __name__ == "__main__":
    unittest.main()
