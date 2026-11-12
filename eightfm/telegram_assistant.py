"""Telegram auto-chooser for 8ФМ Classic, driven by the lu_luna RL policy.

The game bot posts a message with an inline keyboard: one button per card to
play it open, plus an ❌ button per card to play it closed.  This module reads
that message, lets ``lu_luna`` pick the move, and presses the matching button.

Behaviour required by the user:

* every choice message is answered after a short pause (20 seconds by
  default, ``reply_delay_seconds``), which looks less robotic than an instant
  click; the old 40–80 minute timer remains available in ``plan_reply_time``;
* on startup it also looks for a pending choice and answers it right away;
* it only answers when the message is the correct pack (8ФМ Classic) and has no
  modifiers — otherwise the human plays the move themselves.

The Telegram transport is optional: ``telethon`` is imported lazily, so the
offline parts of the project keep working without it.  The decision logic
(parsing, pack validation, button mapping, delay planning) is transport-free
and unit-tested.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .agent import EMOJI_CARDS

# Moscow time is used by the game's "Ход заканчивается" line.
MSK = timezone(timedelta(hours=3))

# emoji -> card_id, with the two "closed" markers removed.
_CARD_EMOJI: dict[str, str] = {
    card_id: emoji
    for emoji, card_id in EMOJI_CARDS.items()
    if card_id not in ("__closed__",)
}

DEFAULT_PACK_MARKER = "8ФМ Classic"
DEFAULT_MODIFIER_PATTERNS: tuple[str, ...] = (
    r"(?i)модификатор",
    r"(?i)\bmod\b",
    r"(?i)модиф\b",
)

_DEADLINE_RE = re.compile(
    r"Ход\s+заканчивается\s*:?\s*(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2})",
    re.IGNORECASE,
)
_CHOICE_RE = re.compile(r"Выберите\s+(?:свою\s+)?карту\s*!?", re.IGNORECASE)
_PLAYED_RE = re.compile(r"Вы\s+сыграли\s+карту", re.IGNORECASE)

# How many recent messages to look through on startup for a pending choice.
STARTUP_SCAN_LIMIT = 25


class PackError(Exception):
    """Raised when a message is not the expected, unmodified Classic game."""


@dataclass
class PackCheck:
    ok: bool
    reason: str


@dataclass
class AssistantConfig:
    api_id: int | None = None
    api_hash: str | None = None
    phone: str | None = None
    chat: str = "@phys8math8bot"
    player_name: str = "Леончик"
    model_path: str = "models/lu_luna.json"
    session: str = "data/telegram-assistant"
    min_delay_minutes: float = 40.0
    max_delay_minutes: float = 80.0
    deadline_margin_minutes: float = 10.0
    reply_delay_seconds: float = 20.0
    seed: int | None = None
    pack_marker: str = DEFAULT_PACK_MARKER
    modifier_patterns: tuple[str, ...] = DEFAULT_MODIFIER_PATTERNS
    dry_run: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssistantConfig":
        known = {field_name for field_name in cls.__dataclass_fields__}
        payload = {key: value for key, value in data.items() if key in known}
        if "modifier_patterns" in payload and payload["modifier_patterns"] is not None:
            payload["modifier_patterns"] = tuple(payload["modifier_patterns"])
        return cls(**payload)

    @classmethod
    def from_file(cls, path: str | Path) -> "AssistantConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


@dataclass
class ButtonTarget:
    open_row: int
    open_col: int
    close_row: int
    close_col: int


@dataclass
class ChoicePlan:
    message_id: int | None
    card_id: str | None
    is_closed: bool
    label: str
    target: ButtonTarget | None
    reply_at: datetime
    deadline: datetime | None
    policy_score: float
    probability: float
    warnings: list[str] = field(default_factory=list)

    @property
    def row(self) -> int | None:
        if self.target is None:
            return None
        return self.target.close_row if self.is_closed else self.target.open_row

    @property
    def col(self) -> int | None:
        if self.target is None:
            return None
        return self.target.close_col if self.is_closed else self.target.open_col

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "card_id": self.card_id,
            "is_closed": self.is_closed,
            "label": self.label,
            "row": self.row,
            "col": self.col,
            "reply_at": self.reply_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "policy_score": self.policy_score,
            "probability": self.probability,
            "warnings": self.warnings,
        }


def card_emoji(card_id: str | None) -> str | None:
    if card_id is None:
        return None
    return _CARD_EMOJI.get(card_id)


def card_label(card_id: str | None, is_closed: bool) -> str:
    if is_closed:
        return f"❌ {card_id or '?'} (закрыто)"
    return f"{card_emoji(card_id) or '?'} {card_id or '?'} (открыто)"


def parse_deadline(text: str) -> datetime | None:
    """Read the '⌛️ Ход заканчивается: DD.MM.YYYY HH:MM:SS МСК' line."""
    match = _DEADLINE_RE.search(text)
    if not match:
        return None
    try:
        naive = datetime.strptime(match.group(1), "%d.%m.%Y %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=MSK)


def is_choice_message(text: str) -> bool:
    """True only for an open choice that has not been played yet.

    After a move the bot keeps the "Выберите свою карту!" line and the inline
    keyboard, but appends "Вы сыграли карту ...".  Such a message is already
    answered and must not be clicked again.
    """
    if _PLAYED_RE.search(text):
        return False
    return bool(_CHOICE_RE.search(text))


def check_pack(text: str, config: AssistantConfig) -> PackCheck:
    """Confirm the message is the expected pack without modifiers."""
    if not re.search(re.escape(config.pack_marker), text, re.IGNORECASE):
        return PackCheck(False, f"Не найден маркер пака '{config.pack_marker}'")
    for pattern in config.modifier_patterns:
        if re.search(pattern, text):
            return PackCheck(False, f"Обнаружен модификатор (шаблон: {pattern})")
    return PackCheck(True, "8ФМ Classic без модификаторов")


def _button_text(button: Any) -> str:
    return str(getattr(button, "text", button)).strip()


def find_button_target(
    button_rows: Sequence[Sequence[Any]],
    card_id: str | None,
) -> ButtonTarget | None:
    """Locate the open/close buttons for one card in an inline keyboard.

    ``button_rows`` is a list of rows, each a list of button labels (or button
    objects).  The open button shows the card emoji; the close button in the
    same column shows ❌.
    """
    emoji = card_emoji(card_id)
    if emoji is None:
        return None
    grid = [[_button_text(button) for button in row] for row in button_rows]
    for row_index, row in enumerate(grid):
        for col_index, label in enumerate(row):
            if label == emoji or emoji in label:
                close_row, close_col = _find_close_button(grid, col_index)
                if close_row is None:
                    return None
                return ButtonTarget(
                    open_row=row_index,
                    open_col=col_index,
                    close_row=close_row,
                    close_col=close_col,
                )
    return None


def _find_close_button(grid: Sequence[Sequence[str]], column: int) -> tuple[int | None, int]:
    for row_index, row in enumerate(grid):
        if column < len(row) and row[column] in ("❌", "🚫"):
            return row_index, column
    # Fallback: any ❌ button, keeping the column when possible.
    for row_index, row in enumerate(grid):
        for col_index, label in enumerate(row):
            if label in ("❌", "🚫"):
                return row_index, col_index
    return None, column


def plan_reply_time(
    now: datetime,
    *,
    deadline: datetime | None,
    rng: random.Random,
    min_delay_minutes: float,
    max_delay_minutes: float,
    deadline_margin_minutes: float,
    immediate: bool = False,
) -> datetime:
    """Decide when to press the button.

    Normal mode waits a random ~1 hour (± the configured window).  ``immediate``
    (used on startup for a pending choice) replies right away.  Either way the
    reply never lands after ``deadline - margin``.
    """
    if immediate:
        return now
    low = max(0.0, min(min_delay_minutes, max_delay_minutes))
    high = max(low, max_delay_minutes)
    reply_at = now + timedelta(minutes=rng.uniform(low, high))
    if deadline is not None:
        latest = deadline - timedelta(minutes=max(0.0, deadline_margin_minutes))
        if reply_at > latest:
            reply_at = max(now, latest)
    return reply_at


def choose_plan(
    message_text: str,
    *,
    net: Any,
    config: AssistantConfig,
    message_id: int | None = None,
    button_rows: Sequence[Sequence[Any]] | None = None,
    now: datetime | None = None,
    immediate: bool = False,
    rng: random.Random | None = None,
) -> ChoicePlan:
    """Full offline decision: validate, rank with lu_luna, map to a button."""
    from .rl_train import rank_message_actions

    pack = check_pack(message_text, config)
    if not pack.ok:
        raise PackError(pack.reason)
    if not is_choice_message(message_text):
        raise PackError("В сообщении нет активного выбора карты")

    ranked = rank_message_actions(net, message_text, config.player_name)
    candidates = ranked["candidates"]
    if not candidates:
        raise PackError("Политика не вернула ни одного хода")
    best = candidates[0]
    warnings = list(ranked.get("warnings", []))
    warnings.extend(w for w in candidates[0].get("warnings", []) if w not in warnings)

    target = None
    if button_rows is not None:
        target = find_button_target(button_rows, best["card_id"])
        if target is None:
            warnings.append("Не найдена кнопка для выбранной карты; ход не будет нажат")

    deadline = parse_deadline(message_text)
    moment = now or datetime.now(timezone.utc)
    rng = rng or random.Random(config.seed)
    reply_at = plan_reply_time(
        moment,
        deadline=deadline,
        rng=rng,
        min_delay_minutes=config.min_delay_minutes,
        max_delay_minutes=config.max_delay_minutes,
        deadline_margin_minutes=config.deadline_margin_minutes,
        immediate=immediate,
    )
    return ChoicePlan(
        message_id=message_id,
        card_id=best["card_id"],
        is_closed=bool(best["is_closed"]),
        label=card_label(best["card_id"], bool(best["is_closed"])),
        target=target,
        reply_at=reply_at,
        deadline=deadline,
        policy_score=float(best["policy_score"]),
        probability=float(best["probability"]),
        warnings=warnings,
    )


def parse_message_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _msk(moment: datetime | None) -> str:
    """Format a moment in Moscow time for the console log."""
    if moment is None:
        return "?"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(MSK).strftime("%d.%m %H:%M:%S")


_LOG_SINK: Callable[[str], None] | None = None


def set_log_sink(sink: Callable[[str], None] | None) -> None:
    """Forward assistant log lines to a UI sink (the desktop app uses this)."""
    global _LOG_SINK
    _LOG_SINK = sink


def _log(message: str) -> None:
    """Print a timestamped assistant log line."""
    stamp = datetime.now(MSK).strftime("%H:%M:%S")
    line = f"[assistant {stamp}] {message}"
    print(line, flush=True)
    sink = _LOG_SINK
    if sink is not None:
        try:
            sink(line)
        except Exception:  # noqa: BLE001 - logging must never break the bot
            pass


def _preview(text: str, limit: int = 60) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def rich_message_text(node: Any) -> str:
    """Flatten Telegram's ``RichMessage`` tree into plain text.

    Newer bots send their content in ``Message.rich_message`` instead of
    ``Message.message``.  The text is a tree of ``TextPlain`` / ``TextConcat``
    / ``TextBold`` / ``TextItalic`` / ``TextButton`` / ``TextDate`` /
    ``TextMentionName`` ... nodes; this walks it and concatenates everything.
    Both Telethon objects and ``to_dict()``-style dicts are supported.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        name = node.get("_")
        if name == "TextPlain":
            return str(node.get("text") or "")
        if name == "TextConcat":
            return "".join(rich_message_text(part) for part in (node.get("texts") or []))
        if name == "TextEmpty":
            return ""
        if "blocks" in node:
            return "\n".join(rich_message_text(block) for block in (node.get("blocks") or []))
        inner = node.get("text")
        if isinstance(inner, str):
            return inner
        return rich_message_text(inner)
    name = type(node).__name__
    if name == "TextPlain":
        return str(getattr(node, "text", "") or "")
    if name == "TextConcat":
        return "".join(rich_message_text(part) for part in (getattr(node, "texts", None) or []))
    if name == "TextEmpty":
        return ""
    blocks = getattr(node, "blocks", None)
    if blocks:
        return "\n".join(rich_message_text(block) for block in blocks)
    inner = getattr(node, "text", None)
    if isinstance(inner, str):
        return inner
    return rich_message_text(inner)


def message_text(message: Any) -> str:
    """Return a message's plain text, falling back to its ``rich_message``."""
    text = getattr(message, "message", None) or ""
    if text:
        return text
    return rich_message_text(getattr(message, "rich_message", None))


# ---------------------------------------------------------------------------
# Telethon transport


def _require_telethon() -> Any:
    try:
        import telethon  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Не установлен telethon. Установите: "
            "python -m pip install -r requirements-telegram.txt"
        ) from error
    return telethon


class TelegramAssistant:
    """Long-running Telethon client that answers pending Classic choices."""

    def __init__(self, config: AssistantConfig, net: Any):
        self.config = config
        self.net = net
        self.rng = random.Random(config.seed)
        self._handled: set[int] = set()
        self.client: Any | None = None

    async def stop(self) -> None:
        """Disconnect the Telethon client, if the assistant is running."""
        client = self.client
        if client is not None:
            try:
                await client.disconnect()
            except Exception as error:  # noqa: BLE001
                _log(f"ошибка при отключении: {error!r}")

    async def _resolve_plan(self, message: Any, *, immediate: bool) -> ChoicePlan | None:
        text = message_text(message)
        if not is_choice_message(text):
            _log(f"сообщение #{message.id}: нет строки выбора карты — пропускаю")
            return None
        buttons = getattr(message, "buttons", None)
        rows = [[_button_text(b) for b in row] for row in buttons] if buttons else None
        if rows:
            _log(f"сообщение #{message.id}: клавиатура ({len(rows)} ряд(ов)): {rows}")
        else:
            _log(f"сообщение #{message.id}: кнопок под сообщением нет")
        try:
            plan = choose_plan(
                text,
                net=self.net,
                config=self.config,
                message_id=message.id,
                button_rows=rows,
                now=datetime.now(timezone.utc),
                immediate=immediate,
                rng=self.rng,
            )
        except PackError as error:
            _log(f"сообщение #{message.id}: пропуск — {error}")
            return None
        except Exception as error:  # noqa: BLE001 - keep the bot alive, but loud
            _log(f"сообщение #{message.id}: НЕОЖИДАННАЯ ошибка — {error!r}")
            traceback.print_exc()
            return None
        if plan.target is None:
            _log(
                f"сообщение #{message.id}: не нашёл кнопку для {plan.label} — "
                "ход не отправляю, выбери сам"
            )
            return None
        _log(
            f"сообщение #{message.id}: lu_luna выбрала {plan.label} "
            f"(score {plan.policy_score:+.3f}, p={plan.probability:.2f})"
        )
        for warning in plan.warnings:
            _log(f"  предупреждение: {warning}")
        return plan

    async def _press(self, client: Any, message: Any, plan: ChoicePlan) -> None:
        if plan.target is None:
            return
        fresh = await client.get_messages(message.chat_id, ids=message.id)
        if fresh is None or getattr(fresh, "buttons", None) is None:
            _log(f"сообщение #{message.id}: кнопок уже нет — выбор закрыт, пропускаю")
            return
        if not is_choice_message(message_text(fresh)):
            _log(f"сообщение #{message.id}: уже сыграно/закрыто — пропускаю")
            return
        current = [[_button_text(b) for b in row] for row in fresh.buttons]
        if find_button_target(current, plan.card_id) is None:
            _log(f"сообщение #{message.id}: кнопки изменились, пропускаю")
            return
        label = plan.label
        if self.config.dry_run:
            _log(f"сообщение #{message.id}: DRY-RUN — нажал бы ({plan.row},{plan.col}) {label}")
            return
        await fresh.click(plan.row, plan.col)
        _log(
            f"сообщение #{message.id}: НАЖАЛ ({plan.row},{plan.col}) — {label} "
            f"(p={plan.probability:.2f})"
        )

    async def _on_new(self, client: Any, message: Any) -> None:
        if message.id in self._handled:
            return
        plan = await self._resolve_plan(message, immediate=True)
        if plan is None:
            return
        self._handled.add(message.id)
        delay = max(0.0, self.config.reply_delay_seconds)
        if delay > 0:
            _log(f"сообщение #{message.id}: жду {delay:g} с перед нажатием")
            await asyncio.sleep(delay)
        else:
            _log(f"сообщение #{message.id}: отвечаю сразу, без задержки")
        await self._press(client, message, plan)

    async def run(self) -> None:
        _require_telethon()
        from telethon import TelegramClient, events

        config = self.config
        if config.api_id is None or not config.api_hash:
            raise RuntimeError("Не заданы api_id/api_hash (env или --config)")
        _log("запуск ассистента")
        _log(f"чат: {config.chat} | игрок: {config.player_name} | модель: {config.model_path}")
        _log(f"сессия: {config.session}")
        _log(f"новые ходы: отвечаю через {config.reply_delay_seconds:g} с")
        if config.dry_run:
            _log("РЕЖИМ DRY-RUN: кнопки нажиматься не будут")
        Path(config.session).parent.mkdir(parents=True, exist_ok=True)
        client = TelegramClient(config.session, config.api_id, config.api_hash)
        self.client = client
        await client.start(phone=config.phone)
        me = await client.get_me()
        _log(
            "аккаунт: "
            f"id={getattr(me, 'id', '?')}, username={getattr(me, 'username', None)}, "
            f"name={getattr(me, 'first_name', None)}, phone={getattr(me, 'phone', None)}"
        )
        entity = await client.get_entity(config.chat)
        _log(f"подключено к {config.chat} (entity id={getattr(entity, 'id', '?')})")

        def _startup_result(task: Any) -> None:
            if task.cancelled():
                return
            error = task.exception()
            if error is not None:
                _log(f"ошибка обработки стартового выбора: {error!r}")
                traceback.print_exception(type(error), error, error.__traceback__)

        # Startup: look for a choice that is open right now and answer it after
        # the same short pause as a live message.  Stop at the newest
        # choice message; if its buttons are already gone it was answered, so
        # older choices must not be touched.
        _log("старт: проверяю, есть ли открытый выбор сейчас...")
        try:
            recent = await client.get_messages(entity, limit=STARTUP_SCAN_LIMIT)
        except Exception as error:  # noqa: BLE001
            _log(f"старт: не удалось получить историю чата — {error!r}")
            traceback.print_exc()
            recent = []
        recent = [m for m in (recent or []) if m is not None]
        _log(f"старт: получено {len(recent)} последних сообщений из чата")

        # Diagnostic: confirm we can read the text of the newest keyboard
        # message.  Bots may send it as a plain ``message`` or as a new-style
        # ``rich_message`` tree; dump the raw object only if both are empty.
        for message in recent:
            if getattr(message, "buttons", None):
                extracted = message_text(message)
                _log(
                    "диагностика: текст последнего сообщения с кнопками — "
                    f"{len(extracted)} симв.: {_preview(extracted, 120)}"
                )
                if not extracted:
                    _log("  текст пуст даже с rich_message — дамп для разбора:")
                    try:
                        payload = message.to_dict()
                    except Exception as error:  # noqa: BLE001
                        payload = {"error": repr(error)}
                    try:
                        dumped = json.dumps(payload, ensure_ascii=False, default=str)
                    except Exception as error:  # noqa: BLE001
                        dumped = f"не удалось сериализовать: {error!r}"
                    _log(f"  to_dict: {dumped}")
                break

        scanned = 0
        pending = None
        for message in recent:
            text = message_text(message)
            has_buttons = bool(getattr(message, "buttons", None))
            choice = is_choice_message(text)
            _log(
                f"  #{message.id} ({_msk(getattr(message, 'date', None))}): "
                f"out={getattr(message, 'out', None)}, кнопки={has_buttons}, "
                f"выбор={choice}, len={len(text)} | {_preview(text)}"
            )
            if getattr(message, "out", False) or not text:
                continue
            scanned += 1
            if choice:
                pending = message
                break
        if pending is None:
            _log(f"открытого выбора нет (подходящих сообщений: {scanned}) — жду новые")
            if not recent:
                _log(
                    "  подсказка: история пуста — проверь, что сессия залогинена в тот же "
                    "аккаунт, где идёт игра (Леончик), и что игра именно в этом чате"
                )
                try:
                    dialogs = await client.get_dialogs(limit=10)
                    _log("  последние диалоги аккаунта:")
                    for dialog in dialogs:
                        _log(
                            f"    - {getattr(dialog, 'name', '?')} "
                            f"(id={getattr(dialog, 'id', '?')})"
                        )
                except Exception as error:  # noqa: BLE001
                    _log(f"  не удалось получить диалоги — {error!r}")
        elif getattr(pending, "buttons", None):
            _log(
                f"найден открытый выбор #{pending.id} — отвечаю через "
                f"{config.reply_delay_seconds:g} с"
            )
            # Answer in the background so the pause does not delay handler
            # registration; otherwise a message arriving during the pause
            # could be missed.
            startup_task = asyncio.create_task(self._on_new(client, pending))
            startup_task.add_done_callback(_startup_result)
        else:
            _log(f"последний выбор #{pending.id} уже закрыт (кнопок нет) — ничего не нажимаю")

        @client.on(events.NewMessage(chats=entity))
        async def _handler(event: Any) -> None:
            if event.message.out:
                return
            _log(f"замечено новое сообщение #{event.message.id}")
            try:
                await self._on_new(client, event.message)
            except Exception as error:  # noqa: BLE001
                _log(f"ошибка обработки нового сообщения #{event.message.id}: {error!r}")
                traceback.print_exc()

        @client.on(events.MessageEdited(chats=entity))
        async def _edit_handler(event: Any) -> None:
            if event.message.out:
                return
            _log(f"сообщение #{event.message.id} отредактировано — перепроверяю")
            try:
                await self._on_new(client, event.message)
            except Exception as error:  # noqa: BLE001
                _log(f"ошибка обработки изменения #{event.message.id}: {error!r}")
                traceback.print_exc()

        _log("жду новые сообщения (Ctrl+C для выхода)")
        await client.run_until_disconnected()


def build_config(
    *,
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> AssistantConfig:
    data: dict[str, Any] = {}
    if config_path:
        data.update(json.loads(Path(config_path).read_text(encoding="utf-8")))
    if overrides:
        data.update({key: value for key, value in overrides.items() if value is not None})
    return AssistantConfig.from_dict(data)
