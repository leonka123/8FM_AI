"""Paste-a-message parsing and score-model inference for 8ФМ Classic.

The module turns a copied Telegram message into the same feature layout used
during training and exposes a small, model-only ``predict_message`` helper.
It never sends a move anywhere.  Rule-based move recommendations were removed;
the archive-trained policy and the RL policy are the supported advisers.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .model import CARD_IDS, ScoreModel, load_model, make_features


CARD_NAMES: dict[str, tuple[str, ...]] = {
    "alaksiej": ("АЛЕКСЕЙ ПШЕНИЧНЫЙ", "АЛЕКСЕЙ ПШЕНИЧЫЙ", "АЛЕКСЕЙ"),
    "olympiad": ("ОЛИМПИАДА",),
    "evgen": ("ЖЕНЯ", "ЕВГЕНИЙ"),
    "sergay": ("СЕРГЕЙ",),
    "jaman": ("ЯН-ПАВЕЛ", "ЯН ПАВЕЛ"),
    "vanya": ("ИВАН ГРЕЧИХА", "ИВАН"),
    "amogus": ("АМОГУС",),
    "marat": ("МАРАТ",),
    "akim": ("АКИМ",),
    "leonidia": ("ЛЕОНИДИЯ",),
    "veeeep": ("ВЕП", "ВЕЕЕП"),
    "detector": ("ДЕТЕКТОР АЛЕКСЕЯ", "ДЕТЕКТОР"),
    "alexandr": ("АЛЕХАНДР", "АЛЕКСАНДР"),
    "ultramax": ("УЛЬТРАМАКС", "ШЕВЧИК МАКСИМ", "ШЕВЧИК"),
}

EMOJI_CARDS = {
    "🐷": "alaksiej", "🏆": "olympiad", "🤫": "evgen", "🤓": "sergay",
    "🥰": "jaman", "🤯": "vanya", "ඞ": "amogus", "🇰🇵": "marat",
    "🍌": "akim", "🧨": "veeeep", "⚠": "detector", "❌": "__closed__",
    "🚫": "__closed__",
    # Verified against game #4734: the cowboy icon is Leonidia, not a
    # generic unknown card.
    "🤠": "leonidia",
    "😮": "alexandr", "🎉": "ultramax",
}

_EMOJI_TOKEN_RE = re.compile(
    "|".join(re.escape(emoji) for emoji in sorted(EMOJI_CARDS, key=len, reverse=True))
    + r"|[^\s\uFE0F\u200D]"
)


def _tokenize_icons(text: str) -> list[str]:
    """Split a row such as ``❌🤠🐷🐷❌`` into one token per played card."""
    return _EMOJI_TOKEN_RE.findall(text)


def _count_icons(tokens: list[str]) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    closed = 0
    for token in tokens:
        card_id = EMOJI_CARDS.get(token)
        if card_id == "__closed__":
            closed += 1
        elif card_id:
            counts[card_id] = counts.get(card_id, 0) + 1
    return counts, closed


@dataclass
class Candidate:
    card_id: str | None
    name: str
    emoji: str
    description: str


@dataclass
class PredictionEstimate:
    card_id: str | None
    name: str
    play: str
    predicted_score_delta: float
    feature_vector: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    round_number: int | None
    turn_number: int | None
    player_name: str | None
    candidates: list[PredictionEstimate]
    warnings: list[str]


@dataclass
class MessageContext:
    """Visible table state converted into the model's training layout."""

    round_number: int | None
    turn_number: int | None
    player_name: str | None
    player_count: int
    own_counts: dict[str, int]
    own_closed: int
    own_score: float
    other_table_counts: dict[str, int]
    other_table_closed: int
    mean_other_score: float
    warnings: list[str]


def _strip_played_card(message: str) -> str:
    """Remove the optional line Telegram appends after a submitted move.

    The bot's confirmation is not a candidate and is not part of the public
    state before the move. If the line is absent, the message is unchanged.
    """
    return re.split(
        r"(?mi)^\s*Вы\s+сыграли(?:\s+карту)?\b.*$",
        message,
        maxsplit=1,
    )[0]


def _card_for_name(name: str) -> str | None:
    normalized = name.upper().replace("Ё", "Е")
    for card_id, aliases in CARD_NAMES.items():
        if any(alias in normalized for alias in aliases):
            return card_id
    return None


def _parse_candidates(message: str) -> list[Candidate]:
    message = _strip_played_card(message)
    heading = re.search(r"Выберите\s+(?:свою\s+)?карту\s*!?", message, re.I)
    if not heading:
        raise ValueError("Не найден раздел 'Выберите свою карту!' в сообщении")
    if not re.search(r"8\s*ФМ\s*Classic", message, re.IGNORECASE):
        raise ValueError("Поддерживаются только сообщения 8ФМ Classic")
    section = message[heading.end():]
    matches = re.finditer(
        r"(?m)^\s*▫\s*(\S+)\s+([^\n]+)\n(.*?)(?=^\s*▫\s+|\Z)",
        section,
        re.S | re.M,
    )
    candidates: list[Candidate] = []
    for match in matches:
        emoji, name, description = match.groups()
        candidates.append(Candidate(_card_for_name(name), name.strip(), emoji.strip(), description.strip()))
    if not candidates:
        raise ValueError("Не удалось разобрать список карт после 'Выберите свою карту!' ")
    return candidates


def _parse_state(message: str, player_name: str | None) -> tuple[int | None, int | None, list[dict[str, Any]], str | None]:
    header = re.search(r"Раунд\s+(\d+)\s*/\s*2\s*\|\s*🃏\s*Карта\s+(\d+)\s*/\s*5", message)
    round_number, turn_number = (int(header.group(1)), int(header.group(2))) if header else (None, None)
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r"(?m)^\s*▫\s*👤\s*(.+?)\s*\(\d+\)(?:\s*:\s*(.*?))?\s*$", message):
        name, tail = match.groups()
        tail = tail or ""
        score_match = re.search(r"\((-?\d+)\)\s*$", tail)
        score = int(score_match.group(1)) if score_match else 0
        icon_text = tail[:score_match.start()].strip() if score_match else tail.strip()
        tokens = _tokenize_icons(icon_text)
        counts, closed = _count_icons(tokens)
        rows.append({
            "name": name.strip(),
            "counts": counts,
            "closed": closed,
            "score": score,
            "icons": icon_text,
            "tokens": tokens,
            "unknown": [token for token in tokens if token not in EMOJI_CARDS],
        })
    chosen = None
    if player_name:
        exact = [row for row in rows if row["name"].casefold() == player_name.casefold()]
        contains = [row for row in rows if player_name.casefold() in row["name"].casefold()]
        matches = exact or contains
        if len(matches) == 1:
            chosen = matches[0]
    return round_number, turn_number, rows, chosen["name"] if chosen else None


def _message_context(message: str, player_name: str | None) -> MessageContext:
    """Extract only the information visible in the pasted message."""
    message = _strip_played_card(message)
    round_number, turn_number, rows, resolved_player_name = _parse_state(message, player_name)
    warnings: list[str] = []
    if player_name and not resolved_player_name:
        warnings.append(
            f"Игрок '{player_name}' не найден среди строк с сыгранными картами; "
            "персональная история принята пустой."
        )
    if not player_name:
        warnings.append("Не указан --player: оценка контекста личных карт будет неполной.")
    if rows and len(rows) < 2:
        warnings.append("Видна только часть стола; сопернический контекст неполный.")

    # The message shows cards already played. At the start of round 2 it still
    # shows all five cards from round 1, while the trained model resets card
    # counts at the round boundary. Keep only the current round's turns.
    current_turn = turn_number or 1
    expected_visible = (
        (round_number - 1) * 5 + (current_turn - 1)
        if round_number is not None
        else None
    )
    if expected_visible is not None and any(len(row["tokens"]) != expected_visible for row in rows):
        warnings.append(
            "Количество иконок в строках не совпадает с ожидаемой историей текущего раунда; "
            "контекст карт может быть неполным."
        )
    unknown_tokens = sorted({token for row in rows for token in row["unknown"]})
    if unknown_tokens:
        warnings.append("Неизвестные иконки карт не учтены в признаках: " + " ".join(unknown_tokens))

    for row in rows:
        row["current_tokens"] = row["tokens"][-(current_turn - 1):] if current_turn > 1 else []

    own = next((row for row in rows if row["name"] == resolved_player_name), None)
    own_tokens = own["current_tokens"] if own else []
    own_counts, own_closed = _count_icons(own_tokens)
    own_score = float(own["score"]) if own else 0.0
    table_counts: dict[str, int] = {}
    table_closed = 0
    for row in rows:
        counts, closed = _count_icons(row["current_tokens"])
        for card_id, count in counts.items():
            table_counts[card_id] = table_counts.get(card_id, 0) + count
        table_closed += closed
    other_table_counts = dict(table_counts)
    for card_id, count in own_counts.items():
        other_table_counts[card_id] = max(0, other_table_counts.get(card_id, 0) - count)
    other_table_closed = max(0, table_closed - own_closed)
    other_scores = [float(row["score"]) for row in rows if row is not own]
    mean_other_score = sum(other_scores) / len(other_scores) if other_scores else 0.0
    return MessageContext(
        round_number=round_number,
        turn_number=turn_number,
        player_name=resolved_player_name,
        player_count=max(3, min(6, len(rows))) if rows else 4,
        own_counts=own_counts,
        own_closed=own_closed,
        own_score=own_score,
        other_table_counts=other_table_counts,
        other_table_closed=other_table_closed,
        mean_other_score=mean_other_score,
        warnings=warnings,
    )


def _features_from_context(
    context: MessageContext,
    *,
    card_id: str | None,
    is_closed: bool,
) -> list[float]:
    """Build exactly the vector used by model.make_features() during training."""
    return make_features(
        card_id=None if is_closed else card_id,
        is_closed=is_closed,
        players=context.player_count,
        round_number=context.round_number or 1,
        turn_number=context.turn_number or 1,
        own_counts=context.own_counts,
        table_counts=context.other_table_counts,
        own_closed=context.own_closed,
        table_closed=context.other_table_closed,
        own_score=context.own_score,
        mean_other_score=context.mean_other_score,
    )


def build_message_features(
    message: str,
    *,
    player_name: str | None,
    card_id: str | None,
    is_closed: bool = False,
) -> list[float]:
    """Convert one pasted message and one action into the model feature vector.

    This is the public, side-effect-free bridge between a Telegram message and
    the same 59-value layout that `train_model` builds from the archive. The
    caller can pass a card code such as ``"akim"`` or ``None`` for a face-down
    play. The optional "Вы сыграли карту ..." confirmation is ignored.
    """
    context = _message_context(message, player_name)
    return _features_from_context(context, card_id=card_id, is_closed=is_closed)


def predict_card(
    message: str,
    *,
    player_name: str | None,
    model: ScoreModel,
    card_id: str | None,
    is_closed: bool = False,
) -> tuple[list[float], float]:
    """Return ``(feature_vector, predicted_score_delta)`` for one action."""
    vector = build_message_features(
        message,
        player_name=player_name,
        card_id=card_id,
        is_closed=is_closed,
    )
    return vector, model.predict_features(vector)


def predict_message(
    message: str,
    *,
    player_name: str | None = None,
    model: ScoreModel | None = None,
    model_path: str = "models/classic-score.json",
) -> Prediction:
    """Parse a pasted message and rank every visible card with the score model.

    The closed action is generic: it represents closing one of the visible
    cards.  The score model predicts a score delta, not a win probability.
    """
    if model is None:
        model = load_model(model_path)
    candidates = _parse_candidates(message)
    context = _message_context(message, player_name)
    warnings = list(context.warnings)
    estimates: list[PredictionEstimate] = []
    for candidate in candidates:
        features = _features_from_context(context, card_id=candidate.card_id, is_closed=False)
        estimates.append(PredictionEstimate(
            card_id=candidate.card_id,
            name=candidate.name,
            play="открыто",
            predicted_score_delta=float(model.predict_features(features)),
            feature_vector=features,
        ))
    closed_features = _features_from_context(context, card_id=None, is_closed=True)
    estimates.append(PredictionEstimate(
        card_id=None,
        name="Карта в закрытую",
        play="закрыто",
        predicted_score_delta=float(model.predict_features(closed_features)),
        feature_vector=closed_features,
    ))
    estimates.sort(key=lambda item: (-item.predicted_score_delta, item.name, item.play != "открыто"))
    if any(candidate.card_id is None for candidate in candidates):
        warnings.append("Одна или несколько карт не распознаны; для них использована категория unknown.")
    warnings.append(
        f"Score-модель проверена на отложенных партиях: RMSE {model.validation_rmse:.2f} "
        f"против {model.validation_baseline_rmse:.2f} у среднего прогноза; это прогноз очков, не победителя."
    )
    warnings.append("Закрытый вариант общий: модель не различает, какая именно карта уходит взакрытую.")
    return Prediction(
        round_number=context.round_number,
        turn_number=context.turn_number,
        player_name=context.player_name,
        candidates=estimates,
        warnings=warnings,
    )
