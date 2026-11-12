"""One interactive 3-7 player Classic game in the terminal.

This module is deliberately separate from the RL trainer: it is a manual
verification tool for the simulator and the scoring rules.  The human occupies
seat 0, two to six bots occupy the other seats, and every turn prints a
message-like state before asking for a numbered legal action.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING, Callable, Sequence

from .baselines import (
    ComboPolicy,
    GreedyPolicy,
    Policy,
    RandomPolicy,
    SupervisedPolicy,
)
from .env import ClassicEnv, Observation
from .rules import Action

if TYPE_CHECKING:
    from .ensemble import EnsemblePolicy
    from .policy import PolicyModel
    from .rl import RLNetwork


CARD_EMOJI: dict[str, str] = {
    "alaksiej": "🐷",
    "olympiad": "🏆",
    "evgen": "🤫",
    "sergay": "🤓",
    "jaman": "🥰",
    "vanya": "🤯",
    "amogus": "ඞ",
    "marat": "🇰🇵",
    "akim": "🍌",
    "leonidia": "🤠",
    "veeeep": "🧨",
    "detector": "⚠",
    "alexandr": "😮",
    "ultramax": "🎉",
}

DISPLAY_NAMES: dict[str, str] = {
    "alaksiej": "АЛЕКСЕЙ ПШЕНИЧНЫЙ",
    "olympiad": "ОЛИМПИАДА",
    "evgen": "ЖЕНЯ",
    "sergay": "СЕРГЕЙ",
    "jaman": "ЯН-ПАВЕЛ",
    "vanya": "ИВАН ГРЕЧИХА",
    "amogus": "АМОГУС",
    "marat": "МАРАТ",
    "akim": "АКИМ",
    "leonidia": "ЛЕОНИДИЯ",
    "veeeep": "ВЕП",
    "detector": "ДЕТЕКТОР АЛЕКСЕЯ",
    "alexandr": "АЛЕХАНДР",
    "ultramax": "ШЕВЧИК МАКСИМ",
}

BOT_KINDS = ("rl", "supervised", "ensemble", "greedy", "combo", "random")

# Convenient shorthands that pin an RL bot to a concrete checkpoint.
OPPONENT_ALIASES: dict[str, tuple[str, str | None]] = {
    "lu_luna": ("rl", "models/lu_luna.json"),
    "lu-luna": ("rl", "models/lu_luna.json"),
    "classic-rl": ("rl", "models/classic-rl.json"),
    "classic_rl": ("rl", "models/classic-rl.json"),
    "ensemble": ("ensemble", None),
}

MIN_OPPONENTS = 2
MAX_OPPONENTS = 6


def resolve_opponent(token: str) -> tuple[str, str | None]:
    """Turn one ``--opponents`` token into ``(kind, model_path | None)``.

    Accepted forms: ``rl``, ``supervised``, ``greedy``, ``combo``, ``random``,
    ``rl:models/lu_luna.json`` (kind with explicit checkpoint) and the named
    shorthands ``lu_luna`` / ``classic-rl``.
    """
    key = token.strip().lower()
    if key in OPPONENT_ALIASES:
        return OPPONENT_ALIASES[key]
    if ":" in key:
        kind, _, path = key.partition(":")
        kind = kind.strip()
        path = path.strip() or None
        if kind not in BOT_KINDS:
            raise ValueError(
                f"Неизвестный тип бота: {kind!r}; доступны {', '.join(BOT_KINDS)}"
            )
        return kind, path
    if key in BOT_KINDS:
        return key, None
    raise ValueError(
        f"Неизвестный бот: {token!r}; доступны {', '.join(BOT_KINDS)} "
        f"и псевдонимы {', '.join(sorted(OPPONENT_ALIASES))}"
    )


def parse_opponents(value: str) -> tuple[str, ...]:
    tokens = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not MIN_OPPONENTS <= len(tokens) <= MAX_OPPONENTS:
        raise ValueError(
            f"Нужно от {MIN_OPPONENTS} до {MAX_OPPONENTS} ботов: "
            "--opponents rl,supervised или rl:models/lu_luna.json,rl,supervised"
        )
    for token in tokens:
        resolve_opponent(token)
    return tokens


def card_label(card_id: str) -> str:
    return f"{CARD_EMOJI.get(card_id, '?')} {DISPLAY_NAMES.get(card_id, card_id)}"


def _points(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _signed(value: float) -> str:
    text = _points(value)
    return f"+{text}" if value >= 0 else text


def _score_label(value: float) -> str:
    text = _points(value)
    number = abs(int(round(value)))
    if number % 10 == 1 and number % 100 != 11:
        word = "очко"
    elif 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        word = "очка"
    else:
        word = "очков"
    return f"{text} {word}"


def _format_play(card_id: str | None, is_closed: bool) -> str:
    if is_closed:
        return "❌"
    return CARD_EMOJI.get(card_id or "", "?")


def _format_table_row(observation: Observation, index: int) -> str:
    icons: list[str] = []
    for position, (card_id, is_closed) in enumerate(observation.all_plays[index]):
        if position == 5:
            icons.append("│")
        icons.append(_format_play(card_id, is_closed))
    return " ".join(icons) if icons else "—"


def _format_hand(hand: Sequence[str]) -> str:
    counts = Counter(hand)
    parts: list[str] = []
    for card_id in sorted(counts, key=lambda item: DISPLAY_NAMES.get(item, item)):
        suffix = f" ×{counts[card_id]}" if counts[card_id] > 1 else ""
        parts.append(f"{card_label(card_id)}{suffix}")
    return ", ".join(parts)


def _format_action(action: Action, *, reveal_card: bool) -> str:
    if action.is_closed:
        if reveal_card:
            return f"❌ {card_label(action.card_id)} (закрыто)"
        return "❌ закрыто"
    return f"{card_label(action.card_id)} (открыто)"


def _action_sort_key(action: Action) -> tuple[str, bool]:
    return (DISPLAY_NAMES.get(action.card_id, action.card_id), action.is_closed)


def _print_state(observation: Observation, names: Sequence[str]) -> None:
    arrow = "⬆️" if observation.round_number == 1 else "⬇️"
    print("\n👾 ИГРА 8 ФИЗМАТОВ #1")
    print(f"{arrow} Раунд {observation.round_number}/2 | 🃏 Карта {observation.turn_number}/5")
    print("🃏 8ФМ Classic\n")
    for index, name in enumerate(names):
        marker = " (вы)" if index == observation.player else ""
        print(
            f"▫ 👤 {name}{marker} ({_points(observation.scores[index])}): "
            f"{_format_table_row(observation, index)}"
        )
    print(f"\nВаша рука ({len(observation.hand)}): {_format_hand(observation.hand)}")
    if observation.turn_number < 5:
        print(
            "Передача после хода: ваша рука → "
            f"{names[observation.pass_receiver]}; вы получите руку от "
            f"{names[observation.pass_supplier]}."
        )
    elif observation.round_number == 2:
        print("После этого хода игра завершится.")
    else:
        print("После этого хода раунд завершится.")


def _print_actions(actions: Sequence[Action]) -> None:
    print("\nВыберите свою карту!")
    for index, action in enumerate(actions, start=1):
        if action.is_closed:
            print(f"  {index:>2}. {card_label(action.card_id)} — закрыто (2 очка)")
        else:
            print(f"  {index:>2}. {card_label(action.card_id)} — открыто")


def _ask_choice(input_fn: Callable[[str], str], count: int) -> int | None:
    while True:
        try:
            raw = input_fn(f"Ваш выбор (1-{count}, q — выход): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nИгра прервана.")
            return None
        if raw.lower() in ("q", "quit", "exit", "выход"):
            return None
        try:
            value = int(raw)
        except ValueError:
            print("Введите номер варианта или q для выхода.")
            continue
        if 1 <= value <= count:
            return value - 1
        print(f"Нужен номер от 1 до {count}.")


def _print_turn_result(
    record: dict,
    names: Sequence[str],
    *,
    human_index: int,
) -> None:
    print(f"\n── Ход {record['turn']}/5, раунд {record['round']} ──")
    for index, name in enumerate(names):
        action = record["actions"][index]
        delta = record["deltas"][index]
        score = record["scores"][index]
        label = _format_action(action, reveal_card=index == human_index)
        print(f"  {name}: {label} → {_signed(delta)} (всего {_points(score)})")


def _print_final(env: ClassicEnv, names: Sequence[str]) -> None:
    print("\n══ Игра окончена ══")
    ranks = env.final_ranks()
    order = sorted(
        range(len(names)),
        key=lambda index: (-env.scores[index], ranks[index], index),
    )
    for index in order:
        print(f"  {ranks[index]}. {names[index]} — {_score_label(env.scores[index])}")
    winners = [names[index] for index in range(len(names)) if ranks[index] == 1]
    print(f"\nПобедитель: {', '.join(winners)}")
    human_score = env.scores[0]
    if ranks[0] == 1:
        print("Вы победили!")
    print(
        f"Ваш счёт: {_points(human_score)}. "
        f"Цель >60: {'выполнена' if human_score > 60 else 'не выполнена'}."
    )


def _spec_label(kind: str, path: str | None) -> str:
    if not path:
        return kind
    stem = path.replace("\\", "/").rsplit("/", 1)[-1]
    if stem.endswith(".json"):
        stem = stem[:-5]
    return f"{kind}:{stem}"


def _load_rl_policy(path: str, cache: dict[str, "RLNetwork"]) -> Policy:
    from .rl import RLGreedyPolicy, load_rl

    if path not in cache:
        cache[path] = load_rl(path)
    return RLGreedyPolicy(cache[path])


def _load_supervised_policy(path: str, seed: int, cache: dict[str, "PolicyModel"]) -> Policy:
    from .policy import load_policy

    if path not in cache:
        cache[path] = load_policy(path)
    return SupervisedPolicy(cache[path], seed=seed)


def _load_ensemble_policy(
    path: str | None, seed: int, cache: dict[str, "EnsemblePolicy"]
) -> Policy:
    from .ensemble import default_ensemble, load_ensemble

    key = path or "__default__"
    if key not in cache:
        cache[key] = load_ensemble(path) if path else default_ensemble(seed=seed)
    return cache[key]


def _make_policy(
    kind: str,
    path: str | None,
    seed: int,
    default_model_path: str,
    default_supervised_path: str,
    rl_cache: dict[str, "RLNetwork"],
    supervised_cache: dict[str, "PolicyModel"],
    ensemble_cache: dict[str, "EnsemblePolicy"],
) -> Policy:
    if kind == "greedy":
        return GreedyPolicy(seed=seed)
    if kind == "combo":
        return ComboPolicy(seed=seed)
    if kind == "random":
        return RandomPolicy(seed=seed)
    if kind == "supervised":
        return _load_supervised_policy(
            path or default_supervised_path, seed, supervised_cache
        )
    if kind == "rl":
        return _load_rl_policy(path or default_model_path, rl_cache)
    if kind == "ensemble":
        return _load_ensemble_policy(path, seed, ensemble_cache)
    raise ValueError(f"Unknown bot kind: {kind}")


def play_interactive_game(
    *,
    name: str = "Леончик",
    opponents: Sequence[str] = ("rl", "supervised"),
    seed: int | None = None,
    model_path: str = "models/classic-rl.json",
    supervised_model_path: str = "models/classic-policy.json",
    input_fn: Callable[[str], str] = input,
) -> int:
    """Play exactly one 3-7 player game with a human in seat 0."""
    specs = [resolve_opponent(token) for token in opponents]
    if not MIN_OPPONENTS <= len(specs) <= MAX_OPPONENTS:
        raise ValueError(
            f"Нужно от {MIN_OPPONENTS} до {MAX_OPPONENTS} ботов"
        )
    players = len(specs) + 1
    if seed is None:
        seed = random.randrange(1 << 30)
    env = ClassicEnv(players, seed=seed)
    rl_cache: dict[str, "RLNetwork"] = {}
    supervised_cache: dict[str, "PolicyModel"] = {}
    ensemble_cache: dict[str, "EnsemblePolicy"] = {}
    names = [name]
    policies: list[Policy | None] = [None]
    for index, (kind, path) in enumerate(specs, start=1):
        names.append(f"Бот {index} ({_spec_label(kind, path)})")
        policies.append(
            _make_policy(
                kind,
                path,
                seed + index,
                model_path,
                supervised_model_path,
                rl_cache,
                supervised_cache,
                ensemble_cache,
            )
        )

    print("👾 ИГРА 8 ФИЗМАТОВ #1 — 8ФМ Classic")
    rivals = ", ".join(names[1:])
    print(f"Вы: {name}. Соперники: {rivals}.")
    print("Два раунда по 5 ходов; закрытая карта даёт 2 очка. Ходы одновременные.")
    print("Закрытые карты на столе показаны как ❌; их вклад в счёт учитывается.")
    print(f"Seed партии: {seed}")

    while not env.done:
        observations = env.observations()
        human_observation = observations[0]
        _print_state(human_observation, names)
        actions = sorted(human_observation.legal, key=_action_sort_key)
        _print_actions(actions)
        choice = _ask_choice(input_fn, len(actions))
        if choice is None:
            print("\nИгра прервана.")
            return 1
        human_action = actions[choice]
        print(f"\nВы выбрали: {_format_action(human_action, reveal_card=True)}")

        selected = [human_action]
        for index in range(1, players):
            policy = policies[index]
            if policy is None:
                raise RuntimeError("Bot seat has no policy")
            selected.append(policy(observations[index]))
        result = env.step(selected)
        record = result.info["record"]
        _print_turn_result(record, names, human_index=0)

        if result.done:
            break
        if result.observations[0].round_number != record["round"]:
            print("\n══ Раунд завершён. Новая раздача, передача вниз. ══")

    _print_final(env, names)
    return 0
