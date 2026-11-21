"""Tests for the tournament engine."""

from __future__ import annotations
from pathlib import Path

from eightfm.tournament import (
    build_participants,
    build_seat_plan,
    run_tournament,
    summarize_tournament,
    SeatRecord,
    DEFAULT_PARTICIPANTS,
    DEFAULT_MODELS,
)


def test_build_participants_loads_all_policies() -> None:
    # Only test with models that exist
    existing_models = {k: v for k, v in DEFAULT_MODELS.items() if v == "" or Path(v).exists()}
    existing_names = tuple(existing_models.keys())
    participants = build_participants(
        names=existing_names,
        models=existing_models,
        seed=42,
    )
    assert len(participants) == len(existing_names)
    names = {p.name for p in participants}
    assert names == set(existing_names)
    # Check all policies are callable
    for p in participants:
        assert callable(p.policy)


def test_build_seat_plan_rotates_correctly() -> None:
    # 3 players: window of 3 from 6, rotating
    names = DEFAULT_PARTICIPANTS
    plan_0 = build_seat_plan(3, 0, names)
    plan_1 = build_seat_plan(3, 1, names)
    plan_2 = build_seat_plan(3, 2, names)
    plan_3 = build_seat_plan(3, 3, names)
    plan_4 = build_seat_plan(3, 4, names)
    plan_5 = build_seat_plan(3, 5, names)

    # Each should have 3 unique names
    assert len(plan_0) == 3
    assert len(set(plan_0)) == 3

    # Over 6 games (one full cycle with 6 participants), each policy appears exactly 3 times at 3p
    all_seats = plan_0 + plan_1 + plan_2 + plan_3 + plan_4 + plan_5
    from collections import Counter
    counts = Counter(all_seats)
    for name in names:
        assert counts[name] == 3, f"{name}: {counts[name]}"


def test_build_seat_plan_six_players() -> None:
    # 6 players: all 6 participants
    names = DEFAULT_PARTICIPANTS
    plan = build_seat_plan(6, 0, names)
    assert len(plan) == 6
    # Should have all 6 unique
    assert len(set(plan)) == 6


def test_build_seat_plan_seven_players() -> None:
    # 7 players: all 6 + 1 duplicate
    names = DEFAULT_PARTICIPANTS
    plan = build_seat_plan(7, 0, names)
    assert len(plan) == 7
    assert len(set(plan)) == 6


def test_seat_plan_deterministic() -> None:
    # Same inputs should give same output
    plan1 = build_seat_plan(4, 10, DEFAULT_PARTICIPANTS)
    plan2 = build_seat_plan(4, 10, DEFAULT_PARTICIPANTS)
    assert plan1 == plan2


def test_tournament_every_policy_500_games() -> None:
    # Quick test with fewer games per player count
    # Only use existing models
    existing_models = {k: v for k, v in DEFAULT_MODELS.items() if v == "" or Path(v).exists()}
    existing_names = tuple(existing_models.keys())

    report = run_tournament(
        games_per_player_count=10,
        player_counts=(3, 4, 5),
        seed_base=999999,
        seed=1,
        names=existing_names,
        models=existing_models,
    )

    # 3 player counts × 10 games = 30 total games
    assert report["games"] == 30
    assert report["games_per_player_count"] == 10
    assert report["player_counts"] == [3, 4, 5]

    # Each policy gets: 3p→10*3/n, 4p→10*4/n, 5p→10*5/n games
    n = len(existing_names)
    expected_total = (10 * 3 // n) + (10 * 4 // n) + (10 * 5 // n)
    for name in existing_names:
        entry = report["participants"][name]
        assert entry["games"] == expected_total, f"{name}: {entry['games']} != {expected_total}"


def test_tournament_full_distribution() -> None:
    # Only use existing models
    existing_models = {k: v for k, v in DEFAULT_MODELS.items() if v == "" or Path(v).exists()}
    existing_names = tuple(existing_models.keys())

    report = run_tournament(
        games_per_player_count=100,
        player_counts=(3, 4, 5, 6, 7),
        seed_base=120000,
        seed=0,
        names=existing_names,
        models=existing_models,
    )

    assert report["games"] == 500
    n = len(existing_names)
    # Each policy: 3*60 + 4*80 + 5*100 + 6*120 + 7*140 all divided by N
    expected = 500 * 5 // n  # Since total games = 500 * 5 / n per policy
    for name in existing_names:
        entry = report["participants"][name]
        assert entry["games"] == expected, f"{name}: {entry['games']} != {expected}"


def test_tournament_per_player_count_distribution() -> None:
    existing_models = {k: v for k, v in DEFAULT_MODELS.items() if v == "" or Path(v).exists()}
    existing_names = tuple(existing_models.keys())

    report = run_tournament(
        games_per_player_count=100,
        player_counts=(3, 4, 5, 6, 7),
        seed_base=120000,
        seed=0,
        names=existing_names,
        models=existing_models,
    )

    # Check per-player-count game counts
    n = len(existing_names)
    expected_per_pc = {3: 100*3//n, 4: 100*4//n, 5: 100*5//n, 6: 100*6//n, 7: 100*7//n}
    for name in existing_names:
        for pc in (3, 4, 5, 6, 7):
            entry = report["by_player_count"][str(pc)][name]
            expected = expected_per_pc[pc]
            assert entry["games"] == expected, (
                f"{name} at {pc}p: {entry['games']} != {expected}"
            )


def test_tournament_no_duplicate_participants_in_same_game_under_5() -> None:
    # For 3-4 players, no policy should appear twice in the same game
    for game_index in range(100):
        plan = build_seat_plan(3, game_index, DEFAULT_PARTICIPANTS)
        assert len(plan) == len(set(plan)), f"Duplicates in game {game_index}: {plan}"

        plan = build_seat_plan(4, game_index, DEFAULT_PARTICIPANTS)
        assert len(plan) == len(set(plan)), f"Duplicates in game {game_index}: {plan}"


def test_tournament_deterministic_seeds() -> None:
    # Same seeds should produce identical results
    existing_models = {k: v for k, v in DEFAULT_MODELS.items() if v == "" or Path(v).exists()}
    existing_names = tuple(existing_models.keys())
    report1 = run_tournament(
        games_per_player_count=5,
        player_counts=(3, 4),
        seed_base=11111,
        seed=22222,
        names=existing_names,
        models=existing_models,
    )
    report2 = run_tournament(
        games_per_player_count=5,
        player_counts=(3, 4),
        seed_base=11111,
        seed=22222,
        names=existing_names,
        models=existing_models,
    )

    assert report1 == report2


def test_tournament_different_seeds_different_results() -> None:
    existing_models = {k: v for k, v in DEFAULT_MODELS.items() if v == "" or Path(v).exists()}
    existing_names = tuple(existing_models.keys())
    report1 = run_tournament(
        games_per_player_count=5,
        player_counts=(3, 4),
        seed_base=11111,
        seed=22222,
        names=existing_names,
        models=existing_models,
    )
    report2 = run_tournament(
        games_per_player_count=5,
        player_counts=(3, 4),
        seed_base=11111,
        seed=33333,
        names=existing_names,
        models=existing_models,
    )

    # Should differ in at least one participant's win_rate
    diff = any(
        report1["participants"][name]["win_rate"]
        != report2["participants"][name]["win_rate"]
        for name in existing_names
    )
    assert diff


def test_summarize_tournament_empty() -> None:
    existing_models = {k: v for k, v in DEFAULT_MODELS.items() if v == "" or Path(v).exists()}
    existing_names = tuple(existing_models.keys())
    report = summarize_tournament(
        records=[],
        names=existing_names,
        player_counts=(3, 4, 5),
        games_per_player_count=10,
        seed_base=0,
    )
    assert report["games"] == 30
    for name in existing_names:
        assert report["participants"][name]["games"] == 0


def test_seat_record_dataclass() -> None:
    record = SeatRecord(
        name="luna",
        player_count=4,
        seat=1,
        rank=2,
        score=58.5,
        margin=3.2,
    )
    assert record.name == "luna"
    assert record.player_count == 4
    assert record.seat == 1
    assert record.rank == 2
    assert record.score == 58.5
    assert record.margin == 3.2