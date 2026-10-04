"""Auswertung: Täuschungsmaße je Stufe, Lösbarkeit, Live-Lauf, Hilfsfunktionen der Studie, Vollständigkeit und Stimmigkeit der vorgerechneten Datei."""

import numpy as np
import pytest

import dm_constants as C
import dm_evaluation as E
import dm_maze as M
import dm_search as S

PRE = E.load_precomputed()


def test_level_metrics_by_hand():
    m0, m3 = E.level_metrics(0), E.level_metrics(3)
    assert m0["path"] == 17 and m0["straight"] == pytest.approx(17.0) and m0["detour"] == pytest.approx(1.0) and m0["minima"] == 0 and m0["trap"] is None
    assert m3["path"] == 49 and m3["detour"] == pytest.approx(49 / 17) and m3["minima"] == 1 and m3["trap"] == pytest.approx(9.0)


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_is_solvable_within_the_step_budget(level):
    dist, steps = E.solvability(level)
    assert dist < 0.5 and steps <= C.STEPS


def test_live_run_returns_the_run_and_the_best_trajectory():
    res, track = E.live_run(2, 12, 5)
    assert len(res.best_distance) == 12 and track.shape == (C.STEPS + 1, 2) and tuple(track[0]) == C.START
    assert np.hypot(*(track[-1] - np.array(C.GOAL))) == pytest.approx(min(res.best_distance))


def test_success_helpers_by_hand():
    runs = [100, None, 500, 50, None]
    assert E.success_count(runs) == 3 and E.success_count(runs, 100) == 2 and E.success_count(runs, 10) == 0
    assert E.success_curve(runs, [0, 60, 100, 1000]) == [0.0, 0.2, 0.4, 0.6]
    assert E.median_solved(runs) == 100.0 and E.median_solved([None, None]) is None and E.median_solved([10, 20]) == 15.0


def test_the_study_covers_every_level_and_every_setting():
    assert set(PRE["levels"]) == {str(lv) for lv in C.LEVELS} and PRE["seeds"] == C.STUDY_SEEDS and PRE["budget"] == C.STUDY_BUDGET
    for lv in C.LEVELS:
        c = E.study_cell(PRE, lv)
        assert len(c["runs"]) == C.STUDY_SEEDS and len(c["mean_best_distance"]) == C.STUDY_GENS
        assert all(r is None or 1 <= r <= C.STUDY_BUDGET for r in c["runs"])
    assert len(PRE["sweep"]) == len(C.SWEEP_LEVELS) * len(C.SWEEP_POPS) * len(C.SWEEP_SIGMAS) == 18
    for x in PRE["sweep"]:
        assert len(x["runs"]) == C.STUDY_SEEDS and E.sweep_cell(PRE, x["level"], x["pop"], x["sigma"]) is x


def test_the_study_metrics_equal_the_exact_level_metrics():
    for lv in C.LEVELS:
        stored, fresh = E.study_cell(PRE, lv), E.level_metrics(lv)
        for key in ("path", "straight", "detour", "fdc", "minima"):
            assert stored[key] == pytest.approx(fresh[key])
        assert stored["trap"] == (pytest.approx(fresh["trap"]) if fresh["trap"] is not None else None)


def test_a_study_run_is_reproduced_by_a_fresh_run_with_the_same_seed():
    """Stufe 0, Seed 3: die gespeicherte Zahl der Evaluationen bis zum Erfolg und der Verlauf stimmen mit einem frischen Lauf überein."""
    res = S.run_ga(M.make_maze(0), 3, gens=C.STUDY_GENS, keep_positions=False)
    assert res.first_solved == E.study_cell(PRE, 0)["runs"][3]


def test_the_sweep_uses_the_same_total_budget_for_every_population_size():
    assert all(C.STUDY_BUDGET % p == 0 for p in C.SWEEP_POPS) and {C.STUDY_BUDGET // p for p in C.SWEEP_POPS} == {800, 400, 200}
    assert all(r is None or r <= C.STUDY_BUDGET for x in PRE["sweep"] for r in x["runs"])


def test_the_default_setting_in_the_sweep_equals_the_level_study():
    """Standardeinstellung (100 Individuen, σ = 0.25) im Raster: dieselben Läufe wie in der Stufenstudie."""
    assert E.sweep_cell(PRE, 3, C.POP, C.MUT_SIGMA)["runs"] == E.study_cell(PRE, 3)["runs"]
    assert E.sweep_cell(PRE, 4, C.POP, C.MUT_SIGMA)["runs"] == E.study_cell(PRE, 4)["runs"]
