"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Täuschungsmaße exakt aus dem Labyrinth, Studienzahlen aus der vorgerechneten Datei (20 Läufe je Zelle, 40 000 Evaluationen, Seeds 0 bis 19).
Die Datei ist fest; ändern sich die Zahlen nach einer neuen Rechnung, müssen README und Hilfetexte nachgezogen werden. Längen in Zellen, Aufwand in Evaluationen."""

import numpy as np
import pytest

import dm_constants as C
import dm_evaluation as E
import dm_maze as M
import dm_search as S

PRE = E.load_precomputed()


def runs(level):
    return PRE["levels"][str(level)]["runs"]


def test_exact_deception_measures_of_the_levels():
    """README: kürzester Weg 17 / 17 / 33 / 49 / 77 Zellen (Luftlinie 17), Umwegfaktor 1.00 / 1.00 / 1.94 / 2.88 / 4.53, Fitness-Distanz-Korrelation 0.97 / 0.97 / 0.89 / 0.69 / 0.68, Fallen 0 / 0 / 1 / 1 / 2."""
    m = [E.level_metrics(lv) for lv in C.LEVELS]
    assert [x["path"] for x in m] == [17, 17, 33, 49, 77] and all(round(x["straight"]) == 17 for x in m)
    assert [round(x["detour"], 2) for x in m] == [1.00, 1.00, 1.94, 2.88, 4.53]
    assert [round(x["fdc"], 2) for x in m] == [0.97, 0.97, 0.89, 0.69, 0.68]
    assert [x["minima"] for x in m] == [0, 0, 1, 1, 2]


def test_trap_depths():
    """README: tiefste Falle 9 Zellen vor dem Ziel (Stufe 2 und 3), 4 Zellen (Stufe 4, dazu eine zweite bei 14); keine Falle bei freiem Feld und bei Öffnung links."""
    assert [E.level_metrics(lv)["trap"] for lv in (0, 1)] == [None, None] and [E.level_metrics(lv)["trap"] for lv in (2, 3, 4)] == [9.0, 9.0, 4.0]
    assert [round(d) for _, d in M.local_minima(M.make_maze(4))] == [4, 14]


def test_the_corridor_to_the_gap_in_level_one_is_on_the_way():
    """README: bei Stufe 1 liegt die Öffnung genau auf dem Weg: Umwegfaktor 1.00 trotz Wand."""
    assert M.make_maze(1).sum() > M.make_maze(0).sum() and E.level_metrics(1)["path"] == E.level_metrics(0)["path"]


def test_every_level_is_solvable_and_the_serpentine_needs_136_of_140_steps():
    """README: eine aus dem kürzesten Pfad abgeleitete Steuerfolge erreicht in allen fünf Stufen das Ziel, mit 27 / 27 / 60 / 82 / 136 von 140 Schritten."""
    out = [E.solvability(lv) for lv in C.LEVELS]
    assert all(d < 0.5 for d, _ in out) and [s for _, s in out] == [27, 27, 60, 82, 136]


def test_success_counts_per_level():
    """README: Erfolg in 20 / 20 / 20 / 11 / 0 von 20 Läufen (40 000 Evaluationen) auf Stufe 0 bis 4."""
    assert [E.success_count(runs(lv)) for lv in C.LEVELS] == [20, 20, 20, 11, 0]


def test_effort_of_the_easy_levels():
    """README: Median der Evaluationen bis zum Erfolg 64 / 126 / 190 (Stufe 0 bis 2), schlimmster Lauf 237 / 252 / 875; Stufe 3: Median 1 267."""
    meds = [E.median_solved(runs(lv)) for lv in C.LEVELS[:4]]
    assert [round(m) for m in meds] == [64, 126, 190, 1267]
    assert [max(r for r in runs(lv) if r is not None) for lv in (0, 1, 2)] == [237, 252, 875]


def test_more_budget_does_not_help_on_level_three():
    """README: Auf Stufe 3 liegen alle 11 Erfolge bei höchstens 3 355 Evaluationen (10 bis 3 000); die übrigen 9 Läufe finden den Weg auch mit 40 000 nicht."""
    ok = sorted(r for r in runs(3) if r is not None)
    assert len(ok) == 11 and ok[-1] == 3355 and E.success_count(runs(3), 3000) == 10 and sum(r is None for r in runs(3)) == 9
    assert E.success_count(runs(3), 3355) == E.success_count(runs(3), 40_000)


def test_mean_best_distance_at_the_end_of_the_study():
    """README: mittlerer bester Abstand nach 400 Generationen 0.01 / 0.01 / 0.02 / 3.86 / 7.00 Zellen; nach 1 / 10 / 50 / 100 Generationen auf Stufe 3: 7.89 / 5.31 / 4.12 / 3.95."""
    ends = [PRE["levels"][str(lv)]["mean_best_distance"][-1] for lv in C.LEVELS]
    assert [round(x, 2) for x in ends] == [0.01, 0.01, 0.02, 3.86, 7.00]
    curve = PRE["levels"]["3"]["mean_best_distance"]
    assert [round(curve[i], 2) for i in (0, 9, 49, 99)] == [7.89, 5.31, 4.12, 3.95]


def test_settings_grid_on_level_three():
    """README: Erfolge von 20 für 50 / 100 / 200 Individuen bei σ = 0.1 / 0.25 / 0.5: 4 / 7 / 16, 10 / 11 / 19, 13 / 18 / 20; Median der Evaluationen (100, 0.5) 4 075 und (200, 0.5) 3 425 gegen 1 267 in der Standardeinstellung."""
    grid = [[E.success_count(E.sweep_cell(PRE, 3, p, s)["runs"]) for s in C.SWEEP_SIGMAS] for p in C.SWEEP_POPS]
    assert grid == [[4, 7, 16], [10, 11, 19], [13, 18, 20]]
    assert E.median_solved(E.sweep_cell(PRE, 3, 100, 0.5)["runs"]) == 4075.0 and round(E.median_solved(E.sweep_cell(PRE, 3, 200, 0.5)["runs"])) == 3424 and E.median_solved(runs(3)) == 1267.0


def test_settings_grid_on_level_four_has_no_success_at_all():
    """README: Stufe 4: 0 Erfolge in allen 9 Einstellungen, also 0 von 180 Läufen."""
    cells = [x for x in PRE["sweep"] if x["level"] == 4]
    assert len(cells) == 9 and all(E.success_count(x["runs"]) == 0 for x in cells) and sum(len(x["runs"]) for x in cells) == 180


def test_the_deception_depends_on_the_genome_length(monkeypatch):
    """README (Befund beim Bau): mit 180 statt 140 Schritten löst derselbe GA Stufe 3 in 20 von 20 Läufen (Median 701 Evaluationen); die Täuschung hängt an der Länge der Steuerfolge."""
    monkeypatch.setattr(C, "STEPS", 180)
    grid = M.make_maze(3)
    solved = [S.run_ga(grid, seed, gens=400, keep_positions=False, stop_when_solved=True).first_solved for seed in range(20)]
    assert all(s is not None for s in solved) and 400 < float(np.median(solved)) < 1100


def test_preset_help_quotes_the_study():
    """PRESET_HELP: Freies Feld (20 von 20, Median 64), Täuschung (11 von 20, alle bis 3 355, übrige 9), Schlangenlinie (77 Zellen, 4.53-fach, 0 von 20, 0 von 180), Kurzer Lauf (10 von 20 bis 3 000, 11 nach 40 000)."""
    assert E.success_count(runs(0)) == 20 and round(E.median_solved(runs(0))) == 64
    assert E.success_count(runs(3)) == 11 and max(r for r in runs(3) if r is not None) == 3355
    assert E.level_metrics(4)["path"] == 77 and round(E.level_metrics(4)["detour"], 2) == 4.53 and E.success_count(runs(4)) == 0
    assert E.success_count(runs(3), 3000) == 10 and C.STUDY_BUDGET == 40_000
    for name, text in C.PRESET_HELP.items():
        assert text and name in C.PRESETS
