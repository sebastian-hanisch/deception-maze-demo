"""Abbildungen: gesperrte Achsen, Spuren, Beschriftungen, Zahl der Fallen."""

import numpy as np

import dm_constants as C
import dm_evaluation as E
import dm_maze as M
import dm_visualization as V

PRE = E.load_precomputed()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    grid = M.make_maze(3)
    res, track = E.live_run(3, 10, 1)
    figs = [V.build_fitness_view(grid), V.build_truth_view(grid), V.build_run_figure(grid, res.end_positions[-1], track), V.build_progress_chart(res.best_distance, 9.0),
            V.build_success_chart(PRE), V.build_budget_chart(PRE), V.build_sweep_chart(PRE, 3), V.build_sweep_chart(PRE, 4)]
    assert all(_locked(f) for f in figs)


def test_fitness_view_marks_the_traps_only_where_there_are_some():
    free = V.build_fitness_view(M.make_maze(0))
    trap = V.build_fitness_view(M.make_maze(4))
    assert [t.name for t in free.data if t.name] == ["Start", "Ziel"]
    names = [t.name for t in trap.data if t.name]
    assert "lokales Optimum (Falle)" in names
    marks = next(t for t in trap.data if t.name == "lokales Optimum (Falle)")
    assert len(marks.x) == 2


def test_views_leave_the_walls_empty_and_show_the_two_distances():
    grid = M.make_maze(3)
    fit, truth = V.build_fitness_view(grid), V.build_truth_view(grid)
    z_fit, z_truth = np.array(fit.data[0].z), np.array(truth.data[0].z)
    assert np.isnan(z_fit).sum() == grid.sum() == np.isnan(z_truth).sum()
    assert np.nanmax(z_truth) > np.nanmax(z_fit)                                           # der wahre Weg ist länger als jede Luftlinie
    assert z_truth[20, 3] == 0 and z_fit[20, 3] == 0                                        # Zielzelle (i = 3, j = 20); die Heatmap-Daten sind nach (j, i) angeordnet


def test_run_figure_has_walls_population_best_path_and_markers():
    grid = M.make_maze(2)
    res, track = E.live_run(2, 5, 3)
    fig = V.build_run_figure(grid, res.end_positions[-1], track)
    assert [t.name for t in fig.data[1:]] == ["Endpositionen der letzten Generation", "beste Fahrt", "Start", "Ziel"] and len(fig.data[1].x) == C.POP and len(fig.data[2].x) == C.STEPS + 1


def test_progress_chart_has_the_goal_circle_and_optionally_the_trap_line():
    with_trap = V.build_progress_chart([5.0, 4.0], 9.0)
    without = V.build_progress_chart([5.0, 4.0], None)
    assert len(with_trap.layout.shapes) == 2 and len(without.layout.shapes) == 1 and list(with_trap.data[0].x) == [1, 2]


def test_success_chart_matches_the_study():
    fig = V.build_success_chart(PRE)
    counts = [E.success_count(PRE["levels"][str(lv)]["runs"]) for lv in C.LEVELS]
    assert list(fig.data[0].y) == [c / PRE["seeds"] for c in counts] and list(fig.data[0].text) == [f"{c} von {PRE['seeds']}" for c in counts]


def test_budget_chart_has_one_line_per_level_and_ends_at_the_final_success_rate():
    fig = V.build_budget_chart(PRE)
    assert len(fig.data) == len(C.LEVELS)
    for lv, trace in zip(C.LEVELS, fig.data):
        assert trace.y[-1] == E.success_count(PRE["levels"][str(lv)]["runs"]) / PRE["seeds"] and trace.y[0] == 0.0


def test_sweep_chart_shows_the_success_counts_of_the_grid():
    fig = V.build_sweep_chart(PRE, 3)
    z = np.array(fig.data[0].z)
    assert z.shape == (3, 3) and z[2, 2] == E.success_count(E.sweep_cell(PRE, 3, 200, 0.5)["runs"]) and z[1, 1] == E.success_count(E.sweep_cell(PRE, 3, 100, 0.25)["runs"])
