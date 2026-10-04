"""Plotly-Abbildungen der Täuschungs-Demo: was die Fitness sieht gegen den wahren Weg, ein Lauf im Labyrinth, Fortschritt, Erfolgsquoten, Budgetkurven, Einstellungs-Raster. Achsen sind gesperrt (fixedrange),
damit Touch-Geräte beim Scrollen nicht zoomen. Das Raster `grid[i, j]` wird mit i als x-Achse gezeichnet."""

import numpy as np
import plotly.graph_objects as go

import dm_constants as C
import dm_maze as M

WALL_COLOR = "#3a3f47"
START_COLOR, GOAL_COLOR, ROBOT_COLOR, PATH_COLOR, TRAP_COLOR = "#54a24b", "#e45756", "#4c78a8", "#f58518", "#b279a2"
LEVEL_COLORS = {0: "#54a24b", 1: "#72b7b2", 2: "#4c78a8", 3: "#f58518", 4: "#e45756"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _arena_axes(fig):
    """Beide Achsen 0 bis N mit gleichem Maßstab (quadratische Zellen), Zeichenfläche passt sich der kleineren Seite an (constrain=domain)."""
    fig.update_xaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="x (Zellen)", constrain="domain")
    fig.update_yaxes(range=[0, C.N], showgrid=False, zeroline=False, title_text="y (Zellen)", scaleanchor="x", scaleratio=1, constrain="domain")


def _markers(fig, with_goal_circle=True):
    fig.add_trace(go.Scatter(x=[C.START[0]], y=[C.START[1]], mode="markers", marker=dict(color=START_COLOR, size=13, symbol="square", line=dict(color="white", width=1.5)), name="Start"))
    fig.add_trace(go.Scatter(x=[C.GOAL[0]], y=[C.GOAL[1]], mode="markers", marker=dict(color=GOAL_COLOR, size=16, symbol="star", line=dict(color="white", width=1.5)), name="Ziel"))
    if with_goal_circle:
        fig.add_shape(type="circle", xref="x", yref="y", x0=C.GOAL[0] - C.SUCCESS_RADIUS, x1=C.GOAL[0] + C.SUCCESS_RADIUS, y0=C.GOAL[1] - C.SUCCESS_RADIUS, y1=C.GOAL[1] + C.SUCCESS_RADIUS,
                      line=dict(color=GOAL_COLOR, dash="dot", width=1.5))


def _field_figure(grid, field, colorbar_title, colorscale, minima=False):
    """Heatmap eines Zellenfelds (Wände ausgespart), Start, Ziel und optional die lokalen Minima der Fitness."""
    z = np.where(grid | ~np.isfinite(field), np.nan, field)
    fig = go.Figure(go.Heatmap(z=z.T, x0=0.5, dx=1, y0=0.5, dy=1, colorscale=colorscale, colorbar=dict(title=dict(text=colorbar_title, side="top"), orientation="h", thickness=10, len=0.9, y=-0.2, yanchor="top"), xgap=0, ygap=0, hoverinfo="skip"))
    _markers(fig)
    if minima:
        cells = [m[0] for m in M.local_minima(grid)]
        if cells:
            fig.add_trace(go.Scatter(x=[c[0] + 0.5 for c in cells], y=[c[1] + 0.5 for c in cells], mode="markers", marker=dict(color=TRAP_COLOR, size=13, symbol="x", line=dict(color="white", width=1)),
                                     name="lokales Optimum (Falle)"))
    _arena_axes(fig)
    fig.update_layout(plot_bgcolor=WALL_COLOR)
    return _base(fig, 470, legend_y=-0.55, top=0)


def build_fitness_view(grid):
    """Was die Fitness sieht: die Luftlinie jeder Zelle zum Ziel (dunkel = nah). Fallen (lokale Minima) sind markiert."""
    return _field_figure(grid, M.euclid_field(grid), "Luftlinie zum Ziel", "Viridis", minima=True)


def build_truth_view(grid):
    """Was wirklich gilt: der Weg jeder Zelle zum Ziel in Zellen (Breitensuche)."""
    return _field_figure(grid, M.bfs_distances(grid, M.cell_of(C.GOAL)), "wahrer Weg (Zellen)", "Viridis")


def build_run_figure(grid, end_positions, track):
    """Ein Lauf: Wände, Endpositionen der letzten Generation (blau) und die beste Fahrt des Laufs (orange)."""
    z = np.where(grid, 1.0, np.nan)
    fig = go.Figure(go.Heatmap(z=z.T, x0=0.5, dx=1, y0=0.5, dy=1, colorscale=[[0, WALL_COLOR], [1, WALL_COLOR]], showscale=False, hoverinfo="skip", xgap=0, ygap=0))
    fig.add_trace(go.Scatter(x=end_positions[:, 0], y=end_positions[:, 1], mode="markers", marker=dict(color=ROBOT_COLOR, size=6, opacity=0.6), name="Endpositionen der letzten Generation"))
    fig.add_trace(go.Scatter(x=track[:, 0], y=track[:, 1], mode="lines", line=dict(color=PATH_COLOR, width=2.5), name="beste Fahrt"))
    _markers(fig)
    _arena_axes(fig)
    return _base(fig, 420, legend_y=-0.3, top=0)


def build_progress_chart(best_distance, trap):
    """Kleinster Abstand zum Ziel je Generation; gestrichelt: Erfolgskreis und (wenn vorhanden) der Abstand der tiefsten Falle."""
    gens = list(range(1, len(best_distance) + 1))
    fig = go.Figure(go.Scatter(x=gens, y=best_distance, mode="lines", line=dict(color=ROBOT_COLOR, width=2.5), name="bester Abstand"))
    fig.add_hline(y=C.SUCCESS_RADIUS, line=dict(color=GOAL_COLOR, dash="dot"), annotation_text="Zielkreis")
    if trap is not None:
        fig.add_hline(y=trap, line=dict(color=TRAP_COLOR, dash="dash"), annotation_text="tiefste Falle")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Abstand zum Ziel (Zellen)", rangemode="tozero")
    return _base(fig, 300)


def build_success_chart(pre):
    """Erfolgsquote der 20 Läufe je Stufe (innerhalb des vollen Budgets von 40 000 Evaluationen)."""
    levels = list(C.LEVELS)
    counts = [sum(1 for r in pre["levels"][str(lv)]["runs"] if r is not None) for lv in levels]
    n = pre["seeds"]
    fig = go.Figure(go.Bar(x=[C.LEVEL_LABELS[lv] for lv in levels], y=[c / n for c in counts], marker_color=[LEVEL_COLORS[lv] for lv in levels], text=[f"{c} von {n}" for c in counts], textposition="outside"))
    fig.update_yaxes(title_text="Anteil erfolgreicher Läufe", tickformat=".0%", range=[0, 1.15])
    return _base(fig, 320)


def build_budget_chart(pre):
    """Anteil erfolgreicher Läufe über dem Evaluationsbudget, je Stufe."""
    budgets = list(range(0, pre["budget"] + 1, 500))
    fig = go.Figure()
    for lv in C.LEVELS:
        runs = pre["levels"][str(lv)]["runs"]
        ys = [sum(1 for r in runs if r is not None and r <= b) / len(runs) for b in budgets]
        fig.add_trace(go.Scatter(x=budgets, y=ys, mode="lines", line=dict(color=LEVEL_COLORS[lv], width=2.5), name=C.LEVEL_LABELS[lv]))
    fig.update_xaxes(title_text="Evaluationen")
    fig.update_yaxes(title_text="Anteil erfolgreicher Läufe", tickformat=".0%", range=[-0.02, 1.05])
    return _base(fig, 340, legend_y=-0.3)


def build_sweep_chart(pre, level):
    """Zahl erfolgreicher Läufe (von 20) für Populationsgröße und Mutationsstärke bei gleichem Gesamtbudget."""
    pops, sigmas = list(C.SWEEP_POPS), list(C.SWEEP_SIGMAS)
    z = [[sum(1 for r in next(x for x in pre["sweep"] if x["level"] == level and x["pop"] == p and x["sigma"] == s)["runs"] if r is not None) for s in sigmas] for p in pops]
    fig = go.Figure(go.Heatmap(z=z, x=[f"σ = {s:g}" for s in sigmas], y=[f"{p} Individuen" for p in pops], text=[[f"{v} von {pre['seeds']}" for v in row] for row in z], texttemplate="%{text}",
                               colorscale="Blues", zmin=0, zmax=pre["seeds"], showscale=False))
    fig.update_xaxes(title_text="Mutationsstärke")
    return _base(fig, 260)
