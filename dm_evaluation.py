"""Auswertung: Täuschungsmaße je Stufe (exakt), Lösbarkeitsnachweis, Live-Lauf, Lesen und Aggregieren der vorgerechneten Studie.
Die Studie (Erfolgsquoten über Seeds, Budgetkurven, Einstellungs-Raster) wird beim Bau gerechnet (`generate_precomputed.py`) und hier nur gelesen."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import dm_constants as C
import dm_maze as M
import dm_robot as R
import dm_search as S

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def level_metrics(level):
    """Exakte Täuschungsmaße der Stufe: kürzester Weg (Zellen), Luftlinie Start–Ziel, Umwegfaktor, Fitness-Distanz-Korrelation, lokale Minima (Zahl) und Abstand des tiefsten Minimums zum Ziel."""
    g = M.make_maze(level)
    path = M.shortest_path(g)
    return {"path": len(path) - 1, "straight": float(np.hypot(C.START[0] - C.GOAL[0], C.START[1] - C.GOAL[1])), "detour": M.detour_ratio(g), "fdc": M.fitness_distance_correlation(g),
            "minima": len(M.local_minima(g)), "trap": M.trap_distance(g)}


def solvability(level):
    """Lösbarkeitsnachweis: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt in diesem Labyrinth ins Ziel. Gibt (Abstand am Ende, benötigte Schritte) zurück."""
    g = M.make_maze(level)
    genome = R.controller_from_path(M.shortest_path(g))
    end = R.simulate(genome[None], g)
    return float(R.distance_to_goal(end)[0]), int((np.abs(genome).sum(axis=1) > 0).sum())


def live_run(level, gens, seed):
    """Ein Lauf des GA mit Fitness = −Abstand zum Ziel auf der gewählten Stufe, samt bester Fahrt (Positionen je Schritt)."""
    g = M.make_maze(level)
    res = S.run_ga(g, seed, gens=gens)
    _, track = R.simulate(res.best_genome[None], g, trajectories=True)
    return res, track[0]


@lru_cache(maxsize=1)
def load_precomputed():
    return json.loads(PRECOMPUTED_PATH.read_text(encoding="utf-8"))


def success_count(runs, budget=None):
    """Zahl der Läufe, die den Zielkreis erreichten (höchstens `budget` Evaluationen, wenn angegeben)."""
    return sum(1 for r in runs if r is not None and (budget is None or r <= budget))


def success_curve(runs, budgets):
    """Anteil erfolgreicher Läufe je Evaluationsbudget."""
    return [success_count(runs, b) / len(runs) for b in budgets]


def median_solved(runs):
    """Median der Evaluationen bis zum Erfolg über die erfolgreichen Läufe (None, wenn kein Lauf erfolgreich war)."""
    ok = [r for r in runs if r is not None]
    return float(np.median(ok)) if ok else None


def study_cell(pre, level):
    return pre["levels"][str(level)]


def sweep_cell(pre, level, pop, sigma):
    return next(x for x in pre["sweep"] if x["level"] == level and x["pop"] == pop and x["sigma"] == sigma)
