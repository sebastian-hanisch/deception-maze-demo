"""Rechnet die Studie vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt `precomputed_sweep.json`.

  levels  je Stufe 0 bis 4: 20 Läufe des GA (Fitness = −Abstand zum Ziel, 400 Generationen × 100 Individuen = 40 000 Evaluationen): Evaluationen bis zum ersten Erfolg (None = nie), mittlerer Verlauf des besten Abstands,
          dazu die exakten Täuschungsmaße der Stufe
  sweep   Stufen 3 und 4, gleiches Gesamtbudget (40 000 Evaluationen): Populationsgröße 50 / 100 / 200 × Mutationsstärke 0.1 / 0.25 / 0.5, je 20 Läufe"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import dm_constants as C
import dm_evaluation as E
import dm_maze as M
import dm_search as S


def _level_task(args):
    level, seed = args
    res = S.run_ga(M.make_maze(level), seed, gens=C.STUDY_GENS, keep_positions=False)
    return level, seed, res.first_solved, res.best_distance


def _sweep_task(args):
    level, pop, sigma, seed = args
    gens = C.STUDY_BUDGET // pop
    res = S.run_ga(M.make_maze(level), seed, gens=gens, pop=pop, sigma=sigma, keep_positions=False)
    return level, pop, sigma, seed, res.first_solved


def main(workers):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        level_res = list(ex.map(_level_task, [(lv, s) for lv in C.LEVELS for s in range(C.STUDY_SEEDS)], chunksize=1))
        sweep_res = list(ex.map(_sweep_task, [(lv, p, sg, s) for lv in C.SWEEP_LEVELS for p in C.SWEEP_POPS for sg in C.SWEEP_SIGMAS for s in range(C.STUDY_SEEDS)], chunksize=1))
    levels = {}
    for lv in C.LEVELS:
        rows = [r for r in level_res if r[0] == lv]
        curve = np.mean([r[3] for r in rows], axis=0)
        levels[str(lv)] = {**E.level_metrics(lv), "runs": [r[2] for r in sorted(rows, key=lambda x: x[1])], "mean_best_distance": [float(x) for x in curve]}
    sweep = []
    for lv in C.SWEEP_LEVELS:
        for p in C.SWEEP_POPS:
            for sg in C.SWEEP_SIGMAS:
                rows = sorted([r for r in sweep_res if r[0] == lv and r[1] == p and r[2] == sg], key=lambda x: x[3])
                sweep.append({"level": lv, "pop": p, "sigma": sg, "runs": [r[4] for r in rows]})
    E.PRECOMPUTED_PATH.write_text(json.dumps({"seeds": C.STUDY_SEEDS, "budget": C.STUDY_BUDGET, "levels": levels, "sweep": sweep}), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {E.PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(14, os.cpu_count() or 1))
