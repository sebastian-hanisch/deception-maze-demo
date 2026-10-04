"""Zielgetriebene Suche: ein Genetischer Algorithmus, dessen Fitness der (negative) Abstand der Endposition zum Ziel ist.

Bewusst einfach, damit die Täuschung sichtbar bleibt und nicht an Einstellungen hängt: Turnierauswahl (3), die besten 2 Individuen bleiben (Elite), jedes Kind ist die Mutation eines Elters (Gauß mit Standardabweichung
`sigma` auf einem Anteil `rate` der Gene), keine Kreuzung. Zufall nur über `numpy.random.default_rng(seed)` (PCG64, plattformstabil)."""

from dataclasses import dataclass, field

import numpy as np

import dm_constants as C
import dm_robot as R


@dataclass
class GAResult:
    best_distance: list                  # kleinster Abstand zum Ziel je Generation
    mean_distance: list                  # mittlerer Abstand je Generation
    first_solved: object                 # Zahl der Evaluationen bis zum ersten Individuum im Zielkreis (None: nie)
    end_positions: list = field(default_factory=list)    # Endpositionen (pop, 2) je Generation
    best_genome: object = None           # Steuerfolge mit dem kleinsten Abstand über den ganzen Lauf
    evaluations: int = 0


def next_generation(rng, genomes, fitness, sigma=C.MUT_SIGMA, rate=C.MUT_RATE):
    """Neue Population: die ELITE besten bleiben unverändert, der Rest sind Mutationen von Turnier-Siegern (Turniergröße TOURNAMENT)."""
    pop = len(genomes)
    order = np.argsort(-fitness, kind="stable")
    nxt = [genomes[i].copy() for i in order[:C.ELITE]]
    while len(nxt) < pop:
        cand = rng.integers(0, pop, C.TOURNAMENT)
        parent = cand[np.argmax(fitness[cand])]
        child = genomes[parent] + rng.normal(0.0, sigma, genomes[parent].shape) * (rng.random(genomes[parent].shape) < rate)
        nxt.append(R.clip_genomes(child))
    return np.array(nxt)


def run_ga(grid, seed, gens=C.STUDY_GENS, pop=C.POP, sigma=C.MUT_SIGMA, rate=C.MUT_RATE, keep_positions=True, stop_when_solved=False):
    """Ein Lauf des GA mit Fitness = −Abstand zum Ziel. Zählt Evaluationen (pop je Generation) bis das erste Individuum im Zielkreis landet."""
    rng = np.random.default_rng(seed)
    genomes = R.random_genomes(rng, pop)
    res = GAResult(best_distance=[], mean_distance=[], first_solved=None)
    best_d = np.inf
    for gen in range(gens):
        end = R.simulate(genomes, grid)
        dist = R.distance_to_goal(end)
        res.evaluations += pop
        if res.first_solved is None and dist.min() < C.SUCCESS_RADIUS:
            res.first_solved = gen * pop + int(np.argmax(dist < C.SUCCESS_RADIUS)) + 1
        res.best_distance.append(float(dist.min()))
        res.mean_distance.append(float(dist.mean()))
        if keep_positions:
            res.end_positions.append(end.copy())
        if dist.min() < best_d:
            best_d = float(dist.min())
            res.best_genome = genomes[int(np.argmin(dist))].copy()
        if stop_when_solved and res.first_solved is not None:
            break
        genomes = next_generation(rng, genomes, -dist, sigma, rate)
    return res
