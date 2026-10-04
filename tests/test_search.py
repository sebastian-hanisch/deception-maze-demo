"""GA: Elite und Mutation von Hand, Grenzen, Reproduzierbarkeit, Zählung der Evaluationen bis zum Erfolg, Monotonie des besten Abstands."""

import numpy as np
import pytest

import dm_constants as C
import dm_maze as M
import dm_robot as R
import dm_search as S


def test_next_generation_keeps_the_elite_unchanged_and_keeps_the_size():
    rng = np.random.default_rng(1)
    genomes = R.random_genomes(rng, 10)
    fitness = np.array([0.0, 5.0, 1.0, 9.0, 2.0, 3.0, 4.0, 8.0, 7.0, 6.0])               # beste: Index 3 (9), 7 (8)
    nxt = S.next_generation(np.random.default_rng(2), genomes, fitness)
    assert nxt.shape == genomes.shape and (nxt[0] == genomes[3]).all() and (nxt[1] == genomes[7]).all()


def test_without_mutation_every_child_is_a_copy_of_a_parent():
    rng = np.random.default_rng(1)
    genomes = R.random_genomes(rng, 8)
    nxt = S.next_generation(np.random.default_rng(2), genomes, np.arange(8, dtype=float), rate=0.0)
    for child in nxt:
        assert any((child == parent).all() for parent in genomes)


def test_the_tournament_prefers_better_individuals():
    """Fitness = Index (0 bis 59). Ohne Mutation sind die Kinder Kopien; der mittlere Index der Eltern liegt weit über dem Mittel 29.5 einer zufälligen Wahl (Erwartung bei Turnier der Größe 3: etwa 44)."""
    genomes = R.random_genomes(np.random.default_rng(5), 60)
    nxt = S.next_generation(np.random.default_rng(6), genomes, np.arange(60, dtype=float), rate=0.0)
    parents = [next(i for i in range(60) if (child == genomes[i]).all()) for child in nxt[C.ELITE:]]
    assert len(parents) == 58 and np.mean(parents) > 36


def test_children_respect_the_gene_bounds():
    genomes = R.random_genomes(np.random.default_rng(7), 20)
    nxt = S.next_generation(np.random.default_rng(8), genomes, np.arange(20, dtype=float), sigma=5.0, rate=1.0)
    assert nxt[..., 0].min() >= -1 and nxt[..., 0].max() <= 1 and nxt[..., 1].min() >= 0 and nxt[..., 1].max() <= 1


def test_mutation_changes_roughly_the_given_share_of_genes():
    genomes = R.random_genomes(np.random.default_rng(9), 30)
    nxt = S.next_generation(np.random.default_rng(10), genomes, np.arange(30, dtype=float), sigma=0.1, rate=0.15)
    children = nxt[C.ELITE:]
    changed_share = np.mean([min(((child != parent).any(axis=1)).mean() for parent in genomes) for child in children])
    assert 0.08 < changed_share < 0.30


def test_run_is_reproducible_and_depends_on_the_seed():
    grid = M.make_maze(2)
    a, b, c = (S.run_ga(grid, s, gens=15, keep_positions=False) for s in (4, 4, 5))
    assert a.best_distance == b.best_distance and a.first_solved == b.first_solved and a.best_distance != c.best_distance


def test_best_distance_never_gets_worse_because_the_elite_survives():
    res = S.run_ga(M.make_maze(3), 3, gens=40, keep_positions=False)
    assert all(b <= a + 1e-12 for a, b in zip(res.best_distance, res.best_distance[1:]))
    assert res.evaluations == 40 * C.POP and len(res.best_distance) == len(res.mean_distance) == 40
    assert all(m >= b - 1e-12 for m, b in zip(res.mean_distance, res.best_distance))


def test_first_solved_counts_evaluations_exactly(monkeypatch):
    """Ziel auf den Start gelegt: schon die Anfangspopulation hat Individuen im Zielkreis. first_solved = Index des ersten solchen Individuums + 1, nachgerechnet aus derselben Anfangspopulation."""
    monkeypatch.setattr(C, "GOAL", C.START)
    grid = M.make_maze(0)
    res = S.run_ga(grid, 11, gens=3, keep_positions=False)
    genomes = R.random_genomes(np.random.default_rng(11), C.POP)
    dist = R.distance_to_goal(R.simulate(genomes, grid))
    expected = int(np.argmax(dist < C.SUCCESS_RADIUS)) + 1
    assert res.first_solved == expected and 1 <= expected <= C.POP


def test_first_solved_in_a_later_generation_is_offset_by_whole_generations():
    """Auf Stufe 2 gibt es unter den ersten Seeds einen Lauf, der erst nach der ersten Generation erfolgreich ist: dort stimmt der Index zur Generation, und davor lag kein Individuum im Zielkreis."""
    found = None
    for seed in range(12):
        res = S.run_ga(M.make_maze(2), seed, gens=40, keep_positions=False, stop_when_solved=True)
        if res.first_solved is not None and res.first_solved > C.POP:
            found = res
            break
    assert found is not None
    gen = (found.first_solved - 1) // C.POP
    assert found.best_distance[gen] < C.SUCCESS_RADIUS and all(d >= C.SUCCESS_RADIUS for d in found.best_distance[:gen])


def test_stop_when_solved_ends_the_run_early():
    res = S.run_ga(M.make_maze(0), 2, gens=60, keep_positions=False, stop_when_solved=True)
    assert res.first_solved is not None and res.evaluations < 60 * C.POP and len(res.best_distance) == (res.first_solved - 1) // C.POP + 1


def test_best_genome_reproduces_the_best_distance():
    grid = M.make_maze(2)
    res = S.run_ga(grid, 6, gens=25, keep_positions=False)
    again = R.distance_to_goal(R.simulate(res.best_genome[None], grid))[0]
    assert again == pytest.approx(min(res.best_distance))


def test_end_positions_are_kept_per_generation_on_request():
    res = S.run_ga(M.make_maze(0), 1, gens=5)
    assert len(res.end_positions) == 5 and res.end_positions[0].shape == (C.POP, 2)
    assert S.run_ga(M.make_maze(0), 1, gens=5, keep_positions=False).end_positions == []
