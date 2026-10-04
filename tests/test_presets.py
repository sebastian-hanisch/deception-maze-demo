"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import dm_constants as C
import dm_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert preset["level"] in C.LEVELS and C.GENS_MIN <= preset["gens"] <= C.GENS_MAX and P.snap_gens(preset["gens"]) == preset["gens"]
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Freies Feld"]["level"] == 0 and C.PRESETS["Täuschung: Öffnung rechts"]["level"] == 3
    assert C.PRESETS["Schlangenlinie"]["level"] == 4 and C.PRESETS["Kurzer Lauf"]["gens"] == 30


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Täuschung: Öffnung rechts"]
    assert (p["level"], p["gens"], p["seed"]) == (C.DEFAULT_LEVEL, C.DEFAULT_GENS, C.DEFAULT_SEED)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("gens_slider") == (10, 200) and P.bounds("level_select") == (0, 4)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


@pytest.mark.parametrize("value,expected", [(0, 10), (14, 10), (16, 20), (104, 100), (106, 110), (999, 200)])
def test_generations_snap_to_the_step_from_the_lower_bound(value, expected):
    assert P.snap_gens(value) == expected


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 1), (3, 3), (4, 4), (2, 2)])
def test_levels_are_taken_over_unchanged(value, expected):
    assert P.snap_to_option("level_select", value) == expected


def test_formatters():
    assert C.fmt_int(40000) == "40.000" and C.fmt_pct(0.55) == "55 %" and C.fmt_pct(0.0898, 1) == "9.0 %"


def test_study_constants_are_consistent():
    assert C.STUDY_BUDGET == C.POP * C.STUDY_GENS == 40_000 and all(C.STUDY_BUDGET % p == 0 for p in C.SWEEP_POPS) and set(C.SWEEP_LEVELS) <= set(C.LEVELS)
    assert set(C.LEVEL_LABELS) == set(C.LEVELS) and C.POP in C.SWEEP_POPS and C.MUT_SIGMA in C.SWEEP_SIGMAS
