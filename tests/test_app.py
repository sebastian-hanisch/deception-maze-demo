"""AppTest-Rauchtests: Voreinstellung, jedes Preset, alle Stufen, Randwerte, lokale Regler, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dm_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, "Kürzester Weg") == "49 Zellen" and _metric(at, "Fitness-Distanz-Korrelation") == "0.69" and _metric(at, "Fallen (lokale Minima)") == "1"
    assert _metric(at, "Evaluationen insgesamt") == "10.000" and _metric(at, "Ergebnis") in ("Ziel erreicht", "Ziel verfehlt")
    assert len(at.get("plotly_chart")) == 7


def test_all_sections_are_present():
    at = _run()
    assert [s.value for s in at.subheader] == ["📐 Ein Lauf der zielgetriebenen Suche", "🔬 Wie oft gelingt es? Die Studie über 20 Läufe", "🔬 Hilft mehr Population oder stärkere Mutation?", "🚧 Wo die Annahmen enden"]
    assert any(m.value.startswith("## 🔗 Was die Fitness sieht") for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["level_select"], at.session_state["gens_slider"]) == (p["level"], p["gens"])
    assert at.metric


@pytest.mark.parametrize("level", list(C.LEVELS))
def test_every_level_runs(level):
    at = _run(level_select=level, gens=10)
    _ok(at)
    assert _metric(at, "Luftlinie Start–Ziel") == "17.0 Zellen"


def test_free_field_has_no_trap_and_the_run_succeeds():
    at = _run(level_select=0)
    _ok(at)
    assert _metric(at, "Fallen (lokale Minima)") == "0" and _metric(at, "Ergebnis") == "Ziel erreicht" and any("Hier gibt es keine Falle" in i.value for i in at.info)


def test_the_serpentine_has_two_traps_and_fails_in_the_run():
    at = _run(level_select=4, gens_slider=100)
    _ok(at)
    assert _metric(at, "Fallen (lokale Minima)") == "2" and _metric(at, "Kürzester Weg") == "77 Zellen" and _metric(at, "Ergebnis") == "Ziel verfehlt"


def test_the_solvability_line_names_the_steps_used():
    at = _run(level_select=4)
    assert any("136 von 140 Schritten benutzt" in c.value for c in at.caption)


@pytest.mark.parametrize("kw", [dict(gens_slider=10), dict(gens_slider=200), dict(level_select=3, gens_slider=200, seed_input=0), dict(level_select=1, gens_slider=10, seed_input=999999)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_the_study_table_lists_every_level_and_the_sweep_regulator_switches_the_level():
    at = _run()
    table = next(m.value for m in at.markdown if m.value.startswith("| Stufe |"))
    assert table.count("\n| ") == len(C.LEVELS) and "| 0 von 20 |" in table and "| 11 von 20 |" in table
    four = _run(sweep_level_select=4)
    _ok(four)
    assert any("keiner" in i.value and "neun Einstellungen" in i.value for i in four.info)


def test_dice_button_changes_the_seed_and_the_run(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run(level_select=2)
    old_seed = at.session_state["seed_input"]
    old = _metric(at, "Evaluationen bis zum Erfolg")
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed and _metric(at, "Evaluationen bis zum Erfolg") != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "9"
    at.query_params["gens"] = "104"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == 4 and at.session_state["gens_slider"] == 100


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["level"] = "viele"
    at.query_params["seed"] = "nan"
    at.run()
    _ok(at)
    assert at.session_state["level_select"] == C.DEFAULT_LEVEL and at.session_state["seed_input"] == C.DEFAULT_SEED


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    source = Path(APP).read_text(encoding="utf-8")
    assert '.replace(",", ".")' not in source
