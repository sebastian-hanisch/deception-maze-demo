"""Täuschende Landschaften – wenn die Fitness in die Sackgasse führt - Stück 1 der Konzepte-Linie "Novelty Search und Quality-Diversity"
Sebastian Hanisch - Operations Research und Machine Learning

Ein Hof-Roboter soll durch ein Labyrinth zum Ziel; seine Steuerfolge sucht ein Genetischer Algorithmus, dessen Fitness der Abstand zum Ziel ist. Zwischen Start und Ziel steht eine Wand: der kürzeste Weg führt
zunächst vom Ziel weg, und die Fitness belohnt genau das Gegenteil. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import dm_constants as C
import dm_evaluation as E
import dm_maze as M
from dm_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from dm_visualization import (build_budget_chart, build_fitness_view, build_progress_chart, build_run_figure, build_success_chart, build_sweep_chart, build_truth_view)

st.set_page_config(page_title="Täuschende Landschaften – Sebastian Hanisch", layout="wide")

PRE = E.load_precomputed()


@st.cache_data(show_spinner=False)
def _run(level, gens, seed):
    res, track = E.live_run(level, gens, seed)
    return res, track


@st.cache_data(show_spinner=False)
def _solvability(level):
    return E.solvability(level)


def _evals(x):
    return "–" if x is None else C.fmt_int(int(round(x)))


st.title("🧭 Täuschende Landschaften – wenn die Fitness in die Sackgasse führt")
st.markdown(
    """
Ein Roboter auf dem Hof soll zu einem Ziel fahren, und eine Suche probiert Steuerfolgen aus: je näher eine Fahrt dem Ziel endet, desto besser (die **Fitness**). Das klingt nach einem guten Wegweiser – bis eine **Wand** zwischen
Roboter und Ziel steht. Dann führt der kürzeste Weg erst einmal **vom Ziel weg**, und die Fitness belohnt jede Fahrt, die an der Wand unter dem Ziel endet. Die Suche läuft in die **Sackgasse**. Das ist eine *täuschende Landschaft*,
und sie ist die Wurzel einer ganzen Linie: Novelty Search und Quality-Diversity fragen nicht mehr „wie nah am Ziel?“, sondern „wie anders als alles Bisherige?“. Dieses erste Stück zeigt nur das Problem, und es misst ehrlich, wie
weit sich das Problem schon mit einfachen Mitteln (mehr Individuen, stärkere Mutation) lösen lässt.
"""
)
st.caption(
    "Stück 1 der Linie „Novelty Search und Quality-Diversity“. Die Suche ist ein einfacher Genetischer Algorithmus wie im [genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/); "
    "jedes Folgestück hebt genau eine Annahme dieses Stücks auf."
)

with st.expander("So ist das Problem gebaut", expanded=True):
    st.markdown(
        """
- **Labyrinth:** 24 × 24 Zellen, Start unten links, Ziel oben links. Stufe 0: freies Feld. Stufen 1 bis 3: eine Querwand mit einer Öffnung (links, in der Mitte, rechts): je weiter rechts, desto länger der Umweg. Stufe 4:
  drei Wände mit Öffnungen abwechselnd rechts und links, eine Schlangenlinie.
- **Roboter:** fährt eine **feste Folge von 140 Befehlen** (Drehung und Tempo je Schritt, kein Sensor); stößt er an eine Wand, bleibt er stehen. Sein **Verhalten** ist die Endposition.
- **Suche:** Genetischer Algorithmus mit Fitness = −Abstand der Endposition zum Ziel; Turnierauswahl, die zwei Besten bleiben, Mutation ohne Kreuzung. **Erfolg:** die Endposition liegt höchstens 2 Zellen vom Ziel.
- **Täuschungsmaße (exakt, ohne Suche):** Umwegfaktor (kürzester Weg durch Luftlinie), **Fitness-Distanz-Korrelation** (Korrelation zwischen Luftlinie und wahrem Weg über alle Zellen) und die **Fallen** (lokale Minima der Luftlinie).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    level = st.select_slider("Labyrinth (Stufe)", options=C.LEVELS, key="level_select", format_func=lambda v: C.LEVEL_LABELS[v], help="0 freies Feld bis 4 Schlangenlinie; je höher, desto länger der Umweg.")
    gens = st.slider("Generationen im Live-Lauf", *bounds("gens_slider"), step=C.GENS_STEP, key="gens_slider", help=f"Je Generation {C.POP} Individuen, also {C.POP} Evaluationen.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1, key="seed_input", help="Bestimmt die Anfangspopulation und alle Zufallsentscheidungen des GA.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

level, gens, seed = int(level), int(gens), int(seed)
sync_query_params({"level_select": level, "gens_slider": gens, "seed_input": seed})

grid = M.make_maze(level)
lm = E.level_metrics(level)

st.markdown("---")
st.markdown("## 🔗 Was die Fitness sieht, und was wirklich gilt")
st.caption(f"{C.LEVEL_LABELS[level]}. Links die Luftlinie jeder Zelle zum Ziel (dunkel = nah), rechts der wahre Weg zum Ziel in Zellen. Violette Kreuze: Fallen, in denen die Fitness nichts Besseres in der Nähe sieht.")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Was die Fitness sieht** (Luftlinie)")
    st.plotly_chart(build_fitness_view(grid), width="stretch", key=f"fitness_view_{level}")
with col_b:
    st.markdown("**Was wirklich gilt** (kürzester Weg)")
    st.plotly_chart(build_truth_view(grid), width="stretch", key=f"truth_view_{level}")
m1 = st.columns(4)
m1[0].metric("Luftlinie Start–Ziel", f"{lm['straight']:.1f} Zellen")
m1[1].metric("Kürzester Weg", f"{lm['path']} Zellen", delta=f"{lm['detour']:.2f}-facher Umweg", delta_color="off")
m1[2].metric("Fitness-Distanz-Korrelation", f"{lm['fdc']:.2f}", help="1 = die Luftlinie führt zum Ziel; je kleiner, desto täuschender.")
m1[3].metric("Fallen (lokale Minima)", f"{lm['minima']}", help="Zellen, in denen die Luftlinie zum Ziel kleiner ist als in allen freien Nachbarzellen.")
if lm["minima"] == 0:
    st.info("Hier gibt es keine Falle: die Fitness führt von jeder Zelle bergab zum Ziel (Korrelation " + f"{lm['fdc']:.2f}" + "). Eine Wand allein macht noch keine Täuschung – bei Stufe 1 liegt die Öffnung genau auf dem Weg.")
else:
    st.info(
        f"Die tiefste Falle liegt {lm['trap']:.1f} Zellen vom Ziel entfernt: dort sieht die Fitness nichts Besseres, obwohl der wahre Weg {lm['path']} Zellen lang ist (Umweg {lm['detour']:.2f}-fach, Korrelation {lm['fdc']:.2f}). "
        "Eine Suche, die nur die Luftlinie kennt, wird an dieser Stelle belohnt und bleibt dort."
    )
dist_ok, steps_used = _solvability(level)
st.caption(f"Lösbar? Ja: eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt ins Ziel (Abstand am Ende {dist_ok:.2f} Zellen, {steps_used} von {C.STEPS} Schritten benutzt). Die Aufgabe liegt also im Suchraum.")

st.markdown("---")
st.subheader("📐 Ein Lauf der zielgetriebenen Suche")
with st.spinner(f"Lasse den GA {gens} Generationen suchen …"):
    res, track = _run(level, gens, seed)
final_dist = res.best_distance[-1]
solved = res.first_solved is not None
r1 = st.columns(4)
r1[0].metric("Ergebnis", "Ziel erreicht" if solved else "Ziel verfehlt")
r1[1].metric("Bester Abstand am Ende", f"{final_dist:.2f} Zellen")
r1[2].metric("Evaluationen bis zum Erfolg", _evals(res.first_solved))
r1[3].metric("Evaluationen insgesamt", C.fmt_int(res.evaluations))
col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(build_run_figure(grid, res.end_positions[-1], track), width="stretch", key=f"run_{level}_{gens}_{seed}")
with col_d:
    st.plotly_chart(build_progress_chart(res.best_distance, lm["trap"]), width="stretch", key=f"progress_{level}_{gens}_{seed}")
if solved:
    st.info(f"Die Suche erreicht das Ziel nach {C.fmt_int(res.first_solved)} Evaluationen (Generation {res.first_solved // C.POP + 1}). Das ist ein Lauf; ob es typisch ist, zeigt die Studie unten.")
elif lm["trap"] is not None and abs(final_dist - lm["trap"]) < 0.6:
    st.info(f"Der beste Abstand ({final_dist:.2f} Zellen) entspricht der Falle ({lm['trap']:.1f}): die Population hat sich vor der Wand versammelt, die blauen Punkte liegen unter dem Ziel. Mehr Generationen helfen nur, wenn sich irgendwann ein Individuum durch die Öffnung verirrt.")
else:
    st.info(f"Nach {gens} Generationen steht der beste Abstand bei {final_dist:.2f} Zellen. Mehr Generationen oder ein anderer Seed können das Ergebnis ändern; die Studie unten fasst 20 Läufe zusammen.")

st.markdown("---")
st.subheader("🔬 Wie oft gelingt es? Die Studie über 20 Läufe")
n = PRE["seeds"]
st.markdown(
    f"Je Stufe {n} Läufe mit unterschiedlichem Seed, jeweils bis zu {C.fmt_int(PRE['budget'])} Evaluationen ({C.STUDY_GENS} Generationen). **Erfolg** heißt: irgendein Individuum endet höchstens {C.SUCCESS_RADIUS:g} Zellen vom Ziel."
)
st.plotly_chart(build_success_chart(PRE), width="stretch", key="success_chart")
rows = []
for lv in C.LEVELS:
    c = E.study_cell(PRE, lv)
    runs = c["runs"]
    trap = "–" if c["trap"] is None else f"{c['trap']:.1f}"
    rows.append(f"| {C.LEVEL_LABELS[lv]} | {c['path']} | {c['detour']:.2f} | {c['fdc']:.2f} | {trap} | {E.success_count(runs)} von {n} | {_evals(E.median_solved(runs))} | {c['mean_best_distance'][-1]:.2f} |")
st.markdown("| Stufe | Kürzester Weg | Umwegfaktor | Korrelation | Tiefe der Falle (Abstand) | Erfolg | Median der Evaluationen bis zum Erfolg | mittlerer bester Abstand am Ende |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
st.plotly_chart(build_budget_chart(PRE), width="stretch", key="budget_chart")
c3, c4 = E.study_cell(PRE, 3), E.study_cell(PRE, 4)
st.info(
    f"Auf den Stufen 0 bis 2 gelingt es in allen {n} Läufen, mit wachsendem Aufwand (Median {_evals(E.median_solved(E.study_cell(PRE, 0)['runs']))} bis {_evals(E.median_solved(E.study_cell(PRE, 2)['runs']))} Evaluationen). "
    f"Auf Stufe 3 gelingt es in {E.success_count(c3['runs'])} von {n} Läufen, auf Stufe 4 in {E.success_count(c4['runs'])}. Die Täuschung wirkt nicht als Mauer, sondern als Kante: bis zu einem gewissen Umweg ist die Fitness noch ein brauchbarer "
    "Wegweiser, danach nicht mehr."
)

st.markdown("---")
st.subheader("🔬 Hilft mehr Population oder stärkere Mutation?")
sweep_level = st.select_slider("Stufe für das Raster", options=list(C.SWEEP_LEVELS), value=C.SWEEP_LEVELS[0], key="sweep_level_select", format_func=lambda v: C.LEVEL_LABELS[v])
st.markdown(
    f"Gleiches Gesamtbudget ({C.fmt_int(C.STUDY_BUDGET)} Evaluationen), aber andere Einstellungen des GA: Populationsgröße 50 / 100 / 200 (entsprechend 800 / 400 / 200 Generationen) und Mutationsstärke σ = 0,1 / 0,25 / 0,5. Je {n} Läufe."
)
st.plotly_chart(build_sweep_chart(PRE, sweep_level), width="stretch", key=f"sweep_{sweep_level}")
sw = lambda lv, p, s: E.success_count(E.sweep_cell(PRE, lv, p, s)["runs"])
total4 = sum(E.success_count(x["runs"]) for x in PRE["sweep"] if x["level"] == 4)
st.info(
    f"Auf Stufe 3 lässt sich die Täuschung zu einem guten Teil wegtunen: mit der Standardeinstellung (100 Individuen, σ = 0,25) {sw(3, 100, 0.25)} von {n}, mit 200 Individuen und σ = 0,5 {sw(3, 200, 0.5)} von {n} "
    f"(allerdings mit mehr Evaluationen bis zum Erfolg, Median {_evals(E.median_solved(E.sweep_cell(PRE, 3, 200, 0.5)['runs']))}). Auf Stufe 4 gelingt es in **keiner** der neun Einstellungen "
    f"({total4} Erfolge in {len(C.SWEEP_POPS) * len(C.SWEEP_SIGMAS) * n} Läufen). Einstellungen verschieben die Kante, sie beseitigen die Täuschung nicht."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Fitness ist ein Wegweiser** (kleiner Abstand = näher am Ziel) | Bei Umwegen führt sie in Sackgassen, die Suche konvergiert dort und liefert Ziele, die nur nah aussehen. | **Novelty Search** (Folgestück): belohnt, anders zu sein als alles Bisherige, statt nah am Ziel |
| **Eine Zielfunktion genügt** | Mit Abstand zum Ziel und einem zweiten Ziel (etwa Fahrzeit) entsteht eine Pareto-Menge, aber die Täuschung bleibt in beiden Zielen. | **[NSGA-II](https://sebastianhanisch-nsga2-demo.streamlit.app/)** (zwei Ziele) und **Novelty plus Ziel** (Folgestück) |
| **Eine einzelne beste Lösung ist gesucht** | Die Suche liefert einen Weg, nicht die Menge der verschiedenen guten Wege (links herum, rechts herum). | **MAP-Elites** (Folgestück): ein Archiv verschiedener Verhaltensweisen |
| **Offene Steuerfolge, kein Sensor** | Mit Sensoren (Abstand zur Wand) könnte ein Netz Wände umfahren; hier kennt der Roboter das Labyrinth nicht und fährt blind. | kein Folgestück |
| **Ein GA ohne Kreuzung** | Ein Genetischer Algorithmus mit Kreuzung und anderen Operatoren würde andere Zahlen liefern, die Täuschung bleibt strukturell. | **[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/)** (Kreuzung, Operatoren) |
"""
)
st.caption(
    "Verwandt im Portfolio: [genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) (der GA ohne Täuschung), [cma-es-demo](https://sebastianhanisch-cma-es-demo.streamlit.app/) "
    "(Suche in kontinuierlichen Räumen), [nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/) (zwei Ziele)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kinematik.** Zustand $(x_t, y_t, \theta_t)$; $\theta_{t+1} = \theta_t + a_t\,\Delta\theta_{\max}$, Fahrt $v_t\,s_{\max}$ in Richtung $\theta_{t+1}$; liegt der neue Punkt in einer belegten Zelle, bleibt die Position unverändert.
Genom: $(a_t, v_t)_{t=1}^{140}$ mit $a_t \in [-1, 1]$, $v_t \in [0, 1]$. Verhalten $b(g)$ = Endposition.

**Fitness.** $f(g) = -\lVert b(g) - z \rVert_2$ mit dem Ziel $z$. Erfolg: $\lVert b(g) - z \rVert_2 < 2$.

**Täuschungsmaße.** Luftlinie $e(c) = \lVert m_c - z \rVert_2$ und wahrer Weg $d(c)$ (Breitensuche, 4er-Nachbarschaft) für jede freie Zelle $c$. Fitness-Distanz-Korrelation (Jones und Forrest 1995):
$\rho = \mathrm{corr}\bigl(e(c), d(c)\bigr)$. Lokales Minimum: freie, vom Start erreichbare Zelle $c \ne z$ mit $e(c) \le e(c')$ für alle freien Nachbarn $c'$ (8er-Nachbarschaft).

**GA.** Population 100, Turnier der Größe 3, Elite 2, Kinder = Mutation eines Elters (Gauß $\sigma$ auf einem Anteil 0,15 der Gene), keine Kreuzung.

Implementiert in `dm_maze.py` (Labyrinth, Maße), `dm_robot.py` (Kinematik, Lösbarkeit), `dm_search.py` (GA).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
