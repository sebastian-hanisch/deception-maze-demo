# Täuschende Landschaften – wenn die Fitness in die Sackgasse führt (Streamlit-Demo)

**[→ Demo live ausprobieren](https://sebastianhanisch-deception-maze-demo.streamlit.app/)**

---

Interaktive Demo zu **täuschenden Suchproblemen**. **Erstes Stück (Wurzel) der Konzepte-Linie „Novelty Search und Quality-Diversity“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Ein Hof-Roboter soll durch ein Labyrinth zum Ziel; ein **Genetischer Algorithmus** sucht seine Steuerfolge, und die Fitness ist der Abstand der Endposition zum Ziel. Zwischen Start und Ziel steht eine Wand: der kürzeste Weg führt zunächst
**vom Ziel weg**, die Fitness belohnt das Gegenteil, und die Suche bleibt in der Sackgasse unter dem Ziel stehen. Das Stück zeigt das Problem und misst ehrlich, **wie weit es sich mit einfachen Mitteln lösen lässt** (mehr Individuen,
stärkere Mutation, mehr Budget) und wo nicht. Alle späteren Stücke der Linie bauen auf diesem Labyrinth auf.

## Kernfrage

Wann führt eine Fitness, die „nah am Ziel“ misst, in die Irre, wie lässt sich das exakt messen, und was davon lösen einfache Einstellungen?

## Modell und Methodik

- **Labyrinth** (`dm_maze.py`): 24 × 24 Zellen, Start (3.5, 3.5) unten links, Ziel (3.5, 20.5) oben links, Rand ist Wand. Stufe 0 freies Feld; Stufen 1 bis 3 eine Querwand (Zeile 12) mit 3 Zellen breiter Öffnung links (Spalten 3 bis 5),
  in der Mitte (11 bis 13) oder rechts (19 bis 21); Stufe 4 drei Wände (Zeilen 7, 12, 17) mit Öffnungen abwechselnd rechts, links, rechts.
- **Roboter** (`dm_robot.py`): feste Steuerfolge aus **140 Befehlen** (Drehung ∈ [−1, 1] mal 0.6 rad, Tempo ∈ [0, 1] mal 0.7 Zellen je Schritt, Anfangsrichtung π/4), kein Sensor; wer in eine belegte Zelle fahren würde, bleibt stehen.
  **Verhalten** = Endposition. Die Simulation fährt ganze Populationen vektorisiert.
- **Suche** (`dm_search.py`): Genetischer Algorithmus, Fitness = −Abstand zum Ziel, 100 Individuen, Turnier der Größe 3, die zwei Besten bleiben, jedes Kind ist die Mutation eines Elters (Gauß σ = 0.25 auf 15 % der Gene), **keine Kreuzung**.
  **Erfolg:** ein Individuum endet höchstens 2 Zellen vom Ziel. Zufall nur über `numpy.random.default_rng(seed)`.
- **Täuschungsmaße, exakt und ohne Suche:** kürzester Weg (Breitensuche, 4er-Nachbarschaft) durch Luftlinie = **Umwegfaktor**; **Fitness-Distanz-Korrelation** (Jones und Forrest 1995) zwischen Luftlinie zum Ziel und wahrem Weg zum Ziel über
  alle freien Zellen; **Fallen** = lokale Minima der Luftlinie (freie, vom Start erreichbare Zellen, die kein freier Nachbar unterbietet).
- **Gegenproben:** (1) Breitensuche, kürzester Weg, Luftlinie und Korrelation auf Handlabyrinthen (5 × 5 mit Pfosten, gerader Gang mit Korrelation genau 1); (2) Kinematik von Hand (ein Schritt, Drehung, Kollision an der Wand bei y = 11.9);
  (3) die vektorisierte Population gleich Einzelfahrten; (4) GA-Einheiten (Elite bleibt, Kinder ohne Mutation sind Kopien, Turnier bevorzugt Bessere, Grenzen der Gene, Zählung der Evaluationen bis zum Erfolg von Hand);
  (5) **Lösbarkeit:** eine aus dem kürzesten Pfad abgeleitete Steuerfolge fährt in allen fünf Stufen ins Ziel, auch in der Schlangenlinie mit 136 von 140 Schritten; (6) ein Studienlauf wird mit demselben Seed frisch reproduziert.
- **Vorgerechnete Studie** (`generate_precomputed.py` → `precomputed_sweep.json`, rund drei Minuten parallel): je Stufe 20 Läufe à 400 Generationen (40 000 Evaluationen); dazu ein Einstellungs-Raster auf den Stufen 3 und 4 bei gleichem Gesamtbudget
  (Populationsgröße 50 / 100 / 200 mit 800 / 400 / 200 Generationen, Mutationsstärke σ = 0.1 / 0.25 / 0.5, je 20 Läufe).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Längen in Zellen, Aufwand in Evaluationen (100 je Generation).

| Frage | Befund |
|---|---|
| Was macht die Wand mit dem Weg? | Kürzester Weg **17 / 17 / 33 / 49 / 77** Zellen bei Luftlinie 17 (Stufe 0 bis 4): Umwegfaktor **1.00 / 1.00 / 1.94 / 2.88 / 4.53**. Bei Stufe 1 liegt die Öffnung genau auf dem Weg, die Wand allein täuscht nicht. |
| Wie täuschend ist das Labyrinth? | Fitness-Distanz-Korrelation **0.97 / 0.97 / 0.89 / 0.69 / 0.68**; Fallen (lokale Minima der Luftlinie): 0 / 0 / 1 / 1 / 2. Tiefste Falle 9 Zellen vor dem Ziel (Stufe 2 und 3), 4 Zellen (Stufe 4, dazu eine zweite bei 14). |
| Ist die Aufgabe lösbar? | Ja: eine aus dem kürzesten Pfad abgeleitete Steuerfolge erreicht in allen Stufen das Ziel, mit 27 / 27 / 60 / 82 / **136** von 140 Schritten. Sie liegt also im Suchraum. |
| Wie oft findet die zielgetriebene Suche das Ziel? | **20 / 20 / 20 / 11 / 0** von 20 Läufen auf Stufe 0 bis 4 (40 000 Evaluationen). Median der Evaluationen bis zum Erfolg 64 / 126 / 190 auf Stufe 0 bis 2 (schlimmster Lauf 237 / 252 / 875), auf Stufe 3 1 267. |
| Hilft mehr Budget? | **Nein, nicht auf Stufe 3.** Alle 11 Erfolge liegen bei höchstens **3 355** Evaluationen (10 davon bis 3 000); die übrigen **9 Läufe** finden den Weg auch mit 40 000 Evaluationen nicht. Der mittlere beste Abstand fällt von 7.89 (Generation 1) auf 5.31 (10), 4.12 (50), 3.95 (100) und 3.86 (400) und bleibt dann stehen (Stufe 4: 7.00). |
| Hilft eine andere Einstellung? | **Auf Stufe 3 zum guten Teil.** Erfolge von 20 für 50 / 100 / 200 Individuen bei σ = 0.1 / 0.25 / 0.5: **4 / 7 / 16, 10 / 11 / 19, 13 / 18 / 20**. Der Preis: der Median der Evaluationen bis zum Erfolg steigt von 1 267 (Standard) auf 4 075 (100 Individuen, σ = 0.5) und 3 425 (200, σ = 0.5). |
| Und auf Stufe 4? | **Keine einzige** der neun Einstellungen gelingt: 0 Erfolge in 180 Läufen. Einstellungen verschieben die Kante der Täuschung, sie beseitigen sie nicht. |
| Wie viel hängt an der Länge der Steuerfolge? | Mit **180 statt 140** Schritten löst derselbe GA Stufe 3 in 20 von 20 Läufen (Median etwa 700 Evaluationen): die Täuschung hängt an der Parametrisierung des Verhaltens, nicht nur am Labyrinth. |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Die erste Fassung des Labyrinths war gar nicht täuschend.** Die Wand lag senkrecht statt waagerecht (Indizes [x, y] vertauscht); Start und Ziel lagen auf derselben Seite, und alle Verfahren schafften 20 von 20. Der kürzeste Weg gegen die Luftlinie
  (Umwegfaktor) und die lokalen Minima sind seitdem fester Teil des Stücks: Täuschung wird gemessen, nicht behauptet.
- **Die Täuschung ist eine Kante, keine Rampe.** Gemessen sind Stufe 0 bis 2 für alle 20 Läufe leicht (Stufe 2 mit Umwegfaktor 1.94 noch vollständig gelöst), Stufe 3 halb gelöst, Stufe 4 nie: die Schwierigkeit wächst nicht gleichmäßig mit dem Umweg.
- **Stufe 3 ist teilweise ein Einstellungsproblem.** Die Vormessung (100 Individuen, σ = 0.25, nur 100 Generationen: 9 von 20) legte nahe, die Täuschung sei hart; das Raster zeigt, dass größere Population und stärkere Mutation sie bis auf 20 von 20 lösen.
  Erst Stufe 4 widersteht allen neun Einstellungen. Der Wurzel-Befund der Linie lautet deshalb nicht „der GA scheitert“, sondern „bei ausreichendem Umweg scheitert er unabhängig von der Einstellung“.
- **Die Länge der Steuerfolge verschiebt die Kante.** Bei 180 Schritten war Stufe 3 für den GA kein Problem mehr; die Stufen und die 140 Schritte wurden so gewählt, dass die Schlangenlinie gerade noch lösbar ist (136 Schritte), aber vom GA nicht gefunden wird.

## Ehrliche Grenzen

- **Offene Steuerfolge, kein Sensor.** Der Roboter fährt blind; ein sensorgestütztes Netz würde Wände umfahren und ein anderes Täuschungsproblem erzeugen.
- **Ein einfacher GA ohne Kreuzung.** Kreuzung und andere Operatoren (siehe [`genetic-algorithm-demo`](https://github.com/sebastian-hanisch/genetic-algorithm-demo)) ändern die Zahlen, aber nicht die Struktur der Täuschung; nicht gemessen.
- **20 Läufe je Zelle.** Die Erfolgsquote 11 von 20 hat bei dieser Stichprobe eine Standardabweichung von etwa 11 Prozentpunkten (Binomial); einzelne Rasterzellen unterscheiden sich innerhalb dieses Rauschens (etwa 10 gegen 11 von 20).
- **Ein einziger Labyrinthtyp** (Querwände mit Öffnung); andere Täuschungen (Spiralen, Schleifen) sind nicht gerechnet.

## Verwandte Demos im Portfolio

- [`genetic-algorithm-demo`](https://github.com/sebastian-hanisch/genetic-algorithm-demo): derselbe GA-Gedanke an einer Lieferroute, ohne Täuschung.
- [`cma-es-demo`](https://github.com/sebastian-hanisch/cma-es-demo): Suche in kontinuierlichen Räumen; die Steuerfolge hier ist ein kontinuierlicher Vektor mit 280 Genen.
- [`nsga2-demo`](https://github.com/sebastian-hanisch/nsga2-demo): zwei Ziele, aber ohne Antwort auf Täuschung.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Die Fitness ist ein Wegweiser | Novelty Search (Folgestück) |
| Eine Zielfunktion genügt | [NSGA-II](https://github.com/sebastian-hanisch/nsga2-demo) und Novelty plus Ziel (Folgestück) |
| Eine einzelne beste Lösung ist gesucht | MAP-Elites (Folgestück) |
| Offene Steuerfolge, kein Sensor | kein Folgestück |
| Ein GA ohne Kreuzung | [Genetischer Algorithmus](https://github.com/sebastian-hanisch/genetic-algorithm-demo) |

## Tests

136 Tests, rund 25 Sekunden: Labyrinth (Aufbau der Stufen, Breitensuche und kürzester Weg auf Handlabyrinthen, Luftlinie, Umwegfaktor, Korrelation im geraden Gang genau 1, lokale Minima), Roboter (Kinematik, Kollision, Population gleich Einzelfahrten, Lösbarkeit über
den Pfad-Regler in allen Stufen), GA (Elite, Mutation, Turnier, Grenzen, Reproduzierbarkeit, Zählung der Evaluationen, Monotonie des besten Abstands), Auswertung und Vollständigkeit der vorgerechneten Datei (inklusive Neurechnung eines Studienlaufs), Presets
und Permalink, Diagramme (gesperrte Achsen), AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `dm_maze.py` | Labyrinth, Breitensuche, Täuschungsmaße |
| `dm_robot.py` | Kinematik mit Kollision, Population, Pfad-Regler (Lösbarkeit) |
| `dm_search.py` | Genetischer Algorithmus mit Fitness = −Abstand zum Ziel |
| `dm_evaluation.py` | Maße je Stufe, Lösbarkeit, Live-Lauf, Zugriff auf die vorgerechnete Studie |
| `generate_precomputed.py` | rechnet die Studie vor → `precomputed_sweep.json` |
| `dm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `dm_presets.py`, `dm_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Lehman, J., Stanley, K. O. (2011): Abandoning objectives: evolution through the search for novelty alone. *Evolutionary Computation* 19(2), 189–223 (die täuschende Labyrinth-Aufgabe, an der objektive Suche scheitert und Novelty Search nicht).
- Lehman, J., Stanley, K. O. (2008): Exploiting open-endedness to solve problems through the search for novelty. *Proceedings of the Eleventh International Conference on Artificial Life (ALIFE XI)*, MIT Press.
- Jones, T., Forrest, S. (1995): Fitness distance correlation as a measure of problem difficulty for genetic algorithms. *Proceedings of the 6th International Conference on Genetic Algorithms*.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studie neu rechnen: `python generate_precomputed.py`.

Gebaut mit Streamlit und Plotly (die Simulation rechnet NumPy).
