"""Konstanten der Täuschungs-Demo: Hof-Roboter im Labyrinth, Steuerfolge, Genetischer Algorithmus, Studien-Achsen, Regler, Voreinstellungen.
Längen in Zellen (Zellraster N × N, Zelle (i, j) belegt den Bereich [i, i + 1) × [j, j + 1))."""


def fmt_int(n):
    """Ganzzahl mit Punkt als Tausendertrenner (10000 -> 10.000)."""
    return f"{n:,}".replace(",", ".")


def fmt_pct(x, digits=0):
    """Anteil als Prozent mit Leerzeichen (0.086 -> "9 %", mit digits=1 "8.6 %")."""
    return f"{x:.{digits}%}".replace("%", " %")


N = 24                                       # Kantenlänge des Labyrinths in Zellen (Rand ist Wand)
START = (3.5, 3.5)                           # Mitte der Startzelle (3, 3)
GOAL = (3.5, 20.5)                           # Mitte der Zielzelle (3, 20)
SUCCESS_RADIUS = 2.0                         # Erfolg: Endposition höchstens so weit vom Ziel
START_HEADING = 0.7853981633974483           # Anfangsrichtung π/4
STEPS = 140                                  # Länge der Steuerfolge
TURN_MAX = 0.6                               # Drehung je Schritt (rad), Gen ∈ [−1, 1]
SPEED_MAX = 0.7                              # Fahrstrecke je Schritt (Zellen), Gen ∈ [0, 1]

POP = 100                                    # Individuen je Generation
ELITE = 2
TOURNAMENT = 3
MUT_SIGMA = 0.25                             # Standardabweichung der Mutation je verändertem Gen
MUT_RATE = 0.15                              # Anteil der Gene, die je Kind mutieren

LEVELS = (0, 1, 2, 3, 4)
LEVEL_LABELS = {0: "0: freies Feld", 1: "1: Öffnung links", 2: "2: Öffnung in der Mitte", 3: "3: Öffnung rechts", 4: "4: Schlangenlinie"}
GAP = 3                                      # Breite einer Öffnung in der Wand (Zellen)

STUDY_SEEDS = 20                             # Läufe je Zelle der Studie
STUDY_GENS = 400                             # Generationen je Lauf (bei POP = 100: 40 000 Evaluationen)
STUDY_BUDGET = POP * STUDY_GENS
SWEEP_LEVELS = (3, 4)                         # Stufen des Einstellungs-Rasters
SWEEP_POPS = (50, 100, 200)                  # Populationsgrößen bei gleichem Gesamtbudget
SWEEP_SIGMAS = (0.1, 0.25, 0.5)

GENS_MIN, GENS_MAX, GENS_STEP, DEFAULT_GENS = 10, 200, 10, 100      # Regler: Generationen des Live-Laufs
DEFAULT_LEVEL = 3
SEED_MAX = 999999
DEFAULT_SEED = 35

PRESET_ORDER = ("Freies Feld", "Täuschung: Öffnung rechts", "Schlangenlinie", "Kurzer Lauf")


def _preset(level=DEFAULT_LEVEL, gens=DEFAULT_GENS):
    return {"level": level, "gens": gens, "seed": DEFAULT_SEED}


PRESETS = {
    "Freies Feld": _preset(level=0),
    "Täuschung: Öffnung rechts": _preset(),
    "Schlangenlinie": _preset(level=4, gens=200),
    "Kurzer Lauf": _preset(gens=30),
}
# Zahlen aus der vorgerechneten Studie (20 Läufe je Stufe) und den exakten Täuschungsmaßen; tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Freies Feld": "Stufe 0: kürzester Weg 17 Zellen gleich Luftlinie, Fitness-Distanz-Korrelation 0.97, keine Falle: alle 20 Läufe erreichen das Ziel, im Median nach 64 Evaluationen.",
    "Täuschung: Öffnung rechts": "Stufe 3: 2.88-facher Umweg, Korrelation 0.69, eine Falle 9 Zellen vor dem Ziel: 11 von 20 Läufen erreichen das Ziel, alle bis 3 355 Evaluationen; die übrigen 9 auch nach 40 000 nicht.",
    "Schlangenlinie": "Stufe 4: kürzester Weg 77 Zellen (4.53-facher Umweg), zwei Fallen: 0 von 20 Läufen erreichen das Ziel, auch in keiner der 9 Einstellungen des Rasters (0 von 180 Läufen).",
    "Kurzer Lauf": "Stufe 3 mit nur 30 Generationen (3 000 Evaluationen): 10 von 20 Seeds haben das Ziel bis dahin erreicht; nach den vollen 40 000 Evaluationen sind es 11.",
}
