"""Faithfulness annotation of ChatScene's 40 released generated scenarios (dynamic_log.csv).

For each scenario and each retrieved component (behavior B, geometry G, spawn S) the verdict
compares the retrieved database entry with the *original* natural-language description.
Annotated by hand from the trace in results/trace40.txt (AI-assisted, every case re-read).

Error codes
  SIDE   left/right/oncoming of the adversary differs from the description
  TURN   left/right turn differs
  LONG   front/behind differs
  ADDED  the retrieved code adds an element the description never mentions
         (a parked car, a slow leading car, an intersection on a straight road)
  CTX    wrong road context (e.g. opposite-direction two-lane road for a same-direction lane change)
  BEHAV  the behavior differs from the described one (when a matching entry exists in the database)
  PART   behavior partly matches (adds or drops a phase, e.g. "brakes" -> "brakes then accelerates")
  ACTOR  a vehicle-only spawn pattern (lane following) is used for a pedestrian, or vice versa
  MIX    retrieved components contradict each other (e.g. geometry has only a right lane, spawn needs a left one)

Stage: "R" = the LLM's sub-description was correct and retrieval lost/changed it,
       "E" = the LLM's extraction already lost/changed it.
"""

# scenario -> {component: [(code, stage, note), ...]}; components without errors are omitted.
ANNOTATIONS = {
    0: {"G": [("ADDED", "R", "straight road -> four-way intersection (no plain straight-road geometry in DB)")]},
    1: {"G": [("ADDED", "R", "straight road -> four-way intersection")],
        "S": [("SIDE", "R", "bus stop on the right front -> left front")]},
    2: {"G": [("ADDED", "R", "straight road -> four-way intersection")],
        "S": [("SIDE", "R", "vending machine on the right front -> left front")]},
    3: {"B": [("PART", "R", "stop then walk diagonally -> stop then resume walking (diagonal entry B2 exists)")],
        "G": [("ADDED", "R", "straight road -> four-way intersection")],
        "S": [("ACTOR", "R", "pedestrian from a driveway -> vehicle in adjacent lane"),
              ("ADDED", "R", "slow leading car added")]},
    4: {"G": [("ADDED", "R", "straight road -> four-way intersection")],
        "S": [("SIDE", "R", "parked car on the right front -> left front")]},
    5: {"G": [("TURN", "R", "ego turning left -> right turn")],
        "S": [("ACTOR", "R", "motorcyclist at road edge -> vehicle in adjacent lane"),
              ("ADDED", "R", "slow leading car added")]},
    6: {"G": [("TURN", "R", "ego turning left -> right turn")],
        "S": [("ADDED", "R", "parked car blocking ego lane added")]},
    7: {"S": [("ADDED", "R", "parked car blocking ego lane added")]},
    8: {"S": [("SIDE", "R", "left front -> oncoming side"),
              ("ACTOR", "R", "pedestrian -> vehicle turning left at intersection")]},
    9: {"G": [("TURN", "R", "ego turning left -> right turn")],
        "S": [("SIDE", "R", "left front -> oncoming side")]},
    10: {"G": [("CTX", "R", "same-direction lane change -> opposite-direction two-lane road")],
         "S": [("SIDE", "R", "car in target lane -> oncoming lane"),
               ("ADDED", "R", "parked car blocking ego lane added")]},
    11: {"S": [("ADDED", "R", "slow leading car added")]},
    12: {"G": [("CTX", "R", "same-direction lane change -> opposite-direction two-lane road")],
         "S": [("SIDE", "R", "car in target lane -> oncoming lane"),
               ("ADDED", "R", "parked car blocking ego lane added")]},
    13: {},
    14: {"S": [("MIX", "R", "geometry G6 guarantees a right lane, spawn S21 uses the left lane")]},
    15: {},
    16: {"B": [("PART", "R", "swerves into the lane -> drives in the oncoming lane (swerve dropped)")]},
    17: {"B": [("PART", "R", "enters and stops -> stops then resumes walking")]},
    18: {"B": [("BEHAV", "R", "oncoming car turns into ego path -> crosses intersection and accelerates")]},
    19: {},
    20: {"B": [("PART", "R", "accelerates -> accelerates then decelerates")],
         "S": [("ACTOR", "R", "crossing vehicle at intersection -> road-crossing pattern next to a parked car"),
               ("ADDED", "R", "parked car blocking ego lane added")]},
    21: {"S": [("CTX", "R", "red-light runner at intersection -> adjacent-lane spawn"),
               ("ADDED", "R", "slow leading car added")]},
    22: {"S": [("ADDED", "R", "slow leading car added")]},
    23: {"S": [("CTX", "R", "red-light runner at intersection -> adjacent-lane spawn"),
               ("ADDED", "R", "slow leading car added")]},
    24: {"B": [("PART", "R", "brakes abruptly -> brakes then accelerates aggressively")],
         "S": [("SIDE", "R", "from the right -> oncoming side")]},
    25: {"B": [("PART", "R", "unexpected acceleration -> alternating acceleration/deceleration")],
         "S": [("SIDE", "R", "oncoming car -> left-front adjacent lane"),
               ("ADDED", "R", "slow leading car added")]},
    26: {"B": [("PART", "R", "brakes -> brakes then accelerates aggressively")],
         "S": [("ADDED", "R", "parked car blocking ego lane added")]},
    27: {"S": [("SIDE", "R", "from the right -> oncoming side")]},
    28: {"S": [("ADDED", "R", "parked car blocking ego lane added")]},
    29: {"S": [("ADDED", "R", "parked car blocking ego lane added")]},
    30: {"S": [("SIDE", "R", "crossing from the left -> oncoming side")]},
    31: {"S": [("ADDED", "R", "slow leading car added")]},
    32: {"S": [("SIDE", "R", "enters from the left -> oncoming side")]},
    33: {"B": [("BEHAV", "R", "reverses abruptly -> stops after a lane change (reversal entry B53 exists)")],
         "S": [("ADDED", "R", "slow leading car added")]},
    34: {"S": [("LONG", "R", "behind on the right -> right front"),
               ("ADDED", "R", "slow leading car added")]},
    35: {"S": [("ADDED", "R", "slow leading car added")]},
    36: {"S": [("ADDED", "R", "slow leading car added")]},
    37: {"G": [("CTX", "R", "multi-lane road -> opposite-direction two-lane road")],
         "S": [("ADDED", "R", "slow leading car added")]},
    38: {"B": [("BEHAV", "R", "turns left and stops -> alternating acceleration (left-turn-and-stop entry B48 exists)")],
         "S": [("ADDED", "R", "parked car blocking ego lane added")]},
    39: {"B": [("BEHAV", "R", "turns left and stops -> alternating acceleration (B48 exists)")],
         "S": [("SIDE", "R", "from the right -> oncoming side"),
               ("TURN", "R", "turns left -> turns right")]},
}

# Code-level defects of retrieved entries (database audit), independent of the description match.
CODE_DEFECTS = {
    ("spawn", 1): "described left front; `left of P by Range(-8,-2)` places it on the right, heading -90 (faces right)",
    ("spawn", 2): "described right front; heading -90 makes the crossing agent face away from the road",
    ("spawn", 7): "described left front; blocker placed on the right, agent faces right",
    ("spawn", 9): "described left front; blocker placed on the right, agent faces right",
    ("spawn", 11): "described left front; blocker placed on the right, agent faces right",
    ("spawn", 19): "described rear right; uses _laneToLeft",
    ("geometry", 0): "described right turn; filters ManeuverType.LEFT_TURN (same description as entry 1)",
}
