"""Ground-truth attributes of ChatScene's retrieval database and template-generated queries.

Every database entry gets the attributes its *description* promises (annotated by hand from
the text). Queries are generated from attribute specs with several phrasings, so the expected
answer of every query is known by construction:

* a query is answered correctly when the top-1 retrieved entry has the query's attributes;
* two phrasings of the same spec should retrieve entries with the same attributes
  (paraphrase invariance);
* flipping one attribute (left -> right, ...) should retrieve an entry with the flipped
  attribute (semantic sensitivity).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------
# Database annotations (index -> attributes), taken from the descriptions in database_v1.pkl
# --------------------------------------------------------------------------------------------
SPAWN_ATTRS = {
    0: dict(group="front_opp"),
    1: dict(group="cross_straight", side="left", blocker="none"),
    2: dict(group="cross_straight", side="right", blocker="none"),
    3: dict(group="rear"),
    4: dict(group="diag_rear", side="left"),
    5: dict(group="diag_rear", side="right"),
    6: dict(group="cross_straight", side="right", blocker="bus stop"),
    7: dict(group="cross_straight", side="left", blocker="bus stop"),
    8: dict(group="cross_straight", side="right", blocker="vending machine"),
    9: dict(group="cross_straight", side="left", blocker="vending machine"),
    10: dict(group="cross_straight", side="right", blocker="parked car"),
    11: dict(group="cross_straight", side="left", blocker="parked car"),
    12: dict(group="zebra_front", side="right"),
    13: dict(group="zebra_front", side="left"),
    14: dict(group="zebra_side", side="right"),
    15: dict(group="zebra_side", side="left"),
    16: dict(group="adj_lane", side="right", long="front", leading="no"),
    17: dict(group="adj_lane", side="left", long="front", leading="no"),
    18: dict(group="adj_lane", side="left", long="rear", leading="no"),
    19: dict(group="adj_lane", side="right", long="rear", leading="no"),
    20: dict(group="adj_lane", side="right", long="front", leading="yes"),
    21: dict(group="adj_lane", side="left", long="front", leading="yes"),
    22: dict(group="cross_blocked_lane", side="right"),
    23: dict(group="cross_blocked_lane", side="left"),
    24: dict(group="oncoming_blocked"),
    25: dict(group="intersection", side="right", turn="straight"),
    26: dict(group="intersection", side="left", turn="straight"),
    27: dict(group="intersection", side="oncoming", turn="straight"),
    28: dict(group="intersection", side="right", turn="left"),
    29: dict(group="intersection", side="left", turn="left"),
    30: dict(group="intersection", side="oncoming", turn="left"),
    31: dict(group="intersection", side="right", turn="right"),
    32: dict(group="intersection", side="left", turn="right"),
    33: dict(group="intersection", side="oncoming", turn="right"),
}

GEOMETRY_ATTRS = {
    0: dict(group="turn", turn="right", junction="any"),
    1: dict(group="turn", turn="right", junction="any"),
    2: dict(group="turn", turn="left", junction="4way"),
    3: dict(group="turn", turn="left", junction="3way"),
    4: dict(group="turn", turn="right", junction="4way"),
    5: dict(group="turn", turn="right", junction="3way"),
    6: dict(group="lanes", lanes="right_same"),
    7: dict(group="lanes", lanes="left_same"),
    8: dict(group="lanes", lanes="both_same"),
    9: dict(group="lanes", lanes="single"),
    10: dict(group="lanes", lanes="two_opposite"),
    11: dict(group="straight_cross", signal="signalized"),
    12: dict(group="straight_cross", signal="unsignalized"),
    13: dict(group="unprotected_left"),
}

# What the *code* of each entry actually does, where it differs from its description.
# Established by reading the snippets (see README, "Database audit").
SPAWN_CODE_OVERRIDES = {19: dict(side="left")}          # "rear right" uses _laneToLeft
GEOMETRY_CODE_OVERRIDES = {0: dict(turn="left")}        # "right turn" uses ManeuverType.LEFT_TURN


# --------------------------------------------------------------------------------------------
# Query generation
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Query:
    db: str                 # "spawn" or "geometry"
    attrs: tuple            # sorted (key, value) pairs: the ground truth
    template: int           # which phrasing
    text: str
    flip_of: tuple | None = field(default=None)  # attrs of the spec this one was flipped from


def _a(d):
    return tuple(sorted(d.items()))


OTHER = {"left": "right", "right": "left", "front": "rear", "rear": "front",
         "4way": "3way", "3way": "4way"}
TURN_OTHER = {"left": "right", "right": "left"}

# Each template is a function of the attribute dict. Templates vary wording, clause order and
# vocabulary, the way an LLM's extracted sub-description would.
_ADV = ["The adversarial agent", "The adversary", "An adversarial road user", "The attacking agent"]


def _cross_straight(a, t):
    side, blk = a["side"], a["blocker"]
    hidden = "" if blk == "none" else [f" from behind a {blk}", f", hidden behind a {blk},",
                                       f" behind a {blk}", f" concealed by a {blk}"][t]
    return [
        f"{_ADV[0]} spawns{hidden} on the {side} front of the ego vehicle on a straight road and crosses.",
        f"{_ADV[1]} starts ahead of the ego car on its {side} side{hidden} and then crosses the straight road.",
        f"On a straight road, {_ADV[2].lower()} appears in front of the ego vehicle to the {side}{hidden} to cross the road.",
        f"{_ADV[3]} is positioned{hidden} at the {side}-hand side ahead of the ego vehicle and walks across the straight road.",
    ][t]


def _zebra_front(a, t):
    s = a["side"]
    return [
        f"{_ADV[0]} spawns on the {s} front of the ego vehicle on the zebra crossing at the end of the ego's lane.",
        f"{_ADV[1]} waits at the {s} end of the crosswalk in front of the ego vehicle and then crosses.",
        f"At the pedestrian crossing ahead of the ego car, {_ADV[2].lower()} starts from the {s} side.",
        f"{_ADV[3]} begins on the {s}-hand side of the zebra crossing directly ahead of the ego vehicle.",
    ][t]


def _adj_lane(a, t):
    s, l = a["side"], a["long"]
    pos = {"front": ["front", "ahead of", "in front of", "ahead"], "rear": ["rear", "behind", "behind", "behind"]}[l]
    lead = "" if a["leading"] == "no" else " A slow car drives ahead of the ego vehicle."
    return [
        f"{_ADV[0]} spawns in the {pos[0]} {s} relative to the ego vehicle and drives straight.{lead}",
        f"{_ADV[1]} drives in the {s} adjacent lane, {pos[1]} the ego car, and keeps going straight.{lead}",
        f"In the lane to the {s} of the ego vehicle, {_ADV[2].lower()} starts {pos[2]} the ego and drives straight.{lead}",
        f"{_ADV[3]} is placed {pos[3]} the ego vehicle in its {s}-hand neighbouring lane, driving straight.{lead}",
    ][t]


def _intersection(a, t):
    s, turn = a["side"], a["turn"]
    side_txt = {"right": ["right", "right", "right-hand", "right"],
                "left": ["left", "left", "left-hand", "left"],
                "oncoming": ["oncoming", "opposite", "oncoming", "opposite"]}[s]
    move = {"straight": ["drives straight", "goes straight through", "continues straight", "drives straight on"],
            "left": ["turns left", "makes a left turn", "turns to the left", "takes a left turn"],
            "right": ["turns right", "makes a right turn", "turns to the right", "takes a right turn"]}[turn]
    return [
        f"{_ADV[0]} comes from the {side_txt[0]} side of a four-way intersection and {move[0]}.",
        f"{_ADV[1]} enters the four-way junction from the {side_txt[1]} direction and {move[1]}.",
        f"At a four-way intersection, {_ADV[2].lower()} approaches from the {side_txt[2]} side and {move[2]}.",
        f"{_ADV[3]} arrives at the crossroads from the {side_txt[3]} side and {move[3]}.",
    ][t]


def _diag_rear(a, t):
    s = a["side"]
    return [
        f"{_ADV[0]} spawns diagonally {s} behind the ego vehicle.",
        f"{_ADV[1]} starts behind the ego car, offset to its {s}.",
        f"Behind the ego vehicle and to the {s}, {_ADV[2].lower()} appears.",
        f"{_ADV[3]} is placed at the {s} rear of the ego vehicle, diagonally behind it.",
    ][t]


def _geo_turn(a, t):
    turn, j = a["turn"], a["junction"]
    jt = {"4way": ["four-way intersection", "4-way junction", "crossroads with four arms", "four-way crossing"],
          "3way": ["three-way intersection", "T-junction", "junction with three arms", "three-way crossing"]}[j]
    return [
        f"The ego vehicle makes a {turn} turn at a {jt[0]}.",
        f"The ego car turns {turn} at a {jt[1]}.",
        f"At a {jt[2]}, the ego vehicle performs a {turn} turn.",
        f"The ego vehicle is turning {turn} through a {jt[3]}.",
    ][t]


SPAWN_GENERATORS = {
    "cross_straight": (_cross_straight, {"side": ["left", "right"],
                                         "blocker": ["none", "bus stop", "vending machine", "parked car"]}),
    "zebra_front": (_zebra_front, {"side": ["left", "right"]}),
    "adj_lane": (_adj_lane, {"side": ["left", "right"], "long": ["front", "rear"], "leading": ["no", "yes"]}),
    "intersection": (_intersection, {"side": ["left", "right", "oncoming"], "turn": ["straight", "left", "right"]}),
    "diag_rear": (_diag_rear, {"side": ["left", "right"]}),
}
GEOMETRY_GENERATORS = {
    "turn": (_geo_turn, {"turn": ["left", "right"], "junction": ["4way", "3way"]}),
}
N_TEMPLATES = 4


def _specs(generators):
    for group, (fn, space) in generators.items():
        keys = list(space)
        for values in itertools.product(*(space[k] for k in keys)):
            a = dict(zip(keys, values), group=group)
            if group == "adj_lane" and a["leading"] == "yes" and a["long"] == "rear":
                continue                     # not in the database; no expected answer
            yield fn, a


def build_queries():
    out = []
    for db, gens in (("spawn", SPAWN_GENERATORS), ("geometry", GEOMETRY_GENERATORS)):
        for fn, a in _specs(gens):
            for t in range(N_TEMPLATES):
                out.append(Query(db, _a(a), t, fn(a, t)))
    return out


def matches(entry_attrs: dict, query_attrs: tuple) -> bool:
    """An entry answers a query when it has every queried attribute ("any" is a wildcard)."""
    return all(entry_attrs.get(k) in (v, "any") for k, v in query_attrs)


# Single-attribute flips used for the sensitivity relation: (attribute, value -> other value).
FLIPS = {"side": {"left": "right", "right": "left"},
         "long": {"front": "rear", "rear": "front"},
         "turn": {"left": "right", "right": "left"},
         "junction": {"4way": "3way", "3way": "4way"},
         "blocker": {"bus stop": "parked car", "parked car": "vending machine",
                     "vending machine": "bus stop"}}
