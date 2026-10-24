# Does the scenario match the sentence? A first faithfulness and consistency check of ChatScene

A small research prototype for the TUM thesis topic *Evaluating the Consistency of Language-to-Scenario Generators for Autonomous Driving Systems* (Chair of Software Engineering). I built it before applying, to understand the problem and to test the three questions of the topic on a real generator:

1. **Consistency.** Does the same situation, described in different words, give the same scenario?
2. **Faithfulness.** Does the scenario contain what the description says, and nothing it does not say?
3. **Complexity.** Where does it break down?

The generator is **ChatScene** (Zhang, Xu & Li, CVPR 2024), the example named in the topic. Everything here runs on a laptop without CARLA and without an LLM API key, because it targets the part of ChatScene that turns sub-descriptions into Scenic code, and because the ChatScene repository ships its own generated scenarios.

## How ChatScene builds a scenario
```
description ──LLM──▶ adversary type + 3 sub-descriptions ──retrieval──▶ 3 Scenic snippets ──▶ Scenic program ──▶ CARLA
                     (behavior, geometry, spawn position)   sentence-t5-large,
                                                            top-1 by dot product
```
The retrieval database (`retrieve/database_v1.pkl`) holds 58 behavior, 14 geometry and 34 spawn entries, each a natural-language description paired with a Scenic snippet. In the default mode (`use_llm=False`) the top-1 snippet is used as is.

## Results

### A. Faithfulness of ChatScene's own 40 released scenarios (`audit/`)
The repository contains 40 scenarios ChatScene generated (`dynamic_log.csv`), with the original description, the LLM's three sub-descriptions and the retrieved snippets. I compared every retrieved component with the original description. Every judgment is listed with its evidence in `results/trace40.txt` and `audit/annotations.py`.

| | scenarios (of 40) |
|---|---|
| fully faithful | **3** |
| meaning changed or contradicted (side, turn, behavior, road context, actor) | **25** |
| only extra elements or partly matching behavior | 12 |

| error type | scenarios |
|---|---|
| element added that the description never mentions (a parked car, a slow leading car, an intersection on a straight road) | 26 |
| wrong side of the adversary (left / right / oncoming) | 13 |
| behavior only partly matches | 7 |
| wrong road context | 5 |
| wrong ego turn direction | 4 |
| different behavior although a matching entry exists | 4 |
| vehicle-only spawn pattern used for a pedestrian or rider | 4 |
| components contradict each other | 1 |
| front / behind swapped | 1 |

**Where the errors come from:** in all 66 error instances the LLM's sub-description was still correct and the **retrieval step** lost or changed the meaning. Examples: the LLM writes "turning left at an intersection" and retrieval returns "a right turn at an intersection" (scenarios 5, 6, 9); the LLM writes "behind a bus stop on the right front" and retrieval returns the left-front entry (1, 2, 4).

### B. Metamorphic test of the retrieval step (`mrt/`)
Scenario generators have no single correct output, so I used **metamorphic relations**, the standard way to test such systems:
- *paraphrase invariance:* rewording a sub-description must not change the retrieved scenario;
- *semantic sensitivity:* changing one fact (left → right, front → behind, left turn → right turn, bus stop → parked car) must change it accordingly.

124 queries are generated from 31 attribute specifications in 4 phrasings each, so the expected answer is known by construction (`mrt/specs.py`). The retrieval reproduces ChatScene's `retrieve_topk` exactly (same encoder, dot product, top-1).

| measure | result |
|---|---|
| top-1 correct | **65%** (81/124) |
| ...with the phrasing closest to the database wording | 94% |
| ...with the other three phrasings | 48%, 48%, 71% |
| pairs of rewordings that retrieve the same entry | **45%** |
| specs where all four rewordings are answered correctly | 7 / 31 |
| left ↔ right swaps where the retrieved entry does **not change at all** | **31%** (30/96) |
| left ↔ right swaps that end on a wrong entry | 40% |
| score margin of the wrong answers over the best correct entry (median) | 0.006 |

**Why:** the encoder barely separates the words that decide the scenario. Cosine similarity between database descriptions that differ only in one critical word (`results/minimal_pairs.json`):

| pair | cosine |
|---|---|
| parked car on the right front / left front | 0.998 |
| vending machine right / left | 0.997 |
| bus stop right / left | 0.996 |
| intersection: from the right / from the left | 0.993 |
| ego turns left / right at a four-way intersection | 0.989 |
| adjacent lane, front / rear | 0.971 |
| bus stop / parked car (both right) | 0.950 |

So the ranking between "left" and "right" is decided by wording noise. Running the encoder in float16 instead of float32 changed some top-1 answers, another sign of how thin the margins are.

### C. Database audit and execution check (`audit/annotations.py`, `execcheck/`)
Reading the snippets, and then executing them, shows that some entries do not do what their description says:

| entry | description | code |
|---|---|---|
| geometry 0 | "right turn at an intersection" | `ManeuverType.LEFT_TURN`; the description is identical to entry 1 (a real right turn), so the two tie |
| spawn 19 | "rear **right**" | uses `_laneToLeft` |
| spawn 1, 7, 9, 11 | "from the **left** front ... for crossing" | `left of P by Range(-8, -2)`: a negative distance to the left is a distance to the right |
| spawn 2 | "from the right front ... for crossing" | heading `-90 deg`: the agent faces away from the road |

To confirm the spawn cases I executed the snippets in Scenic on CARLA's Town03 map (40 sampled scenes each) and measured the adversary relative to the road point each snippet uses:

| spawn entries | described | adversary lands | faces the road |
|---|---|---|---|
| 1, 7, 9, 11 | left front | **right** side (92.5 to 100% of scenes) | **no** (0%) |
| 2 | right front | right side (100%) | **no** (0%) |
| 6, 8, 10 | right front behind an object | right side (100%) | yes (100%) |

`CrossingBehavior` walks the agent along its own heading (`WalkForwardBehavior` → `SetWalkingDirectionAction(self.heading)`), so the agents that face away walk away from the road. In practice, the database cannot produce a pedestrian crossing from the left on a straight road. The left-of semantics are the same in the Scenic 2.1 that ChatScene bundles (`veneer.py`: offset `-width/2 - dist`); the check ran in Scenic 3.1.1 after a mechanical syntax port.

## What this means for the thesis
- A useful evaluation has to look **inside** the pipeline: here every observed error was introduced after the LLM step, so an end-to-end score alone would blame the wrong component.
- **Paraphrase invariance and semantic sensitivity** are cheap, automatic and catch the most severe errors. The thesis benchmark can generate them from attribute specifications, as `mrt/specs.py` does, and extend them with complexity levels (more actors, events, weather).
- Faithfulness needs both **missing** and **added** content. "Added" was the most frequent error, and an accuracy-style metric that only checks requested elements would miss it.
- **Executing** the scenario matters: the left/right defects are only visible from where objects land, not from the text of the code.

## Limitations (honest scope)
- One generator, and only its retrieval-based mode. ChatScene's LLM-rewrite mode (`use_llm=True`), other generators and an LLM-only baseline need an API key and are the next step.
- The audit of the 40 scenarios is my own annotation. It should be double-annotated with an agreement measure in the thesis.
- My query phrasings are template-generated. They are realistic but not a sample of real user prompts.
- The execution check uses Scenic's built-in sampling on the map, not a CARLA simulation, and covers the straight-road crossing entries only.
- AI assistance: I used Claude (Anthropic) as a coding and writing assistant. I chose the questions, checked every finding against the code, and can explain every part (see `WALKTHROUGH.md`).

## Run
```bash
git clone https://github.com/javyduck/ChatScene ../ChatScene   # MIT licence, not vendored
python3.11 -m venv .venv && .venv/bin/pip install scenic sentence-transformers pytest
.venv/bin/python -m pytest -q tests          # 6 tests
python3 audit/summarize.py                    # A: audit of the 40 released scenarios (stdlib only)
.venv/bin/python mrt/run_retrieval.py         # B: metamorphic retrieval test (~2 min on CPU)
.venv/bin/python mrt/minimal_pairs.py         # B: cosine of minimal pairs
cp ../ChatScene/Scenic/tests/formats/opendrive/maps/CARLA/Town03.xodr execcheck/
cd execcheck && ../.venv/bin/python build.py  # C: execute spawn snippets on Town03
```
`tools/safe_pickle.py` reads ChatScene's database without unpickling it: it rebuilds the dicts, lists and strings from the opcode stream and refuses anything else, so no code from the file can run.

`results/trace40.txt` quotes scenario descriptions and database descriptions from ChatScene (MIT licence, Zhang, Xu & Li); the ChatScene code itself is not included.
