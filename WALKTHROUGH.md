# Walkthrough: understand and own this prototype

Work through this in order (about 4 hours). By the end you should be able to explain every number in the README, answer the questions in Part 6 out loud, and change the code yourself.

## Part 1: the domain, from zero

**1.1 Why scenario-based testing.** An automated driving system (ADS) cannot be proven safe by driving millions of kilometres: dangerous situations are too rare. Instead, engineers describe *specific* situations (a pedestrian steps out behind a bus, a car runs a red light) and test the ADS in simulation. A **scenario** = road layout + actors + their behavior over time + environment (weather, light). A **concrete** scenario fixes every number; a **logical** scenario gives ranges (speed 0 to 10 m/s) from which concrete ones are sampled.

**1.2 Safety-critical scenarios.** Most of normal driving is easy. Testing wants the rare, hard cases where the ADS might fail, so generators try to produce *adversarial* scenarios: an "adversary" (car, pedestrian, cyclist, motorcycle) behaves in a way that provokes a collision.

**1.3 How scenarios are written down.** Two families:
- **OpenSCENARIO** (ASAM standard, XML) plus **OpenDRIVE** (`.xodr`, the road network). Industry standard.
- **Scenic** (Fremont et al., PLDI 2019): a *probabilistic programming language* for scenes. `param SPEED = Range(0, 10)` declares a distribution; every run samples a concrete scene. It has driving-specific vocabulary: lanes, intersections, maneuvers, `OrientedPoint following roadDirection from X for 20`.

**1.4 Simulators.** **CARLA** (Unreal Engine, Linux/Windows + NVIDIA GPU) is the standard open-source driving simulator; Scenic drives it through an interface. Scenic also has a simple built-in 2D simulator and a MetaDrive interface. Town01 to Town10 are CARLA's built-in maps, shipped as `.xodr` files.

**1.5 Language-to-scenario generation.** Writing Scenic by hand is slow. LLM-based generators take a sentence ("a pedestrian suddenly crosses from the right front") and produce an executable scenario. ChatScene (CVPR 2024) is one; others generate OpenSCENARIO or Python directly. The survey in the topic's references (Gao et al., 2026) maps the field.

➡️ **Exercise:** open `../ChatScene/safebench/scenario/scenario_data/scenic_data/dynamic_scenario/dynamic_7.scenic`. Find the description, the behavior, the ego's route, and the adversary. Which line puts a parked car into the scene?

## Part 2: ChatScene in detail
1. An LLM (GPT-4 in the paper) splits the description into **adversary type, behavior, geometry, spawn position** (`retrieve/prompts/extraction.txt`, few-shot).
2. Each sub-description is embedded with **sentence-t5-large** and compared by **dot product** with every database description of its kind (`utils.retrieve_topk`).
3. The top-1 snippet of each kind is concatenated with a fixed header into one Scenic program (`retrieve.py`). With `--use_llm`, the LLM instead rewrites a new snippet from the top-k examples.
4. The program is compiled; SafeBench runs it in CARLA and tunes the parameter ranges towards collisions (50 runs, Gaussian refit every 10).

The paper evaluates collision rate, an overall driving score and diversity (ADE). It does **not** measure whether the generated scenario matches the description. That is exactly the gap this thesis topic targets.

➡️ **Exercise:** in `retrieve.py`, where would an error enter if the LLM wrote "left" but the scenario has "right"? List the three places (extraction, retrieval, snippet code).

## Part 3: the concepts behind the measurements

**3.1 Sentence embeddings.** An encoder maps a sentence to a vector (768 numbers for sentence-t5-large) so that similar meanings are close. Similarity = cosine (the dot product of normalised vectors). The weakness: these models are trained for *topic* similarity. "Pedestrian from the left front" and "from the right front" are about the same topic, so they get nearly the same vector (cosine 0.985 to 0.998 in `results/minimal_pairs.json`). Negation, direction and numbers are known blind spots.

**3.2 Metamorphic testing** (Chen et al., 1998; Segura et al., TSE 2016). When you cannot say what the correct output is (the *test oracle problem*), you test **relations between outputs**: if input changes in a known way, the output must change in a known way. Here:
- *MR1 invariance:* paraphrase(q) ⇒ same scenario;
- *MR2 sensitivity:* flip(q, left→right) ⇒ scenario with the flipped attribute.

This is a software-engineering method, which is why it fits a thesis at a software engineering chair.

**3.3 The three quality dimensions of the topic, made measurable.**
| dimension | operational definition | measured here as |
|---|---|---|
| consistency | equivalent inputs → equivalent outputs | MR1: same entry across 4 phrasings (45% of pairs) |
| faithfulness / accuracy | nothing **missing**, nothing **changed**, nothing **added** | audit codes SIDE/TURN/BEHAV (changed), PART (missing), ADDED (added) |
| compositional complexity | quality as the number of actors, events and conditions grows | *not yet*: next step, add complexity levels to `specs.py` |

**3.4 Ground truth by construction.** `mrt/specs.py` writes each query from an attribute dictionary (`side=left, blocker=bus stop`). Because the text is generated from the attributes, the correct answer is known without manual labelling. The thesis benchmark prompt set can work the same way: a structured specification, rendered into many natural-language variants.

**3.5 Where in the pipeline.** Testing only end-to-end (sentence in, simulation out) tells you *that* it failed, not *where*. ChatScene logs every intermediate step, so the audit could attribute all 66 errors to retrieval. A good thesis framework keeps this per-stage view.

**3.6 Scenic semantics you must know.** In Scenic's local frame, x points to the right and y forward. `left of P by d` places the object at offset `(-width/2 - d, 0)` from P, so a **negative** d moves it to the right. Headings are counter-clockwise: `heading + 90 deg` faces left, `- 90 deg` faces right. Verify it yourself: `semantics/probe.scenic`.

➡️ **Exercise:** spawn 1 uses `left of IntSpawnPt by Range(-8, -2)` with heading `-90 deg`. On paper: which side is the pedestrian on, and which way does it walk? Then check `results/execcheck.json`.

## Part 4: read the code (in this order)
| file | what to look for |
|---|---|
| `tools/safe_pickle.py` | why loading an untrusted pickle is dangerous, and how walking the opcodes avoids it |
| `mrt/specs.py` | database annotations; templates; `matches` ("any" wildcard); the `FLIPS` table |
| `mrt/run_retrieval.py` | the exact copy of `retrieve_topk`; how MR1 (pairs per spec) and MR2 (same template, one flipped attribute) are counted |
| `audit/annotations.py` | the error codes; one entry per scenario; `CODE_DEFECTS` |
| `audit/summarize.py` | how per-scenario judgments become the table in the README |
| `execcheck/build.py` | porting Scenic 2 to 3 (`new`), sampling scenes, measuring lateral offset and facing relative to the snippet's own road point |

## Part 5: run and change things
```bash
cd ~/thesis_work/scenario-consistency
.venv/bin/python -m pytest -q tests
python3 audit/summarize.py
.venv/bin/python mrt/run_retrieval.py
```
Predict before you run:
1. Add a fifth template to `_cross_straight` that copies the database wording exactly. Does accuracy for that template reach 100%? Why is that *not* good news?
2. Change `PAIRS` in `minimal_pairs.py` to compare "pedestrian" vs. "cyclist" descriptions. Higher or lower cosine than left/right?
3. In `execcheck/build.py`, change spawn 1's range to `Range(2, 8)` in the generated file. Where does the agent land now?
4. Re-annotate scenarios 0 to 9 yourself without looking at `annotations.py`, then compare. Where do you disagree? (This is inter-annotator agreement, which the thesis needs.)

## Part 6: numbers to know, and questions to practise out loud
**Numbers:** 3/40 fully faithful, 25/40 changed meaning, 26/40 with added elements, 66/66 errors at retrieval. 65% top-1 on 124 queries; 94% with database-like wording vs. 48 to 71% otherwise; 45% paraphrase agreement; 31% of left/right flips change nothing. Left/right cosine 0.985 to 0.998. Spawns 1, 7, 9, 11 land on the right and face away.

1. *"Isn't this just a bug in one repository?"* The code defects are, and they are easy to fix. The retrieval findings are not: any retrieval step built on a topic-similarity encoder will confuse left and right. The thesis question is how to detect such failures systematically for *any* generator.
2. *"How do you know your annotation is right?"* Every judgment has its evidence in `trace40.txt`; the critical ones (left/right) were confirmed by execution. For the thesis: two annotators and Cohen's kappa.
3. *"Your queries are templates, not real prompts."* True, and deliberate: templates give ground truth for free. The thesis should mix them with LLM-paraphrased and human-written prompts and report results per source.
4. *"How would you compare generators that output different formats?"* Parse every output into one intermediate representation (actors, positions relative to ego, behaviors, road type, environment), then compare representations, not code. Execution in a simulator gives the most reliable extraction.
5. *"What about LLM non-determinism?"* Run each prompt several times at fixed temperature and separate *self-consistency* (same prompt, repeated) from *paraphrase consistency* (different prompts).
6. *"What would you do first?"* Agree the generator set and scenario formats with you; build the intermediate representation; extend `specs.py` with complexity levels; run ChatScene with an LLM plus one or two other generators; double-annotate a sample.
