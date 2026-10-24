"""Summarise the faithfulness audit of ChatScene's 40 released scenarios.

Writes results/trace40.txt (description -> LLM sub-descriptions -> retrieved entries, the evidence
for every annotation) and results/audit.json, and prints the headline numbers.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "audit"))
import safe_pickle  # noqa: E402
from annotations import ANNOTATIONS, CODE_DEFECTS  # noqa: E402

CHATSCENE = ROOT.parent / "ChatScene"
LOG = CHATSCENE / "safebench/scenario/scenario_data/scenic_data/dynamic_scenario/dynamic_log.csv"
COMP = {"B": "behavior", "G": "geometry", "S": "spawn"}
SEVERE = {"SIDE", "TURN", "LONG", "BEHAV", "ACTOR", "CTX", "MIX"}   # meaning changed or contradicted


def main():
    db = safe_pickle.load(CHATSCENE / "retrieve/database_v1.pkl")
    index = {k: {s.strip(): i for i, s in enumerate(db[k]["snippet"])} for k in db}
    rows = list(csv.DictReader(open(LOG)))
    assert len(rows) == len(ANNOTATIONS) == 40

    trace, per = [], []
    for n, r in enumerate(rows):
        ids = {c: index[COMP[c]][r[f"{COMP[c].capitalize()} Snippet"].strip()] for c in COMP}
        trace.append(f"## {n} [{r['AdvObject']}] {r['Scenario']}")
        for c in COMP:
            name = COMP[c]
            trace.append(f"   LLM-{c}: {r[f'{name.capitalize()} Description']}")
            trace.append(f"      -> {c}{ids[c]}: {db[name]['description'][ids[c]].strip()}")
        errs = [(c, code, stage, note) for c, lst in ANNOTATIONS[n].items() for code, stage, note in lst]
        defects = [f"{COMP[c]} {ids[c]}" for c in COMP if (COMP[c], ids[c]) in CODE_DEFECTS]
        for c, code, stage, note in errs:
            trace.append(f"   !! {c} {code} ({'retrieval' if stage == 'R' else 'LLM extraction'}): {note}")
        for d in defects:
            trace.append(f"   !! code defect in retrieved {d}: {CODE_DEFECTS[tuple([d.split()[0], int(d.split()[1])])]}")
        per.append(dict(id=n, scenario=r["Scenario"], retrieved=ids, errors=errs, code_defects=defects,
                        faithful=not errs, severe=any(e[1] in SEVERE for e in errs)))

    codes = Counter(e[1] for p in per for e in p["errors"])
    scen_with = {k: sum(any(e[1] == k for e in p["errors"]) for p in per) for k in codes}
    stages = Counter(e[2] for p in per for e in p["errors"])
    comp = {COMP[c]: sum(any(e[0] == c for e in p["errors"]) for p in per) for c in COMP}
    summary = dict(
        n=len(per),
        fully_faithful=sum(p["faithful"] for p in per),
        with_severe_error=sum(p["severe"] for p in per),
        with_only_added_or_partial=sum((not p["faithful"]) and not p["severe"] for p in per),
        scenarios_per_error_code=dict(sorted(scen_with.items(), key=lambda kv: -kv[1])),
        scenarios_with_error_per_component=comp,
        errors_by_stage={"retrieval": stages["R"], "llm_extraction": stages["E"]},
        scenarios_using_defective_code=sum(bool(p["code_defects"]) for p in per),
        error_instances=sum(codes.values()),
    )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "trace40.txt").write_text("\n".join(trace) + "\n")
    (out / "audit.json").write_text(json.dumps(dict(summary=summary, scenarios=per), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
