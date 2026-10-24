"""Metamorphic test of ChatScene's retrieval stage (paraphrase invariance + semantic sensitivity).

Reproduces ChatScene's `retrieve_topk` exactly: sentence-t5-large embeddings of the database
descriptions, dot-product scores, top-1 snippet (`use_llm=False`, the repository's default).
Writes results/retrieval.json and prints a summary.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "mrt"))
import safe_pickle  # noqa: E402
import specs  # noqa: E402

DB_PATH = ROOT.parent / "ChatScene" / "retrieve" / "database_v1.pkl"
OUT = ROOT / "results" / "retrieval.json"
ATTRS = {"spawn": specs.SPAWN_ATTRS, "geometry": specs.GEOMETRY_ATTRS}
CODE = {"spawn": specs.SPAWN_CODE_OVERRIDES, "geometry": specs.GEOMETRY_CODE_OVERRIDES}


def code_attrs(db, i):
    return {**ATTRS[db][i], **CODE[db].get(i, {})}


def main():
    database = safe_pickle.load(DB_PATH)
    encoder = SentenceTransformer("sentence-transformers/sentence-t5-large", device="cpu", model_kwargs={"dtype": torch.float32})
    queries = specs.build_queries()

    emb = {db: encoder.encode(database[db]["description"], convert_to_numpy=True) for db in ATTRS}
    q_emb = encoder.encode([q.text for q in queries], convert_to_numpy=True)

    rows = []
    for q, e in zip(queries, q_emb):
        scores = emb[q.db] @ e                      # same as ChatScene: current_embedding @ embeddings.T
        order = np.argsort(-scores, kind="stable")
        top = int(order[0])
        correct = [i for i in ATTRS[q.db] if specs.matches(ATTRS[q.db][i], q.attrs)]
        best_correct = max(correct, key=lambda i: scores[i]) if correct else None
        rows.append(dict(
            db=q.db, attrs=dict(q.attrs), template=q.template, text=q.text, top1=top,
            top1_description=database[q.db]["description"][top].strip(),
            top1_score=float(scores[top]),
            tie=bool(abs(scores[order[0]] - scores[order[1]]) < 1e-6),
            desc_correct=specs.matches(ATTRS[q.db][top], q.attrs),
            code_correct=specs.matches(code_attrs(q.db, top), q.attrs),
            margin=float(scores[top] - scores[best_correct]) if best_correct is not None else None,
            n_correct_entries=len(correct),
        ))

    # ---- paraphrase invariance: phrasings of the same spec should land on the same entry ----
    by_spec = defaultdict(list)
    for r in rows:
        by_spec[(r["db"], tuple(sorted(r["attrs"].items())))].append(r)
    pair_same, spec_all_same = [], []
    for rs in by_spec.values():
        tops = [r["top1"] for r in rs]
        pair_same += [a == b for a, b in combinations(tops, 2)]
        spec_all_same.append(len(set(tops)) == 1)

    # ---- semantic sensitivity: flip one attribute, keep the phrasing ----
    index = {(r["db"], tuple(sorted(r["attrs"].items())), r["template"]): r for r in rows}
    flips = defaultdict(lambda: dict(n=0, unchanged=0, wrong=0))
    for (db, attrs, t), r in index.items():
        a = dict(attrs)
        for key, mapping in specs.FLIPS.items():
            if key not in a or a[key] not in mapping:
                continue
            b = dict(a, **{key: mapping[a[key]]})
            r2 = index.get((db, tuple(sorted(b.items())), t))
            if r2 is None:
                continue
            f = flips[f"{db}:{key}"]
            f["n"] += 1
            f["unchanged"] += int(r2["top1"] == r["top1"])
            f["wrong"] += int(not r2["desc_correct"])

    def rate(xs):
        return float(np.mean(xs)) if xs else float("nan")

    summary = dict(
        n_queries=len(rows), n_specs=len(by_spec),
        top1_accuracy_vs_description=rate([r["desc_correct"] for r in rows]),
        top1_accuracy_vs_code=rate([r["code_correct"] for r in rows]),
        by_db={db: dict(n=sum(r["db"] == db for r in rows),
                        acc_desc=rate([r["desc_correct"] for r in rows if r["db"] == db]),
                        acc_code=rate([r["code_correct"] for r in rows if r["db"] == db]))
               for db in ATTRS},
        by_group={g: rate([r["desc_correct"] for r in rows if r["attrs"]["group"] == g])
                  for g in sorted({r["attrs"]["group"] for r in rows})},
        paraphrase_pair_agreement=rate(pair_same),
        specs_all_phrasings_agree=rate(spec_all_same),
        flips={k: dict(v, unchanged_rate=v["unchanged"] / v["n"], wrong_rate=v["wrong"] / v["n"])
               for k, v in sorted(flips.items())},
        ties=sum(r["tie"] for r in rows),
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
