"""Cosine similarity of database descriptions that differ in one meaning-critical word."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); import safe_pickle
db = safe_pickle.load(ROOT.parent / "ChatScene/retrieve/database_v1.pkl")
PAIRS = [("spawn", 1, 2, "left/right front, crossing"), ("spawn", 6, 7, "bus stop right/left"),
         ("spawn", 8, 9, "vending machine right/left"), ("spawn", 10, 11, "parked car right/left"),
         ("spawn", 12, 13, "zebra right/left"), ("spawn", 16, 19, "adjacent lane front/rear (right)"),
         ("spawn", 16, 17, "adjacent lane right/left (front)"), ("spawn", 25, 26, "intersection right/left side"),
         ("spawn", 28, 31, "intersection turns left/right"), ("geometry", 2, 4, "ego left/right turn (4-way)"),
         ("geometry", 2, 3, "4-way/3-way junction"), ("spawn", 6, 10, "bus stop/parked car (right)")]
enc = SentenceTransformer("sentence-transformers/sentence-t5-large", device="cpu", model_kwargs={"dtype": torch.float32})
out = []
for dbn, i, j, label in PAIRS:
    a, b = enc.encode([db[dbn]["description"][i], db[dbn]["description"][j]])
    out.append(dict(label=label, db=dbn, i=i, j=j, cosine=float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))
    print(f"{label:38s} {dbn}[{i}] vs [{j}]  cos = {out[-1]['cosine']:.4f}")
(ROOT / "results/minimal_pairs.json").write_text(json.dumps(out, indent=1))
