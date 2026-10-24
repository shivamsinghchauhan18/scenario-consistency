"""Checks of the harness itself: safe pickle reader, query ground truth, annotation integrity."""
import io
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("tools", "mrt", "audit"):
    sys.path.insert(0, str(ROOT / sub))
import safe_pickle  # noqa: E402
import specs  # noqa: E402
from annotations import ANNOTATIONS, CODE_DEFECTS  # noqa: E402

DB = ROOT.parent / "ChatScene" / "retrieve" / "database_v1.pkl"


def test_safe_pickle_roundtrip(tmp_path):
    data = {"a": {"description": ["x", "y"], "snippet": ["p", "q"]}, "b": {"description": [], "snippet": []}}
    f = tmp_path / "d.pkl"
    f.write_bytes(pickle.dumps(data, protocol=4))
    assert safe_pickle.load(f) == data


def test_safe_pickle_refuses_code(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    f = tmp_path / "e.pkl"
    f.write_bytes(pickle.dumps([Evil()]))
    with pytest.raises(ValueError, match="refusing opcode"):
        safe_pickle.load(f)


@pytest.mark.skipif(not DB.exists(), reason="ChatScene clone not present")
def test_annotations_cover_database():
    db = safe_pickle.load(DB)
    assert sorted(specs.SPAWN_ATTRS) == list(range(len(db["spawn"]["description"])))
    assert sorted(specs.GEOMETRY_ATTRS) == list(range(len(db["geometry"]["description"])))


def test_every_query_has_an_answer_and_flips_change_ground_truth():
    queries = specs.build_queries()
    assert len(queries) == 124
    for q in queries:
        attrs = specs.SPAWN_ATTRS if q.db == "spawn" else specs.GEOMETRY_ATTRS
        assert any(specs.matches(a, q.attrs) for a in attrs.values()), q.text


def test_matches_treats_any_as_wildcard():
    assert specs.matches({"turn": "right", "junction": "any"}, (("junction", "4way"), ("turn", "right")))
    assert not specs.matches({"turn": "left", "junction": "any"}, (("turn", "right"),))


def test_annotation_schema():
    assert sorted(ANNOTATIONS) == list(range(40))
    codes = {"SIDE", "TURN", "LONG", "ADDED", "CTX", "BEHAV", "PART", "ACTOR", "MIX"}
    for comps in ANNOTATIONS.values():
        for comp, errs in comps.items():
            assert comp in "BGS"
            for code, stage, note in errs:
                assert code in codes and stage in "RE" and note
    assert all(db in ("spawn", "geometry") for db, _ in CODE_DEFECTS)
