"""Tests that exercise the defects fixed during the audit.

Each test maps to a bug that previously existed and, in most cases, silently produced
wrong output rather than raising.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import pipeline_v2
from pipeline_v2 import EventStore, _parse_time_range, answer, color_name, format_ts
from services import EVENT_COLUMNS, IncidentRules, event_values
from vision_core import MiniFASNet, TrackIdentityStabilizer, align_face, expanded_crop


# --------------------------------------------------------------------------------------
# Regression: answer() used to crash on any question containing a timestamp
# --------------------------------------------------------------------------------------


def test_answer_does_not_crash_on_timestamp(tmp_path):
    db = tmp_path / "events.sqlite"
    store = EventStore(str(db))
    store.add(
        video="clip.mp4",
        track_id=3,
        ts=150.0,
        event_type="standing",
        shirt_color="black",
        identity="menna",
        identity_score=0.71,
        action="standing",
        action_score=0.9,
        vlm=None,
        bbox=[1, 2, 3, 4],
        evidence={"path": "evidence/x.jpg", "frame": 42},
        clip=None,
    )
    result = answer("who was wearing black at 02:30?", str(db))
    assert result["evidence"], "evidence must be returned"
    assert result["query"]["start"] == 150.0


def test_answer_without_timestamp_still_works(tmp_path):
    db = tmp_path / "events.sqlite"
    EventStore(str(db)).add(
        video="clip.mp4", track_id=1, ts=5.0, event_type="walking", shirt_color="red",
        identity="ana", identity_score=0.8, action=None, action_score=0.0, vlm=None,
        bbox=[0, 0, 1, 1], evidence={"path": "a.jpg"}, clip=None,
    )
    result = answer("what happened?", str(db))
    assert result["evidence"]


def test_answer_reports_empty_result_not_error(tmp_path):
    db = tmp_path / "events.sqlite"
    EventStore(str(db))
    result = answer("who wore green at 10:00?", str(db))
    assert result["evidence"] == []
    assert "no evidence" in result["answer"].lower()


# --------------------------------------------------------------------------------------
# Timestamp parsing and formatting
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected_start,expected_end",
    [
        ("at 2:30", 150.0, 450.0),
        ("between 1:00 and 2:00", 60.0, 120.0),
        ("from 00:01:00 to 00:02:30", 60.0, 150.0),
        ("anything", 0.0, float("inf")),
    ],
)
def test_parse_time_range(question, expected_start, expected_end):
    start, end = _parse_time_range(question)
    assert start == expected_start
    if expected_end == float("inf"):
        assert end == float("inf")
    else:
        assert end == expected_end


def test_format_ts_is_video_relative_and_stable():
    assert format_ts(0) == "00:00:00.000"
    assert format_ts(150.5) == "00:02:30.500"
    assert format_ts(3723.25) == "01:02:03.250"
    assert format_ts(-5) == "00:00:00.000"


def test_format_ts_never_uses_wall_clock():
    # 150 seconds into a video must render as 00:02:30, not as a UTC clock reading.
    assert not format_ts(150).startswith("1970")


# --------------------------------------------------------------------------------------
# MiniFASNet preprocessing contract
# --------------------------------------------------------------------------------------


def test_minifasnet_default_scale_is_raw_0_255():
    """Measured on this export: raw 0-255 BGR discriminates; x/255 collapses the output."""
    assert MiniFASNet.SCALE_RAW == "raw"
    crop = np.full((60, 60, 3), 255, np.uint8)
    blob = MiniFASNet._preprocess(crop)
    assert blob.shape == (1, 3, 80, 80), "input must be NCHW 80x80"
    assert blob.dtype == np.float32
    assert float(blob.max()) == pytest.approx(255.0), "default scale must keep 0-255"


def test_minifasnet_scale_variants():
    white = np.full((60, 60, 3), 255, np.uint8)
    assert float(MiniFASNet._preprocess(white, MiniFASNet.SCALE_UNIT).max()) == pytest.approx(1.0)
    assert float(MiniFASNet._preprocess(white, MiniFASNet.SCALE_HALF).max()) == pytest.approx(1.0)
    assert float(MiniFASNet._preprocess(np.zeros((60, 60, 3), np.uint8), MiniFASNet.SCALE_HALF).min()) == pytest.approx(-1.0)


def test_minifasnet_preprocess_does_not_swap_channels():
    bgr = np.zeros((10, 10, 3), np.uint8)
    bgr[..., 0] = 255  # blue only
    blob = MiniFASNet._preprocess(bgr)
    assert float(blob[0, 0].mean()) == pytest.approx(255.0), "blue channel must stay first"
    assert float(blob[0, 2].mean()) == pytest.approx(0.0), "red channel must stay last"


def test_minifasnet_default_live_class_is_one():
    assert MiniFASNet.DEFAULT_LIVE_CLASS == 1


def test_minifasnet_empty_crop_is_not_live():
    live, score = MiniFASNet.__new__(MiniFASNet).predict(np.zeros((0, 0, 3), np.uint8))
    assert live is False and score == 0.0


# --------------------------------------------------------------------------------------
# Track identity stability
# --------------------------------------------------------------------------------------


class FakeIndex:
    """Keys identity on embedding *direction*, matching how a real cosine index behaves."""

    def __init__(self):
        self.calls = 0

    def identify(self, vector, threshold=0.35):
        self.calls += 1
        v = np.asarray(vector, dtype=np.float64)
        # alice sits on the x axis, bob on the y axis
        identity = "alice" if abs(v[0]) >= abs(v[1]) else "bob"
        return {"identity": identity, "score": 0.9}


ALICE = np.array([1.0, 0.0, 0.0], np.float32)
BOB = np.array([0.0, 1.0, 0.0], np.float32)


def test_track_identity_does_not_flicker_on_single_bad_frame():
    stab = TrackIdentityStabilizer(FakeIndex(), min_updates_before_switch=3, agree_ratio=0.6)

    for _ in range(5):
        result = stab.update(1, ALICE, quality=1.0)
    assert result["identity"] == "alice"

    # one stray frame must not relabel the track
    stray = stab.update(1, BOB, quality=1.0)
    assert stray["identity"] == "alice", "a single disagreeing frame must not flip the label"


def test_track_identity_switches_when_evidence_persists():
    stab = TrackIdentityStabilizer(FakeIndex(), min_updates_before_switch=3, agree_ratio=0.6)
    for _ in range(5):
        stab.update(7, ALICE, quality=1.0)
    for _ in range(6):
        result = stab.update(7, BOB, quality=1.0)
    assert result["identity"] == "bob", "sustained new evidence must eventually relabel"


def test_tracks_are_independent():
    stab = TrackIdentityStabilizer(FakeIndex())
    stab.update(1, ALICE, quality=1.0)
    stab.update(2, BOB, quality=1.0)
    snap = stab.snapshot()
    assert snap[1]["identity"] == "alice"
    assert snap[2]["identity"] == "bob"


# --------------------------------------------------------------------------------------
# Event store
# --------------------------------------------------------------------------------------


def test_event_store_round_trip(tmp_path):
    db = tmp_path / "e.sqlite"
    store = EventStore(str(db))
    event_id = store.add(
        video="v.mp4", track_id=2, ts=12.5, event_type="walking", shirt_color="blue",
        identity="zoe", identity_score=0.6, action="walking", action_score=0.4,
        vlm="a person in blue", bbox=[1, 2, 3, 4], evidence={"path": "p.jpg"}, clip="c.mp4",
    )
    rows = store.search(color="blue", start=10, end=20, identity="zoe")
    assert len(rows) == 1
    assert rows[0][0] == event_id
    assert rows[0][6] == "zoe"
    # search() selects: id, track_id, ts, event_type, shirt_color, identity,
    # identity_score, action, action_score, vlm, evidence, clip
    assert rows[0][10] == "a person in blue", "vlm must round-trip"
    assert json.loads(rows[0][11])["path"] == "p.jpg", "evidence must round-trip"
    assert rows[0][12] == "c.mp4", "clip must round-trip"


def test_event_store_migrates_legacy_schema(tmp_path):
    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, video TEXT, track_id INTEGER, ts REAL, event_type TEXT, shirt_color TEXT, bbox TEXT, evidence TEXT)")
    conn.execute("INSERT INTO events(id,video,track_id,ts,event_type,shirt_color,bbox,evidence) VALUES(1,'old.mp4',9,3.0,'walking','red','[1,2,3,4]','{\"path\":\"old.jpg\"}')")
    conn.commit()
    conn.close()
    store = EventStore(str(db))  # must rebuild the table, not fail
    existing = store.all_events()
    assert len(existing) == 1, "legacy rows must survive migration"
    assert existing[0][1] == 9, "legacy track_id must be preserved"
    assert existing[0][6] is None, "new columns must be added and left null"
    store.add(
        video="v", track_id=1, ts=1.0, event_type="x", shirt_color="red",
        identity="i", identity_score=1.0, action=None, action_score=0.0,
        vlm=None, bbox=[0, 0, 1, 1], evidence={}, clip=None,
    )
    assert len(store.all_events()) == 2


# --------------------------------------------------------------------------------------
# Regression: Postgres insert had 8 placeholders but 7 values
# --------------------------------------------------------------------------------------


def test_event_values_matches_event_columns():
    values = event_values("abc", {"video": "v", "track_id": 1, "ts": 2.0, "event_type": "e"})
    assert len(values) == len(EVENT_COLUMNS), (
        f"placeholder/value count must match: got {len(values)} values for {len(EVENT_COLUMNS)} columns"
    )


# --------------------------------------------------------------------------------------
# Incident rules
# --------------------------------------------------------------------------------------


def test_incident_rules_match_and_do_not_match(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps([{"id": "r1", "when": {"shirt_color": "black", "event_type": "moving quickly"}}]))
    rules = IncidentRules(str(rules_path))
    assert rules.evaluate({"shirt_color": "black", "event_type": "moving quickly"})
    assert not rules.evaluate({"shirt_color": "black", "event_type": "standing"})
    assert not rules.evaluate({"shirt_color": "white", "event_type": "moving quickly"})


def test_incident_rules_ignore_empty_conditions(tmp_path):
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps([{"id": "catchall", "when": {}}]))
    rules = IncidentRules(str(rules_path))
    assert rules.evaluate({"anything": 1}) == []


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def test_color_name_on_empty_crop():
    assert color_name(np.zeros((0, 0, 3), np.uint8)) == "unknown"


def test_color_name_black():
    assert color_name(np.zeros((40, 40, 3), np.uint8)) == "black"


def test_expanded_crop_default_scale_matches_model():
    import inspect

    sig = inspect.signature(expanded_crop)
    assert sig.parameters["scale"].default == 2.7


def test_align_face_shape():
    image = np.zeros((160, 160, 3), np.uint8)
    landmarks = np.array([[50, 60], [90, 60], [70, 80], [55, 105], [85, 105]], np.float32)
    assert align_face(image, landmarks).shape == (112, 112, 3)


def test_align_face_rejects_degenerate():
    image = np.zeros((160, 160, 3), np.uint8)
    degenerate = np.array([[50, 60], [50, 60], [50, 60], [50, 60], [50, 60]], np.float32)
    with pytest.raises(ValueError):
        align_face(image, degenerate)


# --------------------------------------------------------------------------------------
# Regression: input normalisation is model-specific and silently breaks embeddings
# --------------------------------------------------------------------------------------


def test_sface_is_configured_for_raw_bgr():
    """SFace wants raw BGR 0-255. Feeding it (x-127.5)/127.5 collapses the embedding."""
    import config

    info = config._EMBEDDERS["sface"]
    assert info["norm"] == "raw_bgr"
    assert info["alignment"] == "arcface"
    assert info["dim"] == 128


def test_blob_normalisation_dispatch():
    from vision_core import OnnxEmbedder512

    e = OnnxEmbedder512.__new__(OnnxEmbedder512)
    e.expected_size = (112, 112)
    bgr = np.zeros((112, 112, 3), np.uint8)
    bgr[..., 0] = 255  # blue channel only

    e.norm = OnnxEmbedder512.NORM_RAW_BGR
    blob = e._to_blob(bgr)
    assert blob.shape == (1, 3, 112, 112)
    assert float(blob[0, 0].mean()) == pytest.approx(255.0), "raw_bgr must keep 0-255 scale"
    assert float(blob[0, 2].mean()) == pytest.approx(0.0), "raw_bgr must keep BGR order"

    e.norm = OnnxEmbedder512.NORM_RAW_RGB
    blob = e._to_blob(bgr)
    assert float(blob[0, 2].mean()) == pytest.approx(255.0), "raw_rgb must swap to RGB order"

    e.norm = OnnxEmbedder512.NORM_HALF_RGB
    blob = e._to_blob(bgr)
    assert float(blob[0, 2].mean()) == pytest.approx(1.0), "half_rgb maps 255 -> +1"
    assert float(blob[0, 0].mean()) == pytest.approx(-1.0), "half_rgb maps 0 -> -1"

    e.norm = OnnxEmbedder512.NORM_UNIT_BGR
    blob = e._to_blob(bgr)
    assert float(blob[0, 0].mean()) == pytest.approx(1.0), "unit_bgr maps 255 -> 1"


def test_unknown_norm_is_rejected():
    from vision_core import OnnxEmbedder512

    e = OnnxEmbedder512.__new__(OnnxEmbedder512)
    e.expected_size = (112, 112)
    e.norm = "not_a_norm"
    with pytest.raises(ValueError):
        e._to_blob(np.zeros((112, 112, 3), np.uint8))


def test_model_identity_embedding_is_not_collapsed():
    """The real defect: a face and a stranger scored 0.94, higher than a face and its own
    brightened copy. Guard the genuine/impostor gap on the shipped weights."""
    pytest.importorskip("onnxruntime")
    if not Path("models/sface.onnx").exists():
        pytest.skip("sface model not present")
    import cv2

    from vision_core import OnnxEmbedder512, YuNetDetector

    det = YuNetDetector("models/yunet.onnx")
    emb = OnnxEmbedder512("models/sface.onnx")
    lfw = Path("data/datasets/lfw")
    files = sorted(lfw.glob("*/*.jpg"))
    if len(files) < 3:
        pytest.skip("LFW not present")

    def vec(path, brighten=False):
        img = cv2.imread(str(path))
        if brighten:
            img = cv2.convertScaleAbs(img, alpha=1.15, beta=10)
        face = max(det.detect(img), key=lambda f: f.confidence)
        return emb.embed(emb.align(img, face))

    a = vec(files[0])
    same_perturbed = vec(files[0], brighten=True)
    stranger = vec(files[500])

    genuine = float(a @ same_perturbed)
    impostor = float(a @ stranger)
    assert genuine > 0.85, f"perturbed copy should stay close, got {genuine:.3f}"
    assert impostor < 0.35, (
        f"a different person must not look identical; got impostor cosine {impostor:.3f}. "
        "The embedding is collapsed - check OnnxEmbedder512 norm/alignment for this model."
    )
    assert genuine - impostor > 0.3, f"separation too small: genuine={genuine:.3f} impostor={impostor:.3f}"
