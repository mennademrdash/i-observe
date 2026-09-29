"""Video -> tracks -> face identity -> action/VLM -> events -> evidence -> answers.

Everything that used to sit in an unused adapter module is wired into this loop
directly: PostgreSQL + MinIO persistence, incident rules, automatic evidence clips,
VideoMAE action recognition and the VLM describer.

Optional backends degrade loudly rather than silently: if PostgreSQL, MinIO, VideoMAE
or the VLM server are unavailable, ``process()`` records that in the returned report
under ``integrations`` instead of pretending the feature ran.
"""
from __future__ import annotations

import config  # noqa: F401  # sets BLAS thread caps before numpy/OpenCV load

import argparse
import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from vision_core import (
    EvidenceWriter,
    MiniFASNet,
    OnnxEmbedder512,
    TurboVecFaceIndex,
    TrackIdentityStabilizer,
    YOLO11FaceDetector,
    align_face,
    expanded_crop,
)

_YOLO = None
_YOLO_ERROR = None


def load_yolo():
    """Import ultralytics lazily.

    ``ultralytics`` pulls in ``torch``, which is a large import and raises ``MemoryError``
    on a machine with little free RAM. Modules such as ``answer()`` and ``EventStore``
    must stay usable without paying that cost, so the import happens on first use.
    """
    global _YOLO, _YOLO_ERROR
    if _YOLO is not None:
        return _YOLO
    if _YOLO_ERROR is not None:
        raise RuntimeError(_YOLO_ERROR)
    try:
        from ultralytics import YOLO
    except Exception as exc:  # ImportError or MemoryError
        _YOLO_ERROR = (
            f"Could not import ultralytics ({type(exc).__name__}: {exc}). "
            "Install it with `pip install ultralytics` and free some memory if this is a MemoryError."
        )
        raise RuntimeError(_YOLO_ERROR) from exc
    _YOLO = YOLO
    return _YOLO


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------

DB_SCHEMA = """CREATE TABLE IF NOT EXISTS events(
 id TEXT PRIMARY KEY, video TEXT, track_id INTEGER, ts REAL,
 wall_ts REAL,
 event_type TEXT, shirt_color TEXT, identity TEXT, identity_score REAL,
 action TEXT, action_score REAL, vlm TEXT,
 bbox TEXT, evidence TEXT, clip TEXT)"""

# Columns added after the first draft; keep old databases working.
_DB_MIGRATIONS = [
    "ALTER TABLE events ADD COLUMN identity TEXT",
    "ALTER TABLE events ADD COLUMN identity_score REAL",
    "ALTER TABLE events ADD COLUMN action TEXT",
    "ALTER TABLE events ADD COLUMN action_score REAL",
    "ALTER TABLE events ADD COLUMN vlm TEXT",
    "ALTER TABLE events ADD COLUMN clip TEXT",
    # wall_ts is the real-world time the event was recorded. ts alone is an offset into a
    # video, so questions like "what happened yesterday" cannot be answered without it.
    "ALTER TABLE events ADD COLUMN wall_ts REAL",
]

_ALL_COLUMNS = (
    "id", "video", "track_id", "ts", "wall_ts", "event_type", "shirt_color", "identity",
    "identity_score", "action", "action_score", "vlm", "bbox", "evidence", "clip",
)
_REQUIRED_COLUMNS = ("id", "video", "track_id", "ts", "event_type")


def _parse_time_token(token: str) -> float:
    """Parse ``MM:SS`` or ``HH:MM:SS`` into seconds from the start of the video."""
    parts = [int(p) for p in token.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unrecognised time token: {token}")


def format_ts(seconds: float) -> str:
    """Render a video-relative timestamp as HH:MM:SS.mmm.

    These are offsets into the recording, not wall-clock times. The old code passed the
    offset to ``time.gmtime`` and presented it as a clock time, which is wrong for any
    video that does not begin at midnight.
    """
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------


class EventStore:
    """SQLite event timeline. Always available, never optional."""

    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self._ensure_schema()
        self._lock = threading.Lock()

    def _ensure_schema(self):
        """Create the table, or migrate a legacy one whose ``id`` is INTEGER PRIMARY KEY.

        SQLite's ``INTEGER PRIMARY KEY`` is a rowid alias and rejects text, so inserting
        a UUID string into a legacy database raised ``datatype mismatch``. The table is
        rebuilt in place so existing rows survive.
        """
        exists = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        ).fetchone()
        # A migration killed between RENAME and recreate leaves only the temp table. Recover
        # by renaming it back instead of raising, so a crash cannot brick the database.
        orphan = self.db.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self._TEMP}'"
        ).fetchone()
        if exists is None and orphan is not None:
            self.db.execute(f"ALTER TABLE {self._TEMP} RENAME TO events")
            exists = ("events",)
        if exists is None:
            self.db.execute(DB_SCHEMA)
        else:
            info = self.db.execute("PRAGMA table_info(events)").fetchall()
            columns = {row[1] for row in info}
            id_type = next((row[2] for row in info if row[1] == "id"), "").upper()
            id_is_rowid = any(row[1] == "id" and row[5] for row in info) or id_type == "INTEGER"
            if id_is_rowid or not set(_REQUIRED_COLUMNS).issubset(columns):
                self._migrate_legacy(info)
        for statement in _DB_MIGRATIONS:
            try:
                self.db.execute(statement)
            except sqlite3.OperationalError:
                pass  # column already present
        self.db.commit()

    _TEMP = "events_legacy_migration"

    def _migrate_legacy(self, info):
        old_cols = [row[1] for row in info]
        # A previous run that was interrupted mid-migration can leave the temp table
        # behind, which makes the RENAME fail with "there is already another table".
        # The rebuild copies from whatever the old table was, so clear the temp first.
        self.db.execute(f"DROP TABLE IF EXISTS {self._TEMP}")
        self.db.execute(f"ALTER TABLE events RENAME TO {self._TEMP}")
        self.db.execute(DB_SCHEMA)
        available = [c for c in old_cols if c in _ALL_COLUMNS]
        if available:
            select = ", ".join(
                f"CAST({c} AS TEXT)" if c == "id" else c for c in available
            )
            try:
                self.db.execute(
                    f"INSERT INTO events({','.join(available)}) SELECT {select} FROM {self._TEMP}"
                )
            except sqlite3.OperationalError:
                # Worst case the rebuild still produces a usable (empty) table rather
                # than leaving the database unusable.
                pass
        self.db.execute(f"DROP TABLE IF EXISTS {self._TEMP}")

    def add(self, **event):
        event_id = event.get("id") or str(uuid.uuid4())
        wall_ts = event.get("wall_ts")
        if wall_ts is None:
            wall_ts = time.time()
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO events(id,video,track_id,ts,wall_ts,event_type,shirt_color,"
                "identity,identity_score,action,action_score,vlm,bbox,evidence,clip)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    event.get("video"),
                    event.get("track_id"),
                    event.get("ts"),
                    float(wall_ts),
                    event.get("event_type"),
                    event.get("shirt_color"),
                    event.get("identity"),
                    event.get("identity_score"),
                    event.get("action"),
                    event.get("action_score"),
                    event.get("vlm"),
                    json.dumps(event.get("bbox")),
                    json.dumps(event.get("evidence")),
                    event.get("clip"),
                ),
            )
            self.db.commit()
        return event_id

    def search(self, color=None, start=0, end=1e12, identity=None):
        # 'video' (the recording session) is appended LAST so existing positional
        # unpacking keeps working; evidence lookup needs it to find the frame folder.
        q = (
            "SELECT id,track_id,ts,wall_ts,event_type,shirt_color,identity,identity_score,"
            "action,action_score,vlm,evidence,clip,video FROM events WHERE ts BETWEEN ? AND ?"
        )
        args = [start, end]
        if color:
            q += " AND shirt_color=?"
            args.append(color)
        if identity:
            q += " AND identity=?"
            args.append(identity)
        return self.db.execute(q + " ORDER BY ts", args).fetchall()

    def search_wall(self, start_wall, end_wall, color=None, identity=None, source=None):
        """Query by real-world time, which is what 'yesterday' or 'today' means."""
        q = (
            "SELECT id,track_id,ts,wall_ts,event_type,shirt_color,identity,identity_score,"
            "action,action_score,vlm,evidence,clip FROM events WHERE wall_ts >= ? AND wall_ts < ?"
        )
        args = [start_wall, end_wall]
        if color:
            q += " AND shirt_color=?"
            args.append(color)
        if identity:
            q += " AND identity=?"
            args.append(identity)
        if source:
            q += " AND video=?"
            args.append(source)
        return self.db.execute(q + " ORDER BY wall_ts", args).fetchall()

    def all_events(self):
        return self.db.execute(
            "SELECT id,track_id,ts,wall_ts,event_type,shirt_color,identity,identity_score,"
            "action,action_score,vlm,evidence,clip,video FROM events ORDER BY ts"
        ).fetchall()


# --------------------------------------------------------------------------------------
# Appearance
# --------------------------------------------------------------------------------------


def color_name(crop):
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    roi = hsv[int(h * 0.25) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    if roi.size == 0:
        return "unknown"
    v = float(np.median(roi[..., 2]))
    s = float(np.median(roi[..., 1]))
    if v < 65:
        return "black"
    if s < 35 and v > 180:
        return "white"
    if s < 50:
        return "gray"
    hue = float(np.median(roi[..., 0]))
    return "red" if hue < 10 or hue > 170 else "green" if hue < 85 else "blue"


# --------------------------------------------------------------------------------------
# Vector index
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------------------


class Tracker:
    """Centre-distance tracker with a max-age window."""

    def __init__(self, max_distance=90, max_age=12):
        self.next_id = 1
        self.items = {}
        self.max_distance = max_distance
        self.max_age = max_age

    def update(self, boxes):
        assigned = set()
        out = []
        for box in boxes:
            x1, y1, x2, y2 = box
            c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            best = None
            bd = self.max_distance
            for tid, item in self.items.items():
                if tid in assigned:
                    continue
                d = float(np.linalg.norm(c - item["center"]))
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.items[best] = {"center": c, "age": 0}
            self.items[best].update(center=c, age=0)
            assigned.add(best)
            out.append((best, box))
        for tid in list(self.items):
            if tid not in assigned:
                self.items[tid]["age"] += 1
                if self.items[tid]["age"] > self.max_age:
                    del self.items[tid]
        return out


# --------------------------------------------------------------------------------------
# Integrations
# --------------------------------------------------------------------------------------


class IntegrationStatus:
    """Tracks what is genuinely connected so the run report cannot lie about it."""

    def __init__(self):
        self.backends = {}

    def record(self, name, ok, detail=""):
        self.backends[name] = {"ok": bool(ok), "detail": str(detail)}
        return ok

    def as_dict(self):
        return dict(self.backends)


class OptionalPostgres:
    """Mirror of the SQLite timeline in PostgreSQL. Never blocks the pipeline.

    A write failure is recorded in the integration status rather than discarded: a bare
    ``except: return None`` made the mirror look healthy while silently losing every row.
    """

    def __init__(self, status: IntegrationStatus, dsn=None):
        self.store = None
        self.status = status
        self.first_error = None
        try:
            from services import PostgresEvents

            self.store = PostgresEvents(dsn)
            status.record("postgres", True, "connected")
        except Exception as exc:
            status.record("postgres", False, f"{type(exc).__name__}: {exc}")

    def add(self, **event):
        if self.store is None:
            return None
        try:
            return self.store.add(**event)
        except Exception as exc:
            if self.first_error is None:
                self.first_error = f"{type(exc).__name__}: {exc}"
                self.status.record("postgres", False, f"write failed - {self.first_error}")
            return None


class OptionalObjectStorage:
    """MinIO mirror of evidence frames. Never blocks the pipeline."""

    def __init__(self, status: IntegrationStatus):
        self.storage = None
        try:
            from services import ObjectStorage

            self.storage = ObjectStorage()
            status.record("minio", True, f"bucket={self.storage.bucket}")
        except Exception as exc:
            status.record("minio", False, f"{type(exc).__name__}: {exc}")

    def upload(self, path, object_name=None):
        if self.storage is None:
            return None
        try:
            return self.storage.upload(path, object_name)
        except Exception:
            return None


class OptionalVideoMAE:
    """VideoMAE action recognition. Loads the real checkpoint or reports why it cannot."""

    def __init__(self, status: IntegrationStatus, model_id=None, enabled=True):
        self.recognizer = None
        if not enabled:
            status.record("videomae", False, "disabled by config")
            return
        try:
            from ai_models import VideoMAEActionRecognizer

            self.recognizer = VideoMAEActionRecognizer(model_id) if model_id else VideoMAEActionRecognizer()
            status.record("videomae", True, f"loaded {model_id or 'MCG-NJU/videomae-base-finetuned-kinetics'}")
        except Exception as exc:
            status.record("videomae", False, f"{type(exc).__name__}: {exc}")

    def predict(self, frames, top_k=1):
        if self.recognizer is None or not frames:
            return []
        try:
            return self.recognizer.predict(frames, top_k=top_k)
        except Exception:
            return []


class OptionalVLM:
    """VLM describer built from the configured provider (ollama / openai / openrouter).

    Reports unavailability instead of returning a fake answer.
    """

    def __init__(self, status: IntegrationStatus, enabled=True):
        self.client = None
        self.info = {}
        if not enabled:
            status.record("vlm", False, "disabled by config")
            return
        try:
            from ai_models import make_vlm, vlm_status

            self.client = make_vlm()
            self.info = vlm_status()
            status.record("vlm", True, f"{self.info.get('label')} · {self.info.get('model')}")
        except Exception as exc:
            self.info = {"detail": f"{type(exc).__name__}: {exc}"}
            status.record("vlm", False, self.info["detail"])

    def describe(self, image_paths, question):
        if self.client is None or not image_paths:
            return None
        try:
            return self.client.describe(image_paths, question)
        except Exception as exc:
            return f"[vlm unavailable: {type(exc).__name__}: {exc}]"


# --------------------------------------------------------------------------------------
# Face sub-pipeline (stages 1-5)
# --------------------------------------------------------------------------------------


class FaceIdentifier:
    """YOLO11 Face detect -> align -> MiniFASNet gate -> ArcFace -> TurboVec.

    The embedder and the vector-index dimension both come from ``config`` so swapping
    models is a single environment variable rather than an edit here.
    """

    def __init__(self, status: IntegrationStatus, live_threshold: float = 0.5):
        import config

        self.detector = YOLO11FaceDetector(str(config.YOLO_FACE_MODEL))
        self.liveness = MiniFASNet(str(config.LIVENESS_PATH), threshold=live_threshold)
        self.embedder = OnnxEmbedder512(str(config.embedder_path()))
        if self.embedder.dim != 512:
            raise RuntimeError(f"Active ArcFace model must output 512-D, got {self.embedder.dim}")
        self.index = TurboVecFaceIndex(path=config.TURBOVEC_PATH, dim=self.embedder.dim)
        self.stabilizer = TrackIdentityStabilizer(self.index)
        info = config.embedder_info()
        status.record(
            "face_stack",
            True,
            f"YOLO11 Face + MiniFASNetV2 + {info['name']} ({self.embedder.dim}-D, {info['license']}) "
            f"+ TurboVec ({self.index.dim}-D, threshold={self.index.threshold:.3f})",
        )
        status.record("embedder_license", True, info["license"])
    def evaluate_all(self, frame, track_id):
        """Process EVERY face in the frame, independently.

        Each face gets its own bbox, liveness verdict, embedding, TurboVec lookup and
        identity, and each is associated with its own track. A spoof or failing face
        never prevents the remaining faces from being recognised.
        """
        faces = self.detector.detect(frame)
        if not faces:
            return []
        out = []
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = [int(round(float(v))) for v in face.bbox]
            try:
                detail = self.liveness.predict_detailed(expanded_crop(frame, face.bbox))
            except Exception as exc:
                out.append({"identity": "liveness_error", "score": 0.0, "live": False,
                             "liveness_score": 0.0, "bbox": [x1, y1, x2, y2], "face_index": i,
                             "error": str(exc)})
                continue
            if not detail["live"]:
                out.append({"identity": "spoof_rejected", "score": 0.0, "live": False,
                         "liveness_score": detail["live_score"],
                         "spoof_score": detail["spoof_score"],
                         "bbox": [x1, y1, x2, y2], "face_index": i})
                continue
            try:
                vector = self.embedder.embed(self.embedder.align(frame, face))
            except Exception as exc:
                out.append({"identity": "alignment_error", "score": 0.0, "live": True,
                         "liveness_score": detail["live_score"], "bbox": [x1, y1, x2, y2],
                         "face_index": i, "error": str(exc)})
                continue
            # a distinct sub-track per face keeps identities from being merged
            stable = self.stabilizer.update(f"{track_id}#{i}", vector, quality=float(face.confidence))
            out.append({
                "identity": stable["identity"],
                "score": stable["score"],
                "live": True,
                "liveness_score": detail["live_score"],
                "spoof_score": detail["spoof_score"],
                "bbox": [x1, y1, x2, y2],
                "face_index": i,
                "track_updates": stable["updates"],
            })
        return out


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------


def process(
    video,
    db_path,
    model_path="yolo11n.pt",
    sample_fps=3,
    evidence_dir="evidence",
    clip_dir="clips",
    use_videomae=True,
    use_vlm=True,
    clip_pad=2.0,
    vlm_every=1,
):
    YOLO = load_yolo()

    status = IntegrationStatus()
    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = Tracker()
    store = EventStore(db_path)
    video_id = Path(video).stem
    evidence = EvidenceWriter(evidence_dir)
    rules = None
    try:
        from services import IncidentRules

        rules = IncidentRules()
        status.record("incident_rules", True, f"{len(rules.rules)} rules loaded")
    except Exception as exc:
        status.record("incident_rules", False, f"{type(exc).__name__}: {exc}")

    postgres = OptionalPostgres(status)
    object_store = OptionalObjectStorage(status)
    face_identifier = FaceIdentifier(status)
    action_model = OptionalVideoMAE(status, enabled=use_videomae)
    vlm = OptionalVLM(status, enabled=use_vlm)

    Path(clip_dir).mkdir(parents=True, exist_ok=True)

    records = []
    frame_buffer = []  # recent sampled frames for VideoMAE
    frame_no = 0
    next_sample = 0
    last_centre = {}
    clip_count = 0
    vlm_count = 0
    videomae_windows = 0
    action_label = None
    action_score = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_no / fps
        frame_no += 1
        if ts < next_sample:
            continue
        next_sample = ts + 1 / sample_fps

        result = model.track(frame, persist=True, verbose=False)[0]
        boxes = []
        objects = []
        if result.boxes is not None:
            all_boxes = result.boxes.xyxy.cpu().numpy().astype(int).tolist()
            classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
            boxes = [b for b, c in zip(all_boxes, classes) if c == 0]
            objects = [(model.names[c], b) for b, c in zip(all_boxes, classes) if c != 0]

        for track, box in tracker.update(boxes):
            x1, y1, x2, y2 = box
            crop = frame[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
            color = color_name(crop)

            # ---- motion-derived action ---------------------------------------------
            prev = last_centre.get(track)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            action = "visible"
            if prev:
                dist = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
                action = "walking" if dist > 12 else "standing"
                if dist > 45:
                    action = "moving quickly"
            last_centre[track] = (cx, cy)

            near = []
            for object_name, ob in objects:
                ox, oy = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
                if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < max(100, (x2 - x1) * 0.8):
                    near.append(object_name)
            if near:
                action = f"{action} near {', '.join(sorted(set(near))[:3])}"

            # ---- face identity + liveness (stages 1-5) ----------------------------
            # Every face in the frame is processed independently; one person per event.
            face_infos = face_identifier.evaluate_all(frame, track) or [
                {"identity": "no_face", "score": 0.0, "live": False, "liveness_score": 0.0, "bbox": None, "face_index": 0}
            ]

            # ---- evidence frame ----------------------------------------------------
            evidence_path = evidence.frame(video_id, track, ts, frame, box)
            object_store.upload(evidence_path, f"{video_id}/track_{track}/{Path(evidence_path).name}")

            # ---- VideoMAE window --------------------------------------------------
            frame_buffer.append(frame)
            frame_buffer = frame_buffer[-16:]
            if len(frame_buffer) == 16 and (frame_no // max(1, int(fps // sample_fps))) % 8 == 0:
                predictions = action_model.predict(frame_buffer, top_k=1)
                if predictions:
                    videomae_windows += 1
                    action_label = predictions[0]["label"]
                    action_score = float(predictions[0]["score"])

            # ---- incident rules -> evidence clip ----------------------------------
            event_type = action_label or action

            for face_info in face_infos:
                probe = {"shirt_color": color, "event_type": event_type,
                        "identity": face_info["identity"]}
                matched_rules = rules.evaluate(probe) if rules else []
                clip_path = None
                if matched_rules:
                    try:
                        from evidence_clips import extract_clip

                        clip_name = f"{video_id}_track{track}_f{face_info.get('face_index', 0)}_{int(ts * 1000):07d}.mp4"
                        clip_path = extract_clip(
                            video, max(0.0, ts - clip_pad), ts + clip_pad, str(Path(clip_dir) / clip_name)
                        )
                        object_store.upload(clip_path, f"{video_id}/clips/{clip_name}")
                        clip_count += 1
                    except Exception as exc:
                        clip_path = f"[clip extraction failed: {type(exc).__name__}: {exc}]"

                # ---- VLM description (per face, throttled) ----------------------
                vlm_text = None
                if face_info["live"] and vlm_count < vlm_every * 3:
                    vlm_text = vlm.describe(
                        [evidence_path],
                        "Describe what this person is doing and wearing. Reference timestamps as video offsets.",
                    )
                    if vlm_text:
                        vlm_count += 1

                proof = {
                    "frame": frame_no,
                    "timestamp": ts,
                    "timestamp_hms": format_ts(ts),
                    "bbox": face_info.get("bbox") or box,
                    "person_bbox": box,
                    "face_index": face_info.get("face_index", 0),
                    "color": color,
                    "objects": near,
                    "path": evidence_path,
                    "clip": clip_path,
                    "rules": [r.get("id") for r in matched_rules],
                }

                # uuid5 keyed on the face index too, so two people in one frame
                # never collide onto a single event id
                event_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{Path(video).resolve()}|{track}|f{face_info.get('face_index', 0)}|{frame_no}|{event_type}",
                    )
                )
                store.add(
                    id=event_id, video=str(video), track_id=track, ts=ts,
                    event_type=event_type, shirt_color=color,
                    identity=face_info["identity"],
                    identity_score=float(face_info.get("score", 0.0)),
                    action=action_label, action_score=float(action_score),
                    vlm=vlm_text, bbox=face_info.get("bbox") or box,
                    evidence=proof, clip=clip_path,
                )
                postgres.add(
                    id=event_id, video=str(video), track_id=track, ts=ts,
                    event_type=event_type, shirt_color=color,
                    identity=face_info["identity"],
                    identity_score=float(face_info.get("score", 0.0)),
                    action=action_label, action_score=float(action_score),
                    vlm=vlm_text, bbox=face_info.get("bbox") or box,
                    evidence=proof, clip=clip_path,
                )
                records.append({
                    "id": event_id,
                    "track_id": track,
                    "face_index": face_info.get("face_index", 0),
                    "event": event_type,
                    "ts": ts,
                    "ts_hms": format_ts(ts),
                    "identity": face_info["identity"],
                    "identity_score": float(face_info.get("score", 0.0)),
                    "live": face_info.get("live"),
                    "liveness_score": float(face_info.get("liveness_score", 0.0)),
                    "face_bbox": face_info.get("bbox"),
                    "clip": clip_path,
                    **proof,
                })

    cap.release()
    manifest_path = evidence.manifest(video_id, records)

    report = {
        "video": str(video),
        "video_id": video_id,
        "fps": fps,
        "resolution": [width, height],
        "declared_frames": total_frames,
        "processed_frames": frame_no,
        "decoded_all_frames": total_frames == 0 or frame_no >= total_frames,
        "events": len(records),
        "evidence_clips": clip_count,
        "videomae_windows": videomae_windows,
        "vlm_descriptions": vlm_count,
        "manifest": str(manifest_path),
        "integrations": status.as_dict(),
    }
    return report, store, records


# --------------------------------------------------------------------------------------
# Question answering
# --------------------------------------------------------------------------------------

_TIME_TOKEN = r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b"

# --------------------------------------------------------------------------------------
# Arabic / Egyptian Arabic support
# --------------------------------------------------------------------------------------

ARABIC_DIGITS = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
                 "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9"}

ARABIC_RANGE_WORDS = {
    "النهارده": "today", "النهاردة": "today", "اليوم": "today", "امبارح": "yesterday",
    "البارحه": "yesterday", "من امبارح": "yesterday", "بعدين": "today",
}

# Transliterations for common Arabic name spellings -> Latin. Used only to match a query
# against already-enrolled identities; nothing stored in Qdrant/PostgreSQL is renamed.
NAME_TRANSLITERATIONS = {
    "منة": "menna", "مenna": "menna", "منه": "menna", "مينا": "menna", "مينه": "menna",
    "أحمد": "ahmed", "احمد": "ahmed", "آحمد": "ahmed", "hamid": "ahmed", "amed": "ahmed",
    "علي": "ali", "على": "ali", "عبدالله": "abdullah", "عبد الله": "abdullah",
    "محمد": "mohamed", "محمود": "mahmoud", "سارة": "sara", "نور": "nour", "نورا": "noura",
    "يوسف": "yousef", "يحيى": "youssef", "فاطمة": "fatima", "مريم": "maryam",
    "خالد": "khaled", "كريم": "kareem", "دينا": "dina", "هدى": "hoda", "هالة": "hala",
    "إيمان": "eman", "ايمان": "eman", "سلمى": "salma", "ريم": "reem", "منى": "mona",
}

ARABIC_QUESTION_WORDS = (
    "مين", "مصر", "ايه", "إيه", "امتي", "إمتى", "في", "وهو", "هي", "الفيديو",
    "بتاع", "بتاعة", "ظهر", "ظهرت", "كان", "يكون", "لحد", "الperiod", "شخص",
)


def normalise_arabic(text: str) -> str:
    """Fold Arabic-Indic digits and strip diacritics/tatweel so matching is robust."""
    out = []
    for ch in text:
        if ch in ARABIC_DIGITS:
            out.append(ARABIC_DIGITS[ch])
        elif ch in ("ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ", "ـ"):
            continue
        else:
            out.append(ch)
    return "".join(out)


def is_arabic_question(question: str) -> bool:
    q = normalise_arabic(question)
    if any("\u0600" <= c <= "\u06ff" for c in q):
        return True
    return any(w in q.lower() for w in ARABIC_QUESTION_WORDS)


def detect_language(question: str) -> str:
    return "ar" if is_arabic_question(question) else "en"


def extract_identities(question: str, known: list[str] | None = None) -> list[str]:
    """Find identity names mentioned in a question, matching Arabic and Latin spellings.

    Returns canonical names as they are stored, so no stored identity is ever renamed.
    """
    q = normalise_arabic(question)
    low = q.lower()
    found: list[str] = []

    def add(name):
        if name and name not in found:
            found.append(name)

    for ar, latin in NAME_TRANSLITERATIONS.items():
        if ar in q:
            canonical = next((name for name in (known or []) if str(name).casefold() == latin.casefold()), latin)
            add(canonical)
    for name in known or []:
        if name and name.lower() in low:
            add(name)
    return found


def _arabic_number(token: str) -> int:
    t = normalise_arabic(token)
    return int(t) if t.isdigit() else 0


def _parse_time_range_arabic(question: str) -> tuple[float, float] | None:
    """Egyptian time expressions: 'بين 2 و 3', 'الساعة 5', 'في الساعة ٢'."""
    q = normalise_arabic(question)
    if any(x in q for x in ("امتى", "إمتى", "اخر حد", "آخر حد", "فيديو", "بتعمل ايه", "بتعمل إيه", "ظهر النهارده", "ظهرت النهارده")):
        return _parse_wall_range(q)
    wall = _parse_wall_range(q)
    if wall is not None:
        return wall

    m = re.search(r"بين\s*(\d{1,2})\s*(?:و|:|إلى|الي)\s*(\d{1,2})", q)
    if m:
        a, b = _arabic_number(m.group(1)), _arabic_number(m.group(2))
        if a > b:
            a, b = b, a
        return float(a * 60), float(b * 60)

    m = re.search(r"(?:الساعه|الساعة)\s*(\d{1,2})", q)
    if m:
        s = _arabic_number(m.group(1))
        return float(s * 60), float(s * 60 + 300)
    return None



def _parse_time_range(question: str) -> tuple[float, float]:
    """Extract a video-relative time window from natural language.

    Supports ``2:30``, ``00:02:30``, ``between 1:00 and 2:00``, ``from 1:00 to 2:00``.
    Returns ``(0, +inf)`` when the question carries no time constraint.
    """
    tokens = re.findall(_TIME_TOKEN, question)
    if not tokens:
        return 0.0, float("inf")

    lowered = question.lower()
    explicit_range = bool(re.search(r"\b(between|from|until|to|-)\b", lowered))

    times = [_parse_time_token(t) for t in tokens]
    start = times[0]
    if len(times) >= 2:
        end = times[1]
    elif explicit_range:
        end = start + 300
    else:
        # A single timestamp asks about a neighbourhood of that moment.
        end = start + 300
    return min(start, end), max(start, end)


def _known_identities(store) -> list[str]:
    """Identity names actually present in the timeline, plus the vector index."""
    names: set[str] = set()
    try:
        for row in store.all_events():
            ident = row[6]
            if ident and ident not in ("unknown", "no_face", "spoof_rejected",
                                       "align_error", "alignment_error", "liveness_error"):
                names.add(str(ident))
    except Exception:
        pass
    try:
        for item in TurboVecFaceIndex(path=config.TURBOVEC_PATH, dim=_current_dim()).list_identities():
            if item.get("name"):
                names.add(str(item["name"]))
    except Exception:
        pass
    return sorted(names)


def _current_dim() -> int:
    try:
        import config as _c

        return int(_c.embedder_info()["dim"])
    except Exception:
        return 512


_AR_SENTINELS = {
    "unknown": "Unknown", "no_face": "مفيش وش", "spoof_rejected": "إنتحال مرفوض",
    "align_error": "محاذاة فاشلة", "alignment_error": "محاذاة فاشلة",
    "liveness_error": "فشل التحقق", "no data": "مفيش بيانات",
}


def _arabic_answer_text(rows, identity, window, kind: str) -> str:
    """Compose an Egyptian-Arabic answer from the rows that were actually retrieved."""
    if not rows:
        who = f"لـ {identity}" if identity else "بالفترة دي"
        return f"ملقتش أحداث مسجلة {who} في الفترة دي."

    people: dict[str, int] = {}
    actions: dict[str, int] = {}
    times = []
    for r in rows:
        ident = r[6]
        if ident and ident not in ("unknown", "no_face", "spoof_rejected", "align_error"):
            people[str(ident)] = people.get(str(ident), 0) + 1
        elif ident:
            people[_AR_SENTINELS.get(str(ident), "Unknown")] = people.get(
                _AR_SENTINELS.get(str(ident), "Unknown"), 0) + 1
        if r[8]:
            actions[r[8]] = actions.get(r[8], 0) + 1
        times.append(r[3] or 0)

    stamp = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    who_txt = " و".join(f"{k} ({v} مرات)" for k, v in sorted(people.items(), key=lambda kv: -kv[1]))
    act_txt = "، ".join(sorted(actions, key=lambda a: -actions[a])[:5]) if actions else "مفيش إجراء محدد"
    first, last = stamp(min(times)), stamp(max(times))

    if kind == "wall":
        return f"النهارده {first} لحد {last}، سجّلت {len(rows)} مشاهدة. الناس اللي ظهرت: {who_txt}. الأفعال: {act_txt}."
    if kind == "who":
        return f"النهارده طلع {who_txt}، والأفعال اللي اتسجّلت: {act_txt}."
    if kind == "clips":
        return f"أيوه، فيه {len(rows)} فيديو وحدث متسجل لـ {identity} بين {first} و {last}."
    if kind == "when":
        return f"{identity} ظهرت {len(rows)} مرة، أول مرة {first} وآخر مرة {last}."
    if kind == "with":
        return f"{identity} ظهرت {len(rows)} مرة. الناس اللي كانت معاها: {who_txt}. الأفعال: {act_txt}."
    if kind == "what":
        return f"اللي قدام الكاميرا: {act_txt}. {who_txt}."
    return f"لقيت {len(rows)} حدث مسجل، من {first} لحد {last}. الناس: {who_txt}. الأفعال: {act_txt}."


def _format_video_answer(question: str, rows, identity, lang: str, window) -> dict:
    """Shared response builder for the video-offset path, in the asker's language."""
    is_ar = lang == "ar"
    if not rows:
        msg_ar = f"ملقتش أحداث مسجلة لـ {identity} في الفترة دي." if identity else \
                 "ملقتش أحداث مسجلة في الفترة دي."
        msg_en = (
            f"I have no recorded events for {identity} in that time range."
            if identity else "I have no recorded events in that time range."
        )
        return {
            "answer": msg_ar if is_ar else msg_en,
            "evidence": [],
            "query": {"color": _parse_color(question), "mode": "video", "identity": identity, "language": lang},
            "window": {"start_hms": format_ts(window[0]), "end_hms": format_ts(window[1])},
            "total_events": 0,
        }

    # Decide what the question was actually asking.
    q = normalise_arabic(question)
    has_video_word = any(w in q for w in ("فيديو", "فديو", "كليب", "video", "clip"))
    has_with = any(w in q for w in ("مع", "with"))
    has_when = any(w in q for w in ("امتي", "امتى", "إمتى", "when"))
    has_what = any(w in q for w in ("ايه", "إيه", "ايلي", "what", "بتعمل", "يعمل"))
    has_who = any(w in q for w in ("مين", "من", "who"))

    if has_video_word:
        kind = "clips"
    elif has_with:
        kind = "with"
    elif has_when:
        kind = "when"
    elif has_what:
        kind = "what"
    elif has_who:
        kind = "who"
    else:
        kind = "all"

    evidence_items = []
    for r in rows[:200]:
        try:
            payload = r[11] if isinstance(r[11], dict) else (json.loads(r[11]) if r[11] else {})
        except Exception:
            payload = {}
        evidence_items.append({
            "event_id": str(r[0]), "ts": r[2],
            "wall_ts": r[3],
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[3] or time.time())),
            "timestamp_hms": format_ts(r[2]),
            "event_type": r[4], "identity": r[6], "identity_score": r[7],
            "shirt_color": r[5], "action": r[8], "action_score": r[9], "vlm": r[10],
            "image": payload.get("path") or payload.get("frame_path"),
            "clip": r[12], "rules": payload.get("rules", []),
        })

    if is_ar:
        answer_text = _arabic_answer_text(rows, identity or "الشخص ده", window, kind)
    else:
        names = sorted({str(r[6]) for r in rows if r[6]})
        who = ", ".join(names) if names else "no identified person"
        acts = [r[8] for r in rows if r[8]]
        acts_txt = ", ".join(sorted(set(acts))[:5]) if acts else "no action label"
        first = time.strftime("%H:%M:%S", time.localtime(min(r[3] or 0 for r in rows)))
        last = time.strftime("%H:%M:%S", time.localtime(max(r[3] or 0 for r in rows)))
        answer_text = (
            f"I found {len(rows)} recorded events between {first} and {last}. "
            f"People: {who}. Actions: {acts_txt}."
        )

    return {
        "answer": answer_text,
        "query": {"color": _parse_color(question), "mode": "video", "identity": identity, "language": lang},
        "window": {"start_hms": format_ts(window[0]), "end_hms": format_ts(window[1])},
        "total_events": len(rows),
        "evidence": evidence_items,
    }


def _parse_wall_range(question: str, now: float | None = None) -> tuple[float, float] | None:
    """Detect a real-world time window ('today', 'yesterday', 'last 2 hours', 'on 2026-09-27').

    Returns epoch seconds, or None when the question is about a video offset instead.
    """
    q = question.lower()
    now = now if now is not None else time.time()
    local = time.localtime(now)
    midnight = time.mktime((local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, 0, 0, -1))

    if "yesterday" in q or "امبارح" in q or "البارحه" in q:
        prev = time.localtime(midnight - 12 * 3600)
        prev_midnight = time.mktime((prev.tm_year, prev.tm_mon, prev.tm_mday, 0, 0, 0, 0, 0, -1))
        return prev_midnight, midnight
    if any(w in q for w in ("today", "just now", "right now", "النهارده", "النهاردة", "اليوم", "امتى", "إمتى", "اخر حد", "آخر حد", "فيديو", "بتعمل ايه", "بتعمل إيه", "ايه", "إيه", "ظهر النهارده", "ظهرت النهارده")):
        next_day = time.localtime(midnight + 36 * 3600)
        tomorrow = time.mktime((next_day.tm_year, next_day.tm_mon, next_day.tm_mday, 0, 0, 0, 0, 0, -1))
        return midnight, tomorrow
    if "this week" in q or "الأسبوع" in q or "الاسبوع" in q:
        return midnight - 6 * 86400, time.mktime((time.localtime(midnight + 36 * 3600).tm_year, time.localtime(midnight + 36 * 3600).tm_mon, time.localtime(midnight + 36 * 3600).tm_mday, 0, 0, 0, 0, 0, -1))

    m = re.search(r"(?:in the )?last (\d+)\s*(minute|min|hour|hr|day)s?", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = n * 60 if unit.startswith("min") else n * 3600 if unit.startswith("h") else n * 86400
        return now - delta, now

    m = re.search(r"\bon (\d{4}-\d{2}-\d{2})\b", q)
    if m:
        try:
            y, mo, d = (int(x) for x in m.group(1).split("-"))
            base = time.mktime((y, mo, d, 0, 0, 0, 0, 0, -1))
            nxt = time.localtime(base + 36 * 3600)
            return base, time.mktime((nxt.tm_year, nxt.tm_mon, nxt.tm_mday, 0, 0, 0, 0, 0, -1))
        except Exception:
            return None
    return None


def _parse_color(question: str):
    m = re.search(r"\b(black|white|gray|grey|red|green|blue)\b", question.lower())
    if not m:
        return None
    return "gray" if m.group(1) == "grey" else m.group(1)


def answer(question: str, db_path: str) -> dict:
    lang = detect_language(question)
    q_norm = normalise_arabic(question)
    from services import PostgresEvents
    # Production always reads the configured system of record (PostgreSQL). An explicit
    # alternate path is used by local/offline analysis and tests, so honor that path
    # instead of silently querying unrelated PostgreSQL data.
    configured_db = Path(config.event_db()).resolve()
    requested_db = Path(db_path).resolve()
    store = PostgresEvents() if requested_db == configured_db else EventStore(str(requested_db))

    # Arabic time expressions ("بين 2 و 3", "الساعة ٥") become the same windows the
    # English path already understands.
    arabic_window = _parse_time_range_arabic(q_norm) if lang == "ar" else None

    known = _known_identities(store)
    wanted = extract_identities(q_norm, known)
    identity = wanted[0] if wanted else None

    color = _parse_color(question)

    if arabic_window is not None:
        if arabic_window[1] > 1e8:
            return _answer_wall(q_norm, db_path, arabic_window[0], arabic_window[1], color)
        rows = store.search(color=color, start=arabic_window[0], end=arabic_window[1], identity=identity)
        return _format_video_answer(q_norm, rows, identity, lang, arabic_window)

    if lang == "ar" and identity is None:
        # A broad "who appeared today" style question with no name: use the real window.
        wall = _parse_wall_range(q_norm)
        if wall is not None:
            return _answer_wall(q_norm, db_path, wall[0], wall[1], color)

    start, end = _parse_time_range(question)

    # A real-world window ("yesterday", "today", "last 2 hours") takes precedence over a
    # video offset, because those questions are about when the recording happened, not
    # where in a clip it happened.
    wall = _parse_wall_range(question)
    if wall is not None:
        return _answer_wall(question, db_path, wall[0], wall[1], color)

    start, end = _parse_time_range(question)

    identity_match = re.search(r"\bwho\s+is\s+([a-z0-9_\- ]{2,40})\?", question, re.I)
    if identity is None and identity_match:
        identity = identity_match.group(1).strip().lower()
    rows = store.search(color=color, start=start, end=end, identity=identity)
    if not rows:
        return {
            "answer": "I found no evidence for that description/time range.",
            "evidence": [],
            "query": {"color": color, "start": start, "end": end, "identity": identity},
        }

    grouped: dict[int, list] = {}
    seen_events = set()
    for row in rows:
        event_id, track_id, ts, wall_ts, event_type, shirt_color, ident, ident_score, action, action_score, vlm_text, evidence_json, clip, video = row
        fingerprint = (track_id, round(float(ts), 3), event_type, action)
        if fingerprint in seen_events:
            continue
        seen_events.add(fingerprint)
        grouped.setdefault(track_id, []).append(row)

    best_track, best_rows = max(grouped.items(), key=lambda kv: len(kv[1]))
    summary = []
    evidence_items = []
    for row in best_rows[:10]:
        event_id, track_id, ts, wall_ts, event_type, shirt_color, ident, ident_score, action, action_score, vlm_text, evidence_json, clip, video = row
        summary.append(f"{event_type} at {format_ts(ts)}")
        try:
            payload = evidence_json if isinstance(evidence_json, dict) else (json.loads(evidence_json) if evidence_json else {})
        except Exception:
            payload = {}
        evidence_items.append(
            {
                "event_id": event_id,
                "ts": ts,
                "wall_ts": wall_ts,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_ts or time.time())),
                "timestamp_hms": format_ts(ts),
                "event_type": event_type,
                "identity": ident,
                "identity_score": ident_score,
                "shirt_color": shirt_color,
                "action": action,
                "action_score": action_score,
                "vlm": vlm_text,
                "frame": payload.get("frame"),
                "bbox": payload.get("bbox"),
                "objects": payload.get("objects", []),
                "image": payload.get("path"),
                "clip": clip or payload.get("clip"),
                "rules": payload.get("rules", []),
            }
        )

    identity_label = best_rows[0][5]
    _SENTINELS = {"no_face", "unknown", "spoof_rejected", "alignment_error", ""}
    if not identity_label or identity_label in _SENTINELS:
        identity_label = "an unidentified person"
    color_label = color or "the described colour"
    answer_text = (
        f"{identity_label} wearing {color_label} (track {best_track}) was observed "
        f"between {format_ts(start if start else 0.0)} and "
        f"{format_ts(end) if end != float('inf') else format_ts(best_rows[-1][2])}: "
        + ", ".join(summary)
        + "."
    )

    return {
        "answer": answer_text,
        "track_id": best_track,
        "identity": identity_label,
        "query": {"color": color, "start": start, "end": end, "identity": identity},
        "window": {"start_hms": format_ts(start), "end_hms": format_ts(end) if end != float("inf") else None},
        "evidence": evidence_items,
    }


def _answer_wall(question: str, db_path: str, start_wall: float, end_wall: float, color) -> dict:
    """Answer a question about when something happened, not where in a clip."""
    from services import PostgresEvents
    store = PostgresEvents()
    names = extract_identities(normalise_arabic(question), _known_identities(store))
    requested_identity = names[0] if names else None
    rows = store.search_wall(start_wall, end_wall, color=color, identity=requested_identity)
    label = time.strftime("%Y-%m-%d %H:%M", time.localtime(start_wall))
    span = time.strftime("%Y-%m-%d %H:%M", time.localtime(end_wall))
    window = {"start": start_wall, "end": end_wall, "start_label": label, "end_label": span}

    if not rows:
        ar = detect_language(question) == "ar"
        return {
            "answer": (
                (f"ملقتش أحداث مسجلة لمنة النهارده ({label} لحد {span})." if "menna" in question.lower() or "منة" in question else f"ملقتش أحداث مسجلة في الفترة دي ({label} لحد {span}).")
                if ar else
                f"I have no recordings for {label} to {span}. "
                "Start a Live Camera session and I will remember what happens."
            ),
            "evidence": [],
            "query": {"color": color, "mode": "wall_clock", "window": window},
            "window": window,
            "total_events": 0,
        }

    # Collapse Arabic/Latin name spellings to the canonical enrolled identity.
    people: dict[str, int] = {}
    actions: dict[str, int] = {}
    evidence_items = []
    for r in rows:
        # id,track_id,ts,wall_ts,event_type,shirt_color,identity,identity_score,action,action_score,vlm,evidence,clip
        ident = r[6]
        if ident and ident not in ("unknown", "no_face", "spoof_rejected", "alignment_error"):
            people[ident] = people.get(ident, 0) + 1
        if r[8]:
            actions[r[8]] = actions.get(r[8], 0) + 1
        try:
            payload = r[11] if isinstance(r[11], dict) else (json.loads(r[11]) if r[11] else {})
        except Exception:
            payload = {}
        evidence_items.append(
            {
                "event_id": str(r[0]),
                "ts": r[2],
                "wall_ts": r[3],
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[3] or time.time())),
                "timestamp_hms": format_ts(r[2]),
                "event_type": r[4],
                "identity": ident,
                "identity_score": r[7],
                "shirt_color": r[5],
                "action": r[8],
                "action_score": r[9],
                "vlm": r[10],
                "nearby_objects": payload.get("nearby_objects", []),
                "objects": payload.get("objects", []),
                "track_id": r[1],
                "image": payload.get("path"),
                "clip": r[12],
                "rules": payload.get("rules", []),
            }
        )

    who = ", ".join(f"{k} ({v} sightings)" for k, v in sorted(people.items(), key=lambda kv: -kv[1])) or "no enrolled person"
    did = ", ".join(f"{k} ({v})" for k, v in sorted(actions.items(), key=lambda kv: -kv[1])[:5]) or "no action label"
    first = time.strftime("%H:%M:%S", time.localtime(rows[0][3] or time.time()))
    last = time.strftime("%H:%M:%S", time.localtime(rows[-1][3] or time.time()))

    ar = detect_language(question) == "ar"
    if ar:
        ar_who = " و".join(f"{k} ({v} مرات)" for k, v in sorted(people.items(), key=lambda kv: -kv[1])) \
            or "مفيش حد متسجل"
        ar_did = "، ".join(sorted(actions, key=lambda a: -actions[a])[:5]) or "مفيش إجراء محدد"
        answer_text = (
            f"النهارده من {label} لحد {span} سجّلت {len(rows)} مشاهدة "
            f"({first} لحد {last}). الناس اللي ظهرت: {ar_who}. الأفعال: {ar_did}."
        )
        q = normalise_arabic(question)
        if "اخر حد" in q or "آخر حد" in q:
            latest = max(rows, key=lambda r: float(r[3] or 0))
            name = latest[6] or "شخص غير معروف"
            at = time.strftime("%H:%M:%S", time.localtime(latest[3] or time.time()))
            answer_text = f"آخر حد ظهر هو {name} الساعة {at}."
            evidence_items = [item for item in evidence_items if str(item["event_id"]) == str(latest[0])]
        if "فيديو" in q:
            has_clip = any(bool(r[12]) for r in rows)
            answer_text = (f"أيوه، فيه فيديو محفوظ لمنة النهارده ({len(rows)} حدث)." if has_clip else f"ملقتش فيديو محفوظ لمنة النهارده؛ لقيت {len(rows)} حدث مسجل." )
        if "بتعمل ايه" in q or "بتعمل إيه" in q or "ايه" in q or "إيه" in q:
            observed_actions = sorted({str(r[8]) for r in rows if r[8]})
            colors = sorted({str(r[5]) for r in rows if r[5]})
            objects = sorted({str(o) for r in rows for o in ((r[11] or {}).get("nearby_objects", []) if isinstance(r[11], dict) else [])})
            parts = []
            if observed_actions: parts.append("الحركة المرصودة: " + "، ".join(observed_actions))
            if colors: parts.append("لون الملابس المقدر: " + "، ".join(colors))
            if objects: parts.append("أشياء قريبة في الصورة: " + "، ".join(objects))
            answer_text = ("؛ ".join(parts) + ".") if parts else "فيه ظهور مسجل، لكن مفيش وصف حركة أو ملابس أو أشياء محفوظ أقدر أأكد منه."
    else:
        answer_text = (
            f"Between {label} and {span} I recorded {len(rows)} events ({first} to {last}). "
            f"People seen: {who}. Actions detected: {did}."
        )

    return {
        "answer": answer_text,
        "evidence": evidence_items[:200],
        "query": {"color": color, "mode": "wall_clock", "window": window},
        "window": window,
        "total_events": len(rows),
        "people": people,
        "actions": actions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--db", default="events.sqlite")
    parser.add_argument("--question")
    parser.add_argument("--no-videomae", action="store_true")
    parser.add_argument("--no-vlm", action="store_true")
    args = parser.parse_args()

    report, _store, _records = process(
        args.video,
        args.db,
        use_videomae=not args.no_videomae,
        use_vlm=not args.no_vlm,
    )
    print(json.dumps(report, indent=2))
    if args.question:
        print(json.dumps(answer(args.question, args.db), indent=2))
