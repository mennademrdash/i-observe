import config  # noqa: F401  # loads .env, sets BLAS caps, before numpy/OpenCV load

import asyncio
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Query as QueryParam, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, generate_latest
from pydantic import BaseModel

from pipeline_v2 import EventStore, answer, format_ts
from services import PostgresEvents, RedisCache
from vision_core import (
    MiniFASNet,
    OnnxEmbedder512,
    TurboVecFaceIndex,
    YOLO11FaceDetector,
    align_face,
    expanded_crop,
)

app = FastAPI(title="I-observe Video Intelligence Console")
REQUESTS = Counter("video_query_requests_total", "Video query requests")
INCIDENTS_PUBLISHED = Counter("incidents_published_total", "Incidents broadcast to websocket subscribers")
subscribers: set[WebSocket] = set()
_face_stack = None
_face_stack_lock = threading.Lock()
_scene_lock = threading.Lock()
_scene_analyzer = None
_redis_cache = None
_redis_lock = threading.Lock()
_face_inference_lock = threading.Lock()
_record_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-record")
_record_future = None
_record_result = {}
_record_error = None

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def event_db_path() -> str:
    return os.getenv("EVENT_DB", "events.sqlite")


def face_stack():
    global _face_stack
    if _face_stack is None:
        # Status, counters, and the first camera frame may arrive concurrently. Without
        # a lock each request deserializes the 260 MB ONNX weights independently; on
        # this Windows demo that exhausts memory and drops the socket (browser: Failed
        # to fetch). Build the process-wide model stack once, then share it.
        with _face_stack_lock:
            if _face_stack is None:
                import config
                embedder = OnnxEmbedder512(str(config.embedder_path()))
                if embedder.dim != 512:
                    raise RuntimeError(f"Active ArcFace model must output 512-D, got {embedder.dim}")
                _face_stack = (
                    YOLO11FaceDetector(str(config.YOLO_FACE_MODEL)),
                    embedder,
                    MiniFASNet(str(config.LIVENESS_PATH)),
                    TurboVecFaceIndex(path=config.TURBOVEC_PATH, dim=embedder.dim),
                )
    return _face_stack


def _check_auth(authorization: str | None) -> None:
    expected = os.getenv("API_TOKEN")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(401, "Invalid bearer token")



@app.on_event("startup")
async def warm_face_models():
    async def warm():
        try:
            await asyncio.to_thread(analyse_faces, np.zeros((480, 640, 3), np.uint8))
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Face model warm-up failed")
    app.state.face_warmup = asyncio.create_task(warm())


def redis_cache():
    global _redis_cache
    if _redis_cache is None:
        with _redis_lock:
            if _redis_cache is None:
                _redis_cache = RedisCache()
    return _redis_cache


class LiveSceneAnalyzer:
    """YOLO11 COCO person/object detection with persistent ByteTrack IDs."""
    def __init__(self):
        from ultralytics import YOLO
        import config
        path = config.ROOT / "models" / "yolo11n.pt"
        self.model = YOLO(str(path) if path.exists() else "yolo11n.pt")
        from pipeline_v2 import color_name
        self.color_name = color_name
        self.history = {}

    def analyze(self, frame, faces):
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False,
                                  conf=0.25, imgsz=320, classes=[0, 24, 26, 28])[0]
        people, objects = [], []
        for box in result.boxes or []:
            cls_id = int(box.cls[0].item())
            name = str(result.names.get(cls_id, cls_id))
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
            if name != "person":
                objects.append({"label": name, "bbox": [x1,y1,x2,y2]})
                continue
            tid = int(box.id[0].item()) if box.id is not None else (y1*frame.shape[1]+x1)
            crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
            shirt = self.color_name(crop[:max(1,int(crop.shape[0]*0.58))]) if crop.size else "unknown"
            center = ((x1+x2)//2,(y1+y2)//2)
            hist = self.history.setdefault(tid, [])
            prev = hist[-1] if hist else center
            hist.append(center)
            if len(hist)>12: del hist[:-12]
            movement = ((center[0]-prev[0])**2+(center[1]-prev[1])**2)**0.5
            face = next((f for f in faces if x1 <= (f["bbox"][0]+f["bbox"][2])//2 <= x2 and y1 <= (f["bbox"][1]+f["bbox"][3])//2 <= y2), None)
            people.append({"track_id":tid,"bbox":[x1,y1,x2,y2],"shirt_color":shirt,
                           "action":"walking" if movement>max(5,(y2-y1)*0.035) else "standing",
                           "action_score":0.0,"identity":face.get("identity") if face else None,
                           "identity_score":face.get("score",0.0) if face else 0.0})
        for person in people:
            x1,y1,x2,y2=person["bbox"]
            person["nearby_objects"]=[o["label"] for o in objects if
                max(0,x1-o["bbox"][2],o["bbox"][0]-x2)<max(80,(x2-x1)*0.5) and
                max(0,y1-o["bbox"][3],o["bbox"][1]-y2)<max(80,(y2-y1)*0.5)]
        return {"tracks":people,"objects":objects}


def live_scene(frame, faces):
    global _scene_analyzer
    with _scene_lock:
        if _scene_analyzer is None:
            _scene_analyzer = LiveSceneAnalyzer()
        return _scene_analyzer.analyze(frame, faces)


def associate_face_tracks(faces, tracks):
    """Attach the persistent person track id to each face inside that person box."""
    for face in faces:
        x1,y1,x2,y2=face["bbox"]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        candidates=[p for p in tracks if p["bbox"][0] <= cx <= p["bbox"][2]
                    and p["bbox"][1] <= cy <= p["bbox"][3]]
        if candidates:
            person=min(candidates,key=lambda p:(p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]))
            face["track_id"]=person["track_id"]
            face["person_bbox"]=person["bbox"]
            if person.get("identity") is None:
                person["identity"]=face.get("identity")
                person["identity_score"]=face.get("score",0.0)
    return faces


class Query(BaseModel):
    question: str


class AnalyzeRequest(BaseModel):
    source: str
    videomae: bool = True
    vlm: bool = False


# --------------------------------------------------------------------------------------
# UI + system status
# --------------------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def home():
    """The I-observe dashboard. Served from the same FastAPI app, so there is one port."""
    index = UI_DIR / "dashboard.html"
    if not index.exists():
        raise HTTPException(500, f"UI asset missing: {index}")
    return FileResponse(str(index), media_type="text/html")


def _safe_media_dir(*parts: str) -> Path:
    """Resolve a path under one of the project's media roots, refusing traversal."""
    base = ROOT.joinpath(*parts).resolve()
    if not str(base).startswith(str(ROOT.resolve())):
        raise HTTPException(400, "path outside the project")
    return base


for _name, _sub in (("evidence", "evidence"), ("clips", "clips"), ("outputs", "outputs")):
    _d = ROOT / _sub
    _d.mkdir(parents=True, exist_ok=True)
    app.mount(f"/{_name}", StaticFiles(directory=str(_d)), name=_name)


@app.get("/api/counters")
def api_counters(authorization: str | None = Header(default=None)):
    """Real counters derived from the event table and the evidence directories."""
    _check_auth(authorization)
    try:
        cached = redis_cache().client.get("iobserve:read:counters")
        if cached: return json.loads(cached)
    except Exception:
        pass
    c = {"events": 0, "face_detections": 0, "identified": 0, "unknown": 0,
         "live": 0, "spoof_rejected": 0, "evidence_images": 0, "evidence_clips": 0,
         "identities": 0, "latest_action": None, "latest_action_score": None,
         "latest_action_ts": None, "max_similarity": 0.0}
    rows = []
    postgres_rows = False
    try:
        # The configured production store is PostgreSQL; reading the legacy SQLite
        # mirror here made the dashboard crash on the live schema's event-type field.
        pg = postgres_events()
        rows = pg.all_events()
        postgres_rows = True
    except Exception:
        # Keep counters responsive when PostgreSQL is unavailable, but never let one
        # malformed historical SQLite row turn a live dashboard request into a 500.
        try:
            store = EventStore(event_db_path())
            rows = store.all_events()
        except Exception:
            rows = []
    c["events"] = len(rows)
    # PostgreSQL Events rows are ordered as returned by PostgresEvents.all_events:
    # id, track_id, ts, wall_ts, event_type, shirt_color, identity, identity_score,
    # action, action_score, vlm, evidence, clip, video.
    for r in rows:
        ident = r[6] if postgres_rows else r[7]
        if ident in (None, "no_face"):
            continue
        c["face_detections"] += 1
        if ident in ("unknown", "spoof_rejected", "alignment_error", "liveness_error", "align_error", "turbovec_error"):
            c["unknown"] += 1
        else:
            c["identified"] += 1
        # PostgreSQL all_events uses: ..., identity, identity_score, action, ...
        score = r[7] if postgres_rows else r[8]
        if score is not None:
            try:
                c["max_similarity"] = max(c["max_similarity"], float(score or 0.0))
            except (TypeError, ValueError):
                pass
    # spoof/live come from the annotated run summary, which records them per detection
    last = JOB.get("report") or {}
    summ = last.get("annotated_summary") or {}
    if summ:
        c["live"] = int(summ.get("live", 0) or 0)
        c["spoof_rejected"] = int(summ.get("spoof_rejected", 0) or 0)
        c["max_similarity"] = max(c["max_similarity"], float(summ.get("max_similarity", 0.0) or 0.0))
    for r in reversed(rows):
        action_idx, score_idx, ts_idx = (8, 9, 2) if postgres_rows else (9, 10, 3)
        if r[action_idx]:
            c["latest_action"] = r[action_idx]
            c["latest_action_score"] = r[score_idx]
            try:
                c["latest_action_ts"] = format_ts(r[ts_idx])
            except (TypeError, ValueError):
                c["latest_action_ts"] = None
            break
    c["evidence_images"] = len(glob.glob(str(ROOT / "evidence" / "**" / "*.jpg"), recursive=True))
    c["evidence_clips"] = len(glob.glob(str(ROOT / "clips" / "*.mp4")))
    try:
        c["identities"] = len(face_stack()[3].list_identities())
    except Exception:
        pass
    try: redis_cache().client.setex("iobserve:read:counters", 5, json.dumps(c))
    except Exception: pass
    return c


@app.get("/api/last-run")
def api_last_run(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    return {"report": JOB.get("report"), "error": JOB.get("error"), "running": JOB.get("running", False)}


@app.get("/api/events/{event_id}")
def api_event_detail(event_id: str, authorization: str | None = Header(default=None)):
    """Full detail for one event, including its evidence payload."""
    _check_auth(authorization)
    pg = postgres_events()
    row = pg.get_event(event_id)
    if row is None:
        raise HTTPException(404, "event not found")
    try:
        payload = row[11] if isinstance(row[11], dict) else (json.loads(row[11]) if row[11] else {})
    except Exception:
        payload = {}
    image_url = None
    if payload.get("path"):
        try:
            image_url = "/evidence/" + Path(payload["path"]).resolve().relative_to((ROOT / "evidence").resolve()).as_posix()
        except Exception:
            image_url = None
    clip_url = None
    if row[12] and not str(row[12]).startswith("["):
        try:
            clip_url = "/clips/" + Path(row[12]).resolve().name
        except Exception:
            clip_url = None
    return {
        "id": str(row[0]), "track_id": row[1], "ts": row[2], "timestamp_hms": format_ts(row[2]),
        "event_type": row[3], "shirt_color": row[4], "identity": row[5], "identity_score": row[6],
        "action": row[7], "action_score": row[8], "vlm": row[9], "bbox": row[10],
        "evidence": payload, "image_url": image_url, "clip_url": clip_url,
    }


LIVE = {"recording": False, "session": None, "started": None, "frames": 0, "events": 0, "people": {}, "last": None}
_pg_events = None
_pg_lock = threading.Lock()


def postgres_events():
    global _pg_events
    if _pg_events is None:
        with _pg_lock:
            if _pg_events is None:
                _pg_events = PostgresEvents()
    return _pg_events


@app.get("/api/live/status")
def api_live_status():
    return {
        "recording": LIVE["recording"],
        "session": LIVE["session"],
        "started": LIVE["started"],
        "frames_seen": LIVE["frames"],
        "events_recorded": LIVE["events"],
        "people": LIVE["people"],
        "last_seen": LIVE["last"],
    }


@app.post("/api/live/start")
def api_live_start(authorization: str | None = Header(default=None)):
    """Begin a live recording session. Events land in the timeline with wall-clock time."""
    _check_auth(authorization)
    session = "live_" + time.strftime("%Y%m%d_%H%M%S")
    LIVE.update({"recording": True, "session": session, "started": time.time(),
                 "frames": 0, "events": 0, "people": {}, "last": None})
    return {"recording": True, "session": session}


@app.post("/api/live/stop")
def api_live_stop(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    LIVE["recording"] = False
    return {"recording": False, "events_recorded": LIVE["events"], "people": LIVE["people"]}


@app.post("/api/live/observe")
async def api_live_observe(
    image: UploadFile = File(...),
    session: str = Form(None),
    authorization: str | None = Header(default=None),
):
    """Record one live-camera observation: detect, recognise, liveness-check, persist.

    This is what makes 'what happened yesterday' answerable - the observation is written
    with the real time it was seen, so it can be retrieved by date later.
    """
    _check_auth(authorization)
    raw = await image.read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "Image could not be decoded")
    return await asyncio.to_thread(record_live_frame, frame, session)



def record_live_frame(frame, session=None, analysis=None):
    recorded_now = []
    try:
        results, face_count, frame_size = analysis if analysis is not None else analyse_faces(frame)
    except Exception as exc:
        raise HTTPException(503, f"Face detection/recognition integration failed: {type(exc).__name__}: {exc}")
    LIVE["frames"] += 1
    if not results:
        try:
            scene = live_scene(frame, [])
        except Exception as exc:
            scene = {"tracks": [], "objects": [], "error": f"{type(exc).__name__}: {exc}"}
        if LIVE["recording"]:
            for person in scene.get("tracks", []):
                now = time.time(); sess = session or LIVE.get("session") or "live_manual"
                folder = ROOT / "evidence" / sess
                folder.mkdir(parents=True, exist_ok=True)
                shot = folder / f"track_{person['track_id']}_{int(now*1000)}.jpg"
                frame_evidence = frame.copy()
                ax1,ay1,ax2,ay2=person["bbox"]
                cv2.rectangle(frame_evidence,(ax1,ay1),(ax2,ay2),(0,210,255),2)
                cv2.putText(frame_evidence,f"#{person['track_id']} {person['shirt_color']} {person['action']}",
                            (ax1,max(18,ay1-7)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,210,255),2)
                if not cv2.imwrite(str(shot),frame_evidence):
                    raise HTTPException(500,"Could not save person evidence frame")
                ev = {"source":"live-camera","session":sess,"objects":scene.get("objects",[]),
                      "nearby_objects":person.get("nearby_objects",[]),"bbox":person["bbox"],
                      "path":str(shot.resolve()),"frame_path":str(shot.resolve()),
                      "recorded_at":time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(now))}
                event = dict(id=str(__import__("uuid").uuid4()),video=sess,track_id=person["track_id"],ts=now-(LIVE.get("started") or now),wall_ts=now,
                             event_type="person_observation",shirt_color=person["shirt_color"],identity=None,identity_score=0,
                             action=person["action"],action_score=0,vlm=None,bbox=person["bbox"],evidence=ev,clip=None)
                pg = postgres_events()
                pg.add(**event)
                if not pg.contains_event(event["id"]):
                    raise HTTPException(500, "PostgreSQL person-event read-back verification failed")
                try: redis_cache().invalidate_reads()
                except Exception: pass
                LIVE["events"] += 1
                recorded_now.append({"identity":None,"track_id":person["track_id"],"shirt_color":person["shirt_color"],
                                     "action":person["action"],"at":ev["recorded_at"],"bbox":person["bbox"]})
            LIVE["last"] = recorded_now[-1] if recorded_now else None
        return {"recorded": len(recorded_now), "face_count": 0, "recording": LIVE["recording"], "scene": scene,
                "observations": recorded_now,"events_recorded":LIVE["events"]}

    # EVERY face in the frame is processed independently: its own bbox, liveness,
    # embedding, identity and observation row. A spoof or failing face never
    # suppresses the others.
    sess = session or LIVE.get("session") or "live_manual"
    outdir = ROOT / "evidence" / sess
    outdir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    start = LIVE.get("started") or now
    stamp = time.strftime("%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
    vis = frame.copy()
    recorded_now = []
    try:
        scene = live_scene(frame, results)
    except Exception as exc:
        scene = {"tracks": [], "objects": [], "error": f"{type(exc).__name__}: {exc}"}
    associate_face_tracks(results, scene.get("tracks", []))
    face_track_ids = set()

    for det in results:
        x1, y1, x2, y2 = det["bbox"]
        identity = det.get("identity", "unknown")
        score = float(det.get("score", 0.0))
        is_live = bool(det.get("live"))
        live_score = float(det.get("liveness_score", 0.0))
        track = next((p for p in scene.get("tracks", []) if p["track_id"] == det.get("track_id")), None)
        if track:
            face_track_ids.add(track["track_id"])

        # one evidence image per person, named by face index so they never collide
        shot = outdir / f"{stamp}_f{det.get('index', 0)}.jpg"
        crop_vis = vis
        if identity == "spoof_rejected":
            color = (0, 0, 255)
        elif identity in ("unknown", "align_error", "liveness_error", "no_face"):
            color = (0, 180, 255)
        else:
            color = (0, 220, 0)
        cv2.rectangle(crop_vis, (x1, y1), (x2, y2), color, 2)
        tag = identity if is_live else "SPOOF"
        label = f"{tag} {score*100:.0f}%" if is_live else tag
        cv2.putText(crop_vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for lx, ly in det.get("landmarks", []):
            cv2.circle(crop_vis, (int(lx), int(ly)), 1, color, -1)
        cv2.imwrite(str(shot), crop_vis)

        # Throttled: a network round trip per face per frame would kill the frame rate.
        vlm_text = None
        if is_live and LIVE["events"] % 5 == 0:
            try:
                from ai_models import make_vlm

                vlm_text = make_vlm().describe(
                    [str(shot)], "In one short sentence, what is this person doing and wearing?"
                )
            except Exception:
                vlm_text = None

        ev_payload = {
            "source": "live-camera",
            "path": str(shot.resolve()),
            "frame_path": str(shot.resolve()),
            "session": sess,
            "face_index": det.get("index", 0),
            "bbox": [x1, y1, x2, y2],
            "live": is_live,
            "liveness_score": live_score,
            "detection_score": det.get("det_score", 0.0),
            "person_track_id": track.get("track_id") if track else None,
            "person_bbox": track.get("bbox") if track else None,
            "objects": scene.get("objects", []),
            "nearby_objects": track.get("nearby_objects", []) if track else [],
            "shirt_color": track.get("shirt_color") if track else None,
            "action": track.get("action") if track else None,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        }
        # track_id is per-face so several people never collapse into one track
        event = dict(
            id=str(__import__("uuid").uuid4()),
            video=sess,
            track_id=track.get("track_id", det.get("index", 0) + 1) if track else det.get("index", 0) + 1,
            ts=now - start,
            wall_ts=now,
            event_type="live_observation",
            shirt_color=track.get("shirt_color") if track else None,
            identity=identity,
            identity_score=score,
            action=track.get("action") if track else None,
            action_score=0.0,
            vlm=vlm_text,
            bbox=[x1, y1, x2, y2],
            evidence=ev_payload,
            clip=None,
        )
        # PostgreSQL is the system of record for live events. Verify each commit by
        # reading the inserted UUID back; report failure instead of silently degrading.
        pg = postgres_events()
        pg.add(**event)
        if not pg.contains_event(event["id"]):
            raise HTTPException(500, "PostgreSQL event read-back verification failed")
        try:
            redis_cache().invalidate_reads()
        except Exception:
            pass
        LIVE["events"] += 1
        if identity not in ("unknown", "spoof_rejected", "align_error", "liveness_error", "no_face"):
            LIVE["people"][identity] = LIVE["people"].get(identity, 0) + 1
        recorded_now.append({
            "identity": identity, "score": score, "live": is_live,
            "liveness_score": live_score, "bbox": [x1, y1, x2, y2],
            "index": det.get("index", 0),
            "at": ev_payload["recorded_at"], "vlm": vlm_text,
        })

    # Persist tracked people even when their faces are turned away or occluded.
    for person in scene.get("tracks", []):
        if person["track_id"] in face_track_ids:
            continue
        now = time.time()
        ev = {"source":"live-camera","session":sess,"bbox":person["bbox"],
              "objects":scene.get("objects",[]),"nearby_objects":person.get("nearby_objects",[]),
              "recorded_at":time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(now))}
        event = dict(id=str(__import__("uuid").uuid4()),video=sess,track_id=person["track_id"],
                     ts=now-start,wall_ts=now,event_type="person_observation",
                     shirt_color=person.get("shirt_color"),identity=None,identity_score=0.0,
                     action=person.get("action"),action_score=0.0,vlm=None,
                     bbox=person["bbox"],evidence=ev,clip=None)
        pg = postgres_events()
        pg.add(**event)
        if not pg.contains_event(event["id"]):
            raise HTTPException(500,"PostgreSQL person-event read-back verification failed")
        LIVE["events"] += 1
        try: redis_cache().invalidate_reads()
        except Exception: pass

    LIVE["last"] = recorded_now[0] if recorded_now else None
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
    import base64

    return {
        "recorded": len(recorded_now),
        "session": sess,
        "recording": LIVE["recording"],
        "face_count": face_count,
        "last": LIVE["last"],
        "observations": recorded_now,
        "detections": results,
        "events_recorded": LIVE["events"],
        "frame_size": list(frame_size),
        "scene": scene,
        "annotated": ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()) if ok else None,
    }


MAX_DETECT_WIDTH = 640
# YuNet's box is a tight detector box; it commonly clips the forehead and chin, which
# makes a correctly-scaled box still look "too small" around the face. Padding is
# proportional to the face's own width/height so it scales with distance.
PAD_X = 0.10   # 10% each side horizontally
PAD_Y = 0.12   # 12% each side vertically


def analyse_faces(frame):
    with _face_inference_lock:
        return _analyse_faces(frame)



def _analyse_faces(frame):
    """Run the full face stack on EVERY face YuNet found in one frame.

    Returns (results, detections_used, original_size). Each result is independent: its own
    bbox, landmarks, liveness verdict, embedding and identity. One failing face never
    prevents the others from being processed.

    Bounding boxes are returned in ORIGINAL frame coordinates, not the downscaled
    inference coordinates, so the browser can map them onto the untouched camera frame.
    """
    orig_h, orig_w = frame.shape[:2]
    work = frame
    scale = 1.0
    if orig_w > MAX_DETECT_WIDTH:
        scale = MAX_DETECT_WIDTH / float(orig_w)
        work = cv2.resize(
            frame, (MAX_DETECT_WIDTH, max(1, int(orig_h * scale))), interpolation=cv2.INTER_AREA
        )
    detector, embedder, liveness, index = face_stack()

    faces = detector.detect(work)
    results = []
    for i, face in enumerate(faces):
        # Pad proportionally so the box frames the whole face, then clamp to the image.
        bx1, by1, bx2, by2 = (float(v) for v in face.bbox)
        bw, bh = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
        ex1, ey1 = bx1 - bw * PAD_X, by1 - bh * PAD_Y
        ex2, ey2 = bx2 + bw * PAD_X, by2 + bh * PAD_Y
        ex1 = max(0.0, ex1); ey1 = max(0.0, ey1)
        ex2 = min(float(work.shape[1]), ex2); ey2 = min(float(work.shape[0]), ey2)
        # back to original resolution
        bx1, by1 = ex1 / scale, ey1 / scale
        bx2, by2 = ex2 / scale, ey2 / scale
        bbox = [
            int(round(max(0, min(bx1, orig_w - 1)))),
            int(round(max(0, min(by1, orig_h - 1)))),
            int(round(max(bx1 + 1, min(bx2, orig_w)))),
            int(round(max(by1 + 1, min(by2, orig_h)))),
        ]

        landmarks_valid = (
            np.asarray(face.landmarks).shape == (5, 2)
            and np.isfinite(face.landmarks).all()
            and not np.allclose(face.landmarks, 0.0)
        )
        entry = {
            "index": i,
            "bbox": bbox,
            "det_score": float(face.confidence),
            "landmarks": ([[float(x) / scale, float(y) / scale] for x, y in face.landmarks] if landmarks_valid else []),
        }
        if not landmarks_valid:
            entry.update({"identity": "align_error", "score": 0.0, "live": False,
                          "liveness_score": 0.0, "error": "YOLO11 did not return five valid face landmarks"})
            results.append(entry)
            continue
        # Liveness and identity are independent signals. MiniFASNet currently has a
        # known false-rejection problem on real video frames, so still calculate a
        # face-match candidate when it says spoof. Keep live=False and expose the
        # warning to the UI/events; this is identification only, never an access grant.
        live_detail = None
        liveness_error = None
        try:
            live_detail = liveness.predict_detailed(expanded_crop(work, face.bbox))
        except Exception as exc:
            liveness_error = f"{type(exc).__name__}: {exc}"
        try:
            res = index.identify(embedder.embed(embedder.align(work, face)))
        except Exception as exc:
            res = {"identity": "turbovec_error", "score": 0.0, "error": str(exc)}
        is_live = bool(live_detail and live_detail.get("live"))
        identity = res.get("identity", "unknown")
        if not is_live and identity == "unknown" and liveness_error is None:
            identity = "spoof_rejected"
        entry.update({
            "live": is_live,
            "liveness_score": float(live_detail.get("live_score", 0.0)) if live_detail else 0.0,
            "spoof_score": float(live_detail.get("spoof_score", 0.0)) if live_detail else 0.0,
            "liveness_error": liveness_error,
            "identity": identity if not liveness_error or identity not in ("unknown", "turbovec_error") else "liveness_error",
            "score": float(res.get("score", 0.0)),
        })
        results.append(entry)
    return results, len(faces), (orig_w, orig_h)


@app.post("/api/camera/detect")
async def api_camera_detect(
    image: UploadFile = File(...),
    annotate: bool = Form(True),
    authorization: str | None = Header(default=None),
):
    """Detect/recognise/liveness-check one frame.

    With ``annotate=false`` only the detection results are returned (a few hundred bytes)
    and the browser draws the boxes over its own local video. That is the fast path: the
    annotated-JPEG path was sending ~226 KB of base64 per frame and dominated the
    round-trip latency.
    """
    _check_auth(authorization)
    raw = await image.read()
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "Image could not be decoded")
    started = time.perf_counter()
    results, face_count, frame_size = await asyncio.to_thread(analyse_faces, frame)
    # One background observation at a time: no stale-frame queue, duplicate face
    # inference, or database/VLM latency on the camera's response path.
    global _record_future, _record_result, _record_error
    if _record_future is not None and _record_future.done():
        try:
            _record_result = _record_future.result()
            _record_error = None
        except Exception as exc:
            _record_error = f"{type(exc).__name__}: {exc}"
        _record_future = None
    if LIVE["recording"] and _record_future is None:
        import copy
        _record_future = _record_executor.submit(
            record_live_frame, frame.copy(), LIVE.get("session"),
            (copy.deepcopy(results), face_count, frame_size),
        )
    payload = {"detections": results, "face_count": face_count, "frame_size": list(frame_size),
               "inference_ms": round((time.perf_counter()-started)*1000, 1),
               "events_recorded": LIVE["events"], "session": LIVE.get("session"),
               "last": _record_result.get("last"),
               "observations": _record_result.get("observations", []),
               "recording_error": _record_error}
    if not annotate:
        return payload
    frame_h, frame_w = frame.shape[:2]
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        ident = r.get("identity", "unknown")
        if ident == "spoof_rejected":
            color = (0, 0, 255)
        elif ident in ("unknown", "align_error", "liveness_error", "no_face"):
            color = (0, 180, 255)
        else:
            color = (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = ident if r.get("live") else (f"UNVERIFIED: {ident}" if r.get("liveness_error") else f"NOT LIVE: {ident}")
        line1 = f"{tag} {r.get('score', 0.0)*100:.0f}%" if r.get("live") else tag
        line2 = f"live={r.get('liveness_score', 0.0)*100:.0f}% det={r.get('det_score', 0.0)*100:.0f}%"
        cv2.putText(frame, line1, (x1, max(20, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, line2, (x1, max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for lx, ly in r.get("landmarks", []):
            cv2.circle(frame, (int(lx), int(ly)), 1, color, -1)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise HTTPException(500, "failed to encode annotated frame")
    import base64

    payload["annotated"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    return payload


@app.get("/api/status")
def system_status():
    """Real health of every backend the UI displays. Never fabricates a green light."""
    out = {
        "api": {"ok": True, "detail": f"serving on {Path(__file__).parent.name}"},
        "postgres": {"ok": False, "detail": "not checked"},
        "redis": {"ok": False, "detail": "not checked"},
        "face_index": {"ok": False, "detail": "not checked"},
        "storage": {"ok": False, "detail": "not checked"},
        "face": {"ok": False, "detail": "not checked"},
        "liveness": {"ok": False, "detail": "Not validated on representative live/spoof data"},
        "videomae": {"ok": False, "detail": "not loaded"},
        "vlm": {"ok": False, "detail": "not configured"},
        "config": {},
        "identities": [],
        "counts": {"events": 0, "clips": 0, "images": 0},
        "last_run": JOB.get("report"),
        "job_running": JOB.get("running", False),
    }

    try:
        out["config"] = config.describe()
    except Exception as exc:
        out["config"] = {"error": f"{type(exc).__name__}: {exc}"}

    # PostgreSQL
    try:
        from services import PostgresEvents

        pg = PostgresEvents()
        try:
            n = len(pg.search(start=0, end=1e12))
        finally:
            pg.close()
        out["postgres"] = {"ok": True, "detail": f"connected Â· {n} events"}
    except Exception as exc:
        out["postgres"] = {"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}

    try:
        redis_cache().client.ping()
        out["redis"] = {"ok": True, "detail": "read cache online"}
    except Exception as exc:
        out["redis"] = {"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}

    # TurboVec local ArcFace index + enrolled identities
    try:
        index = face_stack()[3]
        out["face_index"] = {"ok": True, "detail": f"TurboVec cosine · {len(index.client)} vectors · {index.dim}-D"}
        ident = index.list_identities()
        out["identities"] = ident
    except Exception as exc:
        out["face_index"] = {"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}

    # Evidence object storage
    try:
        from services import ObjectStorage

        o = ObjectStorage()
        n = sum(1 for _ in o.client.list_objects(o.bucket, recursive=True))
        out["storage"] = {"ok": True, "detail": f"s3://{o.bucket} Â· {n} objects"}
    except Exception as exc:
        out["storage"] = {"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}

    # Face stack
    try:
        detector, embedder, liveness, index = face_stack()
        info = config.embedder_info()
        out["face"] = {
            "ok": True,
            "detail": f"YOLO11 Face + MiniFASNetV2 + ArcFace {embedder.dim}-D + TurboVec (threshold={index.threshold:.3f})",
        }
        # A local benchmark found live-class false accepts on all 6 sampled cutout spoofs.
        # Keep the model available for experimentation but never present it as a verified gate.
        out["liveness"] = {
            "ok": False,
            "detail": "MiniFASNet loaded; failed local check: 6/6 sampled cutout spoofs scored live. Not a security gate.",
        }
    except Exception as exc:
        out["face"] = {"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}

    # VideoMAE checkpoint presence
    try:
        model_id = "MCG-NJU/videomae-base-finetuned-kinetics"
        cache = Path(os.environ.get("HF_HOME", str(ROOT / ".hf-cache")))
        hit = list(cache.glob("models--MCG-NJU--videomae*")) if cache.exists() else []
        out["videomae"] = (
            {"ok": True, "detail": f"{model_id} cached in {cache.name}"}
            if hit
            else {"ok": False, "detail": "checkpoint not downloaded (loads on first analysis run)"}
        )
    except Exception as exc:
        out["videomae"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}

    # VLM
    try:
        from ai_models import vlm_status

        out["vlm"] = vlm_status()
    except Exception as exc:
        out["vlm"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}

    try:
        store = EventStore(event_db_path())
        out["counts"]["events"] = len(store.all_events())
    except Exception:
        pass
    out["counts"]["clips"] = len(glob.glob(str(ROOT / "clips" / "*.mp4")))
    out["counts"]["images"] = len(glob.glob(str(ROOT / "evidence" / "**" / "*.jpg"), recursive=True))

    # VideoMAE
    out["videomae_last"] = (JOB.get("report") or {}).get("videomae_windows", 0)
    return out


@app.get("/api/live/recordings")
def api_live_recordings(authorization: str | None = Header(default=None)):
    """Every recorded live-camera session with its evidence frame count and span."""
    _check_auth(authorization)
    base = ROOT / "evidence"
    if not base.is_dir():
        return {"recordings": []}
    out = []
    for folder in sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("live_")]):
        shots = sorted(folder.rglob("*.jpg"))
        if not shots:
            continue
        first, last = shots[0].stat().st_mtime, shots[-1].stat().st_mtime
        out.append(
            {
                "session": folder.name,
                "frames": len(shots),
                "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first)),
                "ended": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)),
                "duration_s": round(last - first, 1),
                "preview": "/evidence/" + shots[-1].relative_to(base).as_posix(),
                "recording_now": LIVE.get("session") == folder.name and LIVE.get("recording"),
            }
        )
    out.sort(key=lambda r: r["started"], reverse=True)
    return {"recordings": out}


class VlmTestRequest(BaseModel):
    language: str = ""


@app.post("/api/vlm/test")
def api_vlm_test(body: VlmTestRequest | None = None, authorization: str | None = Header(default=None)):
    """One minimal real VLM request, so 'READY' means a live response, not a guess.

    The reply language is a per-request display choice, not server configuration.
    """
    _check_auth(authorization)
    from ai_models import vlm_status, vlm_smoke_test

    st = vlm_status()
    if st.get("key_env") and not st.get("key_present"):
        # Missing key is a configuration problem, not a server fault: report it as such
        # so the UI can say exactly what to do instead of showing a stack trace.
        return {
            "ok": False,
            "provider": st.get("provider"),
            "model": st.get("model"),
            "error": (
                f"{st['key_env']} is not set in the server process environment. "
                "Set it in the terminal that runs start_iobserve.ps1, then restart the server."
            ),
            "needs_key": True,
            "key_env": st.get("key_env"),
        }

    # Use an existing evidence frame when one is available, else the first enrolled photo.
    shots = sorted(glob.glob(str(ROOT / "evidence" / "**" / "*.jpg"), recursive=True))
    if not shots:
        shots = sorted(glob.glob(str(ROOT / "data" / "people" / "*" / "*.*")))
    if not shots:
        raise HTTPException(422, "no image available for the VLM test")
    language = (body.language if body else "") or os.getenv("VLM_LANGUAGE") or None
    return vlm_smoke_test(shots[0], language=language)


def _evidence_url(image) -> str | None:
    """Map an on-disk evidence path to its browser URL under the /evidence mount."""
    if not image:
        return None
    try:
        rel = Path(image).resolve().relative_to((ROOT / "evidence").resolve())
        return "/evidence/" + rel.as_posix()
    except Exception:
        return None


def _latest_frame_in_session(session: str) -> str | None:
    """Newest evidence frame for a recording session, used when an event predates
    the payload that stored its own path."""
    folder = ROOT / "evidence" / session
    if not folder.is_dir():
        return None
    shots = sorted(folder.rglob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return _evidence_url(str(shots[0])) if shots else None


@app.get("/api/events")
def api_events(
    limit: int = QueryParam(100, le=1000),
    color: str | None = QueryParam(default=None),
    identity: str | None = QueryParam(default=None),
    start: float = QueryParam(0.0),
    end: float = QueryParam(default=1e12),
    date: str | None = QueryParam(default=None),
    authorization: str | None = Header(default=None),
):
    """Recent events, newest first, with ready-to-open evidence URLs."""
    _check_auth(authorization)
    cache_key = "iobserve:read:events:" + json.dumps([limit,color,identity,start,end,date], separators=(",",":"))
    try:
        cached = redis_cache().client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    if date:
        try:
            day = time.strptime(date, "%Y-%m-%d")
            start = time.mktime((day.tm_year, day.tm_mon, day.tm_mday, 0, 0, 0, 0, 0, -1))
            next_day = time.localtime(start + 36 * 3600)
            end = time.mktime((next_day.tm_year, next_day.tm_mon, next_day.tm_mday, 0, 0, 0, 0, 0, -1))
        except ValueError:
            raise HTTPException(422, "date must be YYYY-MM-DD")
    pg = postgres_events()
    rows = pg.search_wall(start, end, color=color, identity=identity) if date else pg.search(color=color, start=start, end=end, identity=identity)
    out = []
    selected_rows = rows[::-1] if date else rows[-limit:][::-1]
    for row in selected_rows:
        event_id, track_id, ts, wall_ts, event_type, shirt_color, ident, ident_score, action, action_score, vlm_text, evidence_json, clip, video = row
        try:
            payload = evidence_json if isinstance(evidence_json, dict) else (json.loads(evidence_json) if evidence_json else {})
        except Exception:
            payload = {}
        # The recording session is the trailing 'video' column; live sessions name their
        # evidence folder, which is how an event is matched back to a frame on disk.
        session = video or payload.get("session")
        source = payload.get("source") or ("live-camera" if (session or "").startswith("live_") else None)
        image = payload.get("path") or payload.get("frame_path")
        image_url = _evidence_url(image)
        # Older live events were written before 'path' was stored. Recover the frame from
        # the session folder so evidence already on disk is never lost.
        if not image_url and session:
            image_url = _latest_frame_in_session(session)
        clip_url = None
        if clip and not str(clip).startswith("["):
            try:
                clip_url = "/clips/" + Path(clip).resolve().relative_to(ROOT.resolve()).as_posix()
            except Exception:
                clip_url = None
        out.append(
            {
                "id": str(event_id),
                "track_id": track_id,
                "ts": ts,
                "wall_ts": wall_ts,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_ts or time.time())),
                "timestamp_hms": format_ts(ts),
                "event_type": event_type,
                "shirt_color": shirt_color,
                "identity": ident,
                "identity_score": ident_score,
                "action": action,
                "action_score": action_score,
                "vlm": vlm_text,
                "source": source or "video",
                "session": session,
                "objects": payload.get("objects", []),
                "rules": payload.get("rules", []),
                "image_url": image_url,
                "clip_url": clip_url,
            }
        )
    payload = {"count": len(out), "events": out}
    try:
        redis_cache().client.setex(cache_key, 5, json.dumps(payload))
    except Exception:
        pass
    return payload


@app.get("/api/evidence")
def api_evidence(authorization: str | None = Header(default=None)):
    """Evidence images and incident clips discovered on disk."""
    _check_auth(authorization)
    images = []
    for p in sorted(glob.glob(str(ROOT / "evidence" / "**" / "*.jpg"), recursive=True))[:400]:
        images.append(
            {
                "url": "/evidence/" + Path(p).relative_to(ROOT).as_posix(),
                "name": Path(p).name,
            }
        )
    clips = [{"url": "/clips/" + Path(p).name, "name": Path(p).name} for p in sorted(glob.glob(str(ROOT / "clips" / "*.mp4")))]
    return {"count": len(images) + len(clips), "images": images, "clips": clips}


# --------------------------------------------------------------------------------------
# Background video analysis
# --------------------------------------------------------------------------------------

JOB = {"running": False, "report": None, "error": None, "started": None, "finished": None}


def _run_analysis(source: str, use_videomae: bool, use_vlm: bool):
    import pipeline_v2

    try:
        # Annotate an output video so the UI can play the result back.
        out_dir = ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem or "clip"
        annotated = out_dir / f"{stem}_annotated.mp4"

        if use_videomae:
            report, _store, records = pipeline_v2.process(
                source,
                event_db_path(),
                evidence_dir=str(ROOT / "evidence"),
                clip_dir=str(ROOT / "clips"),
                use_videomae=True,
                use_vlm=use_vlm,
            )
        else:
            report, _store, records = pipeline_v2.process(
                source,
                event_db_path(),
                evidence_dir=str(ROOT / "evidence"),
                clip_dir=str(ROOT / "clips"),
                use_videomae=False,
                use_vlm=False,
            )

        # Re-run the face pass to produce the annotated video via the existing function.
        try:
            import analyze_face_video

            summary = analyze_face_video.run(source, str(annotated), sample_every=6)
            report["annotated"] = "/outputs/" + annotated.name
            report["annotated_summary"] = {k: v for k, v in summary.items() if k != "events"}
            dets = []
            for ev in summary.get("events", [])[:300]:
                dets.append(
                    {
                        "time": ev.get("time"),
                        "identity": ev.get("identity"),
                        "similarity": ev.get("similarity"),
                        "live": ev.get("live"),
                        "liveness_score": ev.get("liveness_score"),
                        "action": next((r.get("event") for r in records if abs(r.get("ts", -1) - ev.get("time", -2)) < 0.2), None),
                        "evidence_url": None,
                    }
                )
            report["detections"] = dets
            report["identities"] = summary.get("identities", [])
            report["identified_faces"] = summary.get("identified_faces", 0)
        except Exception as exc:
            report["annotated_error"] = f"{type(exc).__name__}: {exc}"

        JOB["report"] = report
    except Exception as exc:
        JOB["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        JOB["running"] = False
        JOB["finished"] = time.time()


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    if JOB.get("running"):
        raise HTTPException(409, "an analysis is already running")
    src = req.source
    if not (ROOT / src).exists():
        cand = UPLOAD_DIR / src
        if cand.exists():
            src = str(cand)
        elif Path(src).exists():
            pass
        else:
            raise HTTPException(404, f"video not found: {src}")
    JOB.update({"running": True, "report": None, "error": None, "started": time.time()})
    threading.Thread(target=_run_analysis, args=(src, req.videomae, req.vlm), daemon=True).start()
    return {"started": True, "source": src}


@app.get("/api/analyze/status")
def api_analyze_status(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    return {
        "running": JOB.get("running", False),
        "report": JOB.get("report"),
        "error": JOB.get("error"),
        "started": JOB.get("started"),
        "finished": JOB.get("finished"),
    }


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    name = Path(file.filename or "upload.mp4").name
    dest = UPLOAD_DIR / name
    dest.write_bytes(await file.read())
    return {"stored_as": name, "size": dest.stat().st_size}


@app.post("/api/identify-image")
async def api_identify_image(image: UploadFile = File(...), authorization: str | None = Header(default=None)):
    """Run the existing face stack on a browser camera frame."""
    _check_auth(authorization)
    frame = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "Image could not be decoded")
    detections, face_count, frame_size = analyse_faces(frame)
    return {"face_count": face_count, "frame_size": list(frame_size), "detections": detections}


@app.post("/query")
def query(q: Query, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    REQUESTS.inc()
    result = answer(q.question, event_db_path())
    for item in result.get("evidence", []):
        item["image_url"] = _evidence_url(item.get("image"))
        clip = item.get("clip")
        if clip and not str(clip).startswith("["):
            try:
                item["clip_url"] = "/clips/" + Path(clip).resolve().relative_to(ROOT.resolve()).as_posix()
            except (ValueError, OSError):
                item["clip_url"] = None
    return result


@app.get("/events")
def list_events(
    start: float = QueryParam(0.0),
    end: float = QueryParam(default=1e12),
    color: str | None = QueryParam(default=None),
    identity: str | None = QueryParam(default=None),
    authorization: str | None = Header(default=None),
):
    """Read the event timeline with per-event evidence paths and clips."""
    _check_auth(authorization)
    store = EventStore(event_db_path())
    rows = store.search(color=color, start=start, end=end, identity=identity)
    out = []
    for row in rows:
        event_id, track_id, ts, wall_ts, event_type, shirt_color, ident, ident_score, action, action_score, vlm_text, evidence_json, clip = row
        try:
            payload = evidence_json if isinstance(evidence_json, dict) else (json.loads(evidence_json) if evidence_json else {})
        except Exception:
            payload = {}
        out.append(
            {
                "id": event_id,
                "track_id": track_id,
                "ts": ts,
                "wall_ts": wall_ts,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_ts or time.time())),
                "timestamp_hms": format_ts(ts),
                "event_type": event_type,
                "shirt_color": shirt_color,
                "identity": ident,
                "identity_score": ident_score,
                "action": action,
                "action_score": action_score,
                "vlm": vlm_text,
                "evidence": payload,
                "clip": clip or payload.get("clip"),
            }
        )
    return {"count": len(out), "events": out}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")


@app.post("/enroll")
async def enroll(
    identity: str = Form(...),
    images: list[UploadFile] = File(...),
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)
    detector, embedder, _, index = face_stack()
    enrolled = {str(x.get("name", "")).casefold(): str(x.get("name")) for x in index.list_identities()}
    identity = enrolled.get(identity.strip().casefold(), identity.strip())
    vectors = []
    rejected = []
    for image in images:
        frame = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
        faces = detector.detect(frame) if frame is not None else []
        if len(faces) != 1:
            rejected.append({"file": image.filename, "faces": len(faces), "reason": "each enrollment image must contain exactly one face"})
            continue
        try:
            for face in faces:
                if np.asarray(face.landmarks).shape != (5, 2) or not np.isfinite(face.landmarks).all() or np.allclose(face.landmarks, 0.0):
                    raise ValueError("YOLO11 did not return five valid face landmarks")
                vectors.append(embedder.embed(embedder.align(frame, face)))
        except Exception as exc:
            rejected.append({"file": image.filename, "error": str(exc)})
    if not vectors:
        raise HTTPException(422, detail={"error": "no_usable_images", "rejected": rejected})
    # Keep one embedding per accepted view. Averaging pose/lighting variants can pull a
    # valid query away from every enrolled sample and hide genuine within-person spread.
    point_ids = [index.enroll(identity, vector) for vector in vectors]
    return {"identity": identity, "point_id": point_ids[0], "point_ids": point_ids,
            "dimension": int(vectors[0].size), "images_used": len(vectors), "rejected": rejected}


@app.post("/identify")
async def identify(image: UploadFile = File(...)):
    frame = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "Image could not be decoded")
    results, count, size = analyse_faces(frame)
    if count == 0:
        raise HTTPException(422, "No face detected")
    return {"face_count": count, "frame_size": list(size), "detections": results}


@app.post("/events/publish")
async def publish_event(
    event: dict,
    authorization: str | None = Header(default=None),
):
    """Ingest an incident from the video loop and broadcast it to live subscribers."""
    _check_auth(authorization)
    if not isinstance(event, dict):
        raise HTTPException(422, "event must be an object")
    store = EventStore(event_db_path())
    event_id = store.add(
        video=event.get("video"),
        track_id=int(event.get("track_id", 0)),
        ts=float(event.get("ts", 0.0)),
        event_type=event.get("event_type"),
        shirt_color=event.get("shirt_color"),
        identity=event.get("identity"),
        identity_score=float(event.get("identity_score", 0.0)),
        action=event.get("action"),
        action_score=float(event.get("action_score", 0.0)),
        vlm=event.get("vlm"),
        bbox=event.get("bbox"),
        evidence=event.get("evidence", {}),
        clip=event.get("clip"),
    )
    payload = dict(event)
    payload["id"] = event_id
    delivered = await publish_incident(payload)
    return {"id": event_id, "broadcast_to": delivered}


@app.websocket("/ws/incidents")
async def incidents(ws: WebSocket):
    token = ws.query_params.get("token")
    expected = os.getenv("API_TOKEN")
    if expected and token != expected:
        await ws.close(code=4401)
        return
    await ws.accept()
    subscribers.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        subscribers.discard(ws)
    except Exception:
        subscribers.discard(ws)


async def publish_incident(event: dict) -> int:
    """Push an incident to every connected websocket. Returns how many clients got it."""
    if not subscribers:
        return 0
    dead = []
    delivered = 0
    for ws in list(subscribers):
        try:
            await ws.send_json(event)
            delivered += 1
        except Exception:
            dead.append(ws)
    for ws in dead:
        subscribers.discard(ws)
    INCIDENTS_PUBLISHED.inc()
    return delivered
