import sys
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config  # noqa: F401,E402
from services import PostgresEvents, event_values

# Mirror the exact event shape pipeline_v2 sends, including numpy-typed bbox entries
# that come out of the tracker.
box_np = np.array([1, 2, 3, 4], dtype=np.int32)
proof = {
    "frame": 214,
    "timestamp": 7.1,
    "timestamp_hms": "00:00:07.100",
    "bbox": [int(v) for v in box_np],
    "color": "black",
    "objects": [],
    "path": "E:/x/y.jpg",
    "clip": None,
    "rules": [],
}

event_id = str(uuid.uuid4())
payload = dict(
    id=event_id,
    video="data/sample_action.mp4",
    track_id=3,
    ts=7.1,
    event_type="moving quickly",
    shirt_color="black",
    identity="no_face",
    identity_score=0.0,
    action=None,
    action_score=0.0,
    vlm=None,
    bbox=box_np,          # deliberately numpy, as the tracker produces
    evidence=proof,
    clip=None,
)

pg = PostgresEvents()
print("event_values arity:", len(event_values(event_id, payload)), "vs EVENT_COLUMNS:", len(__import__("services").EVENT_COLUMNS))
for label, ev in (("numpy bbox", payload), ("list bbox", {**payload, "bbox": [1, 2, 3, 4]})):
    try:
        got = pg.add(**ev)
        print(f"{label}: OK -> {got}")
    except Exception as exc:
        print(f"{label}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
pg.close()
