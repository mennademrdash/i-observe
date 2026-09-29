import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: F401,E402
import pipeline_v2

report, store, records = pipeline_v2.process(
    "data/sample_action.mp4",
    "tools/_probe.sqlite",
    sample_fps=3,
    evidence_dir="tools/_probe_evidence",
    clip_dir="tools/_probe_clips",
    use_videomae=False,
    use_vlm=False,
)
print("events:", report["events"])
print("postgres:", report["integrations"]["postgres"])
print("minio   :", report["integrations"]["minio"])
