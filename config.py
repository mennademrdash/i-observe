"""Central model and integration configuration.

Model paths were previously hard-coded in four different modules, which made it
impossible to swap the embedder without editing every call site. Everything reads from
here instead.

Embedder choice
---------------
``arcface_glint360k``
    A 260 MB Glint360K ResNet100 export. 512-D, more accurate on hard poses, but the
    weights fall under InsightFace's non-commercial research licence and the model needs
    roughly 300 MB of free RAM simply to deserialise.

Both share the canonical ArcFace 5-point alignment, so ``vision_core.align_face`` is
identical either way. The vector index dimension follows the embedder automatically.
"""
from __future__ import annotations

# Must run before numpy/OpenBLAS are imported anywhere in the process. OpenBLAS sizes a
# per-thread buffer pool at import time and, on a machine with little free RAM, that
# allocation fails outright ("OpenBLAS error: Memory allocation still failed after 10
# retries") and takes `import numpy` down with it. Capping the thread count keeps the
# pool small enough to fit. Set to 0 to let BLAS decide for itself.
import os
import re

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, os.getenv("BLAS_NUM_THREADS", "1"))

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_env(path: Path = ROOT / ".env") -> None:
    """Populate os.environ from the project's secret file without overwriting real env vars.

    Secrets live in ``.env`` (gitignored) and are read at runtime. No credential is ever
    duplicated into source code, and values already present in the process environment win
    so a real deployment can inject them without touching the file.
    """
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env()

# Hugging Face defaults to the user profile on C:, which on this machine has under 100 MB
# free. The VideoMAE checkpoint is 346 MB, so the download dies with "There is not enough
# space on the disk" even though the workspace drive has hundreds of GB free. Keep the
# model cache next to the project unless the caller has already chosen a location.
for _var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
    if not os.environ.get(_var):
        os.environ[_var] = str(ROOT / ".hf-cache")

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
EVIDENCE_DIR = ROOT / "evidence"
CLIP_DIR = ROOT / "clips"

YOLO_FACE_MODEL = Path(os.getenv("YOLO_FACE_MODEL", str(MODEL_DIR / "yolo11n-pose_widerface.pt")))
LIVENESS_PATH = MODEL_DIR / "minifasnet_v2.onnx"
TURBOVEC_PATH = Path(os.getenv("TURBOVEC_PATH", str(DATA_DIR / "face_index_512.tvim")))
ARCFACE_MATCH_THRESHOLD = float(os.getenv("ARCFACE_MATCH_THRESHOLD", "0.25"))

_NORMALIZATIONS = ("raw_bgr", "raw_rgb", "half_bgr", "half_rgb", "unit_bgr", "unit_rgb")

_EMBEDDERS = {
    # Measured on this repo (LFW, 60 identities): genuine/impostor cosine and AUC.
    #   sface + arcface template + raw_bgr  -> genuine 0.624 / impostor 0.079 / AUC 0.994
    #   sface + arcface template + half_rgb -> genuine 0.923 / impostor 0.893 / AUC 0.685  (collapsed)
    "sface": {
        "path": MODEL_DIR / "sface.onnx",
        "dim": 128,
        "alignment": "arcface",
        "norm": "raw_bgr",
        "license": "Apache-2.0 (OpenCV Zoo)",
        "note": "128-D, ~38 MB, commercial-safe, low memory. Wants raw BGR 0-255, NO normalisation.",
    },
    "arcface_glint360k": {
        "path": MODEL_DIR / "auraface_glintr100.onnx",
        "dim": 512,
        "alignment": "arcface",
        "norm": "half_rgb",
        "license": "InsightFace non-commercial research only",
        "note": "512-D, ~260 MB, higher accuracy on hard poses, needs ~300 MB free RAM to load. ArcFace convention.",
    },
}


def embedder_name() -> str:
    name = os.getenv("EMBEDDER", "arcface_glint360k").strip().lower()
    if name == "sface":
        raise ValueError("SFace was removed from the active face-recognition runtime; use arcface_glint360k")
    if name not in _EMBEDDERS:
        raise ValueError(f"Unknown EMBEDDER={name!r}; choose one of {sorted(_EMBEDDERS)}")
    return name


def embedder_path() -> Path:
    return _EMBEDDERS[embedder_name()]["path"]


def embedder_alignment() -> str:
    return _EMBEDDERS[embedder_name()].get("alignment", "arcface")


def embedder_norm() -> str:
    """Input normalisation the weights were trained with. Model-specific, see above."""
    return _EMBEDDERS[embedder_name()].get("norm", "half_rgb")


def embedder_info() -> dict:
    return {"name": embedder_name(), **_EMBEDDERS[embedder_name()]}


def database_url() -> str:
    """PostgreSQL DSN, read from the environment / .env only.

    There is deliberately no credential-bearing default here: if the DSN is missing the
    caller gets a clear error rather than a silently wrong connection using a baked-in
    password.
    """
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is not set. Provide it via the environment or the project .env file."
        )
    return dsn


def event_db() -> str:
    return os.getenv("EVENT_DB", str(ROOT / "events.sqlite"))


def redact_dsn(dsn: str) -> str:
    """Strip the userinfo from a DSN so it can be logged or reported safely."""
    return re.sub(r"(://)[^/@]*@", r"\1***:***@", dsn, count=1)


def describe() -> dict:
    info = embedder_info()
    try:
        dsn = redact_dsn(database_url())
    except RuntimeError:
        dsn = "(not set)"
    return {
        "detector": str(YOLO_FACE_MODEL),
        "liveness": str(LIVENESS_PATH),
        "embedder": info["name"],
        "embedder_path": str(info["path"]),
        "embedder_dim": info["dim"],
        "embedder_license": info["license"],
        "face_index": str(TURBOVEC_PATH),
        "face_match_threshold": ARCFACE_MATCH_THRESHOLD,
        "database_url": dsn,
        "event_db": event_db(),
        "vlm_provider": os.getenv("VLM_PROVIDER", "ollama"),
    }
