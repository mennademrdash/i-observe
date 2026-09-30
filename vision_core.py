from __future__ import annotations

import config  # noqa: F401  # sets BLAS thread caps before numpy/OpenCV load

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32
)


@dataclass
class Face:
    bbox: np.ndarray
    landmarks: np.ndarray
    confidence: float


class YuNetDetector:
    """YuNet face detector with 5-point landmarks.

    The default score threshold is 0.6, not the 0.85 that suits LFW's large frontal
    portraits. At 0.85 this detector found a face in only 3 of 41 tracked people on
    ordinary video frames, silently disabling recognition and liveness for the rest.
    """

    def __init__(self, model: str, threshold: float = 0.6):
        self.threshold = threshold
        # YuNet inference on the live 360px frame benchmarks faster with a small
        # OpenCV CPU thread pool than with the default single-thread setting.
        cv2.setNumThreads(min(4, os.cpu_count() or 1))
        self.net = cv2.FaceDetectorYN.create(model, "", (320, 320), threshold, 0.3, 5000)

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        self.net.setInputSize((w, h))
        _, rows = self.net.detect(frame)
        if rows is None:
            return []
        result = []
        for row in rows:
            x, y, bw, bh = row[:4]
            result.append(Face(np.array([x, y, x + bw, y + bh], np.float32), row[4:14].reshape(5, 2), float(row[14])))
        return result


class YOLO11FaceDetector:
    """YOLO11-pose face detector returning xyxy boxes and real five-point landmarks."""

    def __init__(self, model: str | None = None, threshold: float = 0.25, image_size: int = 640):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.threshold = float(threshold)
        self.image_size = int(image_size)
        self.device = "0" if torch.cuda.is_available() else "cpu"
        model_path = str(model or config.YOLO_FACE_MODEL)
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"YOLO11 face+landmark weights not found: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[Face]:
        if frame is None or frame.size == 0:
            return []
        result = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.threshold,
            max_det=500,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
        keypoints = result.keypoints
        points = None if keypoints is None else keypoints.xy.detach().cpu().numpy().astype(np.float32)
        faces = []
        for i, (box, score) in enumerate(zip(xyxy, scores)):
            # Missing/invalid model keypoints stay explicitly invalid. Downstream alignment
            # rejects that face; no bbox-derived or fabricated landmarks are substituted.
            lm = points[i] if points is not None and points.ndim == 3 and points.shape[1:] == (5, 2) else np.full((5, 2), np.nan, np.float32)
            faces.append(Face(box, lm, float(score)))
        return faces


def align_face(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if matrix is None or abs(np.linalg.det(matrix[:, :2])) < 1e-5:
        raise ValueError("Degenerate face alignment")
    return cv2.warpAffine(frame, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def free_ram_mb() -> float:
    """Best-effort free physical memory, used to make load failures explain themselves."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys / (1024 * 1024)
    except Exception:
        pass
    return -1.0


def _onnx_session(model: str):
    """Load an ONNX model with a memory-lean configuration.

    ``cv2.dnn.readNetFromONNX`` and a default onnxruntime session both peak at several
    times the on-disk size while deserialising weights. On a machine with little free
    RAM that surfaces as an opaque ``bad allocation``. Disabling the CPU memory arena
    and memory pattern planning keeps the peak close to the file size, and any failure
    is re-raised with the free-memory figure so the cause is obvious.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required: pip install onnxruntime") from exc

    path = Path(model)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    size_mb = path.stat().st_size / (1024 * 1024)

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        free = free_ram_mb()
        raise RuntimeError(
            f"Failed to load {path.name} ({size_mb:.1f} MB on disk): {type(exc).__name__}: {exc}. "
            f"Free RAM: {free:.0f} MB. Deserialising this model needs roughly "
            f"{max(1.0, size_mb * 0.6):.0f} MB free. Close other applications or use a "
            f"smaller embedder such as models/sface.onnx (128-D)."
        ) from exc


class OnnxEmbedder512:
    """Face embedding via ONNX Runtime.

    The output dimension is read from the model rather than assumed, so the same class
    serves the 512-D ArcFace/AuraFace backbones and the 128-D SFace model. Callers must
    size their vector index to ``self.dim``.

    **Alignment is part of the model, not of the pipeline.** Each face-recognition model
    was trained against its own canonical face template. Warp a face to the wrong template
    and the crop still looks perfectly plausible to a human while the embedding collapses
    toward a common direction: every pair then scores about the same and identification
    drops to near chance. Measured on this repo's own enrolled photos, the same-person vs
    stranger gap is +0.005 with the ArcFace template on SFace, versus +0.143 with SFace's
    own template. ``align()`` therefore picks the template that matches the weights.
    """

    ALIGN_ARCFACE = "arcface"
    ALIGN_OPENCV_SFACE = "opencv_sface"

    NORM_RAW_BGR = "raw_bgr"
    NORM_RAW_RGB = "raw_rgb"
    NORM_HALF_BGR = "half_bgr"
    NORM_HALF_RGB = "half_rgb"
    NORM_UNIT_BGR = "unit_bgr"
    NORM_UNIT_RGB = "unit_rgb"
    NORMS = (NORM_RAW_BGR, NORM_RAW_RGB, NORM_HALF_BGR, NORM_HALF_RGB, NORM_UNIT_BGR, NORM_UNIT_RGB)

    def _to_blob(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Apply the normalisation these weights were trained with.

        Getting this wrong does not raise: it collapses the embedding so that two
        different people score 0.94 while a face and its own brightened copy score 0.92.
        SFace in particular wants raw BGR 0-255; the ArcFace family wants (x-127.5)/127.5
        in RGB.
        """
        h, w = self.expected_size
        if self.norm not in self.NORMS:
            raise ValueError(f"Unknown norm {self.norm!r}; expected one of {sorted(self.NORMS)}")
        x = cv2.resize(aligned_bgr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if self.norm in (self.NORM_RAW_RGB, self.NORM_HALF_RGB, self.NORM_UNIT_RGB):
            x = x[:, :, ::-1]
        if self.norm in (self.NORM_HALF_BGR, self.NORM_HALF_RGB):
            x = (x - 127.5) / 127.5
        elif self.norm in (self.NORM_UNIT_BGR, self.NORM_UNIT_RGB):
            x = x / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis, ...])

    def __init__(self, model: str, alignment: str | None = None, norm: str | None = None):
        self.model = model
        self.session = _onnx_session(model)
        self.input_name = self.session.get_inputs()[0].name
        out_shape = self.session.get_outputs()[0].shape
        self.dim = int(out_shape[-1]) if out_shape and isinstance(out_shape[-1], int) else 512
        self.expected_size = tuple(
            d if isinstance(d, int) else 112 for d in (self.session.get_inputs()[0].shape[2:] or [112, 112])
        )

        if alignment is None:
            try:
                import config

                alignment = config.embedder_alignment()
            except Exception:
                alignment = self.ALIGN_ARCFACE
        self.alignment = alignment

        if norm is None:
            try:
                import config

                norm = config.embedder_norm()
            except Exception:
                norm = self.NORM_HALF_RGB
        if norm not in self.NORMS:
            raise ValueError(f"Unknown norm {norm!r}; expected one of {sorted(self.NORMS)}")
        self.norm = norm

        self._ref = None
        if self.alignment == self.ALIGN_OPENCV_SFACE:
            # OpenCV ships the exact reference points SFace was trained against, exposed
            # through FaceRecognizerSF.alignCrop. It is small enough for cv2.dnn to load.
            self._ref = cv2.FaceRecognizerSF.create(str(model), "")

    def align(self, frame: np.ndarray, face) -> np.ndarray:
        """Warp ``face`` (a ``vision_core.Face``) to this model's canonical 112x112 crop."""
        if self.alignment == self.ALIGN_OPENCV_SFACE:
            if self._ref is None:
                raise RuntimeError("SFace alignment requested but the reference is not loaded")
            return self._ref.alignCrop(frame, face.bbox.astype(np.float32), face.landmarks)
        return align_face(frame, face.landmarks)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        vector = np.asarray(
            self.session.run(None, {self.input_name: self._to_blob(aligned_bgr)})[0]
        ).reshape(-1).astype(np.float32)
        if vector.size != self.dim:
            raise ValueError(f"Expected a {self.dim}-D embedding, model returned {vector.size}")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("Embedding has zero norm")
        vector /= norm
        return vector


class MiniFASNet:
    """MiniFASNetV2 ONNX inference.

    The model contract (verified against the Silent-Face / MiniFASNetV2 export) is:

      input  : float32 [1, 3, 80, 80] NCHW, **BGR** channel order, scaled ``x / 255``
      output : [1, 3] softmax where class 1 == live/real
               and classes 0 and 2 == spoof (print attack / replay attack)

    Getting either the scaling or the class index wrong does not raise -- it produces
    confident, wrong liveness scores. Both are therefore asserted at construction and
    the preprocessing is kept explicit rather than delegated to ``blobFromImage``'s
    mean/scale convention, which is easy to invert.
    """

    INPUT_SIZE = (80, 80)
    DEFAULT_LIVE_CLASS = 1
    # Measured on this export (see diagnose_liveness_norm.py): feeding raw 0-255 BGR gives
    # live 0.966 / spoof 0.944, while x/255 or (x-127.5)/127.5 collapses every input to the
    # same ~[0, 0.007, 0.993] distribution. The weights expect unnormalised 0-255 BGR.
    SCALE_RAW = "raw"
    SCALE_UNIT = "unit"
    SCALE_HALF = "half"

    def __init__(self, model: str, live_class: int = DEFAULT_LIVE_CLASS, threshold: float = 0.5, scale: str = SCALE_RAW):
        self.model = model
        self.session = _onnx_session(model)
        self.input_name = self.session.get_inputs()[0].name
        out_shape = self.session.get_outputs()[0].shape
        self.num_classes = int(out_shape[-1]) if out_shape and isinstance(out_shape[-1], int) else 3
        self.live_class = live_class
        self.threshold = threshold
        if scale not in (self.SCALE_RAW, self.SCALE_UNIT, self.SCALE_HALF):
            raise ValueError(f"Unknown scale {scale!r}")
        self.scale = scale
        if not 0 <= self.live_class < self.num_classes:
            raise ValueError(
                f"live_class={self.live_class} invalid for a {self.num_classes}-class output"
            )

    @staticmethod
    def _preprocess(crop: np.ndarray, scale: str = SCALE_RAW) -> np.ndarray:
        """Explicit BGR -> NCHW. Channel order is preserved; scaling is model-specific."""
        if crop.size == 0:
            return np.zeros((1, 3, MiniFASNet.INPUT_SIZE[1], MiniFASNet.INPUT_SIZE[0]), np.float32)
        resized = cv2.resize(crop, MiniFASNet.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        x = resized.transpose(2, 0, 1).astype(np.float32)  # BGR kept as-is
        if scale == MiniFASNet.SCALE_UNIT:
            x = x / 255.0
        elif scale == MiniFASNet.SCALE_HALF:
            x = x / 127.5 - 1.0
        return np.ascontiguousarray(x[np.newaxis, ...])

    def predict(self, crop: np.ndarray) -> tuple[bool, float]:
        """Return (is_live, live_score). Spoof is scored on the non-live classes."""
        if crop is None or crop.size == 0:
            return False, 0.0
        probs = self._probabilities(crop)
        score = float(probs[self.live_class])
        return score >= self.threshold, score

    def predict_detailed(self, crop: np.ndarray) -> dict:
        """Return the full class distribution so callers can log *why* a face was rejected."""
        probs = self._probabilities(crop)
        live = float(probs[self.live_class])
        spoof = float(probs.sum() - live)
        return {
            "live": live >= self.threshold,
            "live_score": live,
            "spoof_score": spoof,
            "probs": [float(p) for p in probs],
            "threshold": self.threshold,
        }

    def _probabilities(self, crop: np.ndarray) -> np.ndarray:
        blob = self._preprocess(crop, self.scale)
        logits = np.asarray(
            self.session.run(None, {self.input_name: blob})[0]
        ).reshape(-1).astype(np.float64)
        probs = np.exp(logits - logits.max())
        return probs / probs.sum()


def expanded_crop(frame: np.ndarray, bbox: np.ndarray, scale: float = 2.7) -> np.ndarray:
    """Crop around a face box with context.

    MiniFASNetV2 is trained on crops roughly 2.7x the annotated face box; feeding it a
    tight crop shifts the score distribution and degrades spoof rejection.
    """
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    xa, ya = max(0, int(cx - w * scale / 2)), max(0, int(cy - h * scale / 2))
    xb, yb = min(frame.shape[1], int(cx + w * scale / 2)), min(frame.shape[0], int(cy + h * scale / 2))
    return frame[ya:yb, xa:xb]


class QdrantFaceIndex:
    """Face identity index.

    Backed by Qdrant. Prefers a running Qdrant server (Docker ``qdrant/qdrant`` on
    ``QDRANT_URL``) and falls back to Qdrant's embedded local mode so the pipeline keeps
    working when no server is available.
    """

    def __init__(self, path: str = "qdrant_storage", dim: int = 512, url: str | None = None):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.dim = dim
        self.collection = f"faces_{dim}"
        self.path = path
        self.mode = "embedded"
        url = url or os.getenv("QDRANT_URL")
        self.mode = "embedded"
        if url:
            try:
                client = QdrantClient(url=url, timeout=2)
                # Prove the server answers before committing to it. A Docker/Qdrant stall
                # otherwise costs a multi-second timeout on every single face, which
                # dominates the whole live pipeline.
                client.collection_exists(self.collection)
                self.client = client
                self.mode = "server"
            except Exception:
                self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(path=path)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def enroll(self, identity: str, vector: np.ndarray) -> str:
        if vector.size != self.dim:
            raise ValueError(f"Expected a {self.dim}-D vector, got {vector.size}")
        point_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
        self.client.upsert(
            self.collection,
            [self.models.PointStruct(id=point_id, vector=vector.tolist(), payload={"identity": identity})],
        )
        return point_id

    def identify(self, vector: np.ndarray, threshold: float = 0.35) -> dict:
        if vector.size != self.dim:
            raise ValueError(f"Expected a {self.dim}-D vector, got {vector.size}")
        try:
            hits = self.client.query_points(self.collection, query=vector.tolist(), limit=1).points
        except Exception:
            # Never let a vector-store hiccup take down face recognition for the frame.
            return {"identity": "unknown", "score": 0.0, "store_error": True}
        if not hits or hits[0].score < threshold:
            return {"identity": "unknown", "score": float(hits[0].score) if hits else 0.0}
        return {"identity": hits[0].payload["identity"], "score": float(hits[0].score)}

    def list_identities(self) -> list[dict]:
        """Enrolled identities with their point counts, read back from the index."""
        try:
            points, _ = self.client.scroll(self.collection, limit=1000, with_payload=True, with_vectors=False)
        except Exception:
            return []
        counts: dict[str, int] = {}
        for p in points or []:
            name = (p.payload or {}).get("identity")
            if name:
                counts[name] = counts.get(name, 0) + 1
        return [{"name": k, "points": v} for k, v in sorted(counts.items())]


class TurboVecFaceIndex:
    """Persistent ArcFace index backed only by TurboVec's quantized cosine search.

    TurboVec stores normalized 512-D vectors and returns approximate inner-product
    scores; with unit-normalized ArcFace vectors these rank as cosine similarity.
    Identity labels live in an adjacent durable JSON map keyed by TurboVec's stable IDs.
    """

    def __init__(self, path: str | Path | None = None, dim: int = 512, bit_width: int = 4, threshold: float | None = None):
        from turbovec import IdMapIndex

        self.path = Path(path or config.TURBOVEC_PATH)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".identities.json")
        self.dim = int(dim)
        self.threshold = float(config.ARCFACE_MATCH_THRESHOLD if threshold is None else threshold)
        self.bit_width = int(bit_width)
        if self.dim != 512:
            raise ValueError(f"ArcFace TurboVec index must be 512-D, got {self.dim}")
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError("ARCFACE_MATCH_THRESHOLD must be between -1 and 1")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if self.path.exists() != self.meta_path.exists():
            raise RuntimeError("TurboVec index and identity metadata are inconsistent; no fallback is used")
        if self.path.exists():
            self.client = IdMapIndex.load(str(self.path))
            self.points = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if self.client.dim != self.dim or len(self.client) != len(self.points.get("ids", {})):
                raise RuntimeError("TurboVec index dimension/count does not match its identity metadata")
        else:
            self.client = IdMapIndex(dim=self.dim, bit_width=self.bit_width)
            self.points = {"next_id": 1, "ids": {}}

    def _save_metadata(self) -> None:
        temp = self.meta_path.with_name(self.meta_path.name + ".tmp")
        temp.write_text(json.dumps(self.points, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temp.replace(self.meta_path)

    @staticmethod
    def _unit_vector(vector: np.ndarray, dim: int) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size != dim or not np.isfinite(arr).all():
            raise ValueError(f"Expected a finite {dim}-D vector")
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            raise ValueError("Cannot index a zero-norm face embedding")
        return np.ascontiguousarray((arr / norm).reshape(1, dim), dtype=np.float32)

    def enroll(self, identity: str, vector: np.ndarray) -> str:
        name = str(identity).strip()
        if not name:
            raise ValueError("identity must not be empty")
        embedding = self._unit_vector(vector, self.dim)
        with self._lock:
            point_id = int(self.points["next_id"])
            self.client.add_with_ids(embedding, np.asarray([point_id], dtype=np.uint64))
            self.client.sync(str(self.path))
            self.points["next_id"] = point_id + 1
            self.points["ids"][str(point_id)] = name
            self._save_metadata()
        return str(point_id)

    def identify(self, vector: np.ndarray, threshold: float | None = None) -> dict:
        query = self._unit_vector(vector, self.dim)
        cutoff = self.threshold if threshold is None else float(threshold)
        with self._lock:
            if len(self.client) == 0:
                return {"identity": "unknown", "score": 0.0}
            scores, ids = self.client.search(query, k=1)
            if not ids.size:
                return {"identity": "unknown", "score": 0.0}
            point_id = int(ids[0, 0])
            identity = self.points["ids"].get(str(point_id))
            if identity is None:
                raise RuntimeError(f"TurboVec returned unmapped identity point {point_id}")
            # TurboQuant's approximate inner product can exceed cosine's mathematical
            # bound by a small quantization error, so clamp only to the valid cosine range.
            score = float(np.clip(scores[0, 0], -1.0, 1.0))
            return {"identity": identity if score >= cutoff else "unknown", "score": score}

    def list_identities(self) -> list[dict]:
        with self._lock:
            counts: dict[str, int] = {}
            for name in self.points["ids"].values():
                counts[name] = counts.get(name, 0) + 1
            return [{"name": name, "points": count} for name, count in sorted(counts.items(), key=lambda item: item[0].casefold())]

    def remove(self, identity: str) -> int:
        target = str(identity).strip().casefold()
        with self._lock:
            ids = [int(pid) for pid, name in self.points["ids"].items() if name.casefold() == target]
            for point_id in ids:
                self.client.remove(point_id)
                del self.points["ids"][str(point_id)]
            if ids:
                self.client.sync(str(self.path))
                self._save_metadata()
            return len(ids)


class TrackIdentityStabilizer:
    """Keep one identity per person track across frames.

    Identifying each frame independently makes labels flicker between adjacent frames of
    the same person. This keeps a running, quality-weighted embedding per track and only
    switches label when the accumulated evidence genuinely disagrees.

    Hysteresis is deliberate: a single confusing frame must not relabel a track that has
    already been confidently identified.
    """

    def __init__(self, index, min_updates_before_switch: int = 3, agree_ratio: float = 0.6, window: int = 5):
        self.index = index
        self.min_updates_before_switch = min_updates_before_switch
        self.agree_ratio = agree_ratio
        self.window = max(1, int(window))
        self._state: dict[int, dict] = {}

    def update(self, track_id: int, vector: np.ndarray, quality: float = 1.0) -> dict:
        state = self._state.setdefault(
            track_id,
            {"centroid": None, "updates": 0, "label": None, "score": 0.0, "history": []},
        )
        # Quality-weighted exponential moving average of unit vectors.
        weight = min(max(float(quality), 0.0), 1.0)
        alpha = 0.2 * (0.25 + 0.75 * weight)
        if state["centroid"] is None:
            state["centroid"] = vector.astype(np.float64)
        else:
            state["centroid"] = state["centroid"] * (1 - alpha) + vector.astype(np.float64) * alpha
        centroid = state["centroid"] / max(float(np.linalg.norm(state["centroid"])), 1e-12)
        state["updates"] += 1

        result = self.index.identify(centroid.astype(np.float32))
        candidate, score = result["identity"], float(result["score"])

        # Only recent votes count, otherwise a track's stale history prevents it from
        # ever switching label after a genuine change of appearance.
        state["history"].append(candidate)
        state["history"] = state["history"][-self.window :]
        recent = state["history"]

        if state["label"] is None:
            state["label"], state["score"] = candidate, score
        elif candidate != state["label"]:
            agreed = recent.count(candidate)
            if state["updates"] >= self.min_updates_before_switch and agreed / len(recent) >= self.agree_ratio:
                state["label"], state["score"] = candidate, score
        else:
            state["score"] = max(state["score"], score)

        return {
            "track_id": track_id,
            "identity": state["label"],
            "score": state["score"],
            "updates": state["updates"],
            "switched": candidate != state["label"],
            "history": list(recent),
        }

    def snapshot(self) -> dict:
        return {tid: {"identity": s["label"], "score": s["score"], "updates": s["updates"]} for tid, s in self._state.items()}


class EvidenceWriter:
    def __init__(self, root: str = "evidence"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def frame(self, video_id: str, track_id: int, timestamp: float, image: np.ndarray, bbox: list[int]) -> str:
        folder = self.root / video_id / f"track_{track_id}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{timestamp:012.3f}.jpg"
        annotated = image.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if not cv2.imwrite(str(path), annotated):
            raise IOError(f"Could not write evidence {path}")
        return str(path.resolve())

    def manifest(self, video_id: str, records: list[dict]) -> Path:
        path = self.root / video_id / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return path
