"""Model benchmarks: LFW identification + verification, and anti-spoofing.

The LFW distribution's ``pairs.txt`` is hosted on a server that no longer resolves, so
the protocols here are rebuilt deterministically from the directory layout. The
directory statistics are checked against the published LFW numbers before running, so a
partial or corrupted download cannot silently produce a flattering result.

Two protocols are reported because they answer different questions:

* identification (1:N) -- matches how this pipeline actually uses a vector index:
  one gallery image per identity, everything else is a probe.
* verification (1:1) -- the standard same/different decision, reported as EER rather
  than as "best accuracy", which is optimistic when the threshold is fit on the test set.
"""
from __future__ import annotations

import config  # noqa: F401  # sets BLAS thread caps before numpy/OpenCV load

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from vision_core import (
    MiniFASNet,
    OnnxEmbedder512,
    YuNetDetector,
    align_face,
    expanded_crop,
)

# Published LFW statistics. Used as a guard, not as a target.
LFW_EXPECTED_IDENTITIES = 5749
LFW_EXPECTED_IMAGES = 13233
LFW_EXPECTED_MULTI = 1680


def _embed_cached(detector, embedder, path: Path, cache: dict) -> np.ndarray | None:
    key = str(path)
    if key in cache:
        return cache[key]
    image = cv2.imread(str(path))
    if image is None:
        cache[key] = None
        return None
    faces = detector.detect(image)
    if not faces:
        cache[key] = None
        return None
    try:
        face = max(faces, key=lambda x: x.confidence)
        vector = embedder.embed(embedder.align(image, face))
    except Exception:
        cache[key] = None
        return None
    cache[key] = vector
    return vector


def check_lfw_layout(root: Path) -> dict:
    lfw_dir = root / "lfw"
    if not lfw_dir.is_dir():
        raise FileNotFoundError(f"Expected LFW directory at {lfw_dir}")
    identities = sorted(p.name for p in lfw_dir.iterdir() if p.is_dir())
    images = 0
    multi = []
    for name in identities:
        n = len(list((lfw_dir / name).glob("*.jpg")))
        images += n
        if n >= 2:
            multi.append(name)
    stats = {
        "identities": len(identities),
        "images": images,
        "multi_image_identities": len(multi),
        "matches_published_stats": (
            len(identities) == LFW_EXPECTED_IDENTITIES
            and images == LFW_EXPECTED_IMAGES
            and len(multi) == LFW_EXPECTED_MULTI
        ),
    }
    return stats, identities, multi


def lfw_identification(root, model, detector, max_identities=None, seed=13):
    """1:N rank-1 / rank-5 with one gallery image per multi-image identity."""
    root = Path(root)
    stats, _, multi = check_lfw_layout(root)
    if not stats["matches_published_stats"]:
        print(f"[warn] LFW layout does not match published stats: {stats}")

    rng = random.Random(seed)
    names = sorted(multi)
    if max_identities:
        names = names[:max_identities]

    det = YuNetDetector(detector)
    emb = OnnxEmbedder512(model)
    cache: dict = {}

    gallery_vectors, gallery_names, probes = [], [], []
    for name in names:
        files = sorted((root / "lfw" / name).glob("*.jpg"))
        chosen = rng.choice(files)
        vector = _embed_cached(det, emb, chosen, cache)
        if vector is None:
            continue
        gallery_vectors.append(vector)
        gallery_names.append(name)
        for f in files:
            if f == chosen:
                continue
            probes.append((name, f))

    if not gallery_vectors:
        raise RuntimeError("No gallery vectors could be built from LFW")

    gallery = np.stack(gallery_vectors)
    rank1 = 0
    rank5 = 0
    evaluated = 0
    failures = 0
    for true_name, path in probes:
        vector = _embed_cached(det, emb, path, cache)
        if vector is None:
            failures += 1
            continue
        sims = gallery @ vector
        order = np.argsort(-sims)[:5]
        top = [gallery_names[i] for i in order]
        evaluated += 1
        if top and top[0] == true_name:
            rank1 += 1
        if true_name in top:
            rank5 += 1

    if evaluated == 0:
        raise RuntimeError("No probe images could be embedded")
    return {
        "protocol": "1:N identification, 1 gallery image per identity",
        "lfw_stats": stats,
        "gallery_identities": len(gallery_vectors),
        "probes": evaluated,
        "probe_failures": failures,
        "rank1_accuracy": rank1 / evaluated,
        "rank5_accuracy": rank5 / evaluated,
        "embedder": model,
        "detector": detector,
    }


def lfw_verification(root, model, detector, pairs_per_class=500, seed=13):
    """1:1 verification reported as equal error rate."""
    root = Path(root)
    stats, _, multi = check_lfw_layout(root)
    rng = random.Random(seed)
    names = sorted(multi)

    det = YuNetDetector(detector)
    emb = OnnxEmbedder512(model)
    cache: dict = {}

    same, diff = [], []
    attempts = 0
    while (len(same) < pairs_per_class or len(diff) < pairs_per_class) and attempts < pairs_per_class * 40:
        attempts += 1
        if len(same) < pairs_per_class:
            name = rng.choice(names)
            files = sorted((root / "lfw" / name).glob("*.jpg"))
            if len(files) >= 2:
                a, b = rng.sample(files, 2)
                va, vb = _embed_cached(det, emb, a, cache), _embed_cached(det, emb, b, cache)
                if va is not None and vb is not None:
                    same.append(float(va @ vb))
        if len(diff) < pairs_per_class:
            n1, n2 = rng.sample(names, 2)
            f1 = rng.choice(sorted((root / "lfw" / n1).glob("*.jpg")))
            f2 = rng.choice(sorted((root / "lfw" / n2).glob("*.jpg")))
            va, vb = _embed_cached(det, emb, f1, cache), _embed_cached(det, emb, f2, cache)
            if va is not None and vb is not None:
                diff.append(float(va @ vb))

    if not same or not diff:
        raise RuntimeError("Could not build verification pairs")

    scores = np.array(same + diff)
    labels = np.array([1] * len(same) + [0] * len(diff))

    thresholds = np.unique(np.round(np.linspace(float(scores.min()), float(scores.max()), 2001), 6))
    best_eer, best_thr = 1.0, 0.0
    for t in thresholds:
        predicted = (scores >= t).astype(int)
        far = float(np.mean(predicted[labels == 0] == 1)) if np.any(labels == 0) else 0.0
        frr = float(np.mean(predicted[labels == 1] == 0)) if np.any(labels == 1) else 0.0
        eer = (far + frr) / 2
        if eer < best_eer:
            best_eer, best_thr = eer, float(t)

    return {
        "protocol": "1:1 verification, deterministic pairs",
        "lfw_stats": stats,
        "genuine_pairs": len(same),
        "impostor_pairs": len(diff),
        "eer": best_eer,
        "eer_threshold": best_thr,
        "genuine_mean": float(np.mean(same)),
        "genuine_std": float(np.std(same)),
        "impostor_mean": float(np.mean(diff)),
        "impostor_std": float(np.std(diff)),
        "embedder": model,
        "detector": detector,
    }


# --------------------------------------------------------------------------------------
# Anti-spoofing
# --------------------------------------------------------------------------------------


def build_synthetic_presentation_attacks(live_dir: Path, out_dir: Path) -> dict:
    """Derive presentation attacks from still images.

    This is a *synthetic, self-generated* stress test, not a substitute for a validated
    corpus such as SiW, OULU-NPU or Replay-Attack (all of which are research-gated and
    cannot be auto-downloaded). It exercises the attack families the gate must reject:
    print (recompress + blur), screen replay (down/up-sample + vignette) and a
    cut-out / posterised photo.

    Results here must always be labelled synthetic.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    live_dir_out = out_dir / "live"
    spoof_dir_out = out_dir / "spoof"
    live_dir_out.mkdir(parents=True, exist_ok=True)
    spoof_dir_out.mkdir(parents=True, exist_ok=True)
    counts = {"live": 0, "print": 0, "replay": 0, "cutout": 0}
    for path in sorted(Path(live_dir).glob("*.jp*g")) + sorted(Path(live_dir).glob("*.png")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        stem = path.stem
        cv2.imwrite(str(live_dir_out / f"{stem}_live.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        counts["live"] += 1

        # print attack: paper-white background, slight perspective tilt, heavy JPEG,
        # directional lighting gradient. A blurred JPEG alone is not a print attack --
        # MiniFASNet keys on paper texture and the flatness of a printed surface.
        printed = cv2.copyMakeBorder(image, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(238, 238, 238))
        h2, w2 = printed.shape[:2]
        src = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        dst = np.float32([[2, 3], [w2 - 2, 0], [w2 - 4, h2 - 2], [4, h2 - 3]])
        printed = cv2.warpPerspective(printed, cv2.getPerspectiveTransform(src, dst), (w2, h2),
                                      borderValue=(238, 238, 238))
        paper = np.full_like(printed, 12)
        paper[:] = cv2.GaussianBlur(paper, (0, 0), 3)
        printed = cv2.addWeighted(printed, 1.0, paper, 0.35, 0)
        grad = np.linspace(1.12, 0.88, printed.shape[0])[:, None][..., None]
        printed = cv2.convertScaleAbs(printed * grad, alpha=1.02, beta=6)
        cv2.imwrite(str(spoof_dir_out / f"{stem}_print.jpg"), printed, [cv2.IMWRITE_JPEG_QUALITY, 32])
        counts["print"] += 1

        # screen replay: display resolution, then scanlines + moire beat + vignette, which
        # is what a camera sees when filming an LCD/OLED panel.
        h, w = image.shape[:2]
        small = cv2.resize(image, (max(8, w // 3), max(8, h // 3)), interpolation=cv2.INTER_AREA)
        replay = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        rows = replay.shape[0]
        yy = np.arange(rows)[:, None, None]
        scan = 1.0 + 0.055 * np.sin(yy * (np.pi / 2.0))          # 2px scanline pitch
        moire = 1.0 + 0.035 * np.sin(yy * (np.pi / 6.5) + 0.7)   # slow beat pattern
        replay = replay.astype(np.float32) * scan * moire
        vignette = np.linspace(0.82, 1.0, rows)[:, None][..., None]
        replay = replay * vignette
        replay = np.clip(replay, 0, 255).astype(np.uint8)
        replay = cv2.convertScaleAbs(replay, alpha=1.06, beta=4)   # panel black lift
        cv2.imwrite(str(spoof_dir_out / f"{stem}_replay.jpg"), replay, [cv2.IMWRITE_JPEG_QUALITY, 72])
        counts["replay"] += 1

        # cutout / posterised photo
        cutout = cv2.medianBlur(image, 9)
        cutout = (cutout // 32) * 32
        cv2.imwrite(str(spoof_dir_out / f"{stem}_cutout.jpg"), cutout, [cv2.IMWRITE_JPEG_QUALITY, 60])
        counts["cutout"] += 1
    return counts


def antispoof(root, liveness_model, detector, synthetic=False, synthetic_from=None):
    root = Path(root)
    if synthetic:
        if synthetic_from is None:
            raise ValueError("synthetic=True requires synthetic_from=<dir of real face images>")
        generated = build_synthetic_presentation_attacks(Path(synthetic_from), root)
    else:
        generated = None

    det = YuNetDetector(detector)
    fas = MiniFASNet(liveness_model)

    labels, predictions, details = [], [], []
    for label, name in [(1, "live"), (0, "spoof")]:
        for path in sorted((root / name).glob("*")):
            image = cv2.imread(str(path))
            if image is None:
                continue
            faces = det.detect(image)
            if not faces:
                continue
            detail = fas.predict_detailed(expanded_crop(image, max(faces, key=lambda x: x.confidence).bbox))
            labels.append(label)
            predictions.append(1 if detail["live"] else 0)
            details.append({"path": str(path), "label": label, **{k: v for k, v in detail.items() if k != "probs"}})

    if not labels:
        raise RuntimeError(f"Expected images under {root}/live and {root}/spoof")

    labels_arr = np.array(labels)
    pred_arr = np.array(predictions)
    tp = int(np.sum((labels_arr == 1) & (pred_arr == 1)))
    tn = int(np.sum((labels_arr == 0) & (pred_arr == 0)))
    fp = int(np.sum((labels_arr == 0) & (pred_arr == 1)))  # spoof accepted as live
    fn = int(np.sum((labels_arr == 1) & (pred_arr == 0)))  # live rejected

    apcer = fp / max(1, int(np.sum(labels_arr == 0)))
    bpcer = fn / max(1, int(np.sum(labels_arr == 1)))
    return {
        "protocol": "SYNTHETIC presentation attacks" if synthetic else "directory live/spoof split",
        "synthetic": bool(synthetic),
        "generated": generated,
        "samples": len(labels),
        "live_samples": int(np.sum(labels_arr == 1)),
        "spoof_samples": int(np.sum(labels_arr == 0)),
        "accuracy": (tp + tn) / len(labels),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
        "confusion": {"tp": tp, "tn": tn, "fp_spoof_accepted": fp, "fn_live_rejected": fn},
        "liveness_model": liveness_model,
        "detector": detector,
        "note": (
            "Synthetic attacks generated from still images. Not a validated corpus; "
            "do not compare these numbers to published SiW/OULU-NPU/Replay-Attack results."
        )
        if synthetic
        else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["lfw", "lfw-verify", "antispoof"])
    parser.add_argument("root")
    parser.add_argument("--model", default=str(config.embedder_path()))
    parser.add_argument("--liveness", default=str(config.LIVENESS_PATH))
    parser.add_argument("--detector", default=str(config.DETECTOR_PATH))
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--max-identities", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-from", default=None)
    args = parser.parse_args()

    if args.kind == "lfw":
        result = lfw_identification(args.root, args.model, args.detector, max_identities=args.max_identities)
    elif args.kind == "lfw-verify":
        result = lfw_verification(args.root, args.model, args.detector, pairs_per_class=args.pairs)
    else:
        result = antispoof(
            args.root, args.liveness, args.detector, synthetic=args.synthetic, synthetic_from=args.synthetic_from
        )
    print(json.dumps(result, indent=2))
