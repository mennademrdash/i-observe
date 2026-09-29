"""Decisive 2D sweep: alignment template x input normalisation, scored by identification.

Also sanity-checks the embedding space itself: an image compared with a brightness-perturbed
copy of itself must score far above a different person. If it does not, the model output is
collapsed and no template/preprocessing choice will save it.
"""
import config  # noqa: F401

import random
from pathlib import Path

import cv2
import numpy as np

from vision_core import YuNetDetector, align_face

ROOT = Path("data/datasets/lfw")
MODEL = str(config.embedder_path())


def preprocess(bgr, kind):
    rgb = bgr[:, :, ::-1]
    if kind == "half_rgb":
        x = (rgb - 127.5) / 127.5
    elif kind == "half_bgr":
        x = (bgr - 127.5) / 127.5
    elif kind == "unit_rgb":
        x = rgb / 255.0
    elif kind == "unit_bgr":
        x = bgr / 255.0
    elif kind == "raw_rgb":
        x = rgb.copy()
    elif kind == "raw_bgr":
        x = bgr.copy()
    else:
        raise ValueError(kind)
    return x.transpose(2, 0, 1).astype(np.float32)[None]


def main():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    det = YuNetDetector(str(config.DETECTOR_PATH))
    ref = cv2.FaceRecognizerSF.create(MODEL, "")

    def aligned(img, face, which):
        if which == "arcface":
            return align_face(img, face.landmarks)
        return ref.alignCrop(img, face.bbox.astype(np.float32), face.landmarks)

    def embed(img, face, which, kind):
        a = aligned(img, face, which)
        v = np.asarray(sess.run(None, {iname: preprocess(a.astype(np.float32), kind)})[0]).reshape(-1).astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-12)

    # ---- sanity: identity / perturbation ladder -------------------------------
    sample = sorted(ROOT.glob("*/*.jpg"))[0]
    img = cv2.imread(str(sample))
    face = max(det.detect(img), key=lambda f: f.confidence)
    v0 = embed(img, face, "arcface", "half_rgb")
    bright = cv2.convertScaleAbs(img, alpha=1.15, beta=10)
    vb = embed(bright, face, "arcface", "half_rgb")
    other = cv2.imread(str(sorted(ROOT.glob("*/*.jpg"))[500]))
    vo = embed(other, max(det.detect(other), key=lambda f: f.confidence), "arcface", "half_rgb")
    print("sanity (arcface/half_rgb):")
    print(f"  image vs itself            = {float(v0 @ v0):.4f}   (expect 1.0000)")
    print(f"  image vs brightened copy   = {float(v0 @ vb):.4f}   (expect > 0.90)")
    print(f"  image vs a different person= {float(v0 @ vo):.4f}   (expect < 0.40)")
    print()

    # ---- 2D sweep -------------------------------------------------------------
    rng = random.Random(7)
    names = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and len(list(p.glob("*.jpg"))) >= 2)[:60]
    pairs = []
    for name in names:
        fs = sorted((ROOT / name).glob("*.jpg"))
        a, b = rng.sample(fs, 2)
        pairs.append((1, a, b))
    for _ in range(300):
        n1, n2 = rng.sample(names, 2)
        pairs.append((0, rng.choice(sorted((ROOT / n1).glob("*.jpg"))), rng.choice(sorted((ROOT / n2).glob("*.jpg")))))

    cache = {}

    def face_of(path):
        if path not in cache:
            im = cv2.imread(str(path))
            cache[path] = (im, max(det.detect(im), key=lambda f: f.confidence))
        return cache[path]

    print(f"{'template':10} {'norm':10} {'genuine':>8} {'impostor':>9} {'gap':>8} {'AUC':>7}")
    best = None
    for which in ("arcface", "sface"):
        for kind in ("half_rgb", "half_bgr", "unit_rgb", "unit_bgr", "raw_rgb", "raw_bgr"):
            ec = {}

            def vec(path, which=which, kind=kind):
                k = (path, which, kind)
                if k not in ec:
                    im, f = face_of(path)
                    ec[k] = embed(im, f, which, kind)
                return ec[k]

            g = np.array([float(vec(a) @ vec(b)) for lab, a, b in pairs if lab == 1])
            i = np.array([float(vec(a) @ vec(b)) for lab, a, b in pairs if lab == 0])
            gap = g.mean() - i.mean()
            auc = 1 - ((g[None, :] - i[:, None]) < 0).mean()
            print(f"{which:10} {kind:10} {g.mean():8.4f} {i.mean():9.4f} {gap:8.4f} {auc:7.4f}")
            if best is None or auc > best[2]:
                best = (which, kind, auc, gap)
    print(f"\nbest: template={best[0]} norm={best[1]} AUC={best[2]:.4f} gap={best[3]:.4f}")


if __name__ == "__main__":
    main()
