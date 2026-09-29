"""Sweep input normalisation variants and score each by genuine/impostor separation.

SFace exports differ in whether they want BGR or RGB and which scaling. Feeding the wrong
one does not raise -- it collapses the embedding toward a common direction so every pair
scores near 1.0. This finds the variant the weights were actually trained with.
"""
import config  # noqa: F401

import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/datasets/lfw")
MODEL = str(config.embedder_path())


def variants(x):
    """x is a BGR float32 HxWx3 array in [0,255]."""
    rgb = x[:, :, ::-1]
    half = x / 127.5 - 1.0
    half_rgb = rgb / 127.5 - 1.0
    return {
        "A (x-127.5)/127.5 RGB  [current]": half_rgb.transpose(2, 0, 1)[None],
        "B (x-127.5)/127.5 BGR": half.transpose(2, 0, 1)[None],
        "C x/255 RGB": (rgb / 255.0).transpose(2, 0, 1)[None],
        "D x/255 BGR": (x / 255.0).transpose(2, 0, 1)[None],
        "E (x-127.5)/128 RGB": ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None],
        "F x/128-1 RGB": (rgb / 128.0 - 1.0).transpose(2, 0, 1)[None],
    }


def main():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    from vision_core import YuNetDetector, align_face

    det = YuNetDetector(str(config.DETECTOR_PATH))
    rng = random.Random(7)
    names = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and len(list(p.glob("*.jpg"))) >= 2)[:60]

    crops = {}
    for name in names:
        for f in sorted((ROOT / name).glob("*.jpg")):
            img = cv2.imread(str(f))
            faces = det.detect(img) if img is not None else []
            if faces:
                crops[(name, f.name)] = align_face(img, max(faces, key=lambda x: x.confidence).landmarks)

    pairs = []
    for name in names:
        fs = [k for k in crops if k[0] == name]
        if len(fs) >= 2:
            a, b = rng.sample(fs, 2)
            pairs.append((1, a, b))
    for _ in range(300):
        n1, n2 = rng.sample(names, 2)
        pairs.append((0, (n1, rng.choice([k for k in crops if k[0] == n1])[1]),
                         (n2, rng.choice([k for k in crops if k[0] == n2])[1])))

    print(f"pairs: {sum(1 for p in pairs if p[0]==1)} genuine / {sum(1 for p in pairs if p[0]==0)} impostor")
    print(f"{'variant':38} {'genuine':>8} {'impostor':>9} {'separation':>11} {'AUC':>7}")
    best = None
    for label in variants(np.zeros((2, 2, 3), np.float32)):
        emb_cache = {}

        def vec(key):
            if key not in emb_cache:
                blob = variants(crops[key].astype(np.float32))[label]
                v = np.asarray(sess.run(None, {iname: blob})[0]).reshape(-1).astype(np.float32)
                emb_cache[key] = v / max(float(np.linalg.norm(v)), 1e-12)
            return emb_cache[key]

        g = np.array([float(vec(a) @ vec(b)) for lab, a, b in pairs if lab == 1])
        i = np.array([float(vec(a) @ vec(b)) for lab, a, b in pairs if lab == 0])
        sep = (g.mean() - i.mean()) / max(1e-9, (g.std() + i.std()) / 2)
        auc = 1 - ((g[None, :] - i[:, None]) < 0).mean()
        print(f"{label:38} {g.mean():8.4f} {i.mean():9.4f} {sep:11.3f} {auc:7.4f}")
        if best is None or auc > best[1]:
            best = (label, auc)
    print(f"\nbest by AUC: {best[0]}  (AUC {best[1]:.4f})")


if __name__ == "__main__":
    main()
