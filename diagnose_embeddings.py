"""Is the embedding discriminative at all? Genuine vs impostor score distributions."""
import config  # noqa: F401

import random
from pathlib import Path

import cv2
import numpy as np

from vision_core import OnnxEmbedder512, YuNetDetector, align_face

ROOT = Path("data/datasets/lfw")


def main():
    rng = random.Random(7)
    names = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and len(list(p.glob("*.jpg"))) >= 2)[:60]
    det = YuNetDetector(str(config.DETECTOR_PATH))
    emb = OnnxEmbedder512(str(config.embedder_path()))
    cache = {}

    def vec(path):
        if path in cache:
            return cache[path]
        img = cv2.imread(str(path))
        faces = det.detect(img) if img is not None else []
        v = emb.embed(align_face(img, max(faces, key=lambda f: f.confidence).landmarks)) if faces else None
        cache[path] = v
        return v

    genuine, impostor = [], []
    for name in names:
        files = sorted((ROOT / name).glob("*.jpg"))
        a, b = rng.sample(files, 2)
        va, vb = vec(a), vec(b)
        if va is not None and vb is not None:
            genuine.append(float(va @ vb))
    for _ in range(300):
        n1, n2 = rng.sample(names, 2)
        va = vec(rng.choice(sorted((ROOT / n1).glob("*.jpg"))))
        vb = vec(rng.choice(sorted((ROOT / n2).glob("*.jpg"))))
        if va is not None and vb is not None:
            impostor.append(float(va @ vb))

    g, i = np.array(genuine), np.array(impostor)
    print(f"genuine  n={len(g)} mean={g.mean():.4f} std={g.std():.4f} min={g.min():.4f} max={g.max():.4f}")
    print(f"impostor n={len(i)} mean={i.mean():.4f} std={i.std():.4f} min={i.min():.4f} max={i.max():.4f}")
    print(f"separation (mean diff / pooled std) = {(g.mean()-i.mean())/max(1e-9,(g.std()+i.std())/2):.3f}")
    overlap = ((g[None, :] - i[:, None]) < 0).mean()
    print(f"pairwise overlap (AUC-ish, 0.5=useless) = {1-overlap:.4f}")

    # Do embeddings vary at all across different images?
    allv = np.stack([v for v in list(cache.values()) if v is not None])
    print(f"embedding matrix shape={allv.shape}")
    print(f"row norms: min={np.linalg.norm(allv,axis=1).min():.4f} max={np.linalg.norm(allv,axis=1).max():.4f}")
    print(f"dim std over images: mean={allv.std(axis=0).mean():.6f}")
    print(f"first 8 dims of two different images:")
    print("  ", np.round(allv[0][:8], 4))
    print("  ", np.round(allv[1][:8], 4))


if __name__ == "__main__":
    main()
