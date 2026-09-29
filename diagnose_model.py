"""Isolate whether the weak embedding is a template mismatch or a bad model.

1. Same-person test on the real enrolled photos in data/people/<name>/ (6 photos each).
2. ArcFace template vs OpenCV's FaceRecognizerSF template, both through the same embedder.
"""
import config  # noqa: F401

from pathlib import Path

import cv2
import numpy as np

from vision_core import OnnxEmbedder512, YuNetDetector, align_face


def load_embedder():
    return OnnxEmbedder512(str(config.embedder_path()))


def main():
    det = YuNetDetector(str(config.DETECTOR_PATH))
    emb = load_embedder()
    ref = cv2.FaceRecognizerSF.create(str(config.embedder_path()), "")

    def embed_both(path):
        img = cv2.imread(str(path))
        faces = det.detect(img) if img is not None else []
        if not faces:
            return None, None
        face = max(faces, key=lambda f: f.confidence)
        a = emb.embed(align_face(img, face.landmarks))
        try:
            b = emb.embed(ref.alignCrop(img, face.bbox.astype(np.float32), face.landmarks))
        except Exception as exc:
            b = None
            print("alignCrop failed:", type(exc).__name__, exc)
        return a, b

    print("=== real same-person photos (data/people) vs random LFW ===")
    lfw = Path("data/datasets/lfw")
    lfw_files = sorted(p for p in lfw.glob("*/*.jpg"))[:200]

    people_dirs = [p for p in Path("data/people").iterdir() if p.is_dir()]
    for pd in people_dirs:
        files = sorted(pd.glob("*.jp*g"))
        if len(files) < 2:
            continue
        vecs_a, vecs_b = [], []
        for f in files:
            a, b = embed_both(f)
            if a is not None:
                vecs_a.append(a)
            if b is not None:
                vecs_b.append(b)
        if len(vecs_a) < 2:
            continue
        M = np.stack(vecs_a)
        sims = M @ M.T
        iu = np.triu_indices(len(M), 1)
        gen_a = float(sims[iu].mean())

        lfw_vecs = []
        for f in lfw_files[:60]:
            a, _ = embed_both(f)
            if a is not None:
                lfw_vecs.append(a)
        L = np.stack(lfw_vecs)
        imp_a = float((M @ L.T).mean())

        out = f"{pd.name}: genuine={gen_a:.4f} vs-LFW={imp_a:.4f} gap={gen_a-imp_a:+.4f}"
        if len(vecs_b) >= 2:
            Mb = np.stack(vecs_b)
            sb = Mb @ Mb.T
            gen_b = float(sb[iu].mean())
            imp_b = float((Mb @ L.T).mean())
            out += f"  | OpenCV-template genuine={gen_b:.4f} vs-LFW={imp_b:.4f} gap={gen_b-imp_b:+.4f}"
        print(out)

    print("\n=== embedding norm + saturation check ===")
    img = cv2.imread(str(lfw_files[0]))
    face = max(det.detect(img), key=lambda f: f.confidence)
    v = emb.embed(align_face(img, face.landmarks))
    print("norm:", float(np.linalg.norm(v)), "abs-mean:", float(np.abs(v).mean()), "max:", float(np.abs(v).max()))
    print("fraction of dims with |v|<0.01:", float((np.abs(v) < 0.01).mean()))


if __name__ == "__main__":
    main()
