"""Diagnostic: why is LFW rank-1 so low?

Compares our preprocessing against OpenCV's own ``cv2.FaceRecognizerSF`` reference on the
same images. If the reference is much better, our preprocessing or alignment is wrong.
"""
import config  # noqa: F401

import json
import random
from pathlib import Path

import cv2
import numpy as np

from vision_core import OnnxEmbedder512, YuNetDetector, align_face

ROOT = Path("data/datasets/lfw")


def build_set(n_identities=40, seed=13):
    rng = random.Random(seed)
    names = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and len(list(p.glob("*.jpg"))) >= 2)
    names = names[:n_identities]
    gallery, probes = {}, []
    for name in names:
        files = sorted((ROOT / name).glob("*.jpg"))
        chosen = rng.choice(files)
        gallery[name] = chosen
        probes += [(name, f) for f in files if f != chosen]
    return gallery, probes


def main():
    gallery_files, probes = build_set()
    det = YuNetDetector(str(config.DETECTOR_PATH))
    emb = OnnxEmbedder512(str(config.embedder_path()))

    ref = cv2.FaceRecognizerSF.create(str(config.embedder_path()), "")

    cache = {}

    def detect(path):
        if path not in cache:
            img = cv2.imread(str(path))
            faces = det.detect(img) if img is not None else []
            cache[path] = (img, max(faces, key=lambda f: f.confidence) if faces else None)
        return cache[path]

    # how often does YuNet find a face at all?
    found = sum(1 for _, f in (detect(p) for p in list(gallery_files.values()) + [p for _, p in probes]) if f is not None)
    total = len(gallery_files) + len(probes)
    print(f"YuNet detection rate: {found}/{total} = {found/total:.3f}")

    ours, refs = [], []
    for name, path in gallery_files.items():
        img, face = detect(path)
        if face is None:
            continue
        ours.append((name, emb.embed(align_face(img, face.landmarks))))
        aligned = ref.alignCrop(img, face.bbox.astype(np.float32), face.landmarks)
        refs.append((name, ref.feature(aligned).reshape(-1).astype(np.float32)))

    def rank1(gal, probe_list, label):
        if not gal:
            print(f"{label}: no gallery")
            return
        names = [n for n, _ in gal]
        mat = np.stack([v / max(float(np.linalg.norm(v)), 1e-12) for _, v in gal])
        ok = 0
        n = 0
        for true_name, path in probe_list:
            img, face = detect(path)
            if face is None:
                continue
            if label.startswith("ours"):
                v = emb.embed(align_face(img, face.landmarks))
            else:
                v = ref.feature(ref.alignCrop(img, face.bbox.astype(np.float32), face.landmarks)).reshape(-1).astype(np.float32)
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            n += 1
            if names[int(np.argmax(mat @ v))] == true_name:
                ok += 1
        print(f"{label}: rank-1 = {ok}/{n} = {ok/max(1,n):.4f}")

    rank1(ours, probes, "ours (align_face + OnnxEmbedder512)")
    rank1(refs, probes, "ref  (cv2.FaceRecognizerSF alignCrop+feature)")

    # sanity: same image should give cosine ~1 under both
    img, face = detect(list(gallery_files.values())[0])
    a = emb.embed(align_face(img, face.landmarks))
    b = ref.feature(ref.alignCrop(img, face.bbox.astype(np.float32), face.landmarks)).reshape(-1).astype(np.float32)
    print(f"self-similarity ours={float(a @ a):.4f} ref={float(b / np.linalg.norm(b) @ (b / np.linalg.norm(b))):.4f}")

    # are the two embedding spaces even related?
    common = [n for n, _ in ours if n in {n2 for n2, _ in refs}]
    print(f"common identities embedded: {len(common)}")


if __name__ == "__main__":
    main()
