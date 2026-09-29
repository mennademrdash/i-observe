"""Build a REAL multi-face test image by compositing actual detected face photos.

No mocks: each tile is a real photograph that the detector finds a real face in.
Faces are placed on a neutral background at different offsets so YuNet must find
several independent detections in one frame.
"""
import sys

sys.path.insert(0, r"E:\I-observe")

import cv2
import numpy as np

from vision_core import YuNetDetector

DET = YuNetDetector(r"E:\I-observe\models\yunet.onnx")

SOURCES = [
    (r"E:\I-observe\data\people\menna\01.jpeg", "known"),
    (r"E:\I-observe\data\people\menna\03.jpeg", "known"),
    (r"E:\I-observe\data\datasets\lfw\Aaron_Eckhart\Aaron_Eckhart_0001.jpg", "stranger"),
    (r"E:\I-observe\data\datasets\lfw\Aaron_Guiel\Aaron_Guiel_0001.jpg", "stranger"),
]


def crop_face(path, pad=1.35):
    img = cv2.imread(path)
    if img is None:
        return None
    faces = DET.detect(img)
    if not faces:
        return None
    f = max(faces, key=lambda x: x.confidence)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in f.bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * (pad - 1) / 2))
    y1 = max(0, int(y1 - bh * (pad - 1) / 2))
    x2 = min(w, int(x2 + bw * (pad - 1) / 2))
    y2 = min(h, int(y2 + bh * (pad - 1) / 2))
    return img[y1:y2, x1:x2]


def make(tiles, out, canvas=(1280, 720)):
    W, H = canvas
    bg = np.full((H, W, 3), 235, np.uint8)
    spots = [(60, 60), (420, 70), (760, 60), (200, 380), (620, 380)]
    placed = []
    for (path, label), (px, py) in zip(tiles, spots):
        c = crop_face(path)
        if c is None:
            continue
        ch, cw = c.shape[:2]
        if ch > 260:
            c = cv2.resize(c, (int(cw * 260 / ch), 260))
            ch, cw = c.shape[:2]
        if px + cw > W:
            px = W - cw
        if py + ch > H:
            py = H - ch
        bg[py:py + ch, px:px + cw] = c
        placed.append((label, path, px, py, cw, ch))
    cv2.imwrite(out, bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return placed


if __name__ == "__main__":
    import os

    os.makedirs(r"E:\I-observe\outputs\multiface", exist_ok=True)
    # TEST 1: one face
    p1 = make([SOURCES[0]], r"E:\I-observe\outputs\multiface\one_face.jpg")
    # TEST 2/3: two known + one stranger
    p2 = make([SOURCES[0], SOURCES[1], SOURCES[2]], r"E:\I-observe\outputs\multiface\multi_face.jpg")
    # TEST 4: three strangers
    p3 = make([SOURCES[2], SOURCES[3], SOURCES[0]], r"E:\I-observe\outputs\multiface\three_strangers.jpg")
    for name, placed in (("one_face", p1), ("multi_face", p2), ("three_strangers", p3)):
        print(name, "-> placed", len(placed), "faces:", [p[0] for p in placed])
