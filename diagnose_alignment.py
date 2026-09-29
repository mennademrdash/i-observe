"""Dump aligned crops + a montage so we can see whether alignment is producing real faces."""
import config  # noqa: F401

from pathlib import Path

import cv2
import numpy as np

from vision_core import ARCFACE_TEMPLATE, YuNetDetector, align_face

ROOT = Path("data/datasets/lfw")
OUT = Path("outputs/diag")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    det = YuNetDetector(str(config.DETECTOR_PATH))
    names = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and len(list(p.glob("*.jpg"))) >= 2)[:6]

    rows = []
    stats = []
    for name in names:
        files = sorted((ROOT / name).glob("*.jpg"))[:2]
        cells = []
        for f in files:
            img = cv2.imread(str(f))
            faces = det.detect(img)
            face = max(faces, key=lambda x: x.confidence)
            aligned = align_face(img, face.landmarks)
            cells.append(aligned)
            stats.append(
                {
                    "person": name,
                    "src_shape": list(img.shape),
                    "aligned_mean": float(aligned.mean()),
                    "aligned_std": float(aligned.std()),
                    "bbox": [round(v, 1) for v in face.bbox.tolist()],
                    "kps": [[round(v, 1) for v in p] for p in face.landmarks.tolist()],
                }
            )
            # draw landmarks on the source for visual check
            vis = img.copy()
            for x, y in face.landmarks:
                cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cells.append(cv2.resize(vis, (112, 112)))
        row = np.hstack(cells)
        rows.append(row)

    montage = np.vstack(rows)
    cv2.imwrite(str(OUT / "aligned_montage.jpg"), montage)
    cv2.imwrite(str(OUT / "template_check.jpg"), _template_image())

    for s in stats:
        print(s)
    print("wrote", OUT / "aligned_montage.jpg")


def _template_image():
    img = np.full((112, 112, 3), 30, np.uint8)
    for x, y in ARCFACE_TEMPLATE:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)
    return img


if __name__ == "__main__":
    main()
