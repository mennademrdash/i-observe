"""Throwaway: does YuNet still find the face as the frame shrinks?

Simulates what /api/camera/detect does -- a 1280x720 webcam frame downsized to 640
wide -- and reports the face box in pixels and the detector's own confidence at each
step. Run from the project root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: F401,E402

import cv2  # noqa: E402

from vision_core import YuNetDetector

SRC = Path("data/people/menna/01.jpeg")


def main():
    img = cv2.imread(str(SRC))
    h, w = img.shape[:2]
    print(f"source: {w}x{h}  (menna/01.jpeg)\n")
    print(f"{'frame':>12} {'scale':>6} {'faces':>6} {'box (w x h)':>14} {'det score':>10}")
    print("-" * 56)

    for width in (1280, 960, 640, 480, 320):
        if width > w:
            frame = img
        else:
            frame = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        det = YuNetDetector(str(config.DETECTOR_PATH))
        faces = det.detect(frame)
        if not faces:
            print(f"{width:>12} {width / w:>6.2f} {0:>6} {'-':>14} {'NO FACE':>10}")
            continue
        best = max(faces, key=lambda f: f.confidence)
        x1, y1, x2, y2 = best.bbox
        print(
            f"{width:>12} {width / w:>6.2f} {len(faces):>6} "
            f"{int(x2 - x1):>7} x {int(y2 - y1):<4} {best.confidence:>10.3f}"
        )

    print("\n--- same frames, threshold lowered to 0.3 ---")
    print(f"{'frame':>12} {'scale':>6} {'faces':>6} {'box (w x h)':>14} {'det score':>10}")
    print("-" * 56)
    for width in (1280, 640, 320):
        frame = img if width > w else cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        det = YuNetDetector(str(config.DETECTOR_PATH), threshold=0.3)
        faces = det.detect(frame)
        if not faces:
            print(f"{width:>12} {width / w:>6.2f} {0:>6} {'-':>14} {'NO FACE':>10}")
            continue
        best = max(faces, key=lambda f: f.confidence)
        x1, y1, x2, y2 = best.bbox
        print(
            f"{width:>12} {width / w:>6.2f} {len(faces):>6} "
            f"{int(x2 - x1):>7} x {int(y2 - y1):<4} {best.confidence:>10.3f}"
        )


if __name__ == "__main__":
    main()
