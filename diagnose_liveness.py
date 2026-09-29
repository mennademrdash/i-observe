"""Which output class is 'live' on THIS MiniFASNet export? Print the distribution."""
import config  # noqa: F401

from pathlib import Path

import cv2
import numpy as np

from vision_core import MiniFASNet, YuNetDetector, expanded_crop

ROOT = Path("data/datasets/antispoof")


def main():
    det = YuNetDetector(str(config.DETECTOR_PATH))
    fas = MiniFASNet(str(config.LIVENESS_PATH))
    print(f"num_classes={fas.num_classes} live_class={fas.live_class} threshold={fas.threshold}")

    for label_dir in ("live", "spoof"):
        folder = ROOT / label_dir
        rows = []
        for p in sorted(folder.glob("*.jpg")):
            img = cv2.imread(str(p))
            faces = det.detect(img)
            if not faces:
                rows.append((p.name, "NO_FACE", None))
                continue
            crop = expanded_crop(img, max(faces, key=lambda f: f.confidence).bbox)
            d = fas.predict_detailed(crop)
            rows.append((p.name, [round(x, 3) for x in d["probs"]], round(d["live_score"], 3)))
        print(f"\n--- {label_dir} ---")
        for name, probs, score in rows:
            print(f"  {name:34} probs={probs} live_score={score}")

    print("\nper-class mean probability by true label:")
    for label_dir in ("live", "spoof"):
        acc = []
        for p in sorted((ROOT / label_dir).glob("*.jpg")):
            img = cv2.imread(str(p))
            faces = det.detect(img)
            if faces:
                d = fas.predict_detailed(expanded_crop(img, max(faces, key=lambda f: f.confidence).bbox))
                acc.append(d["probs"])
        if acc:
            m = np.mean(acc, axis=0)
            print(f"  {label_dir:6} mean probs = {[round(float(x),3) for x in m]}")


if __name__ == "__main__":
    main()
