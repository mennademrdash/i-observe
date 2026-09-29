"""Sweep MiniFASNet input scaling/order; score by live-vs-spoof separation."""
import config  # noqa: F401

from pathlib import Path

import cv2
import numpy as np

from vision_core import YuNetDetector, expanded_crop

ROOT = Path("data/datasets/antispoof")


def variants(x):  # x: HxWx3 BGR float 0-255
    rgb = x[:, :, ::-1]
    return {
        "raw 0-255 BGR": x,
        "raw 0-255 RGB": rgb,
        "x/255 [0,1] BGR": x / 255.0,
        "x/255 [0,1] RGB": rgb / 255.0,
        "x/127.5-1 [-1,1] BGR": x / 127.5 - 1.0,
        "x/127.5-1 [-1,1] RGB": rgb / 127.5 - 1.0,
        "(x-127.5)/128 BGR": (x - 127.5) / 128.0,
        "(x-127.5)/128 RGB": (rgb - 127.5) / 128.0,
    }


def main():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    sess = ort.InferenceSession(str(config.LIVENESS_PATH), so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    det = YuNetDetector(str(config.DETECTOR_PATH))
    crops = {"live": [], "spoof": []}
    for label in ("live", "spoof"):
        for p in sorted((ROOT / label).glob("*.jpg")):
            img = cv2.imread(str(p))
            faces = det.detect(img) if img is not None else []
            if faces:
                crops[label].append(expanded_crop(img, max(faces, key=lambda f: f.confidence).bbox))

    print(f"live crops: {len(crops['live'])}   spoof crops: {len(crops['spoof'])}")
    print(f"{'variant':24} {'live_score(mean classX)':>34} {'spoof_score':>12} {'gap':>8}")
    results = []
    for label in variants(np.zeros((2, 2, 3), np.float32)):
        scores = {}
        for cls in range(3):
            for k in ("live", "spoof"):
                vals = []
                for c in crops[k]:
                    x = cv2.resize(c, (80, 80), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    blob = variants(x)[label].transpose(2, 0, 1)[None].astype(np.float32)
                    logits = np.asarray(sess.run(None, {iname: blob})[0]).reshape(-1).astype(np.float64)
                    pr = np.exp(logits - logits.max())
                    pr /= pr.sum()
                    vals.append(float(pr[cls]))
                scores[(cls, k)] = float(np.mean(vals)) if vals else 0.0
        # choose the class with the largest live/spoof gap
        best_cls = max(range(3), key=lambda c: scores[(c, "live")] - scores[(c, "spoof")])
        gap = scores[(best_cls, "live")] - scores[(best_cls, "spoof")]
        print(
            f"{label:24} class{best_cls}: live={scores[(best_cls,'live')]:.4f} "
            f"spoof={scores[(best_cls,'spoof')]:.4f} gap={gap:+.4f}"
        )
        results.append((gap, label, best_cls, scores[(best_cls, "live")], scores[(best_cls, "spoof")]))
    results.sort(reverse=True)
    print(f"\nbest: {results[0][1]}  live_class={results[0][2]}  gap={results[0][0]:+.4f}")
    print(f"      live={results[0][3]:.4f} spoof={results[0][4]:.4f}")


if __name__ == "__main__":
    main()
