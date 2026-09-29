"""Small native-camera viewer over the same recognition stack as the web app."""
from __future__ import annotations

import time

import cv2


def main() -> None:
    # Open and display the camera before importing/loading the AI runtime.
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and device index.")
    ok, frame = camera.read()
    if not ok:
        camera.release()
        raise RuntimeError("Webcam opened but returned no frame.")
    cv2.imshow("I-Observe", frame)
    cv2.waitKey(1)

    from query_api import analyse_faces

    print("YOLO11 Face + MiniFASNetV2 + ArcFace 512-D + TurboVec. Press q/Esc to quit.")
    last_result = []
    last_inference = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam stopped returning frames.")
            started = time.perf_counter()
            try:
                last_result, _, _ = analyse_faces(frame)
                last_inference = (time.perf_counter() - started) * 1000
            except Exception as exc:
                last_result = []
                cv2.putText(frame, f"Recognition error: {type(exc).__name__}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                print(f"Recognition error: {type(exc).__name__}: {exc}")

            for face in last_result:
                x1, y1, x2, y2 = face["bbox"]
                live = bool(face.get("live"))
                name = face.get("identity", "unknown") if live else "SPOOF"
                color = (0, 210, 0) if live and name not in ("unknown", "align_error") else (0, 170, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {face.get('score', 0):.2f} live={face.get('liveness_score', 0):.2f}",
                            (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"faces={len(last_result)} inference={last_inference:.0f} ms | browser UI: Enroll Person",
                        (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
            cv2.imshow("I-Observe", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
