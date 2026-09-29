"""TEST 1-4: real multi-face verification against the running API.

Every image is a real photograph with real faces; nothing is mocked.
"""
import json
import subprocess
import sys
import urllib.request

sys.path.insert(0, r"E:\I-observe")
import config  # noqa: F401,E402

BASE = "http://127.0.0.1:8000"
OK = set("unknown spoof_rejected align_error alignment_error liveness_error no_face".split())


def post(path, field, filename, extra=None):
    cmd = ["curl.exe", "-s", "-X", "POST", "-F", f"{field}=@{filename}", f"{BASE}{path}"]
    for k, v in (extra or {}).items():
        cmd += ["-F", f"{k}={v}"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    return json.loads(out)


def start():
    subprocess.run(
        ["curl.exe", "-s", "-X", "POST", f"{BASE}/api/live/start"],
        capture_output=True, text=True,
    )


def stop():
    return json.loads(
        subprocess.run(["curl.exe", "-s", "-X", "POST", f"{BASE}/api/live/stop"],
                       capture_output=True, text=True).stdout
    )


def report(name, r, expect_faces):
    dets = r.get("detections", [])
    print(f"\n=== {name} ===")
    print(f"face_count = {r.get('face_count')}   detections = {len(dets)}   recorded = {r.get('recorded')}")
    boxes = [tuple(d["bbox"]) for d in dets]
    print("distinct bboxes:", len(set(boxes)), "of", len(boxes))
    for d in dets:
        live = "LIVE" if d.get("live") else "SPOOF/REJ"
        w = round(d["bbox"][2] - d["bbox"][0])
        h = round(d["bbox"][3] - d["bbox"][1])
        print(f"  idx={d.get('index')} {d.get('identity'):<14} {live:<9} "
              f"score={d.get('score', 0):.3f} live={d.get('liveness_score', 0):.3f} "
              f"bbox={d['bbox']} ({w}x{h})")
    assert r.get("face_count", 0) >= expect_faces, f"expected >= {expect_faces} faces, got {r.get('face_count')}"
    assert len(set(boxes)) == len(boxes), "each face must get its own bbox"
    assert len(dets) == r["face_count"], "every detected face must be processed"
    return dets


def main():
    base = r"E:\I-observe\outputs\multiface"
    results = {}

    # TEST 1 - one person
    start()
    d1 = report("TEST 1: one face", post("/api/camera/detect", "image", f"{base}\\one_face.jpg", {"annotate": "true"}), 1)
    results["T1"] = len(d1)

    # TEST 2 + 3 - two enrolled Menna photos + one stranger
    start()
    r2 = post("/api/live/observe", "image", f"{base}\\multi_face.jpg")
    d2 = report("TEST 2+3: two known + one stranger (live recording)", r2, 3)
    known = [d for d in d2 if d.get("identity") not in OK and d.get("identity")]
    unknown = [d for d in d2 if d.get("identity") in OK or not d.get("live")]
    print(f"  -> recognized: {[d['identity'] for d in known]}  not-recognized: {[d['identity'] for d in unknown]}")
    results["T2_faces"] = len(d2)
    results["T3_known"] = len(known)
    results["T3_unknown"] = len(unknown)
    assert len(known) >= 1, "at least one enrolled face must still be recognized"
    assert len(unknown) >= 1, "the stranger must not be silently recognized"

    # TEST 4 - three faces incl. a rejected one; liveness must not suppress others
    r4 = post("/api/live/observe", "image", f"{base}\\three_strangers.jpg")
    d4 = report("TEST 4: three faces, mixed liveness", r4, 3)
    results["T4_faces"] = len(d4)
    results["T4_live"] = sum(1 for d in d4 if d.get("live"))

    ev = stop()
    print(f"\nrecorded total for session: {ev.get('events_recorded')}  people: {ev.get('people')}")
    results["recorded"] = ev.get("events_recorded")

    print("\nRESULTS:", json.dumps(results))
    print("ALL MULTI-FACE TESTS PASSED")


if __name__ == "__main__":
    main()
