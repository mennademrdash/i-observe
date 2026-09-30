"""Regression checks for the camera response and background recorder."""
import asyncio
import io
import threading

import cv2
import numpy as np
from fastapi import UploadFile

import query_api as api


def test_no_faces_does_not_crash_recording(monkeypatch):
    monkeypatch.setitem(api.LIVE, "recording", True)
    monkeypatch.setattr(api, "live_scene", lambda *args: {"tracks": [], "objects": []})
    result = api.record_live_frame(np.zeros((80, 80, 3), np.uint8),
                                   analysis=([], 0, (80, 80)))
    assert result["face_count"] == 0
    assert result["observations"] == []


def test_slow_recording_does_not_block_or_queue_camera_frames(monkeypatch):
    # Authentication is tested separately; this unit targets camera scheduling.
    monkeypatch.setattr(api, "_check_auth", lambda authorization: None)
    started, release = threading.Event(), threading.Event()
    calls = []
    faces = [{"index": 0, "identity": "known"}, {"index": 1, "identity": "unknown"}]
    monkeypatch.setattr(api, "analyse_faces", lambda frame: (faces, 2, (80, 80)))
    monkeypatch.setitem(api.LIVE, "recording", True)
    monkeypatch.setattr(api, "_record_future", None)
    monkeypatch.setattr(api, "_record_result", {})
    monkeypatch.setattr(api, "_record_error", None)

    def record(*args):
        calls.append(args)
        started.set()
        release.wait(10)
        return {"last": None}

    monkeypatch.setattr(api, "record_live_frame", record)
    encoded = cv2.imencode(".jpg", np.zeros((80, 80, 3), np.uint8))[1].tobytes()

    async def run():
        for _ in range(2):
            result = await api.api_camera_detect(
                UploadFile(file=io.BytesIO(encoded), filename="frame.jpg"), False, None)
            assert result["detections"] == faces
            assert result["face_count"] == 2
        assert started.wait(2)
        assert not api._record_future.done()
        assert len(calls) == 1

    try:
        asyncio.run(run())
    finally:
        release.set()
        if api._record_future is not None:
            api._record_future.result(timeout=5)
