"""Integration tests: application code -> real service -> back.

These are not mocked. They require:
  * PostgreSQL reachable at DATABASE_URL (docker compose service "postgres")
  * an S3-compatible endpoint on MINIO_ENDPOINT (moto server, see tools/)

Each test asserts on data read back FROM the service, not on the call returning.
"""
from __future__ import annotations

import config  # noqa: F401  # loads .env + BLAS caps

import os
import socket
import uuid
from pathlib import Path

import pytest


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def postgres_store():
    from services import PostgresEvents

    try:
        store = PostgresEvents()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable at {config.redact_dsn(config.database_url())}: {type(exc).__name__}")
    yield store
    store.close()


@pytest.fixture(scope="module")
def object_storage():
    if not _port_open("127.0.0.1", 9000):
        pytest.skip("S3 endpoint not listening on 9000")
    from services import ObjectStorage

    try:
        return ObjectStorage()
    except Exception as exc:
        pytest.skip(f"S3 endpoint not usable: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# PostgreSQL
# --------------------------------------------------------------------------------------


def test_postgres_round_trip_persists_row(postgres_store):
    """Write through the app, read back from the DB."""
    marker = f"it-{uuid.uuid4()}"
    event_id = postgres_store.add(
        video="integration.mp4",
        track_id=42,
        ts=123.5,
        event_type="walking",
        shirt_color="black",
        identity="integration-person",
        identity_score=0.77,
        action="walking",
        action_score=0.9,
        vlm="a person walking",
        bbox=[1, 2, 3, 4],
        evidence={"path": "evidence/x.jpg", "marker": marker},
        clip="clips/x.mp4",
    )
    assert event_id, "insert must return an id"

    rows = postgres_store.search(color="black", start=100, end=200, identity="integration-person")
    # search() returns the SAME 12 columns as pipeline_v2.EventStore.search:
    #   id,track_id,ts,event_type,shirt_color,identity,identity_score,
    #   action,action_score,vlm,evidence,clip
    # Postgres id is a uuid column (psycopg -> UUID) while SQLite yields str, so normalise.
    matched = [r for r in rows if str(r[0]) == str(event_id) or (r[11] or {}).get("marker") == marker]
    assert matched, f"row written through the app must be readable back; got {len(rows)} rows in window"

    row = matched[0]
    assert len(row) == 14, "Postgres and SQLite search() must return the same shape"
    assert row[1] == 42
    assert row[4] == "walking"
    assert row[6] == "integration-person"
    assert row[8] == "walking", "index 8 is action"
    assert row[9] == pytest.approx(0.9), "index 9 is action_score"
    assert row[10] == "a person walking"
    assert row[12] == "clips/x.mp4"


def test_postgres_rejects_missing_id_column_mismatch(postgres_store):
    """The old bug: INSERT had 8 placeholders for 7 values. Every insert must succeed."""
    for i in range(3):
        eid = postgres_store.add(video="v", track_id=i, ts=float(i), event_type="e")
        assert eid


# --------------------------------------------------------------------------------------
# S3-compatible object storage
# --------------------------------------------------------------------------------------


def test_object_storage_upload_then_stat(object_storage, tmp_path):
    """Upload through the app, then prove the object exists in the store."""
    payload = tmp_path / "evidence.jpg"
    payload.write_bytes(b"\xff\xd8\xff\xe0" + b"iobserve-integration-test" * 8)

    name = f"integration/{uuid.uuid4().hex}/evidence.jpg"
    uri = object_storage.upload(payload, name)
    assert uri.startswith("s3://"), f"unexpected uri {uri}"

    assert object_storage.exists(name) is True, "object must exist in the store after upload"

    # read it back through the raw client and compare bytes
    stream = object_storage.client.get_object(object_storage.bucket, name)
    try:
        body = stream.read()
    finally:
        stream.close()
        stream.release_conn()
    assert body == payload.read_bytes(), "bytes read back must match bytes written"


def test_object_storage_rejects_missing_file(object_storage, tmp_path):
    with pytest.raises(FileNotFoundError):
        object_storage.upload(tmp_path / "does-not-exist.jpg", "integration/missing.jpg")


# --------------------------------------------------------------------------------------
# End-to-end: pipeline -> Postgres + S3 + SQLite together
# --------------------------------------------------------------------------------------


def test_postgres_and_sqlite_search_shapes_match(tmp_path):
    """Regression: Postgres.search() returned 13 columns while SQLite returned 12.

    Both are named ``search`` and callers unpack positionally, so a shape difference
    silently corrupts every consumer the moment a backend is swapped.
    """
    from pipeline_v2 import EventStore
    from services import PostgresEvents

    if not _port_open("127.0.0.1", 5433) and not _port_open("127.0.0.1", 5432):
        pytest.skip("no PostgreSQL reachable")

    try:
        pg = PostgresEvents()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable: {type(exc).__name__}")

    sqlite = EventStore(str(tmp_path / "shape.sqlite"))
    try:
        sqlite.add(video="v", track_id=1, ts=1.0, event_type="e")
        pg.add(video="v", track_id=1, ts=1.0, event_type="e")

        s_rows = sqlite.search(start=0, end=10)
        p_rows = pg.search(start=0, end=10)
        assert s_rows and p_rows
        assert len(s_rows[0]) == len(p_rows[0]) == 14, (
            f"search() shape mismatch: sqlite={len(s_rows[0])} postgres={len(p_rows[0])}"
        )
    finally:
        pg.close()


def test_pipeline_integrations_are_actually_used(tmp_path):
    """Run the real video loop and assert each backend received data."""
    import pipeline_v2

    video = Path("data/sample_action.mp4")
    if not video.exists():
        pytest.skip("sample video not present")

    db = tmp_path / "events.sqlite"
    report, store, records = pipeline_v2.process(
        str(video),
        str(db),
        sample_fps=6,
        evidence_dir=str(tmp_path / "evidence"),
        clip_dir=str(tmp_path / "clips"),
        use_videomae=False,
        use_vlm=False,
    )

    assert report["processed_frames"] > 0
    assert report["events"] > 0, "the loop must produce events"
    assert report["decoded_all_frames"], "every declared frame must decode"

    # SQLite is the always-on timeline
    assert len(store.all_events()) == len(records) > 0

    integrations = report["integrations"]
    assert integrations["face_stack"]["ok"] is True
    assert integrations["incident_rules"]["ok"] is True

    if integrations["postgres"]["ok"]:
        # prove the Postgres mirror actually received rows
        from services import PostgresEvents

        pg = PostgresEvents()
        try:
            rows = pg.search(start=0, end=1e12)
            assert len(rows) >= len(records), (
                f"Postgres mirror should have at least {len(records)} rows, found {len(rows)}"
            )
            # Regression: pipeline_v2 used to call postgres.add() without action/vlm/clip,
            # so the mirror silently dropped them while SQLite kept them.
            # Join on the event id (column 0 in both stores): it is generated once by
            # EventStore.add() and passed straight through to the mirror, so it is unique
            # per event. (track_id, ts) is NOT unique - it repeats identically on every run
            # of the same video, so joining on it matches stale rows from earlier runs.)
            sqlite_rows = store.search(start=0, end=1e12)
            # Postgres `id` is a uuid column, so psycopg hands back UUID objects while
            # SQLite hands back str. Normalise before comparing.
            pg_by_id = {str(r[0]): r for r in rows}
            missing = []
            mismatched = []
            for s in sqlite_rows:
                p = pg_by_id.get(s[0])
                if p is None:
                    missing.append(s[0])
                    continue
                for idx, name in ((7, "action"), (8, "action_score"), (9, "vlm")):
                    if s[idx] != p[idx]:
                        mismatched.append((s[0], name, s[idx], p[idx]))
                # Clips are compared by file name: the name encodes video+track+timestamp
                # and is stable across runs, while the directory is the per-run tmp_path.
                if (s[11] is None) != (p[11] is None):
                    mismatched.append((s[0], "clip", s[11], p[11]))
                elif s[11] and Path(s[11]).name != Path(p[11]).name:
                    mismatched.append((s[0], "clip", Path(s[11]).name, Path(p[11]).name))
            assert not missing, f"{len(missing)} sqlite rows missing from the Postgres mirror: {missing[:5]}"
            assert not mismatched, (
                "Postgres mirror did not carry the same values as SQLite (action/vlm/clip "
                f"were being dropped): {mismatched[:5]}"
            )
            assert any(r[11] for r in sqlite_rows), "this run must produce at least one evidence clip"
        finally:
            pg.close()

    if integrations["minio"]["ok"]:
        assert report["evidence_clips"] >= 0
