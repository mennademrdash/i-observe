from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

# The event row is identical whether it lands in SQLite or PostgreSQL, so both stores
# can be read back the same way. Keeping the columns in one place prevents the
# placeholder/value mismatch that previously made every Postgres insert fail.
EVENT_COLUMNS = (
    "id",
    "video",
    "wall_ts",
    "track_id",
    "ts",
    "event_type",
    "shirt_color",
    "identity",
    "identity_score",
    "action",
    "action_score",
    "vlm",
    "bbox",
    "evidence",
    "clip",
)


def event_values(event_id: str, event: dict) -> tuple:
    return (
        event_id,
        event.get("video"),
        float(event.get("wall_ts") or time.time()),
        int(event.get("track_id", 0)),
        float(event.get("ts", 0.0)),
        event.get("event_type"),
        event.get("shirt_color"),
        event.get("identity"),
        float(event.get("identity_score", 0.0)),
        event.get("action"),
        float(event.get("action_score", 0.0)),
        event.get("vlm"),
        json.dumps(event.get("bbox")),
        json.dumps(event.get("evidence", {})),
        event.get("clip"),
    )


class PostgresEvents:
    """PostgreSQL mirror of the event timeline.

    Uses ``psycopg`` (v3). Connection failure is reported to the caller as an exception
    so the pipeline can record it as an unavailable integration rather than silently
    losing every event.
    """

    def __init__(self, dsn=None):
        from psycopg_pool import ConnectionPool

        import config

        self.pool = ConnectionPool(
            conninfo=dsn or config.database_url(), min_size=1, max_size=int(os.getenv("PG_POOL_MAX", "12")),
            timeout=5, kwargs={"connect_timeout": 5}, open=True,
        )
        read_dsn = os.getenv("DATABASE_READ_URL", "").strip()
        self.read_pool = (ConnectionPool(conninfo=read_dsn, min_size=1, max_size=int(os.getenv("PG_READ_POOL_MAX", "12")),
                         timeout=5, kwargs={"connect_timeout": 5}, open=True) if read_dsn else self.pool)
        with self.pool.connection() as db, db.cursor() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS events(id uuid PRIMARY KEY, video text, wall_ts double precision,
                    track_id int,
                    ts double precision,
                    event_type text,
                    shirt_color text,
                    identity text,
                    identity_score double precision,
                    action text,
                    action_score double precision,
                    vlm text,
                    bbox jsonb,
                    evidence jsonb,
                    clip text
                )"""
            )
            # An existing table from an older schema may lack wall_ts; add it so the
            # timeline can be queried by real-world date instead of only video offset.
            for col, typ in (
                ("wall_ts", "double precision"),
                ("identity_score", "double precision"),
                ("action", "text"),
                ("action_score", "double precision"),
                ("vlm", "text"),
                ("clip", "text"),
                ("bbox", "jsonb"),
            ):
                c.execute(f"ALTER TABLE events ADD COLUMN IF NOT EXISTS {col} {typ}")
            c.execute("CREATE INDEX IF NOT EXISTS events_ts_idx ON events(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS events_wall_idx ON events(wall_ts)")
            c.execute("CREATE INDEX IF NOT EXISTS events_identity_idx ON events(identity)")

    def add(self, **event):
        event_id = event.get("id") or str(uuid.uuid4())
        placeholders = ",".join(["%s"] * len(EVENT_COLUMNS))
        with self.pool.connection() as db, db.cursor() as c:
            c.execute(
                f"INSERT INTO events({','.join(EVENT_COLUMNS)}) VALUES({placeholders}) ON CONFLICT (id) DO NOTHING",
                event_values(event_id, event),
            )
        return str(event_id)

    def get_event(self, event_id):
        with self.pool.connection() as db, db.cursor() as c:
            c.execute("SELECT id,track_id,ts,event_type,shirt_color,identity,identity_score,action,action_score,vlm,bbox,evidence,clip FROM events WHERE id=%s", (event_id,))
            return c.fetchone()

    def contains_event(self, event_id):
        with self.pool.connection() as db, db.cursor() as c:
            c.execute("SELECT id FROM events WHERE id=%s", (event_id,))
            return c.fetchone() is not None

    def search(self, color=None, start=0, end=1e12, identity=None):
        """Return rows in exactly the same column order as ``pipeline_v2.EventStore.search``.

        Both backends expose ``search`` and callers unpack positionally, so the shapes
        must match. Postgres stores ``bbox`` as well but does not return it here to keep
        the two interchangeable; it is still persisted and reachable via ``all_events``.
        """
        sql = (
            "SELECT id,track_id,ts,wall_ts,event_type,shirt_color,identity,identity_score,"
            "action,action_score,vlm,evidence,clip,video FROM events WHERE ts BETWEEN %s AND %s"
        )
        args = [start, end]
        if color:
            sql += " AND shirt_color=%s"
            args.append(color)
        if identity:
            sql += " AND lower(identity)=lower(%s)"
            args.append(identity)
        sql += " ORDER BY ts"
        with self.read_pool.connection() as db, db.cursor() as c:
            c.execute(sql, args)
            rows = c.fetchall()
        return rows

    def search_wall(self, start_wall, end_wall, color=None, identity=None, source=None):
        sql = (
            "SELECT id,track_id,ts,wall_ts,event_type,shirt_color,identity,identity_score,"
            "action,action_score,vlm,evidence,clip,video FROM events WHERE wall_ts >= %s AND wall_ts < %s"
        )
        args = [start_wall, end_wall]
        if color:
            sql += " AND shirt_color=%s"; args.append(color)
        if identity:
            sql += " AND lower(identity)=lower(%s)"; args.append(identity)
        if source:
            sql += " AND video=%s"; args.append(source)
        sql += " ORDER BY wall_ts"
        with self.read_pool.connection() as db, db.cursor() as c:
            c.execute(sql, args)
            rows = c.fetchall()
        return rows

    def all_events(self):
        with self.read_pool.connection() as db, db.cursor() as c:
            c.execute("SELECT id,track_id,ts,wall_ts,event_type,shirt_color,identity,identity_score,action,action_score,vlm,evidence,clip,video FROM events ORDER BY ts")
            rows = c.fetchall()
        return rows

    def close(self):
        try:
            self.pool.close()
            if self.read_pool is not self.pool:
                self.read_pool.close()
        except Exception:
            pass


class RedisCache:
    """Short-lived read cache; database remains the source of truth."""
    def __init__(self):
        import redis
        self.client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"), decode_responses=True,
            socket_connect_timeout=1, socket_timeout=1, health_check_interval=30,
        )

    def invalidate_reads(self):
        keys = list(self.client.scan_iter(match="iobserve:read:*", count=200))
        if keys:
            self.client.delete(*keys)


class ObjectStorage:
    """MinIO object storage for evidence frames and clips."""

    def __init__(self):
        from minio import Minio
        import urllib3

        self.bucket = os.getenv("MINIO_BUCKET", "evidence")
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.client = Minio(
            endpoint,
            access_key=os.getenv("MINIO_ACCESS_KEY", "observe"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "observe-secret"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            http_client=urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=3.0, read=10.0),
                retries=False,
            ),
        )
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload(self, path, object_name=None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot upload missing evidence file: {path}")
        name = object_name or path.name
        self.client.fput_object(self.bucket, name, str(path))
        return f"s3://{self.bucket}/{name}"

    def exists(self, object_name: str) -> bool:
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except Exception:
            return False


class IncidentRules:
    """Declarative incident rules evaluated against every event."""

    def __init__(self, path="incident_rules.json"):
        self.rules = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else []

    def evaluate(self, event):
        matches = []
        for rule in self.rules:
            conditions = rule.get("when", {}) or {}
            if not conditions:
                continue
            if all(str(event.get(k, "")).lower() == str(v).lower() for k, v in conditions.items()):
                matches.append(rule)
        return matches
