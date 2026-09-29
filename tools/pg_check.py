import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: F401,E402
from services import PostgresEvents

pg = PostgresEvents()
c = pg.db.cursor()
c.execute("SELECT count(1) FROM events WHERE clip LIKE %s", ("%pytest-0%",))
print("rows w/ clip in current run   :", c.fetchone()[0])
c.execute("SELECT count(1) FROM events WHERE evidence::text LIKE %s", ("%pytest-0%",))
print("rows w/ evidence in current   :", c.fetchone()[0])
c.execute("SELECT clip FROM events WHERE clip LIKE %s LIMIT 3", ("%pytest-0%",))
for r in c.fetchall():
    print("   ", r[0])
c.execute("SELECT count(1) FROM events WHERE clip IS NOT NULL")
print("total non-null clips          :", c.fetchone()[0])
c.execute("SELECT count(1) FROM events")
print("total rows                    :", c.fetchone()[0])
pg.close()
