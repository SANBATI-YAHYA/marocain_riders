"""Quick smoke test: DB tables + JSON ingestion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import get_connection
from app.core.config import get_settings

s = get_settings()
conn = get_connection(s.sqlite_db_path)
cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

# Quick JSON ingestion test
from app.api.dependencies import get_json_ingestion_service
svc = get_json_ingestion_service()
counts = svc.ingest(s.json_kb_path)
print("Ingestion counts:", counts)

# Verify data
cur2 = conn.execute("SELECT COUNT(*) FROM places")
print("Places in DB:", cur2.fetchone()[0])
cur3 = conn.execute("SELECT COUNT(*) FROM routes")
print("Routes in DB:", cur3.fetchone()[0])
