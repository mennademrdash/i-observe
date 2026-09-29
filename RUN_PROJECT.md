# RUN_PROJECT.md — I-observe Face Recognition & Video Event Pipeline

Every command below was taken from the current repository. Run from `E:\I-observe`
in PowerShell.

---

## 1. First-time setup

```powershell
cd E:\I-observe
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Secrets live in `E:\I-observe\.env` (already present, gitignored). Do not commit it.

## 2. Activate .venv

```powershell
cd E:\I-observe
.\.venv\Scripts\Activate.ps1
```

## 3. Required environment variables

Read automatically from `.env` by `config.py`. You only need to export one extra var,
from the process environment only:

| Variable | Required for | Source |
|---|---|---|
| `DATABASE_URL` | PostgreSQL mirror | `.env` |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` / `MINIO_BUCKET` | evidence upload | `.env` |
| `QDRANT_PATH` | face index (blank `QDRANT_URL` = embedded mode) | `.env` |
| `VLM_PROVIDER` / `VLM_BASE_URL` / `VLM_MODEL` | VLM selection | `.env` |
| `OPENAI_API_KEY` | **only** when `VLM_PROVIDER=openai` | process env, never `.env` |
| `API_TOKEN` | optional bearer auth on the query API | process env |

## 4. Start required infrastructure

PostgreSQL + Qdrant (Docker):

```powershell
cd E:\I-observe
docker compose up -d postgres qdrant
docker ps --filter "name=iobserve"
```

Evidence object storage (S3-compatible, MinIO SDK):

```powershell
cd E:\I-observe
Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList '-m','moto.server','-p','9000' -WindowStyle Hidden
Invoke-WebRequest -Uri http://localhost:9000/ -UseBasicParsing   # expect HTTP 200
```

## 5. Start application

The pipeline is a CLI. The HTTP API is optional:

```powershell
cd E:\I-observe
.\.venv\Scripts\python.exe -m uvicorn query_api:app --host 127.0.0.1 --port 8000
```

Endpoints: `POST /query`, `GET /events`, `GET /health`, `GET /metrics`,
`POST /enroll`, `POST /identify`, `POST /events/publish`, `WS /ws/incidents`.

## 6. Camera / face enrollment demo

Webcam app (SFace, OpenCV Zoo — separate from the 512-D pipeline):

```powershell
cd E:\I-observe
.\.venv\Scripts\python.exe app.py
```

Press `e` to enroll, `r` to reload, `q` / `Esc` to quit.

Enroll a person into the 512-D-equivalent pipeline index (SFace 128-D):

```powershell
.\.venv\Scripts\python.exe enroll_people.py menna data\people\menna\*.jpeg
```

## 7. Face identification demo

```powershell
cd E:\I-observe
.\.venv\Scripts\python.exe analyze_face_video.py data\sample_action.mp4 --output outputs\annotated.mp4
```

Writes `outputs\annotated.mp4` and `outputs\annotated.json`. Boxes are green for an
enrolled identity, cyan for unknown, red for spoof-rejected.

## 8. Liveness demo

Included in steps 6 and 7 — MiniFASNetV2 gates every face before embedding.
Force a spoof report:

```powershell
cd E:\I-observe
.\.venv\Scripts\python.exe analyze_face_video.py data\sample_action.mp4 --output outputs\liveness.mp4
```

`spoof_rejected` and `live` counts appear in the printed summary and in the JSON.

## 9. Menna video demo

```powershell
cd E:\I-observe
Remove-Item -ErrorAction SilentlyContinue events.sqlite
.\.venv\Scripts\python.exe analyze_face_video.py data\sample_action.mp4 --output outputs\menna.mp4
.\.venv\Scripts\python.exe -c "import json;d=json.load(open('outputs/menna.json'));print('identities:',d['identities']);print('identified:',d['identified_faces'])"
```

## 10. VideoMAE / action demo

```powershell
cd E:\I-observe
.\.venv\Scripts\python.exe pipeline_v2.py data\sample_action.mp4 --db events.sqlite --no-vlm
```

First run downloads ~346 MB to `.hf-cache`. Look for `"videomae": {"ok": true}` and
`"videomae_windows": N > 0` in the printed report. Actions land in the `action` /
`action_score` columns.

## 11. Natural-language query demo

```powershell
cd E:\I-observe
.\.venv\Scripts\python.exe pipeline_v2.py data\sample_action.mp4 --db events.sqlite --no-videomae --no-vlm --question "who was wearing black at 00:05?"
```

Or over HTTP:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/query -Method Post `
  -ContentType 'application/json' -Body '{"question":"who was wearing black at 00:05?"}'
```

## 12. Evidence image / video demo

Incident rules are in `incident_rules.json`. Matching events auto-extract a clip.

```powershell
cd E:\I-observe
Get-ChildItem evidence -Recurse -File | Select-Object -First 10   # annotated JPEGs
Get-ChildItem clips -Filter *.mp4                                 # evidence clips
Get-Content evidence\sample_action\manifest.json | Select-Object -First 20
```

Verify the clips actually play:

```powershell
.\.venv\Scripts\python.exe -c "import cv2,glob;[print(f, cv2.VideoCapture(f).isOpened()) for f in glob.glob('clips/*.mp4')]"
```

## 13. Database / storage verification

```powershell
cd E:\I-observe
.\.venv\Scripts\python.exe tools\pg_check.py      # row counts in PostgreSQL
.\.venv\Scripts\python.exe -c "import config;from pipeline_v2 import EventStore;s=EventStore(config.event_db());print('sqlite rows:',len(s.all_events()))"
docker exec iobserve-postgres pg_isready -U observe -d observe
.\.venv\Scripts\python.exe -c "from services import ObjectStorage;o=ObjectStorage();print('s3 objects:',len(list(o.client.list_objects(o.bucket,recursive=True))))"
```

## 14. OpenAI VLM startup / test

Ollama and OpenAI are separate. Pick one.

```powershell
cd E:\I-observe
# OpenAI: key must come from the process environment, never .env
$env:OPENAI_API_KEY = "<your key>"
$env:VLM_PROVIDER   = "openai"
$env:VLM_BASE_URL   = "https://api.openai.com/v1"
$env:VLM_MODEL      = "gpt-4o-mini"
.\.venv\Scripts\python.exe -c "import config;from ai_models import OpenAICompatibleVLM;print(OpenAICompatibleVLM().ping())"
.\.venv\Scripts\python.exe pipeline_v2.py data\sample_action.mp4 --db events.sqlite --no-videomae
```

Confirm `"vlm": {"ok": true}` and a non-null `vlm` column in the events.

To clear the key afterwards:

```powershell
Remove-Item Env:\OPENAI_API_KEY
```

## 15. Fast health check

```powershell
cd E:\I-observe
docker ps --filter "name=iobserve"
Invoke-RestMethod http://127.0.0.1:8000/health
.\.venv\Scripts\python.exe -c "import config;print(config.describe())"
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 16. Safe shutdown

```powershell
cd E:\I-observe
# stop the API if running
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*I-observe*' } | Stop-Process
# stop the S3 server and any other project python
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*moto.server*' } | ForEach-Object { Stop-Process -Id $_.ProcessId }
# stop project containers ONLY (does not touch office-security-app)
docker compose down
```

Leave the shared machine's PostgreSQL 18 (port 5432) and the `office-security-app`
stack alone — this project never modifies them.

---

## === START EVERYTHING ===

```powershell
cd E:\I-observe
.\.venv\Scripts\Activate.ps1
docker compose up -d postgres qdrant
Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList '-m','moto.server','-p','9000' -WindowStyle Hidden
Start-Sleep -Seconds 8
docker ps --filter "name=iobserve"
.\.venv\Scripts\python.exe enroll_people.py menna data\people\menna\*.jpeg
Remove-Item -ErrorAction SilentlyContinue events.sqlite
.\.venv\Scripts\python.exe pipeline_v2.py data\sample_action.mp4 --db events.sqlite --no-vlm
```

## === PRESENTATION DEMO ===

```powershell
cd E:\I-observe
.\.venv\Scripts\Activate.ps1
docker compose up -d postgres qdrant
Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList '-m','moto.server','-p','9000' -WindowStyle Hidden
Start-Sleep -Seconds 8
.\.venv\Scripts\python.exe enroll_people.py menna data\people\menna\*.jpeg
Remove-Item -ErrorAction SilentlyContinue events.sqlite
.\.venv\Scripts\python.exe pipeline_v2.py data\sample_action.mp4 --db events.sqlite --no-vlm --question "who was wearing black at 00:05?"
```

Shortest reliable demo: two commands after the infrastructure is up —
`enroll_people.py` then `pipeline_v2.py --question`. The printed report shows
`processed_frames`, `events`, `evidence_clips`, `videomae_windows` and a per-integration
`ok` block; the answer shows the parsed time window and the evidence image/clip paths.
