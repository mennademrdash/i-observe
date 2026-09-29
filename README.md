# I-Observe

Local face-recognition and video-observation demo. The active face path is:

`YOLO11 Face (bbox + five landmarks) → affine alignment → MiniFASNetV2 → ArcFace 512-D ONNX → TurboVec → identity`

The FastAPI UI, live camera API, enrollment API, video analyzer, and Colab notebook use the same model family. Enrollment stores one ArcFace template per accepted image, and each detected face is processed independently. PostgreSQL remains the event timeline; Qdrant is not used by the active face-index path.

## Start the project on Windows

```powershell
cd E:\I-observe
.\.venv\Scripts\Activate.ps1
.\start_iobserve.ps1
```

Open `http://localhost:8000`. `.env` must contain the existing PostgreSQL/storage configuration. YOLO, ArcFace, MiniFASNet, and TurboVec weights/index are read from `models/` and `data/`.

To enroll a person, open **Enroll Person** in the web UI, enter a name, and upload clear images containing exactly one face each. The local camera viewer is optional:

```powershell
.\.venv\Scripts\python.exe app.py
```

It opens the camera first, then runs the same YOLO11/MiniFASNet/ArcFace/TurboVec inference path. Press `q` or `Esc` to exit; enrollment remains in the web UI.

## Colab notebook

Open `face_recognition_pipeline_colab.ipynb` in Google Colab. Upload these existing project weights when prompted:

- `models/yolo11n-pose_widerface.pt`
- `models/auraface_glintr100.onnx`
- `models/minifasnet_v2.onnx`

Colab runs in an isolated runtime: its TurboVec index, events, and media do not connect to this Windows app unless separately shared/mounted. The notebook is a portable inference demo, not a remote app backend.

## Model and security limitations

- `ARCFACE_MATCH_THRESHOLD` is configurable; the current `0.25` value is a preliminary fit to six MENNA enrollment photos and ten sampled LFW images, not a validated security threshold.
- The current ArcFace ONNX weights are marked **InsightFace non-commercial research only**. Confirm rights before commercial use.
- MiniFASNetV2 currently produces a false-live result for at least one still enrollment photo in local API testing. Do not use this anti-spoof path as an access-control security gate until it passes a representative live/spoof benchmark.
