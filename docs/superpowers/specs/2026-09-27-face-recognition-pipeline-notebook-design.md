# Design Spec — Colab Face Recognition Pipeline Benchmark Notebook

**Date:** 2026-09-27
**Status:** Awaiting partner review
**Deliverable:** `E:\I-observe\face_recognition_pipeline.ipynb`
**Runtime target:** Google Colab free tier (T4, 12 GB VRAM, ~7 GB disk)

---

## 1. Purpose

Produce one Colab-ready notebook that runs a 6-stage face recognition pipeline over
video and **empirically determines which open-source library is best at each stage**,
while keeping the default configuration commercially license-clean.

The notebook answers "which library" with measured numbers, not opinion. It also
transparently flags the accuracy that the clean-license choice costs.

### Success criteria

1. Paste into Colab → Run All → all tiers execute top to bottom with no manual edits.
2. A comparison table ranks every detector×embedder candidate on rank-1 accuracy,
   verification EER, latency, VRAM, and license.
3. The default pipeline uses only commercially-usable components.
4. Every printed metric is cross-checked against a published reference value.
5. Output: annotated video, identity-per-track table, and cosine-score distributions.

### Non-goals

- Training or fine-tuning any model.
- Production deployment (auth, TLS, rate limiting, multi-tenant isolation).
- Perceptual-hash image search (a different problem, not identity recognition).
- 3D face reconstruction / mesh output.
- Any LFW/CelebA model *training* data download.

---

## 2. License position (verified against primary sources, 2026-09-27)

| Component | License | Primary source | Commercial use |
|---|---|---|---|
| AuraFace-v1 weights | Apache-2.0 | HF model card `license: apache-2.0` | Yes |
| Silent-Face-Anti-Spoofing / MiniFASNet | Apache-2.0 | repo `LICENSE`, "Copyright 2020 Minivision" | Yes |
| OpenCV Zoo (YuNet, SFace) | Apache-2.0 | `opencv_zoo/LICENSE` | Yes |
| InsightFace **code** | MIT | InsightFace `README.md` | Yes |
| InsightFace **weights** (SCRFD, `w600k_r50`, `det_*.onnx`) | **Non-commercial research only** | InsightFace `README.md` + 2025-11-24 addendum | **No** |

Verbatim from InsightFace's README:

> The code of InsightFace is released under the MIT License. There is no limitation for
> both academic and commercial usage. The training data containing the annotation (and the
> models trained with these data) are available for non-commercial research purposes only.
> Both manual-downloading models from our github repo and auto-downloading models with our
> python-library follow the above license policy (which is for non-commercial research
> purposes only).
>
> 2025-11-24 Update: For open-sourced face recognition models (e.g., buffalo_l package),
> please contact recognition-oss-pack@insightface.ai for licensing.

### Consequences that shape the architecture

- `FaceAnalysis(name="auraface")` silently downloads InsightFace `det_*.onnx` weights.
  Using it would taint an otherwise-clean chain. The notebook therefore **composes
  detector and embedder independently** and never calls the bundled `FaceAnalysis` shortcut
  for the default path.
- SCRFD is research-only, so the clean stage-1 detector is **YuNet**, not the diagram's
  suggested SCRFD.
- The default chain is **YuNet + AuraFace-v1 + MiniFASNetV2 + Qdrant**. All Apache-2.0.
- Research-only models stay in the notebook **for benchmark comparison only**, clearly
  labelled, never in the default path.

### Bias disclosure (AuraFace model card, verbatim)

> The efficacy of the model in subject preservation may vary on the basis of ethnicity.
> The generalization of the models is limited due to limitations of the training data.

AuraFace is trained on a commercial dataset which "may not extensively cover all
ethnicities." The notebook must surface this in its limitations section and the LFW tier
must report the demographic composition caveat alongside accuracy.

### Anti-spoofing strength (independent benchmark, 2026)

Silent-Face-Anti-Spoofing (MiniFASNet) reported ACER 38.2% / APCER 73.3% by an independent
2026 evaluation. Its upstream self-reported figures are stronger. The notebook must state
that stage 4 is a **demo-grade gate, not a security control**, and must not present
published ACER/BPCER numbers as measurements it produced itself.

---

## 3. Model roster

### Clean chain (default)

| Stage | Model | Source | Dim |
|---|---|---|---|
| 1 Detect | YuNet `face_detection_yunet_2023mar.onnx` | OpenCV Zoo (Apache-2.0) | 5 kps |
| 2 Align | Own `estimateAffinePartial2D` + `warpAffine` | OpenCV / numpy | — |
| 3 Preprocess | 112×112 crop, `(x/127.5) - 1.0`, L2-norm | numpy | — |
| 4 Liveness | MiniFASNetV2 ONNX | Silent-Face repo (Apache-2.0) | scalar |
| 5 Embed | AuraFace-v1 `backbone.onnx` | `fal/AuraFace-v1` (Apache-2.0) | 512 |
| 6 Vector | Qdrant server | qdrant (Apache-2.0) | 512 |

### Benchmark-only candidates

| Candidate | Stage | License | Role |
|---|---|---|---|
| SCRFD `det_2.5g` | 1 | research-only | accuracy ceiling for detection |
| RetinaFace-Mnet | 1 | Apache-2.0 / MIT | clean third detector |
| BlazeFace (MediaPipe short-range) | 1 | Apache-2.0 | CPU-only reference |
| ArcFace `w600k_r50` (buffalo_l) | 5 | research-only | accuracy ceiling for embedding |
| SFace | 5 | Apache-2.0 | clean lightweight embedder (128-d) |
| Milvus Lite | 6 | Apache-2.0 | vector-engine comparison |
| PGVector (in-process HNSW) | 6 | PostgreSQL license | vector-engine comparison |

### Embedder alignment caveat

ArcFace and SFace share the canonical ArcFace 5-point template, so the same alignment
routine serves all of them. Each candidate declares its own input size and normalization;
AuraFace follows ArcFace convention.

---

## 4. Architecture

Single notebook, sequential cells. Stages are plain functions in module-level cells so
that the same code powers the LFW tier, the video tier, and the benchmark tier.

```
§0  Header: purpose, license matrix, bias disclosure, how to run
§1  Config            MODEL_PROFILE, DETECTOR, EMBEDDER, VIDEO_SRC, thresholds
§2  Environment       pip install, GPU assert, version report
§3  Qdrant server     .deb fetch -> dpkg -x -> subprocess -> health ping
§4  Stage 1 Detect    YuNet / SCRFD / RetinaFace behind one interface
§5  Stage 2+3 Align   estimateAffinePartial2D -> warpAffine 112 -> normalize
§6  Stage 4 Liveness  MiniFASNetV2 ONNX, hard gate before embedding
§7  Stage 5 Embed     AuraFace / ArcFace / SFace behind one interface
§8  Tracker           IoU + cosine affinity, greedy assign, max-age drop
§9  Gallery builder   track centroid -> Qdrant upsert (video tier)
§10 LFW tier          1:N protocol, rank-1/rank-5, EER from pairs.txt
§11 Benchmark tier    accuracy + latency + VRAM + license table
§12 Vector tier       Qdrant vs Milvus Lite vs PGVector, HNSW recall@10
§13 Video tier        frame loop, annotate, write mp4, identity table
§14 Limitations       license, bias, stage-4 strength, what was NOT measured
```

### Component interfaces

```python
class Detector(Protocol):
    def detect(self, bgr: np.ndarray) -> list[FaceBox]: ...

@dataclass
class FaceBox:
    bbox: np.ndarray      # (4,) xyxy float32
    kps: np.ndarray       # (5, 2) float32 — left eye, right eye, nose, left mouth, right mouth
    score: float

class Embedder(Protocol):
    dim: int
    def embed(self, aligned_112: np.ndarray) -> np.ndarray: ...  # L2-normalized

class LivenessDetector(Protocol):
    def score(self, bgr: np.ndarray, box: FaceBox) -> float: ...  # >0.5 = live
```

The tracker consumes `FaceBox` + `embedding` and emits `track_id`. Nothing downstream of
stage 5 knows which detector or embedder produced its input.

---

## 5. Stage-by-stage specification

### Stage 1 — Detection

- **YuNet** via `cv2.FaceDetectorYN_create`. Requires `setInputSize((w, h))` per frame size;
  the notebook caches sessions keyed on frame dimensions to avoid reallocating.
- **SCRFD** via InsightFace `model_zoo.get_model`. Benchmark-only.
- **RetinaFace** via `insightface` `RetinaFace` wrapper. Benchmark-only.
- **BlazeFace** via MediaPipe, CPU reference only. Benchmark-only.
- Configurable `det_thresh` (default 0.5) and NMS IoU (default 0.4).
- Detector latency measured on the **same** frame batch across all candidates, with a
  warm-up pass excluded from timing.
- **Known limitation:** a single forward pass is not scale-invariant. Small faces in a wide
  shot will be missed. Documented, not silently ignored. Mitigation shown as an optional
  tiled/downscaled second pass, off by default.

### Stage 2 — Alignment

- Canonical ArcFace 5-point template (`arcface_112x112` / `insightface` reference points).
- `cv2.estimateAffinePartial2D(src_kps, template, method=cv2.LMEDS)`.
- `cv2.warpAffine(..., flags=INTER_LINEAR, borderMode=BORDER_CONSTANT)` to 112×112.
- Degenerate-pose guard: if the fitted matrix has near-zero determinant or extreme
  perspective terms, skip the face and count it in `stats.degenerate`.
- **Written explicitly, never delegated to `app.get()`**, so the notebook maps 1:1 onto
  the user's diagram.

### Stage 3 — Preprocessing

- RGB conversion, `(x / 127.5) - 1.0`, `float32`, `NCHW`.
- L2-normalization applied to the *output embedding*, not the input crop.

### Stage 4 — Liveness

- MiniFASNetV2 ONNX. Weights are **not** distributed as a release asset in the
  Silent-Face repo; the upstream flow is to clone the repo and run its own weight
  conversion, or fetch a third-party ONNX export. The notebook pins to one documented
  third-party ONNX export and records its SHA-256 in the notebook header, so the exact
  artifact in use is never ambiguous. If that artifact becomes unavailable, the cell
  asserts with instructions for the repo's own conversion path rather than silently
  falling back to a different model.
- **Input size is read from the session, not hardcoded.** MiniFASNet variants differ
  (V1 80×80, V2 80×80, V3/V4 80×80, but SE variants 64×64). The cell asserts
  `session.get_inputs()[0].shape` and reshapes accordingly, so a size mismatch surfaces
  as a loud error instead of a plausible-looking wrong score.
- Operates on a face crop expanded to roughly 2.0× the detected box.
- Hard gate: spoof candidates never reach stage 5. Per-frame counts of
  `live / spoof / no_face` are accumulated and reported.
- **No accuracy number is produced by this notebook.** Public spoof corpora (SiW,
  OULU-NPU, Replay-Attack) are research-gated and are not downloaded. The notebook prints
  the independent published ACER/BPCER figures with attribution and labels them
  **third-party, not measured here**.
- A self-referential smoke test (a photo of a screen re-fed to the pipeline) is shown as a
  qualitative indication only, clearly labelled as not a security evaluation.

### Stage 5 — Embedding

- Bare `onnxruntime` session per candidate; no `FaceAnalysis` wrapper.
- Output L2-normalized to unit length.
- `normed_embedding` equivalence: the notebook recomputes the norm and asserts it is
  within `1e-4` of 1.0, catching silent misalignment or wrong preprocessing.

### Stage 6 — Vector search

**Qdrant in Colab without Docker.** Colab has no Docker daemon. Qdrant publishes a
`.deb` on GitHub releases. The notebook:

1. Resolves the latest release asset URL via the GitHub API.
2. Downloads the `.deb`.
3. `dpkg -x qdrant.deb /tmp/qdrant` (extract without install; no root needed).
4. Launches the binary as a `subprocess` on `localhost:6333` with `--config-path`
   pointing at a generated minimal config (storage dir + snapshots dir in `/content`).
5. Polls `GET /healthz` until ready, with a timeout and a clear failure message.
6. Teardown at the end of the notebook: terminate the subprocess, remove storage.

Collection: one per (embedder) so candidates do not collide, sized 512 for the clean
chain. Cosine metric. HNSW with explicit `m` and `ef_construct`, so the
`recall@10`-vs-exact comparison is meaningful.

- **Milvus Lite** and **PGVector** comparison measures insert + top-k query latency only.
- **HNSW recall@10 vs exact brute force** is reported, so the ANN approximation cost is
  visible rather than assumed.

### Tracking (video tier only)

- Greedy assignment per frame: cost = IoU gate AND cosine affinity above threshold.
- Track embedding = quality-weighted centroid, weights = Laplacian variance × face
  area × liveness score. Re-normalized on every update.
- Max-age drop (default 30 frames) so a person leaving and returning gets a fresh track.
- No `supervision` / `boxmot` dependency; ~100 lines, fully inspectable.

---

## 6. The three benchmark tiers

A benchmark needs ground truth. An auto-built video gallery has none. So the notebook
splits measurement into three tiers with different claims.

### Video resolution (video tier)

`VIDEO_SRC` accepts, in priority order: a local path already in `/content`, a Google Drive
path if the user mounts it, or a direct URL. The cell resolves it, asserts the file opens,
asserts `CAP_PROP_FRAME_COUNT > 0` **and** that at least one frame actually decodes, and
prints source, FPS, frame count, and duration. A stream whose container reports frames it
cannot deliver is a real and common failure, and frame-count metadata alone will not catch
it.

- **No bundled default clip.** The user uploads their own via Colab's file uploader, per
  partner decision. The cell prints a one-line hint showing exactly where the file landed.

### Tier 1 — LFW (real accuracy, has ground truth)

- `lfw.tgz` (~200 MB) from the UMass LFW distribution. Documented download command.
- 13,233 images / 5,749 identities.
- **Identification protocol** matching the pipeline's own 1:N use:
  gallery = 1 image per identity for the 1,680 identities with ≥2 images;
  probe = every remaining image. Cosine similarity, no re-ranking.
- **Verification protocol**: standard 10-fold split over the 6,000 official pairs.
  Report EER.
- Every detector×embedder combination is run: detectors `{YuNet, SCRFD, RetinaFace}`,
  embedders `{AuraFace, ArcFace-buffalo_l, SFace}`. 9 combinations, minus any that
  error out (recorded in the table, not silently dropped).
- **Reference check**: AuraFace's published LFW is 0.99650. The notebook asserts the
  measured value lands within a stated tolerance of that, and prints a loud warning
  instead of a green checkmark if it disagrees. A large discrepancy means the
  alignment or preprocessing is wrong, not that the model is different.
- **Demographic caveat printed alongside**: LFW skews heavily toward one demographic,
  so LFW rank-1 overstates real-world performance. The notebook says so.

### Tier 2 — Video (no ground truth, label-free metrics only)

- Gallery auto-built from video track centroids (§9).
- Report **only** metrics that need no labels:
  - silhouette score and Davies-Bouldin index per cluster
  - intra-cluster vs inter-cluster cosine distributions (mean, std, and a 2-sample
    separation measure)
  - cluster-count stability across bootstrap resamples
  - FPS per stage, and the per-frame live/spoof/no-face tally
- **No accuracy, no precision/recall.** Anything else would be fabricated.

### Tier 3 — Systems metrics (always runs, and often the deciding factor)

| Metric | Method |
|---|---|
| Stage latency | warm-up pass excluded; median + p95 over the same frame batch |
| Peak VRAM | `torch.cuda.max_memory_allocated` equivalent — `nvidia-smi` sampled, or ORT's memory arena report |
| Model size | on-disk bytes per ONNX |
| Download size | total bytes pulled |
| License | from the verified matrix in §2, carried in code as a table |

Peak VRAM measurement: `onnxruntime` does not expose a CUDA allocator API, so the notebook
samples `nvidia-smi --query-gpu=memory.used` around the timed region rather than
pretending to use a torch API. Stated plainly in a comment.

### Liveness accuracy — deliberately absent

No tier measures stage-4 accuracy. Public spoof corpora are research-gated. The notebook
publishes the independent ACER/BPCER figures with attribution and states they are
**third-party, not measured here**.

---

## 7. Output

- Annotated MP4: box, track id, identity name, cosine score, liveness colour
  (green = live, red = spoof, grey = no face).
- Identity table: `track_id | centroid_frame | n_samples | top identity | score | live_pct`
- Cosine score histogram: intra-cluster vs inter-cluster, with the open-set threshold
  drawn as a vertical line.
- Benchmark table (Tier 1 + 3) and vector-engine table (Tier 3).
- Limitations section printed as the final cell.

### Open-set decision

The diagram's "Top Matches + Distance/Similarity Scores" implies rejection is possible.
So a threshold is required. The notebook:

- Reports the score distribution per candidate embedder.
- Sets a default open-set threshold of **cosine 0.28** for the clean chain, labelled as
  an **initial guess requiring calibration on your own data** — not a validated value.
- Prints a warning when the best match falls below it.
- Exposes `OPEN_SET_THRESHOLD` in config so it can be recalibrated once a labelled set exists.

A hard-coded threshold that is not calibrated against negatives is exactly the kind of
thing that makes these notebooks quietly wrong. It is labelled as such.

---

## 8. Error handling

Every stage fails loud. Silent zero-detection is the single most common way a face
pipeline notebook reports success while being broken.

| Condition | Behaviour |
|---|---|
| No CUDA device | assert with the Colab GPU-enable instructions |
| Model download fails | assert with the URL, HTTP status, and a retry hint |
| Qdrant `.deb` release asset missing | assert, print the resolved release JSON keys |
| Qdrant fails to become healthy in 60 s | assert with the subprocess stderr tail |
| Video fails to open / has 0 decodable frames | assert with the resolved path and codec |
| Zero faces detected in a whole frame batch | warn loudly, print the count and the frame indices |
| Degenerate affine fit | skip the face, increment `stats.degenerate` |
| Spoof detected | hard gate, face never embedded, counted |
| Candidate model raises on load | record the exception in the benchmark table, keep going |
| Metric disagrees with published reference | warn with both numbers, do not fake a pass |

---

## 9. Verification strategy

The notebook is verified by the numbers it produces, checked against external references.

| Check | Reference | Tolerance |
|---|---|---|
| AuraFace LFW rank-1 | 0.99650 (HF model card) | ±0.005 |
| ArcFace buffalo_l LFW | ~0.9965 (widely reported) | ±0.01, advisory only |
| Alignment output | visually inspected on a grid of poses; landmark overlay cell | manual |
| Embedding norm | 1.0 | 1e-4 |
| LFW pair count | 6,000 pairs / 10 folds × 600 | exact |
| LFW identities with ≥2 images | 1,680 | exact |
| Qdrant recall@10 | ≥0.99 at default HNSW params | assert, tunable |
| Cosine self-similarity | 1.0 for identical image | 1e-3 |
| Distinct-face cosine | well below threshold | qualitative |

**The alignment cell is the highest-risk component.** If `estimateAffinePartial2D` or the
template is wrong, every downstream metric is wrong in a way that still looks plausible.
It gets its own visual inspection cell plus a synthetic-pose test: known rotation applied
to a face should recover the inverse rotation from the fitted matrix.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Qdrant `.deb` release naming changes | Stage 6 breaks | GitHub API asset discovery, not a hardcoded URL; explicit assertion listing available assets |
| `insightface` pip build needs a C++ toolchain on new Colab images | Install fails | Prefer wheel; if source build is attempted, install `g++ cmake` first, and fail with the real build log |
| LFW URL rots | Tier 1 fails | Try UMass primary then a documented mirror; assert clearly if both fail |
| Silent-Face ONNX input size mismatch across variants | Silent garbage scores | Read the input shape from the session and reshape; never hardcode |
| MiniFASNet ONNX artifact unavailable (no upstream release asset) | Stage 4 cannot run | Pin one third-party ONNX export + SHA-256 in the header; on failure assert with the repo's own conversion path |
| T4 VRAM pressure with 3 detectors + 3 embedders loaded | OOM mid-run | Load one candidate at a time, free in a `finally`, report per-candidate peak |
| Colab session disconnect on long Run All | Lost work | Cache downloaded models/artifacts to `/content` and check for existence before re-downloading |
| Alignment subtle bug | All metrics quietly wrong | Visual cell + synthetic-pose recovery test (§9) |
| Benchmark conclusion depends on a noisy single dataset | Over-claiming | State the LFW demographic skew; do not extrapolate to production accuracy |

---

## 11. Explicitly out of scope

- Training or fine-tuning any model.
- Docker-in-Colab (no daemon available).
- GPU-accelerated video decoding (`nvdec`) — CPU decode is sufficient at T4 for this scale.
- Multi-camera or multi-stream tracking.
- Gender, age, or emotion inference.
- Any claim about liveness-detection accuracy.

---

## 12. Decision log

| Decision | Choice | Why |
|---|---|---|
| Embedder default | AuraFace-v1 | Only high-accuracy embedder with genuinely commercial-clean weights |
| Detector default | YuNet | InsightFace detector weights are research-only; YuNet is Apache-2.0 |
| Compose vs `FaceAnalysis` | Compose | `FaceAnalysis` silently pulls research-only detector weights |
| Landmarks | 5-point | ArcFace alignment needs 3; 5 is canonical. 68-point costs ~2x for no gain here |
| Vector store | Qdrant via `.deb` + subprocess | Real server, real gRPC/REST, no Docker, no cloud account |
| Benchmark on auto-built gallery | Split into 3 tiers | Auto-built gallery has no labels; accuracy claims would be fabricated |
| Liveness accuracy | Not measured | Public corpora research-gated; publishing third-party numbers as our own would be dishonest |
| Tracking | Hand-rolled, no dependency | ~100 lines, inspectable, avoids `boxmot` API churn |
| `MODEL_PROFILE` | `clean` default, `research` opt-in | Partner decision: default to the license-clean chain |
| Video source | User upload | Partner decision: no fragile yt-dlp, no licensing ambiguity |
