# ASL Vision — Project Plan

**Video-Based American Sign Language Recognition Using 3D Convolutional Neural Networks**

This plan covers building, testing, and deploying (at no cost) a Streamlit web app that showcases the team's R3D_18 ASL recognition research. The full requirements are in [prompt.txt](prompt.txt). Progress is tracked in [STATUS.md](STATUS.md).

> **Revision 2026-11-19 (after inspecting the 1200-class notebook):** The deployed-track models are
> **not R3D_18**. They are the team's v10 *Geometry-Only Transformer* on MediaPipe landmarks (ASL Citizen;
> 300/500/1000/1200 = class counts). R3D_18 was in the older dual-stream v9 model. The scope now also
> includes two new 1000-class architectures (Video Swin-T on RGB, ST-GCN on landmarks), trained by the
> notebooks in [training_notebooks/](training_notebooks/). Inference must therefore reproduce the
> **cache-building pipeline** (MediaPipe extraction + signer crop), which still needs to be obtained.
> Where §5–§7 below say "R3D_18", read "the selected model's own preprocessing". See [STATUS.md](STATUS.md).

---

## 1. Guiding Rules

These come straight from the requirements and override any convenience shortcut:

1. **Do not fabricate anything.** Every metric, class name, statistic, and prediction shown must come from the team's actual artifacts. If a value is missing, the app says so.
2. **Reproduce the training preprocessing exactly.** Frame count, sampling, resize/crop, normalization, and tensor order must match the notebooks.
3. **Spend nothing.** Use only free services: GitHub, Hugging Face Hub (public model repo), and Streamlit Community Cloud.
4. **Run on CPU only.** No CUDA packages and no GPU assumptions.
5. **Use little memory.** Keep only one model in memory at a time.
6. **Do not modify the model weights.** No retraining and no substitution with ImageNet/Kinetics weights.
7. **Do not overengineer.** No auth, databases, user accounts, or extra APIs.
8. **Describe the system accurately.** This is *isolated sign classification from a video clip*, not continuous real-time translation.

---

## 2. Current State (as of 2026-11-19)

| Item | Status |
|---|---|
| Spec ([prompt.txt](prompt.txt)) | Present |
| Kaggle notebook links ([text.txt](text.txt)) | 300, 500, and 1200 (the 1200 link is listed twice). **1000 is missing.** |
| Notebook source code | Not available locally. Kaggle pages are JS-rendered, and no Kaggle API token is configured. |
| Trained checkpoints | **Not present** |
| Class mappings / metrics / confusion matrices | **Not present** |
| Local Python environment | Python 3.13.5 (Anaconda). PyTorch is **not installed**. |

**Phase 1 is blocked until the inputs in §3 are provided.**

---

## 3. Required Inputs (from the team)

| # | Input | How to provide | Needed for |
|---|---|---|---|
| I1 | All four notebooks (`.ipynb`), **including the 1000 run** | Kaggle → *File → Download Notebook*, saved to `source_notebooks/`. Alternatively, set up a Kaggle API token at `~/.kaggle/access_token` (never paste it in chat or commit it). | Architecture, preprocessing, classes, metrics |
| I2 | Four trained checkpoints | Kaggle notebook → *Output* tab → download into `models/` (local only, not committed) | Inference |
| I3 | Class / label mapping files, if saved separately (JSON, CSV, `LabelEncoder` pickle, etc.) | Put in `source_artifacts/` | Correct class names |
| I4 | Evaluation outputs: metrics, classification reports, confusion matrices, training history, timing | Put in `source_artifacts/`. If they were only printed in notebook outputs, I1 covers them. | Comparison page |
| I5 | Dataset name and source (e.g. WLASL, MS-ASL, or custom), split sizes, license | Usually recorded in the notebooks; otherwise the team supplies them | Overview / About pages |
| I6 | Team info: names, roles, university, supervisor, GitHub URL | Fill in `config/team.json`; placeholders are fine until then | About page |
| I7 | A few short test videos with known labels, ideally held-out clips from the dataset | Put in `samples/` (local only unless the license allows sharing) | Testing and demo |

---

## 4. Target Project Structure

```
project_capstone_c/
├── app.py                      # Streamlit entry point: navigation, theme, and page routing
├── pages_impl/                 # One module per page (kept out of Streamlit's auto "pages/" dir)
│   ├── overview.py
│   ├── recognition.py
│   ├── comparison.py
│   ├── methodology.py
│   └── about.py
├── inference/
│   ├── config.py               # Per-model config: num_classes, frames, size, mean/std, HF file
│   ├── model_loader.py         # Build r3d_18, download and validate the checkpoint, cached load
│   ├── video_processor.py      # Read/validate video, sample frames, transform → tensor
│   └── predictor.py            # Run inference, softmax, top-k, timing
├── ui/
│   ├── theme.py                # CSS injection (navy/white/blue), card components
│   └── charts.py               # Plotly figure builders
├── results/
│   ├── metrics.json            # Real metrics per model, extracted from the notebooks
│   ├── confusion/              # Confusion matrices (.npy/.json) if available
│   └── class_maps/             # class_index → name JSON, one per model
├── config/
│   └── team.json               # Editable team and repo info
├── assets/
│   ├── logo.png
│   └── architecture.png        # Or a Plotly/SVG diagram generated in code
├── models/
│   └── README.md               # Storage instructions; checkpoints are git-ignored
├── scripts/
│   ├── extract_from_notebooks.py   # Parse .ipynb → metrics.json and class maps
│   ├── upload_to_hf.py             # One-off upload of checkpoints to the HF Hub
│   └── test_inference.py           # CLI test of all four checkpoints
├── tests/                      # Lightweight pytest checks
├── .streamlit/config.toml      # Theme and upload size limit
├── requirements.txt
├── .gitignore
├── README.md
├── PLAN.md
└── STATUS.md
```

The structure will be adjusted to whatever the notebooks reveal (for example, if a custom head class is used, it goes in `inference/model_loader.py`).

---

## 5. Phases

Each phase has tasks and an exit criterion. A phase is not marked complete in STATUS.md until its exit criterion is met.

### Phase 1 — Inspect the Existing Project
**Tasks**
- Read all four notebooks end to end, including their outputs.
- Record the following for each model in an *Inspection Report* (`docs/inspection_report.md`):
  1. Checkpoint filename, size, and format (raw `state_dict`, a dict with `model_state_dict`, or a full pickled model)
  2. Exact architecture: `torchvision.models.video.r3d_18`, pretrained init, head replacement (`fc` → `Linear(512, N)`), dropout, any custom layers
  3. Class count and class names
  4. Class-to-index mapping and how it was built (sorted folder names, `LabelEncoder`, gloss JSON, etc.)
  5. Video reading library (OpenCV / decord / torchvision.io / PyAV)
  6. Frame sampling strategy (uniform linspace, random, center clip, padding or looping for short videos)
  7. Frames per clip (T)
  8. Resize / crop dimensions (H×W) and their order
  9. Color space (BGR→RGB conversion?) and normalization mean/std
  10. Tensor layout fed to the model (expected `[B, C, T, H, W]`)
  11. Evaluation metrics and their averaging method (macro / weighted)
  12. Training history and confusion matrices, if present
- **Determine what 300 / 500 / 1000 / 1200 actually mean.** The notebook titles suggest class counts ("300 Class"), but this must be verified from the code (e.g. it could be the top-N glosses of WLASL/MS-ASL).
- List anything missing or inconsistent across the four runs.

**Exit:** The inspection report is complete. All open questions are either answered or explicitly flagged to the team.

### Phase 2 — Understand Models and Preprocessing
**Tasks**
- Encode the findings in `inference/config.py` as one config entry per model. No values are hard-coded anywhere else.
- Run `scripts/extract_from_notebooks.py` to produce `results/metrics.json` and `results/class_maps/*.json` from the real notebook outputs. Each value records its source (notebook + cell).
- Inspect checkpoint sizes to decide storage (§7).

**Exit:** Config, class maps, and metrics are populated from source. The storage decision is recorded.

### Phase 3 — CPU Inference, Standalone
**Tasks**
- Create a clean environment (Python 3.11 recommended for Streamlit Cloud compatibility) with CPU-only PyTorch.
- `model_loader.py`:
  - Build `r3d_18(weights=None)`, replace the head exactly as in training, and load with `map_location="cpu"`.
  - Handle `module.` prefixes (DataParallel) and nested dict keys.
  - Use `load_state_dict(strict=True)`. On failure, report missing or unexpected keys clearly.
  - Validate that `fc.out_features == len(class_map)`.
  - Call `model.eval()`.
- `video_processor.py`: replicate the training pipeline exactly. Validate readability, frame count, and duration.
- `predictor.py`: run under `torch.inference_mode()`, then softmax, top-k, and wall-clock timing.
- `scripts/test_inference.py`: run every checkpoint on sample videos and print shapes, top-5, and timing.
- **Parity check:** if the notebooks print a prediction or accuracy for specific clips, reproduce it. Otherwise confirm that predictions on held-out samples look sensible.

**Exit:** All four checkpoints load strictly, produce outputs with the right shape, and give reasonable predictions on known samples.

### Phase 4 — Streamlit Interface
**Tasks**
- Theme: navy (`#0B1F3A`), white, and subtle blue accents (`#2E6BD8`, `#EAF1FB`), set through `.streamlit/config.toml` plus injected CSS. Use a clean sans-serif font (Inter via CSS with a system fallback). Metric cards are custom HTML/CSS, and the default Streamlit chrome is hidden.
- Navigation: `st.navigation` / sidebar with five pages and a hero call-to-action button linking to the demo.
- Pages:
  1. **Overview:** hero title and subtitle, description, objectives, R3D_18 summary, the four configurations, dataset description, contributions, verified-only stat cards, and a "Try the Demo" button.
  2. **Recognition:** upload (MP4/AVI/MOV, size limit), model dropdown, video preview, video info (duration, FPS, total frames, frames used), "Predict Sign" button with a spinner, a large predicted-class card with confidence %, and a top-5 Plotly bar chart. Errors are shown as friendly messages.
  3. **Comparison:** metrics table, accuracy bar chart, grouped P/R/F1 chart, best-model highlight, confusion matrix heatmap (shown only if available; for 300+ classes, add a top-confused-pairs view and a zoomable heatmap), and measured inference time. Missing data shows an "unavailable" note.
  4. **Methodology:** pipeline diagram (Video → Frame Extraction → Preprocessing → R3D_18 → Classifier → Sign) and short explanations of why video is used, 3D convolutions, R3D_18 (18-layer residual network with full 3D 3×3×3 convolutions, 4 residual stages, global average pooling, fully connected head), spatio-temporal features, and softmax classification.
  5. **About:** background, objectives, dataset, architecture, configurations, evaluation methodology, team (from `config/team.json`), and GitHub link.

**Exit:** All pages render locally with real data where it exists and honest placeholders where it doesn't.

### Phase 5 — Integrate All Four Models
**Tasks**
- Dropdown driven by `inference/config.py`.
- `@st.cache_resource(max_entries=1)` on the loader keyed by model ID, so switching models evicts the previous one. Call `gc.collect()` after a switch.
- Download the selected checkpoint on demand (§7) with a progress spinner, cached on disk.

**Exit:** Switching between all four models works repeatedly without memory growing.

### Phase 6 — Real Evaluation Results and Visualizations
**Tasks**
- Wire `results/metrics.json` and the confusion matrices into the Comparison page.
- Select the best model by a stated rule (e.g. highest test accuracy, F1 as tie-breaker) and display that rule.
- Add measured CPU inference time (from Phase 8) per model, labeled with the hardware it was measured on.

**Exit:** Every number on screen traces back to a source file.

### Phase 7 — Optimize Memory and Performance
**Tasks**
- Profile peak RSS during model load and inference.
- Free decoded frames promptly and stream frames instead of loading the full video when sampling allows it.
- Limit upload size (e.g. 50 MB) and duration (e.g. 15 s), sized to the dataset's clip lengths.
- Set `torch.set_num_threads` sensibly for the cloud CPU.
- **Do not** quantize or half-precision the weights unless the team approves and accuracy parity is verified, because that changes model behavior.

**Exit:** Peak memory fits comfortably within Streamlit Community Cloud limits, and per-prediction latency is acceptable for a live demo.

### Phase 8 — Local Testing
Run the full test matrix in §9 and record the results in `docs/testing_report.md`.

**Exit:** All tests pass or have documented known limitations.

### Phase 9 — Free Deployment
**Tasks**
- Upload the checkpoints to a public Hugging Face model repo (if §7 requires it).
- Push the code to a public GitHub repo with checkpoints git-ignored.
- Deploy on Streamlit Community Cloud, select the Python version, and smoke-test all four models on the live URL.
- Finalize README.md with local run, storage, and deployment instructions, plus known limitations.

**Exit:** The public URL works end to end with all four models.

---

## 6. Inference Design Details

```
upload (bytes) → temp file → cv2.VideoCapture
  → validate (opens, frames > 0, fps > 0, duration ≤ limit)
  → sample T frame indices (training strategy)
  → read frames in order → BGR→RGB (if training did)
  → resize/crop to H×W (training order) → float [0,1]
  → normalize(mean, std) → stack → [C, T, H, W] → unsqueeze → [1, C, T, H, W]
  → model (CPU, inference_mode) → logits [1, N] → softmax → top-k
```

- Every value in this chain (T, H, W, mean, std, interpolation, crop) is read from `inference/config.py`, which is populated from the notebooks. **No defaults are assumed.** (Kinetics-400 stats such as `mean=[0.43216, 0.394666, 0.37645]` are common for r3d_18 but are used only if the notebook used them.)
- Assert the output dimension equals the class-map length before showing results.
- The temp file is deleted after processing.

---

## 7. Zero-Budget Model Storage

- An r3d_18 checkpoint is typically **~127 MB** in fp32 (about 33M parameters). This will be verified once the files are available.
- GitHub rejects files larger than 100 MB, and Git LFS has a small free quota. So the expected plan is:
  - **A public Hugging Face model repo** (e.g. `<team>/asl-vision-r3d18`) containing the four checkpoints, class maps, and a model card. Public repos are free, and downloads need no token.
  - The app uses `huggingface_hub.hf_hub_download(repo_id, filename)` and downloads only the selected model. The file is cached on disk in the cloud container.
  - The repo ID and filenames live in `inference/config.py`. No tokens are kept in code. The one-time upload uses the team's own `huggingface-cli login` locally.
- If the checkpoints turn out to be under ~50 MB each, they can go directly in GitHub instead.
- Training datasets and intermediate checkpoints are never uploaded.

---

## 8. Deployment (Streamlit Community Cloud, free)

1. Push the code to a **public GitHub repo** (`.gitignore` excludes `models/*.pt*`, `samples/`, `.env`, `__pycache__`).
2. `requirements.txt` uses CPU wheels:
   ```
   --extra-index-url https://download.pytorch.org/whl/cpu
   torch==<pinned cpu version>
   torchvision==<matching version>
   streamlit
   opencv-python-headless
   numpy
   pandas
   plotly
   huggingface_hub
   ```
   Exact pins will be chosen to match the training torchvision version where possible.
3. Go to share.streamlit.io → sign in with GitHub → *New app* → choose the repo, branch, and `app.py` → *Advanced settings*: Python 3.11 → Deploy.
4. On the first prediction, the selected checkpoint downloads from the HF Hub.
5. Smoke-test all four models, an invalid file, and model switching on the live URL.

---

## 9. Testing Matrix

| # | Test | Method | Pass criterion |
|---|---|---|---|
| T1 | Each checkpoint loads | `test_inference.py` | `strict=True` load succeeds |
| T2 | Class mapping correct | Compare against notebook mapping and known-label samples | Indices ↔ names match |
| T3 | Preprocessing parity | Compare against notebook transforms (tensor stats, shape) | Identical config, sensible predictions |
| T4 | Tensor dimensions | Assert | `[1, 3, T, H, W]` |
| T5 | Output dimension | Assert | `[1, N]`, N = class count |
| T6 | Confidence | Softmax sums to 1, top-k sorted | Pass |
| T7 | Model switching | Switch 4×4 in the UI and watch RSS | No growth beyond one model |
| T8 | Invalid videos | Corrupt file, wrong extension, 0-frame, too long, too large | Friendly error, no crash |
| T9 | CPU only | Run with `CUDA_VISIBLE_DEVICES=""` and the CPU wheel | Works |
| T10 | Memory | Peak RSS per model | Recorded, within limits |
| T11 | Inference time | Mean of N runs per model | Recorded honestly |
| T12 | Local Streamlit run | `streamlit run app.py`, all pages | No errors |
| T13 | Cloud run | Live URL smoke test | All four models predict |

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| 1000-model notebook or checkpoint is unavailable | Ship with the available models. The 1000 entry shows "not available" rather than fake data. |
| Preprocessing is ambiguous in the notebooks | Flag it to the team and do not guess. Use parity tests on known clips. |
| Class names were not saved (only indices) | Rebuild the mapping with the notebook's own code (e.g. sorted gloss list). Confirm with the team. |
| Checkpoints are too large or memory is too high on the free tier | Load one model at a time, download on demand, cap threads, and limit upload size. |
| Confusion matrix is too large for 1000+ classes | Use a zoomable heatmap plus a "most confused pairs" table and a per-class F1 distribution. |
| AVI/MOV previews don't play in some browsers | Inference still works. The preview shows a note, or the clip is converted to MP4 for preview only. |
| Dataset license restricts redistribution | Don't publish sample clips. Users upload their own. |

---

## 11. Known Limitations (expected, to be confirmed)

- Classifies **one isolated sign per clip**; it does not translate continuous signing.
- Accuracy on user-recorded videos may differ from test-set results because of domain shift (signer, background, lighting, framing).
- CPU inference on the free tier takes seconds, not milliseconds.
- The vocabulary is limited to the trained classes. Out-of-vocabulary signs still get a (wrong) top class, so confidence should be read with care.

---

## 12. Final Deliverables Checklist

- [ ] Application source code (`app.py`, `pages_impl/`, `inference/`, `ui/`)
- [ ] Organized project structure
- [ ] `requirements.txt` (CPU-only)
- [ ] `README.md`
- [ ] Model storage instructions (`models/README.md`)
- [ ] Local execution instructions
- [ ] Streamlit Community Cloud deployment instructions
- [ ] Testing report (`docs/testing_report.md`)
- [ ] Known limitations
- [ ] Inspection report (`docs/inspection_report.md`)
