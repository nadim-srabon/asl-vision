# ASL Vision — Status Tracker

The full plan is in [PLAN.md](PLAN.md). The test details are in [docs/testing_report.md](docs/testing_report.md).

**Legend:** ⬜ Not started · 🟨 In progress · ✅ Done · ⛔ Blocked · ➖ Not applicable

**Last updated:** 2026-11-21
**Current phase:** Phase 9 — the app is built and tested locally. **Deployment needs the team's GitHub and Hugging Face accounts.**

---

## Overview

| Phase | Description | Status | Notes |
|---|---|---|---|
| 0 | Planning | ✅ | |
| 1 | Inspect existing project | ✅ | v10 (300/500/1200), r3d_18, cache builder |
| 1b | New 1000-class models | ✅ | Swin3D-T, v10 @1000, ST-GCN |
| 1c | Extras (R3D full metrics + parity pack) | ✅ | R3D re-eval reproduced its recorded 0.282 exactly |
| 2 | Models and preprocessing understood | ✅ | |
| 3 | Standalone CPU inference | ✅ | `inference/` — 7 models, all load with every weight matched; memory-mapped loading |
| 4 | Streamlit interface | ✅ | 5 pages, navy/white/blue theme, Plotly charts |
| 5 | Integrate all models | ✅ | One model in memory at a time; optional "run all 4" comparison |
| 6 | Real evaluation results and charts | ✅ | All numbers from Kaggle outputs (`results/`) |
| 7 | Memory / performance | ✅ | Peak ≈ 830 MB RSS; memory-mapped loading |
| 8 | Local testing | ✅ | See the testing report: parity, AppTest, server health |
| 9 | Free deployment | 🟨 | Files ready (`requirements.txt`, `packages.txt`, upload script, README). The team must push to GitHub, upload the weights to Hugging Face, and deploy |

---

## Results (all from Kaggle evaluation outputs; test split = 12,560 clips for 1000 classes)

| Model | Input | Classes | Test top-1 | Test top-5 | Macro P / R / F1 | CPU model latency* |
|---|---|---|---|---|---|---|
| **Video Swin-T** | RGB | 1000 | **0.668** | **0.889** | 0.694 / 0.669 / 0.665 | 1.22 s |
| Geometry Transformer v10 | landmarks | 1000 | 0.575 | 0.818 | 0.598 / 0.574 / 0.571 | 3 ms |
| ST-GCN | landmarks | 1000 | 0.488 | 0.740 | 0.513 / 0.488 / 0.484 | 10 ms |
| R3D-18 Dual-Stream (v9) | RGB + landmarks | 1000 | 0.282 | 0.551 | 0.316 / 0.284 / 0.271 | 0.46 s |
| Geometry Transformer v10 | landmarks | 300 | 0.632 | 0.850 | 0.657 / 0.632 / 0.630 | — |
| Geometry Transformer v10 | landmarks | 500 | 0.613 | 0.841 | 0.634 / 0.613 / 0.610 | — |
| Geometry Transformer v10 | landmarks | 1200 | 0.581 | 0.812 | 0.604 / 0.581 / 0.576 | — |

\*Kaggle CPU, 2 threads, forward pass only.

---

## Key decisions

- **One MediaPipe tracker per video, not shared.** The training cache can't be reproduced bit-for-bit, because MediaPipe's tracking depended on the previous video during caching. With a fresh tracker per video, the same upload always gives the same prediction (100% vs 67–87% with a shared tracker), and parity and accuracy are equivalent (testing report §3).
- **No dataset material in git, the app or Hugging Face.** The ASL Citizen licence forbids redistribution, and the videos show identifiable signers.
- **Weights in a private Hugging Face repo by default.** Whether trained weights may be published openly is a team and supervisor decision.

---

## Open items for the team

| # | Item | Status |
|---|---|---|
| D1 | Create a GitHub repo and push (check `git status` first: nothing from `kaggle_outputs/`, `notebooks/`, zips or `*.pt`) | ⬜ |
| D2 | `huggingface-cli login` → `python scripts/upload_to_hf.py <user>/asl-vision-models` | ⬜ |
| D3 | Streamlit Cloud: Python 3.12, secrets `ASL_HF_REPO`, `HF_TOKEN` | ⬜ |
| D4 | Put the GitHub URL in `config/team.json` → `github_url` | ⬜ |
| D5 | Decide: show member emails on the About page? (`show_emails`, currently false) | ⬜ |
| D6 | Decide with the supervisor: private or public weights (licence) | ⬜ |
| D7 | Optional: delete the original zip files (about 1.5 GB) now that everything is extracted | ⬜ |
| D8 | After the project ends: delete the parity videos (`kaggle_outputs/extras/parity_samples`), as the licence requires for personal data | ⬜ |

---

## Work Log

| Date | Phase | Work done |
|---|---|---|
| 2026-11-19 | 0–1b | Plan, status, inspected the 1200 notebook, wrote and smoke-tested the Swin and landmark training notebooks |
| 2026-11-21 | 1b | Swin eval-only notebook (after the training notebook's plotting bug) |
| 2026-11-21 | 1–3 | Extracted all outputs, verified class lists, read the cache builder, wrote `inference/architectures.py`, wrote and ran the extras notebook |
| 2026-11-21 | 3–8 | Video processor (port of the cache builder), model loader (strict, memory-mapped), predictor, artifacts script. Parity test on 30 real videos (fresh vs shared trackers). Built the 5-page Streamlit app. AppTest on all pages + end-to-end recognition with real videos and a corrupted file. Memory/latency measured. README, testing report, requirements, packages.txt, HF upload script |
