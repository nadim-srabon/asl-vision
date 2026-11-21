"""Build the app's artifacts from the raw Kaggle outputs (run once, locally).

    .venv/Scripts/python scripts/prepare_artifacts.py

Reads   kaggle_outputs/   (not committed -- large, and derived from licensed data)
Writes  models/<model_id>.pt            plain fp32 state_dict, strict-load verified (not committed)
        results/<model_id>/*            metrics.json, per_class_metrics.csv, training_history.json,
                                        confusion_matrix.npz (where it exists)
        results/class_lists/classes_K.json
Every number written here is copied from a Kaggle output file; nothing is computed or estimated
except the class lists for K=300/500/1200, which are rebuilt with the notebooks' own top-K rule and
cross-checked against the class names in those runs' per-class reports.
"""
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inference.architectures import DualStreamSLRModel, GeometryOnlySLRModel, STGCN, build_swin3d_t  # noqa: E402
from inference.config import CLASS_LIST_DIR, MODELS, MODELS_DIR, RESULTS_DIR  # noqa: E402

K = ROOT / "kaggle_outputs"
LM = K / "landmark_1000" / "landmark_models_1000cls"
SOURCES = {
    "swin3d_t_1000": {"export": K / "swin3d_1000/swin3d_t_1000cls/app_export"},
    "geometry_transformer_1000": {"export": LM / "geometry_transformer_v10/app_export"},
    "stgcn_1000": {"export": LM / "stgcn/app_export"},
    "r3d18_dualstream_1000": {"export": K / "extras/r3d18_dualstream_1000/app_export",
                              "history": K / "r3d18_dualstream_1000/training_history.json"},
    "geometry_transformer_300": {"v10": K / "v10_300", "k": 300},
    "geometry_transformer_500": {"v10": K / "v10_500", "k": 500},
    "geometry_transformer_1200": {"v10": K / "v10_1200", "k": 1200},
}
BUILDERS = {"swin3d_t": lambda n: build_swin3d_t(n), "r3d18_dualstream": lambda n: DualStreamSLRModel(n),
            "geometry_transformer": lambda n: GeometryOnlySLRModel(n), "stgcn": lambda n: STGCN(n)}


def fingerprint(names):
    return hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:12]


def top_k_class_list(k):
    """The v10/v9 manifest rule, applied to the saved full manifest (identical construction)."""
    m = K / "r3d18_dualstream_1000/manifest"
    nslt = json.load(open(m / "nslt_asl_citizen.json"))
    full = {}
    for line in open(m / "class_list_asl_citizen.txt", encoding="utf-8"):
        if line.strip():
            i, g = line.strip().split("\t", 1)
            full[int(i)] = g
    counts = Counter(v["action"][0] for v in nslt.values())
    return [full[c] for c, _ in counts.most_common(k)]


def save_weights(spec, state):
    model = BUILDERS[spec.architecture](spec.num_classes)
    model.load_state_dict(state, strict=True)
    clean = {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(clean, spec.weights_path)
    return sum(v.numel() for v in clean.values() if v.is_floating_point())


def main():
    CLASS_LIST_DIR.mkdir(parents=True, exist_ok=True)
    class_lists = {}
    for model_id, src in SOURCES.items():
        spec = MODELS[model_id]
        out = spec.results_dir
        out.mkdir(parents=True, exist_ok=True)

        if "export" in src:
            ex = src["export"]
            cl = json.load(open(ex / "class_list.json", encoding="utf-8"))
            names = cl["classes"]
            state = torch.load(ex / "model_weights.pt", map_location="cpu", weights_only=False)
            metrics = json.load(open(ex / "metrics.json"))
            for f in ["per_class_metrics.csv", "confusion_matrix.npz"]:
                shutil.copy2(ex / f, out / f)
            hist = src.get("history", ex / "training_history.json")
            shutil.copy2(hist, out / "training_history.json")
        else:
            v = src["v10"]
            names = top_k_class_list(src["k"])
            # cross-check against the class names that run itself reported
            with open(v / "error_analysis_report/v10_per_class_metrics.csv", newline="", encoding="utf-8") as f:
                reported = {int(r["class_id"]): r["class_name"] for r in csv.DictReader(f)}
            assert all(names[i] == g for i, g in reported.items()), f"{model_id}: class list mismatch"
            ck = torch.load(v / "checkpoints_v10_geometry_only/best_model.pt", map_location="cpu", weights_only=False)
            state = ck["model_state_dict"]
            cmp_csv = next((v / "error_analysis_report").glob("*_vs_v9_comparison.csv"))
            with open(cmp_csv, newline="", encoding="utf-8") as f:
                row = [r for r in csv.DictReader(f) if r["Model"].startswith("v10")][0]
            with open(v / "error_analysis_report/precision_recall_f1_summary.csv", newline="", encoding="utf-8") as f:
                prf = [r for r in csv.DictReader(f) if r["model"] == "v10_geometry_only"][0]
            metrics = {
                "model_id": model_id, "num_classes": src["k"], "class_fingerprint": fingerprint(names),
                "best_epoch": int(ck["epoch"]), "val_top1_best": float(row["Val Top-1 (best ckpt)"]),
                "test": {"top1": float(row["Test Top-1"]), "top5": float(row["Test Top-5"]),
                         **{k: float(prf[k]) for k in ["macro_precision", "macro_recall", "macro_f1",
                                                       "weighted_precision", "weighted_recall", "weighted_f1"]}},
                "trainable_params": int(float(row["Trainable Params"])),
                "source": f"{cmp_csv.name} + precision_recall_f1_summary.csv",
            }
            shutil.copy2(v / "error_analysis_report/v10_per_class_metrics.csv", out / "per_class_metrics.csv")
            shutil.copy2(v / "training_history_v10_geometry_only.json", out / "training_history.json")

        n_params = save_weights(spec, state)
        metrics.setdefault("total_params", n_params)
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=1)

        fp = fingerprint(names)
        if spec.num_classes in class_lists:
            assert class_lists[spec.num_classes][1] == fp, f"{model_id}: differs from other {spec.num_classes}-class models"
        else:
            class_lists[spec.num_classes] = (names, fp)
            with open(spec.class_list_path, "w", encoding="utf-8") as f:
                json.dump({"num_classes": len(names), "fingerprint": fp, "classes": names}, f, indent=1)
        print(f"{model_id:28s} weights {spec.weights_path.stat().st_size / 1e6:6.1f} MB | "
              f"test top-1 {metrics['test']['top1']:.3f} | classes {fp}")

    assert class_lists[1000][1] == "72d3ef78291c"
    print("\nAll artifacts written; all 1000-class models share fingerprint 72d3ef78291c.")


if __name__ == "__main__":
    main()
