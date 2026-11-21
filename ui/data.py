"""Read-only access to the evaluation artifacts in results/ (all produced on Kaggle)."""
import json

import numpy as np
import pandas as pd
import streamlit as st

from inference.config import MODELS, RESULTS_DIR, ROOT


@st.cache_data(show_spinner=False)
def metrics(model_id):
    p = MODELS[model_id].results_dir / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def history(model_id):
    p = MODELS[model_id].results_dir / "training_history.json"
    return pd.DataFrame(json.loads(p.read_text())) if p.exists() else None


@st.cache_data(show_spinner=False)
def per_class(model_id):
    p = MODELS[model_id].results_dir / "per_class_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def confusion(model_id):
    p = MODELS[model_id].results_dir / "confusion_matrix.npz"
    return np.load(p)["cm"].astype(np.int32) if p.exists() else None


@st.cache_data(show_spinner=False)
def class_names(num_classes):
    p = RESULTS_DIR / "class_lists" / f"classes_{num_classes}.json"
    return json.loads(p.read_text(encoding="utf-8"))["classes"]


@st.cache_data(show_spinner=False)
def dataset():
    return json.loads((RESULTS_DIR / "dataset.json").read_text())


@st.cache_data(show_spinner=False)
def team():
    return json.loads((ROOT / "config" / "team.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def parity():
    p = ROOT / "docs" / "parity_results_fresh.json"
    return json.loads(p.read_text()) if p.exists() else None


def summary_table(model_ids):
    rows = []
    for mid in model_ids:
        m, spec = metrics(mid), MODELS[mid]
        if m is None:
            continue
        t = m["test"]
        lat = m.get("cpu_latency", {}).get("mean_s")
        rows.append({
            "model_id": mid, "Model": spec.display_name, "Input": spec.family, "Classes": spec.num_classes,
            "Parameters (M)": round(m.get("total_params", m.get("trainable_params", 0)) / 1e6, 2),
            "Val top-1": m.get("val_top1_best"), "Accuracy (test top-1)": t["top1"], "Test top-5": t["top5"],
            "Precision (macro)": t.get("macro_precision"), "Recall (macro)": t.get("macro_recall"),
            "F1-score (macro)": t.get("macro_f1"),
            "CPU latency (ms)": round(lat * 1000, 1) if lat is not None else None,
            "Test clips": t.get("n_samples"),
        })
    return pd.DataFrame(rows)
