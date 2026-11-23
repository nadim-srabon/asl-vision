import numpy as np
import pandas as pd
import streamlit as st

from inference.config import COMPARISON_1000, MODELS, SCALING_SERIES
from ui import charts, data
from ui.theme import MODEL_COLORS, SERIES_SEQ, html_row, metric, note, page_title

NO_CHART = {"displayModeBar": False}


def render():
    page_title("Results", "Model comparison",
               "All numbers below come from the evaluation runs on the held-out, signer-independent ASL Citizen "
               "test split (12,560 clips for the 1,000-sign models). Nothing on this page is estimated.")

    df = data.summary_table(COMPARISON_1000)
    names = {r.model_id: r.Model for r in df.itertuples()}
    best = df.loc[df["Accuracy (test top-1)"].idxmax()]
    fastest = df.loc[df["CPU latency (ms)"].idxmin()]

    html_row([
        metric("Best accuracy", f"{best['Accuracy (test top-1)']:.1%}", f"{best['Model']} (test top-1)"),
        metric("Best top-5", f"{df['Test top-5'].max():.1%}", df.loc[df['Test top-5'].idxmax(), 'Model']),
        metric("Best macro F1", f"{df['F1-score (macro)'].max():.3f}", df.loc[df['F1-score (macro)'].idxmax(), 'Model']),
        metric("Fastest on CPU", f"{fastest['CPU latency (ms)']:.0f} ms", f"{fastest['Model']} (model only)"),
    ])
    st.write("")

    st.subheader("1,000-sign comparison")
    show = df[["Model", "Input", "Accuracy (test top-1)", "Precision (macro)", "Recall (macro)", "F1-score (macro)",
               "Test top-5", "Val top-1", "Parameters (M)", "CPU latency (ms)"]].copy()

    def _hl(row):
        return ["background-color:rgba(147,52,233,0.18);font-weight:700" if row["Model"] == best["Model"] else "" for _ in row]

    st.dataframe(show.style.apply(_hl, axis=1).format({
        "Accuracy (test top-1)": "{:.1%}", "Precision (macro)": "{:.3f}", "Recall (macro)": "{:.3f}",
        "F1-score (macro)": "{:.3f}", "Test top-5": "{:.1%}", "Val top-1": "{:.1%}", "Parameters (M)": "{:.2f}",
        "CPU latency (ms)": "{:.0f}"}), hide_index=True, width="stretch")
    st.caption(f"Highlighted: best model by test top-1 accuracy ({best['Model']}). Precision / recall / F1 are "
               "macro-averaged over the 1,000 classes. CPU latency = one forward pass of one 16-frame clip on a "
               "2-thread Kaggle CPU, excluding MediaPipe feature extraction.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Accuracy**")
        st.plotly_chart(charts.grouped_bars(
            list(df["Model"]), {"Top-1": list(df["Accuracy (test top-1)"]), "Top-5": list(df["Test top-5"])},
            [SERIES_SEQ[0], "#9CC2F0"]), width="stretch", config=NO_CHART)
    with c2:
        st.markdown("**Precision, recall and F1 (macro)**")
        st.plotly_chart(charts.grouped_bars(
            list(df["Model"]), {"Precision": list(df["Precision (macro)"]), "Recall": list(df["Recall (macro)"]),
                                "F1-score": list(df["F1-score (macro)"])}, SERIES_SEQ, pct=False),
            width="stretch", config=NO_CHART)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Accuracy vs CPU cost** &nbsp;<span class='small-muted'>(marker size = parameters)</span>",
                    unsafe_allow_html=True)
        st.plotly_chart(charts.latency_vs_accuracy(df, MODEL_COLORS), width="stretch", config=NO_CHART)
    with c4:
        st.markdown("**Per-class F1 distribution** &nbsp;<span class='small-muted'>(box = quartiles, dashed = mean)</span>",
                    unsafe_allow_html=True)
        pcs = {m: data.per_class(m) for m in COMPARISON_1000 if data.per_class(m) is not None}
        st.plotly_chart(charts.f1_distribution(pcs, MODEL_COLORS, names), width="stretch", config=NO_CHART)

    st.write("")
    st.subheader("Where the models go wrong")
    sel = st.selectbox("Model", COMPARISON_1000, format_func=lambda m: names[m], key="cm_model")
    cm = data.confusion(sel)
    classes = data.class_names(1000)
    if cm is None:
        st.info("No confusion matrix was saved for this model.")
    else:
        off = cm.copy()
        np.fill_diagonal(off, 0)
        ti, pi = np.unravel_index(np.argsort(off, axis=None)[::-1][:15], off.shape)
        pairs = pd.DataFrame({"true": [classes[i] for i in ti], "pred": [classes[j] for j in pi],
                              "errors": [int(off[i, j]) for i, j in zip(ti, pi)]})
        c5, c6 = st.columns([1, 1.25], gap="large")
        with c5:
            st.markdown("**Most-confused sign pairs**")
            st.plotly_chart(charts.confused_pairs_bar(pairs, MODEL_COLORS[sel]), width="stretch",
                            config=NO_CHART)
        with c6:
            st.markdown("**Confusion matrix -- the 25 classes most involved in errors**")
            involved = pd.Series(np.concatenate([off.sum(1), off.sum(0)]).reshape(2, -1).sum(0))
            idx = involved.sort_values(ascending=False).index[:25].tolist()
            st.plotly_chart(charts.confusion_heatmap(cm[np.ix_(idx, idx)], [classes[i] for i in idx]),
                            width="stretch", config=NO_CHART)
        st.caption(f"The full matrix is 1,000 x 1,000 ({int(cm.sum()):,} test clips, {int(np.trace(cm)):,} correct). "
                   "Many confusions are between signs that are genuinely similar in ASL (e.g. near-synonyms or "
                   "signs sharing a handshape and location).")

    st.write("")
    st.subheader("Training curves")
    sel2 = st.selectbox("Model ", COMPARISON_1000, format_func=lambda m: names[m], key="hist_model")
    h = data.history(sel2)
    if h is not None:
        a, b = st.columns(2, gap="large")
        a.plotly_chart(charts.training_curves(h, ("train_top1", "val_top1"), "Top-1 accuracy"),
                       width="stretch", config=NO_CHART)
        b.plotly_chart(charts.training_curves(h, ("train_loss", "val_loss"), "Loss (label-smoothed CE)"),
                       width="stretch", config=NO_CHART)

    st.write("")
    st.subheader("Vocabulary scaling -- Geometry Transformer")
    sc = data.summary_table(SCALING_SERIES)
    st.plotly_chart(charts.scaling_lines(list(sc["Classes"]), {
        "Test top-5": list(sc["Test top-5"]), "Test top-1": list(sc["Accuracy (test top-1)"]),
        "Macro F1": list(sc["F1-score (macro)"])}), width="stretch", config=NO_CHART)
    st.dataframe(sc[["Classes", "Accuracy (test top-1)", "Test top-5", "Precision (macro)", "Recall (macro)",
                     "F1-score (macro)", "Val top-1"]].style.format({
        "Accuracy (test top-1)": "{:.1%}", "Test top-5": "{:.1%}", "Val top-1": "{:.1%}",
        "Precision (macro)": "{:.3f}", "Recall (macro)": "{:.3f}", "F1-score (macro)": "{:.3f}"}),
        hide_index=True, width="stretch")

    st.write("")
    note("<b>Reading these results.</b> All 1,000-sign models use the identical class list (same 1,000 glosses, "
         "same index order) and the same official signer-independent split. Differences in training protocol: "
         "the R3D-18 dual-stream model (Capstone B) was selected on lowest validation loss and froze the 3D-CNN "
         "up to <i>layer3</i>; all other models were selected on best validation top-1. Video Swin-T reached its "
         "best validation score in its final (25th) epoch, so longer training might improve it further.")
