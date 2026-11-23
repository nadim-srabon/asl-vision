"""Plotly figure builders. Rules: one y-axis per chart, colour fixed per model (never by rank),
recessive grid, hover tooltips on every mark, legend for >= 2 series."""
import numpy as np
import plotly.graph_objects as go

from .theme import ACCENT, BORDER, DIM, MUTED, PANEL, SERIES_SEQ, TEXT

GRID = "rgba(255,255,255,0.06)"
INK_2, INK_3, LINE, NAVY = MUTED, DIM, BORDER, TEXT

FONT = dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12)


def _layout(fig, title=None, height=380, y_title=None, x_title=None, y_range=None, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=NAVY), x=0, xanchor="left") if title else None,
        height=height, margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=FONT, showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor="#9334e9", font=dict(color=TEXT, family="Inter")),
        bargap=0.28, bargroupgap=0.08,
    )
    fig.update_yaxes(title=y_title, gridcolor=GRID, zerolinecolor=LINE, tickfont=dict(color=INK_3),
                     range=y_range)
    fig.update_xaxes(title=x_title, showgrid=False, linecolor=LINE, tickfont=dict(color=INK_2))
    return fig


def top_k_bar(labels, probs, color=ACCENT):
    labels, probs = list(labels)[::-1], [p * 100 for p in probs][::-1]
    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation="h", marker=dict(color=color, line=dict(color="#171021", width=2)),
        text=[f"{p:.1f}%" for p in probs], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2), hovertemplate="<b>%{y}</b><br>confidence %{x:.2f}%<extra></extra>"))
    _layout(fig, height=60 + 46 * len(labels), x_title="Softmax confidence (%)", legend=False)
    fig.update_xaxes(range=[0, max(100, max(probs) * 1.15)], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(tickfont=dict(color=NAVY, size=14))
    return fig


def grouped_bars(categories, series, colors, title=None, y_title="Score", y_range=(0, 1), pct=True, height=400):
    """series: {name: [values per category]} -- one trace per series, colour by series."""
    fig = go.Figure()
    for (name, vals), col in zip(series.items(), colors):
        fig.add_bar(name=name, x=categories, y=vals, marker=dict(color=col, line=dict(color="#171021", width=2)),
                    text=[(f"{v:.1%}" if pct else f"{v:.3f}") if v is not None else "n/a" for v in vals],
                    textposition="outside", textfont=dict(size=11, color=INK_2), cliponaxis=False,
                    hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:{'.1%' if pct else '.3f'}}}<extra></extra>")
    _layout(fig, title, height=height, y_title=y_title, y_range=list(y_range))
    if pct:
        fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group")
    return fig


def latency_vs_accuracy(df, colors):
    fig = go.Figure()
    for _, r in df.iterrows():
        fig.add_scatter(x=[r["CPU latency (ms)"]], y=[r["Accuracy (test top-1)"]], mode="markers+text",
                        name=r["Model"], text=[r["Model"]], textposition="top center",
                        textfont=dict(color=INK_2, size=12),
                        marker=dict(size=10 + 2.2 * np.sqrt(r["Parameters (M)"]), color=colors[r["model_id"]],
                                    line=dict(color="#171021", width=2)),
                        hovertemplate=(f"<b>{r['Model']}</b><br>test top-1 %{{y:.1%}}<br>CPU latency %{{x:.0f}} ms"
                                       f"<br>{r['Parameters (M)']} M parameters<extra></extra>"))
    _layout(fig, None, height=400, y_title="Test top-1 accuracy", x_title="CPU latency per clip, ms (log scale)")
    fig.update_xaxes(type="log", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.8])
    return fig


def training_curves(hist, metric_pair=("train_top1", "val_top1"), title=None):
    fig = go.Figure()
    names = {"train_top1": "Train top-1", "val_top1": "Validation top-1", "train_loss": "Train loss",
             "val_loss": "Validation loss"}
    for key, col in zip(metric_pair, SERIES_SEQ):
        if key in hist:
            fig.add_scatter(x=hist["epoch"], y=hist[key], name=names.get(key, key), mode="lines",
                            line=dict(color=col, width=2),
                            hovertemplate=f"{names.get(key, key)}<br>epoch %{{x}}: %{{y:.3f}}<extra></extra>")
    _layout(fig, title, height=330, x_title="Epoch", y_title="Accuracy" if "top1" in metric_pair[0] else "Loss")
    fig.update_layout(hovermode="x unified")
    return fig


def scaling_lines(classes, series):
    fig = go.Figure()
    for (name, vals), col in zip(series.items(), SERIES_SEQ):
        fig.add_scatter(x=classes, y=vals, name=name, mode="lines+markers", line=dict(color=col, width=2),
                        marker=dict(size=9, line=dict(color="#171021", width=2)),
                        hovertemplate=f"<b>{name}</b><br>%{{x}} classes: %{{y:.1%}}<extra></extra>")
    _layout(fig, None, height=360, x_title="Number of classes", y_title="Test score", y_range=[0.4, 0.9])
    fig.update_xaxes(tickvals=classes, showgrid=False)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(hovermode="x unified")
    return fig


def confused_pairs_bar(pairs_df, color):
    labels = [f"{t} → {p}" for t, p in zip(pairs_df["true"], pairs_df["pred"])][::-1]
    fig = go.Figure(go.Bar(x=pairs_df["errors"][::-1], y=labels, orientation="h",
                           marker=dict(color=color, line=dict(color="#171021", width=2)),
                           hovertemplate="<b>%{y}</b><br>%{x} test clips<extra></extra>"))
    _layout(fig, None, height=80 + 26 * len(labels), x_title="Misclassified test clips (true → predicted)",
            legend=False)
    return fig


def confusion_heatmap(sub_cm, names):
    fig = go.Figure(go.Heatmap(
        z=sub_cm, x=names, y=names, colorscale=[[0, "#1d1429"], [0.25, "#4c1d95"], [0.6, "#9334e9"], [1, "#e9d5ff"]],
        hovertemplate="true <b>%{y}</b><br>predicted <b>%{x}</b><br>%{z} clips<extra></extra>",
        colorbar=dict(title="clips", thickness=12)))
    _layout(fig, None, height=560, x_title="Predicted", y_title="True", legend=False)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    return fig


def f1_distribution(per_class_by_model, colors, names):
    fig = go.Figure()
    for mid, df in per_class_by_model.items():
        fig.add_box(y=df["f1"], name=names[mid], marker_color=colors[mid], line=dict(width=2), boxmean=True,
                    hovertemplate=f"<b>{names[mid]}</b><br>F1 %{{y:.2f}}<extra></extra>")
    _layout(fig, None, height=380, y_title="Per-class F1 (test)", y_range=[0, 1.02], legend=False)
    return fig
