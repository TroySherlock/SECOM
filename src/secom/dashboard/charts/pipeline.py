"""Pipeline charts (page 2): HSIC, PLS, RF/Spearman selection, feature funnel."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from secom.dashboard.charts._base import (
    C_BLUE,
    C_GREEN,
    C_PURPLE,
    C_RED,
    C_YELLOW,
    COLORSCALE_LOW_GREEN_HIGH_RED,
    C,
    _sized,
)


def fig_hsic_selected_rank(
    selected_features: list[str],
    *,
    top_k: int | None = None,
) -> go.Figure:
    """Ordered bar of the real HSIC-Lasso selections (raw vs rz twin, colour-coded).

    HSIC stores only the *order* it picked sensors (strongest nonlinear dependence
    first), not the kernel magnitudes, so the bar length is selection-rank strength
    (``k - position``) and is labelled as such. Raw sensors and rz robust-z twins
    are two separate traces so the legend is clean (no phantom ``trace 0``) and the
    reader can see how often the drift-robust view wins. By default every selected
    feature is shown.
    """
    feats = list(selected_features or [])
    if not feats:
        return _sized(go.Figure(), height=360)
    k = len(feats)
    if top_k:
        feats = feats[:top_k]
    n = len(feats)
    strength = [k - i for i in range(n)]

    def _trace(name: str, color: str, *, want_rz: bool) -> go.Bar:
        xs = [s if f.endswith("_rz") == want_rz else None for f, s in zip(feats, strength)]
        return go.Bar(
            x=xs,
            y=feats,
            orientation="h",
            name=name,
            marker_color=color,
            hovertemplate="%{y}<br>selection-rank strength %{x}<extra></extra>",
        )

    fig = go.Figure()
    fig.add_trace(_trace("raw sensor", C_YELLOW, want_rz=False))
    fig.add_trace(_trace("rz robust-z twin", C_RED, want_rz=True))
    shown = f"top {n} of {k}" if top_k and k > n else f"all {k}"
    fig.update_layout(
        title=dict(text=f"HSIC-Lasso selection order ({shown})"),
        xaxis_title="Selection-rank strength (higher = picked earlier; magnitudes not stored)",
        yaxis_title="Selected feature",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
        bargap=0.25,
    )
    fig.update_yaxes(autorange="reversed", tickmode="linear", dtick=1)
    height = max(360, 80 + 22 * n)
    return _sized(fig, height=height, margin=dict(l=96, r=24, t=86, b=52))


def fig_pls_score_scatter(
    t1: np.ndarray,
    t2: np.ndarray,
    y: np.ndarray,
    *,
    title: str = "PLS latent scores: component 1 vs 2",
) -> go.Figure:
    """Scatter of the first two supervised PLS components, colored pass/fail.

    PLS projects every clustered sensor onto a few components chosen to maximise
    covariance with the fail label, so even two components tend to pull fails away
    from the pass cloud. Fails are drawn on top so the minority class is visible.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    y = np.asarray(y).astype(int)
    if t1.size == 0:
        return _sized(go.Figure(), height=420)
    has_c2 = bool(np.any(t2))
    fail = y == 1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t1[~fail],
            y=t2[~fail],
            mode="markers",
            name="Pass",
            marker=dict(color=C_BLUE, size=6, opacity=0.45),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t1[fail],
            y=t2[fail],
            mode="markers",
            name="Fail",
            marker=dict(color=C_RED, size=9, opacity=0.9, symbol="diamond",
                        line=dict(color="rgba(0,0,0,0.35)", width=0.5)),
        )
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis_title="PLS component 1 (score)",
        yaxis_title="PLS component 2 (score)" if has_c2 else "(single component)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _sized(fig, height=420, margin=dict(l=60, r=24, t=84, b=52))


def fig_rf_topk_selection_example(top_k: int) -> go.Figure:
    """Illustrative feature-importance ranking with top-k cutoff."""
    names = [f"c_{i}" for i in range(20)]
    importances = np.linspace(0.19, 0.03, num=20)
    colors = [C[1] if i < top_k else C[5] for i in range(20)]
    fig = go.Figure(
        data=[
            go.Bar(
                x=importances,
                y=names,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: importance=%{x:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"RF importance ranking (illustrative, top-{top_k} selected)"),
        xaxis_title="Relative importance",
        yaxis_title="Sensor",
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return _sized(fig, height=360, margin=dict(l=80, r=20, t=70, b=45))


def fig_spearman_cluster_example() -> go.Figure:
    """Illustrative Spearman-correlation cluster for c_340/c_204/c_67."""
    sensors = ["c_340", "c_204", "c_67"]
    corr = np.array(
        [
            [1.0, 0.93, 0.89],
            [0.93, 1.0, 0.91],
            [0.89, 0.91, 1.0],
        ]
    )
    return fig_spearman_cluster({"members": sensors, "correlations": corr.tolist()})


def fig_rf_topk_selection(model_artifact: dict) -> go.Figure:
    """RF importance ranking from holdout-fit SelectFromModel."""
    rf = model_artifact.get("rf_selection") or {}
    ranked = rf.get("importances") or []
    top_k = rf.get("top_k", len(rf.get("selected_features", [])))

    if not ranked:
        return fig_rf_topk_selection_example(top_k=top_k or 15)

    names = [r["feature"] for r in ranked]
    importances = [r["importance"] for r in ranked]
    selected_set = set(rf.get("selected_features") or [])
    colors = [C[1] if r["feature"] in selected_set else C[5] for r in ranked]
    fig = go.Figure(
        data=[
            go.Bar(
                x=importances,
                y=names,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: importance=%{x:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"RF importance ranking (top-{top_k} selected, holdout fit)"),
        xaxis_title="Relative importance",
        yaxis_title="Sensor",
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return _sized(fig, height=360, margin=dict(l=80, r=20, t=70, b=45))


def fig_spearman_cluster(cluster_example: dict | None) -> go.Figure:
    """Spearman ρ heatmap from pipeline cluster example."""
    if not cluster_example:
        return fig_spearman_cluster_example()
    sensors = cluster_example.get("members") or []
    corr_raw = cluster_example.get("correlations")
    if not sensors or corr_raw is None:
        return fig_spearman_cluster_example()

    corr = np.asarray(corr_raw, dtype=float)
    title_members = ", ".join(sensors[:4])
    if len(sensors) > 4:
        title_members += ", …"

    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=sensors,
            y=sensors,
            zmin=0.0,
            zmax=1.0,
            colorscale=COLORSCALE_LOW_GREEN_HIGH_RED,
            colorbar=dict(title="Spearman ρ"),
            text=np.round(corr, 2),
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} vs %{x}<br>ρ=%{z:.2f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=dict(text=f"Spearman cluster: {title_members}"),
        xaxis_title="Sensor",
        yaxis_title="Sensor",
        showlegend=False,
    )
    return _sized(fig, height=340, margin=dict(l=70, r=40, t=78, b=60))


def fig_impute_example() -> go.Figure:
    """Illustrative median impute: a sparse sensor whose gaps are filled at the
    training-fold median, so the column stays usable without outliers skewing it."""
    rng = np.random.default_rng(3)
    n = 24
    x = np.arange(n)
    y = 50 + rng.normal(0, 6, n)
    median = float(np.median(y))
    gap_idx = np.array([4, 5, 12, 18, 19])
    present_mask = np.ones(n, dtype=bool)
    present_mask[gap_idx] = False

    fig = go.Figure()
    fig.add_hline(
        y=median,
        line=dict(color=C_YELLOW, dash="dash", width=1.5),
        annotation_text="column median",
        annotation_position="top left",
        annotation_font=dict(size=11, color=C_YELLOW),
    )
    fig.add_trace(
        go.Scatter(
            x=x[present_mask],
            y=y[present_mask],
            mode="markers",
            name="observed",
            marker=dict(color=C_BLUE, size=8, opacity=0.85),
            hovertemplate="sample %{x}<br>value %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gap_idx,
            y=np.full(gap_idx.size, median),
            mode="markers",
            name="imputed (median)",
            marker=dict(color=C_YELLOW, size=11, symbol="diamond",
                        line=dict(color="rgba(0,0,0,0.35)", width=0.6)),
            hovertemplate="sample %{x}<br>filled at median %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Median impute: gaps → column median"),
        xaxis_title="Wafer (time order)",
        yaxis_title="Sensor reading",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _sized(fig, height=280, margin=dict(l=56, r=20, t=70, b=44))


def fig_scale_example() -> go.Figure:
    """Illustrative RobustScaler: a heavy-tailed raw sensor (blue) becomes
    median-centered, IQR-scaled (green) so outliers stop dominating the head."""
    rng = np.random.default_rng(11)
    raw = rng.lognormal(mean=1.0, sigma=0.6, size=600)
    median = float(np.median(raw))
    q75, q25 = np.percentile(raw, [75, 25])
    iqr = float(q75 - q25) or 1.0
    scaled = (raw - median) / iqr

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=raw, name="raw", marker_color=C_BLUE, opacity=0.6, nbinsx=40,
            hovertemplate="raw %{x:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=scaled, name="robust-scaled", marker_color=C_GREEN, opacity=0.7, nbinsx=40,
            hovertemplate="scaled %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Robust scale: median-center, IQR-scale"),
        xaxis_title="Sensor value",
        yaxis_title="Count",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _sized(fig, height=280, margin=dict(l=56, r=20, t=70, b=44))


def fig_calibrate_example() -> go.Figure:
    """Illustrative calibration: a raw head score is mapped through a Platt/sigmoid
    curve to a trustworthy fail probability; the dashed line is the identity."""
    raw = np.linspace(0.0, 1.0, 100)
    calibrated = 1.0 / (1.0 + np.exp(-7.5 * (raw - 0.55)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="identity",
            line=dict(color=C_YELLOW, dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=raw, y=calibrated, mode="lines", name="calibrator",
            line=dict(color=C_PURPLE, width=3),
            hovertemplate="raw %{x:.2f} → P(fail) %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Calibrate: raw score → trustworthy P(fail)"),
        xaxis_title="Raw head score",
        yaxis_title="Calibrated P(fail)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return _sized(fig, height=280, margin=dict(l=56, r=20, t=70, b=44))


def fig_pipeline_stage_counts(
    stages: dict[str, int],
    *,
    title: str = "Feature count through the pipeline",
) -> go.Figure:
    """Horizontal bar of the absolute feature count at each pipeline stage.

    Reads top-to-bottom in flow order so the journey is obvious: staged sensors
    -> dbt mart -> the one *increase* (the causal rolling-Z rz twins) -> cluster
    -> front-end (top-k + T2/hubs, or PLS latent components) -> classifier input.
    The rz-doubling stage is highlighted green and the final classifier input is
    accented so the single increase and the endpoint read at a glance. Handles
    both the selection front-ends and the PLS front-end from the per-model
    ``stages`` dict written to the artifacts.
    """
    g = lambda k, d=0: int(stages.get(k, d))  # noqa: E731
    stg = g("stg_sensors")
    mart = g("mart_sensors")
    after_impute = g("after_impute", 2 * mart)
    after_cluster = g("after_cluster")
    drop_correlated = g("drop_correlated")
    auxiliary = g("auxiliary_features")
    after_preprocess = g("after_preprocess", g("classifier_input"))
    classifier_input = g("classifier_input", after_preprocess)
    after_selection = stages.get("after_selection")
    after_hub = stages.get("after_hub_interactions")

    if not stg or not mart or not after_cluster:
        return _sized(go.Figure(), height=420)

    # (label, count, role) in pipeline order; role drives the bar colour.
    rows: list[tuple[str, int, str]] = [
        ("Staged sensors", stg, "neutral"),
        ("dbt mart (drop >10% missing / zero-var)", mart, "neutral"),
        ("+ rz robust-z twins", after_impute, "increase"),
    ]
    # Split the cluster step so the big cut is legible: VarianceThreshold first
    # (drops near-constant columns), then Spearman SmartCorrelatedSelection.
    if drop_correlated > 0:
        rows.append(("After variance threshold", after_cluster + drop_correlated, "neutral"))
    rows.append(("After Spearman correlation", after_cluster, "neutral"))
    if after_selection is not None:
        rows.append(("After top-k select", int(after_selection), "neutral"))
        if after_hub is not None and int(after_hub) != int(after_selection):
            rows.append(("+ Hotelling T2 / hub interactions", int(after_hub), "neutral"))
    else:
        components = max(0, after_preprocess - auxiliary)
        rows.append(("PLS latent components", components, "neutral"))
    rows.append(("Classifier input (+ auxiliary)", classifier_input, "total"))

    role_color = {"neutral": C_BLUE, "increase": C_GREEN, "total": C_RED}
    labels = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    colors = [role_color[r[2]] for r in rows]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{c:,}" for c in counts],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:,} features<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Feature count",
        yaxis_title="",
        showlegend=False,
        xaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)"),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(counts) * 1.15])
    return _sized(fig, height=420, margin=dict(l=260, r=40, t=72, b=52))
