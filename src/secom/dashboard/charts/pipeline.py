"""Pipeline charts (page 2): HSIC, PLS, RF/Spearman selection, feature funnel."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from secom.dashboard.charts._base import (
    C_BLUE,
    C_GREEN,
    C_PURPLE,
    C_RED,
    COLORSCALE_LOW_GREEN_HIGH_RED,
    C,
    _sized,
)


def fig_hsic_selected_rank(
    selected_features: list[str],
    *,
    top_k: int | None = 20,
) -> go.Figure:
    """Ordered bar of the real HSIC-Lasso selections.

    HSIC stores only the *order* it picked sensors (strongest nonlinear dependence
    first), not the kernel magnitudes, so the bar height is selection-rank strength
    (``k - position``) and is labeled as such. rz robust-z twins are colored apart
    from raw sensors so the reader can see how often the drift-robust view wins.
    """
    feats = list(selected_features or [])
    if not feats:
        return _sized(go.Figure(), height=360)
    k = len(feats)
    if top_k:
        feats = feats[:top_k]
    n = len(feats)
    strength = [k - i for i in range(n)]
    is_rz = [f.endswith("_rz") for f in feats]
    colors = [C_PURPLE if rz else C_BLUE for rz in is_rz]

    fig = go.Figure(
        go.Bar(
            x=strength,
            y=feats,
            orientation="h",
            marker_color=colors,
            customdata=[("rz robust-z twin" if rz else "raw sensor") for rz in is_rz],
            hovertemplate="%{y} (%{customdata})<br>selection rank %{x}<extra></extra>",
        )
    )
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=C_BLUE, name="raw sensor"))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=C_PURPLE, name="rz robust-z twin"))
    shown = f"top {n} of {k}" if top_k and k > n else f"all {k}"
    fig.update_layout(
        title=dict(text=f"HSIC selection order ({shown})"),
        xaxis_title="Selection-rank strength (higher = picked earlier; magnitudes not stored)",
        yaxis_title="Selected feature",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
    )
    fig.update_yaxes(autorange="reversed")
    return _sized(fig, height=420, margin=dict(l=90, r=24, t=86, b=52))


def fig_hsic_dependence_intuition() -> go.Figure:
    """Why HSIC, not correlation: a nonlinear sensor-vs-fail link that a linear
    correlation misses entirely (Pearson r ~ 0) but kernel dependence detects."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-3.0, 3.0, 260)
    # Fails concentrate at both extremes (U-shaped risk): linear corr ~ 0.
    fail_prob = 1.0 / (1.0 + np.exp(-(x**2 - 3.2)))
    fail = rng.uniform(size=x.size) < fail_prob
    r = float(np.corrcoef(x, fail.astype(float))[0, 1])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x[~fail],
            y=rng.normal(0, 0.04, (~fail).sum()),
            mode="markers",
            name="Pass",
            marker=dict(color=C_BLUE, size=7, opacity=0.55),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[fail],
            y=rng.normal(1, 0.04, fail.sum()),
            mode="markers",
            name="Fail",
            marker=dict(color=C_RED, size=8, opacity=0.85, symbol="diamond"),
        )
    )
    fig.add_annotation(
        x=0, y=0.5, showarrow=False,
        text=f"Pearson r ≈ {r:+.2f} (linear: blind)<br>HSIC: dependence detected",
        font=dict(size=13), bgcolor="rgba(0,0,0,0.04)", bordercolor=C_PURPLE, borderpad=6,
    )
    fig.update_layout(
        title=dict(text="Why HSIC, not correlation: nonlinear sensor → fail dependence"),
        xaxis_title="Sensor reading (standardized)",
        yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Pass", "Fail"], title=""),
    )
    return _sized(fig, height=360, margin=dict(l=60, r=24, t=72, b=52))


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


def fig_pipeline_feature_funnel(
    stages: dict[str, int],
    *,
    title: str = "Feature count through the pipeline",
) -> go.Figure:
    """Waterfall of the true per-stage feature count, including the rz doubling.

    Unlike a monotonic funnel, this shows every step as a signed delta so the
    flow is conserved and the one *increase* - the causal rolling-Z (rz) twin
    that the dbt mart appends to every kept raw sensor - is explicit. Handles
    both the selection front-ends (top-k + T2) and the PLS front-end (latent
    components). Reads the per-model ``stages`` dict from the artifacts.
    """
    g = lambda k, d=0: int(stages.get(k, d))  # noqa: E731
    stg = g("stg_sensors")
    mart = g("mart_sensors")
    after_impute = g("after_impute", 2 * mart)
    after_cluster = g("after_cluster")
    drop_correlated = g("drop_correlated")
    auxiliary = g("auxiliary_features")
    after_preprocess = g("after_preprocess", g("classifier_input"))
    after_selection = stages.get("after_selection")
    after_hub = stages.get("after_hub_interactions")

    if not stg or not mart or not after_cluster:
        return _sized(go.Figure(), height=420)

    labels: list[str] = ["Staged sensors"]
    deltas: list[float] = [stg]
    measures: list[str] = ["absolute"]

    def step(label: str, delta: int) -> None:
        labels.append(label)
        deltas.append(delta)
        measures.append("relative")

    step("- dbt drop (>10% missing / zero-var)", -(stg - mart))
    step("+ rz robust-z twins", after_impute - mart)
    variance_drop = max(0, after_impute - drop_correlated - after_cluster)
    if variance_drop:
        step("- variance threshold", -variance_drop)
    step("- Spearman correlated", -drop_correlated)

    if after_selection is not None:
        sel = int(after_selection)
        step("- top-k select", -(after_cluster - sel))
        block = sel
        if after_hub is not None and int(after_hub) != sel:
            step("+ Hotelling T2 / hubs", int(after_hub) - sel)
            block = int(after_hub)
    else:
        components = max(0, after_preprocess - auxiliary)
        step("- PLS latent components", -(after_cluster - components))
        block = components

    if auxiliary:
        step("+ auxiliary (calendar / missing)", auxiliary)

    labels.append("Classifier input")
    deltas.append(block + auxiliary)
    measures.append("total")

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=deltas,
            text=[f"{int(v):+d}" if m == "relative" else f"{int(v)}"
                  for v, m in zip(deltas, measures)],
            textposition="outside",
            connector=dict(line=dict(color="rgba(200,200,200,0.4)")),
            increasing=dict(marker=dict(color=C_GREEN)),
            decreasing=dict(marker=dict(color=C_RED)),
            totals=dict(marker=dict(color=C_BLUE)),
            hovertemplate="%{x}<br>%{y:+d} features<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title),
        yaxis_title="Feature count",
        xaxis=dict(tickangle=-30),
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)"),
        showlegend=False,
    )
    return _sized(fig, height=460, margin=dict(l=56, r=36, t=72, b=140))
