"""MSPC gate charts (pages 5.x): statistic distributions, control charts, factor space."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from secom.dashboard.charts._base import (
    C_BLUE,
    C_GREEN,
    C_ORANGE,
    C_PURPLE,
    C_RED,
    C_YELLOW,
    COLORSCALE_DRIFT_DIVERGING,
    _sized,
)

# Gate-monitor populations: reference (train passing), in-dist, temporal holdout.
_GATE_POP_COLORS = {
    "reference": C_BLUE,
    "in_distribution": C_GREEN,
    "temporal": C_RED,
}
_GATE_POP_LABELS = {
    "reference": "Reference (passing train)",
    "in_distribution": "In-distribution holdout",
    "temporal": "Temporal holdout",
}


def fig_gate_statistic_distributions(
    populations: dict[str, np.ndarray],
    *,
    statistic: str,
    limit: float | None = None,
    limit_side: str = "upper",
    title: str = "Gate statistic distribution",
) -> go.Figure:
    """Overlaid violins of a BGM gate statistic across wafer populations.

    ``populations`` maps ``reference`` / ``in_distribution`` / ``temporal`` to
    per-wafer values (e.g. BGM log-density or Q/SPE). ``limit`` draws the control
    limit; ``limit_side`` is ``"upper"`` (Q UCL) or ``"lower"`` (density LCL).
    Drift shows as the temporal violin pushing past the limit.
    """
    fig = go.Figure()
    has_data = False
    for key in ("reference", "in_distribution", "temporal"):
        values = np.asarray(populations.get(key, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        has_data = True
        fig.add_trace(
            go.Violin(
                y=values,
                name=_GATE_POP_LABELS[key],
                line_color=_GATE_POP_COLORS[key],
                box_visible=True,
                meanline_visible=True,
                points=False,
                hovertemplate=f"{_GATE_POP_LABELS[key]}<br>{statistic}=%{{y:.3f}}<extra></extra>",
            )
        )
    if not has_data:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=440,
            margin=dict(l=56, r=48, t=72, b=48),
        )

    if limit is not None:
        side_label = "UCL" if limit_side == "upper" else "LCL"
        fig.add_hline(
            y=float(limit),
            line_dash="dash",
            line_color=C_YELLOW,
            annotation_text=f"{side_label} {float(limit):.2f}",
            annotation_position="right",
        )
    fig.update_layout(
        title=dict(text=title),
        yaxis_title=statistic,
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        violingap=0.25,
    )
    return _sized(fig, height=440, margin=dict(l=56, r=48, t=72, b=48))


def fig_gate_control_chart(
    values: np.ndarray,
    *,
    limit: float,
    ooc_mask: np.ndarray,
    x: np.ndarray | None = None,
    statistic: str = "Q / SPE",
    limit_side: str = "upper",
    title: str = "MSPC control chart",
) -> go.Figure:
    """Time-ordered control chart with the control limit, OOC points, rolling rate.

    Points are plotted in wafer order (``x`` = timestamps if given, else index);
    points beyond the control limit (``ooc_mask``) are highlighted, and a rolling
    out-of-control rate is overlaid on a secondary axis to show the OOC rate
    climbing into the drifted era.
    """
    vals = np.asarray(values, dtype=float)
    fig = go.Figure()
    if vals.size == 0:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=460,
            margin=dict(l=56, r=64, t=72, b=48),
        )

    idx = np.arange(vals.size)
    xs = np.asarray(x) if x is not None and len(x) == vals.size else idx
    ooc = np.asarray(ooc_mask, dtype=bool)
    in_ctrl = ~ooc

    fig.add_trace(
        go.Scatter(
            x=xs[in_ctrl],
            y=vals[in_ctrl],
            mode="markers",
            name="In control",
            marker=dict(color=C_BLUE, size=6),
            hovertemplate=f"{statistic}=%{{y:.3f}}<extra></extra>",
        )
    )
    if ooc.any():
        fig.add_trace(
            go.Scatter(
                x=xs[ooc],
                y=vals[ooc],
                mode="markers",
                name="Out of control",
                marker=dict(color=C_RED, size=8, symbol="x"),
                hovertemplate=f"OOC<br>{statistic}=%{{y:.3f}}<extra></extra>",
            )
        )

    side_label = "UCL" if limit_side == "upper" else "LCL"
    fig.add_hline(
        y=float(limit),
        line_dash="dash",
        line_color=C_YELLOW,
        annotation_text=f"{side_label} {float(limit):.2f}",
        annotation_position="right",
    )

    # Trailing-window rolling OOC rate (headline trend) + faint cumulative reference,
    # both on a secondary axis. The cumulative average is sticky and hides a late-era
    # spike, so the rolling window is the trend a fab actually reads.
    ooc_f = ooc.astype(float)
    window = int(max(20, ooc_f.size // 10))
    window = min(window, ooc_f.size) or 1
    kernel = np.ones(window, dtype=float)
    counts = np.convolve(np.ones_like(ooc_f), kernel, mode="full")[: ooc_f.size]
    rolling_rate = np.convolve(ooc_f, kernel, mode="full")[: ooc_f.size] / np.maximum(counts, 1.0)
    cumulative_rate = np.cumsum(ooc_f) / (idx + 1.0)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=cumulative_rate,
            mode="lines",
            name="Cumulative OOC rate",
            line=dict(color=C_ORANGE, width=1.2, dash="dot"),
            opacity=0.5,
            yaxis="y2",
            hovertemplate="cumulative=%{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=rolling_rate,
            mode="lines",
            name=f"Rolling OOC rate (window={window})",
            line=dict(color=C_ORANGE, width=2.5),
            yaxis="y2",
            hovertemplate="rolling=%{y:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Wafer order (measurement time)" if x is not None else "Wafer order",
        yaxis=dict(title=statistic, gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis2=dict(
            title="OOC rate",
            overlaying="y",
            side="right",
            range=[0, 1],
            showgrid=False,
            tickformat=".0%",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=460, margin=dict(l=56, r=64, t=72, b=48))


def _ellipse_path(mean2: np.ndarray, cov2: np.ndarray, *, n_sigma: float = 2.0, n: int = 60):
    """2-sigma ellipse (x, y) arrays for a 2-D Gaussian via eigen-decomposition."""
    try:
        vals, vecs = np.linalg.eigh(np.asarray(cov2, dtype=float))
    except np.linalg.LinAlgError:
        return None
    vals = np.clip(vals, 1e-12, None)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # (2, n)
    ellipse = vecs @ (np.sqrt(vals)[:, None] * circle) * n_sigma  # (2, n)
    return mean2[0] + ellipse[0], mean2[1] + ellipse[1]


def fig_sbfa_factor_space(
    reference_scores: np.ndarray,
    holdout_scores: np.ndarray,
    y_true: np.ndarray,
    flagged: np.ndarray,
    *,
    dims: tuple[int, int] = (0, 1),
    bgm_means: np.ndarray | None = None,
    bgm_covariances: np.ndarray | None = None,
    factor_labels: tuple[str, str] | None = None,
    title: str = "sBFA latent factor space",
) -> go.Figure:
    """2-D scatter of two sBFA factors with the BGM 2-sigma envelope overlaid.

    Faint reference points = passing-train envelope; holdout points colored by
    Pass / Fail / Flagged. BGM components are drawn as 2-sigma covariance ellipses
    (the learned multimodal 'normal operating' region). Drift = the holdout cloud
    sliding off the envelope.
    """
    fig = go.Figure()
    ref = np.asarray(reference_scores, dtype=float)
    hold = np.asarray(holdout_scores, dtype=float)
    dx, dy = dims
    if ref.ndim != 2 or hold.ndim != 2 or ref.shape[1] <= max(dims):
        return _sized(fig.update_layout(title=dict(text=title)), height=520, margin=dict(l=56, r=48, t=72, b=48))

    # BGM envelope ellipses (under the points).
    if bgm_means is not None and bgm_covariances is not None:
        means = np.asarray(bgm_means, dtype=float)
        covs = np.asarray(bgm_covariances, dtype=float)
        for c in range(means.shape[0]):
            mean2 = means[c, [dx, dy]]
            cov2 = covs[c][np.ix_([dx, dy], [dx, dy])]
            path = _ellipse_path(mean2, cov2)
            if path is None:
                continue
            ex, ey = path
            fig.add_trace(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    line=dict(color=C_YELLOW, width=1.5),
                    fill="toself",
                    fillcolor="rgba(216, 166, 87, 0.07)",
                    name="BGM 2σ envelope" if c == 0 else None,
                    showlegend=c == 0,
                    hoverinfo="skip",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=ref[:, dx],
            y=ref[:, dy],
            mode="markers",
            name="Reference (passing train)",
            marker=dict(color="rgba(125, 174, 163, 0.28)", size=4),
            hoverinfo="skip",
        )
    )

    _add_factor_points(fig, hold, y_true, flagged, dx, dy)

    xlab, ylab = factor_labels or (f"Factor {dx + 1}", f"Factor {dy + 1}")
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=xlab, gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        yaxis=dict(title=ylab, gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=520, margin=dict(l=56, r=48, t=72, b=48))


def _add_factor_points(fig: go.Figure, hold: np.ndarray, y_true, flagged, dx: int, dy: int) -> None:
    """4-way holdout marker layers (Pass / Flagged pass / Caught fail / Missed fail).

    The split makes the gate's catch/miss behaviour explicit. "Flagged pass" is the
    neutral name (not "false alarm"): a flagged passing wafer can be a genuine
    process excursion that simply did not cause a yield fail.
    """
    y_arr = np.asarray(y_true, dtype=int)
    flag_arr = np.asarray(flagged, dtype=bool)
    fail = y_arr == 1
    groups = [
        ("Pass", ~fail & ~flag_arr, C_BLUE, "circle", 6),
        ("Flagged pass", ~fail & flag_arr, C_ORANGE, "diamond", 8),
        ("Caught fail", fail & flag_arr, C_RED, "x", 11),
        ("Missed fail", fail & ~flag_arr, C_PURPLE, "circle-open", 11),
    ]
    for label, mask, color, symbol, size in groups:
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=hold[mask, dx],
                y=hold[mask, dy],
                mode="markers",
                name=f"Holdout - {label}",
                marker=dict(color=color, size=size, symbol=symbol, line=dict(width=1.0, color=color)),
                hovertemplate=f"{label}<extra></extra>",
            )
        )


def fig_pca_component_space(
    reference_scores: np.ndarray,
    holdout_scores: np.ndarray,
    y_true: np.ndarray,
    flagged: np.ndarray,
    *,
    dims: tuple[int, int] = (0, 1),
    score_mean: np.ndarray | None = None,
    score_cov: np.ndarray | None = None,
    t2_alpha: float = 0.03,
    factor_labels: tuple[str, str] | None = None,
    title: str = "PCA component space",
) -> go.Figure:
    """2-D scatter of two PCA components with the Hotelling T² control ellipse overlaid.

    The fab-standard baseline analogue of ``fig_sbfa_factor_space``: faint
    reference points = passing-train envelope, holdout points colored Pass / Fail /
    Flagged, and a single bivariate χ² (Hotelling) control ellipse at ``t2_alpha``
    for the two plotted components. Drift = the holdout cloud sliding outside the
    control region.
    """
    from scipy.stats import chi2

    fig = go.Figure()
    ref = np.asarray(reference_scores, dtype=float)
    hold = np.asarray(holdout_scores, dtype=float)
    dx, dy = dims
    if ref.ndim != 2 or hold.ndim != 2 or ref.shape[1] <= max(dims):
        return _sized(fig.update_layout(title=dict(text=title)), height=520, margin=dict(l=56, r=48, t=72, b=48))

    if score_mean is not None and score_cov is not None:
        mean = np.asarray(score_mean, dtype=float)
        cov = np.asarray(score_cov, dtype=float)
        if mean.shape[0] > max(dims) and cov.shape[0] > max(dims):
            radius = float(np.sqrt(chi2.ppf(1.0 - t2_alpha, df=2)))
            path = _ellipse_path(mean[[dx, dy]], cov[np.ix_([dx, dy], [dx, dy])], n_sigma=radius)
            if path is not None:
                ex, ey = path
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        line=dict(color=C_YELLOW, width=1.5),
                        fill="toself",
                        fillcolor="rgba(216, 166, 87, 0.07)",
                        name=f"Hotelling T² limit (α={t2_alpha:g})",
                        hoverinfo="skip",
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=ref[:, dx],
            y=ref[:, dy],
            mode="markers",
            name="Reference (passing train)",
            marker=dict(color="rgba(125, 174, 163, 0.28)", size=4),
            hoverinfo="skip",
        )
    )
    _add_factor_points(fig, hold, y_true, flagged, dx, dy)

    xlab, ylab = factor_labels or (f"Component {dx + 1}", f"Component {dy + 1}")
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=xlab, gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        yaxis=dict(title=ylab, gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=520, margin=dict(l=56, r=48, t=72, b=48))


def fig_factor_drift(
    ranking: pd.DataFrame,
    *,
    title: str = "Per-factor drift (reference → holdout KS)",
) -> go.Figure:
    """Horizontal bars of per-factor KS drift; the top-drifting factor highlighted."""
    fig = go.Figure()
    if ranking is None or ranking.empty:
        return _sized(fig.update_layout(title=dict(text=title)), height=360, margin=dict(l=72, r=48, t=64, b=48))
    df = ranking.sort_values("ks_distance", ascending=True)
    colors = [C_RED if i == len(df) - 1 else C_BLUE for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df["ks_distance"],
            y=[f"Factor {int(f)}" for f in df["factor"]],
            orientation="h",
            marker_color=colors,
            customdata=df["top_sensors"],
            hovertemplate="KS=%{x:.3f}<br>top: %{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title="KS distance", gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(title=""),
    )
    return _sized(fig, height=max(280, 60 + 34 * len(df)), margin=dict(l=72, r=48, t=64, b=48))


def fig_sbfa_loadings_heatmap(
    loadings: np.ndarray,
    feature_names: list[str],
    *,
    factor_order: list[int] | None = None,
    highlight_factor: int | None = None,
    max_sensors: int = 25,
    title: str = "Sparse sBFA loadings (sensor × factor)",
) -> go.Figure:
    """Diverging sensor×factor loadings heatmap, factors drift-ordered.

    Only the strongest-loading sensors (by max |loading| across factors) are
    shown for legibility; the Laplace prior makes most loadings ~0.
    """
    fig = go.Figure()
    W = np.asarray(loadings, dtype=float)
    if W.ndim != 2 or W.size == 0 or not feature_names:
        return _sized(fig.update_layout(title=dict(text=title)), height=520, margin=dict(l=140, r=48, t=64, b=56))

    n_factors = W.shape[1]
    order = [f for f in (factor_order or list(range(n_factors))) if 0 <= f < n_factors]
    if not order:
        order = list(range(n_factors))

    sensor_strength = np.abs(W).max(axis=1)
    keep = np.argsort(sensor_strength)[::-1][:max_sensors]
    keep = keep[np.argsort(sensor_strength[keep])]  # ascending so strongest on top
    z = W[np.ix_(keep, order)]
    sensors = [str(feature_names[i]) for i in keep]
    col_labels = [
        f"F{f + 1}★" if (highlight_factor is not None and f == highlight_factor) else f"F{f + 1}"
        for f in order
    ]
    vmax = float(np.abs(z).max()) or 1.0
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=col_labels,
            y=sensors,
            colorscale=COLORSCALE_DRIFT_DIVERGING,
            zmid=0.0,
            zmin=-vmax,
            zmax=vmax,
            colorbar=dict(title="loading"),
            hovertemplate="%{y} · %{x}<br>loading=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title="latent factor (drift-ordered, ★ = top)", side="top"),
        yaxis=dict(title=""),
    )
    return _sized(fig, height=max(420, 60 + 22 * len(sensors)), margin=dict(l=140, r=48, t=72, b=48))


def _disagreement_quadrants(
    fig: go.Figure,
    t2: np.ndarray,
    dens: np.ndarray,
    *,
    t2_ucl: float,
    density_lcl: float,
    sizes: tuple[int, int, int, int] = (5, 7, 7, 9),
    opacity: float = 1.0,
    suffix: str = "",
) -> None:
    """Add the four quadrant traces (by which gate abstains) for one wafer cloud."""
    pca_ooc = t2 > t2_ucl
    bgm_ooc = dens < density_lcl
    groups = [
        (~pca_ooc & ~bgm_ooc, "Both in-control", C_BLUE, sizes[0]),
        (pca_ooc & bgm_ooc, "Both flag OOC", C_RED, sizes[1]),
        (~pca_ooc & bgm_ooc, "BGM only", C_ORANGE, sizes[2]),
        (pca_ooc & ~bgm_ooc, "PCA flags, BGM clears", C_GREEN, sizes[3]),
    ]
    for mask, name, color, size in groups:
        if not mask.any():
            continue
        n = int(mask.sum())
        fig.add_trace(
            go.Scatter(
                x=t2[mask],
                y=dens[mask],
                mode="markers",
                name=f"{name}{suffix} (n={n})",
                marker=dict(color=color, size=size, opacity=opacity, line=dict(width=0)),
                hovertemplate="T²=%{x:.2f}<br>log-density=%{y:.2f}<extra></extra>",
            )
        )


def fig_gate_disagreement_scatter(
    pca_t2: np.ndarray,
    bgm_density: np.ndarray,
    *,
    t2_ucl: float,
    density_lcl: float,
    holdout_t2: np.ndarray | None = None,
    holdout_density: np.ndarray | None = None,
    title: str = "Where the gates disagree (passing wafers)",
) -> go.Figure:
    """Per-wafer Hotelling T2 (x) vs BGM log-density (y), split by quadrant.

    The control limits draw four quadrants. The money quadrant is high-T2 /
    high-density: healthy wafers PCA flags out-of-control (above its single
    Hotelling ellipse) but the multimodal BGM keeps in-control; colour encodes
    which gate(s) would abstain.

    When ``holdout_t2`` / ``holdout_density`` are supplied, the pre-drift
    reference is rendered as a single muted grey ghost cloud and the temporal
    holdout is drawn quadrant-coloured at full opacity, so the downward drift in
    BGM log-density (the cloud sinking below the density limit) is visible.
    """
    fig = go.Figure()
    t2 = np.asarray(pca_t2, dtype=float)
    dens = np.asarray(bgm_density, dtype=float)
    if t2.size == 0 or dens.size == 0 or t2.size != dens.size:
        return _sized(fig.update_layout(title=dict(text=title)), height=520, margin=dict(l=64, r=48, t=72, b=56))

    ho_t2 = np.asarray(holdout_t2, dtype=float) if holdout_t2 is not None else np.empty(0)
    ho_dens = np.asarray(holdout_density, dtype=float) if holdout_density is not None else np.empty(0)
    has_holdout = ho_t2.size > 0 and ho_t2.size == ho_dens.size

    if has_holdout:
        # Reference becomes a muted grey ghost cloud; the holdout carries the
        # quadrant colours so the drift-down reads as the focus.
        fig.add_trace(
            go.Scatter(
                x=t2,
                y=dens,
                mode="markers",
                name=f"Pre-drift reference (n={t2.size})",
                marker=dict(color="rgba(150,150,150,0.22)", size=5, line=dict(width=0)),
                hovertemplate="reference<br>T²=%{x:.2f}<br>log-density=%{y:.2f}<extra></extra>",
            )
        )
        _disagreement_quadrants(
            fig, ho_t2, ho_dens, t2_ucl=t2_ucl, density_lcl=density_lcl, suffix=" (holdout)"
        )
    else:
        _disagreement_quadrants(fig, t2, dens, t2_ucl=t2_ucl, density_lcl=density_lcl)

    fig.add_vline(x=float(t2_ucl), line=dict(color=C_YELLOW, width=1.5, dash="dash"))
    fig.add_hline(y=float(density_lcl), line=dict(color=C_YELLOW, width=1.5, dash="dash"))
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title="PCA Hotelling T² (→ PCA abstains right of line)", gridcolor="rgba(200,200,200,0.15)"),
        yaxis=dict(title="BGM log-density (→ BGM abstains below line)", gridcolor="rgba(200,200,200,0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=520, margin=dict(l=64, r=48, t=84, b=56))


def fig_bgm_mode_weights(
    weights: np.ndarray,
    *,
    active_threshold: float = 0.05,
    title: str = "BGM in-control modes (mixture weights)",
) -> go.Figure:
    """Bar of BGM mixture weights; bars above the active threshold are real modes.

    More than one active component is direct evidence the in-control region is
    multimodal - a structure a single PCA Hotelling ellipse cannot represent.
    """
    fig = go.Figure()
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        return _sized(fig.update_layout(title=dict(text=title)), height=360, margin=dict(l=56, r=48, t=64, b=48))
    order = np.argsort(w)[::-1]
    w = w[order]
    labels = [f"Mode {i + 1}" for i in range(w.size)]
    colors = [C_GREEN if v >= active_threshold else "rgba(125, 174, 163, 0.3)" for v in w]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=w,
            marker_color=colors,
            hovertemplate="%{x}<br>weight=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=float(active_threshold),
        line=dict(color=C_YELLOW, width=1.5, dash="dash"),
        annotation_text=f"active ≥ {active_threshold:g}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=""),
        yaxis=dict(title="mixture weight", gridcolor="rgba(200,200,200,0.15)"),
    )
    return _sized(fig, height=360, margin=dict(l=56, r=48, t=64, b=48))


def fig_sensor_noise_spectrum(
    psi: np.ndarray,
    feature_names: list[str],
    *,
    max_sensors: int = 30,
    title: str = "Per-sensor sBFA noise variance Ψ (log scale)",
) -> go.Figure:
    """Sorted per-sensor noise variance Ψ on a log axis - the heteroscedasticity.

    A wide spread (noisiest / quietest spanning 1-2 orders of magnitude) is why
    equal-weight PCA contributions are misleading: noisy sensors dominate the
    residual purely because they are noisy, not because they drifted.
    """
    fig = go.Figure()
    p = np.asarray(psi, dtype=float).ravel()
    if p.size == 0 or not feature_names or len(feature_names) != p.size:
        return _sized(fig.update_layout(title=dict(text=title)), height=420, margin=dict(l=140, r=48, t=64, b=48))
    order = np.argsort(p)[::-1][:max_sensors]
    order = order[np.argsort(p[order])]  # ascending so noisiest on top
    vals = p[order]
    sensors = [str(feature_names[i]) for i in order]
    spread = float(p.max() / p[p > 0].min()) if np.any(p > 0) else 1.0
    fig.add_trace(
        go.Bar(
            x=vals,
            y=sensors,
            orientation="h",
            marker_color=C_ORANGE,
            hovertemplate="%{y}<br>Ψ=%{x:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"{title} — spread ≈ {spread:.0f}×"),
        xaxis=dict(title="noise variance Ψ", type="log", gridcolor="rgba(200,200,200,0.15)"),
        yaxis=dict(title=""),
    )
    return _sized(fig, height=max(360, 60 + 22 * len(sensors)), margin=dict(l=140, r=48, t=64, b=48))


def fig_contribution_comparison(
    pca_resid: np.ndarray,
    sbfa_resid: np.ndarray,
    psi: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 10,
    title: str = "Per-sensor contribution: equal-weight (PCA) vs noise-weighted (sBFA)",
) -> go.Figure:
    """Paired top-K sensor bars contrasting the two contribution weightings.

    Equal-weight (PCA) contribution is ``resid²``; noise-weighted (sBFA) is
    ``resid² / Ψ``. Both normalised to their own max so the *ranking* is the
    story: an intrinsically noisy sensor can top the PCA ranking yet fall under
    noise-weighting, while a quiet sensor's genuine deviation rises.
    """
    fig = go.Figure()
    rp = np.asarray(pca_resid, dtype=float).ravel()
    rs = np.asarray(sbfa_resid, dtype=float).ravel()
    ps = np.asarray(psi, dtype=float).ravel()
    if rp.size == 0 or rs.size != rp.size or ps.size != rp.size or len(feature_names) != rp.size:
        return _sized(fig.update_layout(title=dict(text=title)), height=440, margin=dict(l=140, r=48, t=72, b=48))

    pca_contrib = rp**2
    sbfa_contrib = rs**2 / np.clip(ps, 1e-12, None)
    pca_norm = pca_contrib / (pca_contrib.max() or 1.0)
    sbfa_norm = sbfa_contrib / (sbfa_contrib.max() or 1.0)

    # Union of each weighting's top-K, ordered by the noise-weighted contribution.
    top_pca = set(np.argsort(pca_norm)[::-1][:top_k].tolist())
    top_sbfa = set(np.argsort(sbfa_norm)[::-1][:top_k].tolist())
    keep = sorted(top_pca | top_sbfa, key=lambda i: sbfa_norm[i])
    sensors = [str(feature_names[i]) for i in keep]
    fig.add_trace(
        go.Bar(
            y=sensors,
            x=[pca_norm[i] for i in keep],
            orientation="h",
            name="Equal-weight (PCA)",
            marker_color=C_BLUE,
            hovertemplate="%{y}<br>PCA contrib=%{x:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=sensors,
            x=[sbfa_norm[i] for i in keep],
            orientation="h",
            name="Noise-weighted (sBFA)",
            marker_color=C_GREEN,
            hovertemplate="%{y}<br>sBFA contrib=%{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title),
        barmode="group",
        xaxis=dict(title="normalised contribution (each to its own max)", gridcolor="rgba(200,200,200,0.15)"),
        yaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=max(420, 80 + 26 * len(sensors)), margin=dict(l=140, r=48, t=84, b=48))
