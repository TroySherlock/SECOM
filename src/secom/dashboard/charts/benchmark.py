"""Benchmark / operating-point charts (pages 3-4 + model deep-dive)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from secom.costs import ESCAPE_OVERKILL_COST_RATIO
from secom.dashboard.charts._base import (
    C_AQUA,
    C_BLUE,
    C_GREEN,
    C_ORANGE,
    C_PURPLE,
    C_RED,
    C_YELLOW,
    CI_YELLOW_RGBA,
    COLORSCALE_LOW_GREEN_HIGH_RED,
    CV_ERROR_PURPLE_RGBA,
    C,
    _sized,
)
from secom.metrics import PRCurve


def fig_benchmark_leaderboard(
    df: pd.DataFrame,
    *,
    metric_col: str,
    label_col: str = "pipeline",
    error_col: str | None = None,
    title: str = "Benchmark leaderboard",
    ascending: bool = False,
    marker_color: str | None = None,
) -> go.Figure:
    """Horizontal leaderboard of point estimates with ±1 SD error bars."""
    if df.empty:
        return _sized(go.Figure(), height=360)

    plot_df = df.sort_values(metric_col, ascending=ascending).copy().reset_index(drop=True)
    color = marker_color or C_PURPLE
    error_rgba = CV_ERROR_PURPLE_RGBA if color == C_PURPLE else "rgba(200, 200, 200, 0.6)"

    error_x = None
    if error_col and error_col in plot_df.columns:
        error_x = dict(
            type="data",
            array=plot_df[error_col].astype(float),
            visible=True,
            color=error_rgba,
            width=5,
            thickness=1.5,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df[metric_col],
            y=plot_df[label_col],
            mode="markers",
            marker=dict(color=color, size=10, symbol="circle"),
            error_x=error_x,
            name="CV Fold Spread (Mean ± 1 SD)",
            hovertemplate="%{y}<br>mean=%{x:.3f}<extra></extra>",
        )
    )

    xaxis_title = metric_col.replace("_", " ").replace("mean ", "Mean ")
    xaxis_kwargs: dict = dict(
        gridcolor="rgba(200, 200, 200, 0.15)",
        zeroline=False,
    )
    if "auc" in metric_col.lower():
        xaxis_kwargs["range"] = [0, 1.0]

    fig.update_layout(
        title=dict(text=title),
        xaxis_title=xaxis_title,
        xaxis=xaxis_kwargs,
        yaxis=dict(autorange="reversed", gridcolor="rgba(200, 200, 200, 0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        showlegend=error_x is not None,
    )
    return _sized(fig, height=380, margin=dict(l=100, r=48, t=72, b=48))


def fig_cv_vs_holdout_validation(
    cv_df: pd.DataFrame,
    ho_df: pd.DataFrame,
    *,
    metric_type: str = "PR-AUC",
    show_ci: bool = True,
) -> go.Figure:
    """Overlay CV fold spread, holdout point estimates, and bootstrap CIs per pipeline."""
    if cv_df.empty or ho_df.empty:
        return _sized(go.Figure(), height=360)

    df = pd.merge(cv_df, ho_df, on="pipeline", suffixes=("_cv", "_ho"))

    if metric_type == "PR-AUC":
        cv_mean, cv_std, ho_val = "mean_pr_auc", "std_pr_auc", "pr_auc"
        ci_low, ci_high = "pr_auc_ci_low", "pr_auc_ci_high"
    else:
        cv_mean, cv_std, ho_val = "mean_roc_auc", "std_roc_auc", "roc_auc"
        ci_low = "roc_auc_ci_low" if "roc_auc_ci_low" in df.columns else None
        ci_high = "roc_auc_ci_high" if "roc_auc_ci_high" in df.columns else None

    df = df.sort_values(ho_val, ascending=False).reset_index(drop=True)

    fig = go.Figure()
    ci_available = (
        show_ci
        and ci_low
        and ci_high
        and ci_low in df.columns
        and ci_high in df.columns
    )

    if ci_available:
        for idx, row in df.iterrows():
            low_val = row.get(ci_low)
            high_val = row.get(ci_high)
            if pd.notna(low_val) and pd.notna(high_val):
                fig.add_trace(
                    go.Scatter(
                        x=[low_val, high_val],
                        y=[row["pipeline"], row["pipeline"]],
                        mode="lines",
                        line=dict(color=CI_YELLOW_RGBA, width=10),
                        name="Holdout 95% Bootstrap CI",
                        showlegend=idx == 0,
                        hoverinfo="skip",
                    )
                )

    fig.add_trace(
        go.Scatter(
            x=df[cv_mean],
            y=df["pipeline"],
            mode="markers",
            marker=dict(color=C_PURPLE, size=10, symbol="circle"),
            error_x=dict(
                type="data",
                array=df[cv_std],
                visible=True,
                color=CV_ERROR_PURPLE_RGBA,
                width=5,
                thickness=1.5,
            ),
            name="CV Fold Spread (Mean ± 1 SD)",
            hovertemplate=(
                "%{y}<br>CV mean=%{x:.3f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[ho_val],
            y=df["pipeline"],
            mode="markers",
            marker=dict(
                color=C_YELLOW,
                size=12,
                symbol="diamond",
                line=dict(color=C_ORANGE, width=1),
            ),
            name="Final Holdout Score",
            hovertemplate="%{y}<br>Holdout=%{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"CV vs Holdout Validation ({metric_type})"),
        xaxis_title=f"Metric Value ({metric_type})",
        xaxis=dict(range=[0, 1.0], gridcolor="rgba(200, 200, 200, 0.15)", zeroline=False),
        yaxis=dict(autorange="reversed", gridcolor="rgba(200, 200, 200, 0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    return _sized(fig, height=380, margin=dict(l=100, r=48, t=72, b=48))


def fig_holdout_confusion(
    confusion_matrix: list[list[int]],
    pipeline_name: str,
    *,
    height: int = 320,
) -> go.Figure:
    """Confusion matrix heatmap for holdout evaluation."""
    cm = np.asarray(confusion_matrix)
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{labels[i][j]}<br>{cm[i, j]}" for j in range(2)] for i in range(2)]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred pass (0)", "Pred fail (1)"],
            y=["Actual pass (0)", "Actual fail (1)"],
            colorscale=COLORSCALE_LOW_GREEN_HIGH_RED,
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}, %{x}<br>count=%{z}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        title=dict(text=f"Holdout confusion matrix — {pipeline_name}"),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    fig.update_yaxes(autorange="reversed")
    return _sized(fig, height=height)


def _pr_axis_limits(
    cv_curve: PRCurve | None,
    ho_curve: PRCurve | None,
    ber_point: tuple[float, float] | None,
) -> tuple[float, float]:
    recall_vals: list[float] = []
    precision_vals: list[float] = []
    for curve in (cv_curve, ho_curve):
        if curve is not None and curve.recall.size:
            recall_vals.append(float(np.max(curve.recall)))
            precision_vals.append(float(np.max(curve.precision)))
    if ber_point is not None:
        recall_vals.append(float(ber_point[0]))
        precision_vals.append(float(ber_point[1]))
    recall_hi = min(1.0, (max(recall_vals) if recall_vals else 0.5) * 1.10 + 0.05)
    precision_hi = min(1.0, (max(precision_vals) if precision_vals else 0.5) * 1.15 + 0.02)
    return recall_hi, precision_hi


def fig_pr_curve_clean(
    curve: PRCurve | None,
    *,
    ap: float | None = None,
    ap_ci: tuple[float, float] | None = None,
    ber_point: tuple[float, float] | None = None,
    baseline: float | None = None,
    draw_line: bool = True,
    line_name: str = "CV (out-of-fold)",
    title: str = "Precision–recall curve",
) -> go.Figure:
    """Decluttered PR view: one line max, one operating point, AP as a number.

    ``curve`` is the single smooth curve to draw (CV-OOF for interpolation);
    pass ``draw_line=False`` (extrapolation) to suppress the jagged small-sample
    staircase and show only the BER operating point against the prevalence
    baseline. Holdout AP (+ optional CI) is annotated as text, not a band.
    """
    fig = go.Figure()
    if draw_line and curve is not None and curve.recall.size:
        fig.add_trace(
            go.Scatter(
                x=curve.recall,
                y=curve.precision,
                mode="lines",
                name=line_name,
                line=dict(color=C_PURPLE, width=2.5, shape="hv"),
                hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>",
            )
        )

    if baseline is None and curve is not None:
        baseline = curve.baseline
    if baseline is not None:
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color=C_BLUE,
            annotation_text="Random baseline",
            annotation_position="right",
        )

    if ber_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[ber_point[0]],
                y=[ber_point[1]],
                mode="markers",
                name="BER-min (holdout)",
                marker=dict(color=C_GREEN, size=13, symbol="diamond"),
                hovertemplate=(
                    "BER-min threshold<br>Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra></extra>"
                ),
            )
        )

    if ap is not None:
        ap_txt = f"Holdout AP = {ap:.3f}"
        if ap_ci is not None and all(v is not None for v in ap_ci):
            ap_txt += f"  [{ap_ci[0]:.3f}, {ap_ci[1]:.3f}]"
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=ap_txt,
            showarrow=False,
            align="right",
            font=dict(size=13, color=C_YELLOW),
            bgcolor="rgba(0,0,0,0.25)",
            borderpad=4,
        )

    recall_hi, precision_hi = _pr_axis_limits(curve if draw_line else None, None, ber_point)
    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, recall_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(range=[0, precision_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=440, margin=dict(l=56, r=48, t=72, b=48))


def _threshold_grid(y_score: np.ndarray, *, cap: int = 400) -> np.ndarray:
    """Unique score grid (with 0/1 endpoints) for sweeping the decision threshold."""
    grid = np.unique(np.clip(y_score, 0.0, 1.0))
    if grid.size > cap:
        grid = np.quantile(grid, np.linspace(0.0, 1.0, cap))
    return np.unique(np.concatenate([[0.0], grid, [1.0]]))


def _tpr_fpr_curve(
    y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """TPR (catch) and FPR (overkill) as fractions at each threshold in ``grid``."""
    pos = y_true == 1
    neg = ~pos
    n_pos = max(int(pos.sum()), 1)
    n_neg = max(int(neg.sum()), 1)
    tpr = np.empty(grid.size, dtype=float)
    fpr = np.empty(grid.size, dtype=float)
    for i, t in enumerate(grid):
        pred = y_score >= t
        tpr[i] = float(np.sum(pred & pos)) / n_pos
        fpr[i] = float(np.sum(pred & neg)) / n_neg
    return tpr, fpr


def _rates_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, thr: float
) -> tuple[float, float]:
    """(catch rate, overkill rate) as fractions at a single threshold."""
    tpr, fpr = _tpr_fpr_curve(y_true, y_score, np.array([float(thr)]))
    return float(tpr[0]), float(fpr[0])


_PROFILE_MARKER_COLORS = (C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_RED, C_YELLOW)


def fig_catch_overkill_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    profile_thresholds: dict[str, float] | None = None,
    title: str = "Catch rate vs overkill",
) -> go.Figure:
    """Fab-vocabulary operating curve: x = overkill rate (FPR), y = catch rate (TPR).

    Same information as an ROC curve, relabelled for a fab: how many real fails
    you catch versus how many good wafers you wrongly flag. The tuned operating
    points (conservative / BER-min / aggressive / economic) are marked.
    """
    fig = go.Figure()
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or y_score.size == 0 or len(np.unique(y_true)) < 2:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=420,
            margin=dict(l=56, r=48, t=72, b=48),
        )

    grid = _threshold_grid(y_score)
    tpr, fpr = _tpr_fpr_curve(y_true, y_score, grid)
    order = np.argsort(fpr, kind="mergesort")
    fig.add_trace(
        go.Scatter(
            x=100 * fpr[order],
            y=100 * tpr[order],
            mode="lines",
            name="Operating curve",
            line=dict(color=C_AQUA, width=2.5),
            hovertemplate="overkill=%{x:.1f}%<br>catch=%{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Random",
            line=dict(color="rgba(200, 200, 200, 0.4)", dash="dash", width=1),
            hoverinfo="skip",
        )
    )
    for idx, (label, thr) in enumerate((profile_thresholds or {}).items()):
        if thr is None or not np.isfinite(thr):
            continue
        catch, overkill = _rates_at_threshold(y_true, y_score, float(thr))
        fig.add_trace(
            go.Scatter(
                x=[100 * overkill],
                y=[100 * catch],
                mode="markers",
                name=label,
                marker=dict(
                    size=13,
                    color=_PROFILE_MARKER_COLORS[idx % len(_PROFILE_MARKER_COLORS)],
                    symbol="circle",
                    line=dict(color="#282828", width=1.5),
                ),
                hovertemplate=(
                    f"{label}<br>overkill=%{{x:.1f}}%<br>catch=%{{y:.1f}}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Overkill rate — good wafers flagged (%)",
        yaxis_title="Catch rate — real fails flagged (%)",
        xaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)", range=[0, 100]),
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=420, margin=dict(l=56, r=48, t=72, b=48))


def fig_expected_cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    prevalence: float | None = None,
    cost_ratio: float = ESCAPE_OVERKILL_COST_RATIO,
    profile_thresholds: dict[str, float] | None = None,
    title: str = "Expected cost vs threshold",
) -> go.Figure:
    """Expected per-wafer cost across the threshold grid at an escape:overkill ratio.

    Cost is in overkill-units: ``cost_ratio*prevalence*(1-TPR) + (1-prevalence)*(1-TNR)``.
    The empirical minimum on this holdout is marked, and each tuned operating
    point is drawn as a colour-coded vertical line + on-curve dot (colours match
    ``fig_catch_overkill_curve``). The x-axis is log-scaled and zoomed to the
    operating region so the basin and the closely-spaced low thresholds read clearly.
    """
    fig = go.Figure()
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or y_score.size == 0 or len(np.unique(y_true)) < 2:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=420,
            margin=dict(l=56, r=48, t=72, b=48),
        )

    p = float(np.mean(y_true)) if prevalence is None else float(prevalence)
    grid = _threshold_grid(y_score)
    tpr, fpr = _tpr_fpr_curve(y_true, y_score, grid)
    tnr = 1.0 - fpr
    cost = cost_ratio * p * (1.0 - tpr) + (1.0 - p) * (1.0 - tnr)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=cost,
            mode="lines",
            name=f"Expected cost ({cost_ratio:g}:1)",
            line=dict(color=C_AQUA, width=2.5),
            hovertemplate="threshold=%{x:.4f}<br>cost=%{y:.3f}<extra></extra>",
        )
    )
    imin = int(np.argmin(cost))
    fig.add_trace(
        go.Scatter(
            x=[grid[imin]],
            y=[cost[imin]],
            mode="markers",
            name="Cost-min (empirical)",
            marker=dict(color=C_GREEN, size=12, symbol="diamond"),
            hovertemplate="cost-min<br>threshold=%{x:.4f}<br>cost=%{y:.3f}<extra></extra>",
        )
    )

    # Colour-coded operating points: one legend line per profile (no overlapping
    # in-plot text) plus a matching dot on the curve so each point's cost reads off.
    cost_lo, cost_hi = float(np.min(cost)), float(np.max(cost))
    pad = 0.04 * (cost_hi - cost_lo or 1.0)
    points = [float(grid[imin])]
    for idx, (label, thr) in enumerate((profile_thresholds or {}).items()):
        if thr is None or not np.isfinite(thr) or float(thr) <= 0.0:
            continue
        thr = float(thr)
        points.append(thr)
        color = _PROFILE_MARKER_COLORS[idx % len(_PROFILE_MARKER_COLORS)]
        catch, overkill = _rates_at_threshold(y_true, y_score, thr)
        cost_at = cost_ratio * p * (1.0 - catch) + (1.0 - p) * overkill
        fig.add_trace(
            go.Scatter(
                x=[thr, thr],
                y=[cost_lo - pad, cost_hi + pad],
                mode="lines",
                name=label,
                line=dict(color=color, width=1.6, dash="dot"),
                hovertemplate=f"{label}<br>threshold={thr:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[thr],
                y=[cost_at],
                mode="markers",
                marker=dict(color=color, size=11, line=dict(color="#282828", width=1.2)),
                showlegend=False,
                hovertemplate=f"{label}<br>threshold={thr:.4f}<br>cost=%{{y:.3f}}<extra></extra>",
            )
        )

    # Zoom + log scale so the operating region is not squished into the left edge.
    lo = max(1e-4, 0.5 * min(points))
    hi = min(float(np.max(y_score)), 1.5 * max(points))
    if not (hi > lo):
        hi = lo * 10.0
    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Decision threshold (log scale)",
        yaxis_title="Expected cost per wafer (overkill units)",
        xaxis=dict(
            type="log",
            range=[float(np.log10(lo)), float(np.log10(hi))],
            gridcolor="rgba(200, 200, 200, 0.15)",
        ),
        yaxis=dict(
            range=[cost_lo - pad, cost_hi + pad],
            gridcolor="rgba(200, 200, 200, 0.15)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=420, margin=dict(l=56, r=48, t=72, b=48))


def fig_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 5,
    title: str = "Calibration (reliability) curve",
) -> go.Figure:
    """Reliability curve (binned observed vs predicted) + diagonal; Brier in title.

    Axes are zoomed to the predicted-probability range actually emitted (a
    sigmoid-calibrated model on imbalanced data predicts mostly small
    probabilities, so a fixed [0,1] view looks empty). A faint predicted-score
    histogram on a secondary axis shows where the mass sits.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    fig = go.Figure()
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or y_score.size == 0 or len(np.unique(y_true)) < 2:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=420,
            margin=dict(l=56, r=48, t=72, b=48),
        )

    brier = float(brier_score_loss(y_true, y_score))
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="quantile"
        )
    except ValueError:
        prob_true, prob_pred = np.array([]), np.array([])

    # Zoom to the data range so the curve fills the panel.
    hi = float(np.max(y_score)) if y_score.size else 1.0
    if prob_pred.size:
        hi = max(hi, float(np.max(prob_pred)), float(np.max(prob_true)))
    axis_hi = float(min(1.0, hi * 1.1 + 0.02))

    # Predicted-score histogram (secondary axis) to show where the mass sits.
    fig.add_trace(
        go.Histogram(
            x=y_score,
            nbinsx=40,
            name="Predicted scores",
            marker=dict(color=C_BLUE),
            opacity=0.25,
            yaxis="y2",
            hovertemplate="score=%{x:.3f}<br>count=%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, axis_hi],
            y=[0, axis_hi],
            mode="lines",
            name="Perfect calibration",
            line=dict(color=C_BLUE, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    if prob_pred.size:
        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="Model",
                line=dict(color=C_PURPLE, width=2.5),
                marker=dict(size=8),
                hovertemplate="predicted=%{x:.3f}<br>observed=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(text=f"{title} — Brier = {brier:.3f}"),
        xaxis_title="Mean predicted fail probability",
        yaxis_title="Observed fail fraction",
        xaxis=dict(range=[0, axis_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(range=[0, axis_hi], gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            title="Predicted-score count",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=420, margin=dict(l=56, r=48, t=72, b=48))


def fig_risk_coverage(
    df: pd.DataFrame,
    *,
    metric: str = "pr_auc",
    operating_coverage: float | None = None,
    title: str = "Gate risk–coverage curve",
) -> go.Figure:
    """One line per pipeline of conditional AUC vs gate coverage.

    Coverage runs high→low left→right so abstention (dropping the most suspicious
    wafers) increases to the right. A dashed vertical line marks the gate's actual
    operating coverage. ``metric`` is ``"pr_auc"`` or ``"roc_auc"``.
    """
    fig = go.Figure()
    if df is None or df.empty or metric not in df.columns:
        return _sized(
            fig.update_layout(title=dict(text=title)),
            height=440,
            margin=dict(l=56, r=48, t=72, b=48),
        )

    metric_label = "PR-AUC" if metric == "pr_auc" else "ROC-AUC"
    pipelines = sorted(df["pipeline"].unique())
    for i, name in enumerate(pipelines):
        sub = df[df["pipeline"] == name].sort_values("coverage", ascending=False)
        fig.add_trace(
            go.Scatter(
                x=sub["coverage"],
                y=sub[metric],
                customdata=sub[["n_kept", "n_fails_kept"]].to_numpy(),
                mode="lines+markers",
                name=name,
                line=dict(color=C[i % len(C)], width=2.5),
                marker=dict(size=6),
                connectgaps=False,
                hovertemplate=(
                    f"{name}<br>Coverage=%{{x:.2f}}<br>{metric_label}=%{{y:.3f}}"
                    "<br>kept=%{customdata[0]} (fails=%{customdata[1]})<extra></extra>"
                ),
            )
        )

    if operating_coverage is not None:
        fig.add_vline(
            x=float(operating_coverage),
            line_dash="dash",
            line_color=C_BLUE,
            annotation_text=f"gate op. coverage {float(operating_coverage):.2f}",
            annotation_position="top",
        )

    fig.update_layout(
        title=dict(text=title),
        xaxis_title="Coverage (fraction of wafers kept)",
        yaxis_title=f"Conditional {metric_label}",
        xaxis=dict(autorange="reversed", gridcolor="rgba(200, 200, 200, 0.15)"),
        yaxis=dict(gridcolor="rgba(200, 200, 200, 0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return _sized(fig, height=440, margin=dict(l=56, r=48, t=72, b=48))


def fig_delta_bar(
    df: pd.DataFrame,
    *,
    value_col: str = "delta",
    label_col: str = "pipeline",
    title: str = "Delta",
    value_label: str = "delta",
    positive_is_good: bool = True,
    height: int = 380,
) -> go.Figure:
    """Diverging horizontal bars for signed deltas, sorted by magnitude.

    Green/red encode whether a bar is desirable: when ``positive_is_good`` is
    True, positive bars are green; flip it for loss-style deltas (e.g. the
    interpolation-minus-extrapolation drift cost) so larger losses read red.
    """
    if df.empty or value_col not in df:
        return _sized(go.Figure(), height=height, title=dict(text=title))
    plot_df = df.dropna(subset=[value_col]).copy()
    if plot_df.empty:
        return _sized(go.Figure(), height=height, title=dict(text=title))
    plot_df = plot_df.reindex(
        plot_df[value_col].abs().sort_values(ascending=True).index
    )
    good = C_GREEN if positive_is_good else C_RED
    bad = C_RED if positive_is_good else C_GREEN
    colors = [good if v >= 0 else bad for v in plot_df[value_col]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df[value_col],
                y=plot_df[label_col],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in plot_df[value_col]],
                textposition="outside",
                hovertemplate="%{y}<br>" + value_label + "=%{x:+.4g}<extra></extra>",
            )
        ]
    )
    fig.add_vline(x=0.0, line_width=1, line_color="rgba(235,235,235,0.45)")
    fig.update_layout(
        title=dict(text=title),
        xaxis_title=value_label,
        yaxis_title="",
        showlegend=False,
    )
    return _sized(fig, height=height, margin=dict(l=150, r=64, t=72, b=48))
