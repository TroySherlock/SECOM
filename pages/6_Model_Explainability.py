"""Root-cause-analysis (RCA) explainability for the SECOM defect-detection grid.

One champion per track (pls_bayes for the temporal/extrapolation track, hsic_rf
for the random/interpolation track). The page leads with deterministic KEY
FINDINGS, then offers three focused views: a per-wafer RCA, the champion's global
drivers, and a constrained plain-English interpretation. PLS attributions are
back-projected through the fitted loadings into one sensor vocabulary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from secom.dashboard import render_blue_note
from secom.dashboard.charts import (
    fig_local_contributions,
    fig_posterior_forest,
    fig_spc_distribution,
    fig_top_features_bar,
)
from secom.dashboard.data import model_info
from secom.dashboard.explainability import (
    OUTCOME_EMOJI,
    OUTCOME_LABELS,
    cached_bayes_hdis,
    cached_global_sensor_importance,
    cached_holdout_outcomes,
    cached_pls_sensor_posterior,
    cached_pls_sensor_robust_map,
    cached_wafer_explanation,
    champion_for_track,
    is_pls_model,
    key_findings,
    load_holdout_split,
    wafer_sensor_posterior,
)
from secom.dashboard.narrator import (
    build_wafer_facts,
    get_wafer_narrative,
    load_narratives,
    narrative_model_for_track,
    narratives_path_for_track,
)
from secom.pipelines import ID_COL, TARGET_COL

_TRACK_LABELS = {
    "Extrapolation (temporal)": "extrapolation",
    "Interpolation (random)": "interpolation",
}
_OUTCOME_FILTERS = {
    "All wafers": None,
    "🔴 Missed fails": "missed_fail",
    "🟠 False alarms": "false_alarm",
    "🟢 Caught fails": "caught_fail",
    "⚪ Correct pass": "correct_pass",
}
_SPC_PLOTS = 3


# --- caching wrappers --------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_track_narratives(track: str) -> dict | None:
    model_id = narrative_model_for_track(track)
    path = narratives_path_for_track(track)
    try:
        return load_narratives(path, expected_model_id=model_id)
    except FileNotFoundError:
        return None
    except (ValueError, OSError):
        return None


# --- header ------------------------------------------------------------------
def _render_header() -> None:
    st.title("Model explainability — root-cause analysis")
    st.caption(
        "An RCA workflow: **symptom** (a wafer the model flags) -> **evidence** "
        "(which sensors drove it, how unusual they are vs the in-control baseline, "
        "whether they are drifting) -> **hypothesis** (where to look) -> **action**. "
        "Each track shows its single champion model, expressed in one sensor "
        "vocabulary (PLS is back-projected from latent components via its loadings)."
    )
    st.warning(
        "**Honesty caveat — associational, not causal.** Every attribution is a "
        "model-derived hypothesis to guide investigation, not a proven root cause. "
        "SECOM ships no sensor-to-tool/chamber map, so there is no real-equipment "
        "grouping here.",
        icon="⚠️",
    )


def _track_word(track: str) -> str:
    return "temporal / extrapolation" if track == "extrapolation" else "random / interpolation"


# --- controls ----------------------------------------------------------------
def _render_controls() -> tuple[str, str, object]:
    c1, c2 = st.columns([2, 3])
    with c1:
        track_label = st.radio(
            "Track",
            options=list(_TRACK_LABELS),
            key="p6_track",
            help="Temporal = latest 20% by time (drift stress); random = in-distribution.",
        )
    track = _TRACK_LABELS[track_label]
    champion = champion_for_track(track)
    info = model_info(champion)
    with c1:
        st.markdown(f"**Champion for this track:** `{champion}` — {info.display_name}")

    outcomes = cached_holdout_outcomes(champion, track)
    outcome_map = dict(zip(outcomes["observation_id"], outcomes["outcome"]))

    with c2:
        f1, f2 = st.columns([2, 3])
        with f1:
            filt_label = st.radio(
                "Show",
                options=list(_OUTCOME_FILTERS),
                key="p6_outcome_filter",
                help="Filter the wafer picker by model verdict. Errors are listed first.",
            )
        filt = _OUTCOME_FILTERS[filt_label]
        sel = outcomes if filt is None else outcomes.loc[outcomes["outcome"] == filt]
        wafer_options = sel["observation_id"].tolist()

        def _fmt(wid: object) -> str:
            oc = outcome_map.get(wid, "")
            return f"{OUTCOME_EMOJI.get(oc, '')} {OUTCOME_LABELS.get(oc, '')} · {wid}"

        with f2:
            if not wafer_options:
                st.info("No wafers match this filter.")
                wafer_id = None
            else:
                wafer_id = st.selectbox(
                    "Wafer (holdout)",
                    wafer_options,
                    format_func=_fmt,
                    key=f"p6_wafer_{track}_{filt}",
                )
        st.caption(
            "Picker legend: 🔴 missed fail · 🟠 false alarm · 🟢 caught fail · "
            "⚪ correct pass. Errors are sorted to the top."
        )
    return champion, track, wafer_id


# --- key findings (deterministic, always on top) -----------------------------
def _annotate_robust(local_df: pd.DataFrame, model_id: str, track: str) -> pd.DataFrame:
    if model_info(model_id).explainability != "bayesian":
        return local_df
    if is_pls_model(model_id):
        robust_map = cached_pls_sensor_robust_map(model_id, track)
    else:
        hdi = cached_bayes_hdis(model_id, track)
        robust_map = (
            {str(r.feature): bool(r.robust) for r in hdi.itertuples()}
            if hdi is not None
            else {}
        )
    if not robust_map:
        return local_df
    out = local_df.copy()
    out["robust"] = out["feature"].astype(str).map(lambda f: robust_map.get(str(f)))
    return out


def _render_key_findings(kf: dict, track: str) -> None:
    st.subheader("Key findings")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual", kf["actual"], border=True)
    m2.metric("Predicted (deploy threshold)", kf["predicted"], border=True)
    m3.metric("P(fail)", f"{100 * kf['p_fail']:.1f}%", border=True)
    m4.metric("Threshold", f"{kf['threshold']:.4f}", border=True)

    verdict = (
        f"Verdict: P(fail) {100 * kf['p_fail']:.1f}% vs threshold "
        f"{100 * kf['threshold']:.1f}% -> **{kf['predicted']}** "
        f"({kf['outcome_label']})."
    )
    if kf["outcome"] == "missed_fail":
        st.error(verdict, icon="🔴")
    elif kf["outcome"] == "false_alarm":
        st.warning(verdict, icon="🟠")
    else:
        st.success(verdict, icon="✅")

    bullets: list[str] = []
    if "top_driver" in kf:
        z = kf.get("top_spc_z")
        z_txt = f"SPC z = {z:+.1f}" if z is not None and np.isfinite(z) else "SPC z n/a"
        drift_txt = ", drifting" if kf.get("top_drifting") else ""
        bullets.append(
            f"**Top driver:** `{kf['top_driver']}` (pushes {kf['top_direction']}), "
            f"{z_txt}{drift_txt}."
        )
        bullets.append(
            f"**{kf.get('n_drifting', 0)} of {kf.get('n_contributors', 0)}** top "
            "contributors are drifting over time."
        )
        if track == "extrapolation" and "n_robust" in kf:
            bullets.append(
                f"**{kf['n_robust']} of {kf.get('n_contributors', 0)}** drivers have "
                "posterior HDIs clear of zero (confident)."
            )
    if bullets:
        st.markdown("\n".join(f"- {b}" for b in bullets))


# --- tab 1: wafer RCA --------------------------------------------------------
def _render_wafer_detail(
    model_id: str, track: str, wafer_id: object, result, local_df: pd.DataFrame
) -> None:
    if result.fail_probability_interval is not None:
        lo, hi = result.fail_probability_interval
        st.caption(
            f"Posterior P(fail) spread (pre-calibration): "
            f"{100 * lo:.1f}%–{100 * hi:.1f}% around an uncalibrated mean of "
            f"{100 * (result.fail_probability_uncalibrated or 0):.1f}% — a rough "
            "uncertainty band, not the calibrated point estimate above."
        )
    st.caption(result.method)

    render_blue_note(
        "Local contributors in **sensor space**. `spc_z` = robust z vs the in-control "
        "(passing) training distribution; `drift_flag` = the sensor's mean shifted "
        "past the era-drift threshold. Positive contribution pushes toward Fail. "
        "Non-sensor context features (calendar, missing-data flags) are excluded."
    )
    st.plotly_chart(
        fig_local_contributions(
            local_df, title=f"Top local contributors — wafer {wafer_id}"
        ),
        width="stretch",
        theme="streamlit",
        key="p6_local_bar",
    )
    display_cols = [
        c
        for c in ["feature", "contribution", "spc_z", "drift_shift", "drift_flag"]
        if c in local_df.columns
    ]
    st.dataframe(local_df[display_cols].round(3), width="stretch", hide_index=True)

    _render_spc_plots(local_df, track, wafer_id)
    _render_drift_linkage(local_df)

    if track == "extrapolation" and is_pls_model(model_id):
        _render_wafer_forest(model_id, track, wafer_id)


def _render_wafer_forest(model_id: str, track: str, wafer_id: object) -> None:
    st.markdown("**Per-sensor posterior — how confident is each driver?**")
    st.caption(
        "Posterior of each sensor's contribution to THIS wafer's logit "
        "(mean ± 95% HDI), back-projected from PLS component draws. A bar that "
        "clears the dashed zero line is a confident push; one straddling zero is "
        "uncertain."
    )
    try:
        forest = wafer_sensor_posterior(model_id, wafer_id, track)
    except Exception as exc:  # noqa: BLE001 - surface live-fit failures inline
        st.exception(exc)
        return
    if forest is None or forest.empty:
        st.info("Posterior forest unavailable for this wafer.")
        return
    st.plotly_chart(
        fig_posterior_forest(
            forest,
            title=f"Per-sensor contribution posterior — wafer {wafer_id}",
            xaxis_title="Contribution to logit",
        ),
        width="stretch",
        theme="streamlit",
        key="p6_wafer_forest",
    )


def _render_spc_plots(local_df: pd.DataFrame, track: str, wafer_id: object) -> None:
    split = load_holdout_split(track)
    train_df = split.train_df
    test_df = split.test_df
    pass_mask = train_df[TARGET_COL].astype(int) == 0
    row = test_df.loc[test_df[ID_COL] == wafer_id]
    if row.empty:
        return
    row = row.iloc[0]

    plottable = [
        r for _, r in local_df.iterrows() if str(r["feature"]) in train_df.columns
    ][:_SPC_PLOTS]
    if not plottable:
        return

    st.markdown("**SPC context — top contributors vs the in-control population**")
    st.caption(
        "Each violin is the passing training distribution for that sensor; the dashed "
        "line is this wafer. A wafer far outside the body is statistically unusual."
    )
    cols = st.columns(len(plottable))
    for i, (col, r) in enumerate(zip(cols, plottable)):
        sensor = str(r["feature"])
        with col:
            st.plotly_chart(
                fig_spc_distribution(
                    train_df.loc[pass_mask, sensor],
                    float(row.get(sensor, np.nan)),
                    sensor=sensor,
                    spc_z=float(r.get("spc_z", np.nan)),
                ),
                width="stretch",
                theme="streamlit",
                key=f"p6_spc_{i}",
            )


def _render_drift_linkage(local_df: pd.DataFrame) -> None:
    drifting = local_df.loc[local_df.get("drift_flag", False) == True]  # noqa: E712
    if drifting.empty:
        st.caption("Drift linkage: none of the top contributors are flagged as drifting.")
        return
    names = ", ".join(str(f) for f in drifting["feature"])
    st.caption(
        f"Drift linkage: {len(drifting)} of the top contributors are drifting over "
        f"time ({names}) — consistent with a temporal process excursion rather than "
        "a stable signature."
    )


# --- tab 2: global drivers ---------------------------------------------------
def _render_global(model_id: str, track: str) -> None:
    st.subheader("What the model learned (global drivers)")
    if track == "extrapolation" and is_pls_model(model_id):
        _render_global_forest(model_id, track)
    else:
        _render_global_bar(model_id, track)


def _render_global_forest(model_id: str, track: str) -> None:
    render_blue_note(
        "Sensor-space **posterior** for the champion: each sensor's signed "
        "coefficient (mean ± 95% HDI), back-projected from PLS component draws via "
        "the fitted loadings. Color-coded bars that clear the dashed zero line are "
        "**robust** (the model is confident in direction); dimmed bars straddle zero. "
        "Non-sensor context features (calendar, missing-data flags) are excluded."
    )
    try:
        forest = cached_pls_sensor_posterior(model_id, track)
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        return
    if forest is None or forest.empty:
        st.warning("Posterior unavailable for this model.")
        return
    n_robust = int(forest["robust"].sum())
    st.plotly_chart(
        fig_posterior_forest(
            forest,
            title=f"Sensor coefficient posterior — {model_info(model_id).display_name}",
        ),
        width="stretch",
        theme="streamlit",
        key="p6_global_forest",
    )
    st.caption(
        f"{n_robust} of {len(forest)} leading sensors have HDIs clear of zero "
        "(robustly nonzero vs posterior noise)."
    )


def _render_global_bar(model_id: str, track: str) -> None:
    render_blue_note(
        "Sensor-space importance: how strongly each sensor moves this model's "
        "fail-probability across the holdout (mean |SHAP| for the tree champion). "
        "Bars are magnitude."
    )
    try:
        top_df, _signed_df, caption = cached_global_sensor_importance(model_id, track)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        return
    if caption:
        st.caption(caption)
    if top_df is None or top_df.empty:
        st.warning("No global importance available for this model.")
        return
    st.plotly_chart(
        fig_top_features_bar(
            top_df,
            title=f"Top {len(top_df)} sensors — {model_info(model_id).display_name}",
        ),
        width="stretch",
        theme="streamlit",
        key="p6_global_bar",
    )


# --- tab 3: plain-English RCA (LLM) -----------------------------------------
def _render_narrative(model_id: str, track: str, wafer_id: object) -> None:
    st.subheader("Plain-English RCA summary")
    render_blue_note(
        "Constrained **Statistical Interpreter** (local Gemma): it restates the "
        "numeric facts only — no invented fab processes, equipment, or causes. This "
        "is an interpretation to guide investigation, not a fab diagnosis."
    )
    st.caption(
        f"For the **{_track_word(track)}** track the narrative model is "
        f"**{model_info(model_id).display_name}** (`{model_id}`)."
    )

    payload = _load_track_narratives(track)
    if payload is None:
        st.warning(
            "No frozen narratives for this track yet. Generate them with the local "
            "Gemma: `python -m secom.cli.build_narratives`"
        )
    elif wafer_id is not None:
        narrative = get_wafer_narrative(wafer_id, payload)
        if narrative is None:
            st.warning(
                f"No frozen narrative for wafer {wafer_id}. Regenerate with: "
                f"`python -m secom.cli.build_narratives --track {track} "
                f"--wafer-id {wafer_id}`"
            )
        else:
            st.markdown(narrative)
            st.caption(
                f"Pre-generated summary (local Gemma) · prompt "
                f"{payload.get('prompt_version', '')}"
            )

    with st.expander("Inspect the structured facts the interpreter receives"):
        st.caption(
            "Builds the facts JSON live for the narrative model (may fit the model). "
            "These are the only inputs the LLM is allowed to restate."
        )
        if wafer_id is not None and st.checkbox("Build facts now", key="p6_facts_go"):
            try:
                result = cached_wafer_explanation(model_id, wafer_id, track)
                if result is None:
                    st.info("Wafer not in this holdout split.")
                else:
                    st.json(build_wafer_facts(model_id, result, track))
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)


def main() -> None:
    _render_header()
    try:
        model_id, track, wafer_id = _render_controls()
    except FileNotFoundError as exc:
        st.error(f"{exc}\n\nRun tuning and `python -m secom.benchmark` first.")
        return

    if wafer_id is None:
        st.info("Select a wafer to see its root-cause analysis.")
        return

    try:
        result = cached_wafer_explanation(model_id, wafer_id, track)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        return
    if result is None:
        st.warning("Wafer not found in this holdout split.")
        return

    local_df = _annotate_robust(result.local_df, model_id, track)
    kf = key_findings(result, local_df, track)
    _render_key_findings(kf, track)

    tab_wafer, tab_global, tab_llm = st.tabs(
        ["Wafer RCA", "Global drivers", "Plain-English RCA"]
    )
    with tab_wafer:
        _render_wafer_detail(model_id, track, wafer_id, result, local_df)
    with tab_global:
        _render_global(model_id, track)
    with tab_llm:
        _render_narrative(model_id, track, wafer_id)


main()
