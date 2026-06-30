"""Root-cause-analysis (RCA) explainability for the SECOM defect-detection grid.

One champion per track (pls_bayes for the temporal/extrapolation track, hsic_rf
for the random/interpolation track). The page leads with deterministic KEY
FINDINGS followed by a constrained plain-English (local Gemma) summary, then two
focused views: a per-wafer RCA and the champion's global drivers. PLS attributions
are back-projected through the fitted loadings into one sensor vocabulary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from secom.costs import THRESHOLD_PROFILES
from secom.dashboard import (
    render_alert,
    render_blue_note,
    render_caveat,
    render_verdict,
)
from secom.dashboard.charts import (
    fig_local_contributions,
    fig_posterior_forest,
    fig_spc_distribution,
    fig_top_features_bar,
    fig_wafer_drift_spikes,
)
from secom.dashboard.data import (
    bgm_ooc_map,
    cached_benchmark_results,
    model_info,
    wafer_gate_facts,
)
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
    deploy_profile_id,
    is_pls_model,
    key_findings,
    load_holdout_split,
    wafer_drift_spikes,
    wafer_sensor_posterior,
)
from secom.dashboard.narrator import (
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
_GATE_FILTERS = {
    "All": None,
    "🚧 Gate-OOC only": "ooc",
    "In-control only": "in_control",
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
    render_caveat(
        "**Honesty caveat — associational, not causal.** Every attribution is a "
        "model-derived hypothesis to guide investigation, not a proven root cause. "
        "SECOM ships no sensor-to-tool/chamber map, so there is no real-equipment "
        "grouping here."
    )


def _track_word(track: str) -> str:
    return "temporal / extrapolation" if track == "extrapolation" else "random / interpolation"


# --- controls ----------------------------------------------------------------
def _render_controls(payload: dict) -> tuple[str, str, object]:
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
    profile = THRESHOLD_PROFILES[deploy_profile_id(track)]
    with c1:
        st.markdown(f"**Champion for this track:** `{champion}` — {info.display_name}")
        st.caption(
            f"**Deploy threshold: {profile.display_name}.** The temporal/extrapolation "
            "champion ships at the cost-optimal *economic* point (escapes priced far "
            "above overkill, so it leans to recall); the random/interpolation champion "
            "ships at the symmetric *BER-balanced* point."
        )

    outcomes = cached_holdout_outcomes(champion, track)
    outcome_map = dict(zip(outcomes["observation_id"], outcomes["outcome"]))
    ooc_map = bgm_ooc_map(payload, track)

    with c2:
        f1, f2 = st.columns([2, 3])
        with f1:
            filt_label = st.radio(
                "Show",
                options=list(_OUTCOME_FILTERS),
                key="p6_outcome_filter",
                help="Filter the wafer picker by model verdict. Errors are listed first.",
            )
            gate_filt = None
            if ooc_map:
                gate_label = st.radio(
                    "Process gate",
                    options=list(_GATE_FILTERS),
                    key="p6_gate_filter",
                    help="Filter by the BGM gate's verdict — independent of the model's "
                    "verdict above. Gate-OOC wafers sit on a drifted process.",
                )
                gate_filt = _GATE_FILTERS[gate_label]
            chrono = st.toggle(
                "Chronological (by wafer ID)",
                key="p6_chrono",
                help="Off = errors-first triage order. On = ascending wafer ID "
                "(time order across the holdout).",
            )
        filt = _OUTCOME_FILTERS[filt_label]
        sel = outcomes if filt is None else outcomes.loc[outcomes["outcome"] == filt]
        if gate_filt == "ooc":
            sel = sel.loc[sel["observation_id"].map(lambda w: ooc_map.get(int(w), False))]
        elif gate_filt == "in_control":
            sel = sel.loc[sel["observation_id"].map(lambda w: not ooc_map.get(int(w), False))]
        if chrono:
            sel = sel.sort_values("observation_id", ascending=True)
        wafer_options = sel["observation_id"].tolist()

        def _fmt(wid: object) -> str:
            oc = outcome_map.get(wid, "")
            mark = " · 🚧 gate-OOC" if ooc_map.get(int(wid), False) else ""
            return f"{OUTCOME_EMOJI.get(oc, '')} {OUTCOME_LABELS.get(oc, '')} · {wid}{mark}"

        with f2:
            if not wafer_options:
                st.info("No wafers match this filter.")
                wafer_id = None
            else:
                wafer_id = st.selectbox(
                    "Wafer (holdout)",
                    wafer_options,
                    format_func=_fmt,
                    key=f"p6_wafer_{track}_{filt}_{gate_filt}_{chrono}",
                )
        legend = (
            "Picker legend: 🔴 missed fail · 🟠 false alarm · 🟢 caught fail · "
            "⚪ correct pass"
        )
        legend += (
            " · 🚧 gate-OOC (BGM flagged the process). Errors are sorted to the top."
            if ooc_map
            else ". Errors are sorted to the top."
        )
        st.caption(legend)
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


_GATE_NAMES = {"bayes": "BGM density gate", "pca": "PCA Hotelling T2 gate"}


def _ooc_gate_labels(gate_facts: dict) -> list[str]:
    """Plain-English '<gate> (<tripped stats>)' for gates that flagged the wafer."""
    labels: list[str] = []
    for gate, facts in (gate_facts or {}).items():
        if not facts.get("out_of_control"):
            continue
        name = _GATE_NAMES.get(gate, gate)
        trips = ", ".join(facts.get("tripped") or []) or "out of control"
        labels.append(f"{name} ({trips})")
    return labels


def _render_key_findings(kf: dict, track: str, gate_facts: dict) -> None:
    st.subheader("🔑 Key findings")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual", kf["actual"], border=True)
    m2.metric("Predicted (deploy threshold)", kf["predicted"], border=True)
    m3.metric("P(fail)", f"{100 * kf['p_fail']:.1f}%", border=True)
    m4.metric("Threshold", f"{kf['threshold']:.4f}", border=True)
    st.caption(
        f"Verdict uses the **{THRESHOLD_PROFILES[deploy_profile_id(track)].display_name}** "
        "deploy threshold for this track."
    )

    verdict = (
        f"Verdict: P(fail) {100 * kf['p_fail']:.1f}% vs threshold "
        f"{100 * kf['threshold']:.1f}% -> **{kf['predicted']}** "
        f"({kf['outcome_label']})."
    )
    if kf["outcome"] == "missed_fail":
        render_alert(f"🔴 {verdict}")
    elif kf["outcome"] == "false_alarm":
        render_caveat(f"🟠 {verdict}")
    else:
        render_verdict(f"✅ {verdict}")

    ooc_labels = _ooc_gate_labels(gate_facts)
    if ooc_labels:
        render_caveat(
            "🚧 **Process gate OOC:** "
            + "; ".join(ooc_labels)
            + ". This wafer sits on a drifted process, so the classifier is "
            "extrapolating — the standalone gate would abstain here and route it "
            "to manual review."
        )

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
    model_id: str,
    track: str,
    wafer_id: object,
    result,
    local_df: pd.DataFrame,
    gate_facts: dict,
) -> None:
    if result.fail_probability_interval is not None:
        lo, hi = result.fail_probability_interval
        st.caption(
            f"95% credible interval for the calibrated P(fail): "
            f"{100 * lo:.1f}%–{100 * hi:.1f}% — same scale as the point estimate "
            "above (a wider band means the model is less certain)."
        )
    st.caption(result.method)

    ooc_labels = _ooc_gate_labels(gate_facts)
    if ooc_labels:
        st.caption(
            "Process gate: "
            + " and ".join(ooc_labels)
            + " — the attribution below is for a wafer the gate considers off-process."
        )

    render_blue_note(
        "Local contributors in **sensor space**. `spc_z` = robust z vs the in-control "
        "(passing) training distribution; `drift_flag` = this top driver is *also* a "
        "sensor whose mean shifted past the era-drift threshold train->holdout. It is "
        "often False: a wafer's strongest prediction drivers are usually not the "
        "globally drifted sensors — for the wafer's actual process drift see the "
        "panel below. Positive contribution pushes toward Fail. Non-sensor context "
        "features (calendar, missing-data flags) are excluded."
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
    _render_wafer_drift_panel(track, wafer_id)

    if track == "extrapolation" and is_pls_model(model_id):
        _render_wafer_forest(model_id, track, wafer_id)


def _render_wafer_drift_panel(track: str, wafer_id: object) -> None:
    st.markdown("**Process drift on this wafer — globally drifting sensors**")
    spikes = wafer_drift_spikes(int(wafer_id), track)
    if spikes.empty:
        st.caption(
            "None of the globally drifting sensors have a usable reading on this "
            "wafer, so there is no direct drift to show here."
        )
        return
    st.caption(
        "This wafer's robust SPC z across the sensors that drifted train->holdout — "
        "independent of whether they drove the prediction above. Bars past ±2σ (red) "
        "are the in-control excursion the density gate keys on; the gate fires on the "
        "*joint* density across these correlated, drifted sensors, so a wafer can be "
        "off-process even when no single sensor screams. Hover for each sensor's "
        "global train->holdout shift."
    )
    st.plotly_chart(
        fig_wafer_drift_spikes(
            spikes["sensor"].astype(str).tolist(),
            spikes["robust_z"].to_numpy(),
            spikes["drift_shift"].to_numpy(),
        ),
        width="stretch",
        theme="streamlit",
        key="p6_wafer_drift_spikes",
    )


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
        st.caption(
            "Drift linkage: none of the top prediction-drivers are among the globally "
            "drifting sensors — the wafer's process drift (if any) is shown directly "
            "in the panel below."
        )
        return
    names = ", ".join(str(f) for f in drifting["feature"])
    st.caption(
        f"Drift linkage: {len(drifting)} of the top contributors are drifting over "
        f"time ({names}) — consistent with a temporal process excursion rather than "
        "a stable signature."
    )


# --- tab 2: global drivers ---------------------------------------------------
def _render_global(model_id: str, track: str) -> None:
    st.subheader("🧠 What the model learned (global drivers)")
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


# --- inline plain-English summary (LLM), shown under Key Findings -----------
def _render_narrative(model_id: str, track: str, wafer_id: object) -> None:


        with st.container(key="card_built"):
            st.markdown("#### LLM-generated summary")
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
                    
        render_blue_note(
            "Constrained **Statistical Interpreter** (local Gemma): it restates the "
            "numeric facts only — no invented fab processes, equipment, or causes. This "
            "is an interpretation to guide investigation, not a fab diagnosis."
        )


def main() -> None:
    _render_header()
    payload = cached_benchmark_results()
    try:
        model_id, track, wafer_id = _render_controls(payload)
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

    gate_facts = wafer_gate_facts(payload, track, wafer_id)
    local_df = _annotate_robust(result.local_df, model_id, track)
    kf = key_findings(result, local_df, track)
    _render_key_findings(kf, track, gate_facts)
    _render_narrative(model_id, track, wafer_id)

    tab_wafer, tab_global = st.tabs(["Wafer RCA", "Global drivers"])
    with tab_wafer:
        _render_wafer_detail(model_id, track, wafer_id, result, local_df, gate_facts)
    with tab_global:
        _render_global(model_id, track)


main()
