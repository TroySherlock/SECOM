"""Unit tests for the page-5.4 gate-contrast df builders (pure functions)."""
from __future__ import annotations

import numpy as np

from secom.dashboard.data import (
    bgm_ooc_map,
    bgm_ooc_wafers,
    contribution_compare_df,
    gate_contrast,
    gate_disagreement_summary,
    sensor_noise_df,
)


def _bgm_payload() -> dict:
    # 4 holdout wafers; only passing (y_true==0) AND ooc ones should survive.
    return {
        "gate_reports": {
            "temporal": {
                "bayes": {
                    "diagnostics": {
                        "holdout": {
                            "observation_id": [10, 11, 12, 13],
                            "ts": ["2008-01-01", "2008-02-01", "2008-03-01", "2008-04-01"],
                            "density": [-9.0, -2.0, -7.0, -1.0],
                            "density_ooc": [True, False, True, False],
                            "q_ooc": [False, False, True, False],
                            "y_true": [0, 0, 1, 0],
                            "ooc": [True, False, True, False],
                        }
                    }
                }
            }
        }
    }


def test_bgm_ooc_wafers_keeps_passing_ooc_sorted_by_density() -> None:
    df = bgm_ooc_wafers(_bgm_payload(), track="extrapolation")
    # Wafer 12 is OOC but a fail (dropped); 11/13 are not OOC. Only wafer 10 remains.
    assert list(df["observation_id"]) == [10]
    assert "y_true" not in df.columns and "ooc" not in df.columns


def test_bgm_ooc_wafers_empty_when_absent() -> None:
    assert bgm_ooc_wafers({}, "extrapolation").empty
    assert bgm_ooc_wafers({"gate_reports": {"temporal": {}}}, "extrapolation").empty


def test_bgm_ooc_map_covers_all_holdout_wafers() -> None:
    # Unlike bgm_ooc_wafers, the map includes failing AND in-control wafers.
    m = bgm_ooc_map(_bgm_payload(), track="extrapolation")
    assert m == {10: True, 11: False, 12: True, 13: False}


def test_bgm_ooc_map_empty_when_absent() -> None:
    assert bgm_ooc_map({}, "extrapolation") == {}
    assert bgm_ooc_map({"gate_reports": {"temporal": {}}}, "extrapolation") == {}
    # interpolation track has no frozen gate diagnostics in this fixture.
    assert bgm_ooc_map(_bgm_payload(), track="interpolation") == {}

# Synthetic contrast block: 4 passing wafers, 3 sensors. t2_ucl=2.0, density_lcl=-5.0.
# Wafer 0: high T2 (3.0) but in-control density (-1.0) -> PCA-only (the money quadrant).
# Wafer 1: low T2, low density -> BGM-only. Wafer 2: high T2, low density -> both.
# Wafer 3: low T2, high density -> neither.
_CONTRAST = {
    "reference": {
        "pca_t2": [3.0, 1.0, 3.0, 1.0],
        "bgm_density": [-1.0, -9.0, -9.0, -1.0],
        "pca_t2_ucl": 2.0,
        "bgm_density_lcl": -5.0,
    },
    "bgm_weights": [0.6, 0.3, 0.05, 0.05],
    "psi": [0.01, 1.0, 100.0],
    "feature_names": ["c_quiet", "c_mid", "c_noisy"],
    "example_wafer": {
        "label": "reference #2",
        "pca_resid": [1.0, 1.0, 5.0],
        "sbfa_resid": [1.0, 1.0, 5.0],
    },
}


def test_gate_contrast_maps_track_to_protocol() -> None:
    # The dashboard track "extrapolation" must resolve to the persisted "temporal" key.
    payload = {"gate_reports": {"temporal": {"contrast": {"x": 1}}}}
    assert gate_contrast(payload, "extrapolation") == {"x": 1}
    assert gate_contrast(payload, "interpolation") == {}
    assert gate_contrast({}, "extrapolation") == {}


def test_disagreement_summary_quadrants() -> None:
    s = gate_disagreement_summary(_CONTRAST)
    assert s["n_total"] == 4
    assert s["pca_only"] == 1
    assert s["bgm_only"] == 1
    assert s["both"] == 1
    assert s["neither"] == 1


def test_disagreement_summary_empty() -> None:
    assert gate_disagreement_summary({}) == {}
    assert gate_disagreement_summary({"reference": {}}) == {}


def test_sensor_noise_df_sorted_descending() -> None:
    df = sensor_noise_df(_CONTRAST)
    assert list(df["sensor"]) == ["c_noisy", "c_mid", "c_quiet"]
    assert df["psi"].is_monotonic_decreasing


def test_contribution_compare_reranks_noisy_sensor() -> None:
    df = contribution_compare_df(_CONTRAST).set_index("sensor")
    # Equal-weight: c_noisy (5^2=25) dominates over c_quiet (1).
    assert df.loc["c_noisy", "pca_contribution"] == 1.0
    # Noise-weighted: c_quiet (1/0.01=100) beats c_noisy (25/100=0.25) -> rank flip.
    assert df.loc["c_quiet", "sbfa_contribution"] == 1.0
    assert df.loc["c_quiet", "sbfa_contribution"] > df.loc["c_noisy", "sbfa_contribution"]


def test_contribution_compare_empty() -> None:
    assert contribution_compare_df({}).empty
    assert sensor_noise_df({}).empty


def test_builders_handle_length_mismatch() -> None:
    bad = {"psi": [1.0, 2.0], "feature_names": ["only_one"]}
    assert sensor_noise_df(bad).empty


def test_disagreement_uses_numpy_arrays() -> None:
    # Arrays (not lists) should also work, mirroring frozen-JSON-loaded payloads.
    contrast = {
        "reference": {
            "pca_t2": np.array([3.0, 1.0]),
            "bgm_density": np.array([-1.0, -1.0]),
            "pca_t2_ucl": 2.0,
            "bgm_density_lcl": -5.0,
        }
    }
    s = gate_disagreement_summary(contrast)
    assert s["pca_only"] == 1
    assert s["neither"] == 1
