"""Regression tests for the bugs documented in README 'Known issues / errata'.

Run with:  python -m pytest -q
Tests that need astropy are skipped automatically if it is not installed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tess_exoplanet_pipeline_prototype as pipe  # noqa: E402


def make_lc(period=0.94, depth=0.01, dur=0.09, noise=6e-4, days=27.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, 2 / 1440)
    t = t[~((t > 13.0) & (t < 14.0))]  # mid-sector gap
    f = 1 + rng.normal(0, noise, len(t)) + 0.0005 * np.sin(2 * np.pi * t / 7)
    f[np.abs(pipe.transit_offset(t, period, 0.3)) < dur / 2] -= depth
    return pd.DataFrame({"time": t, "flux": f * 47000.0}), period, depth, dur


def test_preprocess_does_not_clip_transits():
    """v0.1 used a two-sided 8-sigma-MAD clip that removed deep in-transit
    points (1,501 of 18,299 on WASP-18)."""
    df, P, depth, dur = make_lc()
    t, f, info = pipe.preprocess(df)
    inn = np.abs(pipe.transit_offset(t, P, 0.3)) < dur / 2
    n_raw = (np.abs(pipe.transit_offset(df["time"].to_numpy(), P, 0.3)) < dur / 2).sum()
    assert inn.sum() >= 0.98 * n_raw
    assert np.median(f[inn]) < 1 - 0.8 * depth


def test_upper_clip_removes_flares_only():
    df, *_ = make_lc()
    df.loc[100, "flux"] *= 1.05
    t, f, info = pipe.preprocess(df)
    assert info["n_removed_upper_clip"] >= 1
    assert f.min() < 0.995  # transits intact


def test_masked_detrending_preserves_depth():
    df, P, depth, dur = make_lc()
    t, f, _ = pipe.preprocess(df, transit_masks=[(P, 0.3, dur)])
    inn = np.abs(pipe.transit_offset(t, P, 0.3)) < dur / 4
    assert np.median(f[inn]) < 1 - 0.9 * depth


@pytest.mark.parametrize("ratio", ["5/2", "3/2", "2/1", "7/2"])
def test_alias_detection(ratio):
    a, b = map(int, ratio.split("/"))
    ref = 0.94145
    j, r = pipe.classify_alias(ref * a / b * 1.001, [ref])
    assert j == 0 and r == ratio


def test_alias_detection_leaves_unrelated_period_alone():
    assert pipe.classify_alias(1.2345, [0.94145])[0] is None


def test_period_grid_resolves_transit():
    pytest.importorskip("astropy")
    df, *_ = make_lc()
    t, f, _ = pipe.preprocess(df)
    durs = pipe.build_durations(0.02, 0.30, 8)
    _, _, grid = pipe.search_candidates(t, f, durs, n_candidates=3)
    assert grid["grid_resolves_shortest_transit"]
    # the v0.1 grid (900 evenly spaced periods) smeared by ~0.3 d
    v01_smear = (10 - 0.8) / 899 * (t.max() - t.min()) / 0.94
    assert v01_smear > 5 * grid["max_phase_smear_days"]


def test_recovers_planet_that_is_not_a_v01_grid_node():
    pytest.importorskip("astropy")
    P = 0.94145  # not a node of the v0.1 grid
    df, _, depth, dur = make_lc(period=P)
    t, f, _ = pipe.preprocess(df)
    durs = pipe.build_durations(0.02, 0.30, 8)
    c, _, _ = pipe.search_candidates(t, f, durs, n_candidates=6)
    best = c[~c["is_alias"]].iloc[0]
    assert abs(best["period"] - P) / P < 5e-4
    assert abs(best["depth"] - depth) / depth < 0.25
    assert abs(best["duration"] - dur) / dur < 0.4
    assert c["is_alias"].sum() >= 1  # harmonics are flagged, not listed as new


def test_latex_table_without_jinja(tmp_path):
    ranked = pd.DataFrame({
        "period": [1.0], "duration": [0.1], "depth": [0.01], "power": [10.0],
        "is_alias": [False], "at_duration_edge": [False],
        "ml_probability": [np.nan]})
    out = tmp_path / "t.tex"
    pipe.save_candidate_table(ranked, out)
    assert "\\begin{tabular}" in out.read_text()
