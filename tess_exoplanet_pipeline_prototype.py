"""
TESS Exoplanet Transit Candidate Pipeline — Prototype
Science-project prototype / proof of concept.

Input CSV format:
    time,flux

Optional labeled training CSV:
    depth,duration,period,shape_std,noise_std,label

Outputs:
    candidate_results.csv
    light_curve.png
    pipeline_results.png
    candidate_table.tex

This prototype:
1. cleans and normalizes a TESS-like light curve,
2. detrends it,
3. searches for periodic transit-like signals with BLS,
4. extracts simple candidate features,
5. optionally trains a Random Forest classifier,
6. ranks candidates,
7. creates a two-panel diagnostic figure,
8. exports the top 5 candidates as a LaTeX table.

It is NOT a validated exoplanet-discovery system.
BLS candidates require astronomical validation and cross-checking
against existing catalogs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from astropy.timeseries import BoxLeastSquares
except ImportError:
    raise SystemExit(
        "Install dependencies first: pip install -r requirements.txt"
    )

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit(
        "Install dependencies first: pip install -r requirements.txt"
    )


FEATURES = [
    "depth",
    "duration",
    "period",
    "shape_std",
    "noise_std"
]


# ============================================================
# DATA LOADING
# ============================================================

def load_light_curve(path):
    df = pd.read_csv(path)

    required = {"time", "flux"}

    if not required.issubset(df.columns):
        raise ValueError(
            "Light-curve CSV must contain columns: time, flux"
        )

    df = df[["time", "flux"]].dropna().sort_values("time")

    df = df[
        np.isfinite(df["time"]) &
        np.isfinite(df["flux"])
    ]

    return df


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(df):
    t = df["time"].to_numpy(float)
    f = df["flux"].to_numpy(float)

    # Robust clipping of extreme outliers.
    med = np.median(f)
    mad = np.median(np.abs(f - med))

    if mad > 0:
        keep = np.abs(f - med) < 8 * 1.4826 * mad
        t, f = t[keep], f[keep]

    # Normalize.
    f = f / np.median(f)

    # Simple polynomial detrending.
    coeff = np.polyfit(
        t - np.mean(t),
        f,
        deg=2
    )

    trend = np.polyval(
        coeff,
        t - np.mean(t)
    )

    trend[trend == 0] = 1.0

    fd = f / trend

    return t, fd


# ============================================================
# SYNTHETIC DEMO DATA
# ============================================================

def generate_demo_light_curve(seed=7):
    """Create a synthetic TESS-like curve."""

    rng = np.random.default_rng(seed)

    t = np.linspace(0, 27, 4500)

    baseline = (
        1
        + 0.0015 * np.sin(2 * np.pi * t / 11)
    )

    flux = (
        baseline
        + rng.normal(0, 0.0009, len(t))
    )

    # Synthetic transit.
    period = 3.4
    depth = 0.012
    duration = 0.12

    phase = (
        (t - 0.7 + period / 2) % period
    ) - period / 2

    flux[
        np.abs(phase) < duration / 2
    ] -= depth

    return t, flux


# ============================================================
# BLS SEARCH
# ============================================================

def run_bls(
    t,
    flux,
    min_period=0.8,
    max_period=10.0
):

    bls = BoxLeastSquares(t, flux)

    periods = np.linspace(
        min_period,
        max_period,
        900
    )

    durations = np.linspace(
        0.04,
        0.20,
        12
    )

    result = bls.power(
        periods,
        durations
    )

    order = np.argsort(
        result.power
    )[::-1]

    candidates = []

    # Keep separated peaks rather than
    # returning many nearly identical periods.
    for idx in order:

        p = float(
            result.period[idx]
        )

        if all(
            abs(p - c["period"]) / p > 0.03
            for c in candidates
        ):

            candidates.append({
                "period": p,
                "duration": float(
                    result.duration[idx]
                ),
                "power": float(
                    result.power[idx]
                ),
                "t0": float(
                    result.transit_time[idx]
                ),
            })

        if len(candidates) >= 10:
            break

    return pd.DataFrame(candidates)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    t,
    flux,
    candidates
):

    rows = []

    for _, c in candidates.iterrows():

        period = c["period"]
        duration = c["duration"]
        t0 = c["t0"]

        phase = (
            (t - t0 + period / 2) % period
        ) - period / 2

        in_transit = (
            np.abs(phase)
            < duration / 2
        )

        out_flux = flux[~in_transit]
        in_flux = flux[in_transit]

        if (
            len(in_flux) < 3
            or len(out_flux) < 10
        ):
            continue

        depth = max(
            0.0,
            float(
                np.median(out_flux)
                - np.median(in_flux)
            )
        )

        shape_std = float(
            np.std(in_flux)
        )

        noise_std = float(
            np.std(
                out_flux
                - np.median(out_flux)
            )
        )

        rows.append({
            "period": period,
            "duration": duration,
            "depth": depth,
            "shape_std": shape_std,
            "noise_std": noise_std,
            "power": c["power"],
            "t0": t0,
        })

    return pd.DataFrame(rows)


# ============================================================
# OPTIONAL MACHINE LEARNING
# ============================================================

def train_optional_model(training_csv):

    if training_csv is None:
        return None

    train = pd.read_csv(
        training_csv
    )

    required = set(
        FEATURES + ["label"]
    )

    if not required.issubset(
        train.columns
    ):
        raise ValueError(
            "Training CSV must contain: "
            + ", ".join(
                FEATURES + ["label"]
            )
        )

    X = train[FEATURES]
    y = train["label"]

    if y.nunique() < 2:
        raise ValueError(
            "Training labels must contain "
            "at least two classes."
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    print(
        "\nModel evaluation on held-out labeled data:"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, pred, zero_division=0):.3f}"
    )

    print(
        f"Recall:    "
        f"{recall_score(y_test, pred, zero_division=0):.3f}"
    )

    print(
        f"F1-score:  "
        f"{f1_score(y_test, pred, zero_division=0):.3f}"
    )

    return model


# ============================================================
# CANDIDATE RANKING
# ============================================================

def rank_candidates(
    features,
    model=None
):

    result = features.copy()

    if model is not None:

        result["ml_probability"] = (
            model.predict_proba(
                result[FEATURES]
            )[:, 1]
        )

        result["priority_score"] = (
            0.7
            * result["ml_probability"]
            +
            0.3
            * (
                result["power"]
                / result["power"].max()
            )
        )

        result["classification"] = (
            model.predict(
                result[FEATURES]
            )
        )

    else:

        # Demo ranking when no labeled
        # training set is supplied.
        result["ml_probability"] = np.nan

        result["priority_score"] = (
            result["power"]
            / result["power"].max()
        )

        result["classification"] = (
            "unclassified"
        )

    return result.sort_values(
        "priority_score",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# ORIGINAL LIGHT CURVE PLOT
# ============================================================

def save_plot(
    t,
    flux,
    output
):

    plt.figure(figsize=(10, 4))

    plt.plot(
        t,
        flux,
        linewidth=0.7
    )

    plt.xlabel(
        "Time [days]"
    )

    plt.ylabel(
        "Normalized flux"
    )

    plt.title(
        "Preprocessed TESS Light Curve"
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=180
    )

    plt.close()


# ============================================================
# TRANSIT WINDOW CALCULATION
# ============================================================

def get_transit_centers(
    t,
    period,
    t0
):

    t_min = np.min(t)
    t_max = np.max(t)

    k_min = int(
        np.floor(
            (t_min - t0) / period
        )
    ) - 1

    k_max = int(
        np.ceil(
            (t_max - t0) / period
        )
    ) + 1

    centers = (
        t0
        + np.arange(
            k_min,
            k_max + 1
        ) * period
    )

    centers = centers[
        (centers >= t_min)
        & (centers <= t_max)
    ]

    return centers


# ============================================================
# TWO-PANEL PIPELINE FIGURE
# ============================================================

def save_pipeline_results(
    t,
    flux,
    best_candidate,
    output
):

    period = float(
        best_candidate["period"]
    )

    duration = float(
        best_candidate["duration"]
    )

    depth = float(
        best_candidate["depth"]
    )

    t0 = float(
        best_candidate["t0"]
    )

    # --------------------------------------------------------
    # Phase calculation
    # --------------------------------------------------------

    phase = (
        (t - t0 + period / 2) % period
    ) - period / 2

    in_transit = (
        np.abs(phase)
        < duration / 2
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8)
    )

    # ========================================================
    # PANEL A — FULL LIGHT CURVE
    # ========================================================

    ax = axes[0]

    ax.plot(
        t,
        flux,
        linewidth=0.55,
        alpha=0.85
    )

    transit_centers = (
        get_transit_centers(
            t,
            period,
            t0
        )
    )

    # Shade predicted transit windows.
    for center in transit_centers:

        ax.axvspan(
            center - duration / 2,
            center + duration / 2,
            alpha=0.20
        )

    ax.set_title(
        "(a) Full Preprocessed Light Curve"
    )

    ax.set_xlabel(
        "Time [days]"
    )

    ax.set_ylabel(
        "Normalized flux"
    )

    # Automatic candidate annotation.
    ax.text(
        0.01,
        0.03,
        (
            f"Best candidate\n"
            f"P = {period:.4f} d\n"
            f"Duration = {duration:.4f} d "
            f"({duration * 24:.2f} h)\n"
            f"Depth = {depth * 100:.3f}%"
        ),
        transform=ax.transAxes,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            alpha=0.85
        )
    )

    # ========================================================
    # PANEL B — PHASE FOLDED
    # ========================================================

    ax = axes[1]

    # Out-of-transit points.
    ax.scatter(
        phase[~in_transit],
        flux[~in_transit],
        s=3,
        alpha=0.35,
        rasterized=True
    )

    # In-transit points.
    ax.scatter(
        phase[in_transit],
        flux[in_transit],
        s=6,
        alpha=0.65,
        rasterized=True
    )

    # Expected transit window.
    ax.axvspan(
        -duration / 2,
        duration / 2,
        alpha=0.20
    )

    ax.axvline(
        0,
        linewidth=0.8,
        linestyle="--"
    )

    ax.set_title(
        "(b) Phase-Folded Best Candidate"
    )

    ax.set_xlabel(
        "Phase"
    )

    ax.set_ylabel(
        "Normalized flux"
    )

    ax.set_xlim(
        -0.5,
        0.5
    )

    ax.text(
        0.01,
        0.03,
        (
            f"P = {period:.4f} d\n"
            f"Duration = {duration:.4f} d "
            f"({duration * 24:.2f} h)\n"
            f"Depth = {depth * 100:.3f}%"
        ),
        transform=ax.transAxes,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            alpha=0.85
        )
    )

    fig.suptitle(
        "TESS Exoplanet Transit Candidate Analysis",
        fontsize=15
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# LATEX CANDIDATE TABLE
# ============================================================

def save_candidate_table(
    ranked,
    output,
    top_n=5,
    include_ml=False
):

    top = ranked.head(top_n).copy()

    # Add rank.
    top.insert(
        0,
        "Rank",
        range(1, len(top) + 1)
    )

    # Convert to human-readable units.
    top["duration_hours"] = (
        top["duration"] * 24
    )

    top["depth_percent"] = (
        top["depth"] * 100
    )

    columns = [
        "Rank",
        "period",
        "duration_hours",
        "depth_percent",
        "power",
        "priority_score",
    ]

    column_names = [
        "Rank",
        "Period (hari)",
        "Durasi (jam)",
        "Kedalaman (\\%)",
        "BLS Power",
        "Priority Score",
    ]

    # Only include ML information when an actual
    # trained model was used.
    if include_ml:
        columns += [
            "ml_probability",
            "classification",
        ]

        column_names += [
            "ML Probability",
            "Classification",
        ]

    table = top[columns].copy()

    table.columns = column_names

    latex = table.to_latex(
        index=False,
        escape=False,
        float_format="%.5f"
    )

    header = (
        "% Automatically generated by the "
        "TESS Exoplanet Transit Candidate Pipeline\n"
        f"% Top {top_n} ranked candidates\n\n"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            header + latex
        )

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        help="CSV with time,flux columns"
    )

    parser.add_argument(
        "--training",
        type=str,
        help="Optional labeled feature CSV"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="candidate_results.csv"
    )

    parser.add_argument(
        "--plot",
        type=str,
        default="light_curve.png"
    )

    parser.add_argument(
        "--pipeline-plot",
        type=str,
        default="pipeline_results.png"
    )

    parser.add_argument(
        "--latex-table",
        type=str,
        default="candidate_table.tex"
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD DATA
    # ========================================================

    if args.input:

        print(
            "\nData source: USER-PROVIDED LIGHT CURVE"
        )

        df = load_light_curve(
            args.input
        )

        t, flux = preprocess(
            df
        )

        print(
            f"Loaded {len(t):,} "
            f"valid observations."
        )

    else:

        print(
            "\nData source: SYNTHETIC TESS-LIKE DEMONSTRATION"
        )

        print(
            "This output must NOT be described "
            "as real TESS observations.\n"
        )

        t, raw_flux = generate_demo_light_curve()

        t, flux = preprocess(
            pd.DataFrame({
                "time": t,
                "flux": raw_flux
            })
        )

    # ========================================================
    # BLS
    # ========================================================

    print(
        "Running BLS candidate search..."
    )

    candidates = run_bls(
        t,
        flux
    )

    features = extract_features(
        t,
        flux,
        candidates
    )

    if features.empty:
        raise RuntimeError(
            "No usable candidates were produced."
        )

    # ========================================================
    # OPTIONAL MODEL
    # ========================================================

    model = train_optional_model(
        args.training
    )

    # ========================================================
    # RANK
    # ========================================================

    ranked = rank_candidates(
        features,
        model
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    ranked.to_csv(
        args.output,
        index=False
    )

    # ========================================================
    # ORIGINAL LIGHT CURVE
    # ========================================================

    save_plot(
        t,
        flux,
        args.plot
    )

    # ========================================================
    # BEST CANDIDATE
    # ========================================================

    best_candidate = (
        ranked.iloc[0]
    )

    # ========================================================
    # TWO-PANEL PIPELINE FIGURE
    # ========================================================

    save_pipeline_results(
        t,
        flux,
        best_candidate,
        args.pipeline_plot
    )

    # ========================================================
    # LATEX TABLE
    # ========================================================

    save_candidate_table(
        ranked,
        args.latex_table,
        top_n=5,
        include_ml=(model is not None)
    )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(
        "\nPrioritized candidates:"
    )

    print(
        ranked[
            [
                "period",
                "duration",
                "depth",
                "power",
                "priority_score",
                "classification"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        "\nBest candidate:"
    )

    print(
        f"  Period:   "
        f"{best_candidate['period']:.5f} days"
    )

    print(
        f"  Duration: "
        f"{best_candidate['duration']:.5f} days "
        f"({best_candidate['duration'] * 24:.2f} hours)"
    )

    print(
        f"  Depth:    "
        f"{best_candidate['depth'] * 100:.4f}%"
    )

    print(
        f"\nSaved: {args.output}"
    )

    print(
        f"Saved: {args.plot}"
    )

    print(
        f"Saved: {args.pipeline_plot}"
    )

    print(
        f"Saved: {args.latex_table}"
    )


if __name__ == "__main__":
    main()