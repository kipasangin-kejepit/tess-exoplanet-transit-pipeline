"""
TESS Exoplanet Transit Candidate Pipeline -- Prototype (v0.2.0)

Research prototype. NOT a validated exoplanet-discovery system.

Input CSV:            time,flux            (optional column: flux_err)
Optional training CSV: depth,duration,depth_snr,shape_std,noise_std,odd_even_sigma,label
                       (optional column: group -> group-aware train/test split)

What changed relative to v0.1 (see README "Known issues / errata"):
  * BLS period grid now comes from astropy's autoperiod (resolution set by
    baseline and duration) instead of 900 evenly spaced periods.
  * Outlier clipping is upper-sided only; transits are never clipped.
  * Detrending: time-windowed running median per orbit/segment, with the
    strongest candidate's transits masked on a second pass.
  * Candidates that are simple period ratios of a stronger candidate are
    flagged as possible aliases and are not ranked as independent.
  * Depth/SNR/odd-even statistics come from astropy's BLS statistics.
  * Optional known-planet comparison and injection-recovery test.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.2.0"

# 'period' is deliberately NOT a feature (it invites leakage / overfitting to
# the sampling of the training set).
FEATURES = [
    "depth",
    "duration",
    "depth_snr",
    "shape_std",
    "noise_std",
    "odd_even_sigma",
]

GAP_DAYS = 0.5  # gaps longer than this split the light curve into segments


# ============================================================
# GENERAL HELPERS
# ============================================================

def _import_bls():
    try:
        from astropy.timeseries import BoxLeastSquares
    except ImportError:
        raise SystemExit(
            "astropy is required for the BLS search: "
            "pip install -r requirements.txt"
        )
    return BoxLeastSquares


def transit_offset(t, period, t0):
    """Time from the nearest mid-transit, in days, in [-P/2, P/2)."""
    return ((np.asarray(t) - t0 + 0.5 * period) % period) - 0.5 * period


def robust_sigma(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def split_segments(t, gap=GAP_DAYS):
    """Slices of a sorted time array separated by gaps longer than `gap`."""
    if len(t) == 0:
        return []
    breaks = np.where(np.diff(t) > gap)[0] + 1
    edges = np.concatenate([[0], breaks, [len(t)]])
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# DATA LOADING
# ============================================================

def load_light_curve(path):
    df = pd.read_csv(path)
    if not {"time", "flux"}.issubset(df.columns):
        raise ValueError("Light-curve CSV must contain columns: time, flux")
    cols = ["time", "flux"] + (["flux_err"] if "flux_err" in df.columns else [])
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


# ============================================================
# PREPROCESSING
# ============================================================

def running_median_trend(t, f, window, use=None, min_points=20):
    """Time-windowed running median. Points with use==False do not
    contribute to the trend (used to mask transits) but still get a trend
    value from their neighbours."""
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    if use is None:
        use = np.ones(len(t), bool)
    tu, fu = t[use], f[use]
    if len(tu) < min_points:
        return np.full(len(t), np.median(f))
    half = 0.5 * window
    lo = np.searchsorted(tu, t - half, side="left")
    hi = np.searchsorted(tu, t + half, side="right")
    trend = np.full(len(t), np.nan)
    for i in range(len(t)):
        if hi[i] - lo[i] >= min_points:
            trend[i] = np.median(fu[lo[i]:hi[i]])
    good = np.isfinite(trend)
    if not good.any():
        return np.full(len(t), np.median(f))
    if not good.all():
        trend[~good] = np.interp(t[~good], t[good], trend[good])
    return trend


def preprocess(
    df,
    detrend_window=1.0,
    sigma_upper=5.0,
    sigma_lower=None,
    transit_masks=None,
    gap=GAP_DAYS,
):
    """Normalise, detrend (per segment) and clip.

    Clipping is UPPER-sided by default: low points are potential transits and
    are never removed unless `sigma_lower` is set explicitly.

    transit_masks: list of (period, t0, duration); in-transit points (+50 %
    margin) are excluded when estimating the trend.

    Returns (t, flux, info).
    """
    t = df["time"].to_numpy(float)
    f = df["flux"].to_numpy(float)
    order = np.argsort(t)
    t, f = t[order], f[order]
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    n_input = len(t)

    f = f / np.median(f)

    use = np.ones(len(t), bool)
    for (p, t0, dur) in (transit_masks or []):
        use &= np.abs(transit_offset(t, p, t0)) > 0.75 * dur  # 1.5 x duration

    trend = np.empty_like(f)
    for sl in split_segments(t, gap):
        trend[sl] = running_median_trend(t[sl], f[sl], detrend_window, use[sl])
    fd = f / trend

    sigma = robust_sigma(fd)
    keep = fd < 1.0 + sigma_upper * sigma
    n_upper = int((~keep).sum())
    n_lower = 0
    if sigma_lower is not None:
        low = fd < 1.0 - sigma_lower * sigma
        n_lower = int((low & keep).sum())
        keep &= ~low

    info = {
        "n_input_finite": n_input,
        "n_removed_upper_clip": n_upper,
        "n_removed_lower_clip": n_lower,
        "n_kept": int(keep.sum()),
        "robust_sigma": float(sigma),
        "detrend_window_days": float(detrend_window),
        "sigma_upper": float(sigma_upper),
        "sigma_lower": None if sigma_lower is None else float(sigma_lower),
        "masked_transits": [list(map(float, m)) for m in (transit_masks or [])],
    }
    return t[keep], fd[keep], info


# ============================================================
# SYNTHETIC DEMO DATA
# ============================================================

def generate_demo_light_curve(seed=7):
    """Synthetic TESS-like curve (one injected box transit). Demo only."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27, 4500)
    flux = 1 + 0.0015 * np.sin(2 * np.pi * t / 11) + rng.normal(0, 0.0009, len(t))
    period, depth, duration = 3.4, 0.012, 0.12
    phase = ((t - 0.7 + period / 2) % period) - period / 2
    flux[np.abs(phase) < duration / 2] -= depth
    return t, flux


# ============================================================
# ALIAS DETECTION
# ============================================================

def classify_alias(period, reference_periods, tol=0.003, max_den=2, max_num=20):
    """Return (index_of_reference, 'a/b') if period ~= (a/b) * reference for
    small integers, else (None, ''). Heuristic: a flag, not a proof."""
    for j, ref in enumerate(reference_periods):
        r = period / ref
        for b in range(1, max_den + 1):
            a = int(round(r * b))
            if a < 1 or a > max_num or a == b:
                continue
            target = a / b
            if math.gcd(a, b) == 1 and abs(r - target) / target < tol:
                return j, f"{a}/{b}"
    return None, ""


# ============================================================
# BLS SEARCH
# ============================================================

def build_durations(min_duration, max_duration, n):
    return np.geomspace(min_duration, max_duration, n)


def search_candidates(
    t,
    flux,
    durations,
    min_period=0.8,
    max_period=10.0,
    frequency_factor=0.5,
    n_candidates=10,
    sep_frac=0.02,
    alias_tol=0.003,
):
    """BLS search on an autoperiod grid. Returns (candidates, spectrum, grid)."""
    BLS = _import_bls()
    durations = np.asarray(durations, float)
    if durations.max() >= min_period:
        raise ValueError("max duration must be smaller than min period")

    sigma = robust_sigma(flux)
    dy = np.full(len(flux), sigma)
    bls = BLS(t, flux, dy=dy)
    periods = bls.autoperiod(
        durations,
        minimum_period=min_period,
        maximum_period=max_period,
        minimum_n_transit=2,
        frequency_factor=frequency_factor,
    )
    res = bls.power(periods, durations, objective="snr")
    power = np.nan_to_num(np.asarray(res.power, float), nan=-np.inf)

    baseline = float(t.max() - t.min())
    smear = float(np.max(np.diff(periods) * baseline / periods[:-1]))
    grid = {
        "n_periods": int(len(periods)),
        "period_min": float(periods.min()),
        "period_max": float(periods.max()),
        "baseline_days": baseline,
        "max_phase_smear_days": smear,
        "min_duration_days": float(durations.min()),
        "grid_resolves_shortest_transit": bool(smear < durations.min()),
    }

    finite = power[np.isfinite(power)]
    med = np.median(finite)
    mad = 1.4826 * np.median(np.abs(finite - med)) or 1.0

    order = np.argsort(power)[::-1]
    picked = []
    cand_p = np.empty(0)
    for idx in order:
        if not np.isfinite(power[idx]):
            break
        p = float(res.period[idx])
        if cand_p.size and not np.all(np.abs(p - cand_p) / p > sep_frac):
            continue
        picked.append(int(idx))
        cand_p = np.append(cand_p, p)
        if len(picked) >= n_candidates:
            break

    rows = []
    independent_periods = []
    for idx in picked:
        p = float(res.period[idx])
        d = float(res.duration[idx])
        t0 = float(res.transit_time[idx])

        try:
            st = bls.compute_stats(p, d, t0)
            depth, depth_err = (float(x) for x in st["depth"])
            do, de = st["depth_odd"], st["depth_even"]
            oe = abs(do[0] - de[0]) / math.hypot(do[1], de[1])
            n_tr = int(np.sum(np.asarray(st["per_transit_count"]) > 0))
        except Exception:
            depth, depth_err = float(res.depth[idx]), float(res.depth_err[idx])
            oe, n_tr = np.nan, np.nan

        off = transit_offset(t, p, t0)
        inn = np.abs(off) < d / 2
        out = ~inn
        shape_std = float(np.std(flux[inn])) if inn.sum() > 2 else np.nan
        noise_std = float(np.std(flux[out])) if out.sum() > 10 else np.nan

        j, ratio = classify_alias(p, independent_periods, tol=alias_tol)
        is_alias = j is not None
        ref_p = independent_periods[j] if is_alias else np.nan
        if not is_alias:
            independent_periods.append(p)

        rows.append({
            "period": p,
            "duration": d,
            "t0": t0,
            "depth": depth,
            "depth_err": depth_err,
            "depth_snr": depth / depth_err if depth_err > 0 else np.nan,
            "power": float(power[idx]),
            "sde": float((power[idx] - med) / mad),
            "n_transits_observed": n_tr,
            "odd_even_sigma": oe,
            "shape_std": shape_std,
            "noise_std": noise_std,
            "at_duration_edge": bool(
                d >= 0.98 * durations.max() or d <= 1.02 * durations.min()
            ),
            "at_period_edge": bool(
                p <= 1.01 * min_period or p >= 0.99 * max_period
            ),
            "is_alias": is_alias,
            "alias_of_period": ref_p if is_alias else np.nan,
            "alias_ratio": ratio,
        })

    spectrum = {"period": np.asarray(res.period), "power": power}
    return pd.DataFrame(rows), spectrum, grid


# ============================================================
# OPTIONAL MACHINE LEARNING
# ============================================================

def train_optional_model(training_csv):
    if training_csv is None:
        return None
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    train = pd.read_csv(training_csv)
    missing = set(FEATURES + ["label"]) - set(train.columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    X = train[FEATURES].fillna(0.0)
    y = train["label"]
    if y.nunique() < 2:
        raise ValueError("Training labels must contain at least two classes.")

    if "group" in train.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tr, te = next(gss.split(X, y, groups=train["group"]))
        X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        print("\nTrain/test split: grouped by 'group' column.")
    else:
        print(
            "\nWARNING: no 'group' column; using a random row split. If rows "
            "come from the same star or injection, scores will be optimistic."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

    model = RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Model evaluation on held-out labelled data "
          f"(n_test={len(y_test)}, positives={int((y_test == 1).sum())}):")
    print(f"  Precision: {precision_score(y_test, pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_test, pred, zero_division=0):.3f}")
    print(f"  F1-score:  {f1_score(y_test, pred, zero_division=0):.3f}")
    return model


# ============================================================
# RANKING
# ============================================================

def rank_candidates(features, model=None):
    r = features.copy()
    pmax = r["power"].max()
    rel = r["power"] / pmax if pmax > 0 else 0.0

    if model is not None:
        X = r[FEATURES].fillna(0.0)
        classes = list(model.classes_)
        pos = classes.index(1) if 1 in classes else len(classes) - 1
        r["ml_probability"] = model.predict_proba(X)[:, pos]
        r["priority_score"] = r["ml_probability"]
        r["classification"] = model.predict(X)
    else:
        r["ml_probability"] = np.nan
        # Relative BLS SNR within this star only; not comparable across stars
        # and not a probability.
        r["priority_score"] = rel
        r["classification"] = "unclassified"

    r = r.sort_values(
        ["is_alias", "priority_score", "power"], ascending=[True, False, False]
    ).reset_index(drop=True)
    r.insert(0, "rank", np.arange(1, len(r) + 1))
    return r


def compare_to_known(ranked, known_period, known_depth=None,
                     known_duration=None, tol=0.005):
    """Annotate candidates with their relation to a known planet and print a
    recovered-vs-published comparison."""
    rel = []
    for p in ranked["period"]:
        if abs(p - known_period) / known_period < tol:
            rel.append("match")
        else:
            _, ratio = classify_alias(p, [known_period], tol=0.01)
            rel.append(f"alias {ratio} x known P" if ratio else "-")
    out = ranked.copy()
    out["known_relation"] = rel

    print("\nKnown-planet comparison")
    print(f"  Published period:   {known_period:.6f} d")
    match = out[out["known_relation"] == "match"]
    if match.empty:
        print("  No candidate within "
              f"{tol * 100:.1f}% of the published period.")
    else:
        m = match.iloc[0]
        print(f"  Recovered period:   {m['period']:.6f} d "
              f"({(m['period'] / known_period - 1) * 100:+.3f}%)")
        if known_duration:
            print(f"  Recovered duration: {m['duration']:.4f} d vs "
                  f"{known_duration:.4f} d published "
                  f"({(m['duration'] / known_duration - 1) * 100:+.0f}%)")
        if known_depth:
            print(f"  Recovered depth:    {m['depth'] * 100:.4f}% vs "
                  f"{known_depth * 100:.4f}% published "
                  f"({(m['depth'] / known_depth - 1) * 100:+.0f}%)")
    n_alias = int((out["known_relation"].str.startswith("alias")).sum())
    if n_alias:
        print(f"  {n_alias} other candidate(s) are simple period ratios of "
              "the known planet (not independent signals).")
    return out


# ============================================================
# PLOTS
# ============================================================

def _break_gaps(t, f, gap=GAP_DAYS):
    idx = np.where(np.diff(t) > gap)[0] + 1
    if idx.size:
        t = np.insert(t.astype(float), idx, np.nan)
        f = np.insert(f.astype(float), idx, np.nan)
    return t, f


def save_plot(t, flux, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tp, fp = _break_gaps(t, flux)
    plt.figure(figsize=(10, 4))
    plt.plot(tp, fp, linewidth=0.7)
    plt.xlabel("Time [BTJD, days]")
    plt.ylabel("Normalized flux")
    plt.title("Preprocessed TESS Light Curve")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def get_transit_centers(t, period, t0):
    k_min = int(np.floor((t.min() - t0) / period)) - 1
    k_max = int(np.ceil((t.max() - t0) / period)) + 1
    c = t0 + np.arange(k_min, k_max + 1) * period
    return c[(c >= t.min()) & (c <= t.max())]


def save_pipeline_results(t, flux, ranked, spectrum, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indep = ranked[~ranked["is_alias"]]
    best = indep.iloc[0] if len(indep) else ranked.iloc[0]
    period, dur = float(best["period"]), float(best["duration"])
    t0, depth = float(best["t0"]), float(best["depth"])
    label = (f"P = {period:.5f} d\nDuration = {dur:.4f} d ({dur * 24:.2f} h)\n"
             f"Depth = {depth * 100:.3f}%\nBLS SNR = {best['power']:.1f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    ax = axes[0]
    tp, fp = _break_gaps(t, flux)
    ax.plot(tp, fp, linewidth=0.5, alpha=0.85)
    for c in get_transit_centers(t, period, t0):
        ax.axvspan(c - dur / 2, c + dur / 2, alpha=0.2, color="tab:orange")
    ax.set_title("(a) Preprocessed light curve (shaded: predicted transits of best candidate)")
    ax.set_xlabel("Time [BTJD, days]")
    ax.set_ylabel("Normalized flux")
    ax.text(0.01, 0.03, label, transform=ax.transAxes, va="bottom",
            bbox=dict(boxstyle="round", alpha=0.85, fc="white"))

    ax = axes[1]
    ax.plot(spectrum["period"], spectrum["power"], linewidth=0.5)
    ax.axvline(period, color="tab:red", lw=1, label="best independent")
    al = ranked[ranked["is_alias"]]
    for i, p in enumerate(al["period"]):
        ax.axvline(p, color="gray", lw=0.8, ls=":",
                   label="flagged possible alias" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("BLS SNR")
    ax.set_title("(b) BLS periodogram")
    ax.legend(loc="upper right")

    ax = axes[2]
    off = transit_offset(t, period, t0)
    half = min(0.5 * period, max(4 * dur, 0.15))
    sel = np.abs(off) < half
    inn = np.abs(off) < dur / 2
    ax.scatter(off[sel & ~inn], flux[sel & ~inn], s=3, alpha=0.3, rasterized=True)
    ax.scatter(off[inn], flux[inn], s=6, alpha=0.6, color="tab:orange", rasterized=True)
    nb = max(10, int(2 * half / (dur / 6)))
    edges = np.linspace(-half, half, nb + 1)
    cen, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (off >= a) & (off < b)
        if m.sum() >= 5:
            cen.append(0.5 * (a + b))
            med.append(np.median(flux[m]))
    ax.plot(cen, med, color="k", lw=1.5, label="binned median")
    ax.axvspan(-dur / 2, dur / 2, alpha=0.15)
    ax.axvline(0, lw=0.8, ls="--")
    ax.set_xlabel("Time from mid-transit [days]")
    ax.set_ylabel("Normalized flux")
    ax.set_title("(c) Folded light curve, best independent candidate")
    ax.legend(loc="lower right")

    fig.suptitle("TESS Transit Candidate Analysis (research prototype)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# LATEX TABLE (no jinja2 dependency)
# ============================================================

def save_candidate_table(ranked, output, top_n=5, include_ml=False):
    top = ranked[~ranked["is_alias"]].head(top_n).reset_index(drop=True)
    heads = ["Rank", "Period (hari)", "Durasi (jam)", "Kedalaman (\\%)",
             "BLS SNR", "Catatan"]
    if include_ml:
        heads += ["ML Probability"]
    lines = ["% Automatically generated by the TESS Transit Candidate Pipeline",
             f"% Top {top_n} independent candidates (possible aliases excluded)",
             "\\begin{tabular}{rrrrrl" + ("r" if include_ml else "") + "}", "\\hline",
             " & ".join(heads) + " \\\\", "\\hline"]
    for i, r in top.iterrows():
        notes = []
        if r["at_duration_edge"]:
            notes.append("durasi di tepi grid")
        if r.get("known_relation") == "match":
            notes.append("planet dikenal")
        row = [str(i + 1), f"{r['period']:.5f}", f"{r['duration'] * 24:.2f}",
               f"{r['depth'] * 100:.4f}", f"{r['power']:.1f}",
               ", ".join(notes) or "--"]
        if include_ml:
            row.append(f"{r['ml_probability']:.3f}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# INJECTION-RECOVERY
# ============================================================

def injection_recovery(raw_df, args, durations, mask, n_trials, seed, output):
    """Inject box transits into the RAW flux (so detrending and clipping are
    part of the test), run the full single-pass pipeline, and record whether
    the injected period is recovered.

    `mask` = (period, t0, duration) of the known/strongest signal; its
    in-transit points are removed from the baseline first so it does not
    contaminate the test. Roughly 20 % of trials are noise-only, to measure
    how strong the top peak gets with no injected signal.
    """
    rng = np.random.default_rng(seed)
    base = raw_df.copy()
    if mask is not None:
        p, t0, d = mask
        keep = np.abs(transit_offset(base["time"].to_numpy(), p, t0)) > 0.75 * d
        base = base[keep].reset_index(drop=True)
    tt = base["time"].to_numpy(float)
    baseline = tt.max() - tt.min()
    pmax = min(args.max_period, baseline / 2.5)
    n_null = max(1, n_trials // 5)

    rows = []
    for k in range(n_trials + n_null):
        injected = k < n_trials
        P = float(np.exp(rng.uniform(np.log(args.min_period), np.log(pmax))))
        depth = float(np.exp(rng.uniform(np.log(2e-4), np.log(1e-2))))
        dur = float(min(args.max_duration, 0.076 * P ** (1 / 3)))
        t0 = float(tt.min() + rng.uniform(0, P))
        fl = base["flux"].to_numpy(float).copy()
        if injected:
            inn = np.abs(transit_offset(tt, P, t0)) < dur / 2
            fl = fl * (1.0 - depth * inn)
        df = pd.DataFrame({"time": tt, "flux": fl})
        t, f, _ = preprocess(df, args.detrend_window, args.sigma_upper)
        c, _, _ = search_candidates(
            t, f, durations, args.min_period, args.max_period,
            args.frequency_factor, n_candidates=5, alias_tol=args.alias_tol)
        ind = c[~c["is_alias"]]
        top = ind.iloc[0] if len(ind) else c.iloc[0]
        rec = injected and abs(top["period"] - P) / P < 0.01
        rec_alias = injected and (not rec) and (
            classify_alias(top["period"], [P], tol=0.01)[0] is not None)
        rows.append({
            "trial": k, "injected": injected,
            "inj_period": P if injected else np.nan,
            "inj_depth": depth if injected else np.nan,
            "inj_duration": dur if injected else np.nan,
            "top_period": float(top["period"]), "top_snr": float(top["power"]),
            "top_depth": float(top["depth"]),
            "recovered": bool(rec), "recovered_as_alias": bool(rec_alias),
        })
        print(f"  trial {k + 1}/{n_trials + n_null}", end="\r", flush=True)
    print()

    res = pd.DataFrame(rows)
    res.to_csv(output, index=False)
    inj = res[res["injected"]]
    null = res[~res["injected"]]
    print(f"\nInjection-recovery ({len(inj)} injected, {len(null)} noise-only)")
    print(f"  Overall recovery (period within 1 %): {inj['recovered'].mean():.2f}")
    print(f"  Recovered only as alias:              {inj['recovered_as_alias'].mean():.2f}")
    if len(inj) >= 10:
        bins = pd.cut(inj["inj_depth"], [2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
        print(inj.groupby(bins, observed=True)["recovered"].agg(["mean", "count"])
              .rename(columns={"mean": "recovery_fraction"}).to_string())
    print(f"  Noise-only top-peak SNR: median {null['top_snr'].median():.1f}, "
          f"max {null['top_snr'].max():.1f}  (candidates below this are "
          "indistinguishable from noise)")
    return res


# ============================================================
# MAIN
# ============================================================

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", type=str, help="CSV with time,flux columns")
    ap.add_argument("--training", type=str, help="Optional labelled feature CSV")
    ap.add_argument("--output", default="candidate_results.csv")
    ap.add_argument("--plot", default="light_curve.png")
    ap.add_argument("--pipeline-plot", default="pipeline_results.png")
    ap.add_argument("--latex-table", default="candidate_table.tex")
    ap.add_argument("--metadata", default="run_metadata.json")
    ap.add_argument("--top-n", type=int, default=5)
    g = ap.add_argument_group("search")
    g.add_argument("--min-period", type=float, default=0.8)
    g.add_argument("--max-period", type=float, default=10.0)
    g.add_argument("--min-duration", type=float, default=0.02)
    g.add_argument("--max-duration", type=float, default=0.30)
    g.add_argument("--n-durations", type=int, default=14)
    g.add_argument("--frequency-factor", type=float, default=0.5,
                   help="<1 gives a finer period grid (default 0.5)")
    g.add_argument("--alias-tol", type=float, default=0.003)
    g = ap.add_argument_group("preprocessing")
    g.add_argument("--detrend-window", type=float, default=1.0,
                   help="running-median window in days (>= 3x longest duration)")
    g.add_argument("--sigma-upper", type=float, default=5.0)
    g.add_argument("--sigma-lower", type=float, default=None,
                   help="OFF by default; low points may be transits")
    g.add_argument("--single-pass", action="store_true",
                   help="skip the second pass that masks the best candidate "
                        "while detrending")
    g = ap.add_argument_group("validation")
    g.add_argument("--known-period", type=float,
                   help="published period of a known planet (days)")
    g.add_argument("--known-depth", type=float,
                   help="published depth as a fraction (e.g. 0.01)")
    g.add_argument("--known-duration", type=float,
                   help="published T14 in days")
    g.add_argument("--injection-recovery", type=int, default=0, metavar="N",
                   help="run N injection trials (slow: one full BLS per trial)")
    g.add_argument("--injection-output", default="injection_recovery.csv")
    g.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    durations = build_durations(args.min_duration, args.max_duration,
                                args.n_durations)

    # ---- load -------------------------------------------------
    if args.input:
        print("\nData source: USER-PROVIDED LIGHT CURVE")
        raw = load_light_curve(args.input)
        source = {"input": args.input, "sha256": file_sha256(args.input)}
    else:
        print("\nData source: SYNTHETIC TESS-LIKE DEMONSTRATION")
        print("This output must NOT be described as real TESS observations.\n")
        t_d, f_d = generate_demo_light_curve()
        raw = pd.DataFrame({"time": t_d, "flux": f_d})
        source = {"input": "synthetic demo", "sha256": None}

    search_kw = dict(
        durations=durations, min_period=args.min_period,
        max_period=args.max_period, frequency_factor=args.frequency_factor,
        alias_tol=args.alias_tol,
    )

    # ---- pass 1 -----------------------------------------------
    t, flux, info = preprocess(raw, args.detrend_window, args.sigma_upper,
                               args.sigma_lower)
    print(f"Rows with finite time/flux:       {info['n_input_finite']:,}")
    print(f"Removed by upper-sided clip:      {info['n_removed_upper_clip']:,}")
    print(f"Observations used in BLS search:  {info['n_kept']:,}")
    print("Running BLS candidate search...")
    cands, spectrum, grid = search_candidates(t, flux, **search_kw)

    # ---- pass 2: mask best candidate while detrending ---------
    if not args.single_pass:
        ind = cands[~cands["is_alias"]]
        b = ind.iloc[0] if len(ind) else cands.iloc[0]
        mask = (float(b["period"]), float(b["t0"]), float(b["duration"]))
        print(f"Second pass: masking transits of P={mask[0]:.5f} d "
              "while detrending...")
        t, flux, info = preprocess(raw, args.detrend_window, args.sigma_upper,
                                   args.sigma_lower, transit_masks=[mask])
        print(f"Final pass: removed {info['n_removed_upper_clip']:,} points by "
              f"upper-sided clip; {info['n_kept']:,} observations used.")
        cands, spectrum, grid = search_candidates(t, flux, **search_kw)

    print(f"Period grid: {grid['n_periods']:,} periods; worst-case phase "
          f"smear over baseline = {grid['max_phase_smear_days']:.4f} d "
          f"(shortest duration {grid['min_duration_days']:.3f} d)")
    if not grid["grid_resolves_shortest_transit"]:
        print("WARNING: period grid too coarse for the shortest duration; "
              "decrease --frequency-factor.")
    if cands.empty:
        raise RuntimeError("No usable candidates were produced.")

    # ---- ML + ranking -----------------------------------------
    model = train_optional_model(args.training)
    ranked = rank_candidates(cands, model)
    if args.known_period:
        ranked = compare_to_known(ranked, args.known_period,
                                  args.known_depth, args.known_duration)

    # ---- save -------------------------------------------------
    ranked.to_csv(args.output, index=False)
    save_plot(t, flux, args.plot)
    save_pipeline_results(t, flux, ranked, spectrum, args.pipeline_plot)
    save_candidate_table(ranked, args.latex_table, top_n=args.top_n,
                         include_ml=(model is not None))

    meta = {
        "pipeline_version": __version__,
        "source": source,
        "preprocessing": info,
        "period_grid": grid,
        "durations_days": [float(d) for d in durations],
        "arguments": vars(args),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import astropy
        meta["astropy"] = astropy.__version__
    except ImportError:
        pass
    Path(args.metadata).write_text(json.dumps(meta, indent=2, default=str))

    # ---- report -----------------------------------------------
    cols = ["rank", "period", "duration", "depth", "depth_snr", "power",
            "n_transits_observed", "is_alias", "alias_ratio",
            "at_duration_edge"]
    if "known_relation" in ranked.columns:
        cols.append("known_relation")
    print("\nCandidates (aliases listed last, not independent):")
    print(ranked[cols].to_string(index=False))

    indep = ranked[~ranked["is_alias"]]
    best = indep.iloc[0] if len(indep) else ranked.iloc[0]
    print("\nBest independent candidate:")
    print(f"  Period:   {best['period']:.5f} d")
    print(f"  Duration: {best['duration']:.5f} d ({best['duration'] * 24:.2f} h)")
    print(f"  Depth:    {best['depth'] * 100:.4f}%")
    print(f"  BLS SNR:  {best['power']:.1f}   (SDE {best['sde']:.1f})")
    if best["at_duration_edge"]:
        print("  WARNING: duration is at the edge of the searched grid.")
    print(f"\nSaved: {args.output}, {args.plot}, {args.pipeline_plot}, "
          f"{args.latex_table}, {args.metadata}")

    # ---- injection-recovery -----------------------------------
    if args.injection_recovery > 0:
        print("\nRunning injection-recovery test (this can take a long time)...")
        mask = (float(best["period"]), float(best["t0"]), float(best["duration"]))
        injection_recovery(raw, args, durations, mask,
                           args.injection_recovery, args.seed,
                           args.injection_output)


if __name__ == "__main__":
    main()
