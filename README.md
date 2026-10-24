# TESS Exoplanet Transit Candidate Pipeline

A research prototype for preprocessing TESS light curves, searching for periodic transit-like signals with Box Least Squares (BLS), flagging aliases, ranking candidates, and exporting diagnostics.

**Status:** experimental prototype (v0.2.0). This repository does not claim a new exoplanet discovery.

> ## Known issues / errata (read first)
>
> The project report submitted to EUREKA! ITB 2026 (23 Oct 2026) was produced with **v0.1** of this code. A post-submission review found errors in that run that affect the reported numbers. The report text could not be amended after submission, so the corrections are documented here. **Where this README and the report disagree, this README is correct.**
>
> ### What v0.1 reported
> Top candidate: P = 0.943270 d, duration = 0.200 d (4.8 h), depth = 0.0367 %. Candidates 2–5 were presented as further ranked signals (Report Tabel 4.1 and C.3; Gambar 4.1, C.2).
> These files are kept in `results/original_run/` and marked `_ERRONEOUS`.
>
> ### What was wrong
>
> | # | Issue in v0.1 | Effect on reported results | Fixed in v0.2 |
> |---|---|---|---|
> | 1 | **Period grid too coarse.** `np.linspace(0.8, 10, 900)` has a step of 0.0102 d. Over a ~27 d baseline (~29 cycles) this smears a fold by ≈0.3 d, over three times the real transit duration. The reported 0.943270 d is exactly a grid node (0.8 + 14 × 0.010234). | The reported period is a grid artifact; the true period lies between nodes (≈0.2 % lower). Folded transit is smeared and shows a hole at phase 0 (Gambar 4.1). | `autoperiod` grid whose step is set by baseline and shortest duration; the run prints and warns about worst-case phase smear. |
> | 2 | **Two-sided outlier clip removed the transits.** `\|f − median\| < 8 × 1.4826 × MAD` with a robust σ of ≈0.066 % cuts everything more than ≈0.53 % below the median. | 1,501 of 18,299 rows (8.2 %, comparable to the planet's transit duty cycle) were removed and described in the report as "invalid" observations. The report's "16.798 valid observations" are what survived the clip, not the number of valid observations. | Upper-sided clip only; low points are never removed unless `--sigma-lower` is set. |
> | 3 | **Depth diluted and biased.** Depth was `median(out) − median(in)` for a 0.20 d box over a ≈0.09 d transit, on clipped data. | 0.0367 % is roughly 25× smaller than the planet's depth (≈1 %, published value). | Depth and SNR come from astropy BLS statistics at the fitted duration. |
> | 4 | **Duration grid too narrow / coarse** (12 values, 0.04–0.20 d). | The best duration sat on the upper edge (acknowledged in the report), and was ≈2× the published ≈2.2 h. | Geometric grid 0.02–0.30 d (14 values); an `at_duration_edge` flag is written for every candidate. |
> | 5 | **Candidates 2–5 are aliases of the same planet, not independent signals.** Their periods are 2.5, 1.5, 3.5 and 6.5 × the planet's period (ratios 2.502, 1.502, 3.502, 6.502 relative to the published period). | The "top-5 candidate list" is one planet plus its harmonics. | `classify_alias` flags period ratios a/b (b ≤ 2) of any stronger candidate; aliases are sorted below, and excluded from the LaTeX table and best-candidate plot. This is a heuristic flag, not proof. |
> | 6 | **Global quadratic detrend** across both orbits and the mid-sector gap, with no transit masking. | Residual baseline structure; transit depth sensitivity to the trend. | Per-segment running median (1 d window), then a second pass that masks the best candidate's transits while detrending (disable with `--single-pass`). |
> | 7 | **BLS "power" was unscaled** (no `dy`), giving values like 0.000318 that are not comparable across stars, and `priority_score` was power ÷ max power (so rank 1 = 1.0 by construction). | The power values in the report have no absolute meaning. | BLS runs with `objective="snr"` and a per-point error; SNR and SDE are reported. Priority score remains a *relative* ranking within one star. |
> | 8 | **Figure axis mislabelled.** The fold's x-axis was in days from mid-transit but labelled "Phase". Gaps were drawn as straight lines. | Cosmetic, but misleading. | Axis relabelled; gaps broken; binned median overlaid. |
> | 9 | **Reproducibility gaps.** No data-download script; sector, SAP/PDCSAP and exposure time not recorded; `to_latex` needs `jinja2` (not in requirements); the training CSV used `period` as a feature. | The reported run could not be reproduced from the repository alone. | `fetch_data.py` + JSON sidecar; LaTeX written without `jinja2`; `period` removed from the ML features; `run_metadata.json` records parameters, versions and input hash. |
>
> ### What the report states that this repo cannot support
> * The title, abstract and objectives describe a machine-learning classifier "tested on unseen data". **No classifier was trained or evaluated on any dataset in this repository.** The ML code path exists but is untested on real labelled data.
> * "Unreported" candidates: the only target analysed is WASP-18, a well-known planet host. **No search for unreported candidates was performed**, and no catalogue cross-match was done.
> * The report's remark that results "show the potential" of the pipeline should be read as: v0.1 did **not** correctly recover the known planet's period, depth or duration. v0.2 is the first version intended to.
>
> ### Corrected re-run (v0.2.0, 24 Oct 2026)
> Same input file as the report (`data/tess_real_lightcurve.csv`, 18,299 rows), run with `--known-period 0.9414526`. Environment: Python 3.14, numpy 2.5.4, pandas 2.3.3, astropy 8.0.2. Reference values are approximate and should be checked against the NASA Exoplanet Archive before they are cited.
>
> | Quantity | v0.1 (report) | v0.2 re-run | Published (verify) |
> |---|---|---|---|
> | Period (d) | 0.943270 | 0.941402 (−0.005 % vs 0.941453) | ≈ 0.9415 |
> | Duration (h) | 4.80 (grid edge) | 2.06 (0.086 d, a grid value) | ≈ 2.2 |
> | Depth (%) | 0.0367 | 0.953 | ≈ 1 |
> | BLS SNR | 0.000318 (unscaled "power") | 567.7 (SDE 24.6) | n/a |
> | Points removed by clipping | 1,501 | 0 | n/a |
> | Independent candidates | 5 (claimed) | 1 | 1 known planet |
>
> The other 9 candidates (periods 1.88, 2.82, 3.77, 4.71, 1.41, 5.65, 6.59, 7.53 and 8.47 d) are flagged as period ratios 2/1, 3/1, 4/1, 5/1, 3/2, 6/1, 7/1, 8/1 and 9/1 of the recovered period, so they are not independent signals.
> Remaining caveats: the duration comes from a coarse geometric grid (14 values), so it is only good to roughly 10 %; and the injection–recovery test (`--injection-recovery`) has not yet been run with real astropy, so no detection-efficiency number is claimed.

## What it does

1. Loads a light-curve CSV with `time` and `flux` columns (optional `flux_err`).
2. Normalises, detrends per orbit/segment with a running median, and clips *upward* outliers only.
3. Searches periods 0.8–10 d and durations 0.02–0.30 d with BLS on an `autoperiod` grid (SNR objective).
4. Extracts per-candidate statistics: depth, depth SNR, number of transits observed, odd/even depth difference, shape and noise scatter, and edge-of-grid flags.
5. Flags simple period-ratio aliases of stronger candidates.
6. Ranks candidates by relative BLS SNR. A Random Forest can optionally be trained from a labelled CSV (see below).
7. Optionally compares against a known planet (`--known-period`) and runs an injection–recovery test (`--injection-recovery N`).
8. Writes candidate CSV, diagnostic figures (light curve, periodogram, folded transit), a LaTeX table and `run_metadata.json`.

Without a trained model the ranking is **relative BLS SNR within one star**. It is not a probability that a signal is a planet.

## Installation

Python 3.10+ recommended.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Reproduce the WASP-18 check

```bash
python tess_exoplanet_pipeline_prototype.py \
    --input data/tess_real_lightcurve.csv \
    --known-period 0.9414526        # verify against the NASA Exoplanet Archive
```

Add `--known-depth 0.01 --known-duration 0.09` (also to be verified) to print recovered-vs-published differences. A correct run should report one independent candidate matching the known planet and every other peak flagged as an alias.

To fetch a fresh light curve with provenance recorded:

```bash
python -m pip install -r requirements-data.txt
python fetch_data.py --target "WASP-18" --sector 4 --output data/tess_wasp18_s4.csv
```

## Injection–recovery

```bash
python tess_exoplanet_pipeline_prototype.py --input data/tess_real_lightcurve.csv \
    --known-period 0.9414526 --injection-recovery 100 --seed 1
```

Box transits (log-uniform period 0.8–10 d and depth 0.02–1 %) are injected into the *raw* flux, so detrending and clipping are part of the test. The known planet's transits are removed from the baseline first, and ~20 % extra noise-only trials measure how strong the top peak gets with no signal. Each trial is a full BLS search; 100 trials can take a long time. Results are written to `injection_recovery.csv`. This is the only detection-performance measurement in this repository.

## Optional ML classifier

`--training labelled.csv` needs columns `depth, duration, depth_snr, shape_std, noise_std, odd_even_sigma, label` (label 1 = transit-like), and optionally `group` (rows with the same group, e.g. same star or same injected light curve, are never split between train and test). Without `group` a random row split is used and a warning is printed, because scores will then be optimistic. No labelled dataset is shipped, and the classifier has not been evaluated on real data.

## Tests

```bash
python -m pytest -q
```

The tests encode the failures above: transits survive preprocessing, aliases are flagged, the period grid resolves the transit, and a planet that sits between v0.1 grid nodes is recovered.

## Limitations

* One target, one sector, one known planet; nothing here shows the pipeline finds *new* signals.
* The running-median detrend (1 d window) partly suppresses real astrophysical variability on ≲1 d scales (WASP-18 has a phase curve); the window is configurable.
* BLS assumes box-shaped transits; limb-darkened shape, odd/even and secondary-eclipse vetting are only partly covered (odd/even sigma is reported but not used for rejection).
* Alias flagging is a period-ratio heuristic (tolerance 0.3 %, denominators ≤ 2). It can flag unrelated signals by chance and miss other relations.
* Only the strongest signal is masked in the second pass; multi-planet iterative search is not implemented.
* Candidates still require catalogue cross-matching (TOI / NASA Exoplanet Archive), centroid and contamination checks, and independent follow-up before any astronomical claim.

## Responsible use

BLS detections are candidate signals only. This code is for education and research prototyping.
