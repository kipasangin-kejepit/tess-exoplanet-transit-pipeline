# TESS Exoplanet Transit Candidate Pipeline

A research prototype for preprocessing TESS light curves, searching for periodic transit-like signals with the Box Least Squares (BLS) algorithm, ranking candidate signals, and exporting diagnostic plots and tables.

**Status:** experimental prototype. This repository does not claim a new exoplanet discovery.

## Research team

- Giselle Ruslim
- Michelle Callysta Aurelia
- Perry Quinlan Petrus

## What it does

1. Loads a light-curve CSV with `time` and `flux` columns.
2. Removes non-finite observations and applies robust outlier clipping.
3. Normalizes flux and applies a simple polynomial detrending step.
4. Searches periods from 0.8 to 10 days with BLS and durations from 0.04 to 0.20 days.
5. Extracts candidate features and ranks candidates. A Random Forest classifier can optionally be trained if a suitable labelled training CSV is supplied.
6. Saves candidate results, diagnostic plots, and a LaTeX table.

For the included real-data run, ML classification was **not enabled**; candidates were ranked using BLS power. Do not interpret the ranking as a validated probability that a signal is a planet.

## Data

`data/tess_real_lightcurve.csv` contains the light curve used for the prototype run on the known exoplanet-host system WASP-18. The source observations were retrieved from TESS using Lightkurve (SPOC product; first available observation selected, default quality filtering). The included file contains 18,299 rows before the pipeline's cleaning step; 16,798 valid observations were retained in the reported run.

The observation time values are in BTJD (BJD - 2,457,000). Please consult the TESS archive and Lightkurve documentation for data-product details and provenance.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the pipeline

From the repository root:

```bash
python tess_exoplanet_pipeline_prototype.py --input data/tess_real_lightcurve.csv
```

Default outputs are written to the current working directory:

- `candidate_results.csv`
- `light_curve.png`
- `pipeline_results.png`
- `candidate_table.tex`

To choose output filenames, use `--output`, `--plot`, `--pipeline-plot`, and `--latex-table`. Run `python tess_exoplanet_pipeline_prototype.py --help` to see options.

## Preliminary result (reported run)

The highest-ranked BLS signal in the included run had:

- Period: 0.943270 days
- Duration: 0.200000 days (4.8 hours)
- Transit depth: 0.0367%
- BLS power: 0.000318

The fitted duration equals the upper boundary of the searched duration grid, so it may not be well constrained. WASP-18 is already a known exoplanet-host system; this result is a pipeline demonstration on real data, **not** a claim of an unreported planet.

The file `results/pipeline_results.png` is the diagnostic visualization from the reported run. The CSV in `results/top_candidates_reported.csv` records the rounded top-five values printed by that run; rerunning the code will generate the full-precision output CSV.

## Limitations and next steps

- Candidate signals have not been independently validated or cross-matched against an exoplanet catalogue in this prototype.
- BLS power is used for ranking in the included real-data run; the optional ML model was not evaluated on this dataset.
- The preprocessing and search grid are preliminary and require systematic validation.
- Follow-up work should include catalogue cross-matching, inspection of phase-folded signals and data gaps, testing on labelled data, and reproducibility checks.

## Reproducibility and responsible use

The pipeline is intended for educational and research prototyping. BLS detections are candidate signals only and require further astronomical validation before being interpreted as planetary transits.
