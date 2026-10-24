"""Download a TESS SPOC light curve with Lightkurve and write a pipeline CSV
(time, flux, flux_err) plus a JSON sidecar recording exactly what was fetched.

    pip install -r requirements-data.txt
    python fetch_data.py --target "WASP-18" --sector 4 \
        --output data/tess_wasp18_s4.csv

NOTE: written for reproducibility of the data-retrieval step; it needs network
access and was not executed in the environment where v0.2 of the pipeline was
tested. Verify the printed search table before trusting the output.
"""
import argparse
import datetime
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="WASP-18")
    ap.add_argument("--sector", type=int, default=None,
                    help="TESS sector; strongly recommended so the run is reproducible")
    ap.add_argument("--author", default="SPOC")
    ap.add_argument("--exptime", type=int, default=120, help="seconds")
    ap.add_argument("--flux-column", default="pdcsap_flux",
                    choices=["pdcsap_flux", "sap_flux"])
    ap.add_argument("--output", default="data/tess_real_lightcurve.csv")
    args = ap.parse_args()

    import lightkurve as lk

    sr = lk.search_lightcurve(args.target, mission="TESS", author=args.author,
                              exptime=args.exptime, sector=args.sector)
    if len(sr) == 0:
        raise SystemExit("No matching light curves found.")
    print(sr)
    if len(sr) > 1 and args.sector is None:
        print("\nWARNING: several products match; using the first. "
              "Pass --sector to make this explicit.")

    lc = sr[0].download(flux_column=args.flux_column)  # default quality mask
    n_raw = len(lc)
    lc = lc.remove_nans()

    df = pd.DataFrame({
        "time": lc.time.value,  # BTJD = BJD - 2457000
        "flux": lc.flux.value,
        "flux_err": lc.flux_err.value,
    })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    meta = {
        "target": args.target,
        "sector": lc.meta.get("SECTOR"),
        "author": args.author,
        "exptime_s": args.exptime,
        "flux_column": args.flux_column,
        "quality_bitmask": "default",
        "n_rows_downloaded": n_raw,
        "n_rows_written": len(df),
        "time_system": "BTJD (BJD - 2457000)",
        "lightkurve_version": lk.__version__,
        "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out} ({len(df):,} rows) and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
