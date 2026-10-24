# Data provenance

`tess_real_lightcurve.csv` is the light-curve input used for the WASP-18 prototype run. It has `time` and `flux` columns (18,299 rows) and is provided so the reported run can be reproduced.

**What is known:** retrieved from TESS via Lightkurve, SPOC product, "first available observation", default quality filtering. Time is BTJD (BJD − 2,457,000), spanning roughly 1354–1382. The target is WASP-18, a known exoplanet host. The analysis is a method demonstration, not a new-planet claim.

**What was not recorded at the time** (and cannot be recovered from the CSV alone): the sector number, whether the flux column is SAP or PDCSAP, and the exposure time. Flux values are in e⁻/s (~47,000). For new runs use `fetch_data.py`, which records all of these in a JSON sidecar.
