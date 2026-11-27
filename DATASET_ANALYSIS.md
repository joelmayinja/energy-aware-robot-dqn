# Dataset analysis

## Existing steel-industry data

The existing `notebooks/steel_data.csv` was inspected in place and was not moved. It contains 35,040 rows and 11 columns, covering 2018 at exact 15-minute intervals. Parsing found no missing cells, duplicate rows, or irregular intervals. The `Usage_kWh` field has a mean of 27.387 kWh per recorded interval, a median of 4.57 kWh, and a maximum of 157.18 kWh. The recorded load labels are Light (18,072), Medium (9,696), and Maximum (7,272).

Because the observed interval is 0.25 h, average interval power is derived as `Usage_kWh / 0.25 h`. This is a descriptive conversion for this dataset. The data are not an electricity supply measurement, do not establish blackout events, do not provide robot joint labels, and do not validate the KUKA simulation's energy model.

Reproduce the summary and plots with:

```powershell
python experiments/v1_1/analyze_steel.py
```

Generated outputs for the dated analysis are in `experiments/v1_1/runs/2026-11-26/dataset_analysis/`: the JSON summary, hourly summary CSV, top 1% interval CSV, and four-panel PNG. The plot is descriptive, not inferential.

## Additional robot datasets

Official metadata was inspected for the SDU collaborative-robot dataset. Its DOI resolves to an IEEE DataPort HTML page from this session, and no data file was retrieved.

For the Mendeley high-payload dataset, the public file-list API returned a root manifest and the 5,547-byte `readme.txt` was downloaded and inspected at `data/raw/high_payload_robot/readme.txt` (SHA-256 in the manifest). The README documents electrical total power/energy and six-axis KUKA trace mechanical power/energy, motor and joint variables, plus movement speed and acceleration settings. It explicitly warns that direct electrical measurements and robot trace measurements are not synchronized and have shifted time vectors. This is documentation inspection only; the measurement samples were not obtained. The official Download All archive is 2,709,034,696 bytes in 26 split volumes, so the bulk archive was not fetched or included in the project ZIP.

There is therefore no robot-data power regression, held-out trajectory score, or measured speed-energy result yet. See `DATASET_PROVENANCE.md` and `DATASET_MANIFEST.json` for detailed access and integrity records.
