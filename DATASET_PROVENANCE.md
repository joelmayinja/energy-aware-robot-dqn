# Dataset provenance and access record

## Dataset A — UCI steel-industry energy consumption

- Citation/source: [UCI Machine Learning Repository, Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption), DOI 10.24432/C52G8C.
- Local file inspected: `notebooks/steel_data.csv` (2,731,386 bytes).
- Verified SHA-256 is stored in `DATASET_MANIFEST.json`.
- License listed by UCI: CC BY 4.0.
- File integrity/coverage checks: 35,040 x 11, zero missing cells, zero duplicate rows, one exact 15-minute spacing throughout 2018.
- Use: industrial demand context only. No robot policy labels or simulated power supply are inferred from it.

## Dataset B — SDU collaborative robots

- Citation: Heredia Mena, J. E.; Schlette, C.; Kjærgaard, M. B. (2021), *Dataset of Collaborative Robots for Energy Consumption Modeling*, IEEE DataPort, DOI [10.21227/9wnt-8v86](https://doi.org/10.21227/9wnt-8v86).
- Official metadata page: [University of Southern Denmark](https://portal.findresearcher.sdu.dk/en/datasets/dataset-of-collaborative-robots-for-energy-consumption-modeling/).
- Metadata reports UR3e and UR10e with joint and electrical measurements. No raw file was downloaded or inspected in this run. The metadata page did not state a license; verify dataset terms on IEEE DataPort before reuse.
- On 26 November 2026, a direct public DOI request resolved to the official IEEE DataPort landing page (HTTP 200, `text/html`). The page lists `datasets.zip` as 101.47 MB and explicitly says login is required to access data files. Its public README link returned HTTP 403. No measurement archive was retrieved and no authentication was attempted. Exact response details are in `experiments/v1_1/runs/2026-11-26/dataset_download_attempt.json`.

## Dataset C — high-payload industrial robot

- Citation: Gadaleta, M.; Berselli, G.; Pellicciari, M.; Grassia, F. (2020), *DATA of the paper: “Extensive Experimental Investigation for the Optimization of the Energy Consumption of a High Payload Industrial Robot with Open Research Dataset”*, Mendeley Data v1, DOI [10.17632/hdhchngky8.1](https://doi.org/10.17632/hdhchngky8.1).
- Official landing page: [Mendeley Data](https://data.mendeley.com/datasets/hdhchngky8/1), which states CC BY 4.0 and describes three experiment files plus MATLAB plotting scripts.
- On 26 November 2026, the official public file-list API returned the root file manifest. The 5,547-byte `readme.txt` was downloaded from its listed public-file URL and verified against the API SHA-256. It documents electrical and six-axis KUKA trace fields and warns that their measurement time vectors are shifted/not synchronized. It is documentation, not the experiment samples. The Download All endpoint reports 2,709,034,696 bytes in 26 split volumes; the bulk archive was not downloaded. Exact response details are in `experiments/v1_1/runs/2026-11-26/dataset_download_attempt.json`.

## Reproducible retrieval

Retry through the official landing pages from a network-enabled Colab session and save original files under `data/raw/collaborative_robot/` and `data/raw/high_payload_robot/`. Record the actual access URL, timestamp, byte size, SHA-256, license text, filenames, and inspection outcomes before analysis. Do not use credentials or methods that bypass dataset access terms.
