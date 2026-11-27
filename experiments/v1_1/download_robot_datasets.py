"""Try official public dataset endpoints; never authenticates or bypasses access."""
from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests


TIMEOUT = 30
HEADERS = {"User-Agent": "EnergyAwareRobotResearch/1.1 (dataset provenance; no automated login)"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch_mendeley(out: Path) -> dict:
    landing = "https://data.mendeley.com/datasets/hdhchngky8/1"
    api = "https://data.mendeley.com/public-api/datasets/hdhchngky8/files?folder_id=root&version=1&$start=0&$limit=1000"
    record = {"source": landing, "api_attempted": api, "downloaded": False, "files": []}
    try:
        headers = {**HEADERS, "Accept": "application/vnd.mendeley-public-dataset.1+json"}
        response = requests.get(api, headers=headers, timeout=TIMEOUT)
        record["http_status"] = response.status_code
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            record["api_error"] = payload["error"]
            record["access_note"] = "Official public API returned an application-level error; no files were listed or downloaded."
            return record
        entries = payload if isinstance(payload, list) else payload.get("files", [])
        record["file_listing"] = []
        for entry in entries:
            name = Path(str(entry.get("filename") or entry.get("name") or "")).name
            details = entry.get("content_details") or {}
            if not name:
                continue
            item = {"name": name, "bytes": details.get("size"),
                    "sha256": details.get("sha256_hash"), "file_id": entry.get("id")}
            record["file_listing"].append(item)
            # Retrieve only the small documentation file. The dataset's published
            # Download All archive is multi-gigabyte, so this script does not pull it.
            if name.lower() != "readme.txt" or int(details.get("size") or 0) > 10_000_000:
                continue
            url = details.get("download_url")
            if not url:
                continue
            target = out / name
            source_url = url
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            record["files"].append({"name": name, "bytes": target.stat().st_size,
                                    "sha256": sha256(target), "source_url": source_url})
        record["downloaded"] = bool(record["files"])
        # HEAD the official Download All route to record the archive size without
        # fetching its body; do not download the multi-gigabyte archive automatically.
        try:
            zip_url = "https://data.mendeley.com/public-api/zip/hdhchngky8/download/1"
            probe = requests.get(zip_url, headers={**HEADERS, "Range": "bytes=-1"},
                                 timeout=TIMEOUT, stream=True)
            record["bulk_archive_http_status"] = probe.status_code
            content_range = probe.headers.get("Content-Range", "")
            if "/" in content_range:
                record["bulk_archive_size_bytes"] = int(content_range.rsplit("/", 1)[1])
            probe.close()
        except Exception as exc:
            record["bulk_archive_probe_error"] = f"{type(exc).__name__}: {exc}"
        if not record["files"]:
            record["access_note"] = "Official API response contained no downloadable file entries."
        else:
            record["access_note"] = "Downloaded and hashed the small official README; the multi-gigabyte split data archive was not downloaded."
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def probe_sdu(out: Path) -> dict:
    doi = "https://doi.org/10.21227/9wnt-8v86"
    record = {"source": "https://portal.findresearcher.sdu.dk/en/datasets/dataset-of-collaborative-robots-for-energy-consumption-modeling/",
              "doi_attempted": doi, "downloaded": False, "files": []}
    try:
        response = requests.get(doi, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        record["http_status"] = response.status_code
        record["final_url"] = response.url
        content_type = response.headers.get("Content-Type", "").lower()
        disposition = response.headers.get("Content-Disposition", "")
        record["content_type"] = content_type
        # Save only an obvious direct research archive/file. HTML/login pages are never
        # mistaken for data, and no credentials or access-control bypass is attempted.
        if response.ok and ("zip" in content_type or "octet-stream" in content_type or "attachment" in disposition.lower()):
            filename = re.findall(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', disposition, flags=re.I)
            name = Path(filename[0] if filename else "sdu_dataset_download").name
            target = out / name
            target.write_bytes(response.content)
            record["files"].append({"name": name, "bytes": target.stat().st_size,
                                    "sha256": sha256(target), "url": response.url})
            record["downloaded"] = True
        else:
            record["access_note"] = "DOI resolved to a web page or non-file response; inspect the official page and confirm terms manually. No data file saved."
            page_text = response.text
            page_lower = page_text.lower()
            record["login_required_for_dataset_files"] = "login to access dataset files" in page_lower
            archive_size = re.search(r"datasets\.zip\s*\(\s*size:\s*([0-9.]+\s*(?:mb|gb))", page_text, re.I)
            if archive_size:
                record["dataset_archive_size_listed"] = archive_size.group(1)
            readme_match = re.search(r'href="(https://ieee-dataport\.s3\.amazonaws\.com[^"]+)"[^>]*>\s*README\.md', page_text, re.I)
            if readme_match:
                readme_url = html.unescape(readme_match.group(1))
                try:
                    readme = requests.get(readme_url, headers=HEADERS, timeout=TIMEOUT)
                    record["readme_http_status"] = readme.status_code
                    if readme.ok and len(readme.content) < 10_000_000:
                        target = out / "README.md"
                        target.write_bytes(readme.content)
                        record["files"].append({"name":"README.md","bytes":target.stat().st_size,
                                                "sha256":sha256(target)})
                except Exception as exc:
                    record["readme_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(timezone.utc).isoformat()
    sdu_dir = root / "data/raw/collaborative_robot"
    men_dir = root / "data/raw/high_payload_robot"
    sdu_dir.mkdir(parents=True, exist_ok=True)
    men_dir.mkdir(parents=True, exist_ok=True)
    result = {"attempted_at_utc": timestamp, "policy": "official public endpoints only; no login or access-control bypass",
              "sdu": probe_sdu(sdu_dir), "mendeley": fetch_mendeley(men_dir)}
    dest = root / "experiments/v1_1/runs/2026-11-26/dataset_download_attempt.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
