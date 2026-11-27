"""Safely extract and hash recovered V1.1 deadline results into versioned folders."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RECOVERED_RESEARCH_RESULTS.zip"
DEST = ROOT / "results" / "internal_defence_v1_1_2026-11-26"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def _extract_prefix(zf: zipfile.ZipFile, prefix: str, destination: Path,
                    exclude_nested: bool = False) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for info in zf.infolist():
        safe = _safe_name(info.filename)
        if info.is_dir() or not info.filename.startswith(prefix):
            continue
        relative = PurePosixPath(*safe.parts[len(PurePosixPath(prefix).parts):])
        if exclude_nested and relative.parts and relative.parts[0] == "deadline":
            continue
        if not relative.parts:
            continue
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = zf.read(info)
        if target.exists():
            if target.read_bytes() != payload:
                raise FileExistsError(f"Refusing to replace different existing result: {target}")
        else:
            target.write_bytes(payload)
        entries.append({"path": target.relative_to(DEST).as_posix(),
                        "bytes": target.stat().st_size, "sha256": sha256(target),
                        "source_member": info.filename})
    return entries


def main() -> Path:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Recovered archive not found: {SOURCE}")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as zf:
        bad = zf.testzip()
        if bad:
            raise zipfile.BadZipFile(f"CRC error in {bad}")
        outer = _extract_prefix(zf, "deadline/", DEST / "outer_deadline", exclude_nested=True)
        nested = _extract_prefix(zf, "deadline/deadline/", DEST / "nested_partial_deadline")
    manifest = {
        "created_date": "2026-11-26",
        "source_archive": SOURCE.name,
        "source_archive_bytes": SOURCE.stat().st_size,
        "source_archive_sha256": sha256(SOURCE),
        "archive_crc_check": "passed",
        "outer_deadline_extracted_files": len(outer),
        "nested_partial_extracted_files": len(nested),
        "outer_deadline": outer,
        "nested_partial_deadline": nested,
        "note": "Outer final evaluation and nested partial-stage checkpoints are extracted into separate directories. Original source ZIP is unchanged.",
    }
    (DEST / "RECOVERED_RESULTS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return DEST


if __name__ == "__main__":
    print(main())
