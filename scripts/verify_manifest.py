#!/usr/bin/env python3
"""Verify, or regenerate, the SHA256 manifest of the released files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic_forecast.io import sha256_file, write_json

SKIP = {"MANIFEST.json", "SHA256SUMS", ".git", "__pycache__", ".pytest_cache", "runs"}


def collect() -> list[dict[str, object]]:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if any(part in SKIP for part in Path(relative).parts) or relative.endswith(".pyc"):
            continue
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate MANIFEST.json.")
    args = parser.parse_args()
    manifest_path = ROOT / "MANIFEST.json"
    entries = collect()
    if args.write:
        write_json(manifest_path, {"root": ".", "files": entries})
        print(f"wrote {len(entries)} entries")
        return
    expected = {item["path"]: item for item in json.loads(manifest_path.read_text())["files"]}
    failures = []
    for item in entries:
        record = expected.get(item["path"])
        if record is None:
            failures.append(f"unlisted file: {item['path']}")
        elif record["sha256"] != item["sha256"]:
            failures.append(f"hash mismatch: {item['path']}")
    for path in expected:
        if not (ROOT / path).exists():
            failures.append(f"missing file: {path}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"verified {len(entries)} files")


if __name__ == "__main__":
    main()
