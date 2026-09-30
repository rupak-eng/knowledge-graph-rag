#!/usr/bin/env python3
"""Download the real SEC FY2024 10-K corpus (reproduces data/raw/).

The corpus itself is gitignored (data/raw/); run this to re-fetch it.
SEC asks for a descriptive User-Agent on automated requests.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

FILINGS = [
    (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
        "aapl-20240928.htm",
        "Apple Inc. FY2024 10-K",
    ),
    (
        "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
        "msft-20240630.htm",
        "Microsoft Corp FY2024 10-K",
    ),
    (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm",
        "nvda-20240128.htm",
        "NVIDIA Corp FY2024 10-K",
    ),
]

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = "KRAG research project (contact: repo admin)"


def fetch(url: str, dest: Path) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    print(f"{dest.name}: {len(data):,} bytes  sha256={sha[:16]}...")
    return sha


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {"documents": []}
    doc_ids = ["aapl", "msft", "nvda"]
    for (url, filename, title), doc_id in zip(FILINGS, doc_ids, strict=True):
        dest = RAW / filename
        if dest.exists():
            print(f"{filename}: already present, skipping")
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        else:
            sha = fetch(url, dest)
        manifest["documents"].append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "title": title,
                "url": url,
                "sha256": sha,
            }
        )
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", RAW / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
