"""Download daily cycle-station observations without downloading the full dataset."""

import argparse
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / ".hf_cache"
REPO_ID = "feregrino/london-cycles"


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def download_file(filename: str, attempts: int = 3) -> Path:
    """Download a public dataset file with bounded retries."""
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    cache_dir=str(CACHE_DIR),
                )
            )
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                wait_seconds = 2**attempt
                print(
                    f"Download attempt {attempt} failed: {exc}. "
                    f"Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to download {filename} after {attempts} attempts."
    ) from last_error


def main():
    parser = argparse.ArgumentParser(
        description="Download daily London cycle-hire observations."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if end < start:
        parser.error("--end must be on or after --start")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = []

    for day in date_range(start, end):
        relative = (
            f"data/year={day.year}/month={day.month:02d}/day={day.day:02d}/part.csv"
        )
        destination = (
            RAW_DIR
            / f"year={day.year}"
            / f"month={day.month:02d}"
            / f"day={day.day:02d}"
            / "part.csv"
        )

        if destination.exists() and destination.stat().st_size > 0:
            print(f"[SKIP] Already downloaded: {day}")
            skipped += 1
            continue

        print(f"[DOWNLOAD] {day}: {relative}")

        try:
            cached_file = download_file(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_file, destination)

            if destination.stat().st_size == 0:
                destination.unlink(missing_ok=True)
                raise RuntimeError("Downloaded file is empty.")

            print(f"[OK] {destination.stat().st_size:,} bytes")
            downloaded += 1
        except Exception as exc:
            failed.append((str(day), str(exc)))
            print(f"[FAILED] {day}: {exc}")

    # Station attributes are versioned and separate from the observations.
    station_file = RAW_DIR / "stations.csv"
    if not station_file.exists():
        try:
            print("[DOWNLOAD] Station metadata")
            cached_file = download_file("stations.csv")
            shutil.copy2(cached_file, station_file)
            print(f"[OK] {station_file}")
        except Exception as exc:
            failed.append(("stations.csv", str(exc)))
            print(f"[FAILED] Station metadata: {exc}")

    print("\nINGESTION SUMMARY")
    print(f"Downloaded daily files: {downloaded}")
    print(f"Skipped existing files: {skipped}")
    print(f"Failures: {len(failed)}")

    if failed:
        for item, reason in failed:
            print(f"  - {item}: {reason}")
        raise SystemExit(1)

    if downloaded + skipped == 0:
        raise SystemExit("No daily observation files were obtained.")


if __name__ == "__main__":
    main()
