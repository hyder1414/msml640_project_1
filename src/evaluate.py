import csv
from pathlib import Path

from sift_detector import SIFTConfig, SIFTDetector
from Utils import ensure_dir, list_image_paths, save_image


def write_csv(rows: list[dict], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)