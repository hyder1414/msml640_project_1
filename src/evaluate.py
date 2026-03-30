from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from sift_detector import SIFTConfig, SIFTDetector
from utils import ensure_dir, list_image_paths, save_image


def write_csv(rows: list[dict], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_split(
    detector: SIFTDetector,
    reference_dir: Path,
    scene_dir: Path,
    expected_label: str,
    match_dir: Path,
    overlay_dir: Path,
) -> list[dict]:
    references = detector.load_references([str(p) for p in list_image_paths(reference_dir)])
    scene_paths = list_image_paths(scene_dir)

    rows: list[dict] = []
    for scene_path in scene_paths:
        result = detector.detect_with_best_reference(references, scene_path)

        stem = scene_path.stem
        if result.match_vis is not None:
            save_image(match_dir / f"{stem}_matches.jpg", result.match_vis)
        if result.overlay_vis is not None:
            save_image(overlay_dir / f"{stem}_overlay.jpg", result.overlay_vis)

        rows.append(
            {
                "scene_name": scene_path.name,
                "expected_label": expected_label,
                "predicted_detected": int(result.detected),
                "reference_used": result.reference_path.name,
                "good_matches": result.good_matches,
                "inliers": result.inliers,
                "homography_found": int(result.homography_found),
                "scene_keypoints": result.total_scene_keypoints,
                "reference_keypoints": result.total_reference_keypoints,
                "confidence": round(result.confidence, 4),
                "note": result.note,
            }
        )

        print(
            f"[{expected_label}] {scene_path.name} -> detected={result.detected} | "
            f"ref={result.reference_path.name} | good={result.good_matches} | inliers={result.inliers}"
        )

    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate bag-detection scenes with a SIFT pipeline.")
    parser.add_argument("--reference-dir", type=Path, default=Path("data/reference_v2"))
    parser.add_argument("--positive-dir", type=Path, default=Path("data/scene_positive_v2"))
    parser.add_argument("--negative-dir", type=Path, default=Path("data/scene_negative_v2"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/logs/bag_eval_v2.csv"))
    parser.add_argument("--matches-dir", type=Path, default=Path("results/matches_v2"))
    parser.add_argument("--overlays-dir", type=Path, default=Path("results/figures_v2"))
    parser.add_argument("--ratio-test", type=float, default=0.55)
    parser.add_argument("--min-good", type=int, default=12)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--max-dim", type=int, default=1000)
    parser.add_argument("--blur-ksize", type=int, default=0)
    parser.add_argument(
        "--use-clahe",
        action="store_true",
        help="Enable CLAHE preprocessing. Default is off for the final v2 setup.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    config = SIFTConfig(
        ratio_test=args.ratio_test,
        min_good_matches=args.min_good,
        min_inliers=args.min_inliers,
        max_dim=args.max_dim,
        blur_ksize=args.blur_ksize,
        use_clahe=args.use_clahe,
    )
    detector = SIFTDetector(config)

    reference_dir = args.reference_dir
    positive_dir = args.positive_dir
    negative_dir = args.negative_dir

    positive_match_dir = args.matches_dir / "positive"
    positive_overlay_dir = args.overlays_dir / "positive"
    negative_match_dir = args.matches_dir / "negative"
    negative_overlay_dir = args.overlays_dir / "negative"

    rows = []
    rows.extend(
        evaluate_split(
            detector,
            reference_dir,
            positive_dir,
            expected_label="positive",
            match_dir=positive_match_dir,
            overlay_dir=positive_overlay_dir,
        )
    )
    rows.extend(
        evaluate_split(
            detector,
            reference_dir,
            negative_dir,
            expected_label="negative",
            match_dir=negative_match_dir,
            overlay_dir=negative_overlay_dir,
        )
    )

    write_csv(rows, args.output_csv)
    print(f"\nSaved evaluation log to: {args.output_csv}")


if __name__ == "__main__":
    main()
