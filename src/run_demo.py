from __future__ import annotations

import argparse
from pathlib import Path

from sift_detector import SIFTConfig, SIFTDetector
from utils import list_image_paths, save_image


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SIFT bag detection on one image or an entire folder.")
    parser.add_argument("--reference-dir", type=Path, default=Path("data/reference_v2"))
    parser.add_argument("--scene", type=Path, default=None, help="Single scene image to evaluate.")
    parser.add_argument("--scene-dir", type=Path, default=None, help="Folder of scene images to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo_v2"))
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
    args = build_argparser().parse_args()

    if args.scene is None and args.scene_dir is None:
        raise ValueError("Provide either --scene or --scene-dir.")

    config = SIFTConfig(
        ratio_test=args.ratio_test,
        min_good_matches=args.min_good,
        min_inliers=args.min_inliers,
        max_dim=args.max_dim,
        blur_ksize=args.blur_ksize,
        use_clahe=args.use_clahe,
    )
    detector = SIFTDetector(config)
    references = detector.load_references([str(p) for p in list_image_paths(args.reference_dir)])

    scenes = [args.scene] if args.scene is not None else list_image_paths(args.scene_dir)

    for scene_path in scenes:
        result = detector.detect_with_best_reference(references, scene_path)

        stem = Path(scene_path).stem
        if result.match_vis is not None:
            save_image(args.output_dir / f"{stem}_matches.jpg", result.match_vis)
        if result.overlay_vis is not None:
            save_image(args.output_dir / f"{stem}_overlay.jpg", result.overlay_vis)

        print(
            f"Scene: {Path(scene_path).name}\n"
            f"  detected: {result.detected}\n"
            f"  reference used: {result.reference_path.name}\n"
            f"  good matches: {result.good_matches}\n"
            f"  inliers: {result.inliers}\n"
            f"  confidence: {result.confidence:.3f}\n"
            f"  note: {result.note}\n"
        )


if __name__ == "__main__":
    main()
