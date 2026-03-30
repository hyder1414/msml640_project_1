import argparse
from pathlib import Path

from sift_detector import SIFTConfig, SIFTDetector
from Utils import save_image, list_image_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SIFT bag detection on a single image."
    )
    parser.add_argument("--reference-dir", type=Path, default=Path("data/reference"))
    parser.add_argument("--scene", type=Path, required=True, help="Single scene image to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--min-good", type=int, default=10)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--max-dim", type=int, default=1200)
    parser.add_argument("--blur-ksize", type=int, default=0)
    parser.add_argument("--disable-clahe", action="store_true")
    return parser


def build_detector(args: argparse.Namespace) -> SIFTDetector:
    config = SIFTConfig(
        ratio_test=args.ratio_test,
        min_good_matches=args.min_good,
        min_inliers=args.min_inliers,
        max_dim=args.max_dim,
        use_clahe=not args.disable_clahe,
        blur_ksize=args.blur_ksize,
    )
    return SIFTDetector(config)


def choose_best_result(results):
    return max(
        results,
        key=lambda r: (
            int(r.detected),
            r.inliers,
            r.good_matches,
            r.confidence,
        ),
    )


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    reference_paths = list_image_paths(args.reference_dir)
    if not reference_paths:
        raise FileNotFoundError(f"No reference images found in: {args.reference_dir}")

    detector = build_detector(args)
    references = [detector.load_reference(path) for path in reference_paths]

    results = [detector.detect_with_reference(ref, args.scene) for ref in references]
    best = choose_best_result(results)

    scene_stem = args.scene.stem
    ref_stem = best.reference_path.stem

    overlay_path = args.output_dir / f"{scene_stem}__best_{ref_stem}_overlay.jpg"
    matches_path = args.output_dir / f"{scene_stem}__best_{ref_stem}_matches.jpg"

    if best.overlay_vis is not None:
        save_image(overlay_path, best.overlay_vis)

    if best.match_vis is not None:
        save_image(matches_path, best.match_vis)

    status = "DETECTED" if best.detected else "NOT DETECTED"
    print(f"Result: {status}")
    print(f"Scene: {best.scene_path.name}")
    print(f"Best reference: {best.reference_path.name}")
    print(f"Good matches: {best.good_matches}")
    print(f"Inliers: {best.inliers}")
    print(f"Confidence: {best.confidence:.2f}")
    print(f"Note: {best.note}")
    print(f"Overlay saved to: {overlay_path}")

    if best.match_vis is not None:
        print(f"Matches saved to: {matches_path}")


if __name__ == "__main__":
    main()