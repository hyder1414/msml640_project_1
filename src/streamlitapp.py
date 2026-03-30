from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import streamlit as st
import pandas as pd

from sift_detector import SIFTConfig, SIFTDetector, DetectionResult
from Utils import list_image_paths


st.set_page_config(page_title="Almond Bag Detector", layout="wide")


@st.cache_resource
def build_detector(
    ratio_test: float,
    min_good: int,
    min_inliers: int,
    max_dim: int,
    blur_ksize: int,
    use_clahe: bool,
) -> SIFTDetector:
    config = SIFTConfig(
        ratio_test=ratio_test,
        min_good_matches=min_good,
        min_inliers=min_inliers,
        max_dim=max_dim,
        use_clahe=use_clahe,
        blur_ksize=blur_ksize,
    )
    return SIFTDetector(config)


def choose_best_result(results: List[DetectionResult]) -> DetectionResult:
    return max(
        results,
        key=lambda r: (
            int(r.detected),
            r.inliers,
            r.good_matches,
            r.confidence,
        ),
    )


def save_uploaded_files(uploaded_files, target_dir: Path) -> List[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for uploaded in uploaded_files:
        path = target_dir / uploaded.name
        path.write_bytes(uploaded.getbuffer())
        saved_paths.append(path)
    return saved_paths


def detect_batch(reference_dir: Path, scene_paths: List[Path], detector: SIFTDetector):
    reference_paths = list_image_paths(reference_dir)
    if not reference_paths:
        raise FileNotFoundError(f"No reference images found in: {reference_dir}")

    references = [detector.load_reference(path) for path in reference_paths]

    batch_results: List[DetectionResult] = []
    for scene_path in scene_paths:
        results = [detector.detect_with_reference(ref, scene_path) for ref in references]
        best = choose_best_result(results)
        batch_results.append(best)

    return batch_results


def result_table(results: List[DetectionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "image": r.scene_path.name,
                "detected": "Yes" if r.detected else "No",
                "best_reference": r.reference_path.name,
                "good_matches": r.good_matches,
                "inliers": r.inliers,
                "confidence": round(r.confidence, 2),
                "note": r.note,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.title("Almond Bag Detector")
    st.caption("Upload one or more shelf images and the app will check which ones contain the almond bag using your SIFT pipeline.")

    with st.sidebar:
        st.header("Settings")
        reference_dir_text = st.text_input("Reference folder", value="data/reference")
        ratio_test = st.slider("Ratio test", min_value=0.50, max_value=0.95, value=0.75, step=0.01)
        min_good = st.number_input("Min good matches", min_value=1, max_value=200, value=10, step=1)
        min_inliers = st.number_input("Min inliers", min_value=1, max_value=200, value=6, step=1)
        max_dim = st.number_input("Max dimension", min_value=256, max_value=4000, value=1200, step=64)
        blur_ksize = st.number_input("Gaussian blur kernel", min_value=0, max_value=31, value=0, step=1)
        use_clahe = st.checkbox("Use CLAHE", value=True)
        detected_only = st.checkbox("Show detected images only", value=False)

    uploaded_scenes = st.file_uploader(
        "Upload shelf images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_scenes:
        st.info("Upload one or more images to start.")
        return

    detector = build_detector(
        ratio_test=ratio_test,
        min_good=int(min_good),
        min_inliers=int(min_inliers),
        max_dim=int(max_dim),
        blur_ksize=int(blur_ksize),
        use_clahe=use_clahe,
    )

    reference_dir = Path(reference_dir_text)
    if not reference_dir.exists():
        st.error(f"Reference folder not found: {reference_dir.resolve()}")
        return

    run_button = st.button("Run detection", type="primary")
    if not run_button:
        return

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            upload_dir = Path(tmpdir) / "uploads"
            scene_paths = save_uploaded_files(uploaded_scenes, upload_dir)

            with st.spinner("Running SIFT detection on uploaded images..."):
                results = detect_batch(reference_dir, scene_paths, detector)

        detected_count = sum(1 for r in results if r.detected)
        st.success(f"Done. Detected almond bag in {detected_count} of {len(results)} image(s).")

        df = result_table(results)
        st.subheader("Summary")
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download summary CSV",
            data=csv_bytes,
            file_name="almond_bag_detection_summary.csv",
            mime="text/csv",
        )

        st.subheader("Results")
        for r in results:
            if detected_only and not r.detected:
                continue

            status = " DETECTED" if r.detected else " NOT DETECTED"
            with st.expander(f"{status} — {r.scene_path.name}", expanded=r.detected):
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**Overlay / detected outline**")
                    if r.overlay_vis is not None:
                        st.image(r.overlay_vis, channels="BGR", use_container_width=True)
                    else:
                        st.write("No overlay available.")

                with c2:
                    st.markdown("**Feature match lines**")
                    if r.match_vis is not None:
                        st.image(r.match_vis, channels="BGR", use_container_width=True)
                    else:
                        st.write("No match visualization available.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Best reference", r.reference_path.name)
                m2.metric("Good matches", r.good_matches)
                m3.metric("Inliers", r.inliers)
                m4.metric("Confidence", f"{r.confidence:.2f}")
                st.caption(r.note)

    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
