from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from utils import (
    draw_detection_polygon,
    draw_matches_image,
    load_bgr,
    polygon_is_reasonable,
    preprocess_for_sift,
    resize_max_dim,
)


@dataclass
class SIFTConfig:
    nfeatures: int = 1200
    contrast_threshold: float = 0.04
    edge_threshold: float = 10
    sigma: float = 1.6
    ratio_test: float = 0.75
    min_good_matches: int = 10
    min_inliers: int = 8
    ransac_reproj_threshold: float = 5.0
    max_dim: int = 1200
    use_clahe: bool = True
    blur_ksize: int = 0
    max_draw_matches: int = 50


@dataclass
class ReferenceFeatures:
    path: Path
    bgr: np.ndarray
    gray: np.ndarray
    keypoints: list
    descriptors: Optional[np.ndarray]
    shape: tuple[int, int, int]


@dataclass
class DetectionResult:
    scene_path: Path
    reference_path: Path
    detected: bool
    good_matches: int
    inliers: int
    total_scene_keypoints: int
    total_reference_keypoints: int
    homography_found: bool
    confidence: float
    polygon: Optional[np.ndarray]
    match_vis: Optional[np.ndarray]
    overlay_vis: Optional[np.ndarray]
    note: str


class SIFTDetector:
    def __init__(self, config: SIFTConfig | None = None):
        self.config = config or SIFTConfig()
        self.sift = self._create_sift()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def _create_sift(self):
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "OpenCV SIFT is not available. Install opencv-contrib-python and try again."
            )
        return cv2.SIFT_create(
            nfeatures=self.config.nfeatures,
            contrastThreshold=self.config.contrast_threshold,
            edgeThreshold=self.config.edge_threshold,
            sigma=self.config.sigma,
        )

    def load_reference(self, path: str | Path) -> ReferenceFeatures:
        bgr = load_bgr(path)
        bgr, _ = resize_max_dim(bgr, self.config.max_dim)
        gray = preprocess_for_sift(
            bgr,
            use_clahe=self.config.use_clahe,
            blur_ksize=self.config.blur_ksize,
        )
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return ReferenceFeatures(
            path=Path(path),
            bgr=bgr,
            gray=gray,
            keypoints=keypoints,
            descriptors=descriptors,
            shape=bgr.shape,
        )

    def load_references(self, paths: List[str | Path]) -> List[ReferenceFeatures]:
        return [self.load_reference(p) for p in paths]

    def _prepare_scene(self, path: str | Path):
        bgr = load_bgr(path)
        bgr, _ = resize_max_dim(bgr, self.config.max_dim)
        gray = preprocess_for_sift(
            bgr,
            use_clahe=self.config.use_clahe,
            blur_ksize=self.config.blur_ksize,
        )
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return Path(path), bgr, gray, keypoints, descriptors

    def _ratio_test(self, knn_matches) -> list:
        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.config.ratio_test * n.distance:
                good.append(m)
        return sorted(good, key=lambda m: m.distance)

    def detect_with_reference(
        self,
        reference: ReferenceFeatures,
        scene_path: str | Path,
    ) -> DetectionResult:
        scene_path = Path(scene_path)
        _, scene_bgr, _, scene_kps, scene_desc = self._prepare_scene(scene_path)

        if reference.descriptors is None or len(reference.keypoints) == 0:
            return DetectionResult(
                scene_path=scene_path,
                reference_path=reference.path,
                detected=False,
                good_matches=0,
                inliers=0,
                total_scene_keypoints=len(scene_kps),
                total_reference_keypoints=len(reference.keypoints),
                homography_found=False,
                confidence=0.0,
                polygon=None,
                match_vis=None,
                overlay_vis=draw_detection_polygon(scene_bgr, None, label="NO DETECTION"),
                note="Reference image has no usable SIFT descriptors.",
            )

        if scene_desc is None or len(scene_kps) == 0:
            return DetectionResult(
                scene_path=scene_path,
                reference_path=reference.path,
                detected=False,
                good_matches=0,
                inliers=0,
                total_scene_keypoints=0,
                total_reference_keypoints=len(reference.keypoints),
                homography_found=False,
                confidence=0.0,
                polygon=None,
                match_vis=None,
                overlay_vis=draw_detection_polygon(scene_bgr, None, label="NO DETECTION"),
                note="Scene image has no usable SIFT descriptors.",
            )

        knn = self.matcher.knnMatch(reference.descriptors, scene_desc, k=2)
        good_matches = self._ratio_test(knn)

        match_vis = draw_matches_image(
            reference.bgr,
            reference.keypoints,
            scene_bgr,
            scene_kps,
            good_matches,
            max_draw_matches=self.config.max_draw_matches,
        )

        if len(good_matches) < self.config.min_good_matches:
            return DetectionResult(
                scene_path=scene_path,
                reference_path=reference.path,
                detected=False,
                good_matches=len(good_matches),
                inliers=0,
                total_scene_keypoints=len(scene_kps),
                total_reference_keypoints=len(reference.keypoints),
                homography_found=False,
                confidence=min(0.49, len(good_matches) / max(self.config.min_good_matches, 1)),
                polygon=None,
                match_vis=match_vis,
                overlay_vis=draw_detection_polygon(scene_bgr, None, label="NO DETECTION"),
                note="Too few good matches after Lowe ratio test.",
            )

        ref_pts = np.float32(
            [reference.keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        scene_pts = np.float32(
            [scene_kps[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            ref_pts,
            scene_pts,
            cv2.RANSAC,
            self.config.ransac_reproj_threshold,
        )

        if H is None or mask is None:
            return DetectionResult(
                scene_path=scene_path,
                reference_path=reference.path,
                detected=False,
                good_matches=len(good_matches),
                inliers=0,
                total_scene_keypoints=len(scene_kps),
                total_reference_keypoints=len(reference.keypoints),
                homography_found=False,
                confidence=0.45,
                polygon=None,
                match_vis=match_vis,
                overlay_vis=draw_detection_polygon(scene_bgr, None, label="NO DETECTION"),
                note="Homography could not be estimated.",
            )

        inliers = int(mask.ravel().sum())

        h_ref, w_ref = reference.bgr.shape[:2]
        ref_corners = np.float32(
            [[0, 0], [w_ref - 1, 0], [w_ref - 1, h_ref - 1], [0, h_ref - 1]]
        ).reshape(-1, 1, 2)

        polygon = cv2.perspectiveTransform(ref_corners, H)
        polygon_ok = polygon_is_reasonable(polygon, scene_bgr.shape)

        detected = inliers >= self.config.min_inliers and polygon_ok
        confidence = min(
            1.0,
            0.5 * (len(good_matches) / max(self.config.min_good_matches, 1))
            + 0.5 * (inliers / max(self.config.min_inliers, 1)),
        )

        label = f"DETECTED ({inliers} inliers)" if detected else "NO DETECTION"
        overlay = draw_detection_polygon(scene_bgr, polygon if detected else None, label=label)

        note = (
            "Successful bag detection."
            if detected
            else "Homography exists, but inlier count or polygon quality is too weak."
        )

        return DetectionResult(
            scene_path=scene_path,
            reference_path=reference.path,
            detected=detected,
            good_matches=len(good_matches),
            inliers=inliers,
            total_scene_keypoints=len(scene_kps),
            total_reference_keypoints=len(reference.keypoints),
            homography_found=True,
            confidence=confidence,
            polygon=polygon if detected else None,
            match_vis=match_vis,
            overlay_vis=overlay,
            note=note,
        )

    def detect_with_best_reference(
        self,
        references: List[ReferenceFeatures],
        scene_path: str | Path,
    ) -> DetectionResult:
        if not references:
            raise ValueError("No references were provided.")

        best: DetectionResult | None = None
        for ref in references:
            result = self.detect_with_reference(ref, scene_path)
            if best is None:
                best = result
                continue

            best_tuple = (
                int(best.detected),
                best.inliers,
                best.good_matches,
                best.confidence,
            )
            current_tuple = (
                int(result.detected),
                result.inliers,
                result.good_matches,
                result.confidence,
            )

            if current_tuple > best_tuple:
                best = result

        assert best is not None
        return best