from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils import (
    draw_detection_polygon,
    draw_matches_image,
    load_bgr,
    preprocess_for_sift,
    resize_max_dim,
)


@dataclass
class SIFTConfig:
    nfeatures: int = 1000
    contrast_threshold: float = 0.04
    edge_threshold: float = 10
    sigma: float = 1.6
    ratio_test: float = 0.80
    min_good_matches: int = 8
    max_dim: int = 1200
    use_clahe: bool = True
    blur_ksize: int = 0
    max_draw_matches: int = 40


@dataclass
class ReferenceFeatures:
    path: Path
    bgr: np.ndarray
    gray: np.ndarray
    keypoints: list
    descriptors: Optional[np.ndarray]


@dataclass
class DetectionResult:
    scene_path: Path
    reference_path: Path
    detected: bool
    good_matches: int
    total_scene_keypoints: int
    total_reference_keypoints: int
    confidence: float
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
                "OpenCV SIFT is not available. Install opencv-contrib-python."
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
        )

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

    def detect_with_reference(
        self,
        reference: ReferenceFeatures,
        scene_path: str | Path,
    ) -> DetectionResult:
        scene_path = Path(scene_path)
        _, scene_bgr, _, scene_kps, scene_desc = self._prepare_scene(scene_path)

        if reference.descriptors is None or scene_desc is None:
            return DetectionResult(
                scene_path=scene_path,
                reference_path=reference.path,
                detected=False,
                good_matches=0,
                total_scene_keypoints=len(scene_kps),
                total_reference_keypoints=len(reference.keypoints),
                confidence=0.0,
                match_vis=None,
                overlay_vis=draw_detection_polygon(scene_bgr, None, label="NO DETECTION"),
                note="Missing descriptors in reference or scene.",
            )

        knn = self.matcher.knnMatch(reference.descriptors, scene_desc, k=2)

        good_matches = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.config.ratio_test * n.distance:
                good_matches.append(m)

        match_vis = draw_matches_image(
            reference.bgr,
            reference.keypoints,
            scene_bgr,
            scene_kps,
            good_matches,
            max_draw_matches=self.config.max_draw_matches,
        )

        detected = len(good_matches) >= self.config.min_good_matches
        confidence = len(good_matches) / self.config.min_good_matches

        label = "DETECTED" if detected else "NO DETECTION"
        overlay = draw_detection_polygon(scene_bgr, None, label=label)

        return DetectionResult(
            scene_path=scene_path,
            reference_path=reference.path,
            detected=detected,
            good_matches=len(good_matches),
            total_scene_keypoints=len(scene_kps),
            total_reference_keypoints=len(reference.keypoints),
            confidence=confidence,
            match_vis=match_vis,
            overlay_vis=overlay,
            note="Basic version using only match count.",
        )