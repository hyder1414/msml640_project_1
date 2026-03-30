from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np

VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )


def load_bgr(path: str | Path) -> np.ndarray:
    path = str(path)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def resize_max_dim(image: np.ndarray, max_dim: int = 1200) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dim:
        return image, 1.0

    scale = max_dim / float(current_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_for_sift(
    bgr: np.ndarray,
    *,
    use_clahe: bool = True,
    blur_ksize: int = 0,
) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if blur_ksize and blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    return gray


def polygon_is_reasonable(polygon: np.ndarray, image_shape: Sequence[int]) -> bool:
    if polygon is None:
        return False

    h, w = image_shape[:2]
    pts = polygon.reshape(-1, 2)

    if np.isnan(pts).any() or np.isinf(pts).any():
        return False

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    if max_x < 0 or max_y < 0 or min_x > w or min_y > h:
        return False

    box_w = max_x - min_x
    box_h = max_y - min_y
    if box_w <= 5 or box_h <= 5:
        return False

    area = cv2.contourArea(pts.astype(np.float32))
    if area <= 25:
        return False

    image_area = float(h * w)
    if area > image_area * 0.95:
        return False

    return True


def draw_detection_polygon(
    image: np.ndarray,
    polygon: np.ndarray | None,
    *,
    label: str | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = image.copy()
    if polygon is not None:
        cv2.polylines(out, [np.int32(polygon)], isClosed=True, color=color, thickness=4)

    if label:
        cv2.putText(
            out,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def draw_matches_image(
    ref_bgr: np.ndarray,
    ref_kps,
    scene_bgr: np.ndarray,
    scene_kps,
    matches,
    *,
    max_draw_matches: int = 50,
) -> np.ndarray:
    draw_matches = list(matches[:max_draw_matches])
    return cv2.drawMatches(
        ref_bgr,
        ref_kps,
        scene_bgr,
        scene_kps,
        draw_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def write_text_block(image: np.ndarray, lines: Iterable[str]) -> np.ndarray:
    out = image.copy()
    y = 30
    for line in lines:
        cv2.putText(
            out,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return out


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to save image to {path}")