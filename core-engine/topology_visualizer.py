from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Line = Tuple[Point, Point]


def _compute_degree(lines: List[Line]) -> Dict[Point, int]:
    degree = defaultdict(int)
    for a, b in lines:
        degree[a] += 1
        degree[b] += 1
    return degree


def _collect_points(*line_groups: List[Line]) -> List[Point]:
    points: List[Point] = []
    for lines in line_groups:
        for a, b in lines:
            points.append(a)
            points.append(b)
    return points


def _build_transform(points: List[Point], canvas_w: int, canvas_h: int, margin: int = 50):
    if not points:
        return 1.0, 0.0, 0.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(1, max_x - min_x)
    span_y = max(1, max_y - min_y)

    usable_w = max(1, canvas_w - 2 * margin)
    usable_h = max(1, canvas_h - 2 * margin)

    scale = min(usable_w / span_x, usable_h / span_y)
    off_x = margin - min_x * scale
    off_y = margin - min_y * scale
    return scale, off_x, off_y


def _draw_grid(canvas: np.ndarray, points: List[Point], scale: float, off_x: float, off_y: float, grid: int = 20):
    if not points:
        return

    h, w = canvas.shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    gx = (min_x // grid) * grid
    while gx <= max_x + grid:
        sx = int(round(gx * scale + off_x))
        cv2.line(canvas, (sx, 0), (sx, h), (236, 236, 236), 1)
        gx += grid

    gy = (min_y // grid) * grid
    while gy <= max_y + grid:
        sy = int(round(gy * scale + off_y))
        cv2.line(canvas, (0, sy), (w, sy), (236, 236, 236), 1)
        gy += grid


def _draw_stage(
    canvas: np.ndarray,
    lines: List[Line],
    title: str,
    color: Tuple[int, int, int],
    scale: float,
    off_x: float,
    off_y: float,
):
    for (x1, y1), (x2, y2) in lines:
        p1 = (int(round(x1 * scale + off_x)), int(round(y1 * scale + off_y)))
        p2 = (int(round(x2 * scale + off_x)), int(round(y2 * scale + off_y)))
        cv2.line(canvas, p1, p2, color, 2)

    degree = _compute_degree(lines)
    dangling = 0
    for (x, y), d in degree.items():
        if d != 1:
            continue
        dangling += 1
        p = (int(round(x * scale + off_x)), int(round(y * scale + off_y)))
        cv2.circle(canvas, p, 4, (0, 0, 255), -1)
        cv2.circle(canvas, p, 6, color, 1)

    cv2.putText(canvas, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(canvas, f"segments: {len(lines)}", (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.putText(canvas, f"dangling endpoints: {dangling}", (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)


def show_topology_pipeline(
    raw_lines: List[Line],
    split_lines: List[Line],
    refined_lines: List[Line],
    title: str = "TOPOLOGY PIPELINE",
    stage_size: Tuple[int, int] = (760, 560),
):
    points = _collect_points(raw_lines, split_lines, refined_lines)
    stage_w, stage_h = stage_size

    canvases = [
        np.ones((stage_h, stage_w, 3), dtype=np.uint8) * 255,
        np.ones((stage_h, stage_w, 3), dtype=np.uint8) * 255,
        np.ones((stage_h, stage_w, 3), dtype=np.uint8) * 255,
    ]

    scale, off_x, off_y = _build_transform(points, stage_w, stage_h)

    for c in canvases:
        _draw_grid(c, points, scale, off_x, off_y)

    _draw_stage(canvases[0], raw_lines, "RAW (MERGED)", (255, 130, 0), scale, off_x, off_y)
    _draw_stage(canvases[1], split_lines, "SPLIT (INTERSECTIONS)", (0, 140, 255), scale, off_x, off_y)
    _draw_stage(canvases[2], refined_lines, "REFINED (TOPOLOGY)", (40, 180, 40), scale, off_x, off_y)

    panel = np.hstack(canvases)
    cv2.putText(panel, "Red center marker = dangling endpoint (degree 1)", (16, stage_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1)

    cv2.imshow(title, panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _to_bgr_image(image) -> np.ndarray:
    arr = np.array(image)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    raise ValueError("Unsupported image format for visualization")


def show_refined_on_original(
    original_image,
    refined_lines: List[Line],
    title: str = "TOPOLOGY REFINED OVERLAY",
    line_color: Tuple[int, int, int] = (255, 0, 0),
    line_thickness: int = 2,
):
    canvas = _to_bgr_image(original_image)

    for (x1, y1), (x2, y2) in refined_lines:
        cv2.line(canvas, (x1, y1), (x2, y2), line_color, line_thickness)

    cv2.putText(canvas, f"Refined segments: {len(refined_lines)}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(canvas, f"Refined segments: {len(refined_lines)}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 1)

    cv2.imshow(title, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
