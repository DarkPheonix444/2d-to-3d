import cv2
import numpy as np
from typing import List, Tuple, Dict

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class WallDetector:

    def __init__(self, debug=True):
        self.debug = debug

        self.config_ratios = [
            {"th": 120, "len": 0.04, "gap": 0.005},
            {"th": 100, "len": 0.03, "gap": 0.008},
            {"th": 80,  "len": 0.02, "gap": 0.01},
            {"th": 60,  "len": 0.015, "gap": 0.015},
            {"th": 40,  "len": 0.01, "gap": 0.02},
        ]

    # ===================== MAIN =====================

    def detect(self, images: List[Dict[str, np.ndarray]]) -> List[List[Line]]:
        all_results = []

        for idx, item in enumerate(images):

            for mode in ["skeleton", "stabilized"]:
                img = item[mode]
                h, w = img.shape
                scale = np.hypot(h, w)

                if self.debug:
                    print(f"\n[Image {idx}] Mode: {mode}")

                for i, cfg in enumerate(self.config_ratios):

                    params = {
                        "threshold": cfg["th"],
                        "minLineLength": int(cfg["len"] * scale),
                        "maxLineGap": int(cfg["gap"] * scale),
                    }

                    lines = self._detect_single(img, params)

                    if self.debug:
                        print(f"  Config {i}: {len(lines)} lines")

                        self._show(
                            img,
                            lines,
                            title=f"[Img {idx}] {mode} | Config {i}"
                        )

                    all_results.append(lines)

        return all_results

    # ===================== SINGLE DETECTION =====================

    def _detect_single(self, img, cfg) -> List[Line]:

        lines = cv2.HoughLinesP(
            img,
            rho=1,
            theta=np.pi / 180,
            threshold=cfg["threshold"],
            minLineLength=cfg["minLineLength"],
            maxLineGap=cfg["maxLineGap"]
        )

        detected = []

        if lines is None:
            return detected

        for l in lines:
            x1, y1, x2, y2 = l[0]

            dx, dy = x2 - x1, y2 - y1
            angle = abs(np.degrees(np.arctan2(dy, dx)))

            if not (
                angle < 10 or
                abs(angle - 90) < 10 or
                abs(angle - 180) < 10
            ):
                continue

            detected.append(((x1, y1), (x2, y2)))

        return detected

    # ===================== HELPERS =====================

    # def _remove_small(self, lines: List[Line], shape) -> List[Line]:
    #     h, w = shape
    #     scale = np.hypot(h, w)

    #     min_len = 0.01 * scale

    #     res = []
    #     for (x1, y1), (x2, y2) in lines:
    #         length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    #         if length >= min_len:
    #             res.append(((x1, y1), (x2, y2)))

    #     return res

    # ===================== VISUALIZATION =====================

    def _resize_for_display(self, img, max_width=900, max_height=700):
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        return cv2.resize(img, (int(w * scale), int(h * scale)))

    def _show(self, img, lines, title=""):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        vis = self._resize_for_display(vis)

        cv2.imshow(title, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()