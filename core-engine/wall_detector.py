import cv2
import numpy as np
from typing import List, Tuple

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class WallDetector:

    def __init__(self, debug=True):
        self.debug = debug

    # ===================== MAIN =====================

    def detect(self, images: List[np.ndarray]) -> List[List[Line]]:
        results = []

        for idx, img in enumerate(images):

            skel = self._skeletonize(img)

            lines = self._detect_hough(skel)

            results.append(lines)

            if self.debug:
                print(f"[IMAGE {idx+1}] RAW DETECTED: {len(lines)}")
                self._show(img, lines, f"RAW_{idx+1}")

        return results

    # ===================== CORE =====================

    def _skeletonize(self, img):
        from skimage.morphology import skeletonize
        return (skeletonize(img > 0) * 255).astype(np.uint8)

    def _detect_hough(self, img):

        # 🔥 tuned for HIGH RECALL (not precision)
        lines = cv2.HoughLinesP(
            img,
            rho=1,
            theta=np.pi / 180,
            threshold=30,        # lower = more lines
            minLineLength=10,    # small allowed
            maxLineGap=15        # allow broken walls
        )

        results = []

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]

                # keep everything except zero-length
                if (x1, y1) != (x2, y2):
                    results.append(((x1, y1), (x2, y2)))

        return results

    # ===================== DEBUG =====================

    def _show(self, img, lines, title):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # 🔥 Resize logic (maintain aspect ratio)
        max_size = 900  # max width/height

        h, w = vis.shape[:2]
        scale = min(max_size / w, max_size / h)

        if scale < 1:  # only shrink, never enlarge
            new_w = int(w * scale)
            new_h = int(h * scale)
            vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow(title, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()