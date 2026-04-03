import cv2
import numpy as np
from PIL import Image
from typing import List, Dict
from skimage.morphology import skeletonize


class Normalizer:

    def __init__(self, debug=True):
        self.debug = debug

    def normalize(self, images: List[Image.Image]) -> List[Dict[str, np.ndarray]]:
        results = []

        for idx, img in enumerate(images):

            # ===================== LOAD =====================
            img_np = np.asarray(img.convert("RGB"), dtype=np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # ===================== STEP 1: THRESHOLD =====================
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                15,
                3
            )

            # ===================== STEP 2: CONNECTIVITY (CONTROLLED) =====================
            kernel = np.ones((3, 3), np.uint8)

            # light closing (connect small gaps ONLY)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            # light smoothing
            closed = cv2.medianBlur(closed, 3)

            # ===================== STEP 3: STABILIZED =====================
            # 🔥 NO DILATION — prevents thick walls
            stabilized = closed.copy()

            # ===================== STEP 4: SKELETON =====================
            skeleton = skeletonize(stabilized // 255)
            skeleton = (skeleton * 255).astype(np.uint8)

            # very light cleanup
            skeleton = self._light_cleanup(skeleton)

            results.append({
                "stabilized": stabilized,
                "skeleton": skeleton
            })

            if self.debug:
                self._show(gray, binary, closed, stabilized, skeleton, idx)

        return results

    # ===================== LIGHT CLEANUP =====================

    def _light_cleanup(self, skel):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)

        clean = np.zeros_like(skel)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 10:
                clean[labels == i] = 255

        return clean

    # ===================== DISPLAY =====================

    def _resize_for_display(self, img, max_width=900, max_height=700):
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        return cv2.resize(img, (int(w * scale), int(h * scale)))

    def _show(self, gray, binary, closed, stabilized, skeleton, idx):
        cv2.imshow(f"[{idx}] Gray", self._resize_for_display(gray))
        cv2.imshow(f"[{idx}] Binary", self._resize_for_display(binary))
        cv2.imshow(f"[{idx}] Closed", self._resize_for_display(closed))
        cv2.imshow(f"[{idx}] Stabilized", self._resize_for_display(stabilized))
        cv2.imshow(f"[{idx}] Skeleton", self._resize_for_display(skeleton))
        cv2.waitKey(0)
        cv2.destroyAllWindows()