import cv2
import numpy as np
from PIL import Image
from typing import List


class Normalizer:

    def __init__(self, debug: bool = True):
        self.debug = debug

    def normalize(self, images: List[Image.Image]) -> List[np.ndarray]:
        processed = []

        for idx, img in enumerate(images):

            if not isinstance(img, Image.Image):
                raise TypeError(f"images[{idx}] must be PIL.Image")

            # STEP 1: GRAYSCALE
            img_np = np.asarray(img.convert("RGB"), dtype=np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # STEP 2: LIGHT BLUR (only to reduce noise)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # STEP 3: EDGE DETECTION (IMPORTANT SHIFT)
            edges = cv2.Canny(blurred, 50, 150)

            processed.append(edges)

            if self.debug:
                self._show_debug(idx, gray, blurred, edges)

        return processed

    def _show_debug(self, idx, gray, blurred, edges):
        def resize(img):
            h, w = img.shape[:2]
            scale = 600 / max(h, w)
            return cv2.resize(img, (int(w * scale), int(h * scale)))

        cv2.imshow(f"[{idx}] Gray", resize(gray))
        cv2.imshow(f"[{idx}] Blurred", resize(blurred))
        cv2.imshow(f"[{idx}] Edges", resize(edges))

        cv2.waitKey(0)
        cv2.destroyAllWindows()