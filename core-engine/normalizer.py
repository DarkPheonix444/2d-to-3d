import cv2
import numpy as np
from PIL import Image
from typing import List


class Normalizer:

    def __init__(self, debug=True):
        self.debug = debug

    def normalize(self, images: List[Image.Image]) -> List[np.ndarray]:
        results = []

        for idx, img in enumerate(images):
            img_np = np.asarray(img.convert("RGB"), dtype=np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            edges = cv2.Canny(blurred, 50, 150)

            results.append(edges)

            if self.debug:
                self._show(gray, edges, idx)

        return results

    def _show(self, gray, edges, idx):
        cv2.imshow(f"[{idx}] Gray", gray)
        cv2.imshow(f"[{idx}] Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()