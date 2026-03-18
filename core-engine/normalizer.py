import cv2
import numpy as np
from PIL import Image
from typing import List


class Normalizer:

    def __init__(self,target_width:int=1500):
        self.target_width = target_width

    def normalize(self,images:List[Image.Image])->List[np.ndarray]:
        if not isinstance(images, list):
            raise TypeError("images must be a list of PIL.Image objects")

        processed_images: List[np.ndarray] = []

        for idx, img in enumerate(images):
            if not isinstance(img, Image.Image):
                raise TypeError(f"images[{idx}] must be a PIL.Image.Image")

            if img.width == 0 or img.height == 0:
                raise ValueError(f"images[{idx}] has invalid size: {img.size}")

            try:
                img_rgb = img.convert("RGB")
            except Exception as exc:
                raise ValueError(f"images[{idx}] cannot be converted to RGB") from exc

            img_np = np.asarray(img_rgb, dtype=np.uint8)
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            if min(gray.shape[:2]) >= 3 and self._should_apply_blur(gray):
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

            processed_images.append(gray)

        return processed_images

    def _should_apply_blur(self, gray: np.ndarray) -> bool:
        # Estimate high-frequency speckle by comparing with median-smoothed image.
        median = cv2.medianBlur(gray, 3)
        noise_residual = cv2.absdiff(gray, median)
        noise_score = float(np.mean(noise_residual))
        return noise_score > 4.0