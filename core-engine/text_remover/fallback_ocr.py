import cv2
import numpy as np

from text_remover.mask import MaskApplier
from text_remover.text_rule import CCTextMask


def remove_text_with_fallback(image, debug=False, fallback_ratio=0.01):
    """Remove text from an image using CC detection and a relaxed fallback pass.

    Returns:
        clean_image: the image that should be passed to normalizer
        final_mask: combined text mask
    """

    primary = CCTextMask(debug=False)
    cc_mask, _ = primary.generate_mask(image)

    text_pixels = np.sum(cc_mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    ratio = text_pixels / max(total_pixels, 1)

    if debug:
        print(f"[CC] ratio: {ratio:.4f}")

    final_mask = cc_mask

    if ratio < fallback_ratio:
        if debug:
            print("[Fallback] relaxed text pass triggered")

        fallback_detector = CCTextMask(min_area=10, max_area=15000, debug=False)
        fallback_mask, _ = fallback_detector.generate_mask(image)
        final_mask = cv2.bitwise_or(cc_mask, fallback_mask)

    applier = MaskApplier(debug=debug)
    clean_image = applier.apply(image, final_mask)

    return clean_image, final_mask