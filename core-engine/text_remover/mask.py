import numpy as np
import cv2


class MaskApplier:
    def __init__(self, debug=False):
        self.debug = debug

    def apply(self, image, mask):
        """
        Input:
            image -> BGR image
            mask  -> binary (255 = remove)

        Output:
            cleaned + noise-free image
        """

        assert image.shape[:2] == mask.shape, "Mask mismatch"

        # -------------------------------
        # STEP 1: Strengthen mask
        # -------------------------------
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.medianBlur(mask, 5)

        # -------------------------------
        # STEP 2: Apply mask (white fill)
        # -------------------------------
        output = image.copy()
        output[mask == 255] = 255

        # -------------------------------
        # STEP 3: CLEAN AFTER MASK (CRITICAL)
        # -------------------------------
        output = self._post_process(output)

        if self.debug:
            self._debug(image, mask, output)

        return output

    # -------------------------------
    # POST CLEANING (THIS FIXES DOTS)
    # -------------------------------
    def _post_process(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # re-binarize cleanly
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # # remove small noise
        # kernel = np.ones((3, 3), np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # # smooth tiny artifacts
        # binary = cv2.medianBlur(binary, 3)

        # invert back
        binary = cv2.bitwise_not(binary)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # -------------------------------
    # DEBUG
    # -------------------------------
    def _debug(self, image, mask, output):
        overlay = image.copy()
        overlay[mask == 255] = [0, 0, 255]

        combined = self._stack_debug_views(image, mask, overlay, output)
        self._show_debug("Mask Debug | Original | Mask | Overlay | Clean", combined)

    def _stack_debug_views(self, image, mask, overlay, output):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        top = np.hstack((image, mask_color))
        bottom = np.hstack((overlay, output))

        return np.vstack((top, bottom))

    def _show_debug(self, title, img):
        try:
            cv2.imshow(title, self._resize_for_display(img))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            try:
                from PIL import Image

                display_img = img
                if len(display_img.shape) == 3:
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

                Image.fromarray(display_img).show(title=title)
            except Exception:
                print(f"[preview skipped] {title}")

    def _resize_for_display(self, img, max_width=1200, max_height=800):
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            return cv2.resize(img, (int(w * scale), int(h * scale)))
        return img