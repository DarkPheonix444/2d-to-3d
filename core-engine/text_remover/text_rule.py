import cv2
import numpy as np


class CCTextMask:
    def __init__(self, min_area=30, max_area=5000, debug=False):
        self.min_area = min_area
        self.max_area = max_area
        self.debug = debug

    def generate_mask(self, image):
        """
        Input: BGR or grayscale image
        Output:
            mask   -> 255 = text
            binary -> binarized image
        """

        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Binarize
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 3. DILATION (merge letters → words)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        # 4. Connected Components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # 5. Mask
        mask = np.zeros_like(binary)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            if self.min_area < area < self.max_area:
                aspect_ratio = max(w, h) / (min(w, h) + 1)

                # Text tends to not be ultra-long straight lines
                if aspect_ratio < 15:
                    mask[labels == i] = 255

        if self.debug:
            debug_visualization(image, binary, mask)

        return mask, binary


# -------------------------------
# APPLY MASK
# -------------------------------
def apply_mask(image, mask):
    output = image.copy()
    output[mask == 255] = 255
    return output


# -------------------------------
# DEBUG
# -------------------------------
def debug_visualization(image, binary, mask):

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    overlay = image.copy()
    overlay[mask == 255] = [0, 0, 255]

    top = np.hstack((image, binary_color))
    bottom = np.hstack((mask_color, overlay))
    combined = np.vstack((top, bottom))

    combined = resize_for_display(combined)

    cv2.imshow("DEBUG | Original | Binary | Mask | Overlay", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_for_display(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# -------------------------------
# TEST
# -------------------------------
if __name__ == "__main__":
    image = cv2.imread("test.png")

    cc = CCTextMask(min_area=30, max_area=5000, debug=True)

    mask, binary = cc.generate_mask(image)

    cleaned = apply_mask(image, mask)

    cv2.imwrite("debug_clean.png", cleaned)