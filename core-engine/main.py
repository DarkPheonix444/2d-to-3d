import cv2
import os
from PIL import Image

from normalizer import Normalizer
from wall_detector import WallDetector
from line_normalizer import LineNormalizer
from merger import MergeSystem
from line_refiner import LineRefiner


class TempController:

    def __init__(self, debug=True):
        self.debug = debug

        self.normalizer = Normalizer(debug=debug)
        self.detector = WallDetector(debug=debug)
        self.line_norm = LineNormalizer(grid_ratio=0.02)

        self.merger = MergeSystem(debug=debug)
        self.refiner = LineRefiner(debug=debug)

    # ===================== LOAD =====================

    def load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        # Keep original size, no resize
        return img

    # ===================== VISUALIZER =====================

    @staticmethod
    def _resize_for_display(img, max_w=900, max_h=700):
        """Resize image for display only (NOT for processing)"""
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    def show_lines(self, img, lines, title):
        """Show lines on display-scaled image (coordinates stay original)"""
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw lines in original coordinates
        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Scale ONLY for display visibility
        vis_resized, _ = self._resize_for_display(vis)
        cv2.imshow(title, vis_resized)

    # ===================== PIPELINE =====================

    def run(self, image_path):

        print("\n===== RUNNING PIPELINE =====")

        # ---- LOAD ----
        img = self.load_image(image_path)
        h_pil, w_pil = img.size[1], img.size[0]
        print(f"[Controller] Original image size: {w_pil}x{h_pil}")

        # ---- NORMALIZE ----
        norm_out = self.normalizer.normalize([img])[0]

        # ---- WALL DETECTION ----
        detected_sets = self.detector.detect([norm_out])
        print(f"[Controller] Line sets: {len(detected_sets)} (expected 10)")

        # ---- GRID SNAP ----
        normalized_sets = [
            self.line_norm.normalize(lines)
            for lines in detected_sets
        ]

        # ===================== MERGE =====================

        print("\n--- MERGE ---")
        merged_lines = self.merger.merge(normalized_sets)
        print(f"[Controller] Lines after merge: {len(merged_lines)}")

        # ===================== REFINE =====================

        print("\n--- REFINE ---")
        refined_lines = self.refiner.refine(
            merged_lines,
            norm_out["stabilized"]   # 🔥 CRITICAL FIX
        )
        print(f"[Controller] Lines after refine: {len(refined_lines)}")

        # ===================== VISUAL COMPARISON =====================

        self.show_lines(norm_out["stabilized"], merged_lines, "MERGED")
        self.show_lines(norm_out["stabilized"], refined_lines, "REFINED")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return merged_lines, refined_lines


# ===================== RUN =====================

if __name__ == "__main__":

    controller = TempController(debug=True)

    image_path = "core-engine/images/test7.jpg"  # 🔥 change if needed

    merged, refined = controller.run(image_path)

    print("\n===== DONE =====")