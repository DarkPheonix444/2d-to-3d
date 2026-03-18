import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


Line = Tuple[Tuple[int, int], Tuple[int, int]]


class WallDetector:

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 100,
        min_line_length: int = 50,
        max_line_gap: int = 10,
        orientation_tol: int = 15,
        min_wall_length: int = 40,
        use_adaptive_canny: bool = True
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.orientation_tol = orientation_tol
        self.min_wall_length = min_wall_length
        self.use_adaptive_canny = use_adaptive_canny


    def detect(self, images: List[np.ndarray]) -> List[List[Line]]:

        if not isinstance(images, list):
            raise ValueError("Input must be a list of images.")

        results: List[List[Line]] = []

        for img in images:

            self._validate_image(img)

            gray = self._to_gray(img)

            edges = self._detect_edges(gray)

            lines = self._detect_lines(edges)

            walls = self._filter_walls(lines)

            merged_walls = self._merge_walls(walls)

            results.append(merged_walls)

        return results


    def _validate_image(self, image: np.ndarray) -> None:

        if image is None:
            raise ValueError("Input image is None")

        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be numpy.ndarray")

        if image.size == 0:
            raise ValueError("Empty image provided")

        if image.ndim not in (2, 3):
            raise ValueError("Image must be 2D grayscale or 3D color array")


    def _to_gray(self, image: np.ndarray) -> np.ndarray:

        if image.ndim == 2:
            return image

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    def _detect_edges(self, image: np.ndarray) -> np.ndarray:

        if not self.use_adaptive_canny:
            return cv2.Canny(image, self.canny_low, self.canny_high)

        # Adaptive thresholds preserve weak walls in low-contrast plans
        median_intensity = float(np.median(image))
        sigma = 0.33
        low = int(max(0, (1.0 - sigma) * median_intensity))
        high = int(min(255, (1.0 + sigma) * median_intensity))

        if low >= high:
            low, high = self.canny_low, self.canny_high

        return cv2.Canny(image, low, high)


    def _detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:

        height, width = edges.shape[:2]
        scale = max(height, width)

        min_line_length = max(self.min_line_length, int(scale * 0.05))
        max_line_gap = max(self.max_line_gap, int(scale * 0.01))
        hough_threshold = max(40, min(self.hough_threshold, int(scale * 0.08)))

        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )


    def _filter_walls(self, lines: Optional[np.ndarray]) -> List[Line]:
        """
        Filter lines to keep only axis-aligned walls.
        
        Steps:
        1. Filter by angle - keep only horizontal and vertical
        2. Snap to perfect axis alignment
        3. Remove small segments
        """
        walls: List[Line] = []

        if lines is None:
            return walls

        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Compute line angle using atan2
            dy = y2 - y1
            dx = x2 - x1
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            
            # Normalize angle to [0, 180)
            if angle_deg < 0:
                angle_deg += 180
            
            # Check if horizontal (near 0° or 180°)
            is_horizontal = (
                abs(angle_deg) <= self.orientation_tol
                or abs(angle_deg - 180) <= self.orientation_tol
            )
            
            # Check if vertical (near 90°)
            is_vertical = abs(angle_deg - 90) <= self.orientation_tol
            
            if not (is_horizontal or is_vertical):
                continue  # Discard diagonal/curved lines
            
            # Compute line length
            length = np.hypot(dx, dy)
            
            if length < self.min_wall_length:
                continue
            
            # Snap to perfect axis alignment
            if is_horizontal:
                # Force y1 = y2 = average y
                y_avg = int((y1 + y2) // 2)
                x_start, x_end = sorted((x1, x2))
                walls.append(((x_start, y_avg), (x_end, y_avg)))
            elif is_vertical:
                # Force x1 = x2 = average x
                x_avg = int((x1 + x2) // 2)
                y_start, y_end = sorted((y1, y2))
                walls.append(((x_avg, y_start), (x_avg, y_end)))
        
        return walls


    def _merge_walls(self, walls: List[Line]) -> List[Line]:
        """
        Implement advanced clustering and merging.
        
        Steps:
        1. Separate lines into horizontal and vertical
        2. Cluster by position (±10px tolerance)
        3. Merge overlapping segments within each cluster
        4. Deduplicate results
        """
        if not walls:
            return []
        
        # Step A: Separate into horizontal and vertical
        horizontal_lines = []
        vertical_lines = []
        
        for (x1, y1), (x2, y2) in walls:
            if y1 == y2:  # Horizontal
                horizontal_lines.append(((min(x1, x2), y1), (max(x1, x2), y1)))
            else:  # Vertical
                vertical_lines.append(((x1, min(y1, y2)), (x1, max(y1, y2))))
        
        # Step B & C: Cluster and merge
        merged_horizontal = self._cluster_and_merge_lines(horizontal_lines, is_horizontal=True)
        merged_vertical = self._cluster_and_merge_lines(vertical_lines, is_horizontal=False)
        
        # Combine results
        merged = merged_horizontal + merged_vertical
        
        # Step 5: Deduplicate using set (tuple normalization)
        unique_lines = set()
        for (x1, y1), (x2, y2) in merged:
            normalized = ((x1, y1), (x2, y2))
            unique_lines.add(normalized)
        
        return list(unique_lines)
    
    def _cluster_and_merge_lines(self, lines: List[Line], is_horizontal: bool) -> List[Line]:
        """
        Cluster lines by position and merge overlapping segments.
        
        For horizontal: cluster by y-coordinate (±10px)
        For vertical: cluster by x-coordinate (±10px)
        """
        if not lines:
            return []
        
        # Cluster lines by position
        clusters = self._create_clusters(lines, is_horizontal)
        
        # Merge lines within each cluster
        merged = []
        for cluster in clusters.values():
            merged_cluster = self._merge_cluster(cluster, is_horizontal)
            merged.extend(merged_cluster)
        
        return merged
    
    def _create_clusters(self, lines: List[Line], is_horizontal: bool) -> Dict[int, List[Line]]:
        """
        Group lines by position with ±10px tolerance.
        """
        clusters: Dict[int, List[Line]] = defaultdict(list)
        
        for line in lines:
            (x1, y1), (x2, y2) = line
            
            # Get cluster key based on line orientation
            if is_horizontal:
                # Cluster by y-coordinate
                cluster_key = y1  # y1 == y2 for horizontal lines
            else:
                # Cluster by x-coordinate
                cluster_key = x1  # x1 == x2 for vertical lines
            
            # Find matching cluster (within ±10px)
            found = False
            best_key = None
            best_dist = None
            for existing_key in list(clusters.keys()):
                if abs(existing_key - cluster_key) <= 10:
                    dist = abs(existing_key - cluster_key)
                    if best_dist is None or dist < best_dist:
                        best_key = existing_key
                        best_dist = dist

            if best_key is not None:
                clusters[best_key].append(line)
                found = True
            
            if not found:
                clusters[cluster_key].append(line)
        
        return clusters
    
    def _merge_cluster(self, cluster: List[Line], is_horizontal: bool) -> List[Line]:
        """
        Merge overlapping or nearby segments within a cluster.
        Gap tolerance: 20px
        """
        if not cluster:
            return []
        
        if is_horizontal:
            # Sort by x-coordinate
            sorted_lines = sorted(cluster, key=lambda line: line[0][0])
            # Use y from first line (all same y in cluster)
            y_val = sorted_lines[0][0][1]
            
            merged = []
            current_start, current_end = sorted_lines[0][0][0], sorted_lines[0][1][0]
            
            for i in range(1, len(sorted_lines)):
                next_start, next_end = sorted_lines[i][0][0], sorted_lines[i][1][0]
                
                # Check if gap is <= 20px
                gap = next_start - current_end
                if gap <= max(20, self.max_line_gap * 2):
                    # Merge: extend current segment
                    current_end = max(current_end, next_end)
                else:
                    # Gap too large: save current segment and start new one
                    merged.append(((current_start, y_val), (current_end, y_val)))
                    current_start = next_start
                    current_end = next_end
            
            # Add last segment
            merged.append(((current_start, y_val), (current_end, y_val)))
            return merged
        
        else:  # Vertical
            # Sort by y-coordinate
            sorted_lines = sorted(cluster, key=lambda line: line[0][1])
            # Use x from first line (all same x in cluster)
            x_val = sorted_lines[0][0][0]
            
            merged = []
            current_start, current_end = sorted_lines[0][0][1], sorted_lines[0][1][1]
            
            for i in range(1, len(sorted_lines)):
                next_start, next_end = sorted_lines[i][0][1], sorted_lines[i][1][1]
                
                # Check if gap is <= 20px
                gap = next_start - current_end
                if gap <= max(20, self.max_line_gap * 2):
                    # Merge: extend current segment
                    current_end = max(current_end, next_end)
                else:
                    # Gap too large: save current segment and start new one
                    merged.append(((x_val, current_start), (x_val, current_end)))
                    current_start = next_start
                    current_end = next_end
            
            # Add last segment
            merged.append(((x_val, current_start), (x_val, current_end)))
            return merged