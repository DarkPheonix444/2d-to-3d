import cv2
import numpy as np

class WallAppender:

    def apply(
        self,
        walls,
        consumed_ids,
        new_walls
    ):

        consumed_ids = set(consumed_ids)

        updated = []

        for wall in walls:
            if wall["id"] not in consumed_ids:
                updated.append(wall)

        updated.extend(new_walls)

        return updated

    def apply_batch(
        self,
        walls,
        operations
    ):

        all_consumed = set()
        all_new = []

        for op in operations:

            all_consumed.update(
                op["consumed_ids"]
            )

            all_new.extend(
                op["new_walls"]
            )
       

        return self.apply(
            walls,
            all_consumed,
            all_new
        )


    def visualize_appender(
    self,
    image,
    original_walls,
    updated_walls
    ):

        vis = image.copy()

        original_ids = {
            w["id"] for w in original_walls
        }

        updated_ids = {
            w["id"] for w in updated_walls
        }

        removed_ids = (
            original_ids - updated_ids
        )

        new_ids = (
            updated_ids - original_ids
        )

        # --------------------------------
        # REMOVED WALLS
        # --------------------------------
        for wall in original_walls:

            if wall["id"] not in removed_ids:
                continue

            (x1, y1), (x2, y2) = wall["line"]

            cv2.line(
                vis,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),      # RED
                3
            )

        # --------------------------------
        # UNTOUCHED WALLS
        # --------------------------------
        for wall in updated_walls:

            if wall["id"] in new_ids:
                continue

            (x1, y1), (x2, y2) = wall["line"]

            cv2.line(
                vis,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),      # GREEN
                2
            )

        # --------------------------------
        # NEW WALLS
        # --------------------------------
        for wall in updated_walls:

            if wall["id"] not in new_ids:
                continue

            (x1, y1), (x2, y2) = wall["line"]

            cv2.line(
                vis,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 0),      # BLUE
                4
            )

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.putText(
                vis,
                f"N{wall['id']}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

        return vis

                