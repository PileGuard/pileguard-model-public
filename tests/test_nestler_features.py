import unittest

from pileguard.features.nestler import (
    BoundingBox,
    compute_spatial_features,
    compute_tracking_features,
    parse_boxes,
    parse_frame_boxes,
)


class NESTLERFeatureTest(unittest.TestCase):
    def test_parses_and_clips_valid_boxes(self) -> None:
        rows = [
            [-10, -5, 40, 20, 3, 3],
            [20, 20, 20, 30, 4, 4],
            [1, 2, 3],
        ]

        boxes = parse_boxes(rows, frame_width=100, frame_height=50)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], BoundingBox(0.0, 0.0, 40.0, 20.0, 3))

    def test_computes_spatial_density_features(self) -> None:
        boxes = [
            BoundingBox(0, 0, 20, 20, 1),
            BoundingBox(20, 0, 40, 20, 2),
        ]

        features = compute_spatial_features(
            boxes,
            frame_width=100,
            frame_height=100,
            grid_rows=2,
            grid_columns=2,
        )

        self.assertEqual(features["object_count"], 2)
        self.assertAlmostEqual(features["bbox_area_ratio"], 0.08)
        self.assertAlmostEqual(features["center_x"], 0.2)
        self.assertAlmostEqual(features["center_y"], 0.1)
        self.assertEqual(features["max_grid_fraction"], 1.0)
        self.assertAlmostEqual(features["mean_nearest_neighbor_distance"], 0.2)

    def test_distinguishes_missing_bbox_annotation_from_empty_list(self) -> None:
        missing, missing_boxes = parse_frame_boxes(
            {"frame_index": 10}, frame_width=100, frame_height=50
        )
        available, empty_boxes = parse_frame_boxes(
            {"frame_index": 11, "tracks_bbox": []}, frame_width=100, frame_height=50
        )

        self.assertFalse(missing)
        self.assertEqual(missing_boxes, [])
        self.assertTrue(available)
        self.assertEqual(empty_boxes, [])

    def test_computes_normalized_tracking_motion(self) -> None:
        previous = {
            1: BoundingBox(0, 0, 20, 20, 1),
            2: BoundingBox(20, 0, 40, 20, 2),
        }
        current = [
            BoundingBox(10, 0, 30, 20, 1),
            BoundingBox(30, 0, 50, 20, 2),
        ]

        features = compute_tracking_features(
            previous,
            current,
            frame_width=100,
            frame_height=100,
            fps=10,
        )

        self.assertEqual(features["track_match_count"], 2)
        self.assertAlmostEqual(features["tracked_speed_per_second"], 1.0)
        self.assertAlmostEqual(features["tracked_direction_x"], 1.0)
        self.assertAlmostEqual(features["tracked_direction_y"], 0.0)
        self.assertAlmostEqual(features["tracked_coherence"], 1.0)


if __name__ == "__main__":
    unittest.main()
