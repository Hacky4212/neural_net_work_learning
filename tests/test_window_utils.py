import unittest

from game_ai.window.window_utils import resolve_point_in_bounds


class WindowUtilsTests(unittest.TestCase):
    def test_resolve_point_scales_reference_coordinates_to_screen_bounds(self) -> None:
        bounds = (100, 50, 2020, 1130)

        self.assertEqual(
            resolve_point_in_bounds(640, 360, bounds, 1280, 720, scale=True),
            (1060, 590),
        )

    def test_resolve_point_clamps_to_bounds(self) -> None:
        bounds = (100, 50, 2020, 1130)

        self.assertEqual(
            resolve_point_in_bounds(2000, 2000, bounds, 1280, 720, scale=False),
            (2019, 1129),
        )


if __name__ == "__main__":
    unittest.main()
