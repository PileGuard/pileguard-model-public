import tempfile
import unittest
from pathlib import Path

from pileguard.data_inventory import IMAGE_SUFFIXES, InventoryCheck, count_files, format_report


class DataInventoryTest(unittest.TestCase):
    def test_count_files_filters_non_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.jpg").write_bytes(b"")
            (root / "b.PNG").write_bytes(b"")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual(count_files(root, IMAGE_SUFFIXES), 2)

    def test_format_report_marks_failed_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checks = [InventoryCheck("sample", str(root), expected=2, actual=1)]

            report = format_report(root, checks)

            self.assertIn("[FAIL] sample: 1/2", report)
            self.assertTrue(report.endswith("Inventory: FAIL"))


if __name__ == "__main__":
    unittest.main()
