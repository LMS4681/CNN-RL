import unittest
from pathlib import Path


class RequirementsTests(unittest.TestCase):
    def test_onnx_stays_below_ml_dtypes_dependency_boundary(self):
        requirements = Path(__file__).with_name("requirements.txt").read_text(
            encoding="utf-8"
        )
        onnx_requirement = next(
            line.strip()
            for line in requirements.splitlines()
            if line.strip().lower().startswith("onnx>")
        )

        self.assertIn("<1.18.0", onnx_requirement)


if __name__ == "__main__":
    unittest.main()
