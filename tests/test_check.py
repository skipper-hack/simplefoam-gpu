"""The case check accepts the examples and refuses settings the solver does not implement (CPU only)."""
import shutil
import tempfile
import unittest
from pathlib import Path
from simplefoam_gpu.check import check_case, UnsupportedCase

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
BOUNDARY = """FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
2 ( inlet { type patch; nFaces 1; startFace 0; } sphere { type wall; nFaces 1; startFace 1; } )
"""


class Check(unittest.TestCase):
    def case(self, name):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        shutil.copytree(EXAMPLES / name, d / name, ignore=shutil.ignore_patterns("polyMesh", "[0-9]*", "log.*"))
        (d / name / "constant/polyMesh").mkdir()
        (d / name / "constant/polyMesh/boundary").write_text(BOUNDARY)
        return d / name

    def test_examples_accepted(self):
        for name in ("sphere-laminar", "sphere-kOmegaSST"):
            check_case(self.case(name))

    def test_other_scheme_refused(self):
        c = self.case("sphere-laminar")
        f = c / "system/fvSchemes"
        f.write_text(f.read_text().replace("linearUpwindV grad(U)", "upwind"))
        with self.assertRaisesRegex(UnsupportedCase, "div\\(phi,U\\)"):
            check_case(c)

    def test_other_model_refused(self):
        c = self.case("sphere-kOmegaSST")
        f = c / "constant/turbulenceProperties"
        f.write_text(f.read_text().replace("kOmegaSST", "kEpsilon"))
        with self.assertRaisesRegex(UnsupportedCase, "kEpsilon"):
            check_case(c)

    def test_mrf_refused(self):
        c = self.case("sphere-laminar")
        (c / "constant/MRFProperties").write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }\n"
            "rotor { cellZone rotor; active yes; origin (0 0 0); axis (0 0 1); omega 100; }\n")
        with self.assertRaisesRegex(UnsupportedCase, "MRF"):
            check_case(c)


if __name__ == "__main__":
    unittest.main()
