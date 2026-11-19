"""OpenFOAM file reading and writing (CPU only)."""
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import numpy as np
from simplefoam_gpu import foamio


class FoamIO(unittest.TestCase):
    def test_write_read_roundtrip(self):
        rng = np.random.default_rng(2)
        U = rng.standard_normal((7, 3))
        wall = rng.standard_normal((2, 3))
        with tempfile.TemporaryDirectory() as d:
            foamio.write_field(d, "100", "U", "volVectorField", U,
                               [("inlet", "type fixedValue;", wall), ("outlet", "type zeroGradient;", None)])
            f = foamio.read_field(Path(d) / "100/U")
        self.assertEqual(f["cls"], "volVectorField")
        np.testing.assert_array_equal(foamio.expand(f["internal"], 7, 3), U)
        np.testing.assert_array_equal(foamio.expand(f["boundary"]["inlet"]["value"], 2, 3), wall)
        self.assertEqual(f["boundary"]["outlet"]["type"], ["zeroGradient"])

    def test_ascii_dict_and_uniform(self):
        txt = b"""FoamFile { version 2.0; format ascii; class volScalarField; object p; }
        dimensions [0 2 -2 0 0 0 0];
        internalField uniform 3.5;   // comment
        boundaryField { "(in|out)let" { type fixedValue; value uniform 1; } wall { type zeroGradient; } }
        """
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "p").write_bytes(txt)
            f = foamio.read_field(Path(d) / "p")
        np.testing.assert_array_equal(foamio.expand(f["internal"], 4, 1), np.full(4, 3.5))
        self.assertEqual(foamio.patch_entry(f["boundary"], "outlet")["type"], ["fixedValue"])  # regex patch key

    def _case(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, d)
        (d / "system").mkdir()
        (d / "0/include").mkdir(parents=True)
        return d

    def test_macros_and_includes(self):
        """The motorBike pattern: #include'd initial conditions, $macros, #includeEtc setConstraintTypes."""
        d = self._case()
        (d / "0/include/initialConditions").write_text("flowVelocity (20 0 0);\npressure 0;\n")
        (d / "0/include/fixedInlet").write_text("inlet { type fixedValue; value $internalField; }\n")
        (d / "0/U").write_text("""FoamFile { version 2.0; format ascii; class volVectorField; object U; }
#include "include/initialConditions"
dimensions [0 1 -1 0 0 0 0];
internalField uniform $flowVelocity;
boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"
    #include "include/fixedInlet"
    outlet { type inletOutlet; inletValue uniform (0 0 0); value $internalField; }
    "motorBike.*" { type noSlip; }
}
""")
        with mock.patch.dict(os.environ, {"FOAM_ETC": "", "WM_PROJECT_DIR": ""}):
            f = foamio.read_field(d / "0/U")
        np.testing.assert_array_equal(foamio.expand(f["internal"], 2, 3), [[20, 0, 0]] * 2)
        b = f["boundary"]
        np.testing.assert_array_equal(foamio.expand(foamio.patch_entry(b, "inlet")["value"], 1, 3), [[20, 0, 0]])
        np.testing.assert_array_equal(foamio.expand(b["outlet"]["value"], 1, 3), [[20, 0, 0]])
        self.assertEqual(foamio.patch_entry(b, "motorBike_frt-fairing")["type"], ["noSlip"])
        # a symmetry patch has no entry of its own: it takes the one for its patch group
        self.assertEqual(foamio.patch_entry(b, "frontAndBack", ["symmetryPlane"])["type"], ["symmetryPlane"])
        with self.assertRaises(KeyError):
            foamio.patch_entry(b, "frontAndBack")

    def test_dictionary_paste_and_scoped_macro(self):
        d = self._case()
        (d / "system/fvSolution").write_text("""FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p   { solver GAMG; tolerance 1e-6; relTol 0.05; }
    Phi { $p; relTol 0; }
}
tol $:solvers.p.tolerance;
""")
        fs = foamio.read_dict(d / "system/fvSolution")
        self.assertEqual(fs["solvers"]["Phi"], {"solver": ["GAMG"], "tolerance": ["1e-6"], "relTol": ["0"]})
        self.assertEqual(fs["solvers"]["p"]["relTol"], ["0.05"])     # the paste is a copy
        self.assertEqual(fs["tol"], ["1e-6"])

    def test_unreadable_syntax_is_refused(self):
        d = self._case()
        head = "FoamFile { version 2.0; format ascii; class volScalarField; object p; }\n"
        for body, msg in (("internalField uniform $undefined;\n", r"\$undefined is not defined"),
                          ('#include "include/nothere"\n', "not found"),
                          ('internalField uniform #calc "1+2";\n', "#calc cannot be evaluated"),
                          ('#includeEtc "caseDicts/other"\n', "etc directory")):
            (d / "0/p").write_text(head + body)
            with mock.patch.dict(os.environ, {"FOAM_ETC": "", "WM_PROJECT_DIR": ""}):
                with self.assertRaisesRegex(foamio.FoamSyntaxError, msg):
                    foamio.read_field(d / "0/p")

    def test_boundary_patch_groups(self):
        d = self._case()
        (d / "boundary").write_text("""FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
3
(
    lowerWall    { type wall; inGroups List<word> 1(wall); nFaces 2; startFace 0; }
    frontAndBack { type symmetryPlane; inGroups 1(symmetryPlane); nFaces 1; startFace 2; }
    car          { type wall; inGroups 1(carGroup); nFaces 1; startFace 3; }
)
""")
        groups = [p["groups"] for p in foamio.read_boundary(d / "boundary")]
        self.assertEqual(groups, [["wall"], ["symmetryPlane"], ["carGroup", "wall"]])


if __name__ == "__main__":
    unittest.main()
