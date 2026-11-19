"""The case check accepts the examples and refuses settings the solver does not implement (CPU only)."""
import io
import os
import shutil
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from simplefoam_gpu import check
from simplefoam_gpu.check import check_case, function_objects, UnsupportedCase

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
BOUNDARY = """FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
2 ( inlet { type patch; nFaces 1; startFace 0; } sphere { type wall; nFaces 1; startFace 1; } )
"""
BOUNDARY_FULL = """FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
4 ( inlet { type patch; nFaces 1; startFace 0; } outlet { type patch; nFaces 1; startFace 1; }
    farfield { type patch; nFaces 1; startFace 2; } sphere { type wall; nFaces 1; startFace 3; } )
"""
NO_ETC = {"FOAM_ETC": "", "WM_PROJECT_DIR": ""}


class Check(unittest.TestCase):
    def case(self, name):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        shutil.copytree(EXAMPLES / name, d / name, ignore=shutil.ignore_patterns("polyMesh", "[0-9]*", "log.*"))
        (d / name / "constant/polyMesh").mkdir()
        (d / name / "constant/polyMesh/boundary").write_text(BOUNDARY)
        return d / name

    def case_with_fields(self, name):
        """The example with its 0.orig fields as 0/ and a boundary file naming all of its patches."""
        c = self.case(name)
        shutil.copytree(EXAMPLES / name / "0.orig", c / "0")
        (c / "constant/polyMesh/boundary").write_text(BOUNDARY_FULL)
        return c

    def edit(self, path, old, new):
        txt = path.read_text()
        self.assertIn(old, txt)
        path.write_text(txt.replace(old, new))

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

    def test_example_fields_accepted(self):
        for name in ("sphere-laminar", "sphere-kOmegaSST"):
            check_case(self.case_with_fields(name), "0")

    def test_unsupported_boundary_conditions_listed(self):
        """Unsupported field boundary types are reported together, before anything is set up."""
        c = self.case_with_fields("sphere-kOmegaSST")
        self.edit(c / "0/nut", "sphere   { type nutUSpaldingWallFunction;", "sphere   { type nutkWallFunction;")
        self.edit(c / "0/p", "outlet   { type fixedValue; value uniform 0; }",
                  "outlet   { type fixedFluxPressure; value uniform 0; }")
        self.edit(c / "0/U", "inlet    { type fixedValue; value uniform (0 0 -20); }", "inlet    { type fixedValue; }")
        with self.assertRaises(UnsupportedCase) as e:
            check_case(c, "0")
        msg = str(e.exception)
        self.assertIn("0/nut uses boundary type nutkWallFunction on patch sphere", msg)
        self.assertIn("0/p uses boundary type fixedFluxPressure on patch outlet", msg)
        self.assertIn("0/U patch inlet (fixedValue) has no value", msg)

    def test_missing_entries_and_files_listed(self):
        c = self.case_with_fields("sphere-kOmegaSST")
        self.edit(c / "0/p", "farfield { type zeroGradient; }", "")
        (c / "0/omega").unlink()
        with self.assertRaises(UnsupportedCase) as e:
            check_case(c, "0")
        self.assertIn("0/p has no boundaryField entry for patch farfield", str(e.exception))
        self.assertIn("0/omega is missing", str(e.exception))

    def test_motorbike_style_fields(self):
        """#include, $macros and #includeEtc setConstraintTypes are followed; the result is checked as usual."""
        c = self.case_with_fields("sphere-laminar")
        (c / "constant/polyMesh/boundary").write_text(BOUNDARY_FULL.replace(
            "farfield { type patch;", "farfield { type symmetryPlane; inGroups 1(symmetryPlane);"))
        (c / "0/include").mkdir()
        (c / "0/include/initialConditions").write_text("flowVelocity (0 0 -20);\npressure 0;\n")
        (c / "0/include/fixedInlet").write_text("inlet { type fixedValue; value $internalField; }\n")
        (c / "0/U").write_text("""FoamFile { version 2.0; format ascii; class volVectorField; object U; }
#include "include/initialConditions"
dimensions [0 1 -1 0 0 0 0];
internalField uniform $flowVelocity;
boundaryField
{
    #includeEtc "caseDicts/setConstraintTypes"
    #include "include/fixedInlet"
    outlet { type pressureInletOutletVelocity; value $internalField; }
    sphere { type noSlip; }
}
""")
        with mock.patch.dict(os.environ, NO_ETC):
            check_case(c, "0")      # U symmetryPlane comes from setConstraintTypes; p lists farfield explicitly
            self.edit(c / "0/p", "farfield { type zeroGradient; }", '#includeEtc "caseDicts/setConstraintTypes"')
            with self.assertRaisesRegex(UnsupportedCase, "0/p uses boundary type symmetryPlane on patch farfield"):
                check_case(c, "0")
            self.edit(c / "0/U", "$flowVelocity", "$flowSpeed")
            with self.assertRaisesRegex(UnsupportedCase, r"\$flowSpeed is not defined"):
                check_case(c, "0")

    def test_function_objects_that_will_not_run_are_named(self):
        c = self.case("sphere-laminar")
        cd = c / "system/controlDict"
        txt = cd.read_text()
        start = txt.index("functions")
        cd.write_text(txt[:start] + """functions
{
    #include "forceCoeffs"
    #include "streamLines"
    #includeFunc mag(U)
    cuttingPlane { #includeEtc "caseDicts/postProcessing/visualization/surfaces.cfg" }
    off { type surfaces; enabled false; }
}
""")
        (c / "system/forceCoeffs").write_text("""forceCoeffs1
{
    type forceCoeffs; libs (forces); patches ("motorBike.*"); rho rhoInf; rhoInf 1;
    liftDir (0 0 1); dragDir (1 0 0); CofR (0.72 0 0); pitchAxis (0 1 0); magUInf 20; lRef 1.42; Aref 0.75;
}
""")
        (c / "system/streamLines").write_text("streamLines { type streamLine; libs (fieldFunctionObjects); }\n")
        with mock.patch.dict(os.environ, NO_ETC):
            force, warnings = function_objects(c, rans=False)
        self.assertEqual(force[0], "forceCoeffs1")
        self.assertEqual(force[1]["lRef"], ["1.42"])
        text = "\n".join(warnings)
        self.assertIn("'streamLines' (type streamLine) is not supported", text)
        self.assertIn("'mag(U)' (#includeFunc)", text)
        self.assertIn("'cuttingPlane' (no type)", text)
        self.assertIn("surfaces.cfg", text)
        self.assertNotIn("'off'", text)
        self.assertEqual(len(warnings), 3)

    def run_check_command(self, *args):
        out = io.StringIO()
        with redirect_stdout(out), mock.patch.dict(os.environ, NO_ETC):
            status = check.main([str(a) for a in args])
        return status, out.getvalue()

    def test_check_command(self):
        status, out = self.run_check_command(self.case_with_fields("sphere-laminar"))
        self.assertEqual(status, 0)
        self.assertIn("fields from 0/", out)
        self.assertIn("simplefoam-gpu can run this case.", out)

        c = self.case_with_fields("sphere-kOmegaSST")
        self.edit(c / "0/nut", "sphere   { type nutUSpaldingWallFunction;", "sphere   { type nutkWallFunction;")
        self.edit(c / "system/controlDict", "    yPlus\n", "    streamLines { type streamLine; }\n    yPlus\n")
        status, out = self.run_check_command(c)
        self.assertEqual(status, 1)
        self.assertIn("0/nut uses boundary type nutkWallFunction on patch sphere", out)
        self.assertIn("'streamLines' (type streamLine) is not supported", out)
        self.assertNotIn("can run this case", out)

    def test_check_command_before_allrun(self):
        """With no time folder yet, the fields in 0.orig are checked; without a mesh, it says so."""
        c = self.case("sphere-laminar")
        shutil.copytree(EXAMPLES / "sphere-laminar/0.orig", c / "0.orig")
        (c / "constant/polyMesh/boundary").write_text(BOUNDARY_FULL)
        status, out = self.run_check_command(c)
        self.assertEqual(status, 0, out)
        self.assertIn("fields from 0.orig/", out)
        (c / "constant/polyMesh/boundary").unlink()
        status, out = self.run_check_command(c)
        self.assertEqual(status, 1)
        self.assertIn("mesh the case first", out)
        status, out = self.run_check_command(c / "system")
        self.assertEqual(status, 2)


if __name__ == "__main__":
    unittest.main()
