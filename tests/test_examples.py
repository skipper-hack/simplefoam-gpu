"""Full runs of the examples against OpenFOAM's converged drag (slow; opt in with SIMPLEFOAM_GPU_SLOW=1).

Mesh the examples first (examples/*/Allmesh, needs OpenFOAM). The reference values are OpenFOAM v2412 simpleFoam
runs of the same cases with the transpose fix (openfoam-fix/), after 3000 iterations.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
REFERENCE_CD = {"sphere-laminar": 1.869929, "sphere-kOmegaSST": 0.08372561}


@unittest.skipUnless(os.environ.get("SIMPLEFOAM_GPU_SLOW") == "1", "set SIMPLEFOAM_GPU_SLOW=1 to run")
class Examples(unittest.TestCase):
    def run_case(self, name):
        src = EXAMPLES / name
        if not (src / "constant/polyMesh").exists():
            self.skipTest(f"{name} is not meshed; run examples/{name}/Allmesh first")
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        case = d / name
        shutil.copytree(src, case, ignore=shutil.ignore_patterns("[1-9]*", "postProcessing", "log.*"))
        subprocess.run([sys.executable, "-m", "simplefoam_gpu", str(case), "--every", "1000"], check=True)
        last = (case / "postProcessing/forceCoeffs/0/coefficient.dat").read_text().strip().splitlines()[-1]
        return float(last.split()[1])

    def test_laminar(self):
        self.assertAlmostEqual(self.run_case("sphere-laminar"), REFERENCE_CD["sphere-laminar"], delta=1e-5)

    def test_kOmegaSST(self):
        self.assertAlmostEqual(self.run_case("sphere-kOmegaSST"), REFERENCE_CD["sphere-kOmegaSST"], delta=1e-6)


if __name__ == "__main__":
    unittest.main()
