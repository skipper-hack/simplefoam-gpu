"""OpenFOAM file reading and writing (CPU only)."""
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
