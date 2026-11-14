"""GPU kernels against NumPy (needs CuPy and an NVIDIA GPU)."""
import unittest
import numpy as np

try:
    import cupy as cp
    cp.zeros(1)
except Exception:  # no CuPy or no GPU
    cp = None


@unittest.skipIf(cp is None, "needs CuPy and a CUDA GPU")
class Kernels(unittest.TestCase):
    def setUp(self):
        from simplefoam_gpu import kernels
        self.k = kernels
        self.rng = np.random.default_rng(1)

    def test_scatter_min_max_exact(self):
        n, m = 1000, 5000
        idx = self.rng.integers(0, n, m)
        vals = self.rng.standard_normal((m, 3)) * 10.0 ** self.rng.integers(-300, 300, (m, 3))
        base = self.rng.standard_normal((n, 3))
        for name, ufunc in (("scatter_min", np.minimum), ("scatter_max", np.maximum)):
            want = base.copy()
            ufunc.at(want, idx, vals)
            got = cp.asarray(base)
            getattr(self.k, name)(got, cp.asarray(idx), cp.asarray(vals))
            np.testing.assert_array_equal(cp.asnumpy(got), want)  # bit for bit

    def test_segment_sum(self):
        cells = self.rng.integers(0, 300, 4000)
        b = self.rng.standard_normal((4000, 3))
        want = np.zeros((500, 3))
        np.add.at(want, cells, b)
        got = self.k.SegmentSum(cells, 500) @ cp.asarray(b)
        np.testing.assert_allclose(cp.asnumpy(got), want, rtol=1e-13, atol=1e-13)

    def test_colsum(self):
        X = self.rng.standard_normal((10000, 4))
        np.testing.assert_allclose(cp.asnumpy(self.k.colsum(cp.asarray(X))), X.sum(0), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
