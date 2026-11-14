# SPDX-License-Identifier: MIT
"""Exact float64 scatter-min / scatter-max on the GPU.

cupyx.scatter_min/max on float64 are not exact (results come back rounded), which breaks equality tests and
OpenFOAM-matching limiters. These kernels map doubles to order-preserving 64-bit integers and use 64-bit atomics.
"""
import cupy as cp

_SRC = r"""
__device__ __forceinline__ unsigned long long enc(double x) {
    unsigned long long b = __double_as_longlong(x);
    return (b >> 63) ? ~b : (b | 0x8000000000000000ULL);
}
__device__ __forceinline__ double dec(unsigned long long k) {
    unsigned long long b = (k >> 63) ? (k & 0x7FFFFFFFFFFFFFFFULL) : ~k;
    return __longlong_as_double(b);
}
extern "C" __global__ void encode(const double* x, unsigned long long* k, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) k[i] = enc(x[i]);
}
extern "C" __global__ void decode(const unsigned long long* k, double* x, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) x[i] = dec(k[i]);
}
extern "C" __global__ void smin(unsigned long long* out, const long long* idx, const double* v, long long n,
                                long long ncol) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) { long long r = i / ncol, c = i % ncol; atomicMin(&out[idx[r] * ncol + c], enc(v[i])); }
}
extern "C" __global__ void smax(unsigned long long* out, const long long* idx, const double* v, long long n,
                                long long ncol) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) { long long r = i / ncol, c = i % ncol; atomicMax(&out[idx[r] * ncol + c], enc(v[i])); }
}
extern "C" __global__ void segsum(const double* b, const long long* perm, const long long* off,
                                  const long long* cell, double* out, long long nrows, long long ncol) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= nrows * ncol) return;
    long long r = i / ncol, c = i % ncol;
    double acc = 0.0;
    for (long long j = off[r]; j < off[r + 1]; ++j) acc += b[perm[j] * ncol + c];
    out[cell[r] * ncol + c] = acc;
}
"""
_mod = None


def _k(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC)
    return _mod.get_function(name)


def _launch(name, n, *args):
    if n:
        _k(name)(((n + 255) // 256,), (256,), args)


def _scatter(op, out, idx, vals):
    """In place: out[idx[r], ...] = op(out[idx[r], ...], vals[r, ...]) exactly. out, vals float64, C-contiguous."""
    assert out.dtype == cp.float64 and out.flags.c_contiguous
    vals = cp.ascontiguousarray(cp.broadcast_to(cp.asarray(vals, cp.float64), (len(idx),) + out.shape[1:]))
    idx = cp.ascontiguousarray(idx, dtype=cp.int64)
    ncol = 1
    for s in out.shape[1:]:
        ncol *= s
    keys = cp.empty(out.size, cp.uint64)
    _launch("encode", out.size, out, keys, cp.int64(out.size))
    _launch(op, vals.size, keys, idx, vals, cp.int64(vals.size), cp.int64(ncol))
    _launch("decode", out.size, keys, out, cp.int64(out.size))


def scatter_min(out, idx, vals):
    _scatter("smin", out, idx, vals)


def scatter_max(out, idx, vals):
    _scatter("smax", out, idx, vals)


def colsum(X):
    """Column sums of a 2-D array (fast path: cuBLAS gemv; CuPy's axis-0 reduction is ~70x slower here)."""
    if X.ndim == 1:
        return X.sum()
    return cp.ones(X.shape[0], X.dtype) @ cp.ascontiguousarray(X)


class SegmentSum:
    """out[cell] = sum of b over the faces mapped to that cell (deterministic, face order within a cell).

    Replaces a CSR (n_cells x n_faces) one-hot matrix product, for which cuSPARSE is slow when most rows are empty."""

    def __init__(self, cells, n_out):
        import numpy as np
        cells = np.asarray(cells)
        order = np.argsort(cells, kind="stable")
        uc, start = np.unique(cells[order], return_index=True)
        self.perm = cp.asarray(order.astype(np.int64))
        self.off = cp.asarray(np.r_[start, len(cells)].astype(np.int64))
        self.cell = cp.asarray(uc.astype(np.int64))
        self.n_out = n_out
        self.shape = (n_out, len(cells))

    def __matmul__(self, b):
        b = cp.ascontiguousarray(b, dtype=cp.float64)
        ncol = 1 if b.ndim == 1 else b.shape[1]
        out = cp.zeros((self.n_out,) + b.shape[1:])
        n = len(self.cell) * ncol
        _launch("segsum", n, b, self.perm, self.off, self.cell, out, cp.int64(len(self.cell)), cp.int64(ncol))
        return out


# ---- optional stage profiler (SIMPLEFOAM_GPU_PROFILE=1) shared by all modules -------------------------------------
import os as _os
import time as _time
PROFILE = _os.environ.get("SIMPLEFOAM_GPU_PROFILE") == "1"
_prof = {}
_prof_t = [0.0]


def tick(label=None):
    """With SIMPLEFOAM_GPU_PROFILE=1: synchronise and charge the elapsed time since the last tick to `label`."""
    if not PROFILE:
        return
    cp.cuda.Device().synchronize()
    t = _time.perf_counter()
    if label is not None:
        _prof[label] = _prof.get(label, 0.0) + t - _prof_t[0]
    _prof_t[0] = t


def profile_report():
    tot = sum(_prof.values())
    return "\n".join(f"  {k:26s} {v:8.2f} s  {100 * v / tot:5.1f}%" for k, v in sorted(_prof.items(), key=lambda x: -x[1]))
