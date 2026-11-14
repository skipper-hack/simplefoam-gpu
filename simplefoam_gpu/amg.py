# SPDX-License-Identifier: MIT
"""Agglomeration multigrid (fixed aggregates, coefficients re-summed on the GPU) + preconditioned CG."""
import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as csp
from .kernels import colsum
import os
FP64 = os.environ.get("SIMPLEFOAM_GPU_AMG_FP32") != "1"   # float32 V-cycle is opt-in (measured slower on an RTX 3090)


def _pairwise(A):
    """One pass of heavy-edge matching on a symmetric M-matrix (scipy CSR). Returns aggregate ids."""
    n = A.shape[0]
    C = A.tocoo()
    m = C.row != C.col
    r, c, w = C.row[m].astype(np.int64), C.col[m].astype(np.int64), -C.data[m]
    lo, hi = np.minimum(r, c), np.maximum(r, c)
    h = ((lo * 2654435761 + hi * 40503) % 1000003) / 1000003.0     # symmetric tie-break
    w = w * (1.0 + 1e-3 * h)
    pos = w > 0
    r, c, w = r[pos], c[pos], w[pos]
    agg = -np.ones(n, np.int64)
    nagg = 0
    for _ in range(8):
        ok = (agg[r] < 0) & (agg[c] < 0)
        if not ok.any():
            break
        rr, cc, ww = r[ok], c[ok], w[ok]
        o = np.lexsort((-ww, rr))
        rr, cc = rr[o], cc[o]
        first = np.r_[True, rr[1:] != rr[:-1]]
        best = -np.ones(n, np.int64)
        best[rr[first]] = cc[first]
        i = np.nonzero(best >= 0)[0]
        i = i[best[best[i]] == i]
        i = i[i < best[i]]
        j = best[i]
        agg[i] = nagg + np.arange(len(i))
        agg[j] = agg[i]
        nagg += len(i)
    # leftovers join their strongest already-aggregated neighbour, else become singletons
    ok = (agg[r] < 0) & (agg[c] >= 0)
    if ok.any():
        rr, cc, ww = r[ok], c[ok], w[ok]
        o = np.lexsort((-ww, rr))
        rr, cc = rr[o], cc[o]
        first = np.r_[True, rr[1:] != rr[:-1]]
        agg[rr[first]] = agg[cc[first]]
    left = np.nonzero(agg < 0)[0]
    agg[left] = nagg + np.arange(len(left))
    return agg, nagg + len(left)


def _coarsen(A, passes):
    n = A.shape[0]
    agg = np.arange(n)
    Ac = A
    for _ in range(passes):
        a, na = _pairwise(Ac)
        P = sp.csr_matrix((np.ones(len(a)), (np.arange(len(a)), a)), shape=(len(a), na))
        Ac = (P.T @ Ac @ P).tocsr()
        Ac.sum_duplicates(); Ac.sort_indices()
        agg = a[agg]
    return agg, Ac


class AMG:
    def __init__(self, indices, indptr, data, passes=2, coarsest=400, nu=2, omega=0.8, max_levels=25):
        """indices/indptr: numpy CSR pattern (sorted, with diagonal); data: numpy values of an SPD M-matrix."""
        n = len(indptr) - 1
        A = sp.csr_matrix((np.asarray(data, float), indices, indptr), shape=(n, n))
        self.nu, self.omega = nu, omega
        self.levels = []
        while True:
            lev = dict(n=A.shape[0])
            lev["A"] = csp.csr_matrix((cp.asarray(A.data), cp.asarray(A.indices), cp.asarray(A.indptr)),
                                      shape=A.shape)
            rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
            lev["diagpos"] = cp.asarray(np.nonzero(rows == A.indices)[0])
            self.levels.append(lev)
            if A.shape[0] <= coarsest or len(self.levels) >= max_levels:
                break
            agg, Ac = _coarsen(A, passes)
            nC = Ac.shape[0]
            if nC >= 0.9 * A.shape[0]:
                break
            crow = np.repeat(np.arange(nC), np.diff(Ac.indptr))
            ckey = crow.astype(np.int64) * nC + Ac.indices
            fkey = agg[rows].astype(np.int64) * nC + agg[A.indices]
            idx = np.searchsorted(ckey, fkey)
            assert np.array_equal(ckey[idx], fkey)
            S = sp.csr_matrix((np.ones(len(idx)), (idx, np.arange(len(idx)))), shape=(len(ckey), len(idx)))
            lev["S"] = csp.csr_matrix(S)
            lev["agg"] = cp.asarray(agg)
            R = sp.csr_matrix((np.ones(len(agg)), (agg, np.arange(len(agg)))), shape=(nC, len(agg)))
            lev["R"] = csp.csr_matrix(R)
            A = Ac
        self.sizes = [l["n"] for l in self.levels]
        self._finish()

    def _finish(self):
        """Smoother and coarse-solver data. With SIMPLEFOAM_GPU_AMG_FP32=1 the V-cycle (preconditioner only) runs in
        float32; the surrounding CG iteration and its residuals always stay float64."""
        f32 = not FP64
        for l in self.levels:
            l["invd"] = self.omega / l["A"].data[l["diagpos"]]
            if f32:
                A = l["A"]
                l["A32"] = csp.csr_matrix((A.data.astype(cp.float32), A.indices, A.indptr), shape=A.shape)
                l["invd32"] = l["invd"].astype(cp.float32)
                if "R" in l and "R32" not in l:
                    l["R32"] = l["R"].astype(cp.float32)
        last = self.levels[-1]
        last["inv"] = cp.linalg.inv(last["A"].toarray())
        if f32:
            last["inv32"] = last["inv"].astype(cp.float32)

    def update(self, data):
        """New fine-level values (same pattern, GPU array); re-sum the coarse levels."""
        self.levels[0]["A"].data[:] = data
        for a, b in zip(self.levels[:-1], self.levels[1:]):
            b["A"].data[:] = a["S"] @ a["A"].data
        self._finish()

    def vcycle(self, b, l=0):
        if not FP64 and l == 0:
            return self._vcycle32(b.astype(cp.float32), 0).astype(cp.float64)
        L = self.levels[l]
        if l == len(self.levels) - 1:
            return L["inv"] @ b
        A, invd = L["A"], L["invd"]
        x = invd * b
        for _ in range(self.nu - 1):
            x += invd * (b - A @ x)
        xc = self.vcycle(L["R"] @ (b - A @ x), l + 1)
        x += xc[L["agg"]]
        for _ in range(self.nu):
            x += invd * (b - A @ x)
        return x

    def _vcycle32(self, b, l):
        L = self.levels[l]
        if l == len(self.levels) - 1:
            return L["inv32"] @ b
        A, invd = L["A32"], L["invd32"]
        x = invd * b
        for _ in range(self.nu - 1):
            x += invd * (b - A @ x)
        xc = self._vcycle32(L["R32"] @ (b - A @ x), l + 1)
        x += xc[L["agg"]]
        for _ in range(self.nu):
            x += invd * (b - A @ x)
        return x

    def solve(self, b, x, tol=1e-8, rel=0.05, maxit=200):
        """PCG. Residual norm follows OpenFOAM's normalisation. Returns x, initial residual, final, iterations."""
        A = self.levels[0]["A"]
        Ax = A @ x
        xref = x.mean()
        Aref = (A @ cp.ones_like(x)) * xref
        norm = float(cp.abs(Ax - Aref).sum() + cp.abs(b - Aref).sum()) + 1e-20
        r = b - Ax
        res0 = res = float(cp.abs(r).sum()) / norm
        it = 0
        if res0 > tol:
            z = self.vcycle(r)
            p = z.copy()
            rz = float(r @ z)
            while it < maxit:
                Ap = A @ p
                alpha = rz / float(p @ Ap)
                x = x + alpha * p
                r -= alpha * Ap
                it += 1
                res = float(cp.abs(r).sum()) / norm
                if res < tol or res < rel * res0:
                    break
                z = self.vcycle(r)
                rz_new = float(r @ z)
                p = z + (rz_new / rz) * p
                rz = rz_new
        return x, res0, res, it


def bicgstab(matvec, b, x, invd, rel=0.05, tol=1e-8, maxit=50, r=None):
    """Jacobi-preconditioned BiCGStab for several right-hand sides at once (columns).
    r: the initial residual b - A x when the caller already has it."""
    if r is None:
        r = b - matvec(x)
    res0 = colsum(cp.abs(r))
    bn = colsum(cp.abs(b)) + 1e-300
    target = cp.maximum(rel * res0, tol * bn)
    r0 = r.copy()
    rho = alpha = omega = cp.ones(b.shape[1])
    v = cp.zeros_like(b); p = cp.zeros_like(b)
    it = 0
    while it < maxit and bool((colsum(cp.abs(r)) > target).any()):
        rho_new = colsum(r0 * r)
        beta = _div(rho_new, rho) * _div(alpha, omega)
        p = r + beta * (p - omega * v)
        y = invd * p
        v = matvec(y)
        alpha = _div(rho_new, colsum(r0 * v))
        s = r - alpha * v
        z = invd * s
        t = matvec(z)
        omega = _div(colsum(t * s), colsum(t * t))
        x = x + alpha * y + omega * z
        r = s - omega * t
        rho = rho_new
        it += 1
    return x, res0, it


def _div(a, b):
    return cp.where(b != 0, a / cp.where(b != 0, b, 1.0), 0.0)
