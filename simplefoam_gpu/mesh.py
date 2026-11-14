# SPDX-License-Identifier: MIT
"""Mesh geometry (computed like OpenFOAM's primitiveMesh/surfaceInterpolation) and GPU operators."""
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
import os
from . import foamio
_REFERENCE = os.environ.get("SIMPLEFOAM_GPU_REFERENCE") == "1"


def geometry(m):
    """CPU geometry from raw polyMesh arrays. Returns dict of numpy arrays."""
    pts, own, nei = m["points"], m["owner"], m["neighbour"]
    fl, fo = m["face_labels"], m["face_offsets"]
    nf, nif = len(own), len(nei)
    nc = int(max(own.max(), nei.max())) + 1
    nper = np.diff(fo)
    fidx = np.repeat(np.arange(nf), nper)
    P = pts[fl]
    favg = np.add.reduceat(P, fo[:-1], axis=0) / nper[:, None]
    nxt = np.arange(len(fl)) + 1
    nxt[fo[1:] - 1] = fo[:-1]
    Pn = P[nxt]
    fC = favg[fidx]
    n = np.cross(Pn - P, fC - P)
    a = np.linalg.norm(n, axis=1)
    sumN = np.add.reduceat(n, fo[:-1], axis=0)
    sumA = np.add.reduceat(a, fo[:-1])
    sumAc = np.add.reduceat(a[:, None] * (P + Pn + fC), fo[:-1], axis=0)
    del P, Pn, fC, n, a
    Cf = np.where(sumA[:, None] > 1e-300, sumAc / (3.0 * np.maximum(sumA, 1e-300))[:, None], favg)
    Sf = 0.5 * sumN
    # cells
    cnt = np.bincount(own, minlength=nc) + np.bincount(nei, minlength=nc)
    cEst = np.stack([np.bincount(own, Cf[:, k], nc) + np.bincount(nei, Cf[:nif, k], nc) for k in range(3)], 1)
    cEst /= cnt[:, None]
    pv_o = np.einsum("ij,ij->i", Sf, Cf - cEst[own])
    pv_n = np.einsum("ij,ij->i", Sf[:nif], cEst[nei] - Cf[:nif])
    pc_o = 0.75 * Cf + 0.25 * cEst[own]
    pc_n = 0.75 * Cf[:nif] + 0.25 * cEst[nei]
    V3 = np.bincount(own, pv_o, nc) + np.bincount(nei, pv_n, nc)
    C = np.stack([np.bincount(own, pv_o * pc_o[:, k], nc) + np.bincount(nei, pv_n * pc_n[:, k], nc)
                  for k in range(3)], 1)
    C = np.where(np.abs(V3)[:, None] > 1e-300, C / V3[:, None], cEst)
    V = V3 / 3.0
    magSf = np.linalg.norm(Sf, axis=1)
    nhat = Sf / magSf[:, None]
    # interpolation weights and non-orthogonal coefficients (internal faces)
    oi = own[:nif]
    so = np.abs(np.einsum("ij,ij->i", Sf[:nif], Cf[:nif] - C[oi]))
    sn = np.abs(np.einsum("ij,ij->i", Sf[:nif], C[nei] - Cf[:nif]))
    w = np.where(so + sn > 1e-300, sn / np.maximum(so + sn, 1e-300), 0.5)
    delta = C[nei] - C[oi]
    nd = np.einsum("ij,ij->i", nhat[:nif], delta)
    nodc = 1.0 / np.maximum(nd, 0.05 * np.linalg.norm(delta, axis=1))
    kvec = nhat[:nif] - delta * nodc[:, None]
    # boundary: patch-normal delta
    bcell = own[nif:]
    db = np.einsum("ij,ij->i", nhat[nif:], Cf[nif:] - C[bcell])
    dcb = 1.0 / db
    return dict(nc=nc, nf=nf, nif=nif, own=own, nei=nei, Sf=Sf, magSf=magSf, nhat=nhat, Cf=Cf, C=C, V=V,
                w=w, nodc=nodc, kvec=kvec, dcb=dcb, bcell=bcell, patches=m["patches"])


class Mesh:
    """GPU-side mesh: geometry arrays and fixed sparse operators (face->cell sums as SpMV)."""

    def __init__(self, case):
        raw = foamio.read_mesh(case)
        self.raw = raw
        g = geometry(raw)
        self.cpu = g
        self.nc, self.nf, self.nif = g["nc"], g["nf"], g["nif"]
        self.nb = self.nf - self.nif
        nc, nf, nif, nb = self.nc, self.nf, self.nif, self.nb
        A = cp.asarray
        self.own = A(g["own"]); self.nei = A(g["nei"]); self.own_i = self.own[:nif]
        self.bcell = A(g["bcell"])
        self.Sf = A(g["Sf"]); self.magSf = A(g["magSf"]); self.nhat = A(g["nhat"])
        self.Sf_i, self.Sf_b = self.Sf[:nif], self.Sf[nif:]
        self.magSf_i, self.magSf_b = self.magSf[:nif], self.magSf[nif:]
        self.nb_hat = self.nhat[nif:]
        self.C = A(g["C"]); self.V = A(g["V"]); Cf = A(g["Cf"])
        self.w = A(g["w"]); self.nodc = A(g["nodc"]); self.kvec = A(g["kvec"]); self.dcb = A(g["dcb"])
        self.dCfo = Cf[:nif] - self.C[self.own_i]
        self.dCfn = Cf[:nif] - self.C[self.nei]
        self.dCfb = Cf[nif:] - self.C[self.bcell]
        self.Cf_b = Cf[nif:]
        self.Cf = Cf
        self.patches = []
        for p in g["patches"]:
            s = p["startFace"] - nif
            self.patches.append(dict(name=p["name"], type=p["type"], sl=slice(s, s + p["nFaces"])))
        own, nei = g["own"], g["nei"]

        def onehot(rows, ncols, sign=None):
            n = len(rows)
            order = np.argsort(rows, kind="stable")
            indptr = np.zeros(nc + 1, np.int32)
            np.cumsum(np.bincount(rows, minlength=nc), out=indptr[1:])
            data = np.ones(n) if sign is None else sign[order]
            return csp.csr_matrix((A(data), A(order.astype(np.int32)), A(indptr)), shape=(nc, ncols))

        self.Mown = onehot(own[:nif], nif)          # cell <- internal faces it owns
        self.Mnei = onehot(nei, nif)                # cell <- internal faces where it is neighbour
        from .kernels import SegmentSum
        self.Mb = SegmentSum(own[nif:], nc)         # cell <- its boundary faces
        rows = np.concatenate([own, nei]); cols = np.concatenate([np.arange(nf), np.arange(nif)])
        sign = np.concatenate([np.ones(nf), -np.ones(nif)])
        order = np.lexsort((cols, rows))
        indptr = np.zeros(nc + 1, np.int32)
        np.cumsum(np.bincount(rows, minlength=nc), out=indptr[1:])
        self.Dsum = csp.csr_matrix((A(sign[order]), A(cols[order].astype(np.int32)), A(indptr)), shape=(nc, nf))
        self.Dint = (self.Mown - self.Mnei).tocsr()   # signed sum over internal faces only
        # cell-cell matrix patterns. data order: [upper(nif) | lower(nif) | diag(nc)]
        r = np.concatenate([own[:nif], nei]); c = np.concatenate([nei, own[:nif]])
        order = np.lexsort((c, r))
        indptr = np.zeros(nc + 1, np.int32)
        np.cumsum(np.bincount(r, minlength=nc), out=indptr[1:])
        self.off_perm = A(order)
        self.Aoff = csp.csr_matrix((cp.zeros(2 * nif), A(c[order].astype(np.int32)), A(indptr)), shape=(nc, nc))
        r = np.concatenate([r, np.arange(nc)]); c = np.concatenate([c, np.arange(nc)])
        order = np.lexsort((c, r))
        indptr = np.zeros(nc + 1, np.int32)
        np.cumsum(np.bincount(r, minlength=nc), out=indptr[1:])
        self.full_perm = A(order)
        self.full_indices = c[order].astype(np.int32)
        self.full_indptr = indptr

    # ---- interpolation / differential operators -------------------------------------------------
    def interp(self, c):
        """Linear interpolation of a cell field to internal faces."""
        if c.ndim == 1 and c.dtype == cp.float64 and not _REFERENCE:
            from .fused import interp_scalar
            return interp_scalar(self, c)
        w = self.w.reshape((-1,) + (1,) * (c.ndim - 1))
        return w * c[self.own_i] + (1.0 - w) * c[self.nei]

    def grad(self, c, cb):
        """Gauss linear gradient. scalar -> (nc,3); (nc,k) -> (nc,3,k) with [i,j] = d_i c_j."""
        if not _REFERENCE:
            from . import fused
            if c.ndim == 1:
                return fused.grad_sca(self, c, cb)
            if c.ndim == 2 and c.shape[1] == 3:
                return fused.grad_vec(self, c, cb, limit=False)
        F = cp.concatenate([self.interp(c), cb])
        if c.ndim == 1:
            return (self.Dsum @ (self.Sf * F[:, None])) / self.V[:, None]
        k = c.shape[1]
        X = (self.Sf[:, :, None] * F[:, None, :]).reshape(self.nf, 3 * k)
        return (self.Dsum @ X).reshape(self.nc, 3, k) / self.V[:, None, None]

    def set_off(self, upper, lower):
        self.Aoff.data[:] = cp.concatenate([upper, lower])[self.off_perm]
        return self.Aoff
