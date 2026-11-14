# SPDX-License-Identifier: MIT
"""Fused GPU kernels for the hot loops (same arithmetic as the array code they replace, read once).

grad_limited_vec: Gauss linear gradient of a vector, optionally with OpenFOAM's cellLimited (minmod, k = 1) limiter
                  applied per component (cellLimitedGrad<vector, minmod>). One thread per cell, faces in face order.
momentum_flux:    per internal face, the explicit momentum face flux:
                  - phi * linearUpwindV correction (limited as in linearUpwindV<vector>::correction)
                  + nuEff_f * |Sf| * (k & grad(U)_f)                     (non-orthogonal laplacian correction)
                  + Sf & interpolate(nuEff * dev2(T(grad(U))))             (explicit stress term)
"""
import numpy as np
import cupy as cp

_SRC = r"""
#define SMALL 1e-15
#define VSMALL 1e-300

extern "C" __global__ void grad_lim_vec(
    const long long* off, const long long* face, const long long* other, const double* sign,
    const double* U, const double* Ub, const double* w, const double* Sf, const double* Cf,
    const double* C, const double* V, long long nc, int limit, double* g)
{
    long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (c >= nc) return;
    double uc[3] = {U[3*c], U[3*c+1], U[3*c+2]};
    double gr[9] = {0,0,0,0,0,0,0,0,0};
    for (long long j = off[c]; j < off[c+1]; ++j) {
        long long f = face[j], o = other[j];
        double s = sign[j], val[3];
        if (o >= 0) {
            double wf = w[f];
            for (int k = 0; k < 3; ++k)
                val[k] = (s > 0) ? wf*uc[k] + (1.0 - wf)*U[3*o+k] : wf*U[3*o+k] + (1.0 - wf)*uc[k];
        } else {
            long long b = -o - 1;
            for (int k = 0; k < 3; ++k) val[k] = Ub[3*b+k];
        }
        for (int i = 0; i < 3; ++i) {
            double si = s*Sf[3*f+i];
            for (int k = 0; k < 3; ++k) gr[3*i+k] += si*val[k];
        }
    }
    double iv = 1.0/V[c];
    for (int q = 0; q < 9; ++q) gr[q] *= iv;
    if (limit) {
        double mx[3] = {uc[0], uc[1], uc[2]}, mn[3] = {uc[0], uc[1], uc[2]};
        for (long long j = off[c]; j < off[c+1]; ++j) {
            long long o = other[j];
            for (int k = 0; k < 3; ++k) {
                double v = (o >= 0) ? U[3*o+k] : Ub[3*(-o-1)+k];
                mx[k] = fmax(mx[k], v); mn[k] = fmin(mn[k], v);
            }
        }
        for (int k = 0; k < 3; ++k) { mx[k] -= uc[k]; mn[k] -= uc[k]; }
        double lim[3] = {1.0, 1.0, 1.0};
        for (long long j = off[c]; j < off[c+1]; ++j) {
            long long f = face[j];
            double d0 = Cf[3*f] - C[3*c], d1 = Cf[3*f+1] - C[3*c+1], d2 = Cf[3*f+2] - C[3*c+2];
            for (int k = 0; k < 3; ++k) {
                double e = d0*gr[k] + d1*gr[3+k] + d2*gr[6+k];
                double r;
                if (e > SMALL) r = mx[k]/e;
                else if (e < -SMALL) r = mn[k]/e;
                else continue;
                lim[k] = fmin(lim[k], fmin(r, 1.0));
            }
        }
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) gr[3*i+k] *= lim[k];
    }
    for (int q = 0; q < 9; ++q) g[9*c+q] = gr[q];
}

extern "C" __global__ void momentum_flux(
    const long long* own, const long long* nei, const double* w, const double* phi, const double* U,
    const double* g, const double* nu, const double* Sf, const double* kvec, const double* dCfo,
    const double* dCfn, const double* magSf, long long nif, int do_corr, int do_nonorth, int do_dev2, double* F)
{
    long long f = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (f >= nif) return;
    long long o = own[f], n = nei[f];
    double wf = w[f], ph = phi[f];
    const double* go = g + 9*o; const double* gn = g + 9*n;
    double out[3] = {0, 0, 0};
    if (do_corr) {
        double corr[3], maxc[3];
        if (ph > 0) {
            for (int k = 0; k < 3; ++k) {
                maxc[k] = (1.0 - wf)*(U[3*n+k] - U[3*o+k]);
                corr[k] = dCfo[3*f]*go[k] + dCfo[3*f+1]*go[3+k] + dCfo[3*f+2]*go[6+k];
            }
        } else {
            for (int k = 0; k < 3; ++k) {
                maxc[k] = wf*(U[3*o+k] - U[3*n+k]);
                corr[k] = dCfn[3*f]*gn[k] + dCfn[3*f+1]*gn[3+k] + dCfn[3*f+2]*gn[6+k];
            }
        }
        double s2 = corr[0]*corr[0] + corr[1]*corr[1] + corr[2]*corr[2];
        double mc = corr[0]*maxc[0] + corr[1]*maxc[1] + corr[2]*maxc[2];
        if (s2 > 0) {
            if (mc < 0) { corr[0] = corr[1] = corr[2] = 0; }
            else if (s2 > mc) { double a = mc/(s2 + VSMALL); for (int k = 0; k < 3; ++k) corr[k] *= a; }
        }
        for (int k = 0; k < 3; ++k) out[k] -= ph*corr[k];
    }
    double nuo = nu[o], nun = nu[n];
    if (do_nonorth) {
        double nuf = wf*nuo + (1.0 - wf)*nun;
        double a = nuf*magSf[f];
        for (int k = 0; k < 3; ++k) {
            double gfk0 = wf*go[k] + (1.0 - wf)*gn[k];
            double gfk1 = wf*go[3+k] + (1.0 - wf)*gn[3+k];
            double gfk2 = wf*go[6+k] + (1.0 - wf)*gn[6+k];
            out[k] += a*(kvec[3*f]*gfk0 + kvec[3*f+1]*gfk1 + kvec[3*f+2]*gfk2);
        }
    }
    if (do_dev2) {
        double tro = go[0] + go[4] + go[8], trn = gn[0] + gn[4] + gn[8];
        double ao = wf*nuo, an = (1.0 - wf)*nun;
        double s0 = Sf[3*f], s1 = Sf[3*f+1], s2_ = Sf[3*f+2];
        for (int k = 0; k < 3; ++k) {
            // (Sf & A)_k with A = nu*(g^T - 2/3 tr(g) I), A^T row k = g[k][*]
            double vo = s0*go[3*k] + s1*go[3*k+1] + s2_*go[3*k+2];
            double vn = s0*gn[3*k] + s1*gn[3*k+1] + s2_*gn[3*k+2];
            double sk = (k == 0) ? s0 : (k == 1 ? s1 : s2_);
            out[k] += ao*(vo - (2.0/3.0)*tro*sk) + an*(vn - (2.0/3.0)*trn*sk);
        }
    }
    for (int k = 0; k < 3; ++k) F[3*f+k] = out[k];
}

extern "C" __global__ void grad_sca(
    const long long* off, const long long* face, const long long* other, const double* sign,
    const double* x, const double* xb, const double* w, const double* Sf, const double* V, long long nc, double* g)
{
    long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (c >= nc) return;
    double xc = x[c], g0 = 0, g1 = 0, g2 = 0;
    for (long long j = off[c]; j < off[c+1]; ++j) {
        long long f = face[j], o = other[j];
        double s = sign[j], val;
        if (o >= 0) { double wf = w[f]; val = (s > 0) ? wf*xc + (1.0 - wf)*x[o] : wf*x[o] + (1.0 - wf)*xc; }
        else val = xb[-o - 1];
        g0 += s*Sf[3*f]*val; g1 += s*Sf[3*f+1]*val; g2 += s*Sf[3*f+2]*val;
    }
    double iv = 1.0/V[c];
    g[3*c] = g0*iv; g[3*c+1] = g1*iv; g[3*c+2] = g2*iv;
}

extern "C" __global__ void spalding_utau(
    const double* magUp, const double* magGradU, const double* y, const double* nutw, double nuw,
    const bool* active_face, double kappa, double E, long long n, double* uTau)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uTau[i] = 0.0;
    if (!active_face[i]) return;
    double ut = sqrt((nutw[i] + nuw)*magGradU[i]);
    if (!(1e-150 < ut)) return;
    int iter = 0; double err;
    do {
        double kUu = fmin(kappa*magUp[i]/ut, 50.0);
        double fkUu = exp(kUu) - 1.0 - kUu*(1.0 + 0.5*kUu);
        double f = -ut*y[i]/nuw + magUp[i]/ut + 1.0/E*(fkUu - 1.0/6.0*kUu*kUu*kUu);
        double df = y[i]/nuw + magUp[i]/(ut*ut) + 1.0/E*kUu*fkUu/ut;
        double uNew = ut + f/df;
        err = fabs((ut - uNew)/ut);
        ut = uNew;
    } while (ut > 1e-150 && err > 0.01 && ++iter < 10);
    uTau[i] = fmax(0.0, ut);
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
        _k(name)(((n + 127) // 128,), (128,), args)


class CellFaces:
    """Cell -> faces adjacency (CSR, faces in increasing face order), built once per mesh."""

    def __init__(self, own, nei, nif, nc):
        own = np.asarray(own); nei = np.asarray(nei)
        nf = len(own)
        cells = np.concatenate([own, nei])
        faces = np.concatenate([np.arange(nf), np.arange(nif)])
        other = np.concatenate([np.r_[nei, -(np.arange(nf - nif) + 1)], own[:nif]])
        sign = np.concatenate([np.ones(nf), -np.ones(nif)])
        order = np.lexsort((faces, cells))
        off = np.zeros(nc + 1, np.int64); np.cumsum(np.bincount(cells, minlength=nc), out=off[1:])
        self.off = cp.asarray(off)
        self.face = cp.asarray(faces[order].astype(np.int64))
        self.other = cp.asarray(other[order].astype(np.int64))
        self.sign = cp.asarray(sign[order])


def _cf(m):
    if not hasattr(m, "cellfaces"):
        m.cellfaces = CellFaces(m.cpu["own"], m.cpu["nei"], m.nif, m.nc)
    return m.cellfaces


def grad_sca(m, x, xb):
    """Gauss linear gradient of a cell scalar, (nc, 3)."""
    cf = _cf(m)
    g = cp.empty((m.nc, 3))
    _launch("grad_sca", m.nc, cf.off, cf.face, cf.other, cf.sign, cp.ascontiguousarray(x, dtype=cp.float64),
            cp.ascontiguousarray(xb, dtype=cp.float64), m.w, m.Sf, m.V, cp.int64(m.nc), g)
    return g


def spalding_utau(magUp, magGradU, y, nutw, nuw, active, kappa, E):
    """nutUSpaldingWallFunction::calcUTau per face (Newton, maxIter 10, tolerance 0.01)."""
    n = len(magUp)
    out = cp.empty(n)
    _launch("spalding_utau", n, cp.ascontiguousarray(magUp), cp.ascontiguousarray(magGradU), cp.ascontiguousarray(y),
            cp.ascontiguousarray(nutw), cp.float64(nuw), cp.ascontiguousarray(active), cp.float64(kappa),
            cp.float64(E), cp.int64(n), out)
    return out


_interp1 = cp.ElementwiseKernel("raw float64 a, raw int64 own, raw int64 nei, float64 w", "float64 out",
                                "out = w * a[own[i]] + (1.0 - w) * a[nei[i]]", "simplefoam_gpu_interp1")


def interp_scalar(m, a):
    return _interp1(cp.ascontiguousarray(a, dtype=cp.float64), m.own_i, m.nei, m.w)


def grad_vec(m, U, Ub, limit):
    """Gauss linear gradient of a cell vector field, (nc, 3, 3) with [c, i, j] = d_i U_j; optionally cellLimited."""
    cf = _cf(m)
    g = cp.empty((m.nc, 3, 3))
    _launch("grad_lim_vec", m.nc, cf.off, cf.face, cf.other, cf.sign, cp.ascontiguousarray(U),
            cp.ascontiguousarray(Ub), m.w, m.Sf, m.Cf, m.C, m.V, cp.int64(m.nc), cp.int32(1 if limit else 0), g)
    return g


def momentum_flux(m, U, g, nu_c, phi_i, corr=True, nonorth=True, dev2=True):
    F = cp.empty((m.nif, 3))
    _launch("momentum_flux", m.nif, m.own_i, m.nei, m.w, cp.ascontiguousarray(phi_i), cp.ascontiguousarray(U),
            cp.ascontiguousarray(g), cp.ascontiguousarray(nu_c), m.Sf, m.kvec, m.dCfo, m.dCfn, m.magSf,
            cp.int64(m.nif), cp.int32(corr), cp.int32(nonorth), cp.int32(dev2), F)
    return F
