# SPDX-License-Identifier: MIT
"""k-omega SST (OpenFOAM v2412 kOmegaSSTBase) with kqRWallFunction, omegaWallFunction (binomial2 blending),
nutUSpaldingWallFunction and nutkWallFunction (stepwise blending), on the GPU.

Mirrors kOmegaSSTBase::correct() step by step; comments name the OpenFOAM pieces. Schemes as in the supported
fvSchemes: div(phi,k|omega) bounded Gauss upwind, laplacian Gauss linear corrected, grad default Gauss linear,
grad(U) cellLimited Gauss linear 1 (supplied by the caller).
"""
import math
from pathlib import Path
import numpy as np
import cupy as cp
from . import foamio
from . import walldist
from .amg import bicgstab
from . import kernels

SMALL = 1e-15
# Scalar boundary types that act as zeroGradient: for scalars, OpenFOAM's slip, symmetry and symmetryPlane fields
# take the cell value and have zero normal gradient (basicSymmetryFvPatchField<scalar>, transformFvPatchField<scalar>)
ZERO_GRADIENT_SCALAR = ("zeroGradient", "slip", "symmetry", "symmetryPlane")


from .kernels import tick as _tick
ROOTVSMALL = 1e-150


class ScalarBC:
    """Boundary conditions for a cell-centred scalar (k, omega)."""

    def __init__(self, solver, field, name):
        m = solver.m
        nb = m.nb
        self.name = name
        self.fixed = cp.zeros(nb, bool)      # fixedValue
        self.io = cp.zeros(nb, bool)         # inletOutlet
        self.wf = cp.zeros(nb, bool)         # omegaWallFunction (fixed, value set from the wall cell)
        self.ref = cp.zeros(nb)
        self.text = []
        for pt in m.patches:
            sl, n = pt["sl"], pt["sl"].stop - pt["sl"].start
            e = foamio.patch_entry(field["boundary"], pt["name"], pt["groups"])
            t = e["type"][0]
            self.text.append(t)
            if t == "fixedValue":
                self.fixed[sl] = True; self.ref[sl] = cp.asarray(foamio.expand(e["value"], n, 1))
            elif t == "inletOutlet":
                self.io[sl] = True; self.ref[sl] = cp.asarray(foamio.expand(e["inletValue"], n, 1))
            elif t == "omegaWallFunction":
                self.wf[sl] = True; self.ref[sl] = cp.asarray(foamio.expand(e["value"], n, 1))
            elif t in ZERO_GRADIENT_SCALAR or t == "kqRWallFunction":
                pass
            else:
                raise NotImplementedError(f"{name} boundary type {t}")
        self.entries = [_entry_text(foamio.patch_entry(field["boundary"], pt["name"], pt["groups"])) for pt in m.patches]

    def is_fixed(self, phi_b):
        return self.fixed | self.wf | (self.io & (phi_b < 0))

    def values(self, cells_val, bcell, phi_b):
        return cp.where(self.is_fixed(phi_b), self.ref, cells_val[bcell])

    def coeffs(self, dcb, phi_b):
        """value/gradient internal and boundary coefficients (OpenFOAM fvPatchField conventions)."""
        f = self.is_fixed(phi_b).astype(cp.float64)
        vic = 1.0 - f
        vbc = f * self.ref
        gic = -f * dcb
        gbc = f * dcb * self.ref
        return vic, vbc, gic, gbc


class KOmegaSST:
    alphaK1, alphaK2, alphaOmega1, alphaOmega2 = 0.85, 1.0, 0.5, 0.856
    gamma1, gamma2, beta1, beta2, betaStar = 5.0 / 9.0, 0.44, 0.075, 0.0828, 0.09
    a1, b1, c1 = 0.31, 1.0, 10.0
    Cmu, kappa, E = 0.09, 0.41, 9.8
    omega_beta1 = 0.075   # omegaWallFunction beta1
    # wallFunctionCoefficients::calcYPlusLam: 10 fixed-point iterations from 11 (11.53 for kappa 0.41, E 9.8)
    yPlusLam = 11.0
    for _ in range(10):
        yPlusLam = math.log(max(E * yPlusLam, 1.0)) / kappa

    def __init__(self, solver, start, relax_k, relax_omega, log=print):
        self.s = s = solver
        m = s.m
        self.log = log
        case = Path(s.case)
        fk = foamio.read_field(case / start / "k")
        fo = foamio.read_field(case / start / "omega")
        fn = foamio.read_field(case / start / "nut")
        self.k = cp.asarray(foamio.expand(fk["internal"], m.nc, 1))
        self.omega = cp.asarray(foamio.expand(fo["internal"], m.nc, 1))
        self.nut = cp.asarray(foamio.expand(fn["internal"], m.nc, 1))
        self.bk = ScalarBC(s, fk, "k")
        self.bo = ScalarBC(s, fo, "omega")
        self.relax_k, self.relax_omega = relax_k, relax_omega
        # current boundary values (k, omega: from file; nut: from file, used as the Spalding start value)
        phi_b = s.phi[m.nif:]
        self.kb = self.bk.values(self.k, m.bcell, phi_b)      # evaluated as OpenFOAM does on read
        self.ob = self.bo.values(self.omega, m.bcell, phi_b)
        self.nutb = self._file_boundary(fn, 1)
        self.spalding = cp.zeros(m.nb, bool)     # nutUSpaldingWallFunction
        self.nutk = cp.zeros(m.nb, bool)         # nutkWallFunction
        self.nut_zg = cp.zeros(m.nb, bool)       # slip, symmetry, symmetryPlane: the cell value
        self.nut_text = []
        for pt in m.patches:
            e = foamio.patch_entry(fn["boundary"], pt["name"], pt["groups"])
            t = e["type"][0]
            self.nut_text.append(_entry_text(e))
            if t == "nutUSpaldingWallFunction":
                self.spalding[pt["sl"]] = True
            elif t == "nutkWallFunction":
                self.nutk[pt["sl"]] = True
            elif t in ZERO_GRADIENT_SCALAR[1:]:
                self.nut_zg[pt["sl"]] = True
            elif t != "calculated":
                raise NotImplementedError(f"nut boundary type {t}")
        self.has_nutk = bool(self.nutk.any())
        # wall distances
        raw, geom = m.raw, m.cpu
        y_cell, self.y_wall, self.wall = walldist.compute(raw, geom)
        wd = case / start / "wallDistance"
        if not wd.exists():   # wall distance is time-independent: take it from any saved time (latest first)
            cands = sorted((d for d in case.iterdir() if d.is_dir() and (d / "wallDistance").exists()
                            and d.name.replace(".", "", 1).isdigit()), key=lambda d: -float(d.name))
            wd = cands[0] / "wallDistance" if cands else None
        if wd is not None:
            self.y = cp.asarray(foamio.expand(foamio.read_field(wd)["internal"], m.nc, 1))
            log(f"wall distance: OpenFOAM meshWave field {wd} (GPU estimate differs by mean "
                f"{float((cp.abs(y_cell - self.y) / self.y).mean()):.2e})")
        else:
            self.y = y_cell
            log("wall distance: GPU meshWave estimate (run checkMesh -writeFields '(wallDistance)' for an exact match)")
        # omega wall-function cells and corner weights
        self.owf = self.bo.wf
        wfc = m.bcell[self.owf]
        cnt = cp.zeros(m.nc); kernels_add(cnt, wfc, 1.0)
        self.wf_w = 1.0 / cnt[m.bcell]          # per boundary face (only meaningful on owf faces)
        self.wf_cells = cp.unique(wfc)
        self.wf_cell_mask = cp.zeros(m.nc, bool); self.wf_cell_mask[self.wf_cells] = True

    def _file_boundary(self, f, ncomp):
        m = self.s.m
        out = cp.zeros(m.nb)
        for pt in m.patches:
            e = foamio.patch_entry(f["boundary"], pt["name"], pt["groups"])
            if "value" in e:
                out[pt["sl"]] = cp.asarray(foamio.expand(e["value"], pt["sl"].stop - pt["sl"].start, ncomp))
        return out

    # ---- pieces --------------------------------------------------------------------------------
    def nu_eff(self):
        return self.nut + self.s.nu, self.nutb + self.s.nu

    def _F23(self, k, omega, y):
        nu = self.s.nu
        arg2 = cp.minimum(cp.maximum((2.0 / self.betaStar) * cp.sqrt(k) / (omega * y), 500.0 * nu / (y * y * omega)), 100.0)
        return cp.tanh(arg2 * arg2)

    def _F1(self, k, omega, y, CD):
        nu = self.s.nu
        CDp = cp.maximum(CD, 1.0e-10)
        arg1 = cp.minimum(cp.minimum(cp.maximum((1.0 / self.betaStar) * cp.sqrt(k) / (omega * y), 500.0 * nu / (y * y * omega)),
                                     (4.0 * self.alphaOmega2) * k / (CDp * y * y)), 10.0)
        return cp.tanh(arg1 ** 4)

    def _grad_scalar(self, c, cb):
        """Gauss linear gradient with OpenFOAM's boundary-corrected patch values."""
        m = self.s.m
        g = m.grad(c, cb)
        gc = g[m.bcell]
        sn = (cb - c[m.bcell]) * m.dcb
        gb = gc + m.nb_hat * (sn - (m.nb_hat * gc).sum(1))[:, None]
        return g, gb

    def _mag_up(self, U):
        """|U_cell - U_wall| on boundary faces, as the wall functions use it (a moving wall has fixedValue U)."""
        m = self.s.m
        Ub = self.s.bc_U(U, self.s.phi[m.nif:])[0]
        return cp.linalg.norm(U[m.bcell] - Ub, axis=1)

    def spalding_nut(self, U, nuw):
        """nutUSpaldingWallFunction::calcNut on Spalding faces."""
        uTau, magGradU = self._spalding_utau(U, nuw)
        return cp.maximum(0.0, uTau * uTau / (magGradU + ROOTVSMALL) - nuw)

    def _yplus_k(self, nuw):
        """Cmu^0.25 y sqrt(k_cell)/nu on boundary faces (the y+ of nutkWallFunction)."""
        m = self.s.m
        return self.Cmu ** 0.25 * cp.where(self.nutk, self.y_wall, 1.0) * cp.sqrt(self.k[m.bcell]) / nuw

    def nutk_nut(self, nuw):
        """nutkWallFunction::calcNut, stepwise blending: the log-law nut where y+ > yPlusLam, else zero."""
        yp = self._yplus_k(nuw)
        nutLog = nuw * yp * self.kappa / cp.log(cp.maximum(self.E * yp, 1 + 1e-4))
        return cp.where(yp > self.yPlusLam, nutLog, nuw) - nuw

    def yplus(self, U):
        """y+ on wall faces as the yPlus function object reports it (Spalding patches: y*uTau/nu; nutkWallFunction:
        Cmu^0.25 y sqrt(k)/nu, or y sqrt(nuEff |snGrad U|)/nu below yPlusLam)."""
        m = self.s.m
        nu = self.s.nu
        uTau, magGradU = self._spalding_utau(U, nu)
        ut_other = cp.sqrt((self.nutb + nu) * magGradU)
        y_other = self.y_wall * ut_other / nu
        yk = self._yplus_k(nu)
        yk = cp.where(yk < self.yPlusLam, y_other, yk)
        other = self.wall & ~self.spalding & ~self.nutk
        return cp.where(self.spalding, self.y_wall * uTau / nu,
                        cp.where(self.nutk, yk, cp.where(other, y_other, 0.0)))

    def _spalding_utau(self, U, nuw):
        """nutUSpaldingWallFunction::calcUTau (Newton, maxIter 10, tolerance 0.01) from the current wall nut."""
        m = self.s.m
        sl = self.spalding
        magUp = self._mag_up(U)                             # |U_cell - U_wall|
        magGradU = magUp * m.dcb                            # |snGrad(U_wall)|
        from . import fused
        uTau = fused.spalding_utau(magUp, magGradU, cp.where(sl, self.y_wall, 1.0), self.nutb, nuw, sl,
                                   self.kappa, self.E)
        return uTau, magGradU

    def correct_nut(self, S2, S2b, U):
        """kOmegaSSTBase::correctNut(S2) incl. boundary values, then the nut boundary conditions: wall functions,
        and the cell value on slip/symmetry patches."""
        s = self.s
        F23 = self._F23(self.k, self.omega, self.y)
        self.nut = self.a1 * self.k / cp.maximum(self.a1 * self.omega, self.b1 * F23 * cp.sqrt(S2))
        yb = self.y[s.m.bcell]
        F23b = self._F23(self.kb, self.ob, yb)
        nutb = self.a1 * self.kb / cp.maximum(self.a1 * self.ob, self.b1 * F23b * cp.sqrt(S2b))
        nutb = cp.where(self.nut_zg, self.nut[s.m.bcell], nutb)
        if self.has_nutk:
            nutb = cp.where(self.nutk, self.nutk_nut(s.nu), nutb)
        self.nutb = cp.where(self.spalding, self.spalding_nut(U, s.nu), nutb)

    def validate(self, U, g, gb):
        S2, S2b = _S2(g), _S2(gb)
        self.correct_nut(S2, S2b, U)

    def _scalar_matrix(self, phi, gam_c, gam_b, psi, bc, psib):
        """Assemble div(phi,psi) (bounded upwind) - laplacian(gamma, psi) (corrected). Returns pieces."""
        m = self.s.m
        nif = m.nif
        phi_i, phi_b = phi[:nif], phi[nif:]
        G = m.interp(gam_c) * m.magSf_i * m.nodc
        upper = cp.minimum(phi_i, 0.0) - G
        lower = -cp.maximum(phi_i, 0.0) - G
        D = -(m.Mown @ lower + m.Mnei @ upper) - m.Dsum @ phi
        vic, vbc, gic, gbc = bc.coeffs(m.dcb, phi_b)
        gm = gam_b * m.magSf_b
        iC = phi_b * vic - gm * gic
        bC = -phi_b * vbc + gm * gbc
        g, _ = self._grad_scalar(psi, psib)
        S = m.Dint @ (m.interp(gam_c) * m.magSf_i * (m.kvec * m.interp(g)).sum(1))
        return dict(D=D, upper=upper, lower=lower, iC=iC, bC=bC, S=S)

    def _solve(self, q, psi, alpha, fixed_cells=None, fixed_vals=None):
        """relax (fvMatrix::relax), optional setValues, solve with BiCGStab/Jacobi to relTol 0.05."""
        m = self.s.m
        D, upper, lower, iC, bC, S = q["D"], q["upper"], q["lower"], q["iC"], q["bC"], q["S"]
        sum_off = m.Mnei @ cp.abs(lower) + m.Mown @ cp.abs(upper)
        D0 = D
        D = D + m.Mb @ cp.abs(iC)
        D = cp.maximum(cp.abs(D), sum_off) / alpha
        D = D - m.Mb @ iC
        S = S + (D - D0) * psi
        if fixed_cells is not None:
            fc = self.wf_cell_mask
            fo_, fn_ = fc[m.own_i], fc[m.nei]
            vals = cp.zeros(m.nc); vals[fixed_cells] = fixed_vals
            S = S - m.Mnei @ cp.where(fo_ & ~fn_, lower * vals[m.own_i], 0.0) \
                  - m.Mown @ cp.where(fn_ & ~fo_, upper * vals[m.nei], 0.0)
            touch = fo_ | fn_
            upper = cp.where(touch, 0.0, upper)
            lower = cp.where(touch, 0.0, lower)
            fb = fc[m.bcell]
            iC = cp.where(fb, 0.0, iC); bC = cp.where(fb, 0.0, bC)
            S = cp.where(fc, vals * D, S)
            psi = cp.where(fc, vals, psi)
        A = m.set_off(upper, lower)
        Dc = D + m.Mb @ iC
        rhs = S + m.Mb @ bC

        def mv(x):
            return Dc[:, None] * x + A @ x
        x, res0, it = bicgstab(mv, rhs[:, None], psi[:, None], (1.0 / Dc)[:, None])
        # OpenFOAM-normalised initial residual for the log
        xr = psi.mean()
        Ax = Dc * psi + A @ psi
        Aref = (Dc + A @ cp.ones(m.nc)) * xr
        nrm = float(cp.abs(Ax - Aref).sum() + cp.abs(rhs - Aref).sum()) + 1e-20
        return x[:, 0], float(cp.abs(rhs - Ax).sum()) / nrm, it

    def _bound(self, c, cb, lower):
        """bound(): replace non-positive values by the face-average of max(field, lower), then clip."""
        m = self.s.m
        if float(cp.minimum(c.min(), cb.min())) >= lower:
            return c, cb
        cm, cbm = cp.maximum(c, lower), cp.maximum(cb, lower)
        fv = cp.concatenate([m.interp(cm) * m.magSf_i, cbm * m.magSf_b])
        avg = (m.Mown @ fv[:m.nif] + m.Mnei @ fv[:m.nif] + m.Mb @ fv[m.nif:]) / \
              (m.Mown @ m.magSf_i + m.Mnei @ m.magSf_i + m.Mb @ m.magSf_b)
        c = cp.maximum(cp.maximum(c, avg * (c <= 0)), lower)
        return c, cp.maximum(cb, lower)

    # ---- kOmegaSSTBase::correct() ------------------------------------------------------------
    def correct(self, U, phi, g, gb):
        _tick()
        s = self.s
        m = s.m
        nif = m.nif
        nu = s.nu
        phi_b = phi[nif:]
        divU = (m.Dsum @ phi) / m.V
        S2, S2b = _S2(g), _S2(gb)
        GbyNu0 = (g * (g + g.transpose(0, 2, 1))).sum((1, 2)) - (2.0 / 3.0) * cp.trace(g, axis1=1, axis2=2) ** 2
        G = self.nut * GbyNu0
        _tick("  turb: S2, GbyNu0")
        # omega wall function: omega and G in wall cells, omega on wall faces (updateCoeffs)
        Cmu25 = self.Cmu ** 0.25
        f = self.owf
        cells = m.bcell[f]
        y = self.y_wall[f]
        kc = self.k[cells]
        nuw = nu
        omegaVis = 6.0 * nuw / (self.omega_beta1 * y * y)
        omegaLog = cp.sqrt(kc) / (Cmu25 * self.kappa * y)
        w = self.wf_w[f]
        om0 = cp.zeros(m.nc); kernels_add(om0, cells, w * cp.sqrt(omegaVis ** 2 + omegaLog ** 2))
        magGradUw = self._mag_up(U)[f] * m.dcb[f]
        G0 = cp.zeros(m.nc)
        kernels_add(G0, cells, w * (self.nutb[f] + nuw) * magGradUw * Cmu25 * cp.sqrt(kc) / (self.kappa * y))
        wc = self.wf_cells
        self.omega[wc] = om0[wc]
        G[wc] = G0[wc]
        self.ob = cp.where(f, self.omega[m.bcell], self.ob)
        self.bo.ref = cp.where(f, self.omega[m.bcell], self.bo.ref)
        _tick("  turb: wall functions")
        # blending functions
        self.kb = self.bk.values(self.k, m.bcell, phi_b)
        self.ob = self.bo.values(self.omega, m.bcell, phi_b)
        gk, gkb = self._grad_scalar(self.k, self.kb)
        go, gob = self._grad_scalar(self.omega, self.ob)
        CD = (2.0 * self.alphaOmega2) * (gk * go).sum(1) / self.omega
        CDb = (2.0 * self.alphaOmega2) * (gkb * gob).sum(1) / self.ob
        F1 = self._F1(self.k, self.omega, self.y, CD)
        F23 = self._F23(self.k, self.omega, self.y)
        yb = self.y[m.bcell]
        F1b = cp.where(self.wall, 1.0, self._F1(self.kb, self.ob, yb, CDb))
        blend = lambda F, a, b: F * (a - b) + b
        gamma = blend(F1, self.gamma1, self.gamma2)
        beta = blend(F1, self.beta1, self.beta2)
        GbyNu = cp.minimum(GbyNu0, (self.c1 / self.a1) * self.betaStar * self.omega *
                           cp.maximum(self.a1 * self.omega, self.b1 * F23 * cp.sqrt(S2)))
        _tick("  turb: grads, F1, F23")
        # omega equation
        DoE = blend(F1, self.alphaOmega1, self.alphaOmega2) * self.nut + nu
        DoEb = blend(F1b, self.alphaOmega1, self.alphaOmega2) * self.nutb + nu
        q = self._scalar_matrix(phi, DoE, DoEb, self.omega, self.bo, self.ob)
        _tick("  turb: omega assembly")
        V = m.V
        x1 = (2.0 / 3.0) * gamma * divU
        x2 = (F1 - 1.0) * CD / self.omega
        q["D"] = q["D"] + V * cp.maximum(x1, 0) + V * beta * self.omega + V * cp.maximum(x2, 0)
        q["S"] = q["S"] + V * gamma * GbyNu - V * cp.minimum(x1, 0) * self.omega - V * cp.minimum(x2, 0) * self.omega
        om, res_o, it_o = self._solve(q, self.omega, self.relax_omega, wc, self.omega[wc])
        _tick("  turb: omega solve")
        self.omega = om
        self.ob = self.bo.values(self.omega, m.bcell, phi_b)
        self.omega, self.ob = self._bound(self.omega, self.ob, SMALL)
        # k equation
        DkE = blend(F1, self.alphaK1, self.alphaK2) * self.nut + nu
        DkEb = blend(F1b, self.alphaK1, self.alphaK2) * self.nutb + nu
        q = self._scalar_matrix(phi, DkE, DkEb, self.k, self.bk, self.kb)
        _tick("  turb: k assembly")
        Pk = cp.minimum(G, (self.c1 * self.betaStar) * self.k * self.omega)
        x1 = (2.0 / 3.0) * divU
        q["D"] = q["D"] + V * cp.maximum(x1, 0) + V * self.betaStar * self.omega
        q["S"] = q["S"] + V * Pk - V * cp.minimum(x1, 0) * self.k
        kk, res_k, it_k = self._solve(q, self.k, self.relax_k)
        _tick("  turb: k solve")
        self.k = kk
        self.kb = self.bk.values(self.k, m.bcell, phi_b)
        self.k, self.kb = self._bound(self.k, self.kb, SMALL)
        _tick("  turb: bound, misc")
        self.correct_nut(S2, S2b, U)
        _tick("  turb: nut")
        return dict(res_k=res_k, res_omega=res_o, it_k=it_k, it_omega=it_o)

    def write(self, time_name):
        s = self.s
        m = s.m
        a = cp.asnumpy
        pt = lambda vals, entries: [(p["name"], entries[i], a(vals[p["sl"]])) for i, p in enumerate(m.patches)]
        foamio.write_field(s.case, time_name, "k", "volScalarField", a(self.k), pt(self.kb, self.bk.entries))
        foamio.write_field(s.case, time_name, "omega", "volScalarField", a(self.omega), pt(self.ob, self.bo.entries))
        foamio.write_field(s.case, time_name, "nut", "volScalarField", a(self.nut), pt(self.nutb, self.nut_text))


def _S2(g):
    sym = 0.5 * (g + g.transpose(0, 2, 1))
    return 2.0 * (sym * sym).sum((1, 2))


def kernels_add(out, idx, vals):
    """Scatter-add (float64 atomics in CuPy are exact for addition)."""
    import cupyx
    cupyx.scatter_add(out, idx, vals)


def _entry_text(e):
    def ser(v):
        return "(" + " ".join(ser(x) for x in v) + ")" if isinstance(v, list) else str(v)
    return " ".join(f"{k} {' '.join(ser(x) for x in v)};" for k, v in e.items()
                    if k != "value" and not isinstance(v, dict) and not any(hasattr(x, "shape") for x in v))
