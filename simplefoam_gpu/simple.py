# SPDX-License-Identifier: MIT
"""GPU steady incompressible SIMPLE(C) solver mirroring OpenFOAM simpleFoam's discretisation.

Schemes reproduced: div(phi,U) bounded Gauss linearUpwindV grad(U); grad(U) cellLimited Gauss linear 1;
other gradients Gauss linear; laplacian Gauss linear corrected; consistent (SIMPLEC) pressure correction.
Usage:  simplefoam-gpu <case> [--iters N] [--start TIME] [--every N]   (or: python -m simplefoam_gpu <case> ...)
"""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx
from . import foamio
from .mesh import Mesh
from .amg import AMG, bicgstab
from . import kernels
from . import fused
from .check import check_case, function_objects, UnsupportedCase


REFERENCE = os.environ.get("SIMPLEFOAM_GPU_REFERENCE") == "1"   # use the plain array code instead of fused kernels


from .kernels import PROFILE, tick as _tick, profile_report


def _f(x):
    return float(x)


class Solver:
    def __init__(self, case, start="0", log=print):
        self.case = Path(case)
        self.log = log
        check_case(self.case, start)
        t0 = time.time()
        self.m = m = Mesh(case)
        log(f"mesh: {m.nc} cells, {m.nf} faces, {m.nb} boundary faces  ({time.time() - t0:.1f}s)")
        fs = foamio.read_dict(self.case / "system/fvSolution")
        rf = fs.get("relaxationFactors", {})
        self.alpha_p = foamio.scalar(rf.get("fields", {}).get("p", ["0.3"]))
        eq = rf.get("equations", {})
        self.alpha_U = _relax(eq, "U", 0.7)
        simple = fs.get("SIMPLE", {})
        self.n_nonorth = int(simple.get("nNonOrthogonalCorrectors", ["0"])[0])
        self.consistent = simple.get("consistent", ["no"])[0] in ("yes", "true", "on")
        self.nu = foamio.scalar(foamio.read_dict(self.case / "constant/transportProperties")["nu"][-1:])
        tp = foamio.read_dict(self.case / "constant/turbulenceProperties")
        rans = tp.get("simulationType", ["laminar"])[0] == "RAS"
        self._read_forces(rans)
        self._read_fields(start)
        self.amg = None
        self.it = 0
        self.turb = None
        if rans:
            model = tp["RAS"].get("RASModel", tp["RAS"].get("model", [""]))[0]
            if model != "kOmegaSST":
                raise NotImplementedError(f"RAS model {model}")
            from .turbulence import KOmegaSST
            t0 = time.time()
            self.turb = KOmegaSST(self, start, _relax(eq, "k", 1.0), _relax(eq, "omega", 1.0), log=log)
            Ub, _, _, _, _, sn = self.bc_U(self.U, self.phi[m.nif:])
            g, gb = self.limited_grad_U(self.U, Ub, sn)
            self.turb.validate(self.U, g, gb)
            log(f"turbulence: kOmegaSST with wall functions ({time.time() - t0:.1f}s)")

    # ---- setup ------------------------------------------------------------------------------
    def _read_forces(self, rans):
        """Set up the forceCoeffs function object, and warn about every function object that will not run."""
        self.forces = None
        force, warnings = function_objects(self.case, rans)
        for w in warnings:
            self.log(f"warning: {w}")
        if force is None:
            return
        fname, d = force
        import re
        try:
            pats = [p for p in foamio_flat(d["patches"])]
            mask = cp.zeros(self.m.nb, bool)
            for p in self.m.patches:
                if any(re.fullmatch(x, p["name"]) for x in pats):
                    mask[p["sl"]] = True
            drag = foamio.vector(d["dragDir"]); lift = foamio.vector(d["liftDir"])
            pitch = foamio.vector(d["pitchAxis"])
            self.forces = dict(mask=mask, rho=foamio.scalar(d["rhoInf"]), dragv=drag, liftv=lift, pitchv=pitch,
                               sidev=np.cross(lift, drag),
                               CofR=cp.asarray(foamio.vector(d["CofR"])) if "CofR" in d else cp.zeros(3),
                               drag=cp.asarray(drag), lift=cp.asarray(lift), pitch=cp.asarray(pitch),
                               side=cp.asarray(np.cross(lift, drag)),
                               Uinf=foamio.scalar(d["magUInf"]), Aref=foamio.scalar(d["Aref"]),
                               lRef=foamio.scalar(d["lRef"]), name=fname)
        except (ValueError, TypeError, IndexError, re.error) as e:
            self.forces = None
            self.log(f"warning: forceCoeffs '{fname}' settings could not be read ({e}); it will not run")

    def _read_fields(self, start):
        m = self.m
        nb = m.nb
        fU = foamio.read_field(self.case / start / "U")
        fp = foamio.read_field(self.case / start / "p")
        self.U = cp.asarray(foamio.expand(fU["internal"], m.nc, 3))
        self.p = cp.asarray(foamio.expand(fp["internal"], m.nc, 1))
        eye = cp.eye(3)
        nn = m.nb_hat[:, :, None] * m.nb_hat[:, None, :]
        self.vf_static = cp.zeros((nb, 3, 3)); self.vf_dyn = cp.zeros((nb, 3, 3))
        self.Uref = cp.zeros((nb, 3)); self.U_assignable = cp.zeros(nb, bool)
        self.p_fixed = cp.zeros(nb); self.p_ref = cp.zeros(nb)
        self.bc_text = {"U": [], "p": []}
        for pt in m.patches:
            sl, n = pt["sl"], pt["sl"].stop - pt["sl"].start
            e = foamio.patch_entry(fU["boundary"], pt["name"], pt["groups"]); t = e["type"][0]
            self.bc_text["U"].append(_entry_text(e))
            if t in ("fixedValue", "noSlip", "movingWallVelocity"):
                self.vf_static[sl] = eye
                if t != "noSlip":
                    self.Uref[sl] = cp.asarray(foamio.expand(e["value"], n, 3))
            elif t in ("slip", "symmetry", "symmetryPlane"):
                self.vf_static[sl] = nn[sl]
            elif t == "zeroGradient":
                self.U_assignable[sl] = True
            elif t == "pressureInletOutletVelocity":
                self.vf_dyn[sl] = eye - nn[sl]; self.U_assignable[sl] = True
            elif t == "inletOutlet":
                self.vf_dyn[sl] = eye; self.U_assignable[sl] = True
                self.Uref[sl] = cp.asarray(foamio.expand(e["inletValue"], n, 3))
            else:
                raise NotImplementedError(f"U boundary type {t}")
            e = foamio.patch_entry(fp["boundary"], pt["name"], pt["groups"]); t = e["type"][0]
            self.bc_text["p"].append(_entry_text(e))
            if t == "fixedValue":
                self.p_fixed[sl] = 1.0
                self.p_ref[sl] = cp.asarray(foamio.expand(e["value"], n, 1))
            elif t != "zeroGradient":
                raise NotImplementedError(f"p boundary type {t}")
        phi_file = self.case / start / "phi"
        if phi_file.exists():
            f = foamio.read_field(phi_file)
            phi_i = foamio.expand(f["internal"], m.nif, 1)
            phi_b = np.zeros(nb)
            for pt in m.patches:
                e = foamio.patch_entry(f["boundary"], pt["name"], pt["groups"])
                if "value" in e:
                    phi_b[pt["sl"]] = foamio.expand(e["value"], pt["sl"].stop - pt["sl"].start, 1)
            self.phi = cp.asarray(np.concatenate([phi_i, phi_b]))
        else:
            Ub = self.bc_U(self.U, cp.zeros(nb))[0]
            self.phi = cp.concatenate([(m.interp(self.U) * m.Sf_i).sum(1), (Ub * m.Sf_b).sum(1)])

    # ---- boundary conditions ----------------------------------------------------------------
    def bc_U(self, U, phi_b):
        """Boundary values and OpenFOAM matrix coefficients (value/gradient, internal/boundary)."""
        m = self.m
        Uc = U[m.bcell]
        vf = self.vf_static + (phi_b < 0)[:, None, None] * self.vf_dyn
        Ub = cp.einsum("fij,fj->fi", vf, self.Uref - Uc) + Uc
        td = cp.sqrt(cp.abs(cp.diagonal(vf, axis1=1, axis2=2)))
        dc = m.dcb[:, None]
        vic = 1.0 - td
        vbc = Ub - vic * Uc
        gic = -dc * td
        sn = (Ub - Uc) * dc
        gbc = sn - gic * Uc
        return Ub, vic, vbc, gic, gbc, sn

    def bc_p(self, p):
        pc = p[self.m.bcell]
        return self.p_fixed * self.p_ref + (1.0 - self.p_fixed) * pc

    # ---- discretisation pieces --------------------------------------------------------------
    def limited_grad_U(self, U, Ub, sn):
        """cellLimited Gauss linear 1 (minmod), plus the boundary-corrected patch gradient."""
        if REFERENCE:
            return self._limited_grad_U_ref(U, Ub, sn)
        m = self.m
        g = fused.grad_vec(m, U, Ub, limit=True)
        gc = g[m.bcell]
        n = m.nb_hat
        gb = gc + n[:, :, None] * (sn - cp.einsum("fk,fkj->fj", n, gc))[:, None, :]
        return g, gb

    def _limited_grad_U_ref(self, U, Ub, sn):
        m = self.m
        g = m.grad(U, Ub)
        mx = cp.ascontiguousarray(U).copy(); mn = cp.ascontiguousarray(U).copy()
        Uo, Un = U[m.own_i], U[m.nei]
        kernels.scatter_max(mx, m.own_i, Un); kernels.scatter_max(mx, m.nei, Uo); kernels.scatter_max(mx, m.bcell, Ub)
        kernels.scatter_min(mn, m.own_i, Un); kernels.scatter_min(mn, m.nei, Uo); kernels.scatter_min(mn, m.bcell, Ub)
        mx -= U; mn -= U
        lim = cp.ones_like(U)
        for d, cells in ((m.dCfo, m.own_i), (m.dCfn, m.nei), (m.dCfb, m.bcell)):
            e = cp.einsum("fi,fij->fj", d, g[cells])
            a, b = mx[cells], mn[cells]
            safe = cp.where(e == 0, 1.0, e)
            r = cp.where(e > a, a / safe, cp.where(e < b, b / safe, 1.0))
            kernels.scatter_min(lim, cells, r)
        g *= lim[:, None, :]
        gc = g[m.bcell]
        n = m.nb_hat
        gb = gc + n[:, :, None] * (sn - cp.einsum("fk,fkj->fj", n, gc))[:, None, :]
        return g, gb

    def momentum(self, U, p, phi, nu_c, nu_b):
        """Assemble the relaxed momentum matrix. Returns a dict with everything the p-equation needs."""
        m = self.m
        nif = m.nif
        phi_i, phi_b = phi[:nif], phi[nif:]
        Ub, vic, vbc, gic, gbc, sn = self.bc_U(U, phi_b)
        g, gb = self.limited_grad_U(U, Ub, sn)
        nu_i = m.interp(nu_c)
        Gd = nu_i * m.magSf_i * m.nodc
        upper = cp.minimum(phi_i, 0.0) - Gd       # coefficient of the neighbour in the owner's row
        lower = -cp.maximum(phi_i, 0.0) - Gd      # coefficient of the owner in the neighbour's row
        D = -(m.Mown @ lower + m.Mnei @ upper) - m.Dsum @ phi
        sum_off = m.Mnei @ cp.abs(lower) + m.Mown @ cp.abs(upper)
        H1 = -(m.Mnei @ lower + m.Mown @ upper)
        gm = (nu_b * m.magSf_b)[:, None]
        iC = phi_b[:, None] * vic - gm * gic
        bC = -phi_b[:, None] * vbc + gm * gbc
        if not REFERENCE:
            F = fused.momentum_flux(m, U, g, nu_c, phi_i)
            S = m.Dint @ F
            del F
            Ab = nu_b[:, None, None] * (gb.transpose(0, 2, 1) - (2.0 / 3.0) * cp.trace(gb, axis1=1, axis2=2)[:, None, None] * cp.eye(3))
            S += m.Mb @ cp.einsum("fi,fij->fj", m.Sf_b, Ab)
        else:
            S = self._momentum_explicit_ref(U, g, gb, phi_i, nu_c, nu_b, nu_i)
        # relaxation (fvMatrix::relax)
        D0 = D
        D = D + m.Mb @ cp.abs(iC).max(1)
        D = cp.maximum(cp.abs(D), sum_off) / self.alpha_U
        D = D - m.Mb @ iC.min(1)
        S += (D - D0)[:, None] * U
        return dict(D=D, S=S, upper=upper, lower=lower, H1=H1, bdiag=m.Mb @ iC, bsrc=m.Mb @ bC, gb=gb)

    def _momentum_explicit_ref(self, U, g, gb, phi_i, nu_c, nu_b, nu_i):
        """Reference (array) version of the explicit momentum terms; see fused.momentum_flux."""
        m = self.m
        # explicit linearUpwindV correction
        Uo, Un = U[m.own_i], U[m.nei]
        pos = (phi_i > 0)[:, None]
        w = m.w[:, None]
        dU = Un - Uo
        max_corr = cp.where(pos, (1.0 - w) * dU, -w * dU)
        corr = cp.where(pos, cp.einsum("fi,fij->fj", m.dCfo, g[m.own_i]), cp.einsum("fi,fij->fj", m.dCfn, g[m.nei]))
        s2 = (corr * corr).sum(1, keepdims=True)
        mc = (corr * max_corr).sum(1, keepdims=True)
        corr = cp.where(s2 > 0, cp.where(mc < 0, 0.0, cp.where(s2 > mc, corr * (mc / (s2 + 1e-300)), corr)), corr)
        S = -(m.Dint @ (phi_i[:, None] * corr))
        # non-orthogonal correction of the viscous laplacian
        gf = m.interp(g)
        S += m.Dint @ ((nu_i * m.magSf_i)[:, None] * cp.einsum("fi,fij->fj", m.kvec, gf))
        # explicit div(nuEff*dev2(T(grad(U)))): the product is interpolated to the faces
        eye = cp.eye(3)
        Ac = nu_c[:, None, None] * (g.transpose(0, 2, 1) - (2.0 / 3.0) * cp.trace(g, axis1=1, axis2=2)[:, None, None] * eye)
        Ab = nu_b[:, None, None] * (gb.transpose(0, 2, 1) - (2.0 / 3.0) * cp.trace(gb, axis1=1, axis2=2)[:, None, None] * eye)
        Af = m.interp(Ac)
        del Ac
        S += m.Dsum @ cp.concatenate([cp.einsum("fi,fij->fj", m.Sf_i, Af), cp.einsum("fi,fij->fj", m.Sf_b, Ab)])
        del Af
        return S

    def force_coeffs(self, p, gb, nu_b):
        """forceCoeffs: pressure + viscous force and moments about CofR on the selected patches.
        Columns follow OpenFOAM v2412 (front/rear split: 0.5*C +/- Cm about the matching axis)."""
        f = self.forces
        m = self.m
        pb = self.bc_p(p)
        fp = f["rho"] * m.Sf_b * (pb * f["mask"])[:, None]
        tau = gb + gb.transpose(0, 2, 1)
        tr = cp.trace(gb, axis1=1, axis2=2)
        tau[:, 0, 0] -= (2.0 / 3.0) * tr; tau[:, 1, 1] -= (2.0 / 3.0) * tr; tau[:, 2, 2] -= (2.0 / 3.0) * tr
        fv = cp.einsum("fi,fij->fj", m.Sf_b, tau) * (-f["rho"] * nu_b * f["mask"])[:, None]
        r = m.Cf_b - f["CofR"]
        Fp, Fv = kernels.colsum(fp), kernels.colsum(fv)
        F = Fp + Fv
        M = kernels.colsum(cp.cross(r, fp + fv))
        q = 0.5 * f["rho"] * f["Uinf"] ** 2 * f["Aref"]
        Cd, Cl, Cs = _f(F @ f["drag"]) / q, _f(F @ f["lift"]) / q, _f(F @ f["side"]) / q
        Cmr, Cmp, Cmy = (_f(M @ f[a]) / (q * f["lRef"]) for a in ("drag", "pitch", "lift"))
        return dict(Cd=Cd, Cl=Cl, Cd_p=_f(Fp @ f["drag"]) / q, Cd_v=_f(Fv @ f["drag"]) / q,
                    coeffs=[Cd, 0.5 * Cd + Cmr, 0.5 * Cd - Cmr, Cl, 0.5 * Cl + Cmp, 0.5 * Cl - Cmp,
                            Cmp, Cmr, Cmy, Cs, 0.5 * Cs + Cmy, 0.5 * Cs - Cmy])

    # ---- one SIMPLE iteration ---------------------------------------------------------------
    def step(self):
        m = self.m
        nif = m.nif
        U, p, phi = self.U, self.p, self.phi
        if self.turb is not None:
            nu_c, nu_b = self.turb.nu_eff()
        else:
            nu_c, nu_b = cp.full(m.nc, self.nu), cp.full(m.nb, self.nu)
        p_old = p
        _tick()
        q = self.momentum(U, p, phi, nu_c, nu_b)
        _tick("momentum assembly")
        D, S, bdiag, bsrc = q["D"], q["S"], q["bdiag"], q["bsrc"]
        Aoff = m.set_off(q["upper"], q["lower"])
        pb = self.bc_p(p)
        gradp = m.grad(p, pb)
        rhs = S + bsrc - gradp * m.V[:, None]
        Dc = D[:, None] + bdiag

        def mv(x):
            return Dc * x + Aoff @ x
        AU = mv(U)
        r0 = rhs - AU
        xref = kernels.colsum(U) / m.nc
        Aref = (Dc + (Aoff @ cp.ones(m.nc))[:, None]) * xref
        normU = kernels.colsum(cp.abs(AU - Aref)) + kernels.colsum(cp.abs(rhs - Aref)) + 1e-20
        resU = kernels.colsum(cp.abs(r0)) / normU
        del AU, Aref
        _tick("momentum misc")
        U, _, itU = bicgstab(mv, rhs, U, 1.0 / Dc, r=r0)
        _tick("momentum solve")
        Ub = self.bc_U(U, phi[nif:])[0]
        # pressure equation
        bav = bdiag.mean(1)
        rAU = m.V / (D + bav)
        H = ((bav[:, None] - bdiag) * U - Aoff @ U + S + bsrc) / m.V[:, None]
        HbyA = rAU[:, None] * H
        HbyA_b = cp.where(self.U_assignable[:, None], HbyA[m.bcell], Ub)
        phiH_i = (m.interp(HbyA) * m.Sf_i).sum(1)
        phiH_b = (HbyA_b * m.Sf_b).sum(1)
        rAtU = rAU
        if self.consistent:
            rAtU = 1.0 / (1.0 / rAU - q["H1"] / m.V)
            d = rAtU - rAU
            sn_i = m.nodc * (p[m.nei] - p[m.own_i]) + (m.kvec * m.interp(gradp)).sum(1)
            sn_b = self.p_fixed * m.dcb * (pb - p[m.bcell])
            phiH_i = phiH_i + m.interp(d) * sn_i * m.magSf_i
            phiH_b = phiH_b + d[m.bcell] * sn_b * m.magSf_b
            HbyA = HbyA - (rAU - rAtU)[:, None] * gradp
        gam = m.interp(rAtU) * m.magSf_i
        G = gam * m.nodc
        Gb = rAtU[m.bcell] * m.magSf_b * m.dcb * self.p_fixed
        diag = m.Mown @ G + m.Mnei @ G + m.Mb @ Gb
        data = cp.concatenate([-G, -G, diag])[m.full_perm]
        _tick("pressure assembly")
        if self.amg is None:
            t0 = time.time()
            self.amg = AMG(m.full_indices, m.full_indptr, cp.asnumpy(data))
            self.log(f"multigrid levels: {self.amg.sizes}  ({time.time() - t0:.1f}s)")
        else:
            self.amg.update(data)
        _tick("multigrid setup/update")
        divH = m.Dsum @ cp.concatenate([phiH_i, phiH_b])
        bfix = m.Mb @ (Gb * self.p_ref)
        resP = None
        itP = 0
        for _ in range(self.n_nonorth + 1):
            corr = gam * (m.kvec * m.interp(gradp)).sum(1)
            p, r0p, _, n_it = self.amg.solve(-divH + m.Dint @ corr + bfix, p)
            itP += n_it
            resP = r0p if resP is None else resP
            pb = self.bc_p(p)
            gradp = m.grad(p, pb)
        _tick("pressure solve")
        phi = cp.concatenate([phiH_i - (G * (p[m.nei] - p[m.own_i]) + corr), phiH_b - Gb * (pb - p[m.bcell])])
        cont = _f(cp.abs(m.Dsum @ phi).sum() / m.V.sum())
        p = p_old + self.alpha_p * (p - p_old)
        gradp = m.grad(p, self.bc_p(p))
        U = HbyA - rAtU[:, None] * gradp
        self.U, self.p, self.phi = U, p, phi
        self.it += 1
        out = dict(it=self.it, resU=[_f(x) for x in resU], resP=resP, itU=itU, itP=itP, cont=cont)
        del q
        Ub, _, _, _, _, sn = self.bc_U(U, phi[nif:])
        g, gb = self.limited_grad_U(U, Ub, sn)
        _tick("correctors + grad(U)")
        if self.turb is not None:
            out.update(self.turb.correct(U, phi, g, gb))
            nu_b = self.turb.nu_eff()[1]
        if self.forces is not None:
            out.update(self.force_coeffs(p, gb, nu_b))
            _tick("forces")
        return out

    def write(self, time_name):
        m = self.m
        nif = m.nif
        Ub = self.bc_U(self.U, self.phi[nif:])[0]
        pb = self.bc_p(self.p)
        a = cp.asnumpy
        def patches(vals, kind):
            txt = self.bc_text.get(kind) or ["type calculated;"] * len(m.patches)
            return [(pt["name"], txt[i], a(vals[pt["sl"]])) for i, pt in enumerate(m.patches)]
        foamio.write_field(self.case, time_name, "U", "volVectorField", a(self.U), patches(Ub, "U"))
        foamio.write_field(self.case, time_name, "p", "volScalarField", a(self.p), patches(pb, "p"))
        foamio.write_field(self.case, time_name, "phi", "surfaceScalarField", a(self.phi[:nif]),
                           patches(self.phi[nif:], "phi"))
        if self.turb is not None:
            self.turb.write(time_name)


def _entry_text(e):
    """Boundary entry as OpenFOAM text, without its value (written separately)."""
    def ser(v):
        return "(" + " ".join(ser(x) for x in v) + ")" if isinstance(v, list) else str(v)
    return " ".join(f"{k} {' '.join(ser(x) for x in v)};" for k, v in e.items()
                    if k != "value" and not isinstance(v, dict) and not any(hasattr(x, "shape") for x in v))


def _relax(eq, name, default):
    """Relaxation factor for an equation, honouring regex keys such as "(k|omega)"."""
    import re
    if name in eq:
        return foamio.scalar(eq[name])
    for key, v in eq.items():
        try:
            if re.fullmatch(key, name):
                return foamio.scalar(v)
        except re.error:
            pass
    return default


def _bc(x):
    return x if np.isscalar(x) else x[:, None]


def foamio_flat(v):
    out = []
    for x in v:
        out.extend(foamio_flat(x) if isinstance(x, list) else [x])
    return out


class Outputs:
    """controlDict-driven writing: time directories (writeInterval/purgeWrite), forceCoeffs and yPlus files."""

    def __init__(self, solver, start_name):
        self.s = solver
        self.start = start_name
        self.written = []
        cd = foamio.read_dict(solver.case / "system/controlDict")
        self.purge = int(cd.get("purgeWrite", ["0"])[0])
        self.coef = None
        if solver.forces is not None:
            f = solver.forces
            d = solver.case / "postProcessing" / f["name"] / start_name
            d.mkdir(parents=True, exist_ok=True)
            self.coef = open(d / "coefficient.dat", "w")
            v = lambda x: "(" + " ".join(f"{c:.10e}" for c in x) + ")"
            hdr = [("dragDir", v(f["dragv"])), ("sideDir", v(f["sidev"])), ("liftDir", v(f["liftv"])),
                   ("rollAxis", v(f["dragv"])), ("pitchAxis", v(f["pitchv"])), ("yawAxis", v(f["liftv"])),
                   ("magUInf", f"{f['Uinf']:.10e}"), ("lRef", f"{f['lRef']:.10e}"), ("Aref", f"{f['Aref']:.10e}"),
                   ("CofR", v(cp.asnumpy(f["CofR"])))]
            self.coef.write("# Force and moment coefficients\n" + "".join(f"# {k:<16s}: {x}\n" for k, x in hdr) + "#\n")
            cols = ["Cd", "Cd(f)", "Cd(r)", "Cl", "Cl(f)", "Cl(r)", "CmPitch", "CmRoll", "CmYaw", "Cs", "Cs(f)", "Cs(r)"]
            self.coef.write("# " + "\t".join(f"{c:<18s}" for c in ["Time"] + cols) + "\n")
        self.yp = None
        if solver.turb is not None:
            d = solver.case / "postProcessing/yPlus" / start_name
            d.mkdir(parents=True, exist_ok=True)
            self.yp = open(d / "yPlus.dat", "w")
            self.yp.write("# y+ ()\n# " + "\t".join(f"{c:<18s}" for c in ("Time", "patch", "min", "max", "average")) + "\n")

    def iteration(self, time_name, out):
        if self.coef is not None and "coeffs" in out:
            self.coef.write(f"{time_name:<18s}\t" + "\t".join(f"{c:.10e}" for c in out["coeffs"]) + "\n")
            self.coef.flush()

    def write(self, time_name):
        s = self.s
        m = s.m
        s.write(time_name)
        if s.turb is not None:
            yp = s.turb.yplus(s.U)
            foamio.write_field(s.case, time_name, "yPlus", "volScalarField", np.zeros(m.nc),
                               [(p["name"], "type calculated;", cp.asnumpy(yp[p["sl"]])) for p in m.patches])
            for p in m.patches:
                if p["type"] == "wall":
                    v = yp[p["sl"]]
                    self.yp.write(f"{time_name:<18s}\t{p['name']}\t{float(v.min()):.10e}\t{float(v.max()):.10e}\t{float(v.mean()):.10e}\n")
            self.yp.flush()
        self.written.append(time_name)
        if self.purge > 0:
            import shutil
            while len(self.written) > self.purge:
                shutil.rmtree(s.case / self.written.pop(0), ignore_errors=True)


def _time_name(t):
    return str(int(t)) if float(t) == int(t) else repr(float(t))


def main():
    ap = argparse.ArgumentParser(description="GPU steady incompressible solver (simpleFoam equivalent).")
    ap.add_argument("case")
    ap.add_argument("--start", default=None, help="start time (default: controlDict startFrom/startTime; 'latestTime')")
    ap.add_argument("--iters", type=int, default=None, help="number of iterations (default: up to controlDict endTime)")
    ap.add_argument("--write", default=None, help="only write this time name at the end (benchmarks)")
    ap.add_argument("--every", type=int, default=10, help="print progress every N iterations")
    ap.add_argument("--no-output", action="store_true", help="no time directories or postProcessing files")
    a = ap.parse_args()
    case = Path(a.case)
    try:
        cd = foamio.read_dict(case / "system/controlDict")
    except foamio.FoamSyntaxError as e:
        raise SystemExit(f"simplefoam-gpu: {e}")
    start = a.start
    if start is None:
        start = cd.get("startTime", ["0"])[0] if cd.get("startFrom", ["startTime"])[0] == "startTime" else "latestTime"
    if start == "latestTime":
        times = [d.name for d in case.iterdir() if d.is_dir() and _is_number(d.name) and (d / "U").exists()]
        start = max(times, key=float)
    t_start = float(start)
    end = float(cd.get("endTime", [str(t_start + 1)])[0])
    n_iter = a.iters if a.iters is not None else int(round(end - t_start))
    write_every = int(cd.get("writeInterval", ["0"])[0]) if cd.get("writeControl", ["timeStep"])[0] == "timeStep" else 0
    try:
        s = Solver(case, start)
    except (UnsupportedCase, foamio.FoamSyntaxError) as e:
        raise SystemExit(f"simplefoam-gpu: {e}")
    out_w = None if (a.no_output or a.write) else Outputs(s, _time_name(t_start))
    print(f"start time {start}, {n_iter} iterations, writing every {write_every}", flush=True)
    cp.cuda.Device().synchronize()
    t0 = time.time()
    t1 = t0
    for i in range(n_iter):
        o = s.step()
        tn = _time_name(t_start + o["it"])
        if out_w is not None:
            out_w.iteration(tn, o)
        if i == 0:
            cp.cuda.Device().synchronize(); t1 = time.time()
        last = i == n_iter - 1
        stop = False
        if out_w is not None and o["it"] % 25 == 0:
            stop = foamio.read_dict(case / "system/controlDict").get("stopAt", ["endTime"])[0] == "writeNow"
        if o["it"] % a.every == 0 or last or i < 3 or stop:
            print(f"it {o['it']:5d} (time {tn})  U {o['resU'][0]:.2e} {o['resU'][1]:.2e} {o['resU'][2]:.2e} ({o['itU']})"
                  f"  p {o['resP']:.2e} ({o['itP']})  cont {o['cont']:.2e}"
                  + (f"  k {o['res_k']:.2e} om {o['res_omega']:.2e}" if "res_k" in o else "")
                  + (f"  Cd {o['Cd']:.6f} (p {o['Cd_p']:.6f} v {o['Cd_v']:.6f}) Cl {o['Cl']:.2e}" if "Cd" in o else "")
                  + f"  [{time.time() - t0:.0f} s]", flush=True)
        if out_w is not None and (last or stop or (write_every and (t_start + o["it"]) % write_every == 0)):
            out_w.write(tn)
            print(f"wrote time {tn}", flush=True)
        if stop:
            print("stopAt writeNow requested in controlDict; stopping", flush=True)
            break
    cp.cuda.Device().synchronize()
    if s.it > 1:
        print(f"time per iteration: {(time.time() - t1) / (s.it - 1):.3f} s  (first: {t1 - t0:.1f} s)")
    if a.write:
        s.write(a.write)
    if PROFILE:
        print("time by stage (all iterations):\n" + profile_report())
    print("End", flush=True)


def _is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
