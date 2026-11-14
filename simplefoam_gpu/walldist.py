# SPDX-License-Identifier: MIT
"""Wall distances as OpenFOAM computes them for RAS models.

near_wall_dist: per wall face, the smallest distance from its cell centre to the wall faces that share a point with it
                (OpenFOAM nearWallDist, used by wall functions).
cell_wall_dist: per cell, distance to the nearest wall (OpenFOAM wallDist meshWave with correctWalls): cells touching a
                wall point get the exact distance to the wall faces using those points; all other cells get the
                distance to the nearest wall face centre.
Polygon distance follows face::nearestPoint: triangles fan around the face centre (direct triangle for 3-point faces).
"""
import numpy as np
import cupy as cp
import cupyx
from . import kernels
from scipy.spatial import cKDTree


def _tri_dist(p, a, b, c):
    """Distance from points p to triangles (a, b, c); all (n, 3) cupy arrays (closest-point regions, Ericson)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = p - b
    d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cpp = p - c
    d5 = (ab * cpp).sum(1); d6 = (ac * cpp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    sd = lambda x, y: x / cp.where(y == 0, 1e-300, y)
    v_ab = sd(d1, d1 - d3)
    w_ac = sd(d2, d2 - d6)
    w_bc = sd(d4 - d3, (d4 - d3) + (d5 - d6))
    den = va + vb + vc
    v_in = sd(vb, den); w_in = sd(vc, den)
    q = a + ab * v_in[:, None] + ac * w_in[:, None]                       # inside
    q = cp.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[:, None], b + (c - b) * w_bc[:, None], q)
    q = cp.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], a + ac * w_ac[:, None], q)
    q = cp.where(((d6 >= 0) & (d5 <= d6))[:, None], c, q)
    q = cp.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], a + ab * v_ab[:, None], q)
    q = cp.where(((d3 >= 0) & (d4 <= d3))[:, None], b, q)
    q = cp.where(((d1 <= 0) & (d2 <= 0))[:, None], a, q)
    return cp.linalg.norm(p - q, axis=1)


class _WallFaces:
    def __init__(self, raw, Cf, wall_faces):
        pts, fl, fo = raw["points"], raw["face_labels"], raw["face_offsets"]
        self.faces = wall_faces                               # global face ids
        n = np.diff(fo)[wall_faces]
        start = fo[wall_faces]
        self.npts = n
        idx = np.repeat(start, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        self.labels = fl[idx]                                 # flat point labels, face-major
        self.off = np.zeros(len(wall_faces) + 1, np.int64); np.cumsum(n, out=self.off[1:])
        self.pts = pts
        self.Cf = Cf[wall_faces]
        # triangles: fan around the face centre (for a 3-point face the fan covers the same triangle, so the
        # distance equals OpenFOAM's direct triangle test)
        tri_face = np.repeat(np.arange(len(wall_faces)), n)
        nxt = np.arange(len(self.labels)) + 1
        nxt[self.off[1:] - 1] = self.off[:-1]
        self.tri_face = tri_face
        self.A = cp.asarray(pts[self.labels]); self.B = cp.asarray(pts[self.labels[nxt]])
        self.C = cp.asarray(self.Cf[tri_face])
        ntri = np.bincount(self.tri_face, minlength=len(wall_faces))
        self.tri_off = np.zeros(len(wall_faces) + 1, np.int64); np.cumsum(ntri, out=self.tri_off[1:])
        # point -> wall faces
        order = np.argsort(self.labels, kind="stable")
        self.pf_points = self.labels[order]
        self.pf_faces = np.repeat(np.arange(len(wall_faces)), n)[order]
        up, first_idx = np.unique(self.pf_points, return_index=True)
        self.wall_points = up
        self.pf_start = np.r_[first_idx, len(order)]

    def faces_of_points(self, point_ids):
        """For an array of wall point ids, return (query index, wall face) pairs."""
        k = np.searchsorted(self.wall_points, point_ids)
        s, e = self.pf_start[k], self.pf_start[k + 1]
        cnt = e - s
        q = np.repeat(np.arange(len(point_ids)), cnt)
        f = self.pf_faces[np.repeat(s, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
        return q, f

    def min_dist(self, P, q, f, nq, chunk=4_000_000):
        """Smallest distance from points P[q] to wall faces f, reduced per query index (nq queries)."""
        key = np.unique(q.astype(np.int64) * len(self.faces) + f)
        q, f = key // len(self.faces), key % len(self.faces)
        ntri = self.tri_off[f + 1] - self.tri_off[f]
        out = cp.full(nq, cp.inf)
        Pg = cp.asarray(P)
        # process pairs in chunks of roughly `chunk` triangles
        csum = np.cumsum(ntri)
        bounds = np.searchsorted(csum, np.arange(chunk, csum[-1] + chunk, chunk)) + 1
        lo = 0
        for hi in np.r_[bounds, len(q)]:
            hi = min(int(hi), len(q))
            if hi <= lo:
                continue
            qq, ff, nt = q[lo:hi], f[lo:hi], ntri[lo:hi]
            tq = np.repeat(qq, nt)
            ti = np.repeat(self.tri_off[ff], nt) + (np.arange(nt.sum()) - np.repeat(np.cumsum(nt) - nt, nt))
            tq_g, ti_g = cp.asarray(tq), cp.asarray(ti)
            d = _tri_dist(Pg[tq_g], self.A[ti_g], self.B[ti_g], self.C[ti_g])
            kernels.scatter_min(out, tq_g, d)
            lo = hi
        return out


def _mesh_wave(geom, wall_faces, tol=0.01):
    """Jacobi version of OpenFOAM's FaceCellWave with wallPoint data: every face and cell carries the wall face
    centre ("origin") it has learned of; a neighbour's origin is taken when it is closer by more than the relative
    tolerance used by wallPoint (distSqr < (1 - tol)*current)."""
    nif, nc, nf = geom["nif"], geom["nc"], geom["nf"]
    own = cp.asarray(geom["own"]); nei = cp.asarray(geom["nei"])
    Cc = cp.asarray(geom["C"]); Cf = cp.asarray(geom["Cf"])
    f_org = cp.full((nf, 3), cp.nan); f_d2 = cp.full(nf, cp.inf)
    wf = cp.asarray(wall_faces)
    f_org[wf] = Cf[wf]; f_d2[wf] = 0.0
    c_org = cp.full((nc, 3), cp.nan); c_d2 = cp.full(nc, cp.inf)
    fcell = cp.concatenate([own, nei]); fface = cp.concatenate([cp.arange(nf), cp.arange(nif)])
    for it in range(100000):
        # faces -> cells
        valid = cp.isfinite(f_d2[fface])
        d2 = cp.where(valid, ((Cc[fcell] - f_org[fface]) ** 2).sum(1), cp.inf)
        best = cp.full(nc, cp.inf); kernels.scatter_min(best, fcell, d2)
        take = (d2 == best[fcell]) & (best[fcell] < (1 - tol) * c_d2[fcell])
        ch_c = int(take.sum())
        c_org[fcell[take]] = f_org[fface[take]]; c_d2[fcell[take]] = d2[take]
        # cells -> faces (internal faces from both sides, boundary faces from the owner)
        d2 = cp.where(cp.isfinite(c_d2[fcell]), ((Cf[fface] - c_org[fcell]) ** 2).sum(1), cp.inf)
        best = cp.full(nf, cp.inf); kernels.scatter_min(best, fface, d2)
        take = (d2 == best[fface]) & (best[fface] < (1 - tol) * f_d2[fface])
        ch_f = int(take.sum())
        f_org[fface[take]] = c_org[fcell[take]]; f_d2[fface[take]] = d2[take]
        if ch_c == 0 and ch_f == 0:
            break
    return cp.sqrt(c_d2)


def compute(raw, geom):
    """Return cupy arrays (y_cell (nc,), y_b (nb,), wall_b (nb,) bool).

    y_b holds the near-wall distance on wall faces and 0 elsewhere."""
    nif = geom["nif"]; nc = geom["nc"]; nf = geom["nf"]
    wall_b = np.zeros(nf - nif, bool)
    for p in geom["patches"]:
        if p["type"] == "wall":
            s = p["startFace"] - nif
            wall_b[s:s + p["nFaces"]] = True
    y_b = cp.zeros(nf - nif)
    if not wall_b.any():
        return cp.full(nc, 1e15), y_b, cp.asarray(wall_b)
    wall_faces = np.nonzero(wall_b)[0] + nif
    W = _WallFaces(raw, geom["Cf"], wall_faces)
    C = geom["C"]
    # near-wall distance per wall face: neighbours = wall faces sharing a point with the face
    fq = np.repeat(np.arange(len(wall_faces)), W.npts)          # (wall face, its point)
    q2, f2 = W.faces_of_points(W.labels)
    qf = fq[q2]
    cells = geom["own"][wall_faces]
    ynw = W.min_dist(C[cells], qf, f2, len(wall_faces))
    y_b[cp.asarray(np.nonzero(wall_b)[0])] = ynw
    # cell distance: meshWave (wall face centres propagated face -> cell -> face), corrected for cells touching
    # wall points
    y = _mesh_wave(geom, wall_faces)
    fl, fo = raw["face_labels"], raw["face_offsets"]
    on_wall_pt = np.isin(fl, W.wall_points)
    fidx = np.repeat(np.arange(nf), np.diff(fo))[on_wall_pt]
    pid = fl[on_wall_pt]
    own, nei = geom["own"], geom["nei"]
    cell = np.concatenate([own[fidx], nei[fidx[fidx < nif]]])
    pt = np.concatenate([pid, pid[fidx < nif]])
    key = np.unique(cell.astype(np.int64) * (pid.max() + 1) + pt)
    cell, pt = key // (pid.max() + 1), key % (pid.max() + 1)
    q3, f3 = W.faces_of_points(pt)
    ucells, inv = np.unique(cell, return_inverse=True)
    dmin = W.min_dist(C[ucells], inv[q3], f3, len(ucells))
    y[cp.asarray(ucells)] = dmin
    return y, y_b, cp.asarray(wall_b)
