# SPDX-License-Identifier: MIT
"""Compare the fields of two OpenFOAM cases cell by cell (same mesh).

Usage:  python -m simplefoam_gpu.compare <caseA> <timeA> <caseB> <timeB>
"""
import sys
import numpy as np
from . import foamio
from .mesh import geometry


def main(argv=None):
    a, ta, b, tb = (argv or sys.argv[1:])[:4]
    nc = geometry(foamio.read_mesh(a))["nc"]
    for name, ncomp in (("U", 3), ("p", 1), ("k", 1), ("omega", 1), ("nut", 1)):
        try:
            A = foamio.expand(foamio.read_field(f"{a}/{ta}/{name}")["internal"], nc, ncomp)
            B = foamio.expand(foamio.read_field(f"{b}/{tb}/{name}")["internal"], nc, ncomp)
        except FileNotFoundError:
            continue
        d = np.abs(A - B).reshape(nc, -1).max(1)
        scale = np.abs(A).reshape(nc, -1).max()
        print(f"{name:6s} max |A-B| / max|A| = {d.max() / scale:.2e}   mean = {d.mean() / scale:.2e}")


if __name__ == "__main__":
    main()
