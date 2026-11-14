# SPDX-License-Identifier: MIT
"""Minimal reader/writer for OpenFOAM files (ascii and binary, label=32 scalar=64)."""
import re
from pathlib import Path
import numpy as np

NCOMP = {"scalar": 1, "label": 1, "vector": 3, "symmTensor": 6, "tensor": 9}
_PUNCT = b"{}();"
_WS = b" \t\r\n"


class Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.i = 0
        self.binary = False

    def skip(self):
        d, n = self.d, len(self.d)
        while self.i < n:
            c = d[self.i]
            if c in _WS:
                self.i += 1
            elif d.startswith(b"//", self.i):
                j = d.find(b"\n", self.i)
                self.i = n if j < 0 else j + 1
            elif d.startswith(b"/*", self.i):
                j = d.find(b"*/", self.i)
                self.i = n if j < 0 else j + 2
            else:
                break

    def tok(self):
        self.skip()
        d, n = self.d, len(self.d)
        if self.i >= n:
            return None
        c = d[self.i]
        if c in _PUNCT:
            self.i += 1
            return chr(c)
        if c == 0x22:  # quoted string
            j = d.find(b'"', self.i + 1)
            s = d[self.i + 1:j].decode()
            self.i = j + 1
            return s
        j = self.i
        while j < n and d[j] not in _WS and d[j] not in _PUNCT:
            j += 1
        s = d[self.i:j].decode("latin1")
        self.i = j
        return s

    def peek(self):
        save = self.i
        t = self.tok()
        self.i = save
        return t

    def expect(self, ch):
        t = self.tok()
        if t != ch:
            raise ValueError(f"expected {ch!r}, got {t!r} at byte {self.i}")

    def read_array(self, typ, n):
        """Read '( ... )' holding n items of the given primitive type."""
        nc = NCOMP[typ]
        dt = np.int32 if typ == "label" else np.float64
        self.skip()
        if n == 0:
            if self.peek() == "(":
                self.expect("(")
                self.expect(")")
            return np.zeros((0, nc) if nc > 1 else 0, dt)
        if self.d[self.i:self.i + 1] == b"{":  # uniform list N{value}
            self.expect("{")
            vals = self._generic_until("}")
            a = np.array(_flatten(vals), dt).reshape(-1)
            a = np.tile(a, (n, 1)) if nc > 1 else np.full(n, a[0], dt)
            return a
        self.expect("(")
        if self.binary:
            nb = n * nc * dt().itemsize
            a = np.frombuffer(self.d, dt, n * nc, self.i).copy()
            self.i += nb
            self.expect(")")
        else:
            if nc == 1:
                j = self.d.find(b")", self.i)
                txt = self.d[self.i:j]
                self.i = j + 1
            else:
                m = re.compile(rb"\)\s*\)").search(self.d, self.i)
                txt = self.d[self.i:m.end() - 1].replace(b"(", b" ").replace(b")", b" ")
                self.i = m.end()
            a = np.array(txt.split(), dtype=dt)
        return a.reshape(n, nc) if nc > 1 else a

    def _generic_until(self, close):
        items = []
        while True:
            t = self.tok()
            if t is None or t == close:
                return items
            if t == "(":
                items.append(self._generic_until(")"))
            else:
                items.append(t)

    def read_value(self):
        items = []
        while True:
            t = self.tok()
            if t is None or t == ";":
                return items
            if t == "(":
                items.append(self._generic_until(")"))
                continue
            m = re.fullmatch(r"List<(\w+)>", t)
            if m:
                n = int(self.tok())
                items.append(self.read_array(m.group(1), n))
            else:
                items.append(t)

    def read_dict(self):
        d = {}
        while True:
            t = self.tok()
            if t is None or t == "}":
                return d
            if t.startswith("#"):  # directive: skip rest of line
                j = self.d.find(b"\n", self.i)
                self.i = len(self.d) if j < 0 else j + 1
                continue
            if self.peek() == "{":
                self.tok()
                d[t] = self.read_dict()
            else:
                d[t] = self.read_value()

    def read_header(self):
        t = self.tok()
        if t != "FoamFile":
            raise ValueError("no FoamFile header")
        self.expect("{")
        h = self.read_dict()
        self.binary = h.get("format", ["ascii"])[0] == "binary"
        arch = h.get("arch", [""])[0]
        if self.binary and arch and ("label=32" not in arch or "scalar=64" not in arch):
            raise ValueError(f"unsupported arch {arch}")
        return h


def _flatten(x):
    out = []
    for v in x:
        if isinstance(v, list):
            out.extend(_flatten(v))
        else:
            out.append(float(v))
    return out


def read_dict(path):
    r = Reader(Path(path).read_bytes())
    r.read_header()
    return r.read_dict()


def scalar(v):
    return float(v[0])


def vector(v):
    return np.array(_flatten(v), float)


def _read_list_file(path, typ):
    r = Reader(Path(path).read_bytes())
    r.read_header()
    n = int(r.tok())
    return r.read_array(typ, n)


def read_faces(path):
    """Return (flat point labels, offsets[nFaces+1])."""
    r = Reader(Path(path).read_bytes())
    h = r.read_header()
    if h["class"][0] == "faceCompactList":
        n = int(r.tok())
        off = r.read_array("label", n)
        m = int(r.tok())
        lab = r.read_array("label", m)
        return lab.astype(np.int64), off.astype(np.int64)
    n = int(r.tok())
    r.expect("(")
    j = r.d.rfind(b")")
    t = np.array(r.d[r.i:j].replace(b"(", b" ").replace(b")", b" ").split(), dtype=np.int64)
    off = np.zeros(n + 1, np.int64)
    pos = 0
    keep = np.ones(len(t), bool)
    for f in range(n):
        k = t[pos]
        keep[pos] = False
        off[f + 1] = off[f] + k
        pos += k + 1
    return t[keep], off


def read_boundary(path):
    r = Reader(Path(path).read_bytes())
    r.read_header()
    n = int(r.tok())
    r.expect("(")
    patches = []
    for _ in range(n):
        name = r.tok()
        r.expect("{")
        d = r.read_dict()
        patches.append(dict(name=name, type=d["type"][0], nFaces=int(d["nFaces"][0]),
                            startFace=int(d["startFace"][0])))
    return patches


def read_mesh(case):
    pm = Path(case) / "constant/polyMesh"
    pts = _read_list_file(pm / "points", "vector")
    own = _read_list_file(pm / "owner", "label").astype(np.int64)
    nei = _read_list_file(pm / "neighbour", "label").astype(np.int64)
    fl, fo = read_faces(pm / "faces")
    return dict(points=pts, owner=own, neighbour=nei, face_labels=fl, face_offsets=fo,
                patches=read_boundary(pm / "boundary"))


def read_field(path):
    """Return dict(cls, internal(list or ndarray), boundary{name: dict})."""
    r = Reader(Path(path).read_bytes())
    h = r.read_header()
    d = r.read_dict()
    return dict(cls=h["class"][0], dimensions=d.get("dimensions"), internal=d["internalField"],
                boundary=d.get("boundaryField", {}))


def expand(value, n, ncomp):
    """Turn a parsed 'uniform X' / 'nonuniform List' entry into an array of n items."""
    if value[0] == "uniform":
        v = np.array(_flatten(value[1:]), float)
        return np.tile(v, (n, 1)) if ncomp > 1 else np.full(n, v[0])
    a = value[1]
    if isinstance(a, list):  # e.g. 'nonuniform 0()'
        return np.zeros((n, ncomp) if ncomp > 1 else n)
    return np.asarray(a, float)


def patch_entry(bdict, name):
    if name in bdict:
        return bdict[name]
    for k, v in bdict.items():
        try:
            if re.fullmatch(k, name):
                return v
        except re.error:
            pass
    raise KeyError(f"no boundary entry for patch {name}")


_DIMS = {"U": "[0 1 -1 0 0 0 0]", "p": "[0 2 -2 0 0 0 0]", "phi": "[0 3 -1 0 0 0 0]",
         "k": "[0 2 -2 0 0 0 0]", "omega": "[0 0 -1 0 0 0 0]", "nut": "[0 2 -1 0 0 0 0]",
         "yPlus": "[0 0 0 0 0 0 0]"}


def _blob(a):
    a = np.ascontiguousarray(a, np.float64)
    typ = {1: "scalar", 3: "vector"}[1 if a.ndim == 1 else a.shape[1]]
    return f"nonuniform List<{typ}> \n{a.shape[0]}\n(".encode() + a.tobytes() + b")\n;\n"


def write_field(case, time, name, cls, internal, patches):
    """patches: list of (name, text-of-type-entries, value array or None)."""
    d = Path(case) / str(time)
    d.mkdir(parents=True, exist_ok=True)
    out = [("FoamFile\n{\n    version 2.0;\n    format binary;\n    arch \"LSB;label=32;scalar=64\";\n"
            f"    class {cls};\n    location \"{time}\";\n    object {name};\n}}\n\n"
            f"dimensions {_DIMS[name]};\n\ninternalField ").encode(), _blob(internal),
           b"\nboundaryField\n{\n"]
    for pname, entries, val in patches:
        out.append(f"    {pname}\n    {{\n        {entries}\n".encode())
        if val is not None:
            out.append(b"        value " + _blob(val))
        out.append(b"    }\n")
    out.append(b"}\n")
    (d / name).write_bytes(b"".join(out))
