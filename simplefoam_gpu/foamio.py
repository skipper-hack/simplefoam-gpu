# SPDX-License-Identifier: MIT
"""Minimal reader/writer for OpenFOAM files (ascii and binary, label=32 scalar=64).

Follows #include, #includeEtc and $macro substitution the way OpenFOAM's dictionary reader does, and refuses
(FoamSyntaxError) what it cannot evaluate, such as #calc, instead of skipping it.
"""
import copy
import os
import re
from pathlib import Path
import numpy as np

NCOMP = {"scalar": 1, "label": 1, "vector": 3, "symmTensor": 6, "tensor": 9}
_PUNCT = b"{}();"
_WS = b" \t\r\n"

# Patch types that are always members of the patch group of their own name (polyPatch::constraintType).
CONSTRAINT_TYPES = {"cyclic", "cyclicAMI", "cyclicACMI", "cyclicSlip", "empty", "nonuniformTransformCyclic",
                    "processor", "processorCyclic", "symmetryPlane", "symmetry", "wedge"}

# OpenFOAM's etc/caseDicts/setConstraintTypes, used when no OpenFOAM installation can be found.
_BUILTIN_ETC = {"caseDicts/setConstraintTypes": b"""
cyclic                    { type cyclic; }
cyclicAMI                 { type cyclicAMI; }
cyclicACMI                { type cyclicACMI; value $internalField; }
cyclicSlip                { type cyclicSlip; }
empty                     { type empty; }
nonuniformTransformCyclic { type nonuniformTransformCyclic; }
processor                 { type processor; value $internalField; }
processorCyclic           { type processorCyclic; value $internalField; }
symmetryPlane             { type symmetryPlane; }
symmetry                  { type symmetry; }
wedge                     { type wedge; }
"""}
_MAX_INCLUDE_DEPTH = 32


class FoamSyntaxError(ValueError):
    """The file uses OpenFOAM syntax this reader cannot evaluate, or refers to something that does not exist."""


class _Quoted(str):
    """A token written in quotes (or a #{ #} block): never treated as a $macro or #directive."""


class Reader:
    def __init__(self, data: bytes, path=None, case=None, scopes=None, keys=None, depth=0):
        self.d = data
        self.i = 0
        self.binary = False
        self.path = Path(path) if path is not None else None
        self.case = case                                   # for <case>, <system>, $FOAM_CASE
        self.scopes = [] if scopes is None else scopes     # dictionaries being read, outermost first ($macros)
        self.keys = [] if keys is None else keys           # the keyword of each of those dictionaries
        self.depth = depth

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
            s = _Quoted(d[self.i + 1:j].decode())
            self.i = j + 1
            return s
        if d.startswith(b"#{", self.i):  # verbatim code block #{ ... #}
            j = d.find(b"#}", self.i + 2)
            j = n if j < 0 else j
            s = _Quoted(d[self.i + 2:j].decode("latin1"))
            self.i = min(n, j + 2)
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
            elif _is_macro(t):
                items.extend(self._expand(t))
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
            if _is_macro(t):
                items.extend(self._expand(t))
                continue
            if _is_directive(t) and not self._in_functions():
                raise FoamSyntaxError(f"{self._where()}: {t} cannot be evaluated by simplefoam-gpu; "
                                      "write the value out instead")
            m = re.fullmatch(r"List<(\w+)>", t)
            if m and m.group(1) in NCOMP:
                n = int(self.tok())
                items.append(self.read_array(m.group(1), n))
            elif not m:   # List<word> etc.: the items follow as an ordinary list
                items.append(t)

    def read_dict(self, d=None, key=None):
        d = {} if d is None else d
        self.scopes.append(d)
        self.keys.append(key)
        try:
            self._entries(d)
        finally:
            self.scopes.pop()
            self.keys.pop()
        return d

    def _entries(self, d):
        while True:
            t = self.tok()
            if t is None or t == "}":
                return
            if _is_directive(t):
                self._directive(t, d)
                continue
            if _is_macro(t) and self.peek() == ";":   # "$name;" pastes the entries of a dictionary
                self.tok()
                v = self._lookup(t)
                if not isinstance(v, dict):
                    raise FoamSyntaxError(f"{self._where()}: {t} is not a dictionary")
                _merge(d, copy.deepcopy(v))
                continue
            if self.peek() == "{":
                self.tok()
                old = d.get(t)    # a repeated dictionary keyword merges into the earlier one (#inputMode merge)
                d[t] = self.read_dict(old if isinstance(old, dict) else None, key=t)
            else:
                v = self.read_value()
                d[t] = v[0] if len(v) == 1 and isinstance(v[0], dict) else v

    # ---- directives and macros ------------------------------------------------------------------
    def _where(self):
        return str(self.path) if self.path is not None else "<input>"

    def _in_functions(self):
        """Inside controlDict's functions{}: optional extras, so unreadable parts are noted, not fatal."""
        return len(self.keys) > 1 and self.keys[1] == "functions"

    def _directive(self, t, d):
        if t in ("#include", "#sinclude", "#includeIfPresent"):
            arg = self.tok()
            path = self._local_path(arg)
            if path.is_file():
                self._include(path.read_bytes(), path, d)
            elif t == "#include":
                raise FoamSyntaxError(f"{self._where()}: #include \"{arg}\": {path} not found")
        elif t in ("#includeEtc", "#sincludeEtc"):
            arg = self.tok()
            path = next((p / arg for p in _etc_dirs() if (p / arg).is_file()), None)
            if path is not None:
                self._include(path.read_bytes(), path, d)
            elif arg in _BUILTIN_ETC:
                self._include(_BUILTIN_ETC[arg], Path("<etc>") / arg, d)
            elif self._in_functions():
                d.setdefault("#missing", []).append(f"#includeEtc \"{arg}\"")
            elif t == "#includeEtc":
                raise FoamSyntaxError(f"{self._where()}: #includeEtc \"{arg}\" needs OpenFOAM's etc directory; "
                                      "source OpenFOAM's etc/bashrc or set FOAM_ETC")
        elif t == "#includeFunc":
            j = self.d.find(b"\n", self.i)
            j = len(self.d) if j < 0 else j
            d.setdefault("#includeFunc", []).append(self.d[self.i:j].decode("latin1").strip())
            self.i = j
        elif t == "#remove":
            arg = self.tok()
            for k in _words(self._generic_until(")") if arg == "(" else [arg]):
                d.pop(k, None)
        elif t == "#inputMode":
            self.tok()
        elif t in ("#default", "#overwrite", "#merge"):
            pass   # prefixes for the next entry; reading it normally gives the same result for case files
        elif self._in_functions():
            d.setdefault("#missing", []).append(t)
            j = self.d.find(b"\n", self.i)
            self.i = len(self.d) if j < 0 else j
        else:
            raise FoamSyntaxError(f"{self._where()}: {t} is not supported by simplefoam-gpu")

    def _local_path(self, arg):
        s = str(arg)
        if self.case is not None:
            for tag, sub in (("<case>", self.case), ("<system>", self.case / "system"),
                             ("<constant>", self.case / "constant")):
                if s.startswith(tag):
                    s = str(sub) + s[len(tag):]
            s = s.replace("${FOAM_CASE}", str(self.case)).replace("$FOAM_CASE", str(self.case))
        p = Path(os.path.expanduser(os.path.expandvars(s)))
        if not p.is_absolute() and self.path is not None:
            p = self.path.parent / p
        return p

    def _include(self, data, path, d):
        if self.depth >= _MAX_INCLUDE_DEPTH:
            raise FoamSyntaxError(f"{self._where()}: #include nested too deeply (recursive include?)")
        sub = Reader(data, path, self.case, self.scopes, self.keys, self.depth + 1)
        if sub.peek() == "FoamFile":
            sub.tok()
            sub.expect("{")
            sub.read_dict(key="FoamFile")
        sub._entries(d)

    def _lookup(self, t):
        name = t[1:]
        if name.startswith("{{"):
            raise FoamSyntaxError(f"{self._where()}: the expression {t} cannot be evaluated by simplefoam-gpu")
        if name.startswith("{") and name.endswith("}"):
            name = name[1:-1]
        scopes = self.scopes[::-1]          # the current dictionary first, then each enclosing one
        if name[:1] in (":", "/"):          # scoped from the top level
            scopes, name = self.scopes[:1], name[1:]
        for scope in scopes:
            v = _find(scope, name)
            if v is not None:
                return v
        if name == "FOAM_CASE" and self.case is not None:
            return [str(self.case)]
        if name in os.environ:
            return [os.environ[name]]
        raise FoamSyntaxError(f"{self._where()}: {t} is not defined")

    def _expand(self, t):
        v = copy.deepcopy(self._lookup(t))
        return v if isinstance(v, list) else [v]

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


def _is_macro(t):
    return type(t) is str and len(t) > 1 and t[0] == "$"


def _is_directive(t):
    return type(t) is str and len(t) > 1 and t[0] == "#"


def _find(scope, name):
    """Entry `name` of a dictionary; 'a.b' or 'a/b' looks inside sub-dictionaries."""
    if name in scope:
        return scope[name]
    parts = re.split(r"[./]", name)
    if len(parts) < 2:
        return None
    v = scope
    for p in parts:
        if not isinstance(v, dict) or p not in v:
            return None
        v = v[p]
    return v


def _merge(d, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _merge(d[k], v)
        else:
            d[k] = v


def _words(x):
    out = []
    for v in x:
        out.extend(_words(v) if isinstance(v, list) else [str(v)])
    return out


def _etc_dirs():
    dirs = []
    if os.environ.get("FOAM_ETC"):
        dirs.append(Path(os.environ["FOAM_ETC"]))
    if os.environ.get("WM_PROJECT_DIR"):
        dirs.append(Path(os.environ["WM_PROJECT_DIR"]) / "etc")
    return dirs


def _case_of(path):
    """The case directory a file belongs to (case/system/x, case/0/U, case/constant/polyMesh/x), if any."""
    p = path.resolve().parent
    for c in (p.parent, p.parent.parent):
        if (c / "system").is_dir():
            return c
    return None


def _reader(path):
    path = Path(path)
    return Reader(path.read_bytes(), path, _case_of(path))


def _flatten(x):
    out = []
    for v in x:
        if isinstance(v, list):
            out.extend(_flatten(v))
        else:
            out.append(float(v))
    return out


def read_dict(path):
    r = _reader(path)
    r.read_header()
    return r.read_dict()


def scalar(v):
    return float(v[0])


def vector(v):
    return np.array(_flatten(v), float)


def _read_list_file(path, typ):
    r = _reader(path)
    r.read_header()
    n = int(r.tok())
    return r.read_array(typ, n)


def read_faces(path):
    """Return (flat point labels, offsets[nFaces+1])."""
    r = _reader(path)
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
    r = _reader(path)
    r.read_header()
    n = int(r.tok())
    r.expect("(")
    patches = []
    for _ in range(n):
        name = r.tok()
        r.expect("{")
        d = r.read_dict()
        typ = d["type"][0]
        groups = [g for g in _words(d.get("inGroups", [])) if not g.isdigit()]
        if (typ in CONSTRAINT_TYPES or typ == "wall") and typ not in groups:   # as polyPatch/wallPolyPatch do
            groups.append(typ)
        patches.append(dict(name=name, type=typ, nFaces=int(d["nFaces"][0]),
                            startFace=int(d["startFace"][0]), groups=groups))
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
    r = _reader(path)
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


def patch_entry(bdict, name, groups=()):
    """The boundaryField entry for a patch, in OpenFOAM's order: its name, then its patch groups
    (e.g. 'symmetry' from setConstraintTypes; the last such entry wins), then regular expressions."""
    if name in bdict:
        return bdict[name]
    for k in reversed(list(bdict)):
        if k in groups and isinstance(bdict[k], dict):
            return bdict[k]
    for k, v in bdict.items():
        if not isinstance(v, dict):
            continue
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
