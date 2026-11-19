# SPDX-License-Identifier: MIT
"""Refuse cases this solver does not reproduce exactly, with a clear message, instead of solving them differently."""
import re
from pathlib import Path
from . import foamio

# The schemes the GPU discretisation implements. Anything else would silently give a different answer.
REQUIRED_SCHEMES = {
    ("ddtSchemes", "default"): "steadyState",
    ("gradSchemes", "default"): "Gauss linear",
    ("gradSchemes", "grad(U)"): "cellLimited Gauss linear 1",
    ("divSchemes", "div(phi,U)"): "bounded Gauss linearUpwindV grad(U)",
    ("divSchemes", "div((nuEff*dev2(T(grad(U)))))"): "Gauss linear",
    ("laplacianSchemes", "default"): "Gauss linear corrected",
    ("interpolationSchemes", "default"): "linear",
    ("snGradSchemes", "default"): "corrected",
}
TURBULENCE_SCHEMES = {
    ("divSchemes", "div(phi,k)"): "bounded Gauss upwind",
    ("divSchemes", "div(phi,omega)"): "bounded Gauss upwind",
}
SUPPORTED_PATCHES = {"patch", "wall", "symmetry", "symmetryPlane"}

# Boundary condition types the solver implements per field, with the entries each one reads.
_SCALAR_BCS = {"fixedValue": ("value",), "inletOutlet": ("inletValue",), "omegaWallFunction": ("value",),
               "zeroGradient": (), "kqRWallFunction": ()}
FIELD_BCS = {
    "U": {"fixedValue": ("value",), "noSlip": (), "movingWallVelocity": ("value",), "slip": (), "symmetry": (),
          "symmetryPlane": (), "zeroGradient": (), "pressureInletOutletVelocity": (), "inletOutlet": ("inletValue",)},
    "p": {"fixedValue": ("value",), "zeroGradient": ()},
    "k": _SCALAR_BCS,
    "omega": _SCALAR_BCS,
    "nut": {"nutUSpaldingWallFunction": (), "calculated": ()},
}


class UnsupportedCase(Exception):
    pass


def _sections(path):
    """fvSchemes as {section: {key: value text}}. Parsed as text, because keys such as grad(U) contain brackets."""
    txt = re.sub(r"/\*.*?\*/", " ", Path(path).read_text(), flags=re.S)
    txt = re.sub(r"//[^\n]*", " ", txt)
    out = {}
    for m in re.finditer(r"(\w+)\s*\{", txt):
        depth, i = 1, m.end()
        while depth and i < len(txt):
            depth += {"{": 1, "}": -1}.get(txt[i], 0)
            i += 1
        body = txt[m.end():i - 1]
        entries = {}
        for e in body.split(";"):
            e = e.strip()
            if not e or "{" in e:
                continue
            if e.startswith('"'):
                j = e.index('"', 1)
                key, val = e[1:j], e[j + 1:]
            else:
                key, _, val = e.partition(" ")
            entries[key] = " ".join(val.split())
        out[m.group(1)] = entries
    return out


def _lookup(d, section, key):
    sec = d.get(section, {})
    if key in sec:
        v = sec[key]
    else:
        # OpenFOAM regex keys such as "div\(phi,(k|omega)\)"
        v = next((val for k, val in sec.items() if _is_regex_match(k, key)), None)
    for _ in range(5):   # $macro inside the same section
        if v is not None and v.startswith("$") and " " not in v:
            v = sec.get(v[1:])
    return v


def _is_regex_match(pattern, key):
    try:
        return re.fullmatch(pattern, key) is not None
    except re.error:
        return False


def check_case(case, start=None):
    """Raise UnsupportedCase listing every setting outside what the solver implements.
    start: the time directory the run starts from; its field files are checked too (skipped when None)."""
    case = Path(case)
    problems = []
    try:
        _check(case, start, problems)
    except foamio.FoamSyntaxError as e:
        problems.append(str(e))
    if problems:
        raise UnsupportedCase("this case uses settings the GPU solver does not implement:\n  - "
                              + "\n  - ".join(problems))


def _check(case, start, problems):
    cd = foamio.read_dict(case / "system/controlDict")
    app = cd.get("application", ["simpleFoam"])[0]
    if app != "simpleFoam":
        problems.append(f"application is {app}; only simpleFoam cases are supported")
        raise UnsupportedCase(problems[0])
    tp = foamio.read_dict(case / "constant/turbulenceProperties")
    sim = tp.get("simulationType", ["laminar"])[0]
    rans = sim == "RAS"
    if sim not in ("laminar", "RAS"):
        problems.append(f"simulationType {sim}; only laminar and RAS kOmegaSST are supported")
    if rans:
        model = tp.get("RAS", {}).get("RASModel", tp.get("RAS", {}).get("model", [""]))[0]
        if model != "kOmegaSST":
            problems.append(f"RAS model {model}; only kOmegaSST is supported")
    for f in ("constant/fvOptions", "system/fvOptions", "constant/fvModels", "constant/fvConstraints"):
        if (case / f).exists() and _has_entries(case / f):
            problems.append(f"{f} is present; fvOptions/source terms are not supported")
    mrf = case / "constant/MRFProperties"
    if mrf.exists() and _has_entries(mrf):
        problems.append("constant/MRFProperties defines rotating zones; MRF is not supported")
    if not (case / "constant/transportProperties").exists():
        problems.append("constant/transportProperties is missing (simpleFoam needs nu)")
    elif foamio.read_dict(case / "constant/transportProperties").get("transportModel", ["Newtonian"])[0] != "Newtonian":
        problems.append("only Newtonian transport (constant nu) is supported")
    sch = _sections(case / "system/fvSchemes")
    want = {**REQUIRED_SCHEMES, **(TURBULENCE_SCHEMES if rans else {})}
    for (sec, key), val in want.items():
        got = _lookup(sch, sec, key)
        if got is None and key != "default":
            got = _lookup(sch, sec, "default")
        if got != val:
            problems.append(f"fvSchemes {sec} {key} is '{got}'; the solver implements '{val}'")
    patches = foamio.read_boundary(case / "constant/polyMesh/boundary")
    for p in patches:
        if p["type"] not in SUPPORTED_PATCHES:
            problems.append(f"patch {p['name']} has type {p['type']}; supported: {', '.join(sorted(SUPPORTED_PATCHES))}")
    if start is not None:
        fields = ["U", "p"] + (["k", "omega", "nut"] if rans else [])
        for name in fields:
            _check_field(case, str(start), name, [p for p in patches if p["type"] in SUPPORTED_PATCHES], problems)


def _check_field(case, start, name, patches, problems):
    """Every patch has a boundary entry of a type the solver implements, with readable values."""
    rel = f"{start}/{name}"
    path = case / start / name
    if not path.is_file():
        problems.append(f"{rel} is missing")
        return
    try:
        f = foamio.read_field(path)
    except foamio.FoamSyntaxError as e:
        problems.append(str(e))
        return
    except (ValueError, KeyError, IndexError) as e:
        problems.append(f"{rel} could not be read ({type(e).__name__}: {e})")
        return
    nc = 3 if name == "U" else 1
    if not _readable(f["internal"], None, nc):
        problems.append(f"{rel} internalField '{_text(f['internal'])}' is not a value the solver can read")
    allowed = FIELD_BCS[name]
    unsupported = {}
    for p in patches:
        try:
            e = foamio.patch_entry(f["boundary"], p["name"], p["groups"])
        except KeyError:
            problems.append(f"{rel} has no boundaryField entry for patch {p['name']}")
            continue
        t = e.get("type", [None])[0] if isinstance(e, dict) else None
        if t not in allowed:
            unsupported.setdefault(t, []).append(p["name"])
            continue
        for key in allowed[t]:
            if key not in e:
                problems.append(f"{rel} patch {p['name']} ({t}) has no {key}")
        for key in ("value", "inletValue"):
            if key in e and not _readable(e[key], p["nFaces"], nc):
                problems.append(f"{rel} patch {p['name']} {key} '{_text(e[key])}' is not a value the solver can read")
    for t, names in unsupported.items():
        what = f"boundary type {t}" if t else "a boundary entry without a type"
        problems.append(f"{rel} uses {what} on {_names(names)}; supported: {', '.join(sorted(allowed))}")


FORCE_COEFFS_KEYS = ("patches", "dragDir", "liftDir", "pitchAxis", "rhoInf", "magUInf", "Aref", "lRef")


def function_objects(case, rans):
    """controlDict functions: the forceCoeffs entry the solver will run as (name, dict) or None, and a warning
    for every function object it will not run (it runs one forceCoeffs, and yPlus in turbulent cases)."""
    fn = foamio.read_dict(Path(case) / "system/controlDict").get("functions", {})
    force, warnings = None, []
    if not isinstance(fn, dict):
        return force, warnings
    for name in fn.get("#includeFunc", []):
        warnings.append(f"function object '{name}' (#includeFunc) is not supported and will not run")
    for what in fn.get("#missing", []):
        warnings.append(f"functions: {what} could not be read (OpenFOAM's etc directory is not set); "
                        "what it defines will not run")
    for name, d in fn.items():
        if not isinstance(d, dict):
            continue
        if str(d.get("enabled", ["true"])[0]).lower() in ("false", "no", "off"):
            continue
        typ = d.get("type", [None])[0]
        missing = d.get("#missing", [])
        why = f" ({', '.join(missing)} could not be read)" if missing else ""
        if typ == "yPlus" and rans:
            continue
        if typ != "forceCoeffs":
            kind = f"type {typ}" if typ else "no type"
            warnings.append(f"function object '{name}' ({kind}) is not supported by simplefoam-gpu and will not run{why}")
        elif force is not None:
            warnings.append(f"function object '{name}': only one forceCoeffs is run ('{force[0]}'); this one will not run")
        elif any(k not in d for k in FORCE_COEFFS_KEYS):
            need = ", ".join(k for k in FORCE_COEFFS_KEYS if k not in d)
            warnings.append(f"forceCoeffs '{name}' has no {need}; it will not run{why}")
        else:
            force = (name, d)
    return force, warnings


def _readable(value, n, nc):
    """foamio.expand understands the value and gives the right shape (n items, or any number when n is None)."""
    try:
        a = foamio.expand(value, 1 if n is None else n, nc)
    except (ValueError, TypeError, IndexError, KeyError):
        return False
    if nc > 1 and (a.ndim != 2 or a.shape[1] != nc):
        return False
    if nc == 1 and a.ndim != 1:
        return False
    return n is None or len(a) == n


def _text(v):
    s = " ".join("<list>" if hasattr(x, "shape") else str(x) for x in _tokens(v))
    return s if len(s) <= 60 else s[:57] + "..."


def _tokens(v):
    out = []
    for x in v if isinstance(v, list) else [v]:
        if isinstance(x, list):
            out.extend(["("] + _tokens(x) + [")"])
        else:
            out.append(x)
    return out


def _names(names):
    return "patch " + names[0] if len(names) == 1 else (
        "patches " + ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else ""))


def _has_entries(path):
    try:
        return any(isinstance(v, dict) and v.get("active", ["yes"])[0] not in ("no", "false", "off")
                   for v in foamio.read_dict(path).values())
    except Exception:
        return True
