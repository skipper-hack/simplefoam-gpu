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


def check_case(case):
    """Raise UnsupportedCase listing every setting outside what the solver implements."""
    case = Path(case)
    problems = []
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
    for p in foamio.read_boundary(case / "constant/polyMesh/boundary"):
        if p["type"] not in SUPPORTED_PATCHES:
            problems.append(f"patch {p['name']} has type {p['type']}; supported: {', '.join(sorted(SUPPORTED_PATCHES))}")
    if problems:
        raise UnsupportedCase("this case uses settings the GPU solver does not implement:\n  - "
                              + "\n  - ".join(problems))


def _has_entries(path):
    try:
        return any(isinstance(v, dict) and v.get("active", ["yes"])[0] not in ("no", "false", "off")
                   for v in foamio.read_dict(path).values())
    except Exception:
        return True
