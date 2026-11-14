# SPDX-License-Identifier: MIT
"""Render the README images from the solved examples (ParaView):

    pvbatch docs/render_examples.py examples/sphere-laminar examples/sphere-kOmegaSST

Writes docs/images/<case>-velocity.png. Flow is along -z, shown left to right.
"""
import os
import sys
from paraview.simple import (CellDatatoPointData, ColorBy, CreateView, GetColorTransferFunction, GetScalarBar, Hide,
                             Line, OpenFOAMReader, Render, SaveScreenshot, Show, Slice, StreamTracerWithCustomSource)

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = [1600, 700]


def render(case):
    name = os.path.basename(os.path.normpath(case))
    foam = os.path.join(case, "case.foam")
    open(foam, "a").close()
    r = OpenFOAMReader(FileName=foam)
    r.MeshRegions = ["internalMesh"]
    r.CellArrays = ["U"]
    r.UpdatePipelineInformation()
    t = list(r.TimestepValues)[-1]
    wall = OpenFOAMReader(FileName=foam)
    wall.MeshRegions = ["patch/sphere"]
    wall.UpdatePipeline(t)
    pts = CellDatatoPointData(Input=r)
    sl = Slice(Input=pts)
    sl.SliceType.Origin = [0, 0, 0]
    sl.SliceType.Normal = [0, 1, 0]
    uinf = 1.0 if "laminar" in name else 20.0

    view = CreateView("RenderView")
    view.ViewSize = SIZE
    view.ViewTime = t
    try:
        view.UseColorPaletteForBackground = 0
    except AttributeError:
        pass
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0

    ws = Show(wall, view)
    ws.ColorArrayName = ['POINTS', '']
    ws.DiffuseColor = [0.35, 0.36, 0.40]

    def shot(field, comp, title, lo, hi, cmap, out, lines):
        s = Show(sl, view)
        ColorBy(s, ("POINTS", field, comp) if comp else ("POINTS", field))
        lut = GetColorTransferFunction(field)
        lut.ApplyPreset(cmap, True)
        lut.RescaleTransferFunction(lo, hi)
        bar = GetScalarBar(lut, view)
        bar.Title = title
        bar.ComponentTitle = ""
        bar.TitleColor = bar.LabelColor = [0.15, 0.15, 0.15]
        bar.Orientation = "Horizontal"
        bar.WindowLocation = "Any Location"
        bar.Position = [0.62, 0.05]
        bar.ScalarBarLength = 0.33
        bar.AddRangeLabels = 0
        bar.TitleFontSize = bar.LabelFontSize = 14
        s.SetScalarBarVisibility(view, True)
        st = None
        if lines:
            seed = Line()
            seed.Point1 = [-1.45, 0, 1.2]
            seed.Point2 = [1.45, 0, 1.2]
            seed.Resolution = 40
            st = StreamTracerWithCustomSource(Input=pts, SeedSource=seed)
            st.Vectors = ["POINTS", "U"]
            st.MaximumStreamlineLength = 20
            st.MaximumSteps = 20000
            ss = Show(st, view)
            ss.ColorArrayName = ['POINTS', '']
            ss.DiffuseColor = ss.AmbientColor = [1, 1, 1]
            ss.LineWidth = 1.2
            ss.Opacity = 0.75
        Render(view)                        # first render resets the camera; set it afterwards
        view.CameraFocalPoint = [0, 0, -1.6]
        view.CameraPosition = [0, 12, -1.6]
        view.CameraViewUp = [-1, 0, 0]      # screen right = -z = flow direction
        view.CameraParallelProjection = 1
        view.CameraParallelScale = 1.55
        Render(view)
        SaveScreenshot(out, view, ImageResolution=SIZE)
        print("wrote", out, flush=True)
        Hide(sl, view)
        if st is not None:
            Hide(st, view)
        s.SetScalarBarVisibility(view, False)

    out = os.path.join(HERE, "images")
    shot("U", "Magnitude", "velocity magnitude (m/s)", 0, 1.3 * uinf, "Viridis",
         os.path.join(out, f"{name}-velocity.png"), True)
    os.remove(foam)


for c in sys.argv[1:]:
    render(c)
