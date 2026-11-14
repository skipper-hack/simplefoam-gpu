# OpenFOAM transpose fix

The conda-forge build of OpenFOAM v2412 (`openfoam-2412-h946d6a9_1`) computes the transpose of a tensor field
wrongly when the result and the input are the same field. In every even-numbered cell, the zx component is left
untransposed. OpenFOAM transposes in place inside the viscous stress term `dev2(T(grad(U)))` of every turbulence
model (laminar included), so the x-momentum equation of every `simpleFoam` and `rhoSimpleFoam` run with that build
is slightly wrong:

- **Symmetry breaks:** a mirror-symmetric case gives an asymmetric solution.
- **Residual stalls:** the x-momentum residual of a converged solution stays far above y and z.
- **Answer shifts:** on a car model, the drag coefficient moved by 0.3%.

The same OpenFOAM source compiled fresh is correct, so the fault is in that binary, not in OpenFOAM's code. Other
builds may or may not be affected; `transposeTest` tells you.

## Build

With the OpenFOAM environment sourced:

```bash
./Allwmake
```

This puts `libtransposeFix.so` in `$FOAM_USER_LIBBIN` and `transposeTest` in `$FOAM_USER_APPBIN`.

## Check your OpenFOAM

In any case folder that has a mesh and a `U` field:

```bash
transposeTest
```

On an affected build:

```
transpose function resolved from: .../libfiniteVolume.so
wrong cells, separate fields: 0
wrong cells, in place:        26465
AFFECTED: this OpenFOAM build transposes tensor fields wrongly; preload libtransposeFix.so
```

## Fix

Preload the library into any solver. Nothing needs rebuilding, and solver settings stay as they are:

```bash
LD_PRELOAD=$FOAM_USER_LIBBIN/libtransposeFix.so simpleFoam
mpiexec -n 8 env LD_PRELOAD=$FOAM_USER_LIBBIN/libtransposeFix.so simpleFoam -parallel
```

The library holds correct compiled copies of the three `Foam::T` functions for `volTensorField`. The dynamic
linker uses them in place of the library's copies. With it preloaded, `transposeTest` reports
`OK: transpose is correct`.

Measured effect: starting from a converged solution, one `simpleFoam` iteration gives an x-momentum residual of
0.015 without the fix and 1.3e-15 with it (y and z: 1.3e-15 and 6.9e-16 either way).

## Licence

GPL-3.0-or-later (see [LICENSE](LICENSE)), because this code is compiled against OpenFOAM.
