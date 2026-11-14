// SPDX-License-Identifier: GPL-3.0-or-later
// Compiled against OpenFOAM (GPL-3.0-or-later); see openfoam-fix/LICENSE.
/*
libtransposeFix.so: correct copies of Foam::T for volTensorField, for LD_PRELOAD into OpenFOAM v2412 solvers.

Found in the conda-forge OpenFOAM v2412 binary (openfoam-2412-h946d6a9_1): its libfiniteVolume.so copy of T(result, input) is wrong when result and input are the same field (which
is what T(tmp) does when it reuses a temporary): it leaves the zx entry untransposed in every even-numbered cell.
The viscous stress term dev2(T(grad(U))) of every turbulence model goes through that path, corrupting the
x-momentum equation. Preloading this library makes the dynamic linker use these correct definitions instead.
Check whether your build is affected with ../transposeTest.
*/
#include "volFields.H"

namespace Foam
{
    template void T<fvPatchField, volMesh>
    (
        GeometricField<tensor, fvPatchField, volMesh>&,
        const GeometricField<tensor, fvPatchField, volMesh>&
    );
    template tmp<GeometricField<tensor, fvPatchField, volMesh>> T<fvPatchField, volMesh>
    (
        const GeometricField<tensor, fvPatchField, volMesh>&
    );
    template tmp<GeometricField<tensor, fvPatchField, volMesh>> T<fvPatchField, volMesh>
    (
        const tmp<GeometricField<tensor, fvPatchField, volMesh>>&
    );
}
