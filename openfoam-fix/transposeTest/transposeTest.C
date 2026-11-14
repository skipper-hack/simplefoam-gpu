// SPDX-License-Identifier: GPL-3.0-or-later
// Compiled against OpenFOAM (GPL-3.0-or-later); see openfoam-fix/LICENSE.
/*
transposeTest: checks whether the installed OpenFOAM computes the in-place transpose of a volTensorField correctly.

Run it in any case folder with a mesh and a U field at the start time:
    transposeTest                                      # tests the installed libfiniteVolume.so
    LD_PRELOAD=/path/to/libtransposeFix.so transposeTest   # tests with the fix preloaded

It calls the T(result, input) function that the dynamic linker resolves (the same one every solver uses), once with
separate fields and once with result and input being the same field, which is what T(tmp) does when it reuses a
temporary inside dev2(T(grad(U))).
*/
#include "fvCFD.H"
#include <dlfcn.h>

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    volVectorField U(IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ), mesh);

    typedef void (*TFn)(volTensorField&, const volTensorField&);
    TFn libT = reinterpret_cast<TFn>(dlsym(RTLD_DEFAULT,
        "_ZN4Foam1TINS_12fvPatchFieldENS_7volMeshEEEvRNS_14GeometricFieldINS_6TensorIdEET_T0_EERKS8_"));
    Dl_info info;
    if (!libT || !dladdr(reinterpret_cast<void*>(libT), &info))
    {
        Info<< "T(volTensorField&, const volTensorField&) not found" << endl;
        return 1;
    }
    Info<< "transpose function resolved from: " << info.dli_fname << endl;

    const volTensorField G(IOobject("G", runTime.timeName(), mesh), fvc::grad(U));
    auto countWrong = [&G](const volTensorField& R)
    {
        label n = 0;
        forAll(G, i)
        {
            if (cmptMax(cmptMag(R[i] - G[i].T())) > 1e-12*max(1.0, cmptMax(cmptMag(G[i])))) n++;
        }
        return n;
    };

    volTensorField R(IOobject("R", runTime.timeName(), mesh), G);
    libT(R, G);                         // separate result and input
    const label wrongSeparate = countWrong(R);

    volTensorField A(IOobject("A", runTime.timeName(), mesh), G);
    libT(A, A);                         // result and input are the same field
    const label wrongInPlace = countWrong(A);

    Info<< "cells: " << G.size() << nl
        << "wrong cells, separate fields: " << wrongSeparate << nl
        << "wrong cells, in place:        " << wrongInPlace << nl
        << (wrongSeparate + wrongInPlace
            ? "AFFECTED: this OpenFOAM build transposes tensor fields wrongly; preload libtransposeFix.so"
            : "OK: transpose is correct")
        << endl;
    return 0;
}
