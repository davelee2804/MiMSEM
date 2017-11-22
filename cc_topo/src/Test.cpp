#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscksp.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"
#include "Test.h"

using namespace std;

Test::Test(SWEqn* _sw) {
    sw = _sw;
}

Test::~Test() {
}

void Test::ke(ICfunc* fu, ICfunc* fv) {
    Vec ul, u, u2, k;
    KSP ksp;
    char filename[20];

    VecCreateSeq(MPI_COMM_SELF, sw->topo->n1, &ul);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &u);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &u2);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &k);

    sw->init1(u, fu, fv);

    VecScatterBegin(sw->topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(sw->topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(k);
    sw->K->assemble(ul);
    MatMult(sw->K->M, u, u2);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, sw->M2->M, sw->M2->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOptionsPrefix(ksp,"test_ke_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, u2, k);

    sprintf(filename,"test_ke");
    sw->geom->write2(k, filename, 0);

    KSPDestroy(&ksp);
    VecDestroy(&ul);
    VecDestroy(&u);
    VecDestroy(&u2);
    VecDestroy(&k);
}
