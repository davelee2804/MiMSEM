#include <iostream>

#include <cmath>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"
#include "Test.h"

using namespace std;

#define RAD_SPHERE 1.0

double u_init(double* x) {
    return x[0]; // R.cos(phi).cos(theta)
}

double v_init(double* x) {
    return x[2]; // R.cos(phi).sin(theta)
}

double k_init(double* x) {
    double u = u_init(x);
    double v = v_init(x);

    return 0.5*(u*u + v*v);
}

void kineticEnergy(SWEqn* sw, Vec ug, Vec* ke) {
    Vec ul, u2;
    KSP ksp;

    VecCreateSeq(MPI_COMM_SELF, sw->topo->n1, &ul);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &u2);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, ke);

    sw->init1(ug, u_init, v_init);

    VecScatterBegin(sw->topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(sw->topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(*ke);
    sw->K->assemble(ul);
    MatMult(sw->K->M, ug, u2);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, sw->M2->M, sw->M2->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOptionsPrefix(ksp,"ke_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, u2, *ke);

    KSPDestroy(&ksp);
    VecDestroy(&ul);
    VecDestroy(&u2);
}

int main(int argc, char** argv) {
    int size, rank;
    double err;
    static char help[] = "petsc";
    char fieldname[20];
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Vec ui, ke, ke_a;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo);
    sw = new SWEqn(topo, geom);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &ke_a);

    sw->init1(ui, u_init, v_init);
    sw->init2(ke_a, k_init);
    kineticEnergy(sw, ui, &ke);

    sprintf(fieldname,"velocity");
    geom->write1(ui,fieldname,0);
    sprintf(fieldname,"ke_n");
    geom->write2(ke,fieldname,0);
    sprintf(fieldname,"ke_a");
    geom->write2(ke_a,fieldname,0);

    err = sw->err2(ke, k_init);
    if(!rank) cout << "L2 divergence error: " << err << endl;

    delete topo;
    delete geom;
    delete sw;

    VecDestroy(&ui);
    VecDestroy(&ke);
    VecDestroy(&ke_a);

    PetscFinalize();

    return 0;
}
