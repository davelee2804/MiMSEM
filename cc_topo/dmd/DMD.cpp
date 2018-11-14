#include <iostream>
#include <fstream>

#include <cmath>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <slepcsvd.h>

#include "Basis.h"
#include "Topo.h"
#include "Geom.h"

#define OFFSET 1
#define NQ 12
#define ZERO_FORM

using namespace std;

void LoadVecs(Vec* vecs, int nk, char* fieldname) {
    int ki;
    char filename[100];
    PetscViewer viewer;

    for(ki = 0; ki < nk; ki++) {
        sprintf(filename, "output/%s_%.4u.vec", fieldname, ki+OFFSET);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
        VecLoad(vecs[ki], viewer);
        PetscViewerDestroy(&viewer);
    }
}

int main(int argc, char** argv) {
    int rank, ki, nEig, ii;
    double sigma;
    static char help[] = "petsc";
    char* fieldname = argv[1];
    char eigvecname[100];
    ofstream file;
    Topo* topo;
    Geom* geom;
    Vec* vecs;
    Vec lVec, rVec, vLocal;
    Mat X;
    PetscScalar* vArray;
    SVD svd;

    SlepcInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    topo = new Topo(rank);
    geom = new Geom(rank, topo);

    vecs  = new Vec[NQ];
    for(ki = 0; ki < NQ; ki++) {
#ifdef ZERO_FORM
        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vecs[ki]);
#else
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &vecs[ki]);
#endif
    }
#ifdef ZERO_FORM
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &vLocal);
#else
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &vLocal);
#endif

    LoadVecs(vecs, NQ, fieldname);

    // pack the time slice data into a dense matrix
#ifdef ZERO_FORM
    MatCreateDense(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NQ, topo->nDofs0G, NULL, &X);
#else
    MatCreateDense(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NQ, topo->nDofs2G, NULL, &X);
#endif
    for(ki = 0; ki < NQ; ki++) {
#ifdef ZERO_FORM
        VecScatterBegin(topo->gtol_0, vecs[ki], vLocal, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_0, vecs[ki], vLocal, INSERT_VALUES, SCATTER_FORWARD);
#else
        VecScatterBegin(topo->gtol_2, vecs[ki], vLocal, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, vecs[ki], vLocal, INSERT_VALUES, SCATTER_FORWARD);
#endif

        VecGetArray(vLocal, &vArray);
#ifdef ZERO_FORM
        MatSetValues(X, 1, &ki, topo->n0, topo->loc0, vArray, INSERT_VALUES);
#else
        MatSetValues(X, 1, &ki, topo->n2, topo->loc2, vArray, INSERT_VALUES);
#endif
        VecRestoreArray(vLocal, &vArray);
    }
    MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  X, MAT_FINAL_ASSEMBLY);
    MatCreateVecs(X, &rVec, &lVec);

    // compute the svd
    SVDCreate(MPI_COMM_WORLD, &svd);
    SVDSetOperator(svd, X);
    SVDSetFromOptions(svd);
    SVDSolve(svd);
    SVDGetConverged(svd, &nEig);

    if(!rank) cout << "number of eigenvalues: " << nEig << endl;
    for(ii = 0; ii < nEig; ii++) {
        SVDGetSingularTriplet(svd, ii, &sigma, lVec, rVec);
        if(!rank) cout << ii << "\tsigma: " << sigma << endl;
        sprintf(eigvecname, "%s_eigvec", fieldname);
#ifdef ZERO_FORM
        geom->write0(rVec, eigvecname, ii);
#else
        geom->write2(rVec, eigvecname, ii);
#endif
    }

    SVDDestroy(&svd);
    MatDestroy(&X);
    VecDestroy(&lVec);
    VecDestroy(&rVec);
    VecDestroy(&vLocal);
    for(ki = 0; ki < NQ; ki++) {
        VecDestroy(&vecs[ki]);
    }
    delete[] vecs;
    delete geom;
    delete topo;

    SlepcFinalize();

    return 0;
}
