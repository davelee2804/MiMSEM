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
    ofstream file;
    Topo* topo;
    Geom* geom;
    Vec* vecs;
    Vec lVec, rVec;
    Mat A;
    PetscScalar* vArray;
    SVD svd;

    SlepcInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    topo = new Topo(rank);
    geom = new Geom(rank, topo);

    vecs  = new Vec[NQ];
    for(ki = 0; ki < NQ; ki++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &vecs[ki] );
    }

    LoadVecs(vecs, NQ, fieldname);

    // pack the time slice data into a dense matrix
    MatCreateDense(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NQ, topo->nDofs2G, NULL, &A);
    for(ki = 0; ki < NQ; ki++) {
        VecGetArray(vecs[ki], &vArray);
        MatSetValues(A, 1, &ki, topo->n2, topo->loc2, vArray, INSERT_VALUES);
        VecRestoreArray(vecs[ki], &vArray);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    MatCreateVecs(A, &rVec, &lVec);

    // compute the svd
    SVDCreate(MPI_COMM_WORLD, &svd);
    SVDSetOperator(svd, A);
    SVDSetFromOptions(svd);
    SVDSolve(svd);
    SVDGetConverged(svd, &nEig);

    if(!rank) cout << "number of eigenvalues: " << nEig << endl;

    for(ii = 0; ii < nEig; ii++) {
        SVDGetSingularTriplet(svd, ii, &sigma, lVec, rVec);
        if(!rank) cout << ii << "\tsigma: " << sigma << endl;
        sprintf(fieldname, "right_eigenvector");
        geom->write2(rVec, fieldname, ii);
    }

    SVDDestroy(&svd);
    MatDestroy(&A);
    VecDestroy(&lVec);
    VecDestroy(&rVec);
    for(ki = 0; ki < NQ; ki++) {
        VecDestroy(&vecs[ki]);
    }
    delete[] vecs;
    delete geom;
    delete topo;

    SlepcFinalize();

    return 0;
}
