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
//#define ZERO_FORM
#define TIMESTEP (24.0*60.0*60.0)

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
    int rank, size, ki, nEig, ii, index_i, index_f;
    double vSigmaInv, lambda_r, lambda_i;
    double* sigma;
    static char help[] = "petsc";
    char* fieldname = argv[1];
    char eigvecname[100];
    ofstream file;
    Topo* topo;
    Geom* geom;
    Vec* vecs;
    Vec lVec, rVec, vLocal, vr, vi;
    Mat X, XT, U, UT, VSI, XVSI, Atilde;
    PetscScalar* vArray;
    SVD svd;
    EPS eps;

    SlepcInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    if(!rank) cout << "solving the singular value decomposition...\n";

    SVDCreate(MPI_COMM_WORLD, &svd);
    SVDSetOperator(svd, X);
    SVDSetFromOptions(svd);
    SVDSolve(svd);
    SVDGetConverged(svd, &nEig);

    sigma = new double[nEig];

    if(!rank) cout << "number of eigenvalues: " << nEig << endl;
    for(ki = 0; ki < nEig; ki++) {
        SVDGetSingularTriplet(svd, ki, &sigma[ki], lVec, rVec);
        if(!rank) cout << ki << "\tsigma: " << sigma[ki] << endl;
        sprintf(eigvecname, "%s_svd", fieldname);
#ifdef ZERO_FORM
        geom->write0(rVec, eigvecname, ki);
#else
        geom->write2(rVec, eigvecname, ki);
#endif
    }

    // compute the dmd
    if(!rank) cout << "solving the dynamic mode decomposition.....\n";

#ifdef ZERO_FORM
    MatCreate(MPI_COMM_WORLD, &U);
    MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, nEig, topo->nDofs0G);
    MatSetType(U, MATMPIAIJ);
    MatMPIAIJSetPreallocation(U, topo->nDofs0G/size+1, PETSC_NULL, topo->nDofs0G, PETSC_NULL);
#else
    MatCreate(MPI_COMM_WORLD, &U);
    MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, nEig, topo->nDofs2G);
    MatSetType(U, MATMPIAIJ);
    MatMPIAIJSetPreallocation(U, topo->nDofs2G/size+1, PETSC_NULL, topo->nDofs2G, PETSC_NULL);
#endif
    MatCreate(MPI_COMM_WORLD, &VSI);
    MatSetSizes(VSI, PETSC_DECIDE, PETSC_DECIDE, nEig, nEig);
    MatSetType(VSI, MATMPIAIJ);
    MatMPIAIJSetPreallocation(VSI, nEig, PETSC_NULL, nEig, PETSC_NULL);

    for(ki = 0; ki < nEig; ki++) {
#ifdef ZERO_FORM
        VecScatterBegin(topo->gtol_0, rVec, vLocal, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_0, rVec, vLocal, INSERT_VALUES, SCATTER_FORWARD);
#else
        VecScatterBegin(topo->gtol_2, rVec, vLocal, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, rVec, vLocal, INSERT_VALUES, SCATTER_FORWARD);
#endif

        VecGetArray(vLocal, &vArray);
#ifdef ZERO_FORM
        MatSetValues(U, 1, &ki, topo->n0, topo->loc0, vArray, INSERT_VALUES);
#else
        MatSetValues(U, 1, &ki, topo->n2, topo->loc2, vArray, INSERT_VALUES);
#endif
        VecRestoreArray(vLocal, &vArray);

        VecGetOwnershipRange(lVec, &index_i, &index_f);
        VecGetArray(lVec, &vArray);
        for(ii = index_i; ii < index_f; ii++) {
            vSigmaInv = vArray[ii-index_i]/sigma[ii];
            MatSetValues(VSI, 1, &ki, 1, &ii, &vArray[ii-index_i], INSERT_VALUES);
        }
        VecRestoreArray(lVec, &vArray);
    }
    MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  U, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(VSI, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  VSI, MAT_FINAL_ASSEMBLY);

    MatTranspose(X, MAT_INITIAL_MATRIX, &XT);
    MatTranspose(U, MAT_INITIAL_MATRIX, &UT);

    MatMatMult(XT, VSI, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &XVSI);
    MatMatMult(U, XVSI, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Atilde);

    MatCreateVecs(Atilde, NULL, &vr);
    MatCreateVecs(Atilde, NULL, &vi);

    // compute the eigenvalues
    if(!rank) cout << "solving the eigenvalues....................\n";

    EPSCreate(MPI_COMM_WORLD, &eps);
    EPSSetOperators(eps, Atilde, NULL);
    EPSSetFromOptions(eps);
    EPSSolve(eps);

    for(ii = 0; ii < nEig; ii++) {
        EPSGetEigenpair(eps, ii, &lambda_r, &lambda_i, vr, vi);

        if(!rank) cout << ii << "\tlambda: " << lambda_r << " + " << lambda_i << "i\tfrequency: " << 
                          log(fabs(lambda_r))/TIMESTEP << "\ttimescale: " << TIMESTEP/log(fabs(lambda_r))/(24.0*60.0*60.0) << endl;
        MatMult(XVSI, vr, vecs[ii]);
        VecScale(vecs[ii], 1.0/lambda_r);
        sprintf(eigvecname, "%s_dmd", fieldname);
#ifdef ZERO_FORM
        geom->write0(vecs[ii], eigvecname, ii);
#else
        geom->write2(vecs[ii], eigvecname, ii);
#endif
        // imaginary component
        if(fabs(lambda_i) > 1.0e-16) {
            MatMult(XVSI, vi, vecs[ii]);
            VecScale(vecs[ii], 1.0/lambda_i);
            sprintf(eigvecname, "%s_dmd_i", fieldname);
#ifdef ZERO_FORM
            geom->write0(vecs[ii], eigvecname, ii);
#else
            geom->write2(vecs[ii], eigvecname, ii);
#endif
        }
    }

    SVDDestroy(&svd);
    EPSDestroy(&eps);
    MatDestroy(&X);
    MatDestroy(&U);
    MatDestroy(&XT);
    MatDestroy(&UT);
    MatDestroy(&VSI);
    MatDestroy(&XVSI);
    MatDestroy(&Atilde);
    VecDestroy(&lVec);
    VecDestroy(&rVec);
    VecDestroy(&vLocal);
    VecDestroy(&vr);
    VecDestroy(&vi);
    for(ki = 0; ki < NQ; ki++) {
        VecDestroy(&vecs[ki]);
    }
    delete[] vecs;
    delete geom;
    delete topo;
    delete[] sigma;

    SlepcFinalize();

    return 0;
}
