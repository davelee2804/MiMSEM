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

#define OFFSET 48
//#define OFFSET 9
//#define NQ 12
#define NQ 54
//#define TIMESTEP (24.0*60.0*60.0)
//#define TIMESTEP (12.0*60.0*60.0)
#define TIMESTEP (4.0*60.0*60.0)
#define SHIFT 1

using namespace std;

void LoadVecs(Vec* vecs, int nk, char* fieldname) {
    int ki;
    char filename[100];
    PetscViewer viewer;

    //for(ki = 0; ki < nk; ki++) {
    for(ki = 0; ki < nk; ki += SHIFT) {
        sprintf(filename, "output/%s_%.4u.vec", fieldname, ki+OFFSET);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
        VecLoad(vecs[ki], viewer);
        PetscViewerDestroy(&viewer);
    }
}

int main(int argc, char** argv) {
    int rank, size, ki, kj, nEig, ii, index_i, index_f;
    double vSigmaInv, lambda_r, lambda_i, freq;
    double* sigma;
    static char help[] = "petsc";
    char* fieldname = argv[1];
    int form = atoi(argv[2]);
    char eigvecname[100];
    ofstream file;
    Topo* topo;
    Geom* geom;
    Vec* vecs;
    Vec lVec, rVec, vLocal, vr, vi;
    Mat Xi, Xj, XT, U, UT, VSI, XVSI, Atilde;
    PetscScalar* vArray;
    SVD svd;
    EPS eps;
    int nk, nkl, nDofsKG, *lock;
    VecScatter gtol_k;

    SlepcInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    topo = new Topo(rank);
    geom = new Geom(rank, topo);

    if(form == 2) {
        nk      = topo->n2;
        nkl     = topo->n2l;
        nDofsKG = topo->nDofs2G;
        lock    = topo->loc2;
        gtol_k  = topo->gtol_2;
    } else if(form == 1) {
        nk      = topo->n1;
        nkl     = topo->n1l;
        nDofsKG = topo->nDofs1G;
        lock    = topo->loc1;
        gtol_k  = topo->gtol_1;
    } else if(form == 0) {
        nk      = topo->n0;
        nkl     = topo->n0l;
        nDofsKG = topo->nDofs0G;
        lock    = topo->loc0;
        gtol_k  = topo->gtol_0;
    } else {
        if(!rank) cout << "invalid basis form! must be 0, 1 or 2.\n";
        abort();
    }

    vecs  = new Vec[NQ];
    for(ki = 0; ki < NQ; ki++) {
        VecCreateMPI(MPI_COMM_WORLD, nk, nDofsKG, &vecs[ki]);
    }
    VecCreateSeq(MPI_COMM_SELF, nk, &vLocal);
    LoadVecs(vecs, NQ, fieldname);

    // pack the time slice data into a dense matrix
    MatCreateDense(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NQ, nDofsKG, NULL, &Xi);
    MatCreateDense(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NQ, nDofsKG, NULL, &Xj);

    for(ki = 0; ki < NQ; ki++) {
        kj = ki - 1;

        VecScatterBegin(gtol_k, vecs[ki], vLocal, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  gtol_k, vecs[ki], vLocal, INSERT_VALUES, SCATTER_FORWARD);

        VecGetArray(vLocal, &vArray);
        if(ki < NQ - 1) MatSetValues(Xi, 1, &ki, nk, lock, vArray, INSERT_VALUES);
        if(ki > 0)      MatSetValues(Xj, 1, &kj, nk, lock, vArray, INSERT_VALUES);
        VecRestoreArray(vLocal, &vArray);
    }
    MatAssemblyBegin(Xi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Xi, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Xj, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Xj, MAT_FINAL_ASSEMBLY);
    MatCreateVecs(Xi, &rVec, &lVec);

    // compute the svd
    if(!rank) cout << "solving the singular value decomposition...\n";

    SVDCreate(MPI_COMM_WORLD, &svd);
    SVDSetOperator(svd, Xi);
    SVDSetFromOptions(svd);
    SVDSolve(svd);
    SVDGetConverged(svd, &nEig);

    sigma = new double[nEig];

    if(!rank) cout << "number of eigenvalues: " << nEig << endl;
    for(ki = 0; ki < nEig; ki++) {
        SVDGetSingularTriplet(svd, ki, &sigma[ki], lVec, rVec);
        if(!rank) cout << ki << "\tsigma: " << sigma[ki] << endl;
        sprintf(eigvecname, "%s_svd", fieldname);

        if(form == 2) {
            geom->write2(rVec, eigvecname, ki);
        } else if(form == 1) {
            geom->write1(rVec, eigvecname, ki);
        } else if(!form) {
            geom->write0(rVec, eigvecname, ki);
        }
    }

    // compute the dmd
    if(!rank) cout << "solving the dynamic mode decomposition.....\n";

    MatCreate(MPI_COMM_WORLD, &U);
    MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, NQ-1, nDofsKG);
    MatSetType(U, MATMPIAIJ);
    MatMPIAIJSetPreallocation(U, nDofsKG/size+1, PETSC_NULL, nDofsKG, PETSC_NULL);

    MatCreate(MPI_COMM_WORLD, &VSI);
    MatSetSizes(VSI, PETSC_DECIDE, PETSC_DECIDE, NQ-1, NQ-1);
    MatSetType(VSI, MATMPIAIJ);
    MatMPIAIJSetPreallocation(VSI, NQ-1, PETSC_NULL, NQ-1, PETSC_NULL);

    MatZeroEntries(U);
    MatZeroEntries(VSI);

    for(ki = 0; ki < nEig; ki++) {
        SVDGetSingularTriplet(svd, ki, &sigma[ki], lVec, rVec);

        VecScatterBegin(gtol_k, rVec, vLocal, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  gtol_k, rVec, vLocal, INSERT_VALUES, SCATTER_FORWARD);

        VecGetArray(vLocal, &vArray);
        MatSetValues(U, 1, &ki, nk, lock, vArray, INSERT_VALUES);
        VecRestoreArray(vLocal, &vArray);

        VecGetOwnershipRange(lVec, &index_i, &index_f);
        VecGetArray(lVec, &vArray);
        for(ii = index_i; ii < index_f; ii++) {
            //vSigmaInv = vArray[ii-index_i]/sigma[ii];
            vSigmaInv = vArray[ii-index_i]/sigma[ki]; // TODO: check this!
//if(ii<nEig) {
            //MatSetValues(VSI, 1, &ki, 1, &ii, &vArray[ii-index_i], INSERT_VALUES);
            MatSetValues(VSI, 1, &ki, 1, &ii, &vSigmaInv, INSERT_VALUES);
//}
        }
        VecRestoreArray(lVec, &vArray);
    }
    MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  U, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(VSI, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  VSI, MAT_FINAL_ASSEMBLY);

    MatTranspose(Xj, MAT_INITIAL_MATRIX, &XT);
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

        freq = log(fabs(lambda_r))/TIMESTEP;
        if(!rank) cout << ii << "\tlambda: " << lambda_r << " + " << lambda_i << "i\tfrequency: " << 
                          freq << "\ttimescale: " << 1.0/freq/(24.0*60.0*60.0) << endl;
        MatMult(XVSI, vr, vecs[ii]);
        VecScale(vecs[ii], 1.0/lambda_r);
        sprintf(eigvecname, "%s_dmd", fieldname);

        if(form == 2) {
            geom->write2(vecs[ii], eigvecname, ii);
        } else if(form == 1) {
            geom->write1(vecs[ii], eigvecname, ii);
        } else if(!form) {
            geom->write0(vecs[ii], eigvecname, ii);
        }

        // imaginary component
        if(fabs(lambda_i) > 1.0e-16) {
            MatMult(XVSI, vi, vecs[ii]);
            VecScale(vecs[ii], 1.0/lambda_i);
            sprintf(eigvecname, "%s_dmd_i", fieldname);

            if(form == 2) {
                geom->write2(vecs[ii], eigvecname, ii);
            } else if(form == 1) {
                geom->write1(vecs[ii], eigvecname, ii);
            } else if(!form) {
                geom->write0(vecs[ii], eigvecname, ii);
            }
        }
    }

    SVDDestroy(&svd);
    EPSDestroy(&eps);
    MatDestroy(&Xi);
    MatDestroy(&Xj);
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
