#include <iostream>
#include <cmath>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "VertOps.h"

#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define GRAVITY 9.80616
#define SCALE 1.0e+8

using namespace std;

VertOps::VertOps(Topo* _topo, Geom* _geom) {
    int N2;

    topo = _topo;
    geom = _geom;

    n2 = topo->elOrd*topo->elOrd;
    N2 = (n2 > 1) ? n2 : 4;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    Q = new Wii(node->q, geom);
    W = new M2_j_xy_i(edge);
    Q0 = new double[Q->nDofsI];
    QT = new double[Q->nDofsI];
    QB = new double[Q->nDofsI];
    Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    WtQW_2 = Alloc2D(W->nDofsJ, W->nDofsJ);
    WtQW_3 = Alloc2D(W->nDofsJ, W->nDofsJ);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatCreate(MPI_COMM_SELF, &VA);
    MatSetType(VA, MATSEQAIJ);
    MatSetSizes(VA, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VA, 2*N2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VB);
    MatSetType(VB, MATSEQAIJ);
    MatSetSizes(VB, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(VB, N2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VA_inv);
    MatSetType(VA_inv, MATSEQAIJ);
    MatSetSizes(VA_inv, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VA_inv, 2*N2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VB_inv);
    MatSetType(VB_inv, MATSEQAIJ);
    MatSetSizes(VB_inv, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(VB_inv, N2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VAB);
    MatSetType(VAB, MATSEQAIJ);
    MatSetSizes(VAB, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(VAB, 2*N2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VBA);
    MatSetType(VBA, MATSEQAIJ);
    MatSetSizes(VBA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VBA, 2*N2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VR);
    MatSetType(VR, MATSEQAIJ);
    MatSetSizes(VR, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VR, 2*N2, PETSC_NULL);

    // for the diagnosis of theta without boundary conditions
    MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+1)*n2, (geom->nk+1)*n2, N2, NULL, &VA2);
    MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+1)*n2, (geom->nk+0)*n2, 2*N2, NULL, &VAB2);

    vertOps();
}

VertOps::~VertOps() {
    delete[] Q0;
    delete[] QT;
    delete[] QB;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    delete[] WtQWflat;
    Free2D(W->nDofsJ, WtQW_2);
    Free2D(W->nDofsJ, WtQW_3);

    delete Q;
    delete W;

    delete edge;
    delete node;
    delete quad;

    MatDestroy(&V10);
    MatDestroy(&V01);
    MatDestroy(&V10_full);
    MatDestroy(&VA);
    MatDestroy(&VB);
    MatDestroy(&VA_inv);
    MatDestroy(&VB_inv);
    MatDestroy(&VAB);
    MatDestroy(&VBA);
    MatDestroy(&VR);

    MatDestroy(&VA2);
    MatDestroy(&VAB2);
}

/*
assemble the vertical gradient and divergence orientation matrices
V10 is the strong form vertical divergence from the linear to the
constant basis functions
*/
void VertOps::vertOps() {
    int ii, kk, rows[1], cols[2];
    double vm = -1.0;
    double vp = +1.0;
    Mat V10t;
    
    MatCreate(MPI_COMM_SELF, &V10);
    MatSetType(V10, MATSEQAIJ);
    MatSetSizes(V10, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(V10, 2, PETSC_NULL);

    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < n2; ii++) {
            rows[0] = kk*n2 + ii;
            if(kk > 0) {            // bottom of element
                cols[0] = (kk-1)*n2 + ii;
                MatSetValues(V10, 1, rows, 1, cols, &vm, INSERT_VALUES);
            }
            if(kk < geom->nk - 1) { // top of element
                cols[0] = (kk+0)*n2 + ii;
                MatSetValues(V10, 1, rows, 1, cols, &vp, INSERT_VALUES);
            }
        }
    }
    MatAssemblyBegin(V10, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V10, MAT_FINAL_ASSEMBLY);

    MatTranspose(V10, MAT_INITIAL_MATRIX, &V10t);
    MatDuplicate(V10t, MAT_DO_NOT_COPY_VALUES, &V01);
    MatZeroEntries(V01);
    MatAXPY(V01, -1.0, V10t, SAME_NONZERO_PATTERN);
    MatDestroy(&V10t);

    // create a second div operator over all vertical levels (for theta with non-homogeneous bcs)
    MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+1)*n2, 2*n2, NULL, &V10_full);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < n2; ii++) {
            rows[0] = kk*n2 + ii;
            // bottom of element
            cols[0] = (kk+0)*n2 + ii;
            MatSetValues(V10_full, 1, rows, 1, cols, &vm, INSERT_VALUES);
            // top of element
            cols[0] = (kk+1)*n2 + ii;
            MatSetValues(V10_full, 1, rows, 1, cols, &vp, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(V10_full, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V10_full, MAT_FINAL_ASSEMBLY);
}

/*
assemble a 3D mass matrix as a tensor product of 2 forms in the 
horizotnal and constant basis functions in the vertical
*/
void VertOps::AssembleConst(int ex, int ey, Mat B) {
    int ii, kk, ei, mp12;
    int *inds0;
    double det;
    int inds2k[99];

    ei    = ey*topo->nElsX + ex;
    inds0 = geom->elInds0_l(ex, ey);
    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(B);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii] *= geom->thickInv[kk][inds0[ii]];
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, INSERT_VALUES);
    }
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

/*
assemble a 3D mass matrix as a tensor product of 2 forms in the 
horizotnal and linear basis functions in the vertical
*/
void VertOps::AssembleLinear(int ex, int ey, Mat A) {
    int ii, kk, ei, mp12;
    int *inds0;
    double det;
    int inds2k[99];

    ei    = ey*topo->nElsX + ex;
    inds0 = geom->elInds0_l(ex, ey);
    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(A);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii]  = Q->A[ii]*(SCALE/det);
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            Q0[ii] *= 0.5*geom->thick[kk][inds0[ii]];
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinCon(int ex, int ey, Mat AB) {
    int ii, kk, ei, mp12;
    double det;
    int rows[99], cols[99];

    ei   = ey*topo->nElsX + ex;
    mp12 = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(AB);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            // multiply by the vertical jacobian, then scale the piecewise constant 
            // basis by the vertical jacobian, so do nothing 
            Q0[ii] *= 0.5;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }
        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, ADD_VALUES);
        }
    }
    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinCon2(int ex, int ey, Mat AB) {
    int ii, kk, ei, mp12;
    double det;
    int rows[99], cols[99];

    ei   = ey*topo->nElsX + ex;
    mp12 = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(AB);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            // multiply by the vertical jacobian, then scale the piecewise constant 
            // basis by the vertical jacobian, so do nothing 
            Q0[ii] *= 0.5;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }
        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, ADD_VALUES);
    }
    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRho2(int ex, int ey, Vec rho, Mat A) {
    int ii, jj, kk, ei, mp1, mp12;
    double det, rk;
    int inds2k[99];
    PetscScalar *rArray;

    ei   = ey*topo->nElsX + ex;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(A);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            // multuply by the vertical determinant to integrate, then
            // divide piecewise constant density by the vertical determinant,
            // so these cancel
            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rk += rArray[kk*n2+jj]*W->A[ii*n2+jj];
            }
            Q0[ii] *= 0.5*rk/det;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearInv(int ex, int ey, Mat A) {
    int kk, ii, rows[99], ei, *inds0, mp1, mp12;
    double det;

    ei    = ey*topo->nElsX + ex;
    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    for(kk = 0; kk < geom->nk-1; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii]  = Q->A[ii]*(SCALE/det);
            // for linear field we multiply by the vertical jacobian determinant when
            // integrating, and do no other trasformations for the basis functions
            Q0[ii] *= 0.5*(geom->thick[kk+0][inds0[ii]] + geom->thick[kk+1][inds0[ii]]);
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // take the inverse
        Inv(WtQW, WtQWinv, n2);
        // add to matrix
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, rows, W->nDofsJ, rows, WtQWinv, ADD_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithRhoInv(int ex, int ey, Vec rho, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk;
    int inds2k[99];
    PetscScalar* rArray;

    ei    = ey*topo->nElsX + ex;
    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii] *= geom->thickInv[kk][inds0[ii]];

            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rk += rArray[kk*n2+jj]*W->A[ii*n2+jj];
            }
            Q0[ii] *= rk/(geom->thick[kk][inds0[ii]]*det);
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWinv, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithRho(int ex, int ey, Vec rho, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk;
    int inds2k[99];
    PetscScalar* rArray;

    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii] *= geom->thickInv[kk][inds0[ii]];

            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rk += rArray[kk*n2+jj]*W->A[ii*n2+jj];
            }
            Q0[ii] *= rk/(geom->thick[kk][inds0[ii]]*det);
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConLinWithW(int ex, int ey, Vec velz, Mat BA) {
    int ii, jj, kk, ei, mp1, mp12, rows[99], cols[99];
    double wb, wt, det;
    PetscScalar* wArray;

    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(BA);

    VecGetArray(velz, &wArray);
    for(kk = 0; kk < geom->nk; kk++) {
        if(kk > 0) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Q0[ii] = Q->A[ii]*(SCALE/det);

                // multiply by the vertical jacobian, then scale the piecewise constant
                // basis by the vertical jacobian, so do nothing

                // interpolate the vertical velocity at the quadrature point
                wb = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    wb += wArray[(kk-1)*n2+jj]*W->A[ii*n2+jj];
                }
                Q0[ii] *= 0.5*wb/det; // scale by 0.5 outside for the 0.5 w^2
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
                cols[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, INSERT_VALUES);
        }

        if(kk < geom->nk - 1) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Q0[ii] = Q->A[ii]*(SCALE/det);

                // multiply by the vertical jacobian, then scale the piecewise constant
                // basis by the vertical jacobian, so do nothing

                // interpolate the vertical velocity at the quadrature point
                wt = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    wt += wArray[(kk+0)*n2+jj]*W->A[ii*n2+jj];
                }
                Q0[ii] *= 0.5*wt/det; // scale by 0.5 outside
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
                cols[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, INSERT_VALUES);
        }
    }
    VecRestoreArray(velz, &wArray);
    MatAssemblyBegin(BA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(BA, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRT(int ex, int ey, Vec rt, Mat A, bool do_internal) {
    int ii, jj, kk, ei, mp1, mp12;
    double det, rk;
    int inds2k[99];
    int* inds0 = geom->elInds0_l(ex, ey);
    PetscScalar *rArray;

    ei    = ey*topo->nElsX + ex;
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    // assemble the matrices
    VecGetArray(rt, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        if(kk > 0 && kk < geom->nk-1 && !do_internal) {
            continue;
        }

        // build the 2D mass matrix
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);

            // multuply by the vertical determinant to integrate, then
            // divide piecewise constant density by the vertical determinant,
            // so these cancel
            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rk += rArray[kk*n2+jj]*W->A[ii*n2+jj];
            }
            if(!do_internal) { // TODO: don't understand this scaling ?!?
                rk *= geom->thickInv[kk][inds0[ii]];
            }
            Q0[ii] *= 0.5*rk/det;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);
        }
    }
    VecRestoreArray(rt, &rArray);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithTheta(int ex, int ey, Vec theta, Mat A) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, tb, tt;
    int inds2k[99];
    PetscScalar *tArray;

    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(A);

    // assemble the matrices
    VecGetArray(theta, &tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            QB[ii]  = Q->A[ii]*(SCALE/det);
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            QB[ii] *= 0.5*geom->thick[kk][inds0[ii]];
            QT[ii]  = QB[ii];

            tb = tt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                tb += tArray[(kk+0)*n2+jj]*W->A[ii*n2+jj];
                tt += tArray[(kk+1)*n2+jj]*W->A[ii*n2+jj];
            }
            QB[ii] *= tb/det;
            QT[ii] *= tt/det;
        }

        // assemble the first basis function
        if(kk > 0) {
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QB, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QT, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);
        }
    }
    VecRestoreArray(theta, &tArray);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::Assemble_EOS_RHS(int ex, int ey, Vec rt, Vec eos_rhs, double factor, double exponent) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk, fac;
    double rtq[99], rtj[99];
    PetscScalar *rArray, *eArray;

    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    fac = factor;

    VecZeroEntries(eos_rhs);

    // assemble the eos rhs vector
    VecGetArray(rt, &rArray);
    VecGetArray(eos_rhs, &eArray);
    for(kk = 0; kk < geom->nk; kk++) {
        // test function (0.5 at each vertical quadrature point) by jacobian determinant
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = 0.5*Q->A[ii]*SCALE;
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);

        // interpolate
        for(ii = 0; ii < mp12; ii++) {
            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rk += W->A[ii*n2+jj]*rArray[kk*n2+jj];
            }
            // scale by matric term and vertical basis function at quadrature point ii
            det = geom->det[ei][ii];
            rk *= 1.0/(det*geom->thick[kk][inds0[ii]]);
            rtq[ii] = fac*pow(rk, exponent);
        }

        for(jj = 0; jj < n2; jj++) {
            rtj[jj] = 0.0;
            for(ii = 0; ii < mp12; ii++) {
                rtj[jj] += WtQ[jj*mp12+ii]*rtq[ii];
            }
            // x 2 (once for each vertical quadrature point)
            rtj[jj] *= 2.0;
        }

        // add to the vector
        for(jj = 0; jj < n2; jj++) {
            eArray[kk*n2+jj] = rtj[jj];
        }
    }
    VecRestoreArray(rt, &rArray);
    VecRestoreArray(eos_rhs, &eArray);
}

void VertOps::AssembleConstInv(int ex, int ey, Mat B) {
    int ii, kk, ei, mp1, mp12;
    int *inds0;
    double det;
    int inds2k[99];

    ei    = ey*topo->nElsX + ex;
    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(B);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii]  = Q->A[ii]*(SCALE/det);
            Q0[ii] *= geom->thickInv[kk][inds0[ii]];
        }
        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWinv, ADD_VALUES);
    }
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

/*
Assemble the top layer rayleigh friction operator
*/
void VertOps::AssembleRayleigh(int ex, int ey, Mat A) {
    int ii, ei, mp12;
    int *inds0;
    double det;
    int inds2k[99];

    ei    = ey*topo->nElsX + ex;
    inds0 = geom->elInds0_l(ex, ey);
    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(A);

    // top level
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii]  = Q->A[ii]*(SCALE/det);
        // assembly the contributions from the top two levels only
        Q0[ii] *= 0.5*(geom->thick[geom->nk-1][inds0[ii]] + geom->thick[geom->nk-2][inds0[ii]]);
    }
    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        // interface between top two levels
        inds2k[ii] = ii + (geom->nk-2)*W->nDofsJ;
    }
    MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);

    // second from top level
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii]  = Q->A[ii]*(SCALE/det);
        // assembly the contributions from the top two levels only
        Q0[ii] *= 0.25*(geom->thick[geom->nk-2][inds0[ii]] + geom->thick[geom->nk-3][inds0[ii]]);
    }
    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        // interface between top two levels
        inds2k[ii] = ii + (geom->nk-3)*W->nDofsJ;
    }
    MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);

    // third from top level
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii]  = Q->A[ii]*(SCALE/det);
        // assembly the contributions from the top two levels only
        Q0[ii] *= 0.125*(geom->thick[geom->nk-3][inds0[ii]] + geom->thick[geom->nk-4][inds0[ii]]);
    }
    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        // interface between top two levels
        inds2k[ii] = ii + (geom->nk-4)*W->nDofsJ;
    }
    MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConLin(int ex, int ey, Mat BA) {
    int ii, kk, ei, mp1, mp12, rows[99], cols[99];
    double det;

    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(BA);

    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            Q0[ii] *= 0.5;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
                cols[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, ADD_VALUES);
        }

        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
                cols[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, ADD_VALUES);
        }
    }
    MatAssemblyBegin(BA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(BA, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithTheta(int ex, int ey, Vec theta, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, tb, tt;
    int inds2k[99];
    PetscScalar* tArray;

    ei    = ey*topo->nElsX + ex;
    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(theta, &tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii]  = Q->A[ii]*(SCALE/det);
            Q0[ii] *= geom->thickInv[kk][inds0[ii]];

            tb = tt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                //gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                //tb += tArray[(kk+0)*n2+jj]*gamma;
                //tt += tArray[(kk+1)*n2+jj]*gamma;
                tb += tArray[(kk+0)*n2+jj]*W->A[ii*n2+jj];
                tt += tArray[(kk+1)*n2+jj]*W->A[ii*n2+jj];
            }
            Q0[ii] *= 0.5*(tb + tt)/det;
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);
    }
    VecRestoreArray(theta, &tArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void _matvec(double** A, double* x, double* b, int ni, int nj) {
    int ii, jj;
    for(ii = 0; ii < ni; ii++) {
        b[ii] = 0.0;
        for(jj = 0; jj < nj; jj++) {
            b[ii] += A[ii][jj]*x[jj];
        }
    }
}

void VertOps::Assemble_EOS_Residual(int ex, int ey, Vec rt, Vec exner, Vec eos_rhs) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk, ek;
    double rtq[99], rtj[99];
    PetscScalar *rArray, *eArray, *fArray;

    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    VecZeroEntries(eos_rhs);

    // assemble the eos rhs vector
    VecGetArray(rt, &rArray);
    VecGetArray(exner, &eArray);
    VecGetArray(eos_rhs, &fArray);
    for(kk = 0; kk < geom->nk; kk++) {
        // test function (0.5 at each vertical quadrature point) by jacobian determinant
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = 0.5*Q->A[ii]*SCALE;
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);

        // interpolate
        for(ii = 0; ii < mp12; ii++) {
            rk = ek = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rk += W->A[ii*n2+jj]*rArray[kk*n2+jj];
                ek += W->A[ii*n2+jj]*eArray[kk*n2+jj];
            }
            // scale by matric term and vertical basis function at quadrature point ii
            det = geom->det[ei][ii];
            rk *= 1.0/(det*geom->thick[kk][inds0[ii]]);
            ek *= 1.0/(det*geom->thick[kk][inds0[ii]]);
//if(ek<0.0){ek=1.0e-8;/*cout<<"ERROR! -ve exner pressure: "<<ek<<endl;*/}
//if(rk<0.0){rk=1.0e-8;/*cout<<"ERROR! -ve potential temp: "<<rk<<endl;*/}

            rtq[ii] = log(ek) - (RD/CV)*log(rk) - log(CP) - (RD/CV)*log(RD/P0);
        }

        for(jj = 0; jj < n2; jj++) {
            rtj[jj] = 0.0;
            for(ii = 0; ii < mp12; ii++) {
                rtj[jj] += WtQ[jj*mp12+ii]*rtq[ii];
            }
            // x 2 (once for each vertical quadrature point)
            rtj[jj] *= 2.0;
        }

        // add to the vector
        for(jj = 0; jj < n2; jj++) {
            fArray[kk*n2+jj] = rtj[jj];
        }
    }
    VecRestoreArray(rt, &rArray);
    VecRestoreArray(exner, &eArray);
    VecRestoreArray(eos_rhs, &fArray);
}

void VertOps::Assemble_EOS_BlockInv(int ex, int ey, Vec rt, Vec theta, Mat B) {
    int ii, jj, kk, mp1, mp12, ei;
    int *inds0;
    int inds2k[99];
    double tk, tkp1, gamma, det;
    double *BinvB = new double[W->nDofsJ*W->nDofsJ];
    double *B_BinvB = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar *tArray, *_tArray;

    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(B);

    VecGetArray(rt, &tArray);
    if(theta) VecGetArray(theta, &_tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        // assemble the layer-wise inverse matrix
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            Q0[ii] *= geom->thickInv[kk][inds0[ii]];

            tk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                tk += tArray[kk*n2+jj]*gamma;
            }
            Q0[ii] *= tk/(geom->thick[kk][inds0[ii]]*det);
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);

        // assemble the layer-wise mass matrix
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            Q0[ii] *= geom->thickInv[kk][inds0[ii]];
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // multiply operators
        Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQWinv, WtQW, BinvB);
        Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW, BinvB, B_BinvB);

        // rho correction
        if(theta) {
            Inv(WtQW, WtQWinv, n2);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Q0[ii]  = Q->A[ii]*(SCALE/det);
                Q0[ii] *= geom->thickInv[kk][inds0[ii]];

                tk = tkp1 = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    tk   += _tArray[(kk+0)*n2+jj]*gamma;
                    tkp1 += _tArray[(kk+1)*n2+jj]*gamma;
                }
                Q0[ii] *= 0.5*(tk + tkp1)/det;
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW, WtQWinv, BinvB);
            for(ii = 0; ii < n2; ii++) {
                BinvB[ii*n2+ii] += 1.0;
            }
            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, BinvB, B_BinvB, WtQW);
            for(ii = 0; ii < n2; ii++) {
                for(jj = 0; jj < n2; jj++) {
                    B_BinvB[ii*n2+jj] = WtQW[ii*n2+jj];
                }
            }
        }

        Inv(B_BinvB, WtQWinv, n2);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWinv, ADD_VALUES);
    }
    VecRestoreArray(rt, &tArray);
    if(theta) VecRestoreArray(theta, &_tArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  B, MAT_FINAL_ASSEMBLY);

    delete[] BinvB;
    delete[] B_BinvB;
}

void VertOps::AssembleLinearWithRayleighInv(int ex, int ey, double dt_fric, Mat A) {
    int kk, ii, rows[99], ei, *inds0, mp1, mp12;
    double det;

    ei    = ey*topo->nElsX + ex;
    inds0 = geom->elInds0_l(ex, ey);
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    for(kk = 0; kk < geom->nk-1; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii]  = Q->A[ii]*(SCALE/det);
            Q0[ii] *= 0.5*(geom->thick[kk+0][inds0[ii]] + geom->thick[kk+1][inds0[ii]]);
            if(kk == geom->nk-1)      Q0[ii] *= (1.0 + 1.00*dt_fric);
            else if(kk == geom->nk-2) Q0[ii] *= (1.0 + 0.50*dt_fric);
            else if(kk == geom->nk-3) Q0[ii] *= (1.0 + 0.25*dt_fric);
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // take the inverse
        Inv(WtQW, WtQWinv, n2);
        // add to matrix
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, rows, W->nDofsJ, rows, WtQWinv, ADD_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRho2_up(int ex, int ey, Vec rho, Mat A, double dt, Vec* uhl) {
    int ii, jj, kk, ei, _n, _n2, mp1, mp12;
    double det, **J, rk, gamma, ug[2], ul[2], _ex[99], _ey[99];
    int* inds_0;
    int inds2k[99];
    PetscScalar *rArray, *uArray;

    ei   = ey*topo->nElsX + ex;
    _n   = topo->elOrd;
    _n2  = _n*_n;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;

    inds_0 = geom->elInds0_l(ex, ey);

    MatZeroEntries(A);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(uhl[kk], &uArray);

        // build the 2D mass matrix
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            // multuply by the vertical determinant to integrate, then
            // divide piecewise constant density by the vertical determinant,
            // so these cancel
            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            Q0[ii] *= 0.5*rk/det;

            // upwinded test functions
            J = geom->J[ei][ii];
            geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, ug);
            // map velocity to local element coordinates
	    ul[0] = (+J[1][1]*ug[0] - J[0][1]*ug[1])/det;
            ul[1] = (-J[1][0]*ug[0] + J[0][0]*ug[1])/det;
            ul[0] *= geom->thickInv[kk][inds_0[ii]];
            ul[1] *= geom->thickInv[kk][inds_0[ii]];
            for(jj = 0; jj < _n; jj++) {
                _ex[jj] = edge->eval(quad->x[ii%mp1] + dt*ul[0], jj);
                _ey[jj] = edge->eval(quad->x[ii/mp1] + dt*ul[1], jj);
            }
            for(jj = 0; jj < _n2; jj++) {
                Wt[jj*mp12+ii] = _ex[jj%_n]*_ey[jj/_n];
            }
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQW, ADD_VALUES);

        VecGetArray(uhl[kk], &uArray);
    }
    VecRestoreArray(rho, &rArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
}

void VertOps::AssembleLinCon2_up(int ex, int ey, Mat AB, double dt, Vec* uhl) {
    int ii, jj, kk, ei, _n, _n2, mp1, mp12;
    double det, **J, ug[2], ul[2], _ex[99], _ey[99];
    int* inds_0;
    int rows[99], cols[99];
    PetscScalar* uArray;

    ei   = ey*topo->nElsX + ex;
    _n   = topo->elOrd;
    _n2  = _n*_n;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;

    inds_0 = geom->elInds0_l(ex, ey);

    MatZeroEntries(AB);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(uhl[kk], &uArray);
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = Q->A[ii]*(SCALE/det);
            // multiply by the vertical jacobian, then scale the piecewise constant 
            // basis by the vertical jacobian, so do nothing 
            Q0[ii] *= 0.5;

            // upwinded test functions
            J = geom->J[ei][ii];
            geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, ug);
            // map velocity to local element coordinates
	    ul[0] = (+J[1][1]*ug[0] - J[0][1]*ug[1])/det;
            ul[1] = (-J[1][0]*ug[0] + J[0][0]*ug[1])/det;
            ul[0] *= geom->thickInv[kk][inds_0[ii]];
            ul[1] *= geom->thickInv[kk][inds_0[ii]];
            for(jj = 0; jj < _n; jj++) {
                _ex[jj] = edge->eval(quad->x[ii%mp1] + dt*ul[0], jj);
                _ey[jj] = edge->eval(quad->x[ii/mp1] + dt*ul[1], jj);
            }
            for(jj = 0; jj < _n2; jj++) {
                Wt[jj*mp12+ii] = _ex[jj%_n]*_ey[jj/_n];
            }
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }
        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQW, ADD_VALUES);

        VecGetArray(uhl[kk], &uArray);
    }
    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
}

double compute_k_T(double phi, double exner, double exner_s, double theta) {
    double p        = pow(exner/CP, CP/RD);
    double ps       = pow(exner_s/CP, CP/RD);
    double sigma    = p/ps;
    double sigma_b  = 0.7;
    double theta_eq;
    double k_a      = 2.8935185185185185e-07;
    double k_s      = 2.8935185185185184e-06;
    double k_t      = 0.0;
    double t_eq     = 315.0 - 60.0*sin(phi)*sin(phi) - 10.0*log(p)*cos(phi)*cos(phi);

    t_eq *= pow(p, RD/CP);
    if(t_eq < 200.0) t_eq = 200.0;

    theta_eq = t_eq*pow(1.0/p, RD/CP);

    if(sigma > sigma_b) {
        k_t  = (k_s - k_a)*(sigma - sigma_b)/(1.0 - sigma_b);
        k_t *= pow(cos(phi), 4.0);
    }
    k_t += k_a;

    return k_t*(theta - theta_eq);
}

void VertOps::AssembleTempForcing_HS(int ex, int ey, Vec exner, Vec theta, Vec rho, Vec vec) {
    int ei     = ey*topo->nElsX + ex;
    int mp1    = quad->n + 1;
    int mp12   = mp1*mp1;
    int* inds0 = geom->elInds0_l(ex, ey);
    double _e[99], _r[99], _tb[99], _tt[99], _es[99], k_t[99], det;
    PetscScalar *eArray, *tArray, *rArray, *vArray;

    VecZeroEntries(vec);

    VecGetArray(exner, &eArray);
    VecGetArray(theta, &tArray);
    VecGetArray(rho,   &rArray);
    VecGetArray(vec,   &vArray);

    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];

            _e[ii] = _tb[ii] = _tt[ii] = _r[ii] = _es[ii] = 0.0;
            for(int ll = 0; ll < n2; ll++) {
                _e[ii]  += eArray[kk*n2+ll]*W->A[ii*n2+ll];
                _tb[ii] += tArray[(kk+0)*n2+ll]*W->A[ii*n2+ll];
                _tt[ii] += tArray[(kk+1)*n2+ll]*W->A[ii*n2+ll];
                _r[ii]  += rArray[kk*n2+ll]*W->A[ii*n2+ll];
                _es[ii] += eArray[ll]*W->A[ii*n2+ll];
            }
            _e[ii]  /= (det*geom->thick[kk][inds0[ii]]);
            _tb[ii] /= (det);
            _tt[ii] /= (det);
            _r[ii]  /= (det*geom->thick[kk][inds0[ii]]);
            _es[ii] /= (det*geom->thick[0][inds0[ii]]);

            k_t[ii] = compute_k_T(geom->s[inds0[ii]][1], _e[ii], _es[ii], 0.5*(_tb[ii] + _tt[ii]));
        }
        for(int jj = 0; jj < n2; jj++) {
            for(int ii = 0; ii < mp12; ii++) {
                vArray[kk*n2+jj] += Wt[jj*mp12+ii] * Q->A[ii] * SCALE * _r[ii] * k_t[ii];
            }
        }
    }
    VecRestoreArray(exner, &eArray);
    VecRestoreArray(theta, &tArray);
    VecRestoreArray(rho,   &rArray);
    VecRestoreArray(vec,   &vArray);
}

double compute_sigma(double phi, double z) {
    double _GRAVITY = 9.80616;
    double _RD      = 287.0;
    double _TE      = 310.0;
    double _TP      = 240.0;
    double _T0      = (0.5*(_TE + _TP));
    double _GAMMA   = 0.005;
    double _KP      = 3.0;

    double A        = 1.0/_GAMMA;
    double B        = (_TE - _TP)/((_TE + _TP)*_TP);
    double H        = _RD*_T0/_GRAVITY;
    double b        = 2.0;
    double fac      = z/(b*H);
    double fac2     = fac*fac;

    double int_torr_1 = A*(exp(_GAMMA*z/_T0) - 1.0) + B*z*exp(-fac2);

    double C        = 0.5*(_KP + 2.0)*(_TE - _TP)/(_TE*_TP);

    double int_torr_2 = C*z*exp(-fac2);

    double cp       = cos(phi);
    double cpk      = pow(cp, _KP);
    double cpkp2    = pow(cp, _KP+2.0);
    double _fac     = cpk - (_KP/(_KP+2.0))*cpkp2;

    double sigma    = exp(-_GRAVITY*int_torr_1/_RD + _GRAVITY*int_torr_2*_fac/_RD);

    double SIGMA_B  = 0.7;//3000.0;

    double _sigma   = (sigma - SIGMA_B)/(1.0 - SIGMA_B);

    if(_sigma < 0.0) _sigma = 0.0;

    return _sigma;
}

/*
#define K_A 2.8935185185185185e-07
#define K_S 2.8935185185185184e-06

void VertOps::AssembleConLinWithRho(int ex, int ey, Mat BA, Vec rho) {
    int ii, jj, kk, ei, mp1, mp12, *inds0;
    double det, tk, gamma, k_theta = 0.0, sigma;
    int rows[99], cols[99];
    PetscScalar* tArray;

    ei   = ey*topo->nElsX + ex;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;
    inds0 = geom->elInds0_l(ex, ey);

    MatZeroEntries(BA);

    // assemble the matrices
    VecGetArray(rho, &tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii] = 0.5*Q->A[ii]*(SCALE/det);

            tk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                tk += tArray[kk*n2+jj]*gamma;
            }
            Q0[ii] *= tk/det;
            Q0[ii] *= 1.0/geom->thick[kk][inds0[ii]];

            // bottom level
            //if(geom->levs[kk+0][inds0[ii]] > SIGMA_B) {
            sigma = compute_sigma(geom->s[inds0[ii]][1], geom->levs[kk+0][inds0[ii]]);
            if(sigma < 0.0) {
                k_theta = 0.0;
            } else {
                //k_theta = (SIGMA_B - sigma)/SIGMA_B;
                k_theta  = sigma;
                k_theta *= pow(cos(geom->s[inds0[ii]][1]), 4);
                k_theta *= (K_S - K_A);
            }
            k_theta += K_A;
            QB[ii] = k_theta*Q0[ii];

            // top level
            //if(geom->levs[kk+1][inds0[ii]] > SIGMA_B) {
            sigma = compute_sigma(geom->s[inds0[ii]][1], geom->levs[kk+1][inds0[ii]]);
            if(sigma < 0.0) {
                k_theta = 0.0;
            } else {
                //k_theta = (SIGMA_B - geom->levs[kk+1][inds0[ii]])/SIGMA_B;
                k_theta  = sigma;
                k_theta *= pow(cos(geom->s[inds0[ii]][1]), 4);
                k_theta *= (K_S - K_A);
            }
            k_theta += K_A;
            QT[ii] = k_theta*Q0[ii];
        }

        // bottom level
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QB, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);

        // top level
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QT, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &tArray);

    MatAssemblyBegin(BA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(BA, MAT_FINAL_ASSEMBLY);
}
*/
