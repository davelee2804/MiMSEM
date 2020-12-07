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
    int N2, ii;

    topo = _topo;
    geom = _geom;

    n2 = topo->elOrd*topo->elOrd;
    N2 = (n2 > 1) ? n2 : 4;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    Q = new Wii(node->q, geom);
    W = new M2_j_xy_i(edge);
    Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    QT = Alloc2D(Q->nDofsI, Q->nDofsJ);
    QB = Alloc2D(Q->nDofsI, Q->nDofsJ);
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

    // for the density and temperature equation preconditioner
    MatCreate(MPI_COMM_SELF, &VAB_w);
    MatSetType(VAB_w, MATSEQAIJ);
    MatSetSizes(VAB_w, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(VAB_w, 4*N2, PETSC_NULL);

    // for the diagnosis of theta without boundary conditions
    //MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+1)*n2, (geom->nk+1)*n2, N2, NULL, &VA2);
    MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+1)*n2, (geom->nk+1)*n2, 4*N2, NULL, &VA2);
    //MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+1)*n2, (geom->nk+0)*n2, 2*N2, NULL, &VAB2);
    MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+1)*n2, (geom->nk+0)*n2, 4*N2, NULL, &VAB2);
    MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+1)*n2, 2*N2, NULL, &VBA2);

    vertOps();

    A3 = new double*[3];
    A3inv = new double*[3];
    for(ii = 0; ii < 3; ii++) {
        A3[ii] = new double[3];
        A3inv[ii] = new double[3];
    }
    rho_q = new double[(quad->n+1)*(quad->n+1)*geom->nk];
    rho_e = new double[(quad->n+1)*(quad->n+1)*(geom->nk-1)];

    Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Qab = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Qba = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Qbb = Alloc2D(Q->nDofsI, Q->nDofsJ);
}

VertOps::~VertOps() {
    int ii;

    Free2D(Q->nDofsI, Q0);
    Free2D(Q->nDofsI, QT);
    Free2D(Q->nDofsI, QB);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    delete[] WtQWflat;
    Free2D(W->nDofsJ, WtQW_2);
    Free2D(W->nDofsJ, WtQW_3);

    delete edge;
    delete node;
    delete quad;

    delete Q;
    delete W;

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

    MatDestroy(&VAB_w);

    MatDestroy(&VA2);
    MatDestroy(&VAB2);
    MatDestroy(&VBA2);

    for(ii = 0; ii < 3; ii++) {
        delete[] A3[ii];
        delete[] A3inv[ii];
    }
    delete[] A3;
    delete[] A3inv;
    delete[] rho_q;
    delete[] rho_e;

    Free2D(Q->nDofsI, Qaa);
    Free2D(Q->nDofsI, Qab);
    Free2D(Q->nDofsI, Qba);
    Free2D(Q->nDofsI, Qbb);
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
    inds0 = topo->elInds0_l(ex, ey);
    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(B);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            //Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
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
    inds0 = topo->elInds0_l(ex, ey);
    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(A);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            //Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det/det);
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            Q0[ii][ii] *= 0.5*geom->thick[kk][inds0[ii]];
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
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
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // multiply by the vertical jacobian, then scale the piecewise constant 
            // basis by the vertical jacobian, so do nothing 
            Q0[ii][ii] *= 0.5;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }
        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
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
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // multiply by the vertical jacobian, then scale the piecewise constant 
            // basis by the vertical jacobian, so do nothing 
            Q0[ii][ii] *= 0.5;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }
        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
    }
    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A, bool do_internal) {
    int ii, kk, ei, mp1, mp12;
    double det, rk;
    int inds2k[99];
    int* inds0 = topo->elInds0_l(ex, ey);
    PetscScalar *rArray;

    ei   = ey*topo->nElsX + ex;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(A);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        if(kk > 0 && kk < geom->nk-1 && !do_internal) {
            continue;
        }

        // build the 2D mass matrix
        VecGetArray(rho[kk], &rArray);
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);

            // multuply by the vertical determinant to integrate, then
            // divide piecewise constant density by the vertical determinant,
            // so these cancel
            geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rArray, &rk);
            if(!do_internal) { // TODO: don't understand this scaling?!?
                rk *= 1.0/geom->thick[kk][inds0[ii]];
            }
            Q0[ii][ii] *= 0.5*rk;
        }
        VecRestoreArray(rho[kk], &rArray);

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRho2(int ex, int ey, Vec rho, Mat A) {
    int ii, jj, kk, ei, mp1, mp12;
    double det, rk, gamma;
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
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // multuply by the vertical determinant to integrate, then
            // divide piecewise constant density by the vertical determinant,
            // so these cancel
            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            Q0[ii][ii] *= 0.5*rk/det;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearInv(int ex, int ey, Mat A) {
    int kk, ii, rows[99], ei, *inds0, mp1, mp12;
    double det;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    for(kk = 0; kk < geom->nk-1; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            // for linear field we multiply by the vertical jacobian determinant when
            // integrating, and do no other trasformations for the basis functions
            Q0[ii][ii] *= 0.5*(geom->thick[kk+0][inds0[ii]] + geom->thick[kk+1][inds0[ii]]);
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // take the inverse
        Inv(WtQW, WtQWinv, n2);
        // add to matrix
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, rows, W->nDofsJ, rows, WtQWflat, ADD_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithRhoInv(int ex, int ey, Vec rho, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk, gamma;
    int inds2k[99];
    PetscScalar* rArray;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            //Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            Q0[ii][ii] *= rk/(geom->thick[kk][inds0[ii]]*det);
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithRho(int ex, int ey, Vec rho, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk, gamma;
    int inds2k[99];
    PetscScalar* rArray;

    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            Q0[ii][ii] *= rk/(geom->thick[kk][inds0[ii]]*det);
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConLinWithW(int ex, int ey, Vec velz, Mat BA) {
    int ii, jj, kk, ei, mp1, mp12, rows[99], cols[99];
    double wb, wt, gamma, det;
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
                //Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);
                Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);

                // multiply by the vertical jacobian, then scale the piecewise constant
                // basis by the vertical jacobian, so do nothing

                // interpolate the vertical velocity at the quadrature point
                wb = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    wb += wArray[(kk-1)*n2+jj]*gamma;
                }
                Q0[ii][ii] *= 0.5*wb/det; // scale by 0.5 outside for the 0.5 w^2
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
                cols[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }

        if(kk < geom->nk - 1) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                //Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);
                Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);

                // multiply by the vertical jacobian, then scale the piecewise constant
                // basis by the vertical jacobian, so do nothing

                // interpolate the vertical velocity at the quadrature point
                wt = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    wt += wArray[(kk+0)*n2+jj]*gamma;
                }
                Q0[ii][ii] *= 0.5*wt/det; // scale by 0.5 outside
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
                cols[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }
    }
    VecRestoreArray(velz, &wArray);
    MatAssemblyBegin(BA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(BA, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRT(int ex, int ey, Vec rt, Mat A, bool do_internal) {
    int ii, jj, kk, ei, mp1, mp12;
    double det, rk, gamma;
    int inds2k[99];
    int* inds0 = topo->elInds0_l(ex, ey);
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
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);

            // multuply by the vertical determinant to integrate, then
            // divide piecewise constant density by the vertical determinant,
            // so these cancel
            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            if(!do_internal) { // TODO: don't understand this scaling ?!?
                rk *= 1.0/geom->thick[kk][inds0[ii]];
            }
            Q0[ii][ii] *= 0.5*rk/det;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    VecRestoreArray(rt, &rArray);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithTheta(int ex, int ey, Vec theta, Mat A) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, tb, tt, gamma;
    int inds2k[99];
    PetscScalar *tArray;

    inds0 = topo->elInds0_l(ex, ey);
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
            QB[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            QB[ii][ii] *= 0.5*geom->thick[kk][inds0[ii]];
            QT[ii][ii]  = QB[ii][ii];

            tb = tt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                tb += tArray[(kk+0)*n2+jj]*gamma;
                tt += tArray[(kk+1)*n2+jj]*gamma;
            }
            QB[ii][ii] *= tb/det;
            QT[ii][ii] *= tt/det;
        }

        // assemble the first basis function
        if(kk > 0) {
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QB, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QT, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
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

    inds0 = topo->elInds0_l(ex, ey);
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
            Q0[ii][ii] = 0.5*Q->A[ii][ii]*SCALE;
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);

        // interpolate
        for(ii = 0; ii < mp12; ii++) {
            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rk += W->A[ii][jj]*rArray[kk*n2+jj];
            }
            // scale by matric term and vertical basis function at quadrature point ii
            det = geom->det[ei][ii];
            rk *= 1.0/(det*geom->thick[kk][inds0[ii]]);
            rtq[ii] = fac*pow(rk, exponent);
        }

        for(jj = 0; jj < n2; jj++) {
            rtj[jj] = 0.0;
            for(ii = 0; ii < mp12; ii++) {
                rtj[jj] += WtQ[jj][ii]*rtq[ii];
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
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(B);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
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
    inds0 = topo->elInds0_l(ex, ey);
    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(A);

    // top level
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
        // assembly the contributions from the top two levels only
        Q0[ii][ii] *= 0.5*(geom->thick[geom->nk-1][inds0[ii]] + geom->thick[geom->nk-2][inds0[ii]]);
    }
    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
    Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        // interface between top two levels
        inds2k[ii] = ii + (geom->nk-2)*W->nDofsJ;
    }
    MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

    // second from top level
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
        // assembly the contributions from the top two levels only
        Q0[ii][ii] *= 0.25*(geom->thick[geom->nk-2][inds0[ii]] + geom->thick[geom->nk-3][inds0[ii]]);
    }
    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
    Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        // interface between top two levels
        inds2k[ii] = ii + (geom->nk-3)*W->nDofsJ;
    }
    MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

    // third from top level
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
        // assembly the contributions from the top two levels only
        Q0[ii][ii] *= 0.125*(geom->thick[geom->nk-3][inds0[ii]] + geom->thick[geom->nk-4][inds0[ii]]);
    }
    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
    Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        // interface between top two levels
        inds2k[ii] = ii + (geom->nk-4)*W->nDofsJ;
    }
    MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
/*
*/

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinConWithTheta(int ex, int ey, Mat AB, Vec theta) {
    int ii, jj, kk, ei, mp1, mp12;
    double det, tk, gamma;
    int rows[99], cols[99];
    PetscScalar* tArray;

    ei   = ey*topo->nElsX + ex;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(AB);

    // assemble the matrices
    VecGetArray(theta, &tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Q0[ii][ii] = 0.5*Q->A[ii][ii]*(SCALE/det);

                tk = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    tk += tArray[(kk-1)*n2+jj]*gamma;
                }
                Q0[ii][ii] *= tk/det;
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                //Q0[ii][ii] = 0.5*Q->A[ii][ii]*(SCALE/det/det);
                Q0[ii][ii] = 0.5*Q->A[ii][ii]*(SCALE/det);

                tk = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    tk += tArray[(kk+0)*n2+jj]*gamma;
                }
                Q0[ii][ii] *= tk/det;
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }
    }
    VecRestoreArray(theta, &tArray);

    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinConWithRho(int ex, int ey, Mat AB, Vec rho) {
    int ii, jj, kk, ei, mp1, mp12, *inds0;
    double det, tk, gamma;
    int rows[99], cols[99];
    PetscScalar* tArray;

    ei   = ey*topo->nElsX + ex;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;
    inds0 = topo->elInds0_l(ex, ey);

    MatZeroEntries(AB);

    // assemble the matrices
    VecGetArray(rho, &tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = 0.5*Q->A[ii][ii]*(SCALE/det);

            tk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                tk += tArray[kk*n2+jj]*gamma;
            }
            Q0[ii][ii] *= tk/det;
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }
    }
    VecRestoreArray(rho, &tArray);

    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);
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
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 0.5;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
                cols[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }

        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
                cols[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(BA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(BA, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstEoS(int ex, int ey, Vec rt, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk, gamma, fac, dPidTheta;
    int inds2k[99];
    PetscScalar* rArray;

    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    fac = CP*pow(RD/P0, RD/CV);
    fac *= RD/CV;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(rt, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            rk *= 1.0/(geom->thick[kk][inds0[ii]]*det);

            dPidTheta = fac*pow(rk, RD/CV-1.0);
            Q0[ii][ii] *= dPidTheta;
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rt, &rArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithThetaInv(int ex, int ey, Vec theta, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, tb, tt, gamma;
    int inds2k[99];
    PetscScalar* tArray;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(theta, &tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            tb = tt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                tb += tArray[(kk+0)*n2+jj]*gamma;
                tt += tArray[(kk+1)*n2+jj]*gamma;
            }
            Q0[ii][ii] *= 0.5*(tb + tt)/det;
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(theta, &tArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithBousInv(int ex, int ey, Vec bous, bool add_I, Mat A) {
    int kk, ii, jj, rows[99], ei, *inds0, mp1, mp12;
    double det, bb, bt, gamma;
    PetscScalar* bArray;
    double** WtQW_2;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);
    if(add_I) {
        WtQW_2 = new double*[n2];
        for(ii = 0; ii < n2; ii++) {
            WtQW_2[ii] = new double[n2];
        }
    }
    VecGetArray(bous, &bArray);
    for(kk = 0; kk < geom->nk-1; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            //Q0[ii][ii] *= 0.5*(geom->thick[kk+0][inds0[ii]] + geom->thick[kk+1][inds0[ii]]);

            bb = bt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                bb += bArray[(kk+0)*n2+jj]*gamma;
                bt += bArray[(kk+1)*n2+jj]*gamma;
            }
            //bb /= geom->thick[kk+0][inds0[ii]];
            //bt /= geom->thick[kk+1][inds0[ii]];
            //Q0[ii][ii] *= (bb + bt)/det;
            Q0[ii][ii] *= 0.5*(bb + bt)/det;
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        if(add_I) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
                Q0[ii][ii] *= 0.5*(geom->thick[kk+0][inds0[ii]] + geom->thick[kk+1][inds0[ii]]);
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW_2);
            for(ii = 0; ii < n2; ii++) {
                for(jj = 0; jj < n2; jj++) {
                    WtQW[ii][jj] += WtQW_2[ii][jj];
                } 
            }
        }

        // take the inverse
        Inv(WtQW, WtQWinv, n2);
        // add to matrix
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, rows, W->nDofsJ, rows, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(bous, &bArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    if(add_I) {
        for(ii = 0; ii < n2; ii++) {
            delete[] WtQW_2[ii];
        }
        delete[] WtQW_2;
    }
}

void VertOps::AssembleConLin2(int ex, int ey, Mat BA) {
    int ii, kk, ei, mp1, mp12, rows[99], cols[99];
    double det;

    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(BA);

    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 0.5;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+0)*W->nDofsJ;
        }

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
    }
    MatAssemblyBegin(BA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(BA, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithEOS(int ex, int ey, Vec rt, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk, rtq, gamma;
    int inds2k[99];
    PetscScalar* rArray;
    double fac = CP*pow(RD/P0, RD/CV);

    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(rt, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            rk *= 1.0/(det*geom->thick[kk][inds0[ii]]);
            rtq = fac*pow(rk, RD/CV);
            Q0[ii][ii] *= rtq;
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rt, &rArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithTheta(int ex, int ey, Vec theta, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, tb, tt, gamma;
    int inds2k[99];
    PetscScalar* tArray;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(theta, &tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            tb = tt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                tb += tArray[(kk+0)*n2+jj]*gamma;
                tt += tArray[(kk+1)*n2+jj]*gamma;
            }
            Q0[ii][ii] *= 0.5*(tb + tt)/det;
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
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

    inds0 = topo->elInds0_l(ex, ey);
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
            Q0[ii][ii] = 0.5*Q->A[ii][ii]*SCALE;
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);

        // interpolate
        for(ii = 0; ii < mp12; ii++) {
            rk = ek = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rk += W->A[ii][jj]*rArray[kk*n2+jj];
                ek += W->A[ii][jj]*eArray[kk*n2+jj];
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
                rtj[jj] += WtQ[jj][ii]*rtq[ii];
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
    double **BinvB = new double*[W->nDofsJ];
    double **B_BinvB = new double*[W->nDofsJ];
    PetscScalar *tArray, *_tArray;

    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(B);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        BinvB[ii] = new double[W->nDofsJ];
        B_BinvB[ii] = new double[W->nDofsJ];
    }

    VecGetArray(rt, &tArray);
    if(theta) VecGetArray(theta, &_tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        // assemble the layer-wise inverse matrix
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            tk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                tk += tArray[kk*n2+jj]*gamma;
            }
            Q0[ii][ii] *= tk/(geom->thick[kk][inds0[ii]]*det);
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);

        // assemble the layer-wise mass matrix
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
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
                Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
                Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

                tk = tkp1 = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    tk   += _tArray[(kk+0)*n2+jj]*gamma;
                    tkp1 += _tArray[(kk+1)*n2+jj]*gamma;
                }
                Q0[ii][ii] *= 0.5*(tk + tkp1)/det;
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW, WtQWinv, BinvB);
            for(ii = 0; ii < n2; ii++) {
                BinvB[ii][ii] += 1.0;
            }
            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, BinvB, B_BinvB, WtQW);
            for(ii = 0; ii < n2; ii++) {
                for(jj = 0; jj < n2; jj++) {
                    B_BinvB[ii][jj] = WtQW[ii][jj];
                }
            }
        }

        Inv(B_BinvB, WtQWinv, n2);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rt, &tArray);
    if(theta) VecRestoreArray(theta, &_tArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  B, MAT_FINAL_ASSEMBLY);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        delete[] BinvB[ii];
        delete[] B_BinvB[ii];
    }
    delete[] BinvB;
    delete[] B_BinvB;
}

void VertOps::AssembleLinearWithThetaExp(int ex, int ey, Vec theta, double exponent, Mat A) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, tb, tt, gamma;
    int inds2k[99];
    PetscScalar *tArray;

    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(A);

    // assemble the matrices
    VecGetArray(theta, &tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            QB[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            QB[ii][ii] *= 0.5*geom->thick[kk][inds0[ii]];
            QT[ii][ii]  = QB[ii][ii];

            tb = tt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                tb += tArray[(kk+0)*n2+jj]*gamma;
                tt += tArray[(kk+1)*n2+jj]*gamma;
            }
            QB[ii][ii] *= pow(tb/det, exponent);
            QT[ii][ii] *= pow(tt/det, exponent);
        }
        // assemble the first basis function
        if(kk > 0) {
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QB, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
        // assemble the second basis function
        if(kk < geom->nk - 1) {
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QT, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    VecRestoreArray(theta, &tArray);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRhoExp(int ex, int ey, Vec rho, double exponent, Mat A) {
    int ii, jj, kk, ei, mp1, mp12;
    double det, rk, gamma;
    int* inds0;
    int inds2k[99];
    PetscScalar *rArray;

    inds0 = topo->elInds0_l(ex, ey);
    ei    = ey*topo->nElsX + ex;
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 0.5*geom->thick[kk][inds0[ii]];
            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            rk /= (det * geom->thick[kk][inds0[ii]]);
            Q0[ii][ii] *= pow(rk, exponent);
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
        MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRhoInv(int ex, int ey, Vec rho, Mat A) {
    int kk, ii, jj, rows[99], ei, mp1, mp12;
    double det, rb, rt, gamma;
    PetscScalar* rArray;

    ei    = ey*topo->nElsX + ex;
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk-1; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            rb = rt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rb += rArray[(kk+0)*n2+jj]*gamma;
                rt += rArray[(kk+1)*n2+jj]*gamma;
            }
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 0.5*(rb + rt)/det;
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, rows, W->nDofsJ, rows, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithThetaExp(int ex, int ey, Vec theta, double exponent, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, tb, tt, ftb, ftt, gamma;
    int inds2k[99];
    PetscScalar* tArray;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(theta, &tArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            tb = tt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                tb += tArray[(kk+0)*n2+jj]*gamma;
                tt += tArray[(kk+1)*n2+jj]*gamma;
            }
            tb /= det;
            tt /= det;
            ftb = pow(tb, exponent);
            ftt = pow(tt, exponent);
            //Q0[ii][ii] *= 0.5*(ftb + ftt)/det;
            Q0[ii][ii] *= 0.5*(ftb + ftt);
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(theta, &tArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithRhoExp(int ex, int ey, Vec rho, double exponent, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk, gamma;
    int inds2k[99];
    PetscScalar* rArray;

    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            rk /= (geom->thick[kk][inds0[ii]]*det);
            Q0[ii][ii] *= pow(rk, exponent);
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithW(int ex, int ey, Vec velz, Mat A) {
    int ii, jj, kk, ei, mp1, mp12, cols[99], *inds0;
    double wb, wt, gamma, det;
    PetscScalar* wArray;

    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);

    MatZeroEntries(A);

    VecGetArray(velz, &wArray);
    for(kk = 0; kk < geom->nk; kk++) {
        if(kk > 0) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
                Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

                // interpolate the vertical velocity at the quadrature point
                wb = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    wb += wArray[(kk-1)*n2+jj]*gamma;
                }
                Q0[ii][ii] *= wb/det; // scale by 0.5 outside for the 0.5 w^2
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                cols[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, cols, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }

        if(kk < geom->nk - 1) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
                Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

                // interpolate the vertical velocity at the quadrature point
                wt = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    wt += wArray[(kk+0)*n2+jj]*gamma;
                }
                Q0[ii][ii] *= wt/det; // scale by 0.5 outside
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
            for(ii = 0; ii < W->nDofsJ; ii++) {
                cols[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, cols, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }
    }
    VecRestoreArray(velz, &wArray);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

// assumes that zi are vertical distances relative to the edge at which 
// the flux is to be computed
double interpRhoAtEdge(double** A, double** Ainv, double* ri, double* zi) {
    int ii;
    double re = 0.0;

    for(ii = 0; ii < 3; ii++) {
        A[ii][0] = 1.0;
        A[ii][1] = zi[ii];
        A[ii][2] = zi[ii]*zi[ii];
    }
    Inv(A, Ainv, 3);
    for(ii = 0; ii < 3; ii++) {
        re += Ainv[0][ii]*ri[ii];
    }
    return re;
}

void VertOps::AssembleLinearWithRho_up(int ex, int ey, Vec rho, Vec vel, Mat A) {
    int ii, kk, ei, mp1, mp12;
    double det, zi[3], ri[3];
    int inds2k[99];
    int* inds0 = topo->elInds0_l(ex, ey);
    PetscScalar *rArray, *vArray;

    ei    = ey*topo->nElsX + ex;
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    VecGetArray(vel, &vArray);

    // interpolate rho at quadrature points
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rArray, &rho_q[kk*mp12+ii]);
            rho_q[kk*mp12+ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
    }

    // interpolate upwinded rho at the edges
    // first & last edge (piecewise constant upwinding)
    for(ii = 0; ii < mp12; ii++) {
        rho_e[ii]                   = (vArray[ii] > 0.0) ? rho_q[ii]                   : rho_q[mp12+ii];
        rho_e[(geom->nk-2)*mp12+ii] = (vArray[ii] > 0.0) ? rho_q[(geom->nk-2)*mp12+ii] : rho_q[(geom->nk-1)*mp12+ii];
    }
    // quadratic upwinding for the internal levels
    for(kk = 1; kk < geom->nk-2; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            if(vArray[kk*mp12+ii] > 0.0) {
                ri[0] = rho_q[(kk-1)*mp12+ii];
                ri[1] = rho_q[(kk+0)*mp12+ii];
                ri[2] = rho_q[(kk+1)*mp12+ii];
                zi[0] = -0.5*geom->thick[kk-1][inds0[ii]] - geom->thick[kk+0][inds0[ii]];
                zi[1] = -0.5*geom->thick[kk+0][inds0[ii]];
                zi[2] = +0.5*geom->thick[kk+1][inds0[ii]];
            } else {
                ri[0] = rho_q[(kk+0)*mp12+ii];
                ri[1] = rho_q[(kk+1)*mp12+ii];
                ri[2] = rho_q[(kk+2)*mp12+ii];
                zi[0] = -0.5*geom->thick[kk+0][inds0[ii]];
                zi[1] = +0.5*geom->thick[kk+1][inds0[ii]];
                zi[2] = +0.5*geom->thick[kk+2][inds0[ii]] + geom->thick[kk+1][inds0[ii]];
            }
            rho_e[kk*mp12+ii] = interpRhoAtEdge(A3, A3inv, ri, zi);
        }
    }
    VecRestoreArray(rho, &rArray);
    VecRestoreArray(vel, &vArray);

    // assemble the right hand side
    MatZeroEntries(A);
    for(kk = 0; kk < geom->nk-1; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            QB[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            QB[ii][ii] *= 0.5*geom->thick[kk][inds0[ii]];
            QT[ii][ii]  = QB[ii][ii];

            QB[ii][ii]  = (kk < geom->nk-1) ? rho_e[(kk+0)*mp12+ii]*QB[ii][ii] : 0.0;
            QT[ii][ii]  = (kk > 0         ) ? rho_e[(kk-1)*mp12+ii]*QT[ii][ii] : 0.0;
        }

        // assemble the first basis function
        if(kk > 0) {
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QB, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QT, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRayleighInv(int ex, int ey, double dt_fric, Mat A) {
    int kk, ii, rows[99], ei, *inds0, mp1, mp12;
    double det;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    for(kk = 0; kk < geom->nk-1; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 0.5*(geom->thick[kk+0][inds0[ii]] + geom->thick[kk+1][inds0[ii]]);
            if(kk == geom->nk-1)      Q0[ii][ii] *= (1.0 + 1.00*dt_fric);
            else if(kk == geom->nk-2) Q0[ii][ii] *= (1.0 + 0.50*dt_fric);
            else if(kk == geom->nk-3) Q0[ii][ii] *= (1.0 + 0.25*dt_fric);
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // take the inverse
        Inv(WtQW, WtQWinv, n2);
        // add to matrix
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, rows, W->nDofsJ, rows, WtQWflat, ADD_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithThetaInv(int ex, int ey, Vec theta, Mat A) {
    int kk, ii, jj, rows[99], ei, *inds0, mp1, mp12;
    double det, rb, rt, gamma;
    PetscScalar* tArray;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    VecGetArray(theta, &tArray);
    for(kk = 0; kk < geom->nk-1; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            rb = rt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rb += tArray[(kk+0)*n2+jj]*gamma;
                rt += tArray[(kk+1)*n2+jj]*gamma;
            }
            rb *= geom->thick[kk+0][inds0[ii]];
            rt *= geom->thick[kk+1][inds0[ii]];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 0.5*(rb + rt)/det;
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, rows, W->nDofsJ, rows, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(theta, &tArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void VertOps::Assemble_EOS_Residual_new(int ex, int ey, Vec rt, Vec exner, Vec eos_rhs) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk, ek;
    double exq[99], tmp1[99], tmp2[99], tmp3[99];
    PetscScalar *rArray, *eArray, *fArray;

    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    ei    = ey*topo->nElsX + ex;

    VecZeroEntries(eos_rhs);

    // assemble the eos rhs vector
    VecGetArray(rt, &rArray);
    VecGetArray(exner, &eArray);
    VecGetArray(eos_rhs, &fArray);
    for(kk = 0; kk < geom->nk; kk++) {
        // interpolate the exner pressure and the density weighted potential temperature
        // to the quadrature points
        for(ii = 0; ii < mp12; ii++) {
            rk = ek = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rk += W->A[ii][jj]*rArray[kk*n2+jj];
                ek += W->A[ii][jj]*eArray[kk*n2+jj];
            }
            // scale by matric term and vertical basis function at quadrature point ii
            det = geom->det[ei][ii];
            rk *= 1.0/(det*geom->thick[kk][inds0[ii]]);
            ek *= 1.0/(det*geom->thick[kk][inds0[ii]]);

            Q0[ii][ii]  = rk*Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
            exq[ii] = pow(ek/CP, CV/RD);
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsI, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsI, WtQ, W->A, WtQW);

        for(ii = 0; ii < mp12; ii++) {
            Q0[ii][ii]  = Q->A[ii][ii]*SCALE;
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);

        _matvec(WtQ, exq, tmp1, n2, mp12);
        _matvec(WtQWinv, tmp1, tmp2, n2, n2);
        _matvec(WtQW, tmp2, tmp3, n2, n2);

        for(ii = 0; ii < mp12; ii++) tmp1[ii] = 1.0;
        _matvec(WtQ, tmp1, tmp2, n2, mp12);
        for(ii = 0; ii < n2; ii++) {
            fArray[kk*n2+ii] = tmp2[ii] - (P0/RD)*tmp3[ii];
        }
    }
    VecRestoreArray(rt, &rArray);
    VecRestoreArray(exner, &eArray);
    VecRestoreArray(eos_rhs, &fArray);
}

void VertOps::AssembleN_PiInv(int ex, int ey, Vec rt, Vec pi, Mat A, bool do_inverse) {
    int ii, jj, kk, ei, mp1, mp12, rows[99];
    int* inds0 = topo->elInds0_l(ex, ey);
    double det, rq, eq, rtq[99];
    double fac = -1.0*(RD/P0)*(RD/CV)*pow(CP, CV/RD);
    PetscScalar *rtArray, *piArray;

    ei    = ey*topo->nElsX + ex;
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    VecGetArray(rt, &rtArray);
    VecGetArray(pi, &piArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            rq = eq = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rq += W->A[ii][jj]*rtArray[kk*n2+jj];
                eq += W->A[ii][jj]*piArray[kk*n2+jj];
            }
            // scale by matric term and vertical basis function at quadrature point ii
            det = geom->det[ei][ii];
            rq *= 1.0/(det*geom->thick[kk][inds0[ii]]);
            eq *= 1.0/(det*geom->thick[kk][inds0[ii]]);

            rtq[ii] = rq;

            Q0[ii][ii]  = pow(eq, (CV-RD)/RD) * Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW_2); // pi^{(c_v-R)/R}

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = rtq[ii] * Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);                                      // rt^{-1}

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);   // M_2

        Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW, WtQWinv, WtQW_3);
        Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW_3, WtQW_2, WtQW);
        if(do_inverse) {
            Inv(WtQW, WtQWinv, n2);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);
            for(ii = 0; ii < n2*n2; ii++) WtQWflat[ii] *= fac;
        } else {
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
            for(ii = 0; ii < n2*n2; ii++) WtQWflat[ii] /= fac;
        }
        for(ii = 0; ii < n2; ii++) rows[ii] = kk*n2 + ii;

        MatSetValues(A, W->nDofsJ, rows, W->nDofsJ, rows, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rt, &rtArray);
    VecRestoreArray(pi, &piArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleN_RT(int ex, int ey, Vec rt, Vec pi, Mat A) {
    int ii, jj, kk, ei, mp1, mp12, rows[99];
    int* inds0 = topo->elInds0_l(ex, ey);
    double det, rq, eq, rtq[99];
    double fac = (P0/RD);
    PetscScalar *rtArray, *piArray;

    ei    = ey*topo->nElsX + ex;
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    VecGetArray(rt, &rtArray);
    VecGetArray(pi, &piArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            rq = eq = 0.0;
            for(jj = 0; jj < n2; jj++) {
                rq += W->A[ii][jj]*rtArray[kk*n2+jj];
                eq += W->A[ii][jj]*piArray[kk*n2+jj];
            }
            // scale by matric term and vertical basis function at quadrature point ii
            det = geom->det[ei][ii];
            rq *= 1.0/(det*geom->thick[kk][inds0[ii]]);
            eq *= 1.0/(det*geom->thick[kk][inds0[ii]]);

            rtq[ii] = rq*rq;

            Q0[ii][ii]  = pow(eq/CP, CV/RD) * Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW_2); // (pi/cp)^{c_v/R}

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = rtq[ii] * Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);                                      // rt^{-2}

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];
        }
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);   // M_2

        Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW, WtQWinv, WtQW_3);
        Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW_3, WtQW_2, WtQW);

        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        for(ii = 0; ii < n2*n2; ii++) WtQWflat[ii] *= fac;
        for(ii = 0; ii < n2; ii++) rows[ii] = kk*n2 + ii;

        MatSetValues(A, W->nDofsJ, rows, W->nDofsJ, rows, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rt, &rtArray);
    VecRestoreArray(pi, &piArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleConstWithRhoInv2(int ex, int ey, Vec rho, Mat B) {
    int ii, jj, kk, ei, mp1, mp12;
    int *inds0;
    double det, rk, gamma;
    int inds2k[99];
    PetscScalar* rArray;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(B);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            Q0[ii][ii] *= 1.0/geom->thick[kk][inds0[ii]];

            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            rk /= (geom->thick[kk][inds0[ii]]*det);
            Q0[ii][ii] *= rk*rk;
        }

        // assemble the piecewise constant mass matrix for level k
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Inv(WtQW, WtQWinv, n2);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
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

    inds_0 = topo->elInds0_l(ex, ey);

    MatZeroEntries(A);

    // assemble the matrices
    VecGetArray(rho, &rArray);
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(uhl[kk], &uArray);

        // build the 2D mass matrix
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // multuply by the vertical determinant to integrate, then
            // divide piecewise constant density by the vertical determinant,
            // so these cancel
            rk = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
            }
            Q0[ii][ii] *= 0.5*rk/det;

            // upwinded test functions
            J = geom->J[ei][ii];
            geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, ug);
            // map velocity to local element coordinates
	    ul[0] = (+J[1][1]*ug[0] - J[0][1]*ug[1])/det;
            ul[1] = (-J[1][0]*ug[0] + J[0][0]*ug[1])/det;
            ul[0] /= geom->thick[kk][inds_0[ii]];
            ul[1] /= geom->thick[kk][inds_0[ii]];
            for(jj = 0; jj < _n; jj++) {
                _ex[jj] = edge->eval(quad->x[ii%mp1] + dt*ul[0], jj);
                _ey[jj] = edge->eval(quad->x[ii/mp1] + dt*ul[1], jj);
            }
            for(jj = 0; jj < _n2; jj++) {
                Wt[jj][ii] = _ex[jj%_n]*_ey[jj/_n];
            }
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

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

    inds_0 = topo->elInds0_l(ex, ey);

    MatZeroEntries(AB);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(uhl[kk], &uArray);
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det);
            // multiply by the vertical jacobian, then scale the piecewise constant 
            // basis by the vertical jacobian, so do nothing 
            Q0[ii][ii] *= 0.5;

            // upwinded test functions
            J = geom->J[ei][ii];
            geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, ug);
            // map velocity to local element coordinates
	    ul[0] = (+J[1][1]*ug[0] - J[0][1]*ug[1])/det;
            ul[1] = (-J[1][0]*ug[0] + J[0][0]*ug[1])/det;
            ul[0] /= geom->thick[kk][inds_0[ii]];
            ul[1] /= geom->thick[kk][inds_0[ii]];
            for(jj = 0; jj < _n; jj++) {
                _ex[jj] = edge->eval(quad->x[ii%mp1] + dt*ul[0], jj);
                _ey[jj] = edge->eval(quad->x[ii/mp1] + dt*ul[1], jj);
            }
            for(jj = 0; jj < _n2; jj++) {
                Wt[jj][ii] = _ex[jj%_n]*_ey[jj/_n];
            }
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }
        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);

        VecGetArray(uhl[kk], &uArray);
    }
    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
}

#define phi_0(z) ( 0.5*(1.0 - (z)) )
#define phi_1(z) ( 0.5*(1.0 + (z)) )

void VertOps::AssembleLinCon_up(int ex, int ey, Mat AB, Vec velz, double dt) {
    int ii, jj, kk, ei, mp1, mp12;
    double det;
    int rows[99], cols[99];
    double gamma, wt, wb, N0zb, N0zt, N1zb, N1zt;
    PetscScalar* wArray;

    ei    = ey*topo->nElsX + ex;
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    MatZeroEntries(AB);

    // assemble the matrices
    VecGetArray(velz, &wArray);
    for(kk = 0; kk < geom->nk; kk++) {
        // interpolate the top and bottom velocities
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            wb = wt = 0.0;
            if(kk > 0) {
                for(jj = 0; jj < W->nDofsJ; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    wb += wArray[(kk-1)*n2+jj]*gamma;
                }
            }
            N0zb = phi_0(-1.0 + dt*wb);
            N1zb = phi_1(-1.0 + dt*wb);
            if(kk < geom->nk-1) {
                for(jj = 0; jj < W->nDofsJ; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    wt += wArray[(kk  )*n2+jj]*gamma;
                }
            }
            N0zt = phi_0(+1.0 + dt*wt);
            N1zt = phi_1(+1.0 + dt*wt);

            QB[ii][ii] = 0.5 * (N0zb + N0zt) * Q->A[ii][ii]*(SCALE/det);
            QT[ii][ii] = 0.5 * (N1zb + N1zt) * Q->A[ii][ii]*(SCALE/det);
        }

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }

        // assemble the first basis function
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QB, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+0)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);

        // assemble the second basis function
        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QT, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + (kk+1)*W->nDofsJ;
        }
        MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(velz, &wArray);

    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  AB, MAT_FINAL_ASSEMBLY);
}

void VertOps::AssembleLinearWithRho_up(int ex, int ey, Vec rho, Mat A, Vec velz, double dt) {
    int ii, jj, kk, ei, mp1, mp12;
    double det, rk, gamma;
    int inds_a[99], inds_b[99];
    double wb, wt, N0zb, N0zt, N1zb, N1zt;
    PetscScalar *rArray, *wArray;

    ei    = ey*topo->nElsX + ex;
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(A);

    VecGetArray(rho, &rArray);
    VecGetArray(velz, &wArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            rk = wb = wt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rk += rArray[kk*n2+jj]*gamma;
                if(kk > 0         ) wb += wArray[(kk-1)*n2+jj]*gamma;
                if(kk < geom->nk-1) wt += wArray[(kk  )*n2+jj]*gamma;
            }
            rk *= (0.5/det);

            N0zb = phi_0(-1.0 + dt*wb);
            N1zb = phi_1(-1.0 + dt*wb);
            N0zt = phi_0(+1.0 + dt*wt);
            N1zt = phi_1(+1.0 + dt*wt);

            Qaa[ii][ii] = rk * N0zb * Q->A[ii][ii] * (SCALE/det);
            Qab[ii][ii] = rk * N0zt * Q->A[ii][ii] * (SCALE/det);
            Qba[ii][ii] = rk * N1zb * Q->A[ii][ii] * (SCALE/det);
            Qbb[ii][ii] = rk * N1zt * Q->A[ii][ii] * (SCALE/det);
        }

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds_a[ii] = ii + (kk+0)*W->nDofsJ;
            inds_b[ii] = ii + (kk+1)*W->nDofsJ;
        }

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        MatSetValues(A, W->nDofsJ, inds_a, W->nDofsJ, inds_a, WtQWflat, ADD_VALUES);

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qab, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        MatSetValues(A, W->nDofsJ, inds_a, W->nDofsJ, inds_b, WtQWflat, ADD_VALUES);

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qba, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        MatSetValues(A, W->nDofsJ, inds_b, W->nDofsJ, inds_a, WtQWflat, ADD_VALUES);

        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qbb, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        MatSetValues(A, W->nDofsJ, inds_b, W->nDofsJ, inds_b, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &rArray);
    VecRestoreArray(velz, &wArray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);
}
