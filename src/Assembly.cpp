#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"

using namespace std;

// mass matrix for the 1 form vector (x-normal degrees of
// freedom first then y-normal degrees of freedom)
Umat::Umat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    mat_alloc = true;
    assemble();
    mat_alloc = false;
}

void Umat::assemble() {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y;
    Wii* Q = new Wii(l->q, geom);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double det, **J;
    double** Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    double** Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    double** UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** VtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** VtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Qab = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Qbb = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double* UtQUflat = new double[U->nDofsJ*U->nDofsJ];

    if(mat_alloc) {
        MatCreate(MPI_COMM_WORLD, &M);
        MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
        MatSetType(M, MATMPIAIJ);
        MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
    }
    MatZeroEntries(M);

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii][ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]/det;
                Qab[ii][ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]/det;
                Qbb[ii][ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]/det;
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
            Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

            Mult_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qaa, UtQaa);
            Mult_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qab, UtQab);
            Mult_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qab, VtQba);
            Mult_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qbb, VtQbb);

            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQaa, U->A, UtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQab, V->A, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQba, U->A, VtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQbb, V->A, VtQV);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQaa);
    Free2D(U->nDofsJ, UtQab);
    Free2D(U->nDofsJ, VtQba);
    Free2D(U->nDofsJ, VtQbb);
    Free2D(U->nDofsJ, UtQU);
    Free2D(U->nDofsJ, UtQV);
    Free2D(U->nDofsJ, VtQU);
    Free2D(U->nDofsJ, VtQV);
    Free2D(Q->nDofsI, Qaa);
    Free2D(Q->nDofsI, Qab);
    Free2D(Q->nDofsI, Qbb);
    delete[] UtQUflat;
    delete Q;
    delete U;
    delete V;
}

// upwinded test function matrix
void Umat::assemble_up(double dt, Vec u1, Vec h2) {
    int ex, ey, ei, mp1, mp12, n0, np1, ii, jj;
    int *inds_x, *inds_y;
    double det, **J, ug[2], ul[2], lx[99], ly[99], _ex[99], _ey[99], _h = 1.0;
    PetscScalar *u1Array, *h2Array;
    GaussLobatto* quad = l->q;
    Wii* Q = new Wii(l->q, geom);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double** Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    double** Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    double** UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** VtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** VtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Qab = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Qbb = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double* UtQUflat = new double[U->nDofsJ*U->nDofsJ];

    n0 = l->n;
    np1 = n0 + 1;
    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    VecGetArray(u1, &u1Array);
    if(h2) VecGetArray(h2, &h2Array);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ug);

                // map velocity to local element coordinates
                ul[0] = (+J[1][1]*ug[0] - J[0][1]*ug[1])/det;
                ul[1] = (-J[1][0]*ug[0] + J[0][0]*ug[1])/det;
                // evaluate the nodal bases at the upwinded locations
                for(jj = 0; jj < np1; jj++) {
                    lx[jj] = l->eval_q(quad->x[ii%mp1] + dt*ul[0], jj);
                    ly[jj] = l->eval_q(quad->x[ii/mp1] + dt*ul[1], jj);
                }
                // evaluate the edge bases at the upwinded locations
                for(jj = 0; jj < n0; jj++) {
                    _ex[jj] = e->eval(quad->x[ii%mp1]/* + dt*ul[0]*/, jj);
                    _ey[jj] = e->eval(quad->x[ii/mp1]/* + dt*ul[1]*/, jj);
                }
                // evaluate the 2 form basis at the upwinded locations
                for(jj = 0; jj < n0*np1; jj++) {
                    Ut[jj][ii] = lx[jj%np1]*_ey[jj/np1];
                    Vt[jj][ii] = _ex[jj%n0]*ly[jj/n0];
                }

                if(h2) geom->interp2_g(ex, ey, ii%mp1, ii/mp1, h2Array, &_h);

                Qaa[ii][ii] = _h*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]/det;
                Qab[ii][ii] = _h*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]/det;
                Qbb[ii][ii] = _h*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]/det;
            }

            // reuse the JU and JV matrices for the nonlinear trial function expansion matrices
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Ut, Qaa, UtQaa);
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Ut, Qab, UtQab);
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Vt, Qab, VtQba);
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Vt, Qbb, VtQbb);

            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQaa, U->A, UtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQab, V->A, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQba, U->A, VtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQbb, V->A, VtQV);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);
    if(h2) VecRestoreArray(h2, &h2Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQaa);
    Free2D(U->nDofsJ, UtQab);
    Free2D(U->nDofsJ, VtQba);
    Free2D(U->nDofsJ, VtQbb);
    Free2D(U->nDofsJ, UtQU);
    Free2D(U->nDofsJ, UtQV);
    Free2D(U->nDofsJ, VtQU);
    Free2D(U->nDofsJ, VtQV);
    Free2D(Q->nDofsI, Qaa);
    Free2D(Q->nDofsI, Qab);
    Free2D(Q->nDofsI, Qbb);
    delete[] UtQUflat;
    delete Q;
    delete U;
    delete V;
}

Umat::~Umat() {
    MatDestroy(&M);
}

// 2 form mass matrix
Wmat::Wmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    assemble();
}

void Wmat::assemble() {
    int ex, ey, ei, mp1, mp12, ii, *inds;
    double det;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii][ii] = Q->A[ii][ii]/det;
            }

            Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
            Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQWflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Qaa);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete W;
    delete Q;
    delete[] WtQWflat;
}

Wmat::~Wmat() {
    MatDestroy(&M);
}

// 0 form mass matrix
Pmat::Pmat(Topo* _topo, Geom* _geom, LagrangeNode* _node) {
    topo = _topo;
    geom = _geom;
    node = _node;

    assemble();
}

void Pmat::assemble() {
    int ex, ey, ei, mp1, mp12, ii, *inds;
    double det;
    Wii* Q = new Wii(node->q, geom);
    M0_j_xy_i* P = new M0_j_xy_i(node);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Pt = Alloc2D(P->nDofsJ, P->nDofsI);
    double** PtQ = Alloc2D(P->nDofsJ, Q->nDofsJ);
    double** PtQP = Alloc2D(P->nDofsJ, P->nDofsJ);
    double* PtQPflat = new double[P->nDofsJ*P->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 4*P->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds0_g(ex, ey);
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii][ii] = Q->A[ii][ii]*det;
            }

            Tran_IP(P->nDofsI, P->nDofsJ, P->A, Pt);
            Mult_IP(P->nDofsJ, Q->nDofsJ, P->nDofsI, Pt, Qaa, PtQ);
            Mult_IP(P->nDofsJ, P->nDofsJ, Q->nDofsJ, PtQ, P->A, PtQP);

            Flat2D_IP(P->nDofsJ, P->nDofsJ, PtQP, PtQPflat);

            MatSetValues(M, P->nDofsJ, inds, P->nDofsJ, inds, PtQPflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Qaa);
    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    Free2D(P->nDofsJ, PtQP);
    delete P;
    delete Q;
    delete[] PtQPflat;
}

Pmat::~Pmat() {
    MatDestroy(&M);
}

// 0 form mass matrix with 2 form interpolated
Phmat::Phmat(Topo* _topo, Geom* _geom, LagrangeNode* _node) {
    M0_j_xy_i* P;

    topo = _topo;
    geom = _geom;
    node = _node;

    P = new M0_j_xy_i(node);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 4*P->nDofsJ, PETSC_NULL);

    delete P;
}

void Phmat::assemble(Vec h2) {
    int ex, ey, mp1, mp12, ii, *inds;
    double hi;
    Wii* Q = new Wii(node->q, geom);
    M0_j_xy_i* P = new M0_j_xy_i(node);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Pt = Alloc2D(P->nDofsJ, P->nDofsI);
    double** PtQ = Alloc2D(P->nDofsJ, Q->nDofsJ);
    double** PtQP = Alloc2D(P->nDofsJ, P->nDofsJ);
    double* PtQPflat = new double[P->nDofsJ*P->nDofsJ];
    PetscScalar* hArray;

    MatZeroEntries(M);

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(h2, &hArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds0_g(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                // determinant from the integration and the inverse determinant 
                // from the interpolation of the 2 form cancel
                geom->interp2_l(ex, ey, ii%mp1, ii/mp1, hArray, &hi);
                Qaa[ii][ii] = hi*Q->A[ii][ii];
            }

            Tran_IP(P->nDofsI, P->nDofsJ, P->A, Pt);
            Mult_IP(P->nDofsJ, Q->nDofsJ, P->nDofsI, Pt, Qaa, PtQ);
            Mult_IP(P->nDofsJ, P->nDofsJ, Q->nDofsJ, PtQ, P->A, PtQP);

            Flat2D_IP(P->nDofsJ, P->nDofsJ, PtQP, PtQPflat);

            MatSetValues(M, P->nDofsJ, inds, P->nDofsJ, inds, PtQPflat, ADD_VALUES);
        }
    }
    VecRestoreArray(h2, &hArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Qaa);
    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    Free2D(P->nDofsJ, PtQP);
    delete P;
    delete Q;
    delete[] PtQPflat;
}

void Phmat::assemble_block(Vec h2, int row_offset, int col_offset, Mat Q2) {
    int ex, ey, np1, np12, mp1, mp12, ii, jj, *inds, inds_r[99], inds_c[99];
    double hi;
    Wii* Q = new Wii(node->q, geom);
    M0_j_xy_i* P = new M0_j_xy_i(node);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Pt = Alloc2D(P->nDofsJ, P->nDofsI);
    double** PtQ = Alloc2D(P->nDofsJ, Q->nDofsJ);
    double** PtQP = Alloc2D(P->nDofsJ, P->nDofsJ);
    double* PtQPflat = new double[P->nDofsJ*P->nDofsJ];
    PetscScalar* hArray;

    np1 = node->n + 1;
    np12 = np1*np1;
    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(h2, &hArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds0_g(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                // determinant from the integration and the inverse determinant 
                // from the interpolation of the 2 form cancel
                geom->interp2_l(ex, ey, ii%mp1, ii/mp1, hArray, &hi);
                Qaa[ii][ii] = hi*Q->A[ii][ii];
            }
	    for(jj = 0; jj < np12; jj++) {
                inds_r[jj] = 2*inds[jj]+row_offset;
                inds_c[jj] = 2*inds[jj]+col_offset;
            }

            Tran_IP(P->nDofsI, P->nDofsJ, P->A, Pt);
            Mult_IP(P->nDofsJ, Q->nDofsJ, P->nDofsI, Pt, Qaa, PtQ);
            Mult_IP(P->nDofsJ, P->nDofsJ, Q->nDofsJ, PtQ, P->A, PtQP);

            Flat2D_IP(P->nDofsJ, P->nDofsJ, PtQP, PtQPflat);

            MatSetValues(Q2, P->nDofsJ, inds_r, P->nDofsJ, inds_c, PtQPflat, ADD_VALUES);
        }
    }
    VecRestoreArray(h2, &hArray);

    Free2D(Q->nDofsI, Qaa);
    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    Free2D(P->nDofsJ, PtQP);
    delete P;
    delete Q;
    delete[] PtQPflat;
}

void Phmat::assemble_up(Vec ul, Vec hl, double fac, double dt) {
    int ex, ey, ei, mp1, mp12, np1, np12, ii, *inds;
    double ux[2], lx[99], ly[99], det, **J, ux2[2], hx, tau, tmp, tmp2;
    PetscScalar *uArray, *hArray;
    GaussLobatto* quad = node->q;
    Wii* Q = new Wii(node->q, geom);
    M0_j_xy_i* P = new M0_j_xy_i(node);
    double** QP = Alloc2D(Q->nDofsI, P->nDofsJ);
    double** Pt = Alloc2D(P->nDofsJ, P->nDofsI);
    double** PtQP = Alloc2D(P->nDofsJ, P->nDofsJ);
    double* PtQPflat = new double[P->nDofsJ*P->nDofsJ];

    np1 = node->n + 1;
    np12 = np1*np1;
    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);

    Tran_IP(P->nDofsI, P->nDofsJ, P->A, Pt);

    VecGetArray(hl, &hArray);
    VecGetArray(ul, &uArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds = topo->elInds0_g(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, ux);
                ux2[0] = +J[1][1]*ux[0]/det - J[0][1]*ux[1]/det;
                ux2[1] = -J[1][0]*ux[0]/det + J[0][0]*ux[1]/det;
                geom->interp2_l(ex, ey, ii%mp1, ii/mp1, hArray, &hx);

                //tmp = sqrt(ux[0]*ux[0] + ux[1]*ux[1])/(sqrt(det)*fac);
                //tmp2 = dt*(ux[0]*ux[0] + ux[1]*ux[1])/det/fac;
                tau = 1.0/(1.0/(fac*dt)/* + tmp + tmp2*/);

                for(int jj = 0; jj < np1; jj++) {
                    lx[jj] = node->eval_q(quad->x[ii%mp1] - tau*ux2[0], jj);
                    ly[jj] = node->eval_q(quad->x[ii/mp1] - tau*ux2[1], jj);
                }
                // determinant from the integration and the inverse determinant 
                // from the density cancel
                for(int jj = 0; jj < np12; jj++) {
                    QP[ii][jj]  = hx * Q->A[ii][ii] * lx[jj%np1] * ly[jj/np1];
                }
            }

            Mult_IP(P->nDofsJ, P->nDofsJ, Q->nDofsI, Pt, QP, PtQP);
            Flat2D_IP(P->nDofsJ, P->nDofsJ, PtQP, PtQPflat);
            MatSetValues(M, P->nDofsJ, inds, P->nDofsJ, inds, PtQPflat, ADD_VALUES);
        }
    }
    VecRestoreArray(hl, &hArray);
    VecRestoreArray(ul, &uArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    Free2D(P->nDofsJ, Pt);
    Free2D(Q->nDofsI, QP);
    Free2D(P->nDofsJ, PtQP);
    delete P;
    delete Q;
    delete[] PtQPflat;
}

void Phmat::assemble_up_block(Vec ul, Vec hl, double fac, double dt, int row_offset, int col_offset, Mat Q2) {
    int ex, ey, ei, mp1, mp12, np1, np12, ii, jj, *inds, inds_r[99], inds_c[99];
    double ux[2], lx[99], ly[99], det, **J, ux2[2], hx, tau, tmp, tmp2;
    PetscScalar *uArray, *hArray;
    GaussLobatto* quad = node->q;
    Wii* Q = new Wii(node->q, geom);
    M0_j_xy_i* P = new M0_j_xy_i(node);
    double** QP = Alloc2D(Q->nDofsI, P->nDofsJ);
    double** Pt = Alloc2D(P->nDofsJ, P->nDofsI);
    double** PtQP = Alloc2D(P->nDofsJ, P->nDofsJ);
    double* PtQPflat = new double[P->nDofsJ*P->nDofsJ];

    np1 = node->n + 1;
    np12 = np1*np1;
    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(P->nDofsI, P->nDofsJ, P->A, Pt);

    VecGetArray(hl, &hArray);
    VecGetArray(ul, &uArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds = topo->elInds0_g(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, ux);
                ux2[0] = +J[1][1]*ux[0]/det - J[0][1]*ux[1]/det;
                ux2[1] = -J[1][0]*ux[0]/det + J[0][0]*ux[1]/det;
                geom->interp2_l(ex, ey, ii%mp1, ii/mp1, hArray, &hx);

                //tmp = sqrt(ux[0]*ux[0] + ux[1]*ux[1])/(sqrt(det)*fac);
                //tmp2 = dt*(ux[0]*ux[0] + ux[1]*ux[1])/det/fac;
                tau = 1.0/(1.0/(fac*dt)/* + tmp + tmp2*/);

                for(jj = 0; jj < np1; jj++) {
                    lx[jj] = node->eval_q(quad->x[ii%mp1] - tau*ux2[0], jj);
                    ly[jj] = node->eval_q(quad->x[ii/mp1] - tau*ux2[1], jj);
                }
                // determinant from the integration and the inverse determinant 
                // from the density cancel
                for(jj = 0; jj < np12; jj++) {
                    QP[ii][jj]  = hx * Q->A[ii][ii] * lx[jj%np1] * ly[jj/np1];
                }
            }
	    for(jj = 0; jj < np12; jj++) {
                inds_r[jj] = 2*inds[jj]+row_offset;
                inds_c[jj] = 2*inds[jj]+col_offset;
            }

            Mult_IP(P->nDofsJ, P->nDofsJ, Q->nDofsI, Pt, QP, PtQP);
            Flat2D_IP(P->nDofsJ, P->nDofsJ, PtQP, PtQPflat);
            MatSetValues(Q2, P->nDofsJ, inds_r, P->nDofsJ, inds_c, PtQPflat, ADD_VALUES);
        }
    }
    VecRestoreArray(hl, &hArray);
    VecRestoreArray(ul, &uArray);

    Free2D(P->nDofsJ, Pt);
    Free2D(Q->nDofsI, QP);
    Free2D(P->nDofsJ, PtQP);
    delete P;
    delete Q;
    delete[] PtQPflat;
}

Phmat::~Phmat() {
    MatDestroy(&M);
}

// 1 form mass matrix with 2 forms interpolated to quadrature points
Uhmat::Uhmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    Q = new Wii(l->q, geom);
    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);

    UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQU = Alloc2D(V->nDofsJ, V->nDofsJ);
    VtQV = Alloc2D(V->nDofsJ, V->nDofsJ);
    Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Qab = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Qbb = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);

    UtQUflat = new double[U->nDofsJ*U->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
}

void Uhmat::assemble(Vec h2) {
    int ex, ey, ei, mp1, mp12, ii;
    int *inds_x, *inds_y;
    double hi, det, **J;
    PetscScalar* h2Array;

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    VecGetArray(h2, &h2Array);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, h2Array, &hi);

                Qaa[ii][ii] = hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]/det;
                Qab[ii][ii] = hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]/det;
                Qbb[ii][ii] = hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]/det;
            }

            Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
            Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

            // reuse the JU and JV matrices for the nonlinear trial function expansion matrices
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Ut, Qaa, UtQaa);
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Ut, Qab, UtQab);
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Vt, Qab, VtQba);
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Vt, Qbb, VtQbb);

            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQaa, U->A, UtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQab, V->A, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQba, U->A, VtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQbb, V->A, VtQV);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);
        }
    }
    VecRestoreArray(h2, &h2Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

void Uhmat::assemble_up(double dt, Vec u1, Vec h2) {
    int ex, ey, ei, mp1, mp12, n0, np1, ii, jj;
    int *inds_x, *inds_y;
    double det, **J, ug[2], ul[2], lx[99], ly[99], _ex[99], _ey[99], _h;
    PetscScalar *u1Array, *h2Array;
    GaussLobatto* quad = l->q;
    double** Uup = Alloc2D(U->nDofsI, U->nDofsJ);
    double** Vup = Alloc2D(U->nDofsI, U->nDofsJ);

    n0 = l->n;
    np1 = n0 + 1;
    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    VecGetArray(u1, &u1Array);
    VecGetArray(h2, &h2Array);

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ug);

                // map velocity to local element coordinates
                ul[0] = (+J[1][1]*ug[0] - J[0][1]*ug[1])/det;
                ul[1] = (-J[1][0]*ug[0] + J[0][0]*ug[1])/det;
                // evaluate the nodal bases at the upwinded locations
                for(jj = 0; jj < np1; jj++) {
                    lx[jj] = l->eval_q(quad->x[ii%mp1] + dt*ul[0], jj);
                    ly[jj] = l->eval_q(quad->x[ii/mp1] + dt*ul[1], jj);
                }
                // evaluate the edge bases at the upwinded locations
                for(jj = 0; jj < n0; jj++) {
                    _ex[jj] = e->eval(quad->x[ii%mp1], jj);
                    _ey[jj] = e->eval(quad->x[ii/mp1], jj);
                }
                // evaluate the 2 form basis at the upwinded locations
                for(jj = 0; jj < n0*np1; jj++) {
                    Uup[ii][jj] = lx[jj%np1]*_ey[jj/np1];
                    Vup[ii][jj] = _ex[jj%n0]*ly[jj/n0];
                }

                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, h2Array, &_h);

                Qaa[ii][ii] = _h*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]/det;
                Qab[ii][ii] = _h*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]/det;
                Qbb[ii][ii] = _h*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]/det;
            }

            // reuse the JU and JV matrices for the nonlinear trial function expansion matrices
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Ut, Qaa, UtQaa);
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Ut, Qab, UtQab);
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Vt, Qab, VtQba);
            Mult_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Vt, Qbb, VtQbb);

            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQaa, Uup, UtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQab, Vup, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQba, Uup, VtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQbb, Vup, VtQV);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);
    VecRestoreArray(h2, &h2Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsI, Uup);
    Free2D(U->nDofsI, Vup);
}

Uhmat::~Uhmat() {
    delete[] UtQUflat;

    Free2D(U->nDofsJ, UtQU);
    Free2D(U->nDofsJ, UtQV);
    Free2D(V->nDofsJ, VtQU);
    Free2D(V->nDofsJ, VtQV);
    Free2D(Q->nDofsI, Qaa);
    Free2D(Q->nDofsI, Qab);
    Free2D(Q->nDofsI, Qbb);
    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQaa);
    Free2D(U->nDofsJ, UtQab);
    Free2D(U->nDofsJ, VtQba);
    Free2D(U->nDofsJ, VtQbb);

    delete U;
    delete V;
    delete Q;

    MatDestroy(&M);
}

// Assembly of the diagonal 0 form mass matrix as a vector.
// Assumes inexact integration and a diagonal mass matrix for the 
// 0 form function space (ie: quadrature and basis functions are 
// the same order)
Pvec::Pvec(Topo* _topo, Geom* _geom, LagrangeNode* _l) {
    topo = _topo;
    geom = _geom;
    l = _l;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vg);
    VecZeroEntries(vg);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &vlInv);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vgInv);

    entries = new PetscScalar[(l->n+1)*(l->n+1)];

    Q = new Wii(l->q, geom);

    assemble();
}

void Pvec::assemble() {
    int ii, ex, ey, np1, np12;
    int *inds_x;
    PetscScalar *p1Array, *p2Array;

    VecZeroEntries(vl);

    np1 = l->n + 1;
    np12 = np1*np1;

    // assemble values into local vector
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            for(ii = 0; ii < np12; ii++) {
                entries[ii] = Q->A[ii][ii]*geom->det[ey*topo->nElsX + ex][ii];
            }
            inds_x = topo->elInds0_l(ex, ey);
            VecSetValues(vl, np12, inds_x, entries, ADD_VALUES);
        }
    }

    // scatter values to global vector
    VecScatterBegin(topo->gtol_0, vl, vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_0, vl, vg, ADD_VALUES, SCATTER_REVERSE);

    // and back to local vector
    VecScatterBegin(topo->gtol_0, vg, vl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, vg, vl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(vl, &p1Array);
    VecGetArray(vlInv, &p2Array);
    for(ii = 0; ii < topo->n0; ii++) {
        p2Array[ii] = 1.0/p1Array[ii];
    }
    VecRestoreArray(vl, &p1Array);
    VecRestoreArray(vlInv, &p2Array);
    VecScatterBegin(topo->gtol_0, vlInv, vgInv, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_0, vlInv, vgInv, ADD_VALUES, SCATTER_REVERSE);
}

Pvec::~Pvec() {
    delete[] entries;
    VecDestroy(&vl);
    VecDestroy(&vg);
    VecDestroy(&vlInv);
    VecDestroy(&vgInv);
    delete Q;
}

// Assumes quadrature points and 0 forms are the same (for now)
WtQmat::WtQmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    assemble();
}

void WtQmat::assemble() {
    int ex, ey, mp1, mp12, ii, *inds_2, *inds_0;
    M2_j_xy_i* W = new M2_j_xy_i(e);
    Wii* Q = new Wii(e->l->q, geom);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQflat = new double[W->nDofsJ*Q->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, geom->n0l, topo->nDofs2G, geom->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*W->nDofsJ, PETSC_NULL, 8*W->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            for(ii = 0; ii < mp12; ii++) {
                Qaa[ii][ii] = Q->A[ii][ii];
            }

            Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

            Mult_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qaa, WtQ);
            Flat2D_IP(W->nDofsJ, Q->nDofsJ, WtQ, WtQflat);

            inds_2 = topo->elInds2_g(ex, ey);
            inds_0 = geom->elInds0_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, Q->nDofsJ, inds_0, WtQflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Qaa);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    delete[] WtQflat;
    delete W;
    delete Q;
}

WtQmat::~WtQmat() {
    MatDestroy(&M);
}

// Assumes quadrature points and 0 forms are the same (for now)
PtQmat::PtQmat(Topo* _topo, Geom* _geom, LagrangeNode* _l) {
    topo = _topo;
    geom = _geom;
    l = _l;

    assemble();
}

void PtQmat::assemble() {
    int ex, ey, *inds_0, *inds_q, mp1, mp12;
    M0_j_xy_i* P = new M0_j_xy_i(l);
    Wii* Q = new Wii(l->q, geom);
    double** Pt = Tran(P->nDofsI, P->nDofsJ, P->A);
    double** PtQ = Alloc2D(P->nDofsJ, Q->nDofsJ);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double* PtQflat = new double[P->nDofsJ*Q->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, geom->n0l, topo->nDofs0G, geom->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*Q->nDofsJ, PETSC_NULL, 4*Q->nDofsJ, PETSC_NULL);

    MatZeroEntries(M);

    mp1 = l->q->n+1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            for(int ii = 0; ii < mp12; ii++) {
                Qaa[ii][ii] = Q->A[ii][ii]*geom->det[ey*topo->nElsX + ex][ii];
            }
            Mult_IP(P->nDofsJ, Q->nDofsJ, Q->nDofsI, Pt, Qaa, PtQ);
            Flat2D_IP(P->nDofsJ, Q->nDofsJ, PtQ, PtQflat);

            inds_0 = topo->elInds0_g(ex, ey);
            inds_q = geom->elInds0_g(ex, ey);

            MatSetValues(M, P->nDofsJ, inds_0, Q->nDofsJ, inds_q, PtQflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    Free2D(Q->nDofsI, Qaa);
    delete[] PtQflat;
    delete P;
    delete Q;
}

PtQmat::~PtQmat() {
    MatDestroy(&M);
}

// columns are 2x the number of quadrature points to account for vector quantities
UtQmat::UtQmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    assemble();
}

void UtQmat::assemble() {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0, *inds_0x, *inds_0y;
    double **J;
    Wii* Q = new Wii(l->q, geom);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double** Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    double** Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    double** UtQ = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Qab = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Qba = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Qbb = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double* UtQflat = new double[U->nDofsJ*Q->nDofsJ];

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;
    inds_0x = new int[mp12];
    inds_0y = new int[mp12];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, 2*geom->n0l, topo->nDofs1G, 2*geom->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 24*U->nDofsJ, PETSC_NULL, 24*U->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                J = geom->J[ei][ii];

                Qaa[ii][ii] = J[0][0]*Q->A[ii][ii];
                Qab[ii][ii] = J[1][0]*Q->A[ii][ii];
                Qba[ii][ii] = J[0][1]*Q->A[ii][ii];
                Qbb[ii][ii] = J[1][1]*Q->A[ii][ii];
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_0 = geom->elInds0_g(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                inds_0x[ii] = 2*inds_0[ii]+0;
                inds_0y[ii] = 2*inds_0[ii]+1;
            }

            Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
            Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

            //
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Qaa, UtQ);
            Flat2D_IP(U->nDofsJ, Q->nDofsJ, UtQ, UtQflat);
            MatSetValues(M, U->nDofsJ, inds_x, Q->nDofsJ, inds_0x, UtQflat, ADD_VALUES);

            //
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Qab, UtQ);
            Flat2D_IP(U->nDofsJ, Q->nDofsJ, UtQ, UtQflat);
            MatSetValues(M, U->nDofsJ, inds_x, Q->nDofsJ, inds_0y, UtQflat, ADD_VALUES);

            //
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Vt, Qba, UtQ);
            Flat2D_IP(U->nDofsJ, Q->nDofsJ, UtQ, UtQflat);
            MatSetValues(M, U->nDofsJ, inds_y, Q->nDofsJ, inds_0x, UtQflat, ADD_VALUES);

            //
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Vt, Qbb, UtQ);
            Flat2D_IP(U->nDofsJ, Q->nDofsJ, UtQ, UtQflat);
            MatSetValues(M, U->nDofsJ, inds_y, Q->nDofsJ, inds_0y, UtQflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQ);
    Free2D(Q->nDofsI, Qaa);
    Free2D(Q->nDofsI, Qab);
    Free2D(Q->nDofsI, Qba);
    Free2D(Q->nDofsI, Qbb);
    delete[] UtQflat;
    delete[] inds_0x;
    delete[] inds_0y;
    delete Q;
    delete U;
    delete V;
}

UtQmat::~UtQmat() {
    MatDestroy(&M);
}

// 
WtQUmat::WtQUmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);
    W = new M2_j_xy_i(e);
    Q = new Wii(l->q, geom);
    Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Qab = Alloc2D(Q->nDofsI, Q->nDofsJ);
    WtQaa = Alloc2D(W->nDofsJ, Q->nDofsJ);
    WtQab = Alloc2D(W->nDofsJ, Q->nDofsJ);
    WtQU = Alloc2D(W->nDofsJ, U->nDofsJ);
    WtQV = Alloc2D(W->nDofsJ, V->nDofsJ);
    WtQUflat = new double[W->nDofsJ*U->nDofsJ];
    WtQVflat = new double[W->nDofsJ*V->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n1l, topo->nDofs2G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
}

void WtQUmat::assemble(Vec u1) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_2;
    double det, **J, ux[2];
    PetscScalar* u1Array;

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);

                Qaa[ii][ii] = 0.5*(ux[0]*J[0][0] + ux[1]*J[1][0])*Q->A[ii][ii]/det;
                Qab[ii][ii] = 0.5*(ux[0]*J[0][1] + ux[1]*J[1][1])*Q->A[ii][ii]/det;
            }

            Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
            Mult_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qaa, WtQaa);
            Mult_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qab, WtQab);

            Mult_IP(W->nDofsJ, U->nDofsJ, U->nDofsI, WtQaa, U->A, WtQU);
            Mult_IP(W->nDofsJ, V->nDofsJ, V->nDofsI, WtQab, V->A, WtQV);

            Flat2D_IP(W->nDofsJ, U->nDofsJ, WtQU, WtQUflat);
            Flat2D_IP(W->nDofsJ, V->nDofsJ, WtQV, WtQVflat);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_2 = topo->elInds2_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, U->nDofsJ, inds_x, WtQUflat, ADD_VALUES);
            MatSetValues(M, W->nDofsJ, inds_2, V->nDofsJ, inds_y, WtQVflat, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

void WtQUmat::assemble_up(Vec u1, double dt, Vec u_up) {
    int ex, ey, ei, ii, jj, mp1, mp12, np1, n0;
    int *inds_x, *inds_y, *inds_2;
    double det, **J, ux[2], ul[2], lx[99], ly[99], _ex[99], _ey[99], u_up_g[2];
    PetscScalar* u1Array;
    PetscScalar* u_upArray;
    GaussLobatto* quad = l->q;
    double** Uup = Alloc2D(U->nDofsI, U->nDofsJ);
    double** Vup = Alloc2D(U->nDofsI, U->nDofsJ);

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;
    n0 = l->n;
    np1 = l->n + 1;

    VecGetArray(u1, &u1Array);
    VecGetArray(u_up, &u_upArray);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u_upArray, u_up_g);

                // map velocity to local element coordinates
                ul[0] = (+J[1][1]*u_up_g[0] - J[0][1]*u_up_g[1])/det;
                ul[1] = (-J[1][0]*u_up_g[0] + J[0][0]*u_up_g[1])/det;
                // evaluate the nodal bases at the upwinded locations
                for(jj = 0; jj < np1; jj++) {
                    lx[jj] = l->eval_q(quad->x[ii%mp1] + dt*ul[0], jj);
                    ly[jj] = l->eval_q(quad->x[ii/mp1] + dt*ul[1], jj);
                }
                // evaluate the edge bases at the upwinded locations
                for(jj = 0; jj < n0; jj++) {
                    _ex[jj] = e->eval(quad->x[ii%mp1], jj);
                    _ey[jj] = e->eval(quad->x[ii/mp1], jj);
                }
                // evaluate the 2 form basis at the upwinded locations
                for(jj = 0; jj < n0*np1; jj++) {
                    Uup[ii][jj] = lx[jj%np1]*_ey[jj/np1];
                    Vup[ii][jj] = _ex[jj%n0]*ly[jj/n0];
                }

                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);

                Qaa[ii][ii] = 0.5*(ux[0]*J[0][0] + ux[1]*J[1][0])*Q->A[ii][ii]/det;
                Qab[ii][ii] = 0.5*(ux[0]*J[0][1] + ux[1]*J[1][1])*Q->A[ii][ii]/det;
            }

            Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
            Mult_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qaa, WtQaa);
            Mult_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qab, WtQab);

            Mult_IP(W->nDofsJ, U->nDofsJ, U->nDofsI, WtQaa, Uup, WtQU);
            Mult_IP(W->nDofsJ, V->nDofsJ, V->nDofsI, WtQab, Vup, WtQV);

            Flat2D_IP(W->nDofsJ, U->nDofsJ, WtQU, WtQUflat);
            Flat2D_IP(W->nDofsJ, V->nDofsJ, WtQV, WtQVflat);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_2 = topo->elInds2_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, U->nDofsJ, inds_x, WtQUflat, ADD_VALUES);
            MatSetValues(M, W->nDofsJ, inds_2, V->nDofsJ, inds_y, WtQVflat, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);
    VecRestoreArray(u_up, &u_upArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsI, Uup);
    Free2D(U->nDofsI, Vup);
}

WtQUmat::~WtQUmat() {
    Free2D(W->nDofsJ, Wt);
    Free2D(Q->nDofsI, Qaa);
    Free2D(Q->nDofsI, Qab);
    Free2D(W->nDofsJ, WtQU);
    Free2D(W->nDofsJ, WtQV);
    Free2D(W->nDofsJ, WtQaa);
    Free2D(W->nDofsJ, WtQab);
    delete[] WtQUflat;
    delete[] WtQVflat;
    delete U;
    delete V;
    delete W;
    delete Q;
    MatDestroy(&M);
}

// 1 form mass matrix with 0 form interpolated to quadrature points
// (for rotational term in the momentum equation)
RotMat::RotMat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    Q = new Wii(l->q, geom);
    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);

    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(V->nDofsJ, U->nDofsI);
    Qab = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Qba = Alloc2D(Q->nDofsI, Q->nDofsJ);
    UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQU = Alloc2D(V->nDofsJ, V->nDofsJ);

    UtQUflat = new double[U->nDofsJ*V->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
}

void RotMat::assemble(Vec q0) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y;
    double det, **J, vort;
    PetscScalar* q0Array;

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(q0, &q0Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp0(ex, ey, ii%mp1, ii/mp1, q0Array, &vort);

                Qab[ii][ii] = vort*(-J[0][0]*J[1][1] + J[0][1]*J[1][0])*Q->A[ii][ii]/det;
                Qba[ii][ii] = vort*(+J[0][0]*J[1][1] - J[0][1]*J[1][0])*Q->A[ii][ii]/det;
            }

            Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
            Tran_IP(U->nDofsI, V->nDofsJ, V->A, Vt);

            Mult_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Qab, UtQab);
            Mult_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Qba, VtQba);

            // take cross product by multiplying the x projection of the row vector with
            // the y component of the column vector and vice versa
            Mult_IP(U->nDofsJ, U->nDofsJ, U->nDofsI, UtQab, V->A, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, V->nDofsI, VtQba, U->A, VtQU);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);
        }
    }
    VecRestoreArray(q0, &q0Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

RotMat::~RotMat() {
    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    Free2D(Q->nDofsI, Qab);
    Free2D(Q->nDofsI, Qba);
    Free2D(U->nDofsJ, UtQab);
    Free2D(U->nDofsJ, VtQba);
    Free2D(U->nDofsJ, UtQV);
    Free2D(V->nDofsJ, VtQU);

    delete[] UtQUflat;
    delete Q;
    delete U;
    delete V;
    MatDestroy(&M);
}

// edge to node incidence matrix
E10mat::E10mat(Topo* _topo) {
    int ex, ey, nn, np1, ii, jj, kk, ll, row;
    int *inds_0, *inds_1x, *inds_1y;
    int cols[2];
    double vals[2];
    Mat E10t;

    topo = _topo;

    nn = topo->elOrd;
    np1 = nn + 1;

    MatCreate(MPI_COMM_WORLD, &E10);
    MatSetSizes(E10, topo->n1l, topo->n0l, topo->nDofs1G, topo->nDofs0G);
    MatSetType(E10, MATMPIAIJ);
    MatMPIAIJSetPreallocation(E10, 4, PETSC_NULL, 4, PETSC_NULL);
    MatZeroEntries(E10);
    
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_1x = topo->elInds1x_g(ex, ey);
            inds_1y = topo->elInds1y_g(ex, ey);
            inds_0 = topo->elInds0_g(ex, ey);

            // only set the incidence relations for edges that are local to this 
            // element. east and north edges are set by neighbouring elements.
            for(ii = 0; ii < nn; ii++) {
                for(jj = 0; jj < nn; jj++) {
                    // x-normal edge
                    kk = jj*np1 + ii;
                    ll = jj*np1 + ii;
                    row = inds_1x[kk];
                    cols[0] = inds_0[ll];
                    cols[1] = inds_0[ll+np1];
                    vals[0] = +1.0;
                    vals[1] = -1.0;
                    MatSetValues(E10, 1, &row, 2, cols, vals, INSERT_VALUES);

                    // y-normal edge
                    kk = jj*nn + ii;
                    ll = jj*np1 + ii;
                    row = inds_1y[kk];
                    cols[0] = inds_0[ll];
                    cols[1] = inds_0[ll+1];
                    vals[0] = -1.0;
                    vals[1] = +1.0;
                    MatSetValues(E10, 1, &row, 2, cols, vals, INSERT_VALUES);
                }
            }
        }
    }
    MatAssemblyBegin(E10, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(E10, MAT_FINAL_ASSEMBLY);

    // build the -ve of the transpose
    MatTranspose(E10, MAT_INITIAL_MATRIX, &E10t);
    MatDuplicate(E10t, MAT_DO_NOT_COPY_VALUES, &E01);
    MatZeroEntries(E01);
    MatAXPY(E01, -1.0, E10t, SAME_NONZERO_PATTERN);
    MatDestroy(&E10t);
}

E10mat::~E10mat() {
    MatDestroy(&E10);
    MatDestroy(&E01);
}

// face to edge incidence matrix
E21mat::E21mat(Topo* _topo) {
    int ex, ey, nn, np1, ii, jj, kk, row;
    int *inds_2, *inds_1x, *inds_1y;
    int cols[4];
    double vals[4];
    Mat E21t;

    topo = _topo;

    nn = topo->elOrd;
    np1 = nn + 1;

    MatCreate(MPI_COMM_WORLD, &E21);
    MatSetSizes(E21, topo->n2l, topo->n1l, topo->nDofs2G, topo->nDofs1G);
    MatSetType(E21, MATMPIAIJ);
    MatMPIAIJSetPreallocation(E21, 4, PETSC_NULL, 4, PETSC_NULL);
    MatZeroEntries(E21);
    
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_1x = topo->elInds1x_g(ex, ey);
            inds_1y = topo->elInds1y_g(ex, ey);
            inds_2 = topo->elInds2_g(ex, ey);

            for(ii = 0; ii < nn; ii++) {
                for(jj = 0; jj < nn; jj++) {
                    kk = ii*nn + jj;
                    row = inds_2[kk];
                    cols[0] = inds_1x[ii*np1+jj];
                    cols[1] = inds_1x[ii*np1+jj+1];
                    cols[2] = inds_1y[ii*nn+jj];
                    cols[3] = inds_1y[(ii+1)*nn+jj];
                    vals[0] = -1.0;
                    vals[1] = +1.0;
                    vals[2] = -1.0;
                    vals[3] = +1.0;
                    MatSetValues(E21, 1, &row, 4, cols, vals, INSERT_VALUES);
                }
            }
        }
    }
    MatAssemblyBegin(E21, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(E21, MAT_FINAL_ASSEMBLY);

    // build the -ve of the transpose
    MatTranspose(E21, MAT_INITIAL_MATRIX, &E21t);
    MatDuplicate(E21t, MAT_DO_NOT_COPY_VALUES, &E12);
    MatZeroEntries(E12);
    MatAXPY(E12, -1.0, E21t, SAME_NONZERO_PATTERN);
    MatDestroy(&E21t);
}

E21mat::~E21mat() {
    MatDestroy(&E21);
    MatDestroy(&E12);
}

// 2 form mass matrix with pressure
Whmat::Whmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    M2_j_xy_i* W;

    topo = _topo;
    geom = _geom;
    e = _e;

    W = new M2_j_xy_i(e);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);

    delete W;
}

void Whmat::assemble(Vec h2) {
    int ex, ey, ei, mp1, mp12, ii, *inds;
    double det, hi;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar* hArray;

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    VecGetArray(h2, &hArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, hArray, &hi);
                det = geom->det[ei][ii];
                Qaa[ii][ii] = hi*Q->A[ii][ii]/det;
            }

            Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
            Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQWflat, ADD_VALUES);
        }
    }
    VecRestoreArray(h2, &hArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Qaa);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete W;
    delete Q;
    delete[] WtQWflat;
}

Whmat::~Whmat() {
    MatDestroy(&M);
}

// NOTE: this assumes a diagonal mass matrix
P_up_mat::P_up_mat(Topo* _topo, Geom* _geom, LagrangeNode* _node) {
    topo = _topo;
    geom = _geom;
    node = _node;

    Q = new Wii(node->q, geom);
    QP = Alloc2D(Q->nDofsJ, Q->nDofsJ);
    QPI = Alloc2D(Q->nDofsJ, Q->nDofsJ);
    QPflat = new double[Q->nDofsJ*Q->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*Q->nDofsJ, PETSC_NULL, 8*Q->nDofsJ, PETSC_NULL);

    MatCreate(MPI_COMM_WORLD, &I);
    MatSetSizes(I, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(I, MATMPIAIJ);
    MatMPIAIJSetPreallocation(I, 8*Q->nDofsJ, PETSC_NULL, 8*Q->nDofsJ, PETSC_NULL);
}

void P_up_mat::assemble(Vec ul, double dt) {
    int ex, ey, ei, mp1, mp12, np1, np12, ii, *inds;
    double ux[2], lx[99], ly[99], det, **J, ux2[2];
    PetscScalar* uArray;
    GaussLobatto* quad = node->q;

    np1 = node->n + 1;
    np12 = np1*np1;
    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    MatZeroEntries(I);

    VecGetArray(ul, &uArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, ux);

                ux2[0] = +J[1][1]*ux[0]/det - J[0][1]*ux[1]/det;
                ux2[1] = -J[1][0]*ux[0]/det + J[0][0]*ux[1]/det;
                for(int jj = 0; jj < np1; jj++) {
                    lx[jj] = node->eval_q(quad->x[ii%mp1] - dt*ux2[0], jj);
                    ly[jj] = node->eval_q(quad->x[ii/mp1] - dt*ux2[1], jj);
                }
                for(int jj = 0; jj < np12; jj++) {
                    QP[ii][jj]  = det * Q->A[ii][ii] * lx[jj%np1] * ly[jj/np1];
                    QPI[ii][jj] = lx[jj%np1] * ly[jj/np1];
                }
            }

            inds = topo->elInds0_g(ex, ey);

            Flat2D_IP(Q->nDofsJ, Q->nDofsJ, QP, QPflat);
            MatSetValues(M, Q->nDofsJ, inds, Q->nDofsJ, inds, QPflat, ADD_VALUES);

            Flat2D_IP(Q->nDofsJ, Q->nDofsJ, QPI, QPflat);
            MatSetValues(I, Q->nDofsJ, inds, Q->nDofsJ, inds, QPflat, ADD_VALUES);
        }
    }
    VecRestoreArray(ul, &uArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(I, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  I, MAT_FINAL_ASSEMBLY);
}

void P_up_mat::assemble_h(Vec ul, Vec hl, double dt) {
    int ex, ey, ei, mp1, mp12, np1, np12, ii, *inds;
    double ux[2], lx[99], ly[99], det, **J, ux2[2], hx;
    PetscScalar *uArray, *hArray;
    GaussLobatto* quad = node->q;

    ux2[0] = ux2[1] = 0.0;

    np1 = node->n + 1;
    np12 = np1*np1;
    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    MatZeroEntries(I);

    VecGetArray(hl, &hArray);
    if(ul) VecGetArray(ul, &uArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                if(ul) {
                    geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, ux);
                    ux2[0] = +J[1][1]*ux[0]/det - J[0][1]*ux[1]/det;
                    ux2[1] = -J[1][0]*ux[0]/det + J[0][0]*ux[1]/det;
                }
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, hArray, &hx);

                for(int jj = 0; jj < np1; jj++) {
                    lx[jj] = node->eval_q(quad->x[ii%mp1] - dt*ux2[0], jj);
                    ly[jj] = node->eval_q(quad->x[ii/mp1] - dt*ux2[1], jj);
                }
                for(int jj = 0; jj < np12; jj++) {
                    QP[ii][jj]  = hx * det * Q->A[ii][ii] * lx[jj%np1] * ly[jj/np1];
                    QPI[ii][jj] = lx[jj%np1] * ly[jj/np1];
                }
            }

            inds = topo->elInds0_g(ex, ey);

            Flat2D_IP(Q->nDofsJ, Q->nDofsJ, QP, QPflat);
            MatSetValues(M, Q->nDofsJ, inds, Q->nDofsJ, inds, QPflat, ADD_VALUES);

            Flat2D_IP(Q->nDofsJ, Q->nDofsJ, QPI, QPflat);
            MatSetValues(I, Q->nDofsJ, inds, Q->nDofsJ, inds, QPflat, ADD_VALUES);
        }
    }
    VecRestoreArray(hl, &hArray);
    if(ul) VecRestoreArray(ul, &uArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(I, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  I, MAT_FINAL_ASSEMBLY);
}

P_up_mat::~P_up_mat() {
    MatDestroy(&M);
    MatDestroy(&I);
    Free2D(Q->nDofsJ, QP);
    Free2D(Q->nDofsJ, QPI);
    delete[] QPflat;
    delete Q;
}

RotMat_up::RotMat_up(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    Q = new Wii(l->q, geom);
    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);

    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(V->nDofsJ, U->nDofsI);
    Qab = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Qba = Alloc2D(Q->nDofsI, Q->nDofsJ);
    UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQU = Alloc2D(V->nDofsJ, V->nDofsJ);

    UtQUflat = new double[U->nDofsJ*V->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
}

void RotMat_up::assemble(Vec q0, Vec ul, double fac, double dt) {
    int ex, ey, ei, ii, jj, mp1, mp12, np1, np12;
    int *inds_x, *inds_y, *inds_0;
    double det, **J, vort, ux[2], lx[99], ly[99], ux2[2], tau, tmp, tmp2;
    PetscScalar *q0Array, *u1Array;
    GaussLobatto* quad = l->q;

    np1 = l->n + 1;
    np12 = np1*np1;
    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(q0, &q0Array);
    VecGetArray(ul, &u1Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_0 = topo->elInds0_l(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);
                ux2[0] = +J[1][1]*ux[0]/det - J[0][1]*ux[1]/det;
                ux2[1] = -J[1][0]*ux[0]/det + J[0][0]*ux[1]/det;

                tau = 1.0/(1.0/(fac*dt)/* + tmp + tmp2*/);

                for(jj = 0; jj < np1; jj++) {
                    lx[jj] = l->eval_q(quad->x[ii%mp1] - tau*ux2[0], jj);
                    ly[jj] = l->eval_q(quad->x[ii/mp1] - tau*ux2[1], jj);
                }
                vort = 0.0;
                for(jj = 0; jj < np12; jj++) {
                    vort += q0Array[inds_0[jj]] * lx[jj%np1] * ly[jj/np1];
                }

                Qab[ii][ii] = vort*(-J[0][0]*J[1][1] + J[0][1]*J[1][0])*Q->A[ii][ii]/det;
                Qba[ii][ii] = vort*(+J[0][0]*J[1][1] - J[0][1]*J[1][0])*Q->A[ii][ii]/det;
            }

            Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
            Tran_IP(U->nDofsI, V->nDofsJ, V->A, Vt);

            Mult_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Qab, UtQab);
            Mult_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Qba, VtQba);

            // take cross product by multiplying the x projection of the row vector with
            // the y component of the column vector and vice versa
            Mult_IP(U->nDofsJ, U->nDofsJ, U->nDofsI, UtQab, V->A, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, V->nDofsI, VtQba, U->A, VtQU);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);
        }
    }
    VecRestoreArray(q0, &q0Array);
    VecRestoreArray(ul, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

void RotMat_up::assemble_supg(Vec q0, Vec ul, Vec dql, double fac, double dt, Vec qi) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y;
    double det, **J, vort, ux[2], dq[2], vort_i, tau;
    PetscScalar *q0Array, *u1Array, *dqArray, *qiArray;

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(q0, &q0Array);
    VecGetArray(ul, &u1Array);
    VecGetArray(dql, &dqArray);
    VecGetArray(qi, &qiArray);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                geom->interp0(ex, ey, ii%mp1, ii/mp1, q0Array, &vort);
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, dqArray, dq);
                geom->interp0(ex, ey, ii%mp1, ii/mp1, qiArray, &vort_i);

                //ux2[0] = +J[1][1]*ux[0]/det - J[0][1]*ux[1]/det;
                //ux2[1] = -J[1][0]*ux[0]/det + J[0][0]*ux[1]/det;
                //dq2[0] = +J[1][1]*dq[0]/det - J[0][1]*dq[1]/det;
                //dq2[1] = -J[1][0]*dq[0]/det + J[0][0]*dq[1]/det;
                //vort -= tau*(ux2[0]*dq2[1] - ux2[1]*dq2[0]);

                //geom->interp1_l(ex, ey, ii%mp1, ii/mp1, u1Array, uxl);
                //geom->interp1_l(ex, ey, ii%mp1, ii/mp1, ones, bl);
                //tmp = uxl[0]*bl[0] + uxl[1]*bl[1];
                //tau2 = 1.0/(1.0/tau + 1.0/sqrt(tmp*tmp));

                //geom->interp0(ex, ey, ii%mp1, ii/mp1, ones, bl);
                //tmp = sqrt((ux[0]*ux[0] + ux[1]*ux[1])*(ux[0]*ux[0] + ux[1]*ux[1]));
                //tmp = sqrt(ux[0]*ux[0] + ux[1]*ux[1]);
                //tmp *= bl[0]/sqrt(det);
                //tau = 1.0/(1.0/tau + tmp);
                tau = fabs(fac*dt);

                vort -= tau*(ux[0]*dq[1] - ux[1]*dq[0] + (vort - vort_i)/dt);

                Qab[ii][ii] = vort*(-J[0][0]*J[1][1] + J[0][1]*J[1][0])*Q->A[ii][ii]/det;
                Qba[ii][ii] = vort*(+J[0][0]*J[1][1] - J[0][1]*J[1][0])*Q->A[ii][ii]/det;
            }

            Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
            Tran_IP(U->nDofsI, V->nDofsJ, V->A, Vt);

            Mult_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Qab, UtQab);
            Mult_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Qba, VtQba);

            // take cross product by multiplying the x projection of the row vector with
            // the y component of the column vector and vice versa
            Mult_IP(U->nDofsJ, U->nDofsJ, U->nDofsI, UtQab, V->A, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, V->nDofsI, VtQba, U->A, VtQU);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);
        }
    }
    VecRestoreArray(q0, &q0Array);
    VecRestoreArray(ul, &u1Array);
    VecRestoreArray(dql, &dqArray);
    VecRestoreArray(qi, &qiArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

RotMat_up::~RotMat_up() {
    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    Free2D(Q->nDofsI, Qab);
    Free2D(Q->nDofsI, Qba);
    Free2D(U->nDofsJ, UtQab);
    Free2D(U->nDofsJ, VtQba);
    Free2D(U->nDofsJ, UtQV);
    Free2D(V->nDofsJ, VtQU);

    delete[] UtQUflat;
    delete Q;
    delete U;
    delete V;
    MatDestroy(&M);
}

// piecewise constant 1 form mass matrix
U0mat::U0mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    assemble();
}

void U0mat::assemble() {
    int ex, ey, ei, ii, jj, nx, nxp1;
    int *inds_x, *inds_y;
    double det, **J;
    double** U = Alloc2D(4, 2);
    double** V = Alloc2D(4, 2);
    double** Ut = Alloc2D(2, 4);
    double** Vt = Alloc2D(2, 4);
    double** UtQaa = Alloc2D(2, 4);
    double** UtQab = Alloc2D(2, 4);
    double** VtQba = Alloc2D(2, 4);
    double** VtQbb = Alloc2D(2, 4);
    double** UtQU = Alloc2D(2, 4);
    double** UtQV = Alloc2D(2, 4);
    double** VtQU = Alloc2D(2, 4);
    double** VtQV = Alloc2D(2, 4);
    double** Qaa = Alloc2D(4, 4);
    double** Qab = Alloc2D(4, 4);
    double** Qbb = Alloc2D(4, 4);
    double* UtQUflat = new double[2*2];
    int vertInds[4], edgeIndsX[2], edgeIndsY[2];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 32, PETSC_NULL, 32, PETSC_NULL);
    MatZeroEntries(M);

    nx = l->q->n;
    nxp1 = nx+1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            for(ii = 0; ii < nx*nx; ii++) {
                vertInds[0] = (ii/nx)*nxp1 + ii%nx;
                vertInds[1] = vertInds[0] + 1;
                vertInds[2] = vertInds[0] + nxp1;
                vertInds[3] = vertInds[2] + 1;

                edgeIndsX[0] = inds_x[(ii/nx)*nxp1 + ii%nx];
                edgeIndsX[1] = inds_x[(ii/nx)*nxp1 + ii%nx + 1];

                edgeIndsY[0] = inds_y[(ii/nx)*nx + ii%nx];
                edgeIndsY[1] = inds_y[(ii/nx)*nx + ii%nx + nx];

                for(jj = 0; jj < 4; jj++) {
                    det = geom->det[ei][vertInds[jj]];
                    J = geom->J[ei][vertInds[jj]];

                    Qaa[jj][jj] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])/det;
                    Qab[jj][jj] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])/det;
                    Qbb[jj][jj] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])/det;

                    U[jj][0] = (jj%2==0) ? 0.5 : 0.0; // left
                    U[jj][1] = (jj%2==1) ? 0.5 : 0.0; // right
                    V[jj][0] = (jj/2==0) ? 0.5 : 0.0; // bottom
                    V[jj][1] = (jj/2==1) ? 0.5 : 0.0; // top
                }
 
                Tran_IP(4, 2, U, Ut);
                Tran_IP(4, 2, V, Vt);

                Mult_IP(2, 4, 4, Ut, Qaa, UtQaa);
                Mult_IP(2, 4, 4, Ut, Qab, UtQab);
                Mult_IP(2, 4, 4, Vt, Qab, VtQba);
                Mult_IP(2, 4, 4, Vt, Qbb, VtQbb);

                Mult_IP(2, 2, 4, UtQaa, U, UtQU);
                Mult_IP(2, 2, 4, UtQab, V, UtQV);
                Mult_IP(2, 2, 4, VtQba, U, VtQU);
                Mult_IP(2, 2, 4, VtQbb, V, VtQV);

                Flat2D_IP(2, 2, UtQU, UtQUflat);
                MatSetValues(M, 2, edgeIndsX, 2, edgeIndsX, UtQUflat, ADD_VALUES);

                Flat2D_IP(2, 2, UtQV, UtQUflat);
                MatSetValues(M, 2, edgeIndsX, 2, edgeIndsY, UtQUflat, ADD_VALUES);

                Flat2D_IP(2, 2, VtQU, UtQUflat);
                MatSetValues(M, 2, edgeIndsY, 2, edgeIndsX, UtQUflat, ADD_VALUES);

                Flat2D_IP(2, 2, VtQV, UtQUflat);
                MatSetValues(M, 2, edgeIndsY, 2, edgeIndsY, UtQUflat, ADD_VALUES);
            }
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(4, U);
    Free2D(4, V);
    Free2D(2, Ut);
    Free2D(2, Vt);
    Free2D(2, UtQaa);
    Free2D(2, UtQab);
    Free2D(2, VtQba);
    Free2D(2, VtQbb);
    Free2D(2, UtQU);
    Free2D(2, UtQV);
    Free2D(2, VtQU);
    Free2D(2, VtQV);
    Free2D(4, Qaa);
    Free2D(4, Qab);
    Free2D(4, Qbb);
    delete[] UtQUflat;
}

U0mat::~U0mat() {
    MatDestroy(&M);
}

W_IP_mat::W_IP_mat(Topo* _topo, Geom* _geom, LagrangeEdge* _edge) {
    int m, mp1, mp12;
    M2_j_xy_i* W;
    M1x_j_xy_i* U;

    topo = _topo;
    geom = _geom;
    edge = _edge;

    W = new M2_j_xy_i(edge);
    U = new M1x_j_xy_i(edge->l, edge);

    m    = edge->l->q->n;
    mp1  = m + 1;
    mp12 = mp1*mp1;

    MatCreate(MPI_COMM_WORLD, &M_QW);
    MatSetSizes(M_QW, geom->n0l, topo->n2l, geom->nDofs0G, topo->nDofs2G);
    MatSetType(M_QW, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M_QW, 8*W->nDofsJ, PETSC_NULL, 8*W->nDofsJ, PETSC_NULL);

    MatCreate(MPI_COMM_WORLD, &M_Q);
    MatSetSizes(M_Q, geom->n0l, geom->n0l, geom->nDofs0G, geom->nDofs0G);
    MatSetType(M_Q, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M_Q, 4*mp12, PETSC_NULL, 4*mp12, PETSC_NULL);

    assemble_QW();
    MatTranspose(M_QW, MAT_INITIAL_MATRIX, &M_WQ);

    MatCreate(MPI_COMM_WORLD, &M_UtQ);
    MatSetSizes(M_UtQ, topo->n1l, geom->n0l, topo->nDofs1G, geom->nDofs0G);
    MatSetType(M_UtQ, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M_UtQ, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);

    delete W;
    delete U;
}

void W_IP_mat::assemble_QW() {
    int ex, ey, ei, m, mp1, mp12, ii, jj, *inds_2, *inds_0;
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    Wii* Q = new Wii(edge->l->q, geom);
    double det;
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** QW = Alloc2D(Q->nDofsJ, W->nDofsJ);
    double* QWflat = new double[W->nDofsJ*Q->nDofsJ];

    m    = edge->l->q->n;
    mp1  = m + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M_QW);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

	    for(ii = 0; ii < mp12; ii++) Qaa[ii][ii] = 0.0;

            for(jj = 0; jj < mp1; jj++) {
                // bottom
                ii    = jj;
                det   = geom->det[ei][ii];
		Qaa[ii][ii] += 1.0/det;
                // top
                ii    = jj + m*mp1;
                det   = geom->det[ei][ii];
		Qaa[ii][ii] -= 1.0/det;
                // left
                ii    = jj*mp1;
                det   = geom->det[ei][ii];
		Qaa[ii][ii] += 1.0/det;
                // right
                ii    = jj*mp1 + m;
                det   = geom->det[ei][ii];
		Qaa[ii][ii] -= 1.0/det;
            }
            Mult_IP(Q->nDofsJ, W->nDofsJ, Q->nDofsI, Qaa, W->A, QW);
            Flat2D_IP(Q->nDofsJ, W->nDofsJ, QW, QWflat);

            inds_2 = topo->elInds2_g(ex, ey);
            inds_0 = geom->elInds0_g(ex, ey);

            MatSetValues(M_QW, Q->nDofsI, inds_0, W->nDofsJ, inds_2, QWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M_QW, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M_QW, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Qaa);
    Free2D(W->nDofsJ, QW);
    delete[] QWflat;
    delete W;
    delete Q;
}

void global_norm(double** J, int dim, double* norm_g) {
    double norm_sq, norm_inv;

    norm_g[0] = J[0][dim];
    norm_g[1] = J[1][dim];

    norm_sq = norm_g[0]*norm_g[0] + norm_g[1]*norm_g[1];
    norm_inv = 1.0/sqrt(norm_sq);

    norm_g[0] = norm_inv*norm_g[0];
    norm_g[1] = norm_inv*norm_g[1];
}

void W_IP_mat::assemble_Q(Vec ul) {
    int ex, ey, ei, m, mp1, mp12, ii, jj, *inds_0;
    Wii* Q = new Wii(edge->l->q, geom);
    double **J, det_l, uq[2];
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double* Qflat = new double[Q->nDofsJ*Q->nDofsJ];
    double* wts = edge->l->q->w;
    PetscScalar* uArray;
    double norm_g[2];

    m    = edge->l->q->n;
    mp1  = m + 1;
    mp12 = mp1*mp1;

    uq[0] = uq[1] = 1.0;

    MatZeroEntries(M_Q);
    VecGetArray(ul, &uArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

	    for(ii = 0; ii < mp12; ii++) Qaa[ii][ii] = 0.0;

            for(jj = 0; jj < mp1; jj++) {
                // bottom
                ii           = jj;
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, uq);
                J            = geom->J[ei][ii];
                global_norm(J, 1, norm_g);
                det_l        = sqrt(fabs(J[0][0]*J[0][0] + J[1][0]*J[1][0]));
                Qaa[ii][ii] += 0.5*wts[jj]*det_l*fabs(uq[0]*norm_g[0] + uq[1]*norm_g[1]);
                // top
                ii           = jj + m*mp1;
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, uq);
                J            = geom->J[ei][ii];
                global_norm(J, 1, norm_g);
                det_l        = sqrt(fabs(J[0][0]*J[0][0] + J[1][0]*J[1][0]));
                Qaa[ii][ii] += 0.5*wts[jj]*det_l*fabs(uq[0]*norm_g[0] + uq[1]*norm_g[1]);
                // left
                ii           = jj*mp1;
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, uq);
                J            = geom->J[ei][ii];
                global_norm(J, 0, norm_g);
                det_l        = sqrt(fabs(J[0][1]*J[0][1] + J[1][1]*J[1][1]));
                Qaa[ii][ii] += 0.5*wts[jj]*det_l*fabs(uq[0]*norm_g[0] + uq[1]*norm_g[1]);
                // right
                ii           = jj*mp1 + m;
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, uq);
                J            = geom->J[ei][ii];
                global_norm(J, 0, norm_g);
                det_l        = sqrt(fabs(J[0][1]*J[0][1] + J[1][1]*J[1][1]));
                Qaa[ii][ii] += 0.5*wts[jj]*det_l*fabs(uq[0]*norm_g[0] + uq[1]*norm_g[1]);
            }
            Flat2D_IP(Q->nDofsI, Q->nDofsJ, Qaa, Qflat);

            inds_0 = geom->elInds0_g(ex, ey);

            MatSetValues(M_Q, Q->nDofsI, inds_0, Q->nDofsJ, inds_0, Qflat, ADD_VALUES);
        }
    }
    VecRestoreArray(ul, &uArray);
    MatAssemblyBegin(M_Q, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M_Q, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Qaa);
    delete[] Qflat;
    delete Q;
}

void W_IP_mat::assemble_UtQ(Vec ul, Vec b_jump) {
    int ex, ey, ei, m, mp1, mp12, ii, jj, kk, *inds_0, *inds_x, *inds_y;
    double **J, det_l, uq[2], fac, det;
    double* wts = edge->l->q->w;
    PetscScalar* uArray;
    PetscScalar* bArray;
    double norm_g[2];
    M1x_j_xy_i* U = new M1x_j_xy_i(edge->l, edge);
    M1y_j_xy_i* V = new M1y_j_xy_i(edge->l, edge);
    double** UtQ = Alloc2D(U->nDofsJ, U->nDofsI);
    double** VtQ = Alloc2D(U->nDofsJ, U->nDofsI);
    double* UtQflat = new double[U->nDofsI*U->nDofsJ];
    double* VtQflat = new double[U->nDofsI*U->nDofsJ];
    double sgn, F_dot_n;

    m    = edge->l->q->n;
    mp1  = m + 1;
    mp12 = mp1*mp1;

    uq[0] = uq[1] = 1.0;

    MatZeroEntries(M_UtQ);
    VecGetArray(ul, &uArray);
    VecGetArray(b_jump, &bArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

	    for(ii = 0; ii < mp12; ii++) {
                for(jj = 0; jj < U->nDofsJ; jj++) {
                    UtQ[jj][ii] = 0.0;
                    VtQ[jj][ii] = 0.0;
                }
            }

            for(jj = 0; jj < mp1; jj++) {
                // bottom
                ii           = jj;
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, uq);
                J            = geom->J[ei][ii];
                det          = geom->det[ei][ii];
                global_norm(J, 1, norm_g);
                F_dot_n      = uq[0]*norm_g[0] + uq[1]*norm_g[1];
                sgn          = (F_dot_n > 0.0) ? 1.0 : -1.0;
                det_l        = sqrt(fabs(J[0][0]*J[0][0] + J[1][0]*J[1][0]));
                fac          = 0.5*wts[jj]*det_l*sgn*bArray[ii]/det;
                // U^T * J^T/|J| * n^ * sgn(F.n)
                for(kk = 0; kk < U->nDofsJ; kk++) {
                    UtQ[kk][ii] += fac*U->A[ii][kk]*(J[0][0]*norm_g[0] + J[1][0]*norm_g[1]);
                    VtQ[kk][ii] += fac*V->A[ii][kk]*(J[0][1]*norm_g[0] + J[1][1]*norm_g[1]);
                }

                // top
                ii           = jj + m*mp1;
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, uq);
                J            = geom->J[ei][ii];
                det          = geom->det[ei][ii];
                global_norm(J, 1, norm_g);
                F_dot_n      = uq[0]*norm_g[0] + uq[1]*norm_g[1];
                sgn          = (F_dot_n > 0.0) ? 1.0 : -1.0;
                det_l        = sqrt(fabs(J[0][0]*J[0][0] + J[1][0]*J[1][0]));
                fac          = 0.5*wts[jj]*det_l*sgn*bArray[ii]/det;
                // U^T * J^T/|J| * n^ * sgn(F.n)
                for(kk = 0; kk < U->nDofsJ; kk++) {
                    UtQ[kk][ii] += fac*U->A[ii][kk]*(J[0][0]*norm_g[0] + J[1][0]*norm_g[1]);
                    VtQ[kk][ii] += fac*V->A[ii][kk]*(J[0][1]*norm_g[0] + J[1][1]*norm_g[1]);
                }

                // left
                ii           = jj*mp1;
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, uq);
                J            = geom->J[ei][ii];
                det          = geom->det[ei][ii];
                global_norm(J, 0, norm_g);
                F_dot_n      = uq[0]*norm_g[0] + uq[1]*norm_g[1];
                sgn          = (F_dot_n > 0.0) ? 1.0 : -1.0;
                det_l        = sqrt(fabs(J[0][1]*J[0][1] + J[1][1]*J[1][1]));
                fac          = 0.5*wts[jj]*det_l*sgn*bArray[ii]/det;
                // U^T * J^T/|J| * n^ * sgn(F.n)
                for(kk = 0; kk < U->nDofsJ; kk++) {
                    UtQ[kk][ii] += fac*U->A[ii][kk]*(J[0][0]*norm_g[0] + J[1][0]*norm_g[1]);
                    VtQ[kk][ii] += fac*V->A[ii][kk]*(J[0][1]*norm_g[0] + J[1][1]*norm_g[1]);
                }

                // right
                ii           = jj*mp1 + m;
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, uq);
                J            = geom->J[ei][ii];
                det          = geom->det[ei][ii];
                global_norm(J, 0, norm_g);
                F_dot_n      = uq[0]*norm_g[0] + uq[1]*norm_g[1];
                sgn          = (F_dot_n > 0.0) ? 1.0 : -1.0;
                det_l        = sqrt(fabs(J[0][1]*J[0][1] + J[1][1]*J[1][1]));
                fac          = 0.5*wts[jj]*det_l*sgn*bArray[ii]/det;
                // U^T * J^T/|J| * n^ * sgn(F.n)
                for(kk = 0; kk < U->nDofsJ; kk++) {
                    UtQ[kk][ii] += fac*U->A[ii][kk]*(J[0][0]*norm_g[0] + J[1][0]*norm_g[1]);
                    VtQ[kk][ii] += fac*V->A[ii][kk]*(J[0][1]*norm_g[0] + J[1][1]*norm_g[1]);
                }
            }

            Flat2D_IP(U->nDofsJ, U->nDofsI, UtQ, UtQflat);
            Flat2D_IP(V->nDofsJ, U->nDofsI, VtQ, VtQflat);

            inds_0 = geom->elInds0_g(ex, ey);
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            MatSetValues(M_UtQ, U->nDofsJ, inds_x, U->nDofsI, inds_0, UtQflat, ADD_VALUES);
            MatSetValues(M_UtQ, U->nDofsJ, inds_y, U->nDofsI, inds_0, VtQflat, ADD_VALUES);
        }
    }
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(b_jump, &bArray);
    MatAssemblyBegin(M_Q, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M_Q, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, UtQ);
    Free2D(U->nDofsJ, VtQ);
    delete[] UtQflat;
    delete[] VtQflat;
    delete U;
    delete V;
}

W_IP_mat::~W_IP_mat() {
    MatDestroy(&M_UtQ);
    MatDestroy(&M_WQ);
    MatDestroy(&M_QW);
    MatDestroy(&M_Q);
}
