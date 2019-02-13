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

    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
    MatSetOptionsPrefix(M, "M1_");

    delete U;

    assemble(1.0);
}

void Umat::assemble(double scale) {
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

    MatZeroEntries(M);

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii][ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]*(scale/det/det);
                Qab[ii][ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
                Qbb[ii][ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
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

Umat::~Umat() {
    MatDestroy(&M);
}

// 2 form mass matrix
Wmat::Wmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    M2_j_xy_i* W = new M2_j_xy_i(e);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);

    delete W;

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

    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii][ii] = Q->A[ii][ii]/det/det;
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

void Uhmat::assemble(Vec h2, double scale) {
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
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, h2Array, &hi);

                Qaa[ii][ii] = hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]*(scale/det/det);
                Qab[ii][ii] = hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
                Qbb[ii][ii] = hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
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
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            for(ii = 0; ii < np12; ii++) {
                entries[ii] = Q->A[ii][ii];
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
    int ex, ey, ei, mp1, mp12, ii, *inds_2, *inds_0;
    double det;
    M2_j_xy_i* W = new M2_j_xy_i(e);
    Wii* Q = new Wii(e->l->q, geom);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQflat = new double[W->nDofsJ*Q->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n0l, topo->nDofs2G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii][ii] = Q->A[ii][ii]/det;
            }

            Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

            Mult_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qaa, WtQ);
            Flat2D_IP(W->nDofsJ, Q->nDofsJ, WtQ, WtQflat);

            inds_2 = topo->elInds2_g(ex, ey);
            inds_0 = topo->elInds0_g(ex, ey);

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
    int ex, ey;
    int *inds_0;
    M0_j_xy_i* P = new M0_j_xy_i(l);
    Wii* Q = new Wii(l->q, geom);
    double** Pt = Tran(P->nDofsI, P->nDofsJ, P->A);
    double** PtQ = Alloc2D(P->nDofsJ, Q->nDofsJ);
    double* PtQflat = new double[P->nDofsJ*Q->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 4*P->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);
            Mult_IP(P->nDofsJ, Q->nDofsJ, Q->nDofsI, Pt, Q->A, PtQ);
            Flat2D_IP(P->nDofsJ, Q->nDofsJ, PtQ, PtQflat);

            inds_0 = topo->elInds0_g(ex, ey);
            MatSetValues(M, P->nDofsJ, inds_0, Q->nDofsJ, inds_0, PtQflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
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
    double det, **J;
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
    MatSetSizes(M, topo->n1l, 2*topo->n0l, topo->nDofs1G, 2*topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii][ii] = J[0][0]*Q->A[ii][ii]/det;
                Qab[ii][ii] = J[1][0]*Q->A[ii][ii]/det;
                Qba[ii][ii] = J[0][1]*Q->A[ii][ii]/det;
                Qbb[ii][ii] = J[1][1]*Q->A[ii][ii]/det;
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_0 = topo->elInds0_g(ex, ey);
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
    MatSetOptionsPrefix(M, "WU2_");
}

void WtQUmat::assemble(Vec u1) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_2;
    double det, **J, ux[2];
    PetscScalar* u1Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);

                Qaa[ii][ii] = 0.5*(ux[0]*J[0][0] + ux[1]*J[1][0])*Q->A[ii][ii]/det/det;
                Qab[ii][ii] = 0.5*(ux[0]*J[0][1] + ux[1]*J[1][1])*Q->A[ii][ii]/det/det;
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
    MatSetOptionsPrefix(M, "R1_");
}

void RotMat::assemble(Vec q0) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y;
    double det, **J, vort;
    PetscScalar* q0Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(q0, &q0Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp0(ex, ey, ii%mp1, ii/mp1, q0Array, &vort);

                Qab[ii][ii] = vort*(-J[0][0]*J[1][1] + J[0][1]*J[1][0])*Q->A[ii][ii]/det/det;
                Qba[ii][ii] = vort*(+J[0][0]*J[1][1] - J[0][1]*J[1][0])*Q->A[ii][ii]/det/det;
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

////////////////////////////////////////////////////////////////////////
Krhs::Krhs(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;
}

double GetSign(int jj, int mp1) {
    if(jj == 0) {
        return +2.0;
    }
    else if(jj == mp1*mp1-1) {
        return -2.0;
    }
    else if(jj == mp1-1 || jj == (mp1-1)*mp1) {
        return  0.0;
    }
    else if(jj%mp1 == 0 || jj/mp1 == 0) {
        return +1.0;
    }
    else if(jj%mp1 == mp1-1 || jj/mp1 == mp1-1) {
        return -1.0;
    }
    else {
        return  0.0;
    }
}

void Krhs::assemble(Vec kq, Vec* F) {
    int jj;
    Vec Fl;
    PetscScalar *kArray, *fArray;
    double val;

    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    double det, **J, sgn;
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

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &Fl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, F);

    VecZeroEntries(Fl);

    VecGetArray(Fl, &fArray);
    VecGetArray(kq, &kArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii][ii] = J[0][0]*Q->A[ii][ii]/det;
                Qab[ii][ii] = J[1][0]*Q->A[ii][ii]/det;
                Qba[ii][ii] = J[0][1]*Q->A[ii][ii]/det;
                Qbb[ii][ii] = J[1][1]*Q->A[ii][ii]/det;
            }

            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);
            inds_0 = topo->elInds0_l(ex, ey);

            Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
            Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

            //
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Qaa, UtQ);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                val = 0.0;
                for(jj = 0; jj < Q->nDofsJ; jj++) {
                    sgn = GetSign(jj, mp1);
                    val += sgn*UtQ[ii][jj]*kArray[inds_0[jj]];
                }
                fArray[inds_x[ii]] += val;
            }

            //
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Qab, UtQ);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                val = 0.0;
                for(jj = 0; jj < Q->nDofsJ; jj++) {
                    sgn = GetSign(jj, mp1);
                    val += sgn*UtQ[ii][jj]*kArray[inds_0[jj]];
                }
                fArray[inds_x[ii]] += val;
            }

            //
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Vt, Qba, UtQ);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                val = 0.0;
                for(jj = 0; jj < Q->nDofsJ; jj++) {
                    sgn = GetSign(jj, mp1);
                    val += sgn*UtQ[ii][jj]*kArray[inds_0[jj]];
                }
                fArray[inds_y[ii]] += val;
            }

            //
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Vt, Qbb, UtQ);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                val = 0.0;
                for(jj = 0; jj < Q->nDofsJ; jj++) {
                    sgn = GetSign(jj, mp1);
                    val += sgn*UtQ[ii][jj]*kArray[inds_0[jj]];
                }
                fArray[inds_y[ii]] += val;
            }
        }
    }
    VecRestoreArray(Fl, &fArray);
    VecRestoreArray(kq, &kArray);

    VecScatterBegin(topo->gtol_1, Fl, *F, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_1, Fl, *F, INSERT_VALUES, SCATTER_REVERSE);

    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQ);
    Free2D(Q->nDofsI, Qaa);
    Free2D(Q->nDofsI, Qab);
    Free2D(Q->nDofsI, Qba);
    Free2D(Q->nDofsI, Qbb);
    delete[] UtQflat;
    delete Q;
    delete U;
    delete V;

    VecDestroy(&Fl);
}

Krhs::~Krhs() {
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
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, hArray, &hi);
                det = geom->det[ei][ii];
                Qaa[ii][ii] = hi*Q->A[ii][ii]/det/det;
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

// piecewise constant 2 form mass matrix
W0mat::W0mat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 1, PETSC_NULL, 1, PETSC_NULL);
    MatSetOptionsPrefix(M, "W0_");

    assemble();
}

void W0mat::assemble() {
    int ex, ey, ei, ii, jj, *inds;
    double** W = Alloc2D(4, 1);
    double** Wt = Alloc2D(1, 4);
    double** Qaa = Alloc2D(4, 4);
    double** WtQ = Alloc2D(1, 4);
    double** WtQW = Alloc2D(1, 1);
    double* WtQWflat = new double[1*1];
    int nx = e->l->q->n;
    int nxp1 = nx+1;
    int faceInds[4];

    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            inds = topo->elInds2_g(ex, ey);

            for(ii = 0; ii < nx*nx; ii++) {
                faceInds[0] = (ii/nx)*nxp1 + ii%nx;
                faceInds[1] = faceInds[0] + 1;
                faceInds[2] = faceInds[0] + nxp1;
                faceInds[3] = faceInds[2] + 1;

                for(jj = 0; jj < 4; jj++) {
                    W[jj][0] = 0.25;
                    Qaa[jj][jj] = geom->det[ei][faceInds[jj]];
                }

                Tran_IP(4, 1, W, Wt);
                Mult_IP(1, 4, 4, Wt, Qaa, WtQ);
                Mult_IP(1, 1, 4, WtQ, W, WtQW);
                Flat2D_IP(1, 1, WtQW, WtQWflat);
                MatSetValues(M, 1, &inds[ii], 1, &inds[ii], WtQWflat, ADD_VALUES);
            }
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(4, W);
    Free2D(1, Wt);
    Free2D(4, Qaa);
    Free2D(1, WtQ);
    Free2D(1, WtQW);
    delete[] WtQWflat;
}

W0mat::~W0mat() {
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
    MatSetOptionsPrefix(M, "U0_");
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

                    Qaa[jj][jj] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])/det/det;
                    Qab[jj][jj] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])/det/det;
                    Qbb[jj][jj] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])/det/det;

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

WU0mat::WU0mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    assemble();
}

void WU0mat::assemble() {
    int ex, ey, ei, ii, jj, nx, nxp1;
    int *inds_x, *inds_y, *inds_2;
    double det, **J;
    double** W = Alloc2D(4, 1);
    double** Wt = Alloc2D(1, 4);
    double** U = Alloc2D(4, 2);
    double** V = Alloc2D(4, 2);
    double** WtQa = Alloc2D(2, 4);
    double** WtQb = Alloc2D(2, 4);
    double** WtQU = Alloc2D(2, 4);
    double** WtQV = Alloc2D(2, 4);
    double** Qa = Alloc2D(4, 4);
    double** Qb = Alloc2D(4, 4);
    double* WtQUflat = new double[1*2];
    int vertInds[4], edgeIndsX[2], edgeIndsY[2];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n1l, topo->nDofs2G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 32, PETSC_NULL, 32, PETSC_NULL);
    MatSetOptionsPrefix(M, "WU0_");
    MatZeroEntries(M);

    nx = l->q->n;
    nxp1 = nx+1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_2 = topo->elInds2_g(ex, ey);

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

                    Qa[jj][jj] = (J[0][0] + J[1][0])/det/det;
                    Qb[jj][jj] = (J[0][1] + J[1][1])/det/det;

                    W[jj][0] = 0.25;

                    U[jj][0] = (jj%2==0) ? 0.5 : 0.0; // left
                    U[jj][1] = (jj%2==1) ? 0.5 : 0.0; // right
                    V[jj][0] = (jj/2==0) ? 0.5 : 0.0; // bottom
                    V[jj][1] = (jj/2==1) ? 0.5 : 0.0; // top
                }
 
                Tran_IP(4, 1, W, Wt);

                Mult_IP(1, 4, 4, Wt, Qa, WtQa);
                Mult_IP(1, 4, 4, Wt, Qb, WtQb);

                Mult_IP(1, 2, 4, WtQa, U, WtQU);
                Mult_IP(1, 2, 4, WtQb, V, WtQV);

                Flat2D_IP(1, 2, WtQU, WtQUflat);
                MatSetValues(M, 1, &inds_2[ii], 2, edgeIndsX, WtQUflat, ADD_VALUES);

                Flat2D_IP(1, 2, WtQV, WtQUflat);
                MatSetValues(M, 1, &inds_2[ii], 2, edgeIndsY, WtQUflat, ADD_VALUES);
            }
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(4, W);
    Free2D(1, Wt);
    Free2D(4, U);
    Free2D(4, V);
    Free2D(2, WtQa);
    Free2D(2, WtQb);
    Free2D(2, WtQU);
    Free2D(2, WtQV);
    Free2D(4, Qa);
    Free2D(4, Qb);
    delete[] WtQUflat;
}

WU0mat::~WU0mat() {
    MatDestroy(&M);
}

// piecewise constant 2 form mass matrix with height field
W0hmat::W0hmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 1, PETSC_NULL, 1, PETSC_NULL);
}

void W0hmat::assemble(Vec h) {
    int ex, ey, ei, ii, jj, *inds;
    double** W = Alloc2D(4, 1);
    double** Wt = Alloc2D(1, 4);
    double** Qaa = Alloc2D(4, 4);
    double** WtQ = Alloc2D(1, 4);
    double** WtQW = Alloc2D(1, 1);
    double* WtQWflat = new double[1*1];
    int nx = e->l->q->n;
    int nxp1 = nx+1;
    int faceInds[4];
    double hi;
    PetscScalar* hArray;

    VecGetArray(h, &hArray);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            inds = topo->elInds2_g(ex, ey);

            for(ii = 0; ii < nx*nx; ii++) {
                faceInds[0] = (ii/nx)*nxp1 + ii%nx;
                faceInds[1] = faceInds[0] + 1;
                faceInds[2] = faceInds[0] + nxp1;
                faceInds[3] = faceInds[2] + 1;

                for(jj = 0; jj < 4; jj++) {
                    W[jj][0] = 0.25;
                    Qaa[jj][jj] = geom->det[ei][faceInds[jj]];

                    geom->interp2_g(ex, ey, faceInds[jj]%nxp1, faceInds[jj]/nxp1, hArray, &hi);
                    Qaa[jj][jj] *= hi;
                }

                Tran_IP(4, 1, W, Wt);
                Mult_IP(1, 4, 4, Wt, Qaa, WtQ);
                Mult_IP(1, 1, 4, WtQ, W, WtQW);
                Flat2D_IP(1, 1, WtQW, WtQWflat);
                MatSetValues(M, 1, &inds[ii], 1, &inds[ii], WtQWflat, ADD_VALUES);
            }
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    VecRestoreArray(h, &hArray);

    Free2D(4, W);
    Free2D(1, Wt);
    Free2D(4, Qaa);
    Free2D(1, WtQ);
    Free2D(1, WtQW);
    delete[] WtQWflat;
}

W0hmat::~W0hmat() {
    MatDestroy(&M);
}

// boudary integral of the pressure projected onto 1 forms
UtQh_vec::UtQh_vec(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ug);
}

void UtQh_vec::assemble(Vec h2) {
    int ex, ey, ei, ii, jj, mp1, mp12;
    int *inds_x, *inds_y;
    double **J, *u1Array, *h2Array, hi, det = 0.0, avg;
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double** Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    double** Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    double** UtQ = Alloc2D(U->nDofsJ, U->nDofsI);
    double** VtQ = Alloc2D(U->nDofsJ, U->nDofsI);
    double** Qa = Alloc2D(U->nDofsI, U->nDofsI);
    double** Qb = Alloc2D(U->nDofsI, U->nDofsI);
    double hq[99];

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    VecZeroEntries(hl);
    VecScatterBegin(topo->gtol_2, h2, hl, ADD_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h2, hl, ADD_VALUES, SCATTER_FORWARD);

    VecZeroEntries(ul);
    VecZeroEntries(ug);
    VecGetArray(ul, &u1Array);
    VecGetArray(hl, &h2Array);

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) { hq[ii] = Qa[ii][ii] = Qb[ii][ii] = 0.0; }
            for(ii = 0; ii < mp1; ii++) {
                avg = (ii == 0 || ii == mp1-1) ? 0.25 : 0.50;
                // bottom
                jj = ii;
                geom->interp2_g(ex, ey, jj%mp1, jj/mp1, h2Array, &hi);
                hq[jj] += +1.0 * avg * l->q->w[ii] * hi;
                // top
                jj = ii + (mp1-1)*mp1;
                geom->interp2_g(ex, ey, jj%mp1, jj/mp1, h2Array, &hi);
                hq[jj] += -1.0 * avg * l->q->w[ii] * hi;
            }

            for(ii = 0; ii < mp12; ii++) {
                J = geom->J[ei][ii];
                det = geom->det[ei][ii];
                Qa[ii][ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])/det;
                Qb[ii][ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])/det;
            }
            Mult_IP(U->nDofsJ, U->nDofsI, U->nDofsI, Ut, Qa, UtQ);
            Mult_IP(U->nDofsJ, U->nDofsI, U->nDofsI, Vt, Qb, VtQ);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                for(jj = 0; jj < U->nDofsI; jj++) {
                    u1Array[inds_x[ii]] += UtQ[ii][jj]*hq[jj];
                    u1Array[inds_y[ii]] += VtQ[ii][jj]*hq[jj];
                }
            }

            for(ii = 0; ii < mp12; ii++) { hq[ii] = Qa[ii][ii] = Qb[ii][ii] = 0.0; }
            for(ii = 0; ii < mp1; ii++) {
                avg = (ii == 0 || ii == mp1-1) ? 0.25 : 0.50;
                // left
                jj = ii*mp1;
                geom->interp2_g(ex, ey, jj%mp1, jj/mp1, h2Array, &hi);
                hq[jj] += +1.0 * avg * l->q->w[ii] * hi;
                // right
                jj = ii*mp1 + mp1 - 1;
                geom->interp2_g(ex, ey, jj%mp1, jj/mp1, h2Array, &hi);
                hq[jj] += -1.0 * avg * l->q->w[ii] * hi;
            }

            for(ii = 0; ii < mp12; ii++) {
                J = geom->J[ei][ii];
                det = geom->det[ei][ii];
                Qa[ii][ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])/det;
                Qb[ii][ii] = (J[0][1]*J[0][0] + J[1][1]*J[1][0])/det;
            }
            Mult_IP(U->nDofsJ, U->nDofsI, U->nDofsI, Ut, Qa, UtQ);
            Mult_IP(U->nDofsJ, U->nDofsI, U->nDofsI, Vt, Qb, VtQ);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                for(jj = 0; jj < U->nDofsI; jj++) {
                    u1Array[inds_x[ii]] += UtQ[ii][jj]*hq[jj];
                    u1Array[inds_y[ii]] += VtQ[ii][jj]*hq[jj];
                }
            }
        }
    }
    VecRestoreArray(ul, &u1Array);
    VecRestoreArray(hl, &h2Array);

    VecScatterBegin(topo->gtol_1, ul, ug, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, ul, ug, ADD_VALUES, SCATTER_REVERSE);

    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQ);
    Free2D(U->nDofsJ, VtQ);
    Free2D(U->nDofsI, Qa);
    Free2D(U->nDofsI, Qb);
    delete U;
    delete V;
}

UtQh_vec::~UtQh_vec() {
    VecDestroy(&ul);
    VecDestroy(&hl);
    VecDestroy(&ug);
}

// 
WtUmat::WtUmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
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
    MatSetOptionsPrefix(M, "WU2_");

    assemble();
}

void WtUmat::assemble() {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_2;
    double det, **J;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii][ii] = (J[0][0] + J[1][0])*Q->A[ii][ii]/det/det;
                Qab[ii][ii] = (J[0][1] + J[1][1])*Q->A[ii][ii]/det/det;
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

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

WtUmat::~WtUmat() {
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

// 1 form mass matrix with upwinded quadrature
U_up_mat::U_up_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
    MatSetOptionsPrefix(M, "M1_");

    delete U;
}

void U_up_mat::assemble(Vec ui) {
    int ex, ey, ei, ii, jj[2], kk, mm, mp1, mp12;
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
    PetscScalar* uArray;

    MatZeroEntries(M);

    mm   = l->n;
    mp1  = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(ui, &uArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);
            ei = ey*topo->nElsX + ex;

            // initialise global quadrature weights
            for(ii = 0; ii < mp12; ii++) { Qaa[ii][ii] = Qab[ii][ii] = Qbb[ii][ii] = 0.0; }

            // internal quadrature points
            for(ii = 0; ii < mp12; ii++) {
                // do the boundary points after
                if(ii/mp1 == 0 || ii/mp1 == mm || ii%mp1 == 0 || ii%mp1 == mm) continue;

                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii][ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]/det/det;
                Qab[ii][ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]/det/det;
                Qbb[ii][ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]/det/det;
            }

            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);

            // boundary quadrature weights - upwind
            for(ii = 0; ii < mm; ii++) {
                // bottom
                jj[0] = ii;
                jj[1] = jj[0] + 1;
                if(uArray[inds_y[ii]] < 0.0) {
                //if(uArray[inds_y[ii]] > 0.0) {
                    for(kk = 0; kk < 2; kk++) {
                        J   = geom->J[ei][jj[kk]];
                        det = geom->det[ei][jj[kk]];
                        Qaa[jj[kk]][jj[kk]] += (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qab[jj[kk]][jj[kk]] += (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qbb[jj[kk]][jj[kk]] += (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                    }
                }

                // top
                jj[0] = ii + mp1*mm;
                jj[1] = jj[0] + 1;
                if(uArray[inds_y[ii + mm*mm]] > 0.0) {
                //if(uArray[inds_y[ii + mm*mm]] < 0.0) {
                    for(kk = 0; kk < 2; kk++) {
                        J   = geom->J[ei][jj[kk]];
                        det = geom->det[ei][jj[kk]];
                        Qaa[jj[kk]][jj[kk]] += (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qab[jj[kk]][jj[kk]] += (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qbb[jj[kk]][jj[kk]] += (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                    }
                }

                // left
                jj[0] = ii*mp1;
                jj[1] = jj[0] + mp1;
                if(uArray[inds_x[ii*mp1]] < 0.0) {
                //if(uArray[inds_x[ii*mp1]] > 0.0) {
                    for(kk = 0; kk < 2; kk++) {
                        J   = geom->J[ei][jj[kk]];
                        det = geom->det[ei][jj[kk]];
                        Qaa[jj[kk]][jj[kk]] += (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qab[jj[kk]][jj[kk]] += (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qbb[jj[kk]][jj[kk]] += (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                    }
                }

                // right
                jj[0] = ii*mp1 + mm;
                jj[1] = jj[0] + mp1;
                if(uArray[inds_x[ii*mp1+mm]] > 0.0) {
                //if(uArray[inds_x[ii*mp1+mm]] < 0.0) {
                    for(kk = 0; kk < 2; kk++) {
                        J   = geom->J[ei][jj[kk]];
                        det = geom->det[ei][jj[kk]];
                        Qaa[jj[kk]][jj[kk]] += (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qab[jj[kk]][jj[kk]] += (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qbb[jj[kk]][jj[kk]] += (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                    }
                }
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

    VecRestoreArray(ui, &uArray);

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

U_up_mat::~U_up_mat() {
    MatDestroy(&M);
}

Uh_up_mat::Uh_up_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
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

void Uh_up_mat::assemble(Vec h2, Vec u1) {
    int ex, ey, ei, mm, mp1, mp12, ii, jj[2], kk;
    int *inds_x, *inds_y;
    double hi, det, **J;
    PetscScalar* h2Array;
    PetscScalar* u1Array;

    mm   = l->q->n;
    mp1  = l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    VecGetArray(h2, &h2Array);
    VecGetArray(u1, &u1Array);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);
            ei = ey*topo->nElsX + ex;

            // initialise global quadrature weights
            for(ii = 0; ii < mp12; ii++) { Qaa[ii][ii] = Qab[ii][ii] = Qbb[ii][ii] = 0.0; }

            for(ii = 0; ii < mp12; ii++) {
                // do the boundary points after
                if(ii/mp1 == 0 || ii/mp1 == mm || ii%mp1 == 0 || ii%mp1 == mm) continue;

                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, h2Array, &hi);

                Qaa[ii][ii] = hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]/det/det;
                Qab[ii][ii] = hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]/det/det;
                Qbb[ii][ii] = hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]/det/det;
            }

            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);

            // boundary quadrature weights - upwind
            for(ii = 0; ii < mm; ii++) {
                // bottom
                jj[0] = ii;
                jj[1] = jj[0] + 1;
                if(u1Array[inds_y[ii]] < 0.0) {
                //if(u1Array[inds_y[ii]] > 0.0) {
                    for(kk = 0; kk < 2; kk++) {
                        J   = geom->J[ei][jj[kk]];
                        det = geom->det[ei][jj[kk]];
                        geom->interp2_g(ex, ey, jj[kk]%mp1, jj[kk]/mp1, h2Array, &hi);
                        Qaa[jj[kk]][jj[kk]] += hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qab[jj[kk]][jj[kk]] += hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qbb[jj[kk]][jj[kk]] += hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                    }
                }

                // top
                jj[0] = ii + mp1*mm;
                jj[1] = jj[0] + 1;
                if(u1Array[inds_y[ii + mm*mm]] > 0.0) {
                //if(u1Array[inds_y[ii + mm*mm]] < 0.0) {
                    for(kk = 0; kk < 2; kk++) {
                        J   = geom->J[ei][jj[kk]];
                        det = geom->det[ei][jj[kk]];
                        geom->interp2_g(ex, ey, jj[kk]%mp1, jj[kk]/mp1, h2Array, &hi);
                        Qaa[jj[kk]][jj[kk]] += hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qab[jj[kk]][jj[kk]] += hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qbb[jj[kk]][jj[kk]] += hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                    }
                }

                // left
                jj[0] = ii*mp1;
                jj[1] = jj[0] + mp1;
                if(u1Array[inds_x[ii*mp1]] < 0.0) {
                //if(u1Array[inds_x[ii*mp1]] > 0.0) {
                    for(kk = 0; kk < 2; kk++) {
                        J   = geom->J[ei][jj[kk]];
                        det = geom->det[ei][jj[kk]];
                        geom->interp2_g(ex, ey, jj[kk]%mp1, jj[kk]/mp1, h2Array, &hi);
                        Qaa[jj[kk]][jj[kk]] += hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qab[jj[kk]][jj[kk]] += hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qbb[jj[kk]][jj[kk]] += hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                    }
                }

                // right
                jj[0] = ii*mp1 + mm;
                jj[1] = jj[0] + mp1;
                if(u1Array[inds_x[ii*mp1+mm]] > 0.0) {
                //if(u1Array[inds_x[ii*mp1+mm]] < 0.0) {
                    for(kk = 0; kk < 2; kk++) {
                        J   = geom->J[ei][jj[kk]];
                        det = geom->det[ei][jj[kk]];
                        geom->interp2_g(ex, ey, jj[kk]%mp1, jj[kk]/mp1, h2Array, &hi);
                        Qaa[jj[kk]][jj[kk]] += hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qab[jj[kk]][jj[kk]] += hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                        Qbb[jj[kk]][jj[kk]] += hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[jj[kk]][jj[kk]]/det/det;
                    }
                }
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
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

Uh_up_mat::~Uh_up_mat() {
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
