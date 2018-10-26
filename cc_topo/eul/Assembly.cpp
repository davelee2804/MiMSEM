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

    delete U;
}

void Umat::assemble(int lev, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
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

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds_0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii][ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]*(scale/det/det);
                Qab[ii][ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
                Qbb[ii][ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]*(scale/det/det);

                // horiztonal velocity is piecewise constant in the vertical
                Qaa[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
                Qab[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
                Qbb[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qaa, UtQaa);
            Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qab, UtQab);
            Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qab, VtQba);
            Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qbb, VtQbb);

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
}

void Wmat::assemble(int lev, double scale, bool vert_scale) {
    int ex, ey, ei, mp1, mp12, ii, *inds, *inds0;
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

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii][ii]  = Q->A[ii][ii]*(scale/det/det);
                if(vert_scale) {
                    Qaa[ii][ii] *= 1.0/geom->thick[lev][inds0[ii]];
                }
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
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

void Uhmat::assemble(Vec h2, int lev, bool const_vert, double scale) {
    int ex, ey, ei, mp1, mp12, ii;
    int *inds_x, *inds_y, *inds_0;
    double hi, det, **J;
    PetscScalar *h2Array;

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    VecGetArray(h2, &h2Array);

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds_0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, h2Array, &hi);

                // density field is piecewise constant in the vertical
                if(const_vert) {
                    hi *= 1.0/geom->thick[lev][inds_0[ii]];
                }

                Qaa[ii][ii] = hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]*(scale/det/det);
                Qab[ii][ii] = hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
                Qbb[ii][ii] = hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]*(scale/det/det);

                // horiztonal velocity is piecewise constant in the vertical
                Qaa[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
                Qab[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
                Qbb[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
            }

            // reuse the JU and JV matrices for the nonlinear trial function expansion matrices
            Mult_FD_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Ut, Qaa, UtQaa);
            Mult_FD_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Ut, Qab, UtQab);
            Mult_FD_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Vt, Qab, VtQba);
            Mult_FD_IP(U->nDofsJ, U->nDofsI, Q->nDofsJ, Vt, Qbb, VtQbb);

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

    entries = new PetscScalar[(l->n+1)*(l->n+1)];

    Q = new Wii(l->q, geom);
}

void Pvec::assemble(int lev, double scale) {
    int ii, ex, ey, np1, np12, *inds_l;

    VecZeroEntries(vl);
    VecZeroEntries(vg);

    np1 = l->n + 1;
    np12 = np1*np1;

    // assemble values into local vector
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            inds_l = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < np12; ii++) {
                entries[ii]  = scale*Q->A[ii][ii];
                entries[ii] *= 1.0/geom->thick[lev][inds_l[ii]];
            }
            VecSetValues(vl, np12, inds_l, entries, ADD_VALUES);
        }
    }

    // scatter values to global vector
    VecScatterBegin(topo->gtol_0, vl, vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, vl, vg, ADD_VALUES, SCATTER_REVERSE);

    // and back to local vector
    VecScatterBegin(topo->gtol_0, vg, vl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, vg, vl, INSERT_VALUES, SCATTER_FORWARD);
}

Pvec::~Pvec() {
    delete[] entries;
    VecDestroy(&vl);
    VecDestroy(&vg);
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

            // piecewise constant field in the vertical, so vertical transformation is det/det = 1
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
            // piecewise constant field in the vertical, so vertical transformation is det/det = 1
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

            // piecewise constant field in the vertical, so vertical transformation is det/det = 1
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
}

void WtQUmat::assemble(Vec u1, int lev, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_2, *inds_0;
    double det, **J, ux[2];
    PetscScalar *u1Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds_0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);
                // horiztontal velocity is piecewise constant in the vertical
                ux[0] *= 1.0/geom->thick[lev][inds_0[ii]];
                ux[1] *= 1.0/geom->thick[lev][inds_0[ii]];

                Qaa[ii][ii] = 0.5*(ux[0]*J[0][0] + ux[1]*J[1][0])*Q->A[ii][ii]*(scale/det/det);
                Qab[ii][ii] = 0.5*(ux[0]*J[0][1] + ux[1]*J[1][1])*Q->A[ii][ii]*(scale/det/det);

                // rescale by the inverse of the vertical determinant (piecewise 
                // constant in the vertical)
                Qaa[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
                Qab[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qaa, WtQaa);
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qab, WtQab);

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
}

void RotMat::assemble(Vec q0, int lev, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    double det, **J, vort;
    PetscScalar* q0Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(q0, &q0Array);
    MatZeroEntries(M);

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, V->nDofsJ, V->A, Vt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds_0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp0(ex, ey, ii%mp1, ii/mp1, q0Array, &vort);

                // vertical vorticity is piecewise constant in the vertical
                vort *= 1.0/geom->thick[lev][inds_0[ii]];

                Qab[ii][ii] = vort*(-J[0][0]*J[1][1] + J[0][1]*J[1][0])*Q->A[ii][ii]*(scale/det/det);
                Qba[ii][ii] = vort*(+J[0][0]*J[1][1] - J[0][1]*J[1][0])*Q->A[ii][ii]*(scale/det/det);

                Qab[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
                Qba[ii][ii] *= 1.0/geom->thick[lev][inds_0[ii]];
            }

            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Qab, UtQab);
            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Qba, VtQba);

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

//
Whmat::Whmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    M2_j_xy_i* W = new M2_j_xy_i(e);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);

    delete W;
}

void Whmat::assemble(Vec rho, int lev, double scale) {
    int ex, ey, ei, mp1, mp12, ii, *inds, *inds0;
    double det, p;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar* pArray;

    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecGetArray(rho, &pArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];

                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, pArray, &p);
                // density is piecewise constant in the vertical
                p *= 1.0/geom->thick[lev][inds0[ii]];

                Qaa[ii][ii]  = p*Q->A[ii][ii]*(scale/det/det);
                // W is piecewise constant in the vertical
                Qaa[ii][ii] *= 1.0/geom->thick[lev][inds0[ii]];
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQWflat, ADD_VALUES);
        }
    }
    VecRestoreArray(rho, &pArray);

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

// these matrices are for the horizontal vorticity components
Ut_mat::Ut_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);

    delete U;
}

void Ut_mat::assemble(int lev, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
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

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds_0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii][ii] = (+J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
                Qab[ii][ii] = (-J[0][0]*J[0][1] - J[1][0]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
                Qbb[ii][ii] = (+J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]*(scale/det/det);

                // horiztonal velocity is piecewise constant in the vertical
                Qaa[ii][ii] *= 0.5*(geom->thick[lev][inds_0[ii]] + geom->thick[lev+1][inds_0[ii]]);
                Qab[ii][ii] *= 0.5*(geom->thick[lev][inds_0[ii]] + geom->thick[lev+1][inds_0[ii]]);
                Qbb[ii][ii] *= 0.5*(geom->thick[lev][inds_0[ii]] + geom->thick[lev+1][inds_0[ii]]);
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qaa, UtQaa);
            Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qab, UtQab);
            Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qab, VtQba);
            Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qbb, VtQbb);

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

Ut_mat::~Ut_mat() {
    MatDestroy(&M);
}

//
UtQWmat::UtQWmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);
    W = new M2_j_xy_i(e);
    Q = new Wii(l->q, geom);
    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(V->nDofsJ, V->nDofsI);
    Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Qba = Alloc2D(Q->nDofsI, Q->nDofsJ);
    UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(V->nDofsJ, Q->nDofsJ);
    UtQW = Alloc2D(U->nDofsJ, W->nDofsJ);
    VtQW = Alloc2D(V->nDofsJ, W->nDofsJ);
    UtQWflat = new double[U->nDofsJ*W->nDofsJ];
    VtQWflat = new double[V->nDofsJ*W->nDofsJ];

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(V->nDofsI, V->nDofsJ, V->A, Vt);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n2l, topo->nDofs1G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
}

void UtQWmat::assemble(Vec u1, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_2, *inds_0;
    double det, **J, ux[2];
    PetscScalar *u1Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds_0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                // components are [-dv/dz, +du/dz]
                geom->interp1_g_t(ex, ey, ii%mp1, ii/mp1, u1Array, ux);

                // horizontal velocity is piecewise constant, and vertical velocity is 
                // piecewise linear, so vertical transformations cancel

                // 0.5 scaling done outside
                //Qaa[ii][ii] = (ux[1]*J[0][0] - ux[0]*J[1][0])*Q->A[ii][ii]*(scale/det/det);
                //Qba[ii][ii] = (ux[1]*J[0][1] - ux[0]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
                Qaa[ii][ii] = (ux[1]*J[0][0] - ux[0]*J[0][1])*Q->A[ii][ii]*(scale/det/det);
                Qba[ii][ii] = (ux[1]*J[1][0] - ux[0]*J[1][1])*Q->A[ii][ii]*(scale/det/det);
            }

            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Qaa, UtQaa);
            Mult_FD_IP(V->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Qba, VtQba);

            Mult_IP(U->nDofsJ, W->nDofsJ, W->nDofsI, UtQaa, U->A, UtQW);
            Mult_IP(V->nDofsJ, W->nDofsJ, W->nDofsI, VtQba, V->A, VtQW);

            Flat2D_IP(U->nDofsJ, W->nDofsJ, UtQW, UtQWflat);
            Flat2D_IP(V->nDofsJ, W->nDofsJ, VtQW, VtQWflat);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_2 = topo->elInds2_g(ex, ey);

            MatSetValues(M, U->nDofsJ, inds_x, W->nDofsJ, inds_2, UtQWflat, ADD_VALUES);
            MatSetValues(M, V->nDofsJ, inds_y, W->nDofsJ, inds_2, VtQWflat, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

UtQWmat::~UtQWmat() {
    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    Free2D(Q->nDofsI, Qaa);
    Free2D(Q->nDofsI, Qba);
    Free2D(U->nDofsJ, UtQW);
    Free2D(V->nDofsJ, VtQW);
    Free2D(U->nDofsJ, UtQaa);
    Free2D(V->nDofsJ, VtQba);
    delete[] UtQWflat;
    delete[] VtQWflat;
    delete U;
    delete V;
    delete W;
    delete Q;
    MatDestroy(&M);
}

//
WtQdUdz_mat::WtQdUdz_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
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

void WtQdUdz_mat::assemble(Vec u1, int lev, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_2, *inds_0;
    double det, **J, ux[2];
    PetscScalar *u1Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds_0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);
                // vertical scalings cancel between piecewise constant velocity and jacobian

                // vorticity = [a,b,c], velocity = [u,v,w]
                // +J^{-T}.v.a
                //Qaa[ii][ii] = +1.0*(+ux[1]*J[1][1] - ux[1]*J[1][0])*Q->A[ii][ii]*(scale/det/det);
                Qaa[ii][ii] = (+ux[1]*J[1][1] + ux[0]*J[1][0])*Q->A[ii][ii]*(scale/det/det);
                // -J^{-T}.u.b
                //Qab[ii][ii] = -1.0*(-ux[0]*J[0][1] + ux[0]*J[0][0])*Q->A[ii][ii]*(scale/det/det);
                Qab[ii][ii] = (-ux[1]*J[0][1] - ux[0]*J[0][0])*Q->A[ii][ii]*(scale/det/det);
                // vertical rescaling of jacobian determinant cancels with scaling of
                // the H(curl) test function
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qaa, WtQaa);
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qab, WtQab);

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

WtQdUdz_mat::~WtQdUdz_mat() {
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
