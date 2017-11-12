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

#define VIEW_MAT

// mass matrix for the 1 form vector (x-normal degrees of
// freedom first then y-normal degrees of freedom)
Umat::Umat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    assemble();
}

void Umat::assemble() {
    int ex, ey;
    int *inds_x, *inds_y;
    Wii* Q = new Wii(l->q, geom);
    JacM1* J = new JacM1(l->q, geom);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double** JxU = Alloc2D(J->nDofsI, U->nDofsJ);
    double** JxV = Alloc2D(J->nDofsI, U->nDofsJ);
    double** JyU = Alloc2D(J->nDofsI, U->nDofsJ);
    double** JyV = Alloc2D(J->nDofsI, U->nDofsJ);
    double** JxUt = Alloc2D(U->nDofsJ, J->nDofsI);
    double** JyVt = Alloc2D(U->nDofsJ, J->nDofsI);
    double** JxUtQ = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** JyVtQ = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** VtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** VtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* UtQUflat = new double[U->nDofsJ*U->nDofsJ];
#ifdef VIEW_MAT
    PetscViewer viewer;
#endif

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);
            J->assemble(ex, ey);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aaa, U->A, JxU);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aab, V->A, JxV);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aba, U->A, JyU);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Abb, V->A, JyV);

            Tran_IP(J->nDofsI, U->nDofsJ, JxU, JxUt);
            Tran_IP(J->nDofsI, U->nDofsJ, JyV, JyVt);

            Mult_IP(U->nDofsJ, J->nDofsI, J->nDofsJ, JxUt, Q->A, JxUtQ);
            Mult_IP(U->nDofsJ, J->nDofsI, J->nDofsJ, JyVt, Q->A, JyVtQ);

            Mult_IP(U->nDofsJ, U->nDofsJ, J->nDofsJ, JxUtQ, JxU, UtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, J->nDofsJ, JxUtQ, JxV, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, J->nDofsJ, JyVtQ, JyU, VtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, J->nDofsJ, JyVtQ, JyV, VtQV);

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

#ifdef VIEW_MAT
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "mat.dat", &viewer);
    MatView(M, viewer);
    PetscViewerDestroy(&viewer);
    MatView(M, PETSC_VIEWER_DRAW_WORLD);
#endif

    Free2D(J->nDofsI, JxU);
    Free2D(J->nDofsI, JxV);
    Free2D(J->nDofsI, JyU);
    Free2D(J->nDofsI, JyV);
    Free2D(U->nDofsJ, JxUt);
    Free2D(U->nDofsJ, JyVt);
    Free2D(U->nDofsJ, JxUtQ);
    Free2D(U->nDofsJ, JyVtQ);
    Free2D(U->nDofsJ, UtQU);
    Free2D(U->nDofsJ, UtQV);
    Free2D(U->nDofsJ, VtQU);
    Free2D(U->nDofsJ, VtQV);
    delete[] UtQUflat;
    delete Q;
    delete U;
    delete V;
    delete J;
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
    int ex, ey;
    int* inds;

    Wii* Q = new Wii(e->l->q, geom);
    JacM2* J = new JacM2(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double** JW = Alloc2D(J->nDofsI, W->nDofsJ);
    double** JWt = Alloc2D(W->nDofsJ, J->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);
            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);
            J->assemble(ex, ey);

            Mult_IP(J->nDofsI, W->nDofsJ, J->nDofsJ, J->A, W->A, JW);
            Tran_IP(J->nDofsI, W->nDofsJ, JW, JWt);

            Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, JWt, Q->A, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, JW, WtQW);

            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQWflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(J->nDofsI, JW);
    Free2D(W->nDofsJ, JWt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete W;
    delete Q;
    delete J;
    delete[] WtQWflat;
}

Wmat::~Wmat() {
    MatDestroy(&M);
}

// 0 form mass matrix
Pmat::Pmat(Topo* _topo, Geom* _geom, LagrangeNode* _l) {
    topo = _topo;
    geom = _geom;
    l = _l;

    assemble();
}

void Pmat::assemble() {
    int ex, ey;
    int* inds;

    Wii* Q = new Wii(l->q, geom);
    M0_j_xy_i* P = new M0_j_xy_i(l);
    double** Pt = Tran(P->nDofsI, P->nDofsJ, P->A);
    double** PtQ = Mult(P->nDofsJ, Q->nDofsJ, P->nDofsI, Pt, Q->A);
    double** PtQP = Mult(P->nDofsJ, P->nDofsJ, Q->nDofsJ, PtQ, P->A);
    double* PtQPflat = Flat2D(P->nDofsJ, P->nDofsJ, PtQP);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 2*P->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds0_g(ex, ey);
            // TODO: incorporate the jacobian transformation for each element
            MatSetValues(M, P->nDofsJ, inds, P->nDofsJ, inds, PtQPflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    Free2D(P->nDofsJ, PtQP);
    delete P;
    delete[] PtQPflat;
}

Pmat::~Pmat() {
    MatDestroy(&M);
}

// 1 form mass matrix with 2 forms interpolated to quadrature points
Uhmat::Uhmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    Q = new Wii(l->q, geom);
    J = new JacM1(l->q, geom);
    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);
    Uh = new M1x_j_Fxy_i(l, e, geom);
    Vh = new M1y_j_Fxy_i(l, e, geom);

    JxU = Alloc2D(J->nDofsI, U->nDofsJ);
    JxV = Alloc2D(J->nDofsI, U->nDofsJ);
    JyU = Alloc2D(J->nDofsI, V->nDofsJ);
    JyV = Alloc2D(J->nDofsI, V->nDofsJ);
    JxUt = Alloc2D(U->nDofsJ, J->nDofsI);
    JyVt = Alloc2D(V->nDofsJ, J->nDofsI);
    UtQ = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQ = Alloc2D(V->nDofsJ, Q->nDofsJ);
    UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQU = Alloc2D(V->nDofsJ, V->nDofsJ);
    VtQV = Alloc2D(V->nDofsJ, V->nDofsJ);
    UtQUflat = new double[U->nDofsJ*U->nDofsJ];
    ck = new double[l->n*l->n];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
}

void Uhmat::assemble(Vec h2) {
    int ex, ey, kk, n2;
    int *inds_x, *inds_y, *inds2;
    PetscScalar* h2Array;

    n2 = topo->elOrd*topo->elOrd;

    MatZeroEntries(M);
    VecGetArray(h2, &h2Array);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            // incorporate the jacobian transformation for each element
            Q->assemble(ex, ey);
            J->assemble(ex, ey);

            inds2 = topo->elInds2_l(ex, ey);
            for(kk = 0; kk < n2; kk++) {
                ck[kk] = h2Array[inds2[kk]];
            }
            Uh->assemble(ex, ey, ck);
            Vh->assemble(ex, ey, ck);

            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aaa, U->A, JxU);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Abb, V->A, JyV);

            Tran_IP(J->nDofsI, U->nDofsJ, JxU, JxUt);
            Tran_IP(J->nDofsI, V->nDofsJ, JyV, JyVt);

            Mult_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, JxUt, Q->A, UtQ);
            Mult_IP(V->nDofsJ, Q->nDofsJ, Q->nDofsI, JyVt, Q->A, VtQ);

            // reuse the JU and JV matrices for the nonlinear trial function expansion matrices
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aaa, Uh->A, JxU);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aab, Vh->A, JxV);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aba, Uh->A, JyU);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Abb, Vh->A, JyV);

            Mult_IP(U->nDofsJ, Uh->nDofsJ, Q->nDofsJ, UtQ, JxU, UtQU);
            Mult_IP(U->nDofsJ, Uh->nDofsJ, Q->nDofsJ, UtQ, JxV, UtQV);
            Mult_IP(U->nDofsJ, Uh->nDofsJ, Q->nDofsJ, VtQ, JyU, VtQU);
            Mult_IP(U->nDofsJ, Uh->nDofsJ, Q->nDofsJ, VtQ, JyV, VtQV);

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
    delete[] ck;

    Free2D(J->nDofsI, JxU);
    Free2D(J->nDofsI, JxV);
    Free2D(J->nDofsI, JyU);
    Free2D(J->nDofsI, JyV);
    Free2D(Uh->nDofsJ, JxUt);
    Free2D(Vh->nDofsJ, JyVt);
    Free2D(Uh->nDofsJ, UtQ);
    Free2D(Vh->nDofsJ, VtQ);
    Free2D(Uh->nDofsJ, UtQU);
    Free2D(Uh->nDofsJ, UtQV);
    Free2D(Vh->nDofsJ, VtQU);
    Free2D(Vh->nDofsJ, VtQV);

    delete U;
    delete V;
    delete Uh;
    delete Vh;
    delete Q;
    delete J;

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

    entries = new PetscScalar[(l->n+1)*(l->n+1)];

    Q = new Wii(l->q, geom);

    assemble();
}

void Pvec::assemble() {
    int ii, ex, ey, np1, np12;
    int *inds_x;

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

// Assembly of the diagonal 0 form mass matrix as a vector 
// with 2 form vector interpolated onto quadrature points.
// Assumes inexact integration and a diagonal mass matrix for the 
// 0 form function space (ie: quadrature and basis functions are 
// the same order)
Phvec::Phvec(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vg);
    VecZeroEntries(vg);

    ck = new double[(l->n)*(l->n)];
    entries = new PetscScalar[(l->n+1)*(l->n+1)];
}

void Phvec::assemble(Vec h2) {
    int ii, kk, ex, ey, np1, np12, n2;
    int *inds2, *inds_x;
    double wt, hq;
    PetscScalar* h2Array;
    double* weights = l->q->w;
    double** ejxi = e->ejxi;

    VecGetArray(h2, &h2Array);
    VecZeroEntries(vl);

    n2 = (l->n)*(l->n);
    np1 = l->n + 1;
    np12 = np1*np1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // TODO: incorporate the jacobian transformation for each element
            inds2 = topo->elInds2_l(ex, ey);
            for(kk = 0; kk < n2; kk++) {
                ck[kk] = h2Array[inds2[kk]];
            }

            for(ii = 0; ii < np12; ii++) {
                // weight at quaadrature point
                wt = weights[ii%np1]*weights[ii/np1];

                // interpolate 2 form at quadrature point
                hq = 0.0;
                for(kk = 0; kk < n2; kk++) {
                    hq += ck[kk]*ejxi[ii][kk];
                }

                entries[ii] = wt*hq;
            }

            inds_x = topo->elInds0_l(ex, ey);
            VecSetValues(vl, np12, inds_x, entries, ADD_VALUES);
        }
    }
    VecRestoreArray(h2, &h2Array);

    // scatter values to global vector
    VecScatterBegin(topo->gtol_0, vl, vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, vl, vg, ADD_VALUES, SCATTER_REVERSE);

    // and back to local vector
    VecScatterBegin(topo->gtol_0, vg, vl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, vg, vl, INSERT_VALUES, SCATTER_FORWARD);
}

Phvec::~Phvec() {
    delete[] ck;
    delete[] entries;
    VecDestroy(&vl);
    VecDestroy(&vg);
}

// Assumes quadrature points and 0 forms are the same (for now)
WtQmat::WtQmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    assemble();
}

void WtQmat::assemble() {
    int ex, ey;
    int *inds_2, *inds_0;

    M2_j_xy_i* W = new M2_j_xy_i(e);
    Wii* Q = new Wii(e->l->q, geom);
    JacM2* J = new JacM2(e->l->q, geom);
    double** JW = Alloc2D(J->nDofsI, W->nDofsJ);
    double** JWt = Alloc2D(W->nDofsJ, J->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQflat = new double[W->nDofsJ*Q->nDofsJ];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n0l, topo->nDofs2G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);
            J->assemble(ex, ey);

            Mult_IP(J->nDofsI, W->nDofsJ, W->nDofsJ, J->A, W->A, JW);
            Tran_IP(J->nDofsI, W->nDofsJ, JW, JWt);

            Mult_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, JWt, Q->A, WtQ);
            Flat2D_IP(W->nDofsJ, Q->nDofsJ, WtQ, WtQflat);

            inds_2 = topo->elInds2_g(ex, ey);
            inds_0 = topo->elInds0_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, Q->nDofsJ, inds_0, WtQflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(J->nDofsI, JW);
    Free2D(W->nDofsJ, JWt);
    Free2D(W->nDofsJ, WtQ);
    delete[] WtQflat;
    delete W;
    delete Q;
    delete J;
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
    int ex, ey, ii, mp12;
    int *inds_x, *inds_y, *inds_qx, *inds_qy;
    Wii* Q = new Wii(l->q, geom);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    JacM1* J = new JacM1(l->q, geom);
    double** JU = Alloc2D(J->nDofsI, U->nDofsJ);
    double** JUt = Alloc2D(U->nDofsJ, J->nDofsI);
    double** UtQ = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* UtQflat = new double[U->nDofsJ*Q->nDofsJ];

    mp12 = (l->q->n + 1)*(l->q->n + 1);
    inds_qy = new int[mp12];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, 2*topo->n0l, topo->nDofs1G, 2*topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 4*U->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);
            J->assemble(ex, ey);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_qx = topo->elInds0_g(ex, ey);
            // derive degrees of freedom for y-component of vector at quadrature points
            // by shifting x-components
            // TODO: use vertex major indexing??
            for(ii = 0; ii < mp12; ii++) {
                inds_qy[ii] = inds_qx[ii] + topo->nDofs0G;
            }

            //
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aaa, U->A, JU);
            Tran_IP(J->nDofsI, U->nDofsJ, JU, JUt);
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, JUt, Q->A, UtQ);
            Flat2D_IP(U->nDofsJ, Q->nDofsJ, UtQ, UtQflat);
            MatSetValues(M, U->nDofsJ, inds_x, Q->nDofsJ, inds_qx, UtQflat, ADD_VALUES);
          
            //
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Abb, V->A, JU);
            Tran_IP(J->nDofsI, U->nDofsJ, JU, JUt);
            Mult_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, JUt, Q->A, UtQ);
            Flat2D_IP(U->nDofsJ, Q->nDofsJ, UtQ, UtQflat);
            MatSetValues(M, U->nDofsJ, inds_y, Q->nDofsJ, inds_qy, UtQflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(J->nDofsI, JU);
    Free2D(U->nDofsJ, JUt);
    Free2D(U->nDofsJ, UtQ);
    delete[] UtQflat;
    delete[] inds_qy;
    delete Q;
    delete U;
    delete V;
    delete J;
}

UtQmat::~UtQmat() {
    MatDestroy(&M);
}

// project the potential vorticity gradient velocity product onto the 0 forms
PtQUmat::PtQUmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    double **Pt;
    Wii* Q;

    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    U = new M1x_j_Exy_i(l, e);
    V = new M1y_j_Exy_i(l, e);
    P = new M0_j_xy_i(l);
    Pt = Tran(P->nDofsI, P->nDofsJ, P->A);
    Q = new Wii(l->q, geom);
    PtQ = Mult(P->nDofsJ, Q->nDofsJ, Q->nDofsI, Pt, Q->A);

    PtQUflat = new double[P->nDofsJ*U->nDofsJ];
    PtQVflat = new double[P->nDofsJ*V->nDofsJ];
    ckx = new double[(l->n+1)*(l->n)];
    cky = new double[(l->n)*(l->n+1)];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n1l, topo->nDofs0G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);

    Free2D(P->nDofsJ, Pt);
    delete Q;
}

void PtQUmat::assemble(Vec q1) {
    int ex, ey, ii, jj, kk, n2;
    int *inds_x, *inds_y, *inds_0;
    double **PtQU, **PtQV;
    PetscScalar* q1Array;

    n2 = (l->n+1)*(l->n);

    VecGetArray(q1, &q1Array);
    MatZeroEntries(M);

    // TODO: incorportate jacobian tranformation for each element
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);
            // TODO: check the sign of these pv gradient terms after the E10 
            // incidence matrix has been implemented
            for(kk = 0; kk < n2; kk++) {
                ckx[kk] = -1.0*q1Array[inds_x[kk]];
                cky[kk] = +1.0*q1Array[inds_y[kk]];
            }
            U->assemble(cky);
            V->assemble(ckx);
            // TODO: incorporate the jacobian transformation for each element
            PtQU = Mult(P->nDofsJ, U->nDofsJ, U->nDofsI, PtQ, U->A);
            PtQV = Mult(P->nDofsJ, V->nDofsJ, V->nDofsI, PtQ, V->A);

            kk = 0;
            for(ii = 0; ii < P->nDofsJ; ii++) {
                for(jj = 0; jj < U->nDofsJ; jj++) {
                    PtQUflat[kk] = PtQU[ii][jj];
                    PtQVflat[kk] = PtQV[ii][jj];
                    kk++;
                }
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_0 = topo->elInds0_g(ex, ey);

            MatSetValues(M, P->nDofsJ, inds_0, U->nDofsJ, inds_x, PtQUflat, ADD_VALUES);
            MatSetValues(M, P->nDofsJ, inds_0, V->nDofsJ, inds_y, PtQVflat, ADD_VALUES);

            Free2D(P->nDofsJ, PtQU);
            Free2D(P->nDofsJ, PtQV);
        }
    }
    VecRestoreArray(q1, &q1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

PtQUmat::~PtQUmat() {
    Free2D(P->nDofsJ, PtQ);
    delete[] PtQUflat;
    delete[] PtQVflat;
    delete[] ckx;
    delete[] cky;
    delete U;
    delete V;
    delete P;
    MatDestroy(&M);
}

// 
WtQUmat::WtQUmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    U = new M1x_j_Cxy_i(l, e, geom);
    V = new M1y_j_Cxy_i(l, e, geom);
    W = new M2_j_xy_i(e);
    Q = new Wii(l->q, geom);
    J2 = new JacM2(l->q, geom);
    JW = Alloc2D(J2->nDofsI, W->nDofsJ);
    JWt = Alloc2D(W->nDofsJ, J2->nDofsI);
    WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    WtQU = Alloc2D(W->nDofsJ, U->nDofsJ);
    WtQV = Alloc2D(W->nDofsJ, V->nDofsJ);
    WtQUflat = new double[W->nDofsJ*U->nDofsJ];
    WtQVflat = new double[W->nDofsJ*V->nDofsJ];
    ckx = new double[(l->n+1)*(l->n)];
    cky = new double[(l->n)*(l->n+1)];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n1l, topo->nDofs2G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
}

void WtQUmat::assemble(Vec u1) {
    int ex, ey, kk, n2;
    int *inds_x, *inds_y, *inds_2;
    PetscScalar* u1Array;

    n2 = (l->n+1)*(l->n);

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);

            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);
            J2->assemble(ex, ey);

            // note that we are assembling K = (u.u)/2 onto 2 forms
            for(kk = 0; kk < n2; kk++) {
                ckx[kk] = 0.5*u1Array[inds_x[kk]];
                cky[kk] = 0.5*u1Array[inds_y[kk]];
            }
            //U->assemble(cky);
            //V->assemble(ckx);
            U->assemble(ex, ey, ckx, cky);
            V->assemble(ex, ey, ckx, cky);

            // incorporate the jacobian transformation for basis functions
            Mult_IP(J2->nDofsI, W->nDofsJ, W->nDofsI, J2->A, W->A, JW);
            Tran_IP(J2->nDofsI, W->nDofsJ, JW, JWt);

            Mult_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, JWt, Q->A, WtQ);
            Mult_IP(W->nDofsJ, U->nDofsJ, U->nDofsI, WtQ, U->A, WtQU);
            Mult_IP(W->nDofsJ, V->nDofsJ, V->nDofsI, WtQ, V->A, WtQV);

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
    Free2D(W->nDofsJ, WtQU);
    Free2D(W->nDofsJ, WtQV);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, JWt);
    Free2D(J2->nDofsI, JW);
    delete[] WtQUflat;
    delete[] WtQVflat;
    delete[] ckx;
    delete[] cky;
    delete U;
    delete V;
    delete W;
    delete Q;
    delete J2;
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
    J = new JacM1(l->q, geom);
    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);
    Uq = new M1x_j_Dxy_i(l, e);
    Vq = new M1y_j_Dxy_i(l, e);
    JxU = Alloc2D(J->nDofsI, U->nDofsJ);
    JxV = Alloc2D(J->nDofsI, V->nDofsJ);
    JyU = Alloc2D(J->nDofsI, U->nDofsJ);
    JyV = Alloc2D(J->nDofsI, V->nDofsJ);
    JxUt = Alloc2D(U->nDofsJ, J->nDofsI);
    JyVt = Alloc2D(V->nDofsJ, J->nDofsI);
    UtQ = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQ = Alloc2D(V->nDofsJ, Q->nDofsJ);
    UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQU = Alloc2D(V->nDofsJ, V->nDofsJ);
    VtQV = Alloc2D(V->nDofsJ, V->nDofsJ);

    UtQUflat = new double[U->nDofsJ*V->nDofsJ];
    ckx = new double[(l->n+1)*(l->n+1)];
    cky = new double[(l->n+1)*(l->n+1)];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
}

void RotMat::assemble(Vec q0) {
    int ex, ey, kk, np12;
    int *inds_x, *inds_y, *inds_0;
    PetscScalar* q0Array;

    np12 = (l->n+1)*(l->n+1);

    VecGetArray(q0, &q0Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            // incorportate jacobian tranformation for each element
            Q->assemble(ex, ey);
            J->assemble(ex, ey);

            inds_0 = topo->elInds0_l(ex, ey);
            for(kk = 0; kk < np12; kk++) {
                ckx[kk] = +1.0*q0Array[inds_0[kk]];
                cky[kk] = -1.0*q0Array[inds_0[kk]];
            }
            Uq->assemble(cky);
            Vq->assemble(ckx);

            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aaa, U->A, JxU);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Abb, V->A, JyV);

            Tran_IP(J->nDofsI, U->nDofsJ, JxU, JxUt);
            Tran_IP(J->nDofsI, V->nDofsJ, JyV, JyVt);

            Mult_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, JxUt, Q->A, UtQ);
            Mult_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, JyVt, Q->A, VtQ);

            // reuse the JU and JV matrices for the nonlinear trial function expansion matrices
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aaa, Uq->A, JxU);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aab, Vq->A, JxV);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Aba, Uq->A, JyU);
            Mult_IP(J->nDofsI, U->nDofsJ, U->nDofsI, J->Abb, Vq->A, JyV);

            // take cross product by multiplying the x projection of the row vector with
            // the y component of the column vector and vice versa
            Mult_IP(U->nDofsJ, U->nDofsJ, U->nDofsI, UtQ, JyU, UtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, U->nDofsI, UtQ, JyV, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, V->nDofsI, VtQ, JxU, VtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, V->nDofsI, VtQ, JxV, VtQV);

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
    VecRestoreArray(q0, &q0Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

RotMat::~RotMat() {
    Free2D(J->nDofsI, JxU);
    Free2D(J->nDofsI, JxV);
    Free2D(J->nDofsI, JyU);
    Free2D(J->nDofsI, JyV);
    Free2D(U->nDofsJ, JxUt);
    Free2D(V->nDofsJ, JyVt);
    Free2D(U->nDofsJ, UtQ);
    Free2D(V->nDofsJ, VtQ);
    Free2D(U->nDofsJ, UtQU);
    Free2D(U->nDofsJ, UtQV);
    Free2D(V->nDofsJ, VtQU);
    Free2D(V->nDofsJ, VtQV);
    delete[] UtQUflat;
    delete[] ckx;
    delete[] cky;
    delete Q;
    delete J;
    delete U;
    delete V;
    delete Uq;
    delete Vq;
    MatDestroy(&M);
}

// edge to node incidence matrix
E10mat::E10mat(Topo* _topo) {
    int ex, ey, nn, np1, ii, jj, kk, row;
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
                    kk = ii*np1 + jj;
                    row = inds_1x[kk];
                    cols[0] = inds_0[kk];
                    cols[1] = inds_0[kk+np1];
                    vals[0] = +1.0;
                    vals[1] = -1.0;
                    MatSetValues(E10, 1, &row, 2, cols, vals, INSERT_VALUES);

                    // y-normal edge
                    kk = ii*nn + jj;
                    row = inds_1y[kk];
                    cols[0] = inds_0[kk];
                    cols[1] = inds_0[kk+1];
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
