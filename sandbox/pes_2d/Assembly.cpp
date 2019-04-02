#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "Assembly.h"

using namespace std;

// mass matrix for the 1 form vector (x-normal degrees of
// freedom first then y-normal degrees of freedom)
Umat::Umat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    MatCreate(MPI_COMM_SELF, &M);
    MatSetSizes(M, topo->n1, topo->n1, topo->n1, topo->n1);
    MatSetType(M, MATSEQAIJ);
    MatSeqAIJSetPreallocation(M, 8*(l->q->n+1), PETSC_NULL);
}

void Umat::assemble(int lev, bool vert_det) {
    int ex, ii, mp1;
    int *inds_x, *inds_0;
    double det;
    double** Ut = Alloc2D(l->n+1, l->q->n+1);
    double** UtQaa = Alloc2D(l->n+1, l->q->n+1);
    double** UtQU = Alloc2D(l->n+1, l->n+1);
    double** Qaa = Alloc2D(l->q->n+1, l->q->n+1);
    double* UtQUflat = new double[(l->n+1)*(l->n+1)];

    MatZeroEntries(M);

    mp1 = l->q->n + 1;

    for(ex = 0; ex < topo->nElsX; ex++) {
        inds_0 = topo->elInds0(ex);
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];

            Qaa[ii][ii] = det*l->q->w[ii];

            // horiztonal velocity is piecewise constant in the vertical
            if(vert_det) {
                Qaa[ii][ii] *= 2.0/geom->thick[lev][inds_0[ii]];
            }
        }

        inds_x = topo->elInds1(ex);

        Tran_IP(l->q->n+1, l->n+1, l->ljxi, Ut);
        Mult_IP(l->n+1, l->q->n+1, l->q->n+1, Ut, Qaa, UtQaa);
        Mult_IP(l->n+1, l->n+1, l->q->n+1, UtQaa, l->ljxi, UtQU);

        Flat2D_IP(l->n+1, l->n+1, UtQU, UtQUflat);
        MatSetValues(M, l->n+1, inds_x, l->n+1, inds_x, UtQUflat, ADD_VALUES);
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(l->n+1, Ut);
    Free2D(l->n+1, UtQaa);
    Free2D(l->n+1, UtQU);
    Free2D(l->q->n+1, Qaa);
    delete[] UtQUflat;
}

Umat::~Umat() {
    MatDestroy(&M);
}

// 2 form mass matrix
Wmat::Wmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    MatCreate(MPI_COMM_SELF, &M);
    MatSetSizes(M, topo->n2, topo->n2, topo->n2, topo->n2);
    MatSetType(M, MATSEQAIJ);
    MatSeqAIJSetPreallocation(M, 4*e->n, PETSC_NULL);
}

void Wmat::assemble(int lev, bool vert_det) {
    int ex, mp1, ii, *inds, *inds0;
    double det;
    double** Qaa = Alloc2D(e->l->q->n+1, e->l->q->n+1);
    double** Wt = Alloc2D(e->n, e->l->q->n+1);
    double** WtQ = Alloc2D(e->n, e->l->q->n+1);
    double** WtQW = Alloc2D(e->n, e->n);
    double* WtQWflat = new double[e->n*e->n];

    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;

    for(ex = 0; ex < topo->nElsX; ex++) {
        inds  = topo->elInds2(ex);
        inds0 = topo->elInds0(ex);
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            Qaa[ii][ii]  = det*e->l->q->w[ii]/det/det;
            if(vert_det) {
                Qaa[ii][ii] *= 2.0/geom->thick[lev][inds0[ii]];
            }
        }

        Tran_IP(e->l->q->n+1, e->n, e->ejxi, Wt);
        Mult_IP(e->n, e->l->q->n+1, e->l->q->n+1, Wt, Qaa, WtQ);
        Mult_IP(e->n, e->n, e->l->q->n+1, WtQ, e->ejxi, WtQW);

        Flat2D_IP(e->n, e->n, WtQW, WtQWflat);

        MatSetValues(M, e->n, inds, e->n, inds, WtQWflat, ADD_VALUES);
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(e->l->q->n+1, Qaa);
    Free2D(e->n, Wt);
    Free2D(e->n, WtQ);
    Free2D(e->n, WtQW);
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

    UtQU = Alloc2D(l->n+1, l->n+1);
    Qaa = Alloc2D(l->q->n+1, l->q->n+1);
    Ut = Alloc2D(l->n+1, l->q->n+1);
    UtQaa = Alloc2D(l->n+1, l->q->n+1);
    UtQUflat = new double[(l->n+1)*(l->n+1)];

    MatCreate(MPI_COMM_SELF, &M);
    MatSetSizes(M, topo->n1, topo->n1, topo->n1, topo->n1);
    MatSetType(M, MATSEQAIJ);
    MatSeqAIJSetPreallocation(M, 8*(l->q->n+1), PETSC_NULL);
}

void Uhmat::assemble(Vec h2, int lev, bool const_vert) {
    int ex, mp1, ii;
    int *inds_x, *inds_0;
    double hi, det;
    PetscScalar *h2Array;

    mp1 = l->q->n + 1;

    MatZeroEntries(M);
    VecGetArray(h2, &h2Array);

    for(ex = 0; ex < topo->nElsX; ex++) {
        inds_0 = topo->elInds0(ex);
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            geom->interp2(ex, ii, h2Array, &hi);

            // density field is piecewise constant in the vertical
            if(const_vert) {
                hi *= 2.0/geom->thick[lev][inds_0[ii]];
            }

            Qaa[ii][ii] = hi*det*l->q->w[ii];

            // horiztonal velocity is piecewise constant in the vertical
            Qaa[ii][ii] *= 2.0/geom->thick[lev][inds_0[ii]];
        }

        Tran_IP(l->q->n+1, l->n+1, l->ljxi, Ut);

        // reuse the JU and JV matrices for the nonlinear trial function expansion matrices
        Mult_IP(l->n+1, l->q->n+1, l->q->n+1, Ut, Qaa, UtQaa);
        Mult_IP(l->n+1, l->n+1, l->q->n+1, UtQaa, l->ljxi, UtQU);
        inds_x = topo->elInds1(ex);

        Flat2D_IP(l->n+1, l->n+1, UtQU, UtQUflat);
        MatSetValues(M, l->n+1, inds_x, l->n+1, inds_x, UtQUflat, ADD_VALUES);
    }
    VecRestoreArray(h2, &h2Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

Uhmat::~Uhmat() {
    Free2D(l->n+1, UtQU);
    Free2D(l->q->n+1, Qaa);
    Free2D(l->n+1, Ut);
    Free2D(l->n+1, UtQaa);
    delete[] UtQUflat;

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

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &vg);
    VecZeroEntries(vg);

    entries = new PetscScalar[(l->n+1)*(l->n+1)];
}

void Pvec::assemble(int lev, bool vert_det) {
    int ii, ex, np1, np12, *inds_l;
    double det;

    VecZeroEntries(vl);

    np1 = l->n + 1;
    np12 = np1*np1;

    // assemble values into local vector
    for(ex = 0; ex < topo->nElsX; ex++) {
        inds_l = topo->elInds0(ex);
        for(ii = 0; ii < np12; ii++) {
            det = geom->det[ex][ii];
            entries[ii]  = det*l->q->w[ii];
            if(vert_det) {
                entries[ii] *= 2.0/geom->thick[lev][inds_l[ii]];
            }
        }
        VecSetValues(vg, np12, inds_l, entries, ADD_VALUES);
    }
}

Pvec::~Pvec() {
    delete[] entries;
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
    int ex, mp1, ii, *inds_2, *inds_0;
    double det;
    double** Qaa = Alloc2D(e->l->q->n+1, e->l->q->n+1);
    double** Wt = Alloc2D(e->n, e->l->q->n+1);
    double** WtQ = Alloc2D(e->n, e->l->q->n+1);
    double* WtQflat = new double[e->n*(e->l->q->n+1)];

    MatCreate(MPI_COMM_SELF, &M);
    MatSetSizes(M, topo->n2, topo->n0, topo->n2, topo->n0);
    MatSetType(M, MATSEQAIJ);
    MatSeqAIJSetPreallocation(M, 4*e->n, PETSC_NULL);
    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;

    for(ex = 0; ex < topo->nElsX; ex++) {
        // piecewise constant field in the vertical, so vertical transformation is det/det = 1
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            Qaa[ii][ii] = det*e->l->q->w[ii]/det;
        }

        Tran_IP(e->l->q->n+1, e->n, e->ejxi, Wt);

        Mult_IP(e->n, e->l->q->n+1, e->l->q->n+1, Wt, Qaa, WtQ);
        Flat2D_IP(e->n, e->l->q->n+1, WtQ, WtQflat);

        inds_2 = topo->elInds2(ex);
        inds_0 = topo->elInds0(ex);

        MatSetValues(M, e->n, inds_2, e->l->q->n+1, inds_0, WtQflat, ADD_VALUES);
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(e->l->q->n+1, Qaa);
    Free2D(e->n, Wt);
    Free2D(e->n, WtQ);
    delete[] WtQflat;
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
    int ex, ii;
    int *inds_0;
    double** Pt = Tran(l->q->n+1, l->q->n+1, l->ljxi);
    double** PtQ = Alloc2D(l->q->n+1, l->q->n+1);
    double* PtQflat = new double[(l->q->n+1)*(l->q->n+1)];
    double** Q = Alloc2D(l->q->n+1, l->q->n+1);

    MatCreate(MPI_COMM_SELF, &M);
    MatSetSizes(M, topo->n0, topo->n0, topo->n0, topo->n0);
    MatSetType(M, MATSEQAIJ);
    MatSeqAIJSetPreallocation(M, 4*(l->q->n+1), PETSC_NULL);
    MatZeroEntries(M);

    for(ex = 0; ex < topo->nElsX; ex++) {
        // incorportate jacobian tranformation for each element
        // piecewise constant field in the vertical, so vertical transformation is det/det = 1
        for(ii = 0; ii < l->q->n+1; ii++) {
            Q[ii][ii] = geom->det[ex][ii]*l->q->w[ii];
        }
        Mult_IP(l->q->n+1, l->q->n+1, l->q->n+1, Pt, Q, PtQ);
        Flat2D_IP(l->q->n+1, l->q->n+1, PtQ, PtQflat);

        inds_0 = topo->elInds0(ex);
        MatSetValues(M, l->n+1, inds_0, l->q->n+1, inds_0, PtQflat, ADD_VALUES);
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(l->q->n+1, Q);
    Free2D(l->q->n+1, Pt);
    Free2D(l->q->n+1, PtQ);
    delete[] PtQflat;
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
    int ex, ii, mp1;
    int *inds_x, *inds_0;
    double det;
    double** Ut = Alloc2D(l->n+1, l->q->n+1);
    double** UtQ = Alloc2D(l->n+1, l->q->n+1);
    double** Qaa = Alloc2D(l->q->n+1, l->q->n+1);
    double* UtQflat = new double[(l->n+1)*(l->q->n+1)];

    mp1 = l->q->n + 1;

    MatCreate(MPI_COMM_SELF, &M);
    MatSetSizes(M, topo->n1, topo->n0, topo->n1, topo->n0);
    MatSetType(M, MATSEQAIJ);
    MatSeqAIJSetPreallocation(M, 8*(l->n+1), PETSC_NULL);
    MatZeroEntries(M);

    for(ex = 0; ex < topo->nElsX; ex++) {
        // piecewise constant field in the vertical, so vertical transformation is det/det = 1
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];

            Qaa[ii][ii] = det*l->q->w[ii]/det;
        }

        inds_x = topo->elInds1(ex);
        inds_0 = topo->elInds0(ex);

        Tran_IP(l->q->n+1, l->n+1, l->ljxi, Ut);

        Mult_IP(l->n+1, l->q->n+1, l->q->n+1, Ut, Qaa, UtQ);
        Flat2D_IP(l->n+1, l->q->n+1, UtQ, UtQflat);
        MatSetValues(M, l->n+1, inds_x, e->l->q->n+1, inds_0, UtQflat, ADD_VALUES);
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(l->n+1, Ut);
    Free2D(l->n+1, UtQ);
    Free2D(l->q->n+1, Qaa);
    delete[] UtQflat;
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

    Wt = Alloc2D(e->n, e->l->q->n+1);
    Qaa = Alloc2D(e->l->q->n+1, e->l->q->n+1);
    WtQaa = Alloc2D(e->n, e->l->q->n+1);
    WtQU = Alloc2D(e->n, l->n+1);
    WtQUflat = new double[e->n*(l->n+1)];

    MatCreate(MPI_COMM_SELF, &M);
    MatSetSizes(M, topo->n2, topo->n1, topo->n2, topo->n1);
    MatSetType(M, MATSEQAIJ);
    MatSeqAIJSetPreallocation(M, 4*(l->n+1), PETSC_NULL);
}

void WtQUmat::assemble(Vec u1, int lev) {
    int ex, ii, mp1;
    int *inds_x, *inds_2, *inds_0;
    double det, ux[1];
    PetscScalar *u1Array;

    mp1 = l->n + 1;

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    for(ex = 0; ex < topo->nElsX; ex++) {
        inds_0 = topo->elInds0(ex);
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            geom->interp1(ex, ii, u1Array, ux);

            // horiztontal velocity is piecewise constant in the vertical
            ux[0] *= 2.0/geom->thick[lev][inds_0[ii]];

            Qaa[ii][ii] = 0.5*ux[0]*det*l->q->w[ii]/det/det;

            // rescale by the inverse of the vertical determinant (piecewise 
            // constant in the vertical)
            Qaa[ii][ii] *= 2.0/geom->thick[lev][inds_0[ii]];
        }

        Tran_IP(e->l->q->n+1, e->n, e->ejxi, Wt);
        Mult_IP(e->n, e->l->q->n+1, e->l->q->n+1, Wt, Qaa, WtQaa);

        Mult_IP(e->n, l->n+1, e->l->q->n+1, WtQaa, l->ljxi, WtQU);

        Flat2D_IP(e->n, l->n+1, WtQU, WtQUflat);

        inds_x = topo->elInds1(ex);
        inds_2 = topo->elInds2(ex);

        MatSetValues(M, e->n, inds_2, l->n+1, inds_x, WtQUflat, ADD_VALUES);
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

WtQUmat::~WtQUmat() {
    Free2D(e->n, Wt);
    Free2D(e->l->q->n+1, Qaa);
    Free2D(e->n, WtQU);
    Free2D(e->n, WtQaa);
    delete[] WtQUflat;
    MatDestroy(&M);
}

// face to edge incidence matrix
E21mat::E21mat(Topo* _topo) {
    int ex, nn, ii, row;
    int *inds2, *inds1;
    int cols[2];
    double vals[2];
    Mat E21t;

    topo = _topo;
    nn = topo->elOrd;

    MatCreate(MPI_COMM_SELF, &E21);
    MatSetSizes(E21, topo->n2, topo->n1, topo->n2, topo->n1);
    MatSetType(E21, MATSEQAIJ);
    MatSeqAIJSetPreallocation(E21, 4, PETSC_NULL);
    MatZeroEntries(E21);
   
    for(ex = 0; ex < topo->nElsX; ex++) {
        inds1 = topo->elInds1(ex);
        inds2 = topo->elInds2(ex);

        for(ii = 0; ii < nn; ii++) {
            row = inds2[ii];
            cols[0] = inds1[ii+0];
            cols[1] = inds1[ii+1];
            vals[0] = -1.0;
            vals[1] = +1.0;
            MatSetValues(E21, 1, &row, 2, cols, vals, INSERT_VALUES);
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

    MatCreate(MPI_COMM_SELF, &M);
    MatSetSizes(M, topo->n2, topo->n2, topo->n2, topo->n2);
    MatSetType(M, MATSEQAIJ);
    MatSeqAIJSetPreallocation(M, 4*e->n, PETSC_NULL);
}

void Whmat::assemble(Vec rho, int lev, bool vert_const) {
    int ex, mp1, ii, *inds, *inds0;
    double det, p;
    double** Qaa = Alloc2D(e->l->q->n+1, e->l->q->n+1);
    double** Wt = Alloc2D(e->n, e->l->q->n+1);
    double** WtQ = Alloc2D(e->n, e->l->q->n+1);
    double** WtQW = Alloc2D(e->n, e->n);
    double* WtQWflat = new double[e->n*e->n];
    PetscScalar* pArray;

    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;

    VecGetArray(rho, &pArray);
    for(ex = 0; ex < topo->nElsX; ex++) {
        inds  = topo->elInds2(ex);
        inds0 = topo->elInds0(ex);
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];

            geom->interp2(ex, ii, pArray, &p);
            // density is piecewise constant in the vertical
            p *= 2.0/geom->thick[lev][inds0[ii]];

            Qaa[ii][ii]  = p*det*e->l->q->w[ii]/det/det;
            if(vert_const) {
                Qaa[ii][ii] *= 2.0/geom->thick[lev][inds0[ii]];
            }
        }

        Tran_IP(e->l->q->n+1, e->n, e->ejxi, Wt);
        Mult_IP(e->n, e->l->q->n+1, e->l->q->n+1, Wt, Qaa, WtQ);
        Mult_IP(e->n, e->n, e->l->q->n+1, WtQ, e->ejxi, WtQW);

        Flat2D_IP(e->n, e->n, WtQW, WtQWflat);

        MatSetValues(M, e->n, inds, e->n, inds, WtQWflat, ADD_VALUES);
    }
    VecRestoreArray(rho, &pArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(e->l->q->n+1, Qaa);
    Free2D(e->n, Wt);
    Free2D(e->n, WtQ);
    Free2D(e->n, WtQW);
    delete[] WtQWflat;
}

Whmat::~Whmat() {
    MatDestroy(&M);
}
