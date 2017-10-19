#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>

#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"

void Free2D(int ni, double** A) {
    int ii;

    for(ii = 0; ii < ni; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

double* Flat2D(int ni, int nj, double** A) {
    int ii, jj, kk;
    double* Aflat = new double[ni*nj];

    kk = 0;
    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            Aflat[kk] = A[ii][jj];
            kk++;
        }
    }

    return Aflat;
}

// mass matrix for the 1 form vector (x-normal degrees of
// freedom first then y-normal degrees of freedom)
Umat::Umat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    l = _l;
    e = _e;

    assemble();
}

void Umat::assemble() {
    int ex, ey;
    int *inds_x, *inds_y;

    Wii* Q = new Wii(l->q);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double** Ut = tran(U->nDofsI, U->nDofsJ, U->A);
    double** Vt = tran(V->nDofsI, V->nDofsJ, V->A);
    double** UtQ = mult(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Q->A);
    double** VtQ = mult(V->nDofsJ, Q->nDofsJ, V->nDofsI, Vt, Q->A);
    double** UtQU = mult(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQ, U->A);
    double** VtQV = mult(V->nDofsJ, V->nDofsJ, Q->nDofsJ, VtQ, V->A);
    double* UtQUflat = Flat2D(U->nDofsJ, U->nDofsJ, UtQU);
    double* VtQVflat = Flat2D(V->nDofsJ, V->nDofsJ, VtQV);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1, topo->n1, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map1, topo->map1);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            // TODO: incorporate the jacobian transformation for each element
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);
            MatSetValues(M, V->nDofsJ, inds_y, V->nDofsJ, inds_y, VtQVflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQ);
    Free2D(V->nDofsJ, VtQ);
    Free2D(U->nDofsJ, UtQU);
    Free2D(V->nDofsJ, VtQV);
    delete Q;
    delete U;
    delete V;
    delete[] UtQUflat;
    delete[] VtQVflat;
}

Umat::~Umat() {
    MatDestroy(&M);
}

// 2 form mass matrix
Wmat::Wmat(Topo* _topo, LagrangeEdge* _e) {
    topo = _topo;
    e = _e;

    assemble();
}

void Wmat::assemble() {
    int ex, ey;
    int* inds;

    Wii* Q = new Wii(e->l->q);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double** Wt = tran(W->nDofsI, W->nDofsJ, W->A);
    double** WtQ = mult(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q->A);
    double** WtQW = mult(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A);
    double* WtQWflat = Flat2D(W->nDofsJ, W->nDofsJ, WtQW);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2, topo->n2, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map2, topo->map2);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);
            // TODO: incorporate the jacobian transformation for each element
            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQWflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete W;
    delete[] WtQWflat;
}

Wmat::~Wmat() {
    MatDestroy(&M);
}

// 0 form mass matrix
Pmat::Pmat(Topo* _topo, LagrangeNode* _l) {
    topo = _topo;
    l = _l;

    assemble();
}

void Pmat::assemble() {
    int ex, ey;
    int* inds;

    Wii* Q = new Wii(l->q);
    M0_j_xy_i* P = new M0_j_xy_i(l);
    double** Pt = tran(P->nDofsI, P->nDofsJ, P->A);
    double** PtQ = mult(P->nDofsJ, Q->nDofsJ, P->nDofsI, Pt, Q->A);
    double** PtQP = mult(P->nDofsJ, P->nDofsJ, Q->nDofsJ, PtQ, P->A);
    double* PtQPflat = Flat2D(P->nDofsJ, P->nDofsJ, PtQP);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0, topo->n0, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 2*P->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map0, topo->map0);
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
Uhmat::Uhmat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    l = _l;
    e = _e;

    Wii* Q = new Wii(l->q);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double** Ut = tran(U->nDofsI, U->nDofsJ, U->A);
    double** Vt = tran(V->nDofsI, V->nDofsJ, V->A);
    UtQ = mult(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Q->A);
    VtQ = mult(V->nDofsJ, Q->nDofsJ, V->nDofsI, Vt, Q->A);
    UtQUflat = new double[U->nDofsJ*U->nDofsJ];
    VtQVflat = new double[V->nDofsJ*V->nDofsJ];
    ck = new double[l->n*l->n];

    Uh = new M1x_j_Fxy_i(l, e);
    Vh = new M1y_j_Fxy_i(l, e);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1, topo->n1, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map1, topo->map1);

    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    delete Q;
    delete U;
    delete V;
}

void Uhmat::assemble(Vec h2) {
    int ex, ey, ii, jj, kk, n2;
    int *inds_x, *inds_y, *inds2;
    double **UtQU, **VtQV;
    PetscScalar* h2Array;

    n2 = topo->elOrd*topo->elOrd;

    MatZeroEntries(M);
    VecGetArray(h2, &h2Array);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // TODO: incorporate the jacobian transformation for each element
            inds2 = topo->elInds2_l(ex, ey);
            for(kk = 0; kk < n2; kk++) {
                ck[kk] = h2Array[inds2[kk]];
            }
            Uh->assemble(ck);
            Vh->assemble(ck);
            UtQU = mult(Uh->nDofsJ, Uh->nDofsJ, Uh->nDofsI, UtQ, Uh->A);
            VtQV = mult(Vh->nDofsJ, Vh->nDofsJ, Vh->nDofsI, VtQ, Vh->A);

            kk = 0;
            for(ii = 0; ii < Uh->nDofsJ; ii++) {
                for(jj = 0; jj < Uh->nDofsJ; jj++) {
                    UtQUflat[kk] = UtQU[ii][jj];
                    VtQVflat[kk] = VtQV[ii][jj];
                    kk++;
                }
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            MatSetValues(M, Uh->nDofsJ, inds_x, Uh->nDofsJ, inds_x, UtQUflat, ADD_VALUES);
            MatSetValues(M, Vh->nDofsJ, inds_y, Vh->nDofsJ, inds_y, VtQVflat, ADD_VALUES);

            Free2D(Uh->nDofsJ, UtQU);
            Free2D(Vh->nDofsJ, VtQV);
        }
    }
    VecRestoreArray(h2, &h2Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

Uhmat::~Uhmat() {
    delete[] UtQUflat;
    delete[] VtQVflat;
    delete[] ck;

    Free2D(Uh->nDofsJ, UtQ);
    Free2D(Vh->nDofsJ, VtQ);

	delete Uh;
    delete Vh;

    MatDestroy(&M);
}

// Assembly of the diagonal 0 form mass matrix as a vector.
// Assumes inexact integration and a diagonal mass matrix for the 
// 0 form function space (ie: quadrature and basis functions are 
// the same order)
Pvec::Pvec(Topo* _topo, LagrangeNode* _l) {
    topo = _topo;
    l = _l;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, PETSC_DETERMINE, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vg);
    VecZeroEntries(vg);
    VecScatterCreate(vg, topo->is_g_0, vl, topo->is_l_0, &gtol);
    //VecSetLocalToGlobalMapping(v, topo->map0);

    entries = new PetscScalar[(l->n+1)*(l->n+1)];

    assemble();
}

void Pvec::assemble() {
    int ii, ex, ey, np1, np12;
    int *inds_x;
    double* weights = l->q->w;

    VecZeroEntries(vl);

    np1 = l->n + 1;
    np12 = np1*np1;

    // assemble values into local vector
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // TODO: incorporate the jacobian transformation for each element
            for(ii = 0; ii < np12; ii++) {
                // weight at quadrature point
                entries[ii] = weights[ii%np1]*weights[ii/np1];
            }
            inds_x = topo->elInds0_l(ex, ey);
            VecSetValuesLocal(vl, np12, inds_x, entries, ADD_VALUES);
        }
    }

    // scatter values to global vector
    VecScatterBegin(gtol, vl, vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(gtol, vl, vg, ADD_VALUES, SCATTER_REVERSE);

    // and back to local vector
    VecScatterBegin(gtol, vg, vl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol, vg, vl, INSERT_VALUES, SCATTER_FORWARD);
}

Pvec::~Pvec() {
    delete[] entries;
    VecDestroy(&vl);
    VecDestroy(&vg);
    VecScatterDestroy(&gtol);
}

// Assembly of the diagonal 0 form mass matrix as a vector 
// with 2 form vector interpolated onto quadrature points.
// Assumes inexact integration and a diagonal mass matrix for the 
// 0 form function space (ie: quadrature and basis functions are 
// the same order)
Phvec::Phvec(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    l = _l;
    e = _e;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, PETSC_DETERMINE, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vg);
    VecZeroEntries(vg);
    VecScatterCreate(vg, topo->is_g_0, vl, topo->is_l_0, &gtol);
    //VecSetLocalToGlobalMapping(v, topo->map0);

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
            VecSetValuesLocal(vl, np12, inds_x, entries, ADD_VALUES);
        }
    }
    VecRestoreArray(h2, &h2Array);

    // scatter values to global vector
    VecScatterBegin(gtol, vl, vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(gtol, vl, vg, ADD_VALUES, SCATTER_REVERSE);

    // and back to local vector
    VecScatterBegin(gtol, vg, vl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol, vg, vl, INSERT_VALUES, SCATTER_FORWARD);
}

Phvec::~Phvec() {
    delete[] ck;
    delete[] entries;
    VecDestroy(&vl);
    VecDestroy(&vg);
    VecScatterDestroy(&gtol);
}

// Assumes quadrature points and 0 forms are the same (for now)
WtQmat::WtQmat(Topo* _topo, LagrangeEdge* _e) {
    topo = _topo;
    e = _e;

    assemble();
}

void WtQmat::assemble() {
    int ex, ey;
    int *inds_2, *inds_0;

    M2_j_xy_i* W = new M2_j_xy_i(e);
    Wii* Q = new Wii(e->l->q);
    double** Wt = tran(W->nDofsI, W->nDofsJ, W->A);
    double** WtQ = mult(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Q->A);
    double* WtQflat = Flat2D(W->nDofsJ, Q->nDofsJ, WtQ);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2, topo->n0, topo->nDofs2G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map2, topo->map0);
    MatZeroEntries(M);

    // TODO: incorportate jacobian tranformation for each element
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_2 = topo->elInds2_g(ex, ey);
            inds_0 = topo->elInds0_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, Q->nDofsJ, inds_0, WtQflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

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
PtQmat::PtQmat(Topo* _topo, LagrangeNode* _l) {
    topo = _topo;
    l = _l;

    assemble();
}

void PtQmat::assemble() {
    int ex, ey;
    int *inds_0;

    M0_j_xy_i* P = new M0_j_xy_i(l);
    Wii* Q = new Wii(l->q);
    double** Pt = tran(P->nDofsI, P->nDofsJ, P->A);
    double** PtQ = mult(P->nDofsJ, Q->nDofsJ, Q->nDofsI, Pt, Q->A);
    double* PtQflat = Flat2D(P->nDofsJ, Q->nDofsJ, PtQ);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0, topo->n0, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 2*P->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map0, topo->map0);
    MatZeroEntries(M);

    // TODO: incorportate jacobian tranformation for each element
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
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

//
UtQmat::UtQmat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    l = _l;
    e = _e;

    assemble();
}

void UtQmat::assemble() {
    int ex, ey;
    int *inds_x, *inds_y, *inds_q;

    Wii* Q = new Wii(l->q);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double** Ut = tran(U->nDofsI, U->nDofsJ, U->A);
    double** Vt = tran(V->nDofsI, V->nDofsJ, V->A);
    double** UtQ = mult(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Q->A);
    double** VtQ = mult(V->nDofsJ, Q->nDofsJ, V->nDofsI, Vt, Q->A);
    double* UtQflat = Flat2D(U->nDofsJ, Q->nDofsJ, UtQ);
    double* VtQflat = Flat2D(V->nDofsJ, Q->nDofsJ, VtQ);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1, topo->n0, topo->nDofs1G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map1, topo->map0);
    MatZeroEntries(M);

    // TODO: incorportate jacobian tranformation for each element
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_q = topo->elInds0_g(ex, ey);

            MatSetValues(M, U->nDofsJ, inds_x, Q->nDofsJ, inds_q, UtQflat, ADD_VALUES);
            MatSetValues(M, V->nDofsJ, inds_y, Q->nDofsJ, inds_q, VtQflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQ);
    Free2D(V->nDofsJ, VtQ);
    delete[] UtQflat;
    delete[] VtQflat;
    delete Q;
    delete U;
    delete V;
}

UtQmat::~UtQmat() {
    MatDestroy(&M);
}

// project the potential vorticity gradient velocity product onto the 0 forms
PtQUmat::PtQUmat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    double **Pt;
    Wii* Q;

    topo = _topo;
    l = _l;
    e = _e;

    U = new M1x_j_Exy_i(l, e);
    V = new M1y_j_Exy_i(l, e);
    P = new M0_j_xy_i(l);
    Pt = tran(P->nDofsI, P->nDofsJ, P->A);
    Q = new Wii(l->q);
    PtQ = mult(P->nDofsJ, Q->nDofsJ, Q->nDofsI, Pt, Q->A);

    PtQUflat = new double[P->nDofsJ*U->nDofsJ];
    PtQVflat = new double[P->nDofsJ*V->nDofsJ];
    ckx = new double[(l->n+1)*(l->n)];
    cky = new double[(l->n)*(l->n+1)];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0, topo->n1, topo->nDofs0G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map0, topo->map1);

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
            PtQU = mult(P->nDofsJ, U->nDofsJ, U->nDofsI, PtQ, U->A);
            PtQV = mult(P->nDofsJ, V->nDofsJ, V->nDofsI, PtQ, V->A);

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
WtQUmat::WtQUmat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    double **Wt;
    Wii* Q;

    topo = _topo;
    l = _l;
    e = _e;

    U = new M1x_j_Cxy_i(l, e);
    V = new M1y_j_Cxy_i(l, e);
    W = new M2_j_xy_i(e);
    Wt = tran(W->nDofsI, W->nDofsJ, W->A);
    Q = new Wii(l->q);
    WtQ = mult(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Q->A);

    WtQUflat = new double[W->nDofsJ*U->nDofsJ];
    WtQVflat = new double[W->nDofsJ*V->nDofsJ];
    ckx = new double[(l->n+1)*(l->n)];
    cky = new double[(l->n)*(l->n+1)];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2, topo->n1, topo->nDofs2G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map2, topo->map1);

    Free2D(W->nDofsJ, Wt);
    delete Q;
}

void WtQUmat::assemble(Vec u1) {
    int ex, ey, ii, jj, kk, n2;
    int *inds_x, *inds_y, *inds_2;
    double **WtQU, **WtQV;
    PetscScalar* u1Array;

    n2 = (l->n+1)*(l->n);

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    // TODO: incorportate jacobian tranformation for each element
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);
            // note that we are assembling K = (u.u)/2 onto 2 forms
            for(kk = 0; kk < n2; kk++) {
                ckx[kk] = 0.5*u1Array[inds_x[kk]];
                cky[kk] = 0.5*u1Array[inds_y[kk]];
            }
            U->assemble(cky);
            V->assemble(ckx);
            // TODO: incorporate the jacobian transformation for each element
            WtQU = mult(W->nDofsJ, U->nDofsJ, U->nDofsI, WtQ, U->A);
            WtQV = mult(W->nDofsJ, V->nDofsJ, V->nDofsI, WtQ, V->A);

            kk = 0;
            for(ii = 0; ii < W->nDofsJ; ii++) {
                for(jj = 0; jj < U->nDofsJ; jj++) {
                    WtQUflat[kk] = WtQU[ii][jj];
                    WtQVflat[kk] = WtQV[ii][jj];
                    kk++;
                }
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_2 = topo->elInds0_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, U->nDofsJ, inds_x, WtQUflat, ADD_VALUES);
            MatSetValues(M, W->nDofsJ, inds_2, V->nDofsJ, inds_y, WtQVflat, ADD_VALUES);

            Free2D(W->nDofsJ, WtQU);
            Free2D(W->nDofsJ, WtQV);
        }
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

WtQUmat::~WtQUmat() {
    Free2D(W->nDofsJ, WtQ);
    delete[] WtQUflat;
    delete[] WtQVflat;
    delete[] ckx;
    delete[] cky;
    delete U;
    delete V;
    delete W;
    MatDestroy(&M);
}

// 1 form mass matrix with 0 form interpolated to quadrature points
// (for rotational term in the momentum equation)
RotMat::RotMat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    double **Ut, **Vt;
    Wii* Q;

    topo = _topo;
    l = _l;
    e = _e;

    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);
    Ut = tran(U->nDofsI, U->nDofsJ, U->A);
    Vt = tran(V->nDofsI, V->nDofsJ, V->A);
    Q = new Wii(l->q);
    UtQ = mult(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Q->A);
    VtQ = mult(V->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Q->A);
    Uq = new M1x_j_Dxy_i(l, e);
    Vq = new M1y_j_Dxy_i(l, e);

    UtQVflat = new double[U->nDofsJ*V->nDofsJ];
    VtQUflat = new double[V->nDofsJ*U->nDofsJ];
    ckx = new double[(l->n+1)*(l->n+1)];
    cky = new double[(l->n+1)*(l->n+1)];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1, topo->n1, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map1, topo->map1);

    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    delete Q;
}

void RotMat::assemble(Vec q0) {
    int ex, ey, ii, jj, kk, np12;
    int *inds_x, *inds_y, *inds_0;
    double **UtQV, **VtQU;
    PetscScalar* q0Array;

    np12 = (l->n+1)*(l->n+1);

    VecGetArray(q0, &q0Array);
    MatZeroEntries(M);

    // TODO: incorportate jacobian tranformation for each element
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_0 = topo->elInds0_l(ex, ey);
            for(kk = 0; kk < np12; kk++) {
                ckx[kk] = +1.0*q0Array[inds_0[kk]];
                cky[kk] = -1.0*q0Array[inds_0[kk]];
            }
            Uq->assemble(cky);
            Vq->assemble(ckx);
            // TODO: incorporate the jacobian transformation for each element
            UtQV = mult(U->nDofsJ, Vq->nDofsJ, U->nDofsI, UtQ, Vq->A);
            VtQU = mult(V->nDofsJ, Uq->nDofsJ, V->nDofsI, VtQ, Uq->A);

            kk = 0;
            for(ii = 0; ii < U->nDofsJ; ii++) {
                for(jj = 0; jj < U->nDofsJ; jj++) {
                    UtQVflat[kk] = UtQV[ii][jj];
                    VtQUflat[kk] = VtQU[ii][jj];
                    kk++;
                }
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            MatSetValues(M, U->nDofsJ, inds_x, Vq->nDofsJ, inds_y, UtQVflat, ADD_VALUES);
            MatSetValues(M, V->nDofsJ, inds_y, Uq->nDofsJ, inds_x, VtQUflat, ADD_VALUES);

            Free2D(U->nDofsJ, UtQV);
            Free2D(V->nDofsJ, VtQU);
        }
    }
    VecRestoreArray(q0, &q0Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

RotMat::~RotMat() {
    Free2D(U->nDofsJ, UtQ);
    Free2D(V->nDofsJ, VtQ);
    delete[] UtQVflat;
    delete[] VtQUflat;
    delete[] ckx;
    delete[] cky;
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
    MatSetSizes(E10, topo->n1, topo->n0, topo->nDofs1G, topo->nDofs0G);
    MatSetType(E10, MATMPIAIJ);
    MatMPIAIJSetPreallocation(E10, 4, PETSC_NULL, 4, PETSC_NULL);
    MatSetLocalToGlobalMapping(E10, topo->map1, topo->map0);
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
    MatSetSizes(E21, topo->n2, topo->n1, topo->nDofs2G, topo->nDofs1G);
    MatSetType(E21, MATMPIAIJ);
    MatMPIAIJSetPreallocation(E21, 4, PETSC_NULL, 4, PETSC_NULL);
    MatSetLocalToGlobalMapping(E21, topo->map2, topo->map1);
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
