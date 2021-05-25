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

#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0

#define SCALE 1.0e+8

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

    MT = NULL;

    //MatDuplicate(M, MAT_DO_NOT_COPY_VALUES, &_M);
    MatCreate(MPI_COMM_WORLD, &_M);
    MatSetSizes(_M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(_M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(_M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
    _assemble(0, SCALE, false);

    delete U;
}

void Umat::assemble(int lev, double scale, bool vert_scale) {
    //MatCopy(_M, M, SAME_NONZERO_PATTERN);

    //MatAssemblyBegin(_M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyEnd(  _M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);
    //MatCopy(_M, M, DIFFERENT_NONZERO_PATTERN);
    //MatScale(M, 1.0/geom->thick[lev][0]);
    //MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    _assemble(lev, scale, vert_scale);
}

void Umat::_assemble(int lev, double scale, bool vert_scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    Wii* Q = new Wii(l->q, geom);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double det, **J;
    double* Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    double* Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    double* UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* VtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* VtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* Qaa = new double[Q->nDofsI];
    double* Qab = new double[Q->nDofsI];
    double* Qbb = new double[Q->nDofsI];

    MatZeroEntries(M);

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                Qbb[ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                // horiztonal velocity is piecewise constant in the vertical
                if(vert_scale) {
                    Qaa[ii] *= geom->thickInv[lev][inds_0[ii]];
                    Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
                    Qbb[ii] *= geom->thickInv[lev][inds_0[ii]];
                }
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

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQV, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, VtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, VtQV, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyBegin(_M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyEnd(_M, MAT_FINAL_ASSEMBLY);

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
    delete[] Qaa;
    delete[] Qab;
    delete[] Qbb;
    delete Q;
    delete U;
    delete V;
}

// upwinded test function matrix
void Umat::assemble_up(int lev, double scale, double dt, Vec u1) {
    int ex, ey, ei, m0, mp1, mp12, ii, jj;
    int *inds_x, *inds_y, *inds_0;
    double det, **J, ug[2], ul[2], lx[99], ly[99], _ex[99], _ey[99];
    PetscScalar *u1Array;
    MatReuse reuse = (!MT) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    GaussLobatto* quad = l->q;
    Wii* Q = new Wii(l->q, geom);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double* Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    double* Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    double* UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* VtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* VtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* Qaa = new double[Q->nDofsI];
    double* Qab = new double[Q->nDofsI];
    double* Qbb = new double[Q->nDofsI];

    m0 = l->q->n;
    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    VecGetArray(u1, &u1Array);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ug);
                ug[0] *= geom->thickInv[lev][inds_0[ii]];
                ug[1] *= geom->thickInv[lev][inds_0[ii]];

                // map velocity to local element coordinates
                ul[0] = (+J[1][1]*ug[0] - J[0][1]*ug[1])/det;
                ul[1] = (-J[1][0]*ug[0] + J[0][0]*ug[1])/det;
                // evaluate the nodal bases at the upwinded locations
                for(jj = 0; jj < mp1; jj++) {
                    lx[jj] = l->eval_q(quad->x[ii%mp1] + dt*ul[0], jj);
                    ly[jj] = l->eval_q(quad->x[ii/mp1] + dt*ul[1], jj);
                }
                // evaluate the edge bases at the upwinded locations
                for(jj = 0; jj < m0; jj++) {
                    _ex[jj] = e->eval(quad->x[ii%mp1] + dt*ul[0], jj);
                    _ey[jj] = e->eval(quad->x[ii/mp1] + dt*ul[1], jj);
                }
                // evaluate the 2 form basis at the upwinded locations
                for(jj = 0; jj < m0*mp1; jj++) {
                    Ut[jj*mp12+ii] = lx[jj%mp1]*_ey[jj/mp1];
                    Vt[jj*mp12+ii] = _ex[jj%m0]*ly[jj/m0];
                }

                Qaa[ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                Qbb[ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                // horiztonal velocity is piecewise constant in the vertical
                Qaa[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qbb[ii] *= geom->thickInv[lev][inds_0[ii]];
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

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQV, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, VtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, VtQV, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    MatTranspose(M, reuse, &MT);

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
    delete[] Qaa;
    delete[] Qab;
    delete[] Qbb;
    delete Q;
    delete U;
    delete V;
}

Umat::~Umat() {
    MatDestroy(&M);
    if(MT) MatDestroy(&MT);
    MatDestroy(&_M);
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

    MatCreate(MPI_COMM_WORLD, &_M);
    MatSetSizes(_M, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(_M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(_M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);

    //MatDuplicate(M, MAT_DO_NOT_COPY_VALUES, &_M);
    _assemble(0, SCALE, false);

    delete W;
}

void Wmat::assemble(int lev, double scale, bool vert_scale) {
    //MatAssemblyBegin(_M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyEnd(  _M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);
    //MatCopy(_M, M, DIFFERENT_NONZERO_PATTERN);
    //MatScale(M, 1.0/geom->thick[lev][0]);
    //MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    //MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    _assemble(lev, scale, vert_scale);
}

void Wmat::_assemble(int lev, double scale, bool vert_scale) {
    int ex, ey, ei, mp1, mp12, ii, *inds, *inds0;
    double det;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);

    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii]  = Q->A[ii]*(scale/det);
                if(vert_scale) {
                    Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
                }
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQW, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete W;
    delete Q;
}

Wmat::~Wmat() {
    MatDestroy(&M);
    MatDestroy(&_M);
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
    Qaa = new double[Q->nDofsI];
    Qab = new double[Q->nDofsI];
    Qbb = new double[Q->nDofsI];
    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);

    MT = NULL;

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);
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

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, h2Array, &hi);

                // density field is piecewise constant in the vertical
                if(const_vert) hi *= geom->thickInv[lev][inds_0[ii]];

                Qaa[ii] = hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                Qbb[ii] = hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                // horiztonal velocity is piecewise constant in the vertical
                Qaa[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qbb[ii] *= geom->thickInv[lev][inds_0[ii]];
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

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQV, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, VtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, VtQV, ADD_VALUES);
        }
    }
    VecRestoreArray(h2, &h2Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

// upwinded test function matrix
void Uhmat::assemble_up(Vec h2, int lev, double scale, double dt, Vec u1) {
    int ex, ey, ei, m0, mp1, mp12, ii, jj;
    int *inds_x, *inds_y, *inds_0;
    double hi, det, **J, ug[2], ul[2], lx[99], ly[99], _ex[99], _ey[99];
    PetscScalar *h2Array, *u1Array;
    MatReuse reuse = (!MT) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    GaussLobatto* quad = l->q;

    m0 = l->q->n;
    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    VecGetArray(h2, &h2Array);
    VecGetArray(u1, &u1Array);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, h2Array, &hi);
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ug);
                ug[0] *= geom->thickInv[lev][inds_0[ii]];
                ug[1] *= geom->thickInv[lev][inds_0[ii]];
                // map velocity to local element coordinates
                ul[0] = (+J[1][1]*ug[0] - J[0][1]*ug[1])/det;
                ul[1] = (-J[1][0]*ug[0] + J[0][0]*ug[1])/det;
                // evaluate the nodal bases at the upwinded locations
                for(jj = 0; jj < mp1; jj++) {
                    lx[jj] = l->eval_q(quad->x[ii%mp1] + dt*ul[0], jj);
                    ly[jj] = l->eval_q(quad->x[ii/mp1] + dt*ul[1], jj);
                }
                // evaluate the edge bases at the upwinded locations
                for(jj = 0; jj < m0; jj++) {
                    _ex[jj] = e->eval(quad->x[ii%mp1] + dt*ul[0], jj);
                    _ey[jj] = e->eval(quad->x[ii/mp1] + dt*ul[1], jj);
	        }
                // evaluate the 2 form basis at the upwinded locations
                for(jj = 0; jj < m0*mp1; jj++) {
                    Ut[jj*mp12+ii] = lx[jj%mp1]*_ey[jj/mp1];
                    Vt[jj*mp12+ii] = _ex[jj%m0]*ly[jj/m0];
                }

                Qaa[ii] = hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                Qbb[ii] = hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                // horiztonal velocity is piecewise constant in the vertical
                Qaa[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qbb[ii] *= geom->thickInv[lev][inds_0[ii]];
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

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQV, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, VtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, VtQV, ADD_VALUES);
        }
    }
    VecRestoreArray(h2, &h2Array);
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    MatTranspose(M, reuse, &MT);
}

Uhmat::~Uhmat() {
    Free2D(U->nDofsJ, UtQU);
    Free2D(U->nDofsJ, UtQV);
    Free2D(V->nDofsJ, VtQU);
    Free2D(V->nDofsJ, VtQV);
    delete[] Qaa;
    delete[] Qab;
    delete[] Qbb;
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
    if(MT) MatDestroy(&MT);
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
    int ii, ex, ey, ei, np1, np12, *inds_l;

    VecZeroEntries(vl);
    VecZeroEntries(vg);

    np1 = l->n + 1;
    np12 = np1*np1;

    // assemble values into local vector
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            inds_l = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < np12; ii++) {
                entries[ii]  = scale*Q->A[ii]*geom->det[ei][ii];
                entries[ii] *= geom->thickInv[lev][inds_l[ii]];
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

Pmat::Pmat(Topo* _topo, Geom* _geom, LagrangeNode* _node) {
    topo = _topo;
    geom = _geom;
    node = _node;

    Q = new Wii(node->q, geom);
    P = new M0_j_xy_i(node);
    Pt = Alloc2D(P->nDofsJ, P->nDofsI);
    Qaa = new double[Q->nDofsI];
    PtQ = Alloc2D(P->nDofsJ, Q->nDofsJ);
    PtQP = Alloc2D(P->nDofsJ, P->nDofsJ);

    Tran_IP(P->nDofsI, P->nDofsJ, P->A, Pt);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 4*P->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);
}

void Pmat::assemble(int lev, double scale) {
    int ex, ey, ei, mp1, mp12, ii, *inds;
    double det;

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = geom->elInds0_l(ex, ey);
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii] = scale*Q->A[ii]*det;
                Qaa[ii] *= geom->thickInv[lev][inds[ii]];
            }
            Mult_FD_IP(P->nDofsJ, Q->nDofsJ, Q->nDofsI, Pt, Qaa, PtQ);
            Mult_IP(P->nDofsJ, P->nDofsJ, Q->nDofsJ, PtQ, P->A, PtQP);

            inds = topo->elInds0_g(ex, ey);

            MatSetValues(M, P->nDofsJ, inds, P->nDofsJ, inds, PtQP, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

Pmat::~Pmat() {
    MatDestroy(&M);

    delete[] Qaa;
    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    Free2D(P->nDofsJ, PtQP);
    delete P;
    delete Q;
}

Phvec::Phvec(Topo* _topo, Geom* _geom, LagrangeNode* _l) {
    topo = _topo;
    geom = _geom;
    l = _l;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vg);

    entries = new PetscScalar[(l->n+1)*(l->n+1)];

    Q = new Wii(l->q, geom);
}

void Phvec::assemble(Vec hl, int lev, double scale) {
    int ii, ex, ey, ei, np1, np12, *inds_l;
    double hi;
    PetscScalar* hArray;

    VecZeroEntries(vl);
    VecZeroEntries(vg);

    np1 = l->n + 1;
    np12 = np1*np1;

    VecGetArray(hl, &hArray);

    // assemble values into local vector
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            inds_l = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < np12; ii++) {
                entries[ii]  = scale*Q->A[ii]*geom->det[ei][ii];
                entries[ii] *= geom->thickInv[lev][inds_l[ii]];

                geom->interp2_g(ex, ey, ii%np1, ii/np1, hArray, &hi);
                hi *= geom->thickInv[lev][inds_l[ii]];
                entries[ii] *= hi;
            }
            VecSetValues(vl, np12, inds_l, entries, ADD_VALUES);
        }
    }
    VecRestoreArray(hl, &hArray);

    // scatter values to global vector
    VecScatterBegin(topo->gtol_0, vl, vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_0, vl, vg, ADD_VALUES, SCATTER_REVERSE);
}

Phvec::~Phvec() {
    delete[] entries;
    VecDestroy(&vl);
    VecDestroy(&vg);
    delete Q;
}

Phmat::Phmat(Topo* _topo, Geom* _geom, LagrangeNode* _node) {
    topo = _topo;
    geom = _geom;
    node = _node;

    Q = new Wii(node->q, geom);
    P = new M0_j_xy_i(node);
    Pt = Alloc2D(P->nDofsJ, P->nDofsI);
    Qaa = new double[Q->nDofsI];
    PtQ = Alloc2D(P->nDofsJ, Q->nDofsJ);
    PtQP = Alloc2D(P->nDofsJ, P->nDofsJ);

    Tran_IP(P->nDofsI, P->nDofsJ, P->A, Pt);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 4*P->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);
}

void Phmat::assemble(Vec hl, int lev, double scale) {
    int ex, ey, ei, mp1, mp12, ii, *inds;
    double det, hi;
    PetscScalar* hArray;

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(hl, &hArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = geom->elInds0_l(ex, ey);
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii] = scale*Q->A[ii]*det;
                Qaa[ii] *= geom->thickInv[lev][inds[ii]];

                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, hArray, &hi);
                hi *= geom->thickInv[lev][inds[ii]];
                Qaa[ii] *= hi;
            }
            Mult_FD_IP(P->nDofsJ, Q->nDofsJ, Q->nDofsI, Pt, Qaa, PtQ);
            Mult_IP(P->nDofsJ, P->nDofsJ, Q->nDofsJ, PtQ, P->A, PtQP);

            inds = topo->elInds0_g(ex, ey);

            MatSetValues(M, P->nDofsJ, inds, P->nDofsJ, inds, PtQP, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    VecRestoreArray(hl, &hArray);
}

Phmat::~Phmat() {
    MatDestroy(&M);

    delete[] Qaa;
    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    Free2D(P->nDofsJ, PtQP);
    delete P;
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
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, geom->n0l, topo->nDofs2G, geom->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*Q->nDofsJ, PETSC_NULL, 4*Q->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // piecewise constant field in the vertical, so vertical transformation is det/det = 1
            for(ii = 0; ii < mp12; ii++) {
                Qaa[ii] = Q->A[ii];
            }
            Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qaa, WtQ);

            inds_2 = topo->elInds2_g(ex, ey);
            inds_0 = geom->elInds0_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, Q->nDofsJ, inds_0, WtQ, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
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
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_0, *inds_g;
    M0_j_xy_i* P = new M0_j_xy_i(l);
    Wii* Q = new Wii(l->q, geom);
    double* Pt = Tran(P->nDofsI, P->nDofsJ, P->A);
    double* PtQ = Alloc2D(P->nDofsJ, Q->nDofsJ);
    double* Qaa = new double[Q->nDofsI];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, geom->n0l, topo->nDofs0G, geom->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*Q->nDofsJ, PETSC_NULL, 4*Q->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            // incorportate jacobian tranformation for each element
            // piecewise constant field in the vertical, so vertical transformation is det/det = 1
            for(ii = 0; ii < mp12; ii++) {
                Qaa[ii] = Q->A[ii]*geom->det[ei][ii];
            }
            Mult_FD_IP(P->nDofsJ, Q->nDofsJ, Q->nDofsI, Pt, Qaa, PtQ);

            inds_0 = topo->elInds0_g(ex, ey);
            inds_g = geom->elInds0_g(ex, ey);
            MatSetValues(M, P->nDofsJ, inds_0, Q->nDofsJ, inds_g, PtQ, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    delete[] Qaa;
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
    double* Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    double* Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    double* UtQ = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* Qaa = new double[Q->nDofsI];
    double* Qab = new double[Q->nDofsI];
    double* Qba = new double[Q->nDofsI];
    double* Qbb = new double[Q->nDofsI];

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;
    inds_0x = new int[mp12];
    inds_0y = new int[mp12];

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, 2*geom->n0l, topo->nDofs1G, 2*geom->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*Q->nDofsJ, PETSC_NULL, 8*Q->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // piecewise constant field in the vertical, so vertical transformation is det/det = 1
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                J = geom->J[ei][ii];
                Qaa[ii] = J[0][0]*Q->A[ii];
                Qab[ii] = J[1][0]*Q->A[ii];
                Qba[ii] = J[0][1]*Q->A[ii];
                Qbb[ii] = J[1][1]*Q->A[ii];
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
            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Qaa, UtQ);
            MatSetValues(M, U->nDofsJ, inds_x, Q->nDofsJ, inds_0x, UtQ, ADD_VALUES);

            //
            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Qab, UtQ);
            MatSetValues(M, U->nDofsJ, inds_x, Q->nDofsJ, inds_0y, UtQ, ADD_VALUES);

            //
            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Vt, Qba, UtQ);
            MatSetValues(M, U->nDofsJ, inds_y, Q->nDofsJ, inds_0x, UtQ, ADD_VALUES);

            //
            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, U->nDofsI, Vt, Qbb, UtQ);
            MatSetValues(M, U->nDofsJ, inds_y, Q->nDofsJ, inds_0y, UtQ, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQ);
    delete[] Qaa;
    delete[] Qab;
    delete[] Qba;
    delete[] Qbb;
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
    Qaa = new double[Q->nDofsI];
    Qab = new double[Q->nDofsI];
    WtQaa = Alloc2D(W->nDofsJ, Q->nDofsJ);
    WtQab = Alloc2D(W->nDofsJ, Q->nDofsJ);
    WtQU = Alloc2D(W->nDofsJ, U->nDofsJ);
    WtQV = Alloc2D(W->nDofsJ, V->nDofsJ);

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
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);
                // horiztontal velocity is piecewise constant in the vertical
                ux[0] *= geom->thickInv[lev][inds_0[ii]];
                ux[1] *= geom->thickInv[lev][inds_0[ii]];

                Qaa[ii] = 0.5*(ux[0]*J[0][0] + ux[1]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = 0.5*(ux[0]*J[0][1] + ux[1]*J[1][1])*Q->A[ii]*(scale/det);

                // rescale by the inverse of the vertical determinant (piecewise 
                // constant in the vertical)
                Qaa[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qaa, WtQaa);
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qab, WtQab);

            Mult_IP(W->nDofsJ, U->nDofsJ, U->nDofsI, WtQaa, U->A, WtQU);
            Mult_IP(W->nDofsJ, V->nDofsJ, V->nDofsI, WtQab, V->A, WtQV);

            //Flat2D_IP(W->nDofsJ, U->nDofsJ, WtQU, WtQUflat);
            //Flat2D_IP(W->nDofsJ, V->nDofsJ, WtQV, WtQVflat);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_2 = topo->elInds2_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, U->nDofsJ, inds_x, WtQU, ADD_VALUES);
            MatSetValues(M, W->nDofsJ, inds_2, V->nDofsJ, inds_y, WtQV, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

WtQUmat::~WtQUmat() {
    Free2D(W->nDofsJ, Wt);
    delete[] Qaa;
    delete[] Qab;
    Free2D(W->nDofsJ, WtQU);
    Free2D(W->nDofsJ, WtQV);
    Free2D(W->nDofsJ, WtQaa);
    Free2D(W->nDofsJ, WtQab);
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
    Qab = new double[Q->nDofsI];
    Qba = new double[Q->nDofsI];
    UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQU = Alloc2D(V->nDofsJ, V->nDofsJ);

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

            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp0(ex, ey, ii%mp1, ii/mp1, q0Array, &vort);

                // vertical vorticity is piecewise constant in the vertical
                vort *= geom->thickInv[lev][inds_0[ii]];

                Qab[ii] = vort*(-J[0][0]*J[1][1] + J[0][1]*J[1][0])*Q->A[ii]*(scale/det);
                Qba[ii] = vort*(+J[0][0]*J[1][1] - J[0][1]*J[1][0])*Q->A[ii]*(scale/det);

                Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qba[ii] *= geom->thickInv[lev][inds_0[ii]];
            }

            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Qab, UtQab);
            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Qba, VtQba);

            // take cross product by multiplying the x projection of the row vector with
            // the y component of the column vector and vice versa
            Mult_IP(U->nDofsJ, U->nDofsJ, U->nDofsI, UtQab, V->A, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, V->nDofsI, VtQba, U->A, VtQU);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQV, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, VtQU, ADD_VALUES);
        }
    }
    VecRestoreArray(q0, &q0Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

RotMat::~RotMat() {
    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    delete[] Qab;
    delete[] Qba;
    Free2D(U->nDofsJ, UtQab);
    Free2D(U->nDofsJ, VtQba);
    Free2D(U->nDofsJ, UtQV);
    Free2D(V->nDofsJ, VtQU);

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

void Whmat::assemble(Vec rho, int lev, double scale, bool vert_scale_rho) {
    int ex, ey, ei, mp1, mp12, ii, *inds, *inds0;
    double det, p;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    PetscScalar* pArray;

    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecGetArray(rho, &pArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];

                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, pArray, &p);
                // density is piecewise constant in the vertical
                if(vert_scale_rho) {
                    p *= geom->thickInv[lev][inds0[ii]];
                }

                Qaa[ii]  = p*Q->A[ii]*(scale/det);
                // W is piecewise constant in the vertical
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            //Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQW, ADD_VALUES);
        }
    }
    VecRestoreArray(rho, &pArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete W;
    delete Q;
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

    Q = new Wii(l->q, geom);
    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);
    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);
    UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    Qaa = new double[Q->nDofsI];
    Qab = new double[Q->nDofsI];
    Qbb = new double[Q->nDofsI];

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
}

void Ut_mat::assemble(int lev, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    double det, **J;

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                Qbb[ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                // horiztonal velocity is piecewise constant in the vertical
                Qaa[ii] *= 0.5*(geom->thick[lev][inds_0[ii]] + geom->thick[lev+1][inds_0[ii]]);
                Qab[ii] *= 0.5*(geom->thick[lev][inds_0[ii]] + geom->thick[lev+1][inds_0[ii]]);
                Qbb[ii] *= 0.5*(geom->thick[lev][inds_0[ii]] + geom->thick[lev+1][inds_0[ii]]);
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

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQV, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, VtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, VtQV, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

void Ut_mat::assemble_h(int lev, double scale, Vec rho) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y;
    double det, **J, hi;
    PetscScalar* rArray;

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    MatZeroEntries(M);
    VecGetArray(rho, &rArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                // horiztonal velocity is piecewise constant in the vertical, cancels with
                // vertical metric term from the density
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rArray, &hi);

                Qaa[ii] = hi*(J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = hi*(J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                Qbb[ii] = hi*(J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);
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

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQV, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, VtQU, ADD_VALUES);

            //Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, VtQV, ADD_VALUES);
        }
    }
    VecGetArray(rho, &rArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

Ut_mat::~Ut_mat() {
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
    delete[] Qaa;
    delete[] Qab;
    delete[] Qbb;
    delete Q;
    delete U;
    delete V;

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
    Qaa = new double[Q->nDofsI];
    Qba = new double[Q->nDofsI];
    UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(V->nDofsJ, Q->nDofsJ);
    UtQW = Alloc2D(U->nDofsJ, W->nDofsJ);
    VtQW = Alloc2D(V->nDofsJ, W->nDofsJ);

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(V->nDofsI, V->nDofsJ, V->A, Vt);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n1l, topo->n2l, topo->nDofs1G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
}

void UtQWmat::assemble(Vec u1, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_2;
    double det, **J, ux[2];
    PetscScalar *u1Array;

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
                geom->interp1_g_t(ex, ey, ii%mp1, ii/mp1, u1Array, ux);

                // horizontal velocity is piecewise constant, and vertical velocity is 
                // piecewise linear, so vertical transformations cancel

                // once we have mapped degrees of freedom from inner orientations
                // to outer orientations, this transformation is the same as for
                // the H(div) space, and so the mass matrix is the same in the horizontal
                Qaa[ii] = (ux[0]*J[0][0] + ux[1]*J[1][0])*Q->A[ii]*(scale/det);
                Qba[ii] = (ux[0]*J[0][1] + ux[1]*J[1][1])*Q->A[ii]*(scale/det);
            }

            Mult_FD_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Qaa, UtQaa);
            Mult_FD_IP(V->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Qba, VtQba);

            Mult_IP(U->nDofsJ, W->nDofsJ, W->nDofsI, UtQaa, W->A, UtQW);
            Mult_IP(V->nDofsJ, W->nDofsJ, W->nDofsI, VtQba, W->A, VtQW);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_2 = topo->elInds2_g(ex, ey);

            MatSetValues(M, U->nDofsJ, inds_x, W->nDofsJ, inds_2, UtQW, ADD_VALUES);
            MatSetValues(M, V->nDofsJ, inds_y, W->nDofsJ, inds_2, VtQW, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

UtQWmat::~UtQWmat() {
    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    delete[] Qaa;
    delete[] Qba;
    Free2D(U->nDofsJ, UtQW);
    Free2D(V->nDofsJ, VtQW);
    Free2D(U->nDofsJ, UtQaa);
    Free2D(V->nDofsJ, VtQba);
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
    Qaa = new double[Q->nDofsI];
    Qab = new double[Q->nDofsI];
    WtQaa = Alloc2D(W->nDofsJ, Q->nDofsJ);
    WtQab = Alloc2D(W->nDofsJ, Q->nDofsJ);
    WtQU = Alloc2D(W->nDofsJ, U->nDofsJ);
    WtQV = Alloc2D(W->nDofsJ, V->nDofsJ);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n1l, topo->nDofs2G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
}

void WtQdUdz_mat::assemble(Vec u1, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_2;
    double det, **J, ux[2];
    PetscScalar *u1Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);
                // vertical scalings cancel between piecewise constant velocity and jacobian

                // TODO: make sure that the metric terms are applied to the degrees of freedom
                // as being inner oriented
                // vorticity = [a,b,c], velocity = [u,v,w]
                // metric term: [v, u][ dy/db, -dy/da][dv/dz]
                //                    [-dx/db,  dx/da][du/dz]
                // +J^{-T}.v.a
                //Qab[ii][ii] = (-ux[1]*J[1][1] + ux[0]*J[0][1])*Q->A[ii][ii]*(scale/det/det);
//                Qab[ii][ii] = (-ux[1]*J[1][1] + ux[0]*J[0][1])*Q->A[ii][ii]*(scale/det);
                // -J^{-T}.u.b
                //Qaa[ii][ii] = (+ux[1]*J[1][0] - ux[0]*J[0][0])*Q->A[ii][ii]*(scale/det/det);
//                Qaa[ii][ii] = (+ux[1]*J[1][0] - ux[0]*J[0][0])*Q->A[ii][ii]*(scale/det);
                // vertical rescaling of jacobian determinant cancels with scaling of
                // the H(curl) test function

                Qaa[ii] = (ux[0]*J[0][0] + ux[1]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = (ux[0]*J[0][1] + ux[1]*J[1][1])*Q->A[ii]*(scale/det);
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qaa, WtQaa);
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Qab, WtQab);

            Mult_IP(W->nDofsJ, U->nDofsJ, U->nDofsI, WtQaa, U->A, WtQU);
            Mult_IP(W->nDofsJ, V->nDofsJ, V->nDofsI, WtQab, V->A, WtQV);

            //Flat2D_IP(W->nDofsJ, U->nDofsJ, WtQU, WtQUflat);
            //Flat2D_IP(W->nDofsJ, V->nDofsJ, WtQV, WtQVflat);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_2 = topo->elInds2_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, U->nDofsJ, inds_x, WtQU, ADD_VALUES);
            MatSetValues(M, W->nDofsJ, inds_2, V->nDofsJ, inds_y, WtQV, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

WtQdUdz_mat::~WtQdUdz_mat() {
    Free2D(W->nDofsJ, Wt);
    delete[] Qaa;
    delete[] Qab;
    Free2D(W->nDofsJ, WtQU);
    Free2D(W->nDofsJ, WtQV);
    Free2D(W->nDofsJ, WtQaa);
    Free2D(W->nDofsJ, WtQab);
    delete U;
    delete V;
    delete W;
    delete Q;
    MatDestroy(&M);
}

//
EoSvec::EoSvec(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    Q = new Wii(e->l->q, geom);
    W = new M2_j_xy_i(e);

    Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &vg);
}

void EoSvec::assemble(Vec rt, int lev, double scale) {
    int ex, ey, mp1, mp12, ii, jj, *inds2, *inds0;
    double p, rtq[99], fac;
    PetscScalar *rtArray, *vArray;

    VecZeroEntries(vg);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    fac = CP*pow(RD/P0, RD/CV);

    VecGetArray(rt, &rtArray);
    VecGetArray(vg, &vArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q->A, WtQ);

            inds0 = geom->elInds0_l(ex, ey);
            inds2 = topo->elInds2_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rtArray, &p);
                // density is piecewise constant in the vertical
                p *= geom->thickInv[lev][inds0[ii]];
                rtq[ii] = fac*pow(p, RD/CV);
                rtq[ii] *= scale;
            }

            // 2 form metric term cancels with jacobian determinant at quadrature point
            for(ii = 0; ii < W->nDofsJ; ii++) {
                for(jj = 0; jj < Q->nDofsI; jj++) {
                    vArray[inds2[ii]] += WtQ[ii*Q->nDofsI+jj]*rtq[jj];
                }
            }
        }
    }
    VecRestoreArray(rt, &rtArray);
    VecRestoreArray(vg, &vArray);
}

EoSvec::~EoSvec() {
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);

    delete W;
    delete Q;

    VecDestroy(&vl);
    VecDestroy(&vg);
}

// assemble the derivative of the equation of state into a volume
// form matrix (for use in the Theta preconditioner operator)
EoSmat::EoSmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;

    Q = new Wii(e->l->q, geom);
    W = new M2_j_xy_i(e);

    Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    Qaa = new double[Q->nDofsI];
    WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);
}

void EoSmat::assemble(Vec rt, int lev, double scale) {
    int ex, ey, ei, mp1, mp12, ii, *inds2, *inds0;
    double p, rtq, fac, det;
    PetscScalar *rtArray;

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    fac = CP*pow(RD/P0, RD/CV);
    fac *= RD/CV;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatZeroEntries(M);

    VecGetArray(rt, &rtArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            inds0 = geom->elInds0_l(ex, ey);
            inds2 = topo->elInds2_g(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rtArray, &p);
                // density is piecewise constant in the vertical
                p *= geom->thickInv[lev][inds0[ii]];
                rtq = fac*pow(p, RD/CV-1.0);
                rtq *= scale;

                Qaa[ii] = rtq * Q->A[ii] / (det * geom->thick[lev][inds0[ii]]);
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            MatSetValues(M, W->nDofsJ, inds2, W->nDofsJ, inds2, WtQW, ADD_VALUES);
        }
    }
    VecRestoreArray(rt, &rtArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

EoSmat::~EoSmat() {
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] Qaa;

    delete W;
    delete Q;

    MatDestroy(&M);
}

// 2 form mass matrix inverse
WmatInv::WmatInv(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
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

void WmatInv::assemble(int lev, double scale) {
    int ex, ey, ei, mp1, mp12, ii, *inds, *inds0;
    double det;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);

    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii]  = Q->A[ii]*(scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            Inv(WtQW, WtQWinv, W->nDofsJ);

            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQWinv, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    delete W;
    delete Q;
}

WmatInv::~WmatInv() {
    MatDestroy(&M);
}

//
WhmatInv::WhmatInv(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
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

void WhmatInv::assemble(Vec rho, int lev, double scale) {
    int ex, ey, ei, mp1, mp12, ii, *inds, *inds0;
    double det, p;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    PetscScalar* pArray;

    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecGetArray(rho, &pArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];

                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, pArray, &p);
                // density is piecewise constant in the vertical
                p *= geom->thickInv[lev][inds0[ii]];

                Qaa[ii]  = p*Q->A[ii]*(scale/det);
                // W is piecewise constant in the vertical
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            Inv(WtQW, WtQWinv, W->nDofsJ);

            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQWinv, ADD_VALUES);
        }
    }
    VecRestoreArray(rho, &pArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    delete W;
    delete Q;
}

WhmatInv::~WhmatInv() {
    MatDestroy(&M);
}

//
N_rt_Inv::N_rt_Inv(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
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

void N_rt_Inv::assemble(Vec rho, int lev, double scale, bool do_inverse) {
    int ex, ey, ei, mp1, mp12, ii, jj, *inds, *inds0;
    double det, p;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* BinvB = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* BBinvB = Alloc2D(W->nDofsJ, W->nDofsJ);
    PetscScalar* pArray;

    MatZeroEntries(M);

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecGetArray(rho, &pArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds0 = geom->elInds0_l(ex, ey);

            // 
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, pArray, &p);
                p *= geom->thickInv[lev][inds0[ii]];
                Qaa[ii]  = p*Q->A[ii]*(scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Inv(WtQW, WtQWinv, W->nDofsJ);

            // 
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii]  = Q->A[ii]*(scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            //
            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQWinv, WtQW, BinvB);
            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW, BinvB, BBinvB);
            if(do_inverse) {
                for(ii = 0; ii < W->nDofsJ; ii++) for(jj = 0; jj < W->nDofsJ; jj++) WtQWinv[ii*W->nDofsJ+jj] = 0.0;
                Inv(BBinvB, WtQWinv, W->nDofsJ);
                MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQWinv, ADD_VALUES);
            } else {
                MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, BBinvB, ADD_VALUES);
            }
        }
    }
    VecRestoreArray(rho, &pArray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    Free2D(W->nDofsJ, BinvB);
    Free2D(W->nDofsJ, BBinvB);
    delete W;
    delete Q;
}

N_rt_Inv::~N_rt_Inv() {
    MatDestroy(&M);
}

//
PtQUt_mat::PtQUt_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);
    Q = new Wii(l->q, geom);
    Qaa = new double[Q->nDofsI];
    Qab = new double[Q->nDofsI];
    QU = Alloc2D(Q->nDofsJ, U->nDofsJ);
    QV = Alloc2D(Q->nDofsJ, V->nDofsJ);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n1l, topo->nDofs0G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    //MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
}

void PtQUt_mat::assemble(Vec u1, int lev, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    double **J, ux[2];
    PetscScalar *u1Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);
                // horiztontal velocity is piecewise constant in the vertical
                ux[0] *= geom->thickInv[lev][inds_0[ii]];
                ux[1] *= geom->thickInv[lev][inds_0[ii]];

                Qaa[ii] = (-ux[1]*J[0][0] + ux[0]*J[1][0])*Q->A[ii]*scale;
                Qab[ii] = (-ux[1]*J[0][1] + ux[0]*J[1][1])*Q->A[ii]*scale;

                // rescale by the inverse of the vertical determinant (piecewise 
                // constant in the vertical)
                Qaa[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
            }

            Mult_DF_IP(Q->nDofsJ, U->nDofsJ, U->nDofsI, Qaa, U->A, QU);
            Mult_DF_IP(Q->nDofsJ, V->nDofsJ, V->nDofsI, Qab, V->A, QV);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_0 = topo->elInds0_g(ex, ey);

            MatSetValues(M, Q->nDofsJ, inds_0, U->nDofsJ, inds_x, QU, ADD_VALUES);
            MatSetValues(M, Q->nDofsJ, inds_0, V->nDofsJ, inds_y, QV, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

PtQUt_mat::~PtQUt_mat() {
    delete[] Qaa;
    delete[] Qab;
    Free2D(Q->nDofsJ, QU);
    Free2D(Q->nDofsJ, QV);
    delete U;
    delete V;
    delete Q;
    MatDestroy(&M);
}

// 
PtQUmat::PtQUmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);
    Q = new Wii(l->q, geom);
    Qaa = new double[Q->nDofsI];
    Qab = new double[Q->nDofsI];
    QU = Alloc2D(Q->nDofsJ, U->nDofsJ);
    QV = Alloc2D(Q->nDofsJ, V->nDofsJ);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0l, topo->n1l, topo->nDofs0G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    //MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatMPIAIJSetPreallocation(M, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);
}

void PtQUmat::assemble(Vec u1, int lev, double scale) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    double **J, ux[2];
    PetscScalar *u1Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(u1, &u1Array);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                J = geom->J[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, u1Array, ux);
                // horiztontal velocity is piecewise constant in the vertical
                ux[0] *= geom->thickInv[lev][inds_0[ii]];
                ux[1] *= geom->thickInv[lev][inds_0[ii]];

                Qaa[ii] = 0.5*(ux[0]*J[0][0] + ux[1]*J[1][0])*Q->A[ii]*scale;
                Qab[ii] = 0.5*(ux[0]*J[0][1] + ux[1]*J[1][1])*Q->A[ii]*scale;

                // rescale by the inverse of the vertical determinant (piecewise 
                // constant in the vertical)
                Qaa[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
            }

            Mult_DF_IP(Q->nDofsJ, U->nDofsJ, U->nDofsI, Qaa, U->A, QU);
            Mult_DF_IP(Q->nDofsJ, V->nDofsJ, V->nDofsI, Qab, V->A, QV);

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            inds_0 = topo->elInds0_g(ex, ey);

            MatSetValues(M, Q->nDofsJ, inds_0, U->nDofsJ, inds_x, QU, ADD_VALUES);
            MatSetValues(M, Q->nDofsJ, inds_0, V->nDofsJ, inds_y, QV, ADD_VALUES);
        }
    }
    VecRestoreArray(u1, &u1Array);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}

PtQUmat::~PtQUmat() {
    delete[] Qaa;
    delete[] Qab;
    Free2D(Q->nDofsJ, QU);
    Free2D(Q->nDofsJ, QV);
    delete U;
    delete V;
    delete Q;
    MatDestroy(&M);
}

WtQPmat::WtQPmat(Topo* _topo, Geom* _geom, LagrangeNode* _node, LagrangeEdge* _edge) {
    topo = _topo;
    geom = _geom;
    node = _node;
    edge = _edge;

    M2_j_xy_i* W = new M2_j_xy_i(edge);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2l, topo->n0l, topo->nDofs2G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);

    delete W;
}

void WtQPmat::assemble(int lev, double scale) {
    int ex, ey, mp1, mp12, ii, *inds, *inds0;
    Wii* Q = new Wii(node->q, geom);
    M0_j_xy_i* P = new M0_j_xy_i(node);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQP = Alloc2D(W->nDofsJ, P->nDofsJ);

    MatZeroEntries(M);

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);

            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                Qaa[ii]  = Q->A[ii]*scale;
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            inds0 = topo->elInds0_g(ex, ey);

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, P->nDofsJ, Wt, Qaa, WtQP);
            MatSetValues(M, W->nDofsJ, inds, P->nDofsJ, inds0, WtQP, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQP);
    delete W;
    delete P;
    delete Q;
}

WtQPmat::~WtQPmat() {
    MatDestroy(&M);
}

N_RTmat::N_RTmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
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

void N_RTmat::assemble(int lev, double scale, Vec rt, Vec pi) {
    int ex, ey, ei, n2, mp1, mp12, ii, jj, *inds2_l, *inds2_g, *inds0;
    double det, rt_i, pi_i, rt_q[99];
    double fac = (P0/RD);
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQW_2 = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQW_3 = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    PetscScalar *rtArray, *piArray;

    MatZeroEntries(M);

    n2   = W->nDofsJ;
    mp1  = e->l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecGetArray(rt, &rtArray);
    VecGetArray(pi, &piArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0   = geom->elInds0_l(ex, ey);
            inds2_l = topo->elInds2_l(ex, ey);
            inds2_g = topo->elInds2_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                rt_i = pi_i = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    rt_i += W->A[ii*n2+jj]*rtArray[inds2_l[jj]];
                    pi_i += W->A[ii*n2+jj]*piArray[inds2_l[jj]];
                }
                det = geom->det[ei][ii];
                rt_i *= 1.0/(det*geom->thick[lev][inds0[ii]]);
                pi_i *= 1.0/(det*geom->thick[lev][inds0[ii]]);

                rt_q[ii] = rt_i*rt_i;

                Qaa[ii]  = pow(pi_i/CP, CV/CP) * Q->A[ii] * (scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW_2); // (pi/cp)^{c_v/R}

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii]  = rt_q[ii] * Q->A[ii]*(scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Inv(WtQW, WtQWinv, n2);                                      // rt^{-2}

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii]  = Q->A[ii]*(scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);   // M_2

            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW, WtQWinv, WtQW_3);
            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW_3, WtQW_2, WtQW);

            for(ii = 0; ii < n2*n2; ii++) WtQW[ii] *= fac;

            MatSetValues(M, W->nDofsJ, inds2_g, W->nDofsJ, inds2_g, WtQW, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    VecRestoreArray(rt, &rtArray);
    VecRestoreArray(pi, &piArray);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQW_2);
    Free2D(W->nDofsJ, WtQW_3);
    Free2D(W->nDofsJ, WtQWinv);
    delete W;
    delete Q;
}

N_RTmat::~N_RTmat() {
    MatDestroy(&M);
}

N_PiInv_mat::N_PiInv_mat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
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

void N_PiInv_mat::assemble(int lev, double scale, Vec rt, Vec pi) {
    int ex, ey, ei, n2, mp1, mp12, ii, jj, *inds2_l, *inds2_g, *inds0;
    double det, rt_i, pi_i, rt_q[99];
    double fac = -1.0*(RD/P0)*(RD/CV)*pow(CP, CV/RD);
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQW_2 = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQW_3 = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    PetscScalar *rtArray, *piArray;

    MatZeroEntries(M);

    n2   = W->nDofsJ;
    mp1  = e->l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecGetArray(rt, &rtArray);
    VecGetArray(pi, &piArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0   = geom->elInds0_l(ex, ey);
            inds2_l = topo->elInds2_l(ex, ey);
            inds2_g = topo->elInds2_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                rt_i = pi_i = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    rt_i += W->A[ii*n2+jj]*rtArray[inds2_l[jj]];
                    pi_i += W->A[ii*n2+jj]*piArray[inds2_l[jj]];
                }
                det = geom->det[ei][ii];
                rt_i *= 1.0/(det*geom->thick[lev][inds0[ii]]);
                pi_i *= 1.0/(det*geom->thick[lev][inds0[ii]]);

                rt_q[ii] = rt_i;

                Qaa[ii]  = pow(pi_i, (CV-RD)/RD) * Q->A[ii] * (scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW_2); // pi^{(c_v-R)/R}

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii]  = rt_q[ii] * Q->A[ii]*(scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Inv(WtQW, WtQWinv, n2);                                      // rt^{-1}

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii]  = Q->A[ii]*(scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);   // M_2

            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW, WtQWinv, WtQW_3);
            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW_3, WtQW_2, WtQW);
            Inv(WtQW, WtQWinv, n2);

            for(ii = 0; ii < n2*n2; ii++) WtQWinv[ii] *= fac;

            MatSetValues(M, W->nDofsJ, inds2_g, W->nDofsJ, inds2_g, WtQWinv, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    VecRestoreArray(rt, &rtArray);
    VecRestoreArray(pi, &piArray);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQW_2);
    Free2D(W->nDofsJ, WtQW_3);
    Free2D(W->nDofsJ, WtQWinv);
    delete W;
    delete Q;
}

N_PiInv_mat::~N_PiInv_mat() {
    MatDestroy(&M);
}

N_RT2_mat::N_RT2_mat(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
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

void N_RT2_mat::assemble(int lev, double scale, Vec rt) {
    int ex, ey, ei, n2, mp1, mp12, ii, jj, *inds2_l, *inds2_g, *inds0;
    double det, rt_i;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    PetscScalar *rtArray;

    MatZeroEntries(M);

    n2   = W->nDofsJ;
    mp1  = e->l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecGetArray(rt, &rtArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0   = geom->elInds0_l(ex, ey);
            inds2_l = topo->elInds2_l(ex, ey);
            inds2_g = topo->elInds2_g(ex, ey);

            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                rt_i = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    rt_i += W->A[ii*n2+jj]*rtArray[inds2_l[jj]];
                }
                det = geom->det[ei][ii];
                rt_i *= 1.0/(det*geom->thick[lev][inds0[ii]]);

                Qaa[ii]  = rt_i * rt_i * Q->A[ii] * (scale/det);
                Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
            }
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

            Inv(WtQW, WtQWinv, n2);

            MatSetValues(M, W->nDofsJ, inds2_g, W->nDofsJ, inds2_g, WtQWinv, ADD_VALUES);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    VecRestoreArray(rt, &rtArray);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    delete W;
    delete Q;
}

N_RT2_mat::~N_RT2_mat() {
    MatDestroy(&M);
}

double _compute_sigma(double phi, double z) {
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

double compute_k_v(double exner, double exner_s) {
    double p        = pow(exner/CP, CP/RD);
    double ps       = pow(exner_s/CP, CP/RD);
    double sigma    = p/ps;
    double sigma_b  = 0.7;
    double k_f      = 1.1574074074074073e-05;

    if(sigma < sigma_b) return 0.0;

    return k_f*(sigma - sigma_b)/(1.0 - sigma_b);
}

Umat_ray::Umat_ray(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
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

#define K_F 1.1574074074074073e-05

void Umat_ray::assemble(int lev, double scale, double dt, Vec exner_k, Vec exner_s) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    Wii* Q = new Wii(l->q, geom);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double det, **J, k_v;
    double* Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    double* Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    double* UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* VtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* VtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* Qaa = new double[Q->nDofsI];
    double* Qab = new double[Q->nDofsI];
    double* Qbb = new double[Q->nDofsI];
    double _e, _es;
    PetscScalar *eArray, *esArray;

    MatZeroEntries(M);

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    VecGetArray(exner_k, &eArray);
    VecGetArray(exner_s, &esArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                Qbb[ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                //sigma_b = _compute_sigma(geom->s[inds_0[ii]][1], geom->levs[lev+0][inds_0[ii]]);
                //sigma_t = _compute_sigma(geom->s[inds_0[ii]][1], geom->levs[lev+1][inds_0[ii]]);
                //k_v = 0.5*dt*K_F*(sigma_b + sigma_t);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, eArray, &_e);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, esArray, &_es);
                _e *= geom->thickInv[lev][inds_0[ii]];
                _es *= geom->thickInv[0][inds_0[ii]];
                k_v = compute_k_v(_e, _es);
                k_v *= dt;

                Qaa[ii] *= k_v*geom->thickInv[lev][inds_0[ii]];
                Qab[ii] *= k_v*geom->thickInv[lev][inds_0[ii]];
                Qbb[ii] *= k_v*geom->thickInv[lev][inds_0[ii]];
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

            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQU, ADD_VALUES);
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQV, ADD_VALUES);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, VtQU, ADD_VALUES);
            MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_y, VtQV, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    VecRestoreArray(exner_k, &eArray);
    VecRestoreArray(exner_s, &esArray);

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
    delete[] Qaa;
    delete[] Qab;
    delete[] Qbb;
    delete Q;
    delete U;
    delete V;
}

Umat_ray::~Umat_ray() {
    MatDestroy(&M);
}

//////////////////////////////////////////////////////////////
Uvec::Uvec(Topo* _topo, Geom* _geom, LagrangeNode* _node, LagrangeEdge* _edge) {
    topo = _topo;
    geom = _geom;
    node = _node;
    edge = _edge;

    Q = new Wii(node->q, geom);
    U = new M1x_j_xy_i(node, edge);
    V = new M1y_j_xy_i(node, edge);
    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(U->nDofsJ, U->nDofsI);

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &vg);
}

void Uvec::assemble(int lev, double scale, bool vert_scale, Vec vel) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    double det, **J;
    PetscScalar *vArray, *velArray;
    double _u[2], Qaa[99], Qab[99], Qba[99], Qbb[99], rhs[99];

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    VecZeroEntries(vl);
    VecZeroEntries(vg);

    VecGetArray(vl, &vArray);
    VecGetArray(vel, &velArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                Qbb[ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                // horiztonal velocity is piecewise constant in the vertical
                Qaa[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qbb[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qba[ii]  = Qab[ii];

                // multiply by the velocity vector
                geom->interp1_l(ex, ey, ii%mp1, ii/mp1, velArray, _u);

                Qaa[ii] *= _u[0];
                Qba[ii] *= _u[0];
                Qab[ii] *= _u[1];
                Qbb[ii] *= _u[1];
            }

            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);

            Ax_b(U->nDofsJ, Q->nDofsI, Ut, Qaa, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_x[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Ut, Qab, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_x[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Vt, Qba, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_y[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Vt, Qbb, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_y[ii]] += rhs[ii];
            }
        }
    }
    VecRestoreArray(vl, &vArray);
    VecRestoreArray(vel, &velArray);

    VecScatterBegin(topo->gtol_1, vl, vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, vl, vg, ADD_VALUES, SCATTER_REVERSE);
}

void Uvec::assemble_hu(int lev, double scale, Vec vel, Vec rho, bool zero_and_scatter, double fac) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    double det, **J;
    PetscScalar *vArray, *velArray, *rhoArray;
    double _u[2], _r, Qaa[99], Qab[99], Qba[99], Qbb[99], rhs[99];

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    if(zero_and_scatter) {
        VecZeroEntries(vl);
        VecZeroEntries(vg);
    }

    VecGetArray(vl, &vArray);
    VecGetArray(vel, &velArray);
    VecGetArray(rho, &rhoArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                Qbb[ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                // horiztonal velocity is piecewise constant in the vertical
                Qaa[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qab[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qbb[ii] *= geom->thickInv[lev][inds_0[ii]];
                Qba[ii]  = Qab[ii];

                // multiply by the velocity vector
                geom->interp1_l(ex, ey, ii%mp1, ii/mp1, velArray, _u);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rhoArray, &_r);
                _r *= geom->thickInv[lev][inds_0[ii]];
                _r *= fac;

                Qaa[ii] *= (_u[0] * _r);
                Qba[ii] *= (_u[0] * _r);
                Qab[ii] *= (_u[1] * _r);
                Qbb[ii] *= (_u[1] * _r);
            }

            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);

            Ax_b(U->nDofsJ, Q->nDofsI, Ut, Qaa, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_x[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Ut, Qab, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_x[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Vt, Qba, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_y[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Vt, Qbb, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_y[ii]] += rhs[ii];
            }
        }
    }
    VecRestoreArray(vl, &vArray);
    VecRestoreArray(vel, &velArray);
    VecRestoreArray(rho, &rhoArray);

    if(zero_and_scatter) {
        VecScatterBegin(topo->gtol_1, vl, vg, ADD_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  topo->gtol_1, vl, vg, ADD_VALUES, SCATTER_REVERSE);
    }
}

void Uvec::assemble_wxu(int lev, double scale, Vec vel, Vec vort) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    double det, **J;
    double _u[2], _q, Qab[99], Qba[99], rhs[99];
    PetscScalar *vortArray, *velArray, *vArray;

    mp1 = node->n + 1;
    mp12 = mp1*mp1;

    VecZeroEntries(vl);
    VecZeroEntries(vg);

    VecGetArray(vort, &vortArray);
    VecGetArray(vel, &velArray);
    VecGetArray(vl, &vArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp0(ex, ey, ii%mp1, ii/mp1, vortArray, &_q);
                _q *= geom->thickInv[lev][inds_0[ii]];
                geom->interp1_l(ex, ey, ii%mp1, ii/mp1, velArray, _u);

                Qab[ii] = (-J[0][0]*J[1][1] + J[0][1]*J[1][0])*Q->A[ii]*(scale/det);
                Qba[ii] = (+J[0][0]*J[1][1] - J[0][1]*J[1][0])*Q->A[ii]*(scale/det);

                Qab[ii] *= (_q*_u[1])*geom->thickInv[lev][inds_0[ii]];
                Qba[ii] *= (_q*_u[0])*geom->thickInv[lev][inds_0[ii]];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Ut, Qab, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_x[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Vt, Qba, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                vArray[inds_y[ii]] += rhs[ii];
            }
        }
    }
    VecRestoreArray(vort, &vortArray);
    VecRestoreArray(vel, &velArray);
    VecRestoreArray(vl, &vArray);

    VecScatterBegin(topo->gtol_1, vl, vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, vl, vg, ADD_VALUES, SCATTER_REVERSE);
}

Uvec::~Uvec() {
    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);

    delete U;
    delete V;
    delete Q;

    VecDestroy(&vl);
    VecDestroy(&vg);
}

Wvec::Wvec(Topo* _topo, Geom* _geom, LagrangeEdge* _edge) {
    topo = _topo;
    geom = _geom;
    edge = _edge;

    Q = new Wii(edge->l->q, geom);
    W = new M2_j_xy_i(edge);

    Wt = Alloc2D(W->nDofsJ, W->nDofsI);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &vg);
}

void Wvec::assemble(int lev, double scale, bool vert_scale, Vec rho) {
    int ex, ey, ei, mp1, mp12, ii, *inds, *inds0;
    double det, Qaa[99], rhs[99], _r;
    PetscScalar *vArray, *rhoArray;

    mp1 = edge->l->q->n + 1;
    mp12 = mp1*mp1;

    VecZeroEntries(vg);

    VecGetArray(vg, &vArray);
    VecGetArray(rho, &rhoArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_l(ex, ey);

            ei = ey*topo->nElsX + ex;
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Qaa[ii]  = Q->A[ii]*(scale/det);
                if(vert_scale) {
                    Qaa[ii] *= geom->thickInv[lev][inds0[ii]];
                }
                geom->interp2_l(ex, ey, ii%mp1, ii/mp1, rhoArray, &_r);
                Qaa[ii] *= _r;
            }

            Ax_b(W->nDofsJ, Q->nDofsI, Wt, Qaa, rhs);
            for(ii = 0; ii < W->nDofsJ; ii++) {
                vArray[inds[ii]] += rhs[ii];
            }
        }
    }

    VecRestoreArray(vg, &vArray);
    VecRestoreArray(rho, &rhoArray);
}

void Wvec::assemble_K(int lev, double scale, Vec vel1, Vec vel2) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_2, *inds_0;
    double det, **J, _uxg[2], _uxl[2], Qaa[99], Qab[99], rhs_a[99], rhs_b[99];
    PetscScalar *vArray, *vel1Array, *vel2Array;

    mp1 = edge->l->n + 1;
    mp12 = mp1*mp1;

    VecZeroEntries(vg);

    VecGetArray(vg, &vArray);
    VecGetArray(vel1, &vel1Array);
    VecGetArray(vel2, &vel2Array);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = geom->elInds0_l(ex, ey);
            inds_2 = topo->elInds2_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];
                geom->interp1_l(ex, ey, ii%mp1, ii/mp1, vel1Array, _uxl);
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, vel2Array, _uxg);
                // horiztontal velocity is piecewise constant in the vertical
                _uxg[0] *= geom->thickInv[lev][inds_0[ii]];
                _uxg[1] *= geom->thickInv[lev][inds_0[ii]];

                Qaa[ii] = 0.5*(_uxg[0]*J[0][0] + _uxg[1]*J[1][0])*Q->A[ii]*(scale/det);
                Qab[ii] = 0.5*(_uxg[0]*J[0][1] + _uxg[1]*J[1][1])*Q->A[ii]*(scale/det);

                // rescale by the inverse of the vertical determinant (piecewise 
                // constant in the vertical)
                Qaa[ii] *= (_uxl[0]*geom->thickInv[lev][inds_0[ii]]);
                Qab[ii] *= (_uxl[1]*geom->thickInv[lev][inds_0[ii]]);
            }

            Ax_b(W->nDofsJ, Q->nDofsI, Wt, Qaa, rhs_a);
            Ax_b(W->nDofsJ, Q->nDofsI, Wt, Qab, rhs_b);
            for(ii = 0; ii < W->nDofsJ; ii++) {
                vArray[inds_2[ii]] += (rhs_a[ii] + rhs_b[ii]);
            }
        }
    }
    VecRestoreArray(vg, &vArray);
    VecRestoreArray(vel1, &vel1Array);
    VecRestoreArray(vel2, &vel2Array);
}

Wvec::~Wvec() {
    Free2D(W->nDofsJ, Wt);
    delete W;
    delete Q;
    VecDestroy(&vg);
}
