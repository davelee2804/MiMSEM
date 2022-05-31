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
#include "Assembly_Coupled.h"

#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0

#define SCALE 1.0e+8

using namespace std;

// mass matrix for the 1 form vector (x-normal degrees of
// freedom first then y-normal degrees of freedom)
Umat_coupled::Umat_coupled(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;
}

void Umat_coupled::assemble(double scale, Mat M) {
    int ex, ey, ei, kk, ii, mp1, mp12;
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

    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    for(kk = 0; kk < geom->nk; kk++) {
        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                ei = ey*topo->nElsX + ex;
                inds_0 = topo->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    J = geom->J[ei][ii];

                    Qaa[ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                    Qab[ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                    Qbb[ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                    // horiztonal velocity is piecewise constant in the vertical
                    Qaa[ii] *= geom->thickInv[kk][inds_0[ii]];
                    Qab[ii] *= geom->thickInv[kk][inds_0[ii]];
                    Qbb[ii] *= geom->thickInv[kk][inds_0[ii]];
                }

                inds_x = topo->elInds_velx_g(ex, ey, kk);
                inds_y = topo->elInds_vely_g(ex, ey, kk);

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
    }

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

Umat_coupled::~Umat_coupled() {
}

// 1 form mass matrix with 0 form interpolated to quadrature points
// (for rotational term in the momentum equation)
RotMat_coupled::RotMat_coupled(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    Q = new Wii(l->q, geom);
    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);

    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(V->nDofsJ, V->nDofsI);
    Qab = new double[Q->nDofsI];
    Qba = new double[Q->nDofsI];
    UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQU = Alloc2D(V->nDofsJ, V->nDofsJ);
}

void RotMat_coupled::assemble(double scale, double fac, Vec* q0, Mat M) {
    int ex, ey, ei, ii, kk, mp1, mp12;
    int *inds_x, *inds_y, *inds_0;
    double det, **J, vort;
    PetscScalar* q0Array;

    mp1 = l->n + 1;
    mp12 = mp1*mp1;

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, V->nDofsJ, V->A, Vt);

    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(q0[kk], &q0Array);
        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                inds_x = topo->elInds_velx_g(ex, ey, kk);
                inds_y = topo->elInds_vely_g(ex, ey, kk);

                ei = ey*topo->nElsX + ex;
                inds_0 = topo->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    J = geom->J[ei][ii];
                    geom->interp0(ex, ey, ii%mp1, ii/mp1, q0Array, &vort);

                    // vertical vorticity is piecewise constant in the vertical
                    vort *= fac*geom->thickInv[kk][inds_0[ii]];

                    Qab[ii] = vort*(-J[0][0]*J[1][1] + J[0][1]*J[1][0])*Q->A[ii]*(scale/det);
                    Qba[ii] = vort*(+J[0][0]*J[1][1] - J[0][1]*J[1][0])*Q->A[ii]*(scale/det);

                    Qab[ii] *= geom->thickInv[kk][inds_0[ii]];
                    Qba[ii] *= geom->thickInv[kk][inds_0[ii]];
                }

                Mult_FD_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Qab, UtQab);
                Mult_FD_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Qba, VtQba);

                // take cross product by multiplying the x projection of the row vector with
                // the y component of the column vector and vice versa
                Mult_IP(U->nDofsJ, U->nDofsJ, U->nDofsI, UtQab, V->A, UtQV);
                Mult_IP(U->nDofsJ, U->nDofsJ, V->nDofsI, VtQba, U->A, VtQU);

                MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_y, UtQV, ADD_VALUES);
                MatSetValues(M, U->nDofsJ, inds_y, U->nDofsJ, inds_x, VtQU, ADD_VALUES);
            }
        }
        VecRestoreArray(q0[kk], &q0Array);
    }
}

RotMat_coupled::~RotMat_coupled() {
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
}

// 2 form mass matrix
Wmat_coupled::Wmat_coupled(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    topo = _topo;
    geom = _geom;
    e = _e;
}

// var_ind = 0: rho
// var_ind = 1: theta
// var_ind = 2: exner
// var_ind = 3: velz
void Wmat_coupled::assemble(double scale, int var_ind, Mat M) {
    int ex, ey, ei, mp1, mp12, ii, kk, *inds, *inds0, num_levs;
    double det;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    bool vert_scale;

    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;
    num_levs   = (var_ind == 3) ? geom->nk-1 : geom->nk;
    vert_scale = (var_ind == 3) ? false : true;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(kk = 0; kk < num_levs; kk++) {
        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
		if(var_ind == 0) {
                    inds = topo->elInds_rho_g(ex, ey, kk);
		} else if(var_ind == 1) {
                    inds = topo->elInds_theta_g(ex, ey, kk);
		} else if(var_ind == 2) {
                    inds = topo->elInds_exner_g(ex, ey, kk);
		} else {
                    inds = topo->elInds_velz_g(ex, ey, kk);
		}
                ei = ey*topo->nElsX + ex;
                inds0 = topo->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Qaa[ii]  = Q->A[ii]*(scale/det);
                    if(vert_scale) {
                        Qaa[ii] *= geom->thickInv[kk][inds0[ii]];
                    }
                }

                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
                Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

                MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQW, ADD_VALUES);
            }
        }
    }

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete W;
    delete Q;
}

Wmat_coupled::~Wmat_coupled() {
}

// 2 form mass matrix
EoSmat_coupled::EoSmat_coupled(Topo* _topo, Geom* _geom, LagrangeEdge* _edge) {
    topo = _topo;
    geom = _geom;
    edge = _edge;

    Q       = new Wii(edge->l->q, geom);
    W       = new M2_j_xy_i(edge);
    Wt      = new double[W->nDofsI*W->nDofsJ];
    WtQ     = new double[W->nDofsI*W->nDofsJ];
    WtQW    = new double[W->nDofsJ*W->nDofsJ];
    WtQWinv = new double[W->nDofsJ*W->nDofsJ];
    AAinv   = new double[W->nDofsJ*W->nDofsJ];
    AAinvA  = new double[W->nDofsJ*W->nDofsJ];
}

// Note: p2 is a vertical vector
void EoSmat_coupled::assemble(double scale, double fac, int col_ind, Vec* p2, Mat M) {
    int ii, jj, kk, ex, ey, ei, mp1, mp12, n2;
    int *inds0, *inds_row, *inds_col;
    double det, val, QA[99], QB[99];
    PetscScalar* pArray;

    n2    = topo->elOrd*topo->elOrd;
    mp1   = edge->l->q->n + 1;
    mp12  = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei    = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);

            VecGetArray(p2[ei], &pArray);
            for(kk = 0; kk < geom->nk; kk++) {
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    QA[ii]  = Q->A[ii]*(scale/det);
                    QA[ii] *= geom->thickInv[kk][inds0[ii]];

                    val = 0.0;
                    for(jj = 0; jj < n2; jj++) {
                        val += pArray[kk*n2+jj]*W->A[ii*n2+jj];
                    }
                    QB[ii] = QA[ii]*val/(fac*geom->thick[kk][inds0[ii]]*det);
                }
                // assemble the piecewise constant mass matrix for level k
                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QB, WtQ);
                Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                Inv(WtQW, WtQWinv, n2);

                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QA, WtQ);
                Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

                Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQW, WtQWinv, AAinv);
                Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, AAinv, WtQW, AAinvA);

                inds_row = topo->elInds_exner_g(ex, ey, kk);
                if(col_ind == 1) {
                    inds_col = topo->elInds_theta_g(ex, ey, kk);
                } else {
                    inds_col = inds_row;
                }
                MatSetValues(M, W->nDofsJ, inds_row, W->nDofsJ, inds_col, AAinvA, ADD_VALUES);
            }
            VecRestoreArray(p2[ei], &pArray);
        }
    }
}

// Note: p2 is a horiztonal vector
void EoSmat_coupled::assemble_rho_inv_mm(double scale, double fac, int lev, Vec p2, Mat M2) {
    int ii, jj, ex, ey, ei, mp1, mp12, n2;
    int *inds0, *inds2;
    double det, val, QA[99], QB[99];
    PetscScalar* pArray;

    MatZeroEntries(M2);

    n2    = topo->elOrd*topo->elOrd;
    mp1   = edge->l->q->n + 1;
    mp12  = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecGetArray(p2, &pArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei    = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);
            inds2 = topo->elInds2_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                QA[ii]  = Q->A[ii]*(scale/det);
                QA[ii] *= geom->thickInv[lev][inds0[ii]];

                val = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    val += pArray[inds2[jj]]*W->A[ii*n2+jj];
                }
                QB[ii] = QA[ii]*val/(fac*geom->thick[lev][inds0[ii]]*det);
            }
            // assemble the piecewise constant mass matrix for level k
            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QB, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Inv(WtQW, WtQWinv, n2);

            Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QA, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Mult_IP(W->nDofsJ, W->nDofsJ, W->nDofsJ, WtQWinv, WtQW, AAinvA);

            inds2 = topo->elInds2_g(ex, ey);
            MatSetValues(M2, W->nDofsJ, inds2, W->nDofsJ, inds2, AAinvA, ADD_VALUES);
        }
    }
    VecRestoreArray(p2, &pArray);

    MatAssemblyBegin(M2, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M2, MAT_FINAL_ASSEMBLY);
}

EoSmat_coupled::~EoSmat_coupled() {
    delete Q;
    delete W;
    delete[] Wt;
    delete[] WtQ;
    delete[] WtQW;
    delete[] WtQWinv;
    delete[] AAinv;
    delete[] AAinvA;
}

void AddGradx_Coupled(Topo* topo, int lev, int var_ind, Mat G, Mat M) {
    int mi, mf, mm, nCols, dof_proc, dof_locl, ri, ci;
    const int *cols;
    const double* vals;
    int cols2[999];
    int dofs_per_lev = 4;

    MatGetOwnershipRange(G, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        dof_proc = mm / topo->n1l;
        dof_locl = mm % topo->n1l;
	ri = dof_proc*topo->dofs_per_proc + lev*topo->n1l + dof_locl;
        MatGetRow(G, mm, &nCols, &cols, &vals);
	for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            dof_locl = cols[ci] % topo->n2l;
            cols2[ci] = dof_proc*topo->dofs_per_proc + topo->nk*topo->n1l + (4*topo->nk-1)*dof_locl + dofs_per_lev*lev + var_ind;
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(G, mm, &nCols, &cols, &vals);
    }
}

void AddDivx_Coupled(Topo* topo, int lev, int var_ind, Mat D, Mat M) {
    int mi, mf, mm, nCols, dof_proc, dof_locl, ri, ci;
    const int *cols;
    const double* vals;
    int cols2[999];
    int dofs_per_lev = 4;

    MatGetOwnershipRange(D, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        dof_proc = mm / topo->n2l;
        dof_locl = mm % topo->n2l;
	ri = dof_proc*topo->dofs_per_proc + topo->nk*topo->n1l + (4*topo->nk-1)*dof_locl + dofs_per_lev*lev + var_ind;
        MatGetRow(D, mm, &nCols, &cols, &vals);
	for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            dof_locl = cols[ci] % topo->n1l;
            cols2[ci] = dof_proc*topo->dofs_per_proc + lev*topo->n1l + dof_locl;
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(D, mm, &nCols, &cols, &vals);
    }
}

void AddQx_Coupled(Topo* topo, int lev, Mat Q, Mat M) {
    int mi, mf, mm, nCols, dof_proc, dof_locl, ri, ci;
    const int *cols;
    const double* vals;
    int cols2[999];
    int dofs_per_lev = 4;

    MatGetOwnershipRange(Q, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        dof_proc = mm / topo->n2l;
        dof_locl = mm % topo->n2l;
        ri = dof_proc*topo->dofs_per_proc + topo->nk*topo->n1l + (4*topo->nk-1)*dof_locl + dofs_per_lev*lev + 1;            // Theta
        MatGetRow(Q, mm, &nCols, &cols, &vals);
	for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            dof_locl = cols[ci] % topo->n2l;
            cols2[ci] = dof_proc*topo->dofs_per_proc + topo->nk*topo->n1l + (4*topo->nk-1)*dof_locl + dofs_per_lev*lev + 0; // rho
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(Q, mm, &nCols, &cols, &vals);
    }
}

void AddGradz_Coupled(Topo* topo, int ex, int ey, int var_ind, Mat G, Mat M) {
    int nr, nc, mm, nCols, ri, ci, lev, fce, n2;
    const int *cols;
    const double* vals;
    int cols2[1999];
    int *inds, shift, dofs_per_col;

    n2 = topo->elOrd*topo->elOrd;
    inds = topo->elInds2_l(ex, ey);
    shift = topo->pi*topo->dofs_per_proc + topo->nk*topo->n1l;
    dofs_per_col = 4*topo->nk-1;

    MatGetSize(G, &nr, &nc);
    for(mm = 0; mm < nr; mm++) {
        MatGetRow(G, mm, &nCols, &cols, &vals);
        lev = mm / n2;
        fce = mm % n2;
        ri = shift + dofs_per_col*inds[fce] + 4*lev + 3;
	for(ci = 0; ci < nCols; ci++) {
            lev = cols[ci] / n2;
            fce = cols[ci] % n2;
            cols2[ci] = shift + dofs_per_col*inds[fce] + 4*lev + var_ind;
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(G, mm, &nCols, &cols, &vals);
    }
}

void AddDivz_Coupled(Topo* topo, int ex, int ey, int var_ind, Mat D, Mat M) {
    int nr, nc, mm, nCols, ri, ci, lev, fce, n2;
    const int *cols;
    const double* vals;
    int cols2[1999];
    int *inds, shift, dofs_per_col;

    n2 = topo->elOrd*topo->elOrd;
    inds = topo->elInds2_l(ex, ey);
    shift = topo->pi*topo->dofs_per_proc + topo->nk*topo->n1l;
    dofs_per_col = 4*topo->nk-1;

    MatGetSize(D, &nr, &nc);
    for(mm = 0; mm < nr; mm++) {
        MatGetRow(D, mm, &nCols, &cols, &vals);
        lev = mm / n2;
        fce = mm % n2;
        ri = shift + dofs_per_col*inds[fce] + 4*lev + var_ind;
	for(ci = 0; ci < nCols; ci++) {
            lev = cols[ci] / n2;
            fce = cols[ci] % n2;
            cols2[ci] = shift + dofs_per_col*inds[fce] + 4*lev + 3;
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(D, mm, &nCols, &cols, &vals);
    }
}

void AddMz_Coupled(Topo* topo, int ex, int ey, int var_ind, Mat Mz, Mat M) {
    int nr, nc, mm, nCols, ri, ci, lev, fce, n2;
    const int *cols;
    const double* vals;
    int cols2[1999];
    int *inds, shift, dofs_per_col;

    n2 = topo->elOrd*topo->elOrd;
    inds = topo->elInds2_l(ex, ey);
    shift = topo->pi*topo->dofs_per_proc + topo->nk*topo->n1l;
    dofs_per_col = 4*topo->nk-1;

    MatGetSize(Mz, &nr, &nc);
    for(mm = 0; mm < nr; mm++) {
        MatGetRow(Mz, mm, &nCols, &cols, &vals);
        lev = mm / n2;
        fce = mm % n2;
        ri = shift + dofs_per_col*inds[fce] + 4*lev + var_ind;
	for(ci = 0; ci < nCols; ci++) {
            lev = cols[ci] / n2;
            fce = cols[ci] % n2;
            cols2[ci] = shift + dofs_per_col*inds[fce] + 4*lev + var_ind;
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(Mz, mm, &nCols, &cols, &vals);
    }
}

void AddQz_Coupled(Topo* topo, int ex, int ey, int row_ind, int col_ind, Mat Q, Mat M) {
    int nr, nc, mm, nCols, ri, ci, lev, fce, n2;
    const int *cols;
    const double* vals;
    int cols2[1999];
    int *inds, shift, dofs_per_col;

    n2 = topo->elOrd*topo->elOrd;
    inds = topo->elInds2_l(ex, ey);
    shift = topo->pi*topo->dofs_per_proc + topo->nk*topo->n1l;
    dofs_per_col = 4*topo->nk-1;

    MatGetSize(Q, &nr, &nc);
    for(mm = 0; mm < nr; mm++) {
        MatGetRow(Q, mm, &nCols, &cols, &vals);
        lev = mm / n2;
        fce = mm % n2;
        ri = shift + dofs_per_col*inds[fce] + 4*lev + row_ind;
	for(ci = 0; ci < nCols; ci++) {
            lev = cols[ci] / n2;
            fce = cols[ci] % n2;
            cols2[ci] = shift + dofs_per_col*inds[fce] + 4*lev + col_ind;
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(Q, mm, &nCols, &cols, &vals);
    }
}

E32_Coupled::E32_Coupled(Topo* _topo) {
    int ex, ey, kk, nn, np1, ii, jj, row;
    int *inds_2, *inds_1x, *inds_1y;
    int cols[6];
    int n_rows_locl, n_cols_locl, rank, size, col_proc, col_dof, nCols;
    double vals[6];

    topo = _topo;
    nn = topo->elOrd;
    np1 = nn + 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    n_rows_locl = topo->nk*topo->n2l;
    n_cols_locl = topo->nk*topo->n1l + (topo->nk-1)*topo->n2l;

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, n_rows_locl, n_cols_locl, size*n_rows_locl, topo->nk*topo->nDofs1G + (topo->nk-1)*topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 6, PETSC_NULL, 6, PETSC_NULL);
    MatZeroEntries(M);
    
    for(kk = 0; kk < topo->nk; kk++) {
        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                inds_1x = topo->elInds1x_g(ex, ey);
                inds_1y = topo->elInds1y_g(ex, ey);
                inds_2 = topo->elInds2_l(ex, ey);

                for(ii = 0; ii < nn; ii++) {
                    for(jj = 0; jj < nn; jj++) {
                        row = rank*n_rows_locl + kk*topo->n2l + inds_2[ii*nn+jj];

			col_proc = inds_1x[ii*np1+jj]/topo->n1l;
			col_dof  = inds_1x[ii*np1+jj]%topo->n1l;
			cols[0]  = col_proc*n_cols_locl + kk*topo->n1l + col_dof;
                        vals[0]  = -1.0;

			col_proc = inds_1x[ii*np1+jj+1]/topo->n1l;
			col_dof  = inds_1x[ii*np1+jj+1]%topo->n1l;
			cols[1]  = col_proc*n_cols_locl + kk*topo->n1l + col_dof;
                        vals[1]  = +1.0;

			col_proc = inds_1y[ii*nn+jj]/topo->n1l;
			col_dof  = inds_1y[ii*nn+jj]%topo->n1l;
			cols[2]  = col_proc*n_cols_locl + kk*topo->n1l + col_dof;
                        vals[2]  = -1.0;

			col_proc = inds_1y[(ii+1)*nn+jj]/topo->n1l;
			col_dof  = inds_1y[(ii+1)*nn+jj]%topo->n1l;
			cols[3]  = col_proc*n_cols_locl + kk*topo->n1l + col_dof;
                        vals[3]  = +1.0;

			if(kk == 0) {
                            nCols = 5;
			    cols[4] = col_proc*n_cols_locl + topo->nk*topo->n1l + inds_2[ii*nn+jj];
			    vals[4] = +1.0;
			} else if(kk == topo->nk-1) {
                            nCols = 5;
			    cols[4] = col_proc*n_cols_locl + topo->nk*topo->n1l + (kk-1)*topo->n2l + inds_2[ii*nn+jj];
			    vals[4] = -1.0;
                        } else {
                            nCols = 6;
			    cols[4] = col_proc*n_cols_locl + topo->nk*topo->n1l + (kk-1)*topo->n2l + inds_2[ii*nn+jj];
			    vals[4] = -1.0;
			    cols[5] = col_proc*n_cols_locl + topo->nk*topo->n1l + (kk+0)*topo->n2l + inds_2[ii*nn+jj];
			    vals[5] = +1.0;
                        }
                        MatSetValues(M, 1, &row, nCols, cols, vals, INSERT_VALUES);
                    }
                }
            }
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    // build the -ve of the transpose
    MatTranspose(M, MAT_INITIAL_MATRIX, &MT);
    MatScale(MT, -1.0);
    MatAssemblyBegin(MT, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  MT, MAT_FINAL_ASSEMBLY);
}

E32_Coupled::~E32_Coupled() {
    MatDestroy(&M);
    MatDestroy(&M);
}

M3mat_coupled::M3mat_coupled(Topo* _topo, Geom* _geom, LagrangeEdge* _e) {
    int n2;

    topo = _topo;
    geom = _geom;
    e = _e;

    n2 = topo->elOrd*topo->elOrd;

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->nk*topo->n2l, topo->nk*topo->n2l, topo->nk*topo->nDofs2G, topo->nk*topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, n2, PETSC_NULL, 0, PETSC_NULL);
    MatZeroEntries(M);

    MatCreate(MPI_COMM_WORLD, &Minv);
    MatSetSizes(Minv, topo->nk*topo->n2l, topo->nk*topo->n2l, topo->nk*topo->nDofs2G, topo->nk*topo->nDofs2G);
    MatSetType(Minv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Minv, n2, PETSC_NULL, 0, PETSC_NULL);
    MatZeroEntries(Minv);
}

void M3mat_coupled::assemble(double scale, Vec* p3, bool vert_scale, double fac) {
    int ex, ey, ei, n2, mp1, mp12, ii, jj, kk, *inds, *inds0, inds_g[99], shift;
    double det, pVal, tVal, val;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    PetscScalar *pArray, *tArray;

    n2 = topo->elOrd*topo->elOrd;
    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;
    shift = topo->pi*topo->nk*topo->n2l;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatZeroEntries(M);

    for(kk = 0; kk < topo->nk; kk++) {
        pArray = tArray = NULL;
        if(p3 && vert_scale) {
            VecGetArray(p3[kk], &pArray);
        } else if(p3) {
	    VecGetArray(p3[kk+0], &pArray);
	    VecGetArray(p3[kk+1], &tArray);
        }
        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                ei = ey*topo->nElsX + ex;
                inds0 = topo->elInds0_l(ex, ey);
                inds = topo->elInds2_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Qaa[ii]  = Q->A[ii]*(scale/det);
                    Qaa[ii] *= geom->thickInv[kk][inds0[ii]];

		    pVal = tVal = 0.0;
		    if(pArray) {
                        for(jj = 0; jj < n2; jj++) {
                            pVal += pArray[inds[jj]]*W->A[ii*n2+jj];
                        }
                    }
		    if(tArray) {
                        for(jj = 0; jj < n2; jj++) {
                            tVal += tArray[inds[jj]]*W->A[ii*n2+jj];
                        }
                    }
		    if(p3) {
			if(vert_scale) {
                            val = pVal*geom->thickInv[kk][inds0[ii]]/det;
			} else {
                            val = 0.5*(pVal+tVal)/det;
			}
                        Qaa[ii] *= fac*val;
                    }
                }
                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
                Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

		for(ii = 0; ii < n2; ii++) {
                    inds_g[ii] = shift + kk*topo->n2l + inds[ii];
                }
                MatSetValues(M, W->nDofsJ, inds_g, W->nDofsJ, inds_g, WtQW, ADD_VALUES);
            }
        }
        if(p3) {
            if(tArray) VecRestoreArray(p3[kk+1], &tArray);
            if(pArray) VecRestoreArray(p3[kk+0], &pArray);
	}
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete W;
    delete Q;
}

void M3mat_coupled::assemble_inv(double scale, Vec* p3) {
    int ex, ey, ei, n2, mp1, mp12, ii, jj, kk, *inds, *inds0, inds_g[99], shift;
    double det, val;
    Wii* Q = new Wii(e->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double* Qaa = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    PetscScalar *pArray;

    n2 = topo->elOrd*topo->elOrd;
    mp1 = e->l->q->n + 1;
    mp12 = mp1*mp1;
    shift = topo->pi*topo->nk*topo->n2l;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatZeroEntries(Minv);

    for(kk = 0; kk < topo->nk; kk++) {
        if(p3) VecGetArray(p3[kk], &pArray);
        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                ei = ey*topo->nElsX + ex;
                inds0 = topo->elInds0_l(ex, ey);
                inds = topo->elInds2_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Qaa[ii]  = Q->A[ii]*(scale/det);
                    Qaa[ii] *= geom->thickInv[kk][inds0[ii]];

		    if(p3) {
		        val = 0.0;
                        for(jj = 0; jj < n2; jj++) {
                            val += pArray[inds[jj]]*W->A[ii*n2+jj];
                        }
                        val *= geom->thickInv[kk][inds0[ii]]/det;
                        Qaa[ii] *= val;
                    }
                }
                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
                Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                Inv(WtQW, WtQWinv, n2);

		for(ii = 0; ii < n2; ii++) {
                    inds_g[ii] = shift + kk*topo->n2l + inds[ii];
                }
                MatSetValues(Minv, W->nDofsJ, inds_g, W->nDofsJ, inds_g, WtQWinv, ADD_VALUES);
            }
        }
        if(p3) VecRestoreArray(p3[kk], &pArray);
    }
    MatAssemblyBegin(Minv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Minv, MAT_FINAL_ASSEMBLY);

    delete[] Qaa;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    delete W;
    delete Q;
}

M3mat_coupled::~M3mat_coupled() {
    MatDestroy(&M);
    MatDestroy(&Minv);
}

M2mat_coupled::M2mat_coupled(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    int nn, np1, n_dofs_locl, n_dofs_glob;

    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    nn = topo->elOrd;
    np1 = nn + 1;
    n_dofs_locl = topo->nk*topo->n1l + (topo->nk-1)*topo->n2l;
    n_dofs_glob = topo->nk*topo->nDofs1G + (topo->nk-1)*topo->nDofs2G;

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, n_dofs_locl, n_dofs_locl, n_dofs_glob, n_dofs_glob);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*nn*np1, PETSC_NULL, 2*nn*np1, PETSC_NULL);
    MatZeroEntries(M);

    MatCreate(MPI_COMM_WORLD, &Minv);
    MatSetSizes(Minv, n_dofs_locl, n_dofs_locl, n_dofs_glob, n_dofs_glob);
    MatSetType(Minv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Minv, nn*nn, PETSC_NULL, 0, PETSC_NULL);
    MatZeroEntries(Minv);
}

void M2mat_coupled::assemble(double scale, double fac, Vec* ph, Vec* pz, bool vert_scale) {
    int ex, ey, ei, kk, ii, jj, n2, mp1, mp12, dofs_per_proc, proc_ind, dof_ind, shift;
    int *inds_x, *inds_y, *inds_0, *inds_2, inds_x_g[99], inds_y_g[99], inds_z_g[99];
    Wii* Q = new Wii(l->q, geom);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double det, **J, val, tVal, pVal;
    double* Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    double* Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double* WtQ   = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* VtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* VtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double* WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* Qaa = new double[Q->nDofsI];
    double* Qab = new double[Q->nDofsI];
    double* Qbb = new double[Q->nDofsI];
    PetscScalar *pArray, *tArray;

    n2 = topo->elOrd*topo->elOrd;
    mp1 = l->q->n + 1;
    mp12 = mp1*mp1;
    dofs_per_proc = topo->nk*topo->n1l + (topo->nk-1)*topo->n2l;
    shift = topo->pi*dofs_per_proc + topo->nk*topo->n1l;

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);
    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatZeroEntries(M);

    // horizontal vector components
    for(kk = 0; kk < geom->nk; kk++) {
        pArray = tArray = NULL;
        if(ph && vert_scale) {
            VecGetArray(ph[kk], &pArray);
	} else if(ph) {
	    VecGetArray(ph[kk+1], &tArray);
	    VecGetArray(ph[kk+0], &pArray);
        }
        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                ei = ey*topo->nElsX + ex;
                inds_0 = topo->elInds0_l(ex, ey);
                if(ph) inds_2 = topo->elInds2_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    J = geom->J[ei][ii];

                    Qaa[ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii]*(scale/det);
                    Qab[ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii]*(scale/det);
                    Qbb[ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii]*(scale/det);

                    // horiztonal velocity is piecewise constant in the vertical
                    Qaa[ii] *= geom->thickInv[kk][inds_0[ii]];
                    Qab[ii] *= geom->thickInv[kk][inds_0[ii]];
                    Qbb[ii] *= geom->thickInv[kk][inds_0[ii]];

		    pVal = tVal = 0.0;
		    if(pArray) {
                        for(jj = 0; jj < n2; jj++) {
                            pVal += pArray[inds_2[jj]]*W->A[ii*n2+jj];
                        }
                    }
		    if(tArray) {
                        for(jj = 0; jj < n2; jj++) {
                            tVal += tArray[inds_2[jj]]*W->A[ii*n2+jj];
                        }
                    }
		    if(ph) {
			if(vert_scale) {
                            val = pVal*geom->thickInv[kk][inds_0[ii]]/det;
			} else {
                            val = 0.5*(pVal+tVal)/det;
			}
                        Qaa[ii] *= fac*val;
                        Qab[ii] *= fac*val;
                        Qbb[ii] *= fac*val;
                    }
                }

                inds_x = topo->elInds1x_g(ex, ey);
                inds_y = topo->elInds1y_g(ex, ey);
		for(ii = 0; ii < U->nDofsJ; ii++) {
                    proc_ind = inds_x[ii]/topo->n1l;
                    dof_ind  = inds_x[ii]%topo->n1l;
                    inds_x_g[ii] = proc_ind*dofs_per_proc + kk*topo->n1l + dof_ind;
                    proc_ind = inds_y[ii]/topo->n1l;
                    dof_ind  = inds_y[ii]%topo->n1l;
                    inds_y_g[ii] = proc_ind*dofs_per_proc + kk*topo->n1l + dof_ind;
                }

                Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qaa, UtQaa);
                Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qab, UtQab);
                Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qab, VtQba);
                Mult_FD_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qbb, VtQbb);

                Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQaa, U->A, UtQU);
                Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQab, V->A, UtQV);
                Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQba, U->A, VtQU);
                Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQbb, V->A, VtQV);
/*
for(ii=0;ii<U->nDofsJ;ii++)
for(jj=0;jj<U->nDofsJ;jj++)
UtQU[ii*U->nDofsJ+jj]=
UtQV[ii*U->nDofsJ+jj]=
VtQU[ii*U->nDofsJ+jj]=
VtQV[ii*U->nDofsJ+jj]=0.0;
for(ii=0;ii<U->nDofsJ;ii++)
UtQU[ii*U->nDofsJ+ii]=
VtQV[ii*U->nDofsJ+ii]=1.0;
*/
                MatSetValues(M, U->nDofsJ, inds_x_g, U->nDofsJ, inds_x_g, UtQU, ADD_VALUES);
                MatSetValues(M, U->nDofsJ, inds_x_g, U->nDofsJ, inds_y_g, UtQV, ADD_VALUES);
                MatSetValues(M, U->nDofsJ, inds_y_g, U->nDofsJ, inds_x_g, VtQU, ADD_VALUES);
                MatSetValues(M, U->nDofsJ, inds_y_g, U->nDofsJ, inds_y_g, VtQV, ADD_VALUES);
            }
        }
        if(ph) {
            if(tArray) VecRestoreArray(ph[kk+1], &tArray);
            if(pArray) VecRestoreArray(ph[kk+0], &pArray);
	}
    }

    // vertical vector components
/*
    for(kk = 0; kk < geom->nk; kk++) {
        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                ei = ey*topo->nElsX + ex;
                inds_0 = topo->elInds0_l(ex, ey);
                inds_2 = topo->elInds2_l(ex, ey);

                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Qaa[ii]  = Q->A[ii]*(SCALE/det);
                    Qaa[ii] *= 0.5*fac*geom->thick[kk][inds_0[ii]];
                }

                if(pz) VecGetArray(pz[ei], &pArray);
		// bottom
		if(kk > 0) {
                    lev = kk - 1;
                    for(ii = 0; ii < mp12; ii++) {
                        //det = geom->det[ei][ii];
                        //Qaa[ii]  = Q->A[ii]*(scale/det);
                        //Qaa[ii] *= 0.5*fac*geom->thick[lev][inds_0[ii]];

		        if(pz && vert_scale) {
                            val = 0.0;
                            for(jj = 0; jj < n2; jj++) {
                                val += pArray[lev*n2+jj]*W->A[ii*n2+jj];
                            }
			    val /= det;
			    val *= geom->thickInv[lev][inds_0[ii]];
                            Qbb[ii] = val*Qaa[ii];
                        } else if(pz) {
                            val = 0.0;
                            for(jj = 0; jj < n2; jj++) {
                                val += pArray[(lev+1)*n2+jj]*W->A[ii*n2+jj]; // theta has nk+1 levels
                            }
			    val /= det;
                            Qbb[ii] = val*Qaa[ii];
                        } else {
                            Qbb[ii] = Qaa[ii];
                        }
                    }

		    for(jj = 0; jj < n2; jj++) {
                        inds_z_g[jj] = shift + lev*topo->n2l + inds_2[jj];
                    }
                    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qbb, WtQ);
                    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                    MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_z_g, WtQW, ADD_VALUES);
		}
		// top
		if(kk < topo->nk-1) {
                    lev = kk;
                    for(ii = 0; ii < mp12; ii++) {
                        //det = geom->det[ei][ii];
                        //Qaa[ii]  = Q->A[ii]*(scale/det);
                        //Qaa[ii] *= 0.5*fac*geom->thick[lev][inds_0[ii]];

		        if(pz && vert_scale) {
                            val = 0.0;
                            for(jj = 0; jj < n2; jj++) {
                                val += pArray[lev*n2+jj]*W->A[ii*n2+jj];
                            }
			    val /= det;
			    val *= geom->thickInv[lev][inds_0[ii]];
                            Qbb[ii] = val*Qaa[ii];
                        } else if(pz) {
                            val = 0.0;
                            for(jj = 0; jj < n2; jj++) {
                                val += pArray[(lev+1)*n2+jj]*W->A[ii*n2+jj]; // theta has nk+1 levels
                            }
			    val /= det;
                            Qbb[ii] = val*Qaa[ii];
                        } else {
                            Qbb[ii] = Qaa[ii];
                        }
                    }

		    for(jj = 0; jj < n2; jj++) {
                        inds_z_g[jj] = shift + lev*topo->n2l + inds_2[jj];
                    }
                    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qbb, WtQ);
                    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                    MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_z_g, WtQW, ADD_VALUES);
                }
                if(pz) VecRestoreArray(pz[ei], &pArray);
            }
        }
    }
*/
    if(!pz) {
        for(kk = 0; kk < geom->nk; kk++) {
            for(ey = 0; ey < topo->nElsX; ey++) {
                for(ex = 0; ex < topo->nElsX; ex++) {
                    ei = ey*topo->nElsX + ex;
                    inds_0 = topo->elInds0_l(ex, ey);
                    inds_2 = topo->elInds2_l(ex, ey);

                    for(ii = 0; ii < mp12; ii++) {
                        det = geom->det[ei][ii];
                        Qaa[ii]  = Q->A[ii]*(SCALE/det);
                        Qaa[ii] *= 0.5*fac*geom->thick[kk][inds_0[ii]];
                    }
                    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
                    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
		    if(kk > 0) {
                        for(jj = 0; jj < n2; jj++) {
                            inds_z_g[jj] = shift + (kk-1)*topo->n2l + inds_2[jj];
                        }
                        MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_z_g, WtQW, ADD_VALUES);
                    }
		    if(kk < topo->nk-1) {
                        for(jj = 0; jj < n2; jj++) {
                            inds_z_g[jj] = shift + (kk+0)*topo->n2l + inds_2[jj];
                        }
                        MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_z_g, WtQW, ADD_VALUES);
                    }
                }
            }
        }
    } else if(vert_scale) {
        for(kk = 0; kk < geom->nk; kk++) {
            for(ey = 0; ey < topo->nElsX; ey++) {
                for(ex = 0; ex < topo->nElsX; ex++) {
                    ei = ey*topo->nElsX + ex;
                    inds_0 = topo->elInds0_l(ex, ey);
                    inds_2 = topo->elInds2_l(ex, ey);

                    VecGetArray(pz[ei], &pArray);
                    for(ii = 0; ii < mp12; ii++) {
                        det = geom->det[ei][ii];
                        Qaa[ii] = Q->A[ii]*(SCALE/det);

                        val = 0.0;
                        for(jj = 0; jj < n2; jj++) {
                            val += pArray[kk*n2+jj]*W->A[ii*n2+jj];
                        }
                        Qaa[ii] *= 0.5*val/det;
                    }
		    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
                    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

		    if(kk > 0) {
                        for(jj = 0; jj < n2; jj++) {
                            inds_z_g[jj] = shift + (kk-1)*topo->n2l + inds_2[jj];
                        }
                        MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_z_g, WtQW, ADD_VALUES);
                    }
		    if(kk < topo->nk-1) {
                        for(jj = 0; jj < n2; jj++) {
                            inds_z_g[jj] = shift + (kk+0)*topo->n2l + inds_2[jj];
                        }
                        MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_z_g, WtQW, ADD_VALUES);
                    }
                    VecRestoreArray(pz[ei], &pArray);
                }
            }
        }
    } else {
        for(kk = 0; kk < geom->nk; kk++) {
            for(ey = 0; ey < topo->nElsX; ey++) {
                for(ex = 0; ex < topo->nElsX; ex++) {
                    ei = ey*topo->nElsX + ex;
                    inds_0 = topo->elInds0_l(ex, ey);
                    inds_2 = topo->elInds2_l(ex, ey);

                    VecGetArray(pz[ei], &pArray);
                    for(ii = 0; ii < mp12; ii++) {
                        det = geom->det[ei][ii];
                        Qaa[ii]  = Q->A[ii]*(SCALE/det);
                        Qaa[ii] *= 0.5*geom->thick[kk][inds_0[ii]];
                        Qbb[ii]  = Qaa[ii];

                        pVal = tVal = 0.0;
                        for(jj = 0; jj < n2; jj++) {
                            pVal += pArray[(kk+0)*n2+jj]*W->A[ii*n2+jj];
                            tVal += pArray[(kk+1)*n2+jj]*W->A[ii*n2+jj];
                        }
                        Qaa[ii] *= pVal/det;
                        Qbb[ii] *= tVal/det;
                    }

                    // assemble the first basis function
                    if(kk > 0) {
                        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
                        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                        for(jj = 0; jj < n2; jj++) {
                            inds_z_g[jj] = shift + (kk-1)*topo->n2l + inds_2[jj];
                        }
                        MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_z_g, WtQW, ADD_VALUES);
                    }

                    // assemble the second basis function
                    if(kk < geom->nk - 1) {
                        Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qbb, WtQ);
                        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                        for(jj = 0; jj < n2; jj++) {
                            inds_z_g[jj] = shift + (kk+0)*topo->n2l + inds_2[jj];
                        }
                        MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_z_g, WtQW, ADD_VALUES);
                    }
                    VecRestoreArray(pz[ei], &pArray);
                }
            }
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    Free2D(W->nDofsJ, Wt);
    Free2D(U->nDofsJ, UtQaa);
    Free2D(U->nDofsJ, UtQab);
    Free2D(U->nDofsJ, VtQba);
    Free2D(U->nDofsJ, VtQbb);
    Free2D(W->nDofsJ, WtQ);
    Free2D(U->nDofsJ, UtQU);
    Free2D(U->nDofsJ, UtQV);
    Free2D(U->nDofsJ, VtQU);
    Free2D(U->nDofsJ, VtQV);
    Free2D(W->nDofsJ, WtQW);
    delete[] Qaa;
    delete[] Qab;
    delete[] Qbb;
    delete Q;
    delete U;
    delete V;
    delete W;
}

void M2mat_coupled::assemble_inv(double scale, Umat* Mk) {
    int kk, mm, mi, mf, nCols, ii, ri, inds[99], *inds_2, *inds_0, mp1, mp12, ex, ey, ei, n_dofs_locl;
    const int *cols;
    const double* vals;
    double val_sum, Q0[99], det;
    Wii*       Q    = new Wii(l->q, geom);
    M2_j_xy_i* W    = new M2_j_xy_i(e);
    double* Wt      = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ     = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQW    = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);

    mp1 = l->n + 1;
    mp12 = mp1*mp1;
    n_dofs_locl = topo->nk*topo->n1l + (topo->nk-1)*topo->n2l;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatZeroEntries(Minv);

    MatGetOwnershipRange(Mk->M, &mi, &mf);

    for(kk = 0; kk < topo->nk; kk++) {
        Mk->assemble(kk, scale, true);
        for(mm = mi; mm < mf; mm++) {
            MatGetRow(Mk->M, mm, &nCols, &cols, &vals);
            val_sum = 0.0;
            for(ii = 0; ii < nCols; ii++) {
		if(cols[ii]==mm) {
                    val_sum = vals[ii];
                }
            }
            val_sum = 1.0/val_sum;
            MatRestoreRow(Mk->M, mm, &nCols, &cols, &vals);

            ri = topo->pi*n_dofs_locl + kk*topo->n1l + mm - mi;
            MatSetValues(Minv, 1, &ri, 1, &ri, &val_sum, ADD_VALUES);
        }
    }

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_2 = topo->elInds2_l(ex, ey);
            inds_0 = topo->elInds0_l(ex, ey);
            for(kk = 0; kk < topo->nk-1; kk++) {
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Q0[ii]  = Q->A[ii]*(scale/det);
                    Q0[ii] *= 0.5*(geom->thick[kk+0][inds_0[ii]] + geom->thick[kk+1][inds_0[ii]]);
                }
                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
                Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

                Inv(WtQW, WtQWinv, W->nDofsJ);

                for(ii = 0; ii < W->nDofsJ; ii++) {
                    inds[ii] = topo->pi*n_dofs_locl + topo->nk*topo->n1l + kk*topo->n2l + inds_2[ii];
                }
                MatSetValues(Minv, W->nDofsJ, inds, W->nDofsJ, inds, WtQWinv, ADD_VALUES);
            }
        }
    }
    MatAssemblyBegin(Minv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Minv, MAT_FINAL_ASSEMBLY);

    delete Q;
    delete W;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
}

M2mat_coupled::~M2mat_coupled() {
    MatDestroy(&M);
    MatDestroy(&Minv);
}

Kmat_coupled::Kmat_coupled(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e) {
    int n_rows_locl, n_rows_glob, n_cols_locl, n_cols_glob;

    topo = _topo;
    geom = _geom;
    l = _l;
    e = _e;

    n_rows_locl = topo->nk*topo->n1l + (topo->nk-1)*topo->n2l;
    n_rows_glob = topo->nk*topo->nDofs1G + (topo->nk-1)*topo->nDofs2G;
    n_cols_locl = topo->nk*topo->n2l;
    n_cols_glob = topo->nk*topo->nDofs2G;

    U = new M1x_j_xy_i(l, e);
    V = new M1y_j_xy_i(l, e);
    W = new M2_j_xy_i(e);
    Q = new Wii(l->q, geom);
    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(V->nDofsJ, V->nDofsI);
    Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    Qaa = new double[Q->nDofsI];
    Qba = new double[Q->nDofsI];
    UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(V->nDofsJ, Q->nDofsJ);
    WtQ   = Alloc2D(W->nDofsJ, Q->nDofsJ);
    UtQW = Alloc2D(U->nDofsJ, W->nDofsJ);
    VtQW = Alloc2D(V->nDofsJ, W->nDofsJ);
    WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(V->nDofsI, V->nDofsJ, V->A, Vt);
    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, n_rows_locl, n_cols_locl, n_rows_glob, n_cols_glob);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatZeroEntries(M);
}

void Kmat_coupled::assemble(Vec* ul, Vec* wl, double fac, double scale) {
    int ex, ey, ei, ii, jj, kk, n2, mp1, mp12, proc_ind, dof_ind, dofs_per_proc, shift_row, shift_col;
    int *inds_x, *inds_y, *inds_0, *inds_2, inds_x_g[99], inds_y_g[99], inds_z_g[99], inds_2_g[99];
    double det, **J, ux[2], val;
    PetscScalar *uArray;

    n2 = topo->elOrd*topo->elOrd;
    mp1 = l->n + 1;
    mp12 = mp1*mp1;
    dofs_per_proc = geom->nk*topo->n1l + (geom->nk-1)*topo->n2l;
    shift_row = topo->pi*dofs_per_proc + geom->nk*topo->n1l;
    shift_col = topo->pi*geom->nk*topo->n2l;

    MatZeroEntries(M);

    // horiztonal row dofs
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(ul[kk], &uArray);
        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                ei = ey*topo->nElsX + ex;
                inds_0 = topo->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    J = geom->J[ei][ii];
                    geom->interp1_g_t(ex, ey, ii%mp1, ii/mp1, uArray, ux);
                    ux[0] *= geom->thickInv[kk][inds_0[ii]];
                    ux[1] *= geom->thickInv[kk][inds_0[ii]];

                    Qaa[ii]  = fac*(ux[0]*J[0][0] + ux[1]*J[1][0])*Q->A[ii]*(scale/det);
                    Qba[ii]  = fac*(ux[0]*J[0][1] + ux[1]*J[1][1])*Q->A[ii]*(scale/det);
                    Qaa[ii] *= geom->thickInv[kk][inds_0[ii]];
                    Qba[ii] *= geom->thickInv[kk][inds_0[ii]];
                }

                Mult_FD_IP(U->nDofsJ, Q->nDofsJ, Q->nDofsI, Ut, Qaa, UtQaa);
                Mult_FD_IP(V->nDofsJ, Q->nDofsJ, Q->nDofsI, Vt, Qba, VtQba);

                Mult_IP(U->nDofsJ, W->nDofsJ, W->nDofsI, UtQaa, W->A, UtQW);
                Mult_IP(V->nDofsJ, W->nDofsJ, W->nDofsI, VtQba, W->A, VtQW);

                inds_x = topo->elInds1x_g(ex, ey);
                inds_y = topo->elInds1y_g(ex, ey);
                inds_2 = topo->elInds2_l(ex, ey);
		for(ii = 0; ii < U->nDofsJ; ii++) {
                    proc_ind = inds_x[ii]/topo->n1l;
                    dof_ind  = inds_x[ii]%topo->n1l;
                    inds_x_g[ii] = proc_ind*dofs_per_proc + kk*topo->n1l + dof_ind;
                    proc_ind = inds_y[ii]/topo->n1l;
                    dof_ind  = inds_y[ii]%topo->n1l;
                    inds_y_g[ii] = proc_ind*dofs_per_proc + kk*topo->n1l + dof_ind;
                }
		for(ii = 0; ii < W->nDofsJ; ii++) {
                    inds_2_g[ii] = shift_col + kk*topo->n2l + inds_2[ii];
                }

                MatSetValues(M, U->nDofsJ, inds_x_g, W->nDofsJ, inds_2_g, UtQW, ADD_VALUES);
                MatSetValues(M, V->nDofsJ, inds_y_g, W->nDofsJ, inds_2_g, VtQW, ADD_VALUES);
            }
        }
        VecRestoreArray(ul[kk], &uArray);
    }

    // vertical row dofs
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_2 = topo->elInds2_l(ex, ey);
            VecGetArray(wl[ei], &uArray);
            for(kk = 0; kk < geom->nk; kk++) {
                for(jj = 0; jj < W->nDofsJ; jj++) {
                    inds_2_g[jj] = shift_col + kk*topo->n2l + inds_2[jj];
                }
                if(kk > 0) {
                    for(ii = 0; ii < mp12; ii++) {
                        det = geom->det[ei][ii];
                        Qaa[ii] = 0.5*Q->A[ii]*(SCALE/det);

                        val = 0.0;
                        for(jj = 0; jj < n2; jj++) {
                            val += uArray[(kk-1)*n2+jj]*W->A[ii*n2+jj];
                        }
                        Qaa[ii] *= val/det;
                    }

                    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
                    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

                    for(jj = 0; jj < W->nDofsJ; jj++) {
                        inds_z_g[jj] = shift_row + (kk-1)*topo->n2l + inds_2[jj];
                    }
                    MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_2_g, WtQW, ADD_VALUES);
                }
                if(kk < geom->nk - 1) {
                    for(ii = 0; ii < mp12; ii++) {
                        det = geom->det[ei][ii];
                        Qaa[ii] = 0.5*Q->A[ii]*(SCALE/det);

                        val = 0.0;
                        for(jj = 0; jj < n2; jj++) {
                            val += uArray[(kk+0)*n2+jj]*W->A[ii*n2+jj];
                        }
                        Qaa[ii] *= val/det;
                    }

                    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
                    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

                    for(jj = 0; jj < W->nDofsJ; jj++) {
                        inds_z_g[jj] = shift_row + (kk+0)*topo->n2l + inds_2[jj];
                    }
                    MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_2_g, WtQW, ADD_VALUES);
                }
/*
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Qaa[ii] = 0.5*fac*Q->A[ii]*(scale/det);
                }
		for(ii = 0; ii < W->nDofsJ; ii++) {
                    inds_2_g[ii] = shift_col + kk*topo->n2l + inds_2[ii];
                }
		// bottom
		if(kk > 0) {
                    for(ii = 0; ii < mp12; ii++) {
                        det = geom->det[ei][ii];
                        val = 0.0;
                        for(jj = 0; jj < n2; jj++) {
                            val += uArray[(kk-1)*n2+jj]*W->A[ii*n2+jj];
                        }
                        Qba[ii] = Qaa[ii]*val/det;
                    }

                    for(jj = 0; jj < n2; jj++) {
                        inds_z_g[jj] = shift_row + (kk-1)*topo->n2l + inds_2[jj];
                    }
                    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qba, WtQ);
                    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                    MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_2_g, WtQW, ADD_VALUES);
		}
		// top
		if(kk < topo->nk-1) {
                    for(ii = 0; ii < mp12; ii++) {
                        det = geom->det[ei][ii];
                        val = 0.0;
                        for(jj = 0; jj < n2; jj++) {
                            val += uArray[(kk+0)*n2+jj]*W->A[ii*n2+jj];
                        }
                        Qba[ii] = Qaa[ii]*val/det;
                    }

                    for(jj = 0; jj < n2; jj++) {
                        inds_z_g[jj] = shift_row + (kk+0)*topo->n2l + inds_2[jj];
                    }
                    Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qba, WtQ);
                    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                    MatSetValues(M, W->nDofsJ, inds_z_g, W->nDofsJ, inds_2_g, WtQW, ADD_VALUES);
                }
*/
            }
            VecRestoreArray(wl[ei], &uArray);
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);
}

Kmat_coupled::~Kmat_coupled() {
    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    Free2D(W->nDofsJ, Wt);
    delete[] Qaa;
    delete[] Qba;
    Free2D(U->nDofsJ, UtQW);
    Free2D(V->nDofsJ, VtQW);
    Free2D(V->nDofsJ, WtQW);
    Free2D(U->nDofsJ, UtQaa);
    Free2D(V->nDofsJ, VtQba);
    Free2D(V->nDofsJ, WtQ);
    delete U;
    delete V;
    delete W;
    delete Q;
    MatDestroy(&M);
}

void AddM3_Coupled(Topo* topo, int row_ind, int col_ind, Mat M3, Mat M) {
    int mi, mf, mm, mp, nCols, ri, ci, lev, fce, m3_dofs_per_proc, shift, proc_i;
    const int *cols;
    const double* vals;
    int cols2[999];

    shift = topo->pi*topo->dofs_per_proc + topo->nk*topo->n1l;
    m3_dofs_per_proc = topo->nk*topo->n2l;

    MatGetOwnershipRange(M3, &mi, &mf);
if(mi != topo->pi*m3_dofs_per_proc)cout<<"AddM3: ERROR!\t"<<mi<<"\t"<<topo->pi*m3_dofs_per_proc<<"\n";
    for(mm = mi; mm < mf; mm++) {
        mp  = mm - topo->pi*m3_dofs_per_proc;
        lev = mp / topo->n2l;
        fce = mp % topo->n2l;
        ri  = shift + (4*topo->nk-1)*fce + 4*lev + row_ind;
        MatGetRow(M3, mm, &nCols, &cols, &vals);
	for(ci = 0; ci < nCols; ci++) {
            proc_i = cols[ci]/m3_dofs_per_proc;
            mp     = cols[ci] - proc_i*m3_dofs_per_proc;
            lev    = mp / topo->n2l;
            fce    = mp % topo->n2l;
            cols2[ci] = proc_i*topo->dofs_per_proc + topo->nk*topo->n1l + (4*topo->nk-1)*fce + 4*lev + col_ind;
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(M3, mm, &nCols, &cols, &vals);
    }
}

void AddM2_Coupled(Topo* topo, Mat M2, Mat M) {
    int mi, mf, mm, mp, nCols, ri, ci, m2_dofs_per_proc, xy_dofs_per_proc, lev, fce, shift, proc_i;
    const int *cols;
    const double* vals;
    int cols2[999];

    shift            = topo->pi*topo->dofs_per_proc;
    xy_dofs_per_proc = topo->nk*topo->n1l;
    m2_dofs_per_proc = xy_dofs_per_proc + (topo->nk-1)*topo->n2l;

    MatGetOwnershipRange(M2, &mi, &mf);
if(mi != topo->pi*m2_dofs_per_proc)cout<<"AddM2: ERROR!\t"<<mi<<"\t"<<topo->pi*m2_dofs_per_proc<<"\n";
    for(mm = mi; mm < mf; mm++) {
        mp = mm - topo->pi*m2_dofs_per_proc;
	if(mp < xy_dofs_per_proc) {
            ri  = topo->pi*topo->dofs_per_proc + mp;
	} else {
            lev = (mp-xy_dofs_per_proc) / topo->n2l;
            fce = (mp-xy_dofs_per_proc) % topo->n2l;
            ri  = shift + xy_dofs_per_proc + (4*topo->nk-1)*fce + 4*lev + 3;
        }
        MatGetRow(M2, mm, &nCols, &cols, &vals);
	for(ci = 0; ci < nCols; ci++) {
            proc_i = cols[ci]/m2_dofs_per_proc;
            mp     = cols[ci] - proc_i*m2_dofs_per_proc;
           if(mp < xy_dofs_per_proc) {
                cols2[ci] = proc_i*topo->dofs_per_proc + mp;
            } else {
                lev       = (mp-xy_dofs_per_proc) / topo->n2l;
                fce       = (mp-xy_dofs_per_proc) % topo->n2l;
                cols2[ci] = proc_i*topo->dofs_per_proc + xy_dofs_per_proc + (4*topo->nk-1)*fce + 4*lev + 3;
            }
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(M2, mm, &nCols, &cols, &vals);
    }
}

void AddG_Coupled(Topo* topo, int col_ind, Mat G, Mat M) {
    int mi, mf, mm, mp, nCols, ri, ci, m2_dofs_per_proc, m3_dofs_per_proc, xy_dofs_per_proc, lev, fce, proc_i;
    const int *cols;
    const double* vals;
    int cols2[999];

    m3_dofs_per_proc = topo->nk*topo->n2l;
    xy_dofs_per_proc = topo->nk*topo->n1l;
    m2_dofs_per_proc = xy_dofs_per_proc + (topo->nk-1)*topo->n2l;

    MatGetOwnershipRange(G, &mi, &mf);
if(mi != topo->pi*m2_dofs_per_proc)cout<<"AddG: ERROR!\t"<<mi<<"\t"<<topo->pi*m2_dofs_per_proc<<"\n";
    for(mm = mi; mm < mf; mm++) {
        mp = mm - topo->pi*m2_dofs_per_proc;
	if(mp < xy_dofs_per_proc) {
            ri  = topo->pi*topo->dofs_per_proc + mp;
	} else {
            lev = (mp-xy_dofs_per_proc) / topo->n2l;
            fce = (mp-xy_dofs_per_proc) % topo->n2l;
            ri  = topo->pi*topo->dofs_per_proc + xy_dofs_per_proc + (4*topo->nk-1)*fce + 4*lev + 3;
        }
        MatGetRow(G, mm, &nCols, &cols, &vals);
	for(ci = 0; ci < nCols; ci++) {
            proc_i    = cols[ci]/m3_dofs_per_proc;
            mp        = cols[ci] - proc_i*m3_dofs_per_proc;
            lev       = mp / topo->n2l;
            fce       = mp % topo->n2l;
            cols2[ci] = proc_i*topo->dofs_per_proc + xy_dofs_per_proc + (4*topo->nk-1)*fce + 4*lev + col_ind;
        }
        MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(G, mm, &nCols, &cols, &vals);
    }
}

void AddD_Coupled(Topo* topo, int row_ind, Mat D, Mat M) {
    int mi, mf, mm, mp, nCols, ri, ci, m2_dofs_per_proc, m3_dofs_per_proc, xy_dofs_per_proc, lev, fce, shift, proc_i;
    const int *cols;
    const double* vals;
    int cols2[999];

    m3_dofs_per_proc = topo->nk*topo->n2l;
    xy_dofs_per_proc = topo->nk*topo->n1l;
    m2_dofs_per_proc = xy_dofs_per_proc + (topo->nk-1)*topo->n2l;
    shift = topo->pi*topo->dofs_per_proc + xy_dofs_per_proc;

    MatGetOwnershipRange(D, &mi, &mf);
if(mi != topo->pi*m3_dofs_per_proc)cout<<"AddD: ERROR!\t"<<mi<<"\t"<<topo->pi*m3_dofs_per_proc<<"\n";
    for(mm = mi; mm < mf; mm++) {
        mp  = mm - topo->pi*m3_dofs_per_proc;
        lev = mp / topo->n2l;
        fce = mp % topo->n2l;
        ri  = shift + (4*topo->nk-1)*fce + 4*lev + row_ind;
        MatGetRow(D, mm, &nCols, &cols, &vals);
	for(ci = 0; ci < nCols; ci++) {
            proc_i = cols[ci]/m2_dofs_per_proc;
            mp     = cols[ci] - proc_i*m2_dofs_per_proc;
            if(mp < xy_dofs_per_proc) {
                cols2[ci] = proc_i*topo->dofs_per_proc + mp;
            } else {
                lev       = (mp-xy_dofs_per_proc) / topo->n2l;
                fce       = (mp-xy_dofs_per_proc) % topo->n2l;
                cols2[ci] = proc_i*topo->dofs_per_proc + xy_dofs_per_proc + (4*topo->nk-1)*fce + 4*lev + 3;
            }
        }
        MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(D, mm, &nCols, &cols, &vals);
    }
}
