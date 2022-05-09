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
                inds_y = topo->elInds_velx_g(ex, ey, kk);

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
    Vt = Alloc2D(V->nDofsJ, U->nDofsI);
    Qab = new double[Q->nDofsI];
    Qba = new double[Q->nDofsI];
    UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    VtQU = Alloc2D(V->nDofsJ, V->nDofsJ);
}

void RotMat_coupled::assemble(Vec* q0, double scale, Mat M) {
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
                    vort *= geom->thickInv[kk][inds_0[ii]];

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
                    inds_col = topo->elInds_exner_g(ex, ey, kk);
		}
                MatSetValues(M, W->nDofsJ, inds_row, W->nDofsJ, inds_col, AAinvA, ADD_VALUES);
            }
            VecRestoreArray(p2[ei], &pArray);
        }
    }
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

    MatGetOwnershipRange(G, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        dof_proc = mm / topo->n1l;
        dof_locl = mm % topo->n1l;
        MatGetRow(G, mm, &nCols, &cols, &vals);
	ri = dof_proc*topo->dofs_per_proc + lev*topo->n1l + dof_locl;
	for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            dof_locl = cols[ci] % topo->n2l;
            cols2[ci] = dof_proc*topo->dofs_per_proc + topo->nk*topo->n1l + (4*topo->nk-1)*dof_locl + 4*lev + var_ind;
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

    MatGetOwnershipRange(D, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        dof_proc = mm / topo->n2l;
        dof_locl = mm % topo->n2l;
        MatGetRow(D, mm, &nCols, &cols, &vals);
	ri = dof_proc*topo->dofs_per_proc + topo->nk*topo->n1l + (4*topo->nk-1)*dof_locl + 4*lev + var_ind;
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

    MatGetOwnershipRange(Q, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        dof_proc = mm / topo->n2l;
        dof_locl = mm % topo->n2l;
        MatGetRow(Q, mm, &nCols, &cols, &vals);
        ri = dof_proc*topo->dofs_per_proc + topo->nk*topo->n1l + (4*topo->nk-1)*dof_locl + 4*lev + 1;            // Theta
	for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            dof_locl = cols[ci] % topo->n2l;
            cols2[ci] = dof_proc*topo->dofs_per_proc + topo->nk*topo->n1l + (4*topo->nk-1)*dof_locl + 4*lev + 0; // rho
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
    int *inds_row, *inds_col;

    n2 = topo->elOrd*topo->elOrd;

    MatGetSize(G, &nr, &nc);
    for(mm = 0; mm < nr; mm++) {
        MatGetRow(G, mm, &nCols, &cols, &vals);
        lev = mm / n2;
        fce = mm % n2;
        inds_row = topo->elInds_velz_g(ex, ey, lev);
	ri = inds_row[fce];
	for(ci = 0; ci < nCols; ci++) {
            lev = cols[ci] / n2;
            fce = cols[ci] % n2;
	    if(var_ind == 1) {
                inds_col = topo->elInds_theta_g(ex, ey, lev);
            } else {
                inds_col = topo->elInds_exner_g(ex, ey, lev);
            }
            cols2[ci] = inds_col[fce];
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
    int *inds_row, *inds_col;

    n2 = topo->elOrd*topo->elOrd;

    MatGetSize(D, &nr, &nc);
    for(mm = 0; mm < nr; mm++) {
        MatGetRow(D, mm, &nCols, &cols, &vals);
        lev = mm / n2;
        fce = mm % n2;
	if(var_ind == 0) {
            inds_row = topo->elInds_rho_g(ex, ey, lev);
        } else {
            inds_row = topo->elInds_theta_g(ex, ey, lev);
        }
	ri = inds_row[fce];
	for(ci = 0; ci < nCols; ci++) {
            lev = cols[ci] / n2;
            fce = cols[ci] % n2;
            inds_col = topo->elInds_velz_g(ex, ey, lev);
            cols2[ci] = inds_col[fce];
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(D, mm, &nCols, &cols, &vals);
    }
}

void AddQz_Coupled(Topo* topo, int ex, int ey, Mat Q, Mat M) {
    int nr, nc, mm, nCols, ri, ci, lev, fce, n2;
    const int *cols;
    const double* vals;
    int cols2[1999];
    int *inds_row, *inds_col;

    n2 = topo->elOrd*topo->elOrd;

    MatGetSize(Q, &nr, &nc);
    for(mm = 0; mm < nr; mm++) {
        MatGetRow(Q, mm, &nCols, &cols, &vals);
        lev = mm / n2;
        fce = mm % n2;
        inds_row = topo->elInds_theta_g(ex, ey, lev);
	ri = inds_row[fce];
	for(ci = 0; ci < nCols; ci++) {
            lev = cols[ci] / n2;
            fce = cols[ci] % n2;
            inds_col = topo->elInds_rho_g(ex, ey, lev);
            cols2[ci] = inds_col[fce];
        }
	MatSetValues(M, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(Q, mm, &nCols, &cols, &vals);
    }
}

