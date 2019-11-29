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
#include "Boundary.h"

using namespace std;

Boundary::Boundary(Topo* _topo, Geom* _geom, LagrangeNode* _node, LagrangeEdge* _edge) {
    int ii, jj, ix, iy;
    int mm = _geom->quad->n;
    int mp1 = mm+1;

    topo = _topo;
    geom = _geom;
    node = _node;
    edge = _edge;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &bl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &qg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bg);

    U = new M1x_j_xy_i(node, edge);
    V = new M1y_j_xy_i(node, edge);

    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    Qa = new double[U->nDofsI];
    Qb = new double[U->nDofsI];
    UtQa = new double[U->nDofsJ];
    VtQb = new double[U->nDofsJ];

    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    // zero out the internal dofs
    for(ii = 0; ii < mm*mp1; ii++) {
        ix = ii%mp1;
        if(ix > 0 && ix < mm) {
            for(jj = 0; jj < mp1*mp1; jj++) {
                Ut[ii][jj] = 0.0;
            }
        } else if (ix == 0) { // left side quad points only
            for(jj = 0; jj < mp1*mp1; jj++) {
                if(jj%mp1 !=  0) Ut[ii][jj] = 0.0;
            }
        } else {              // right side quad points only
            for(jj = 0; jj < mp1*mp1; jj++) {
                if(jj%mp1 != mm) Ut[ii][jj] = 0.0;
            }
        }

        iy = ii/mm;
        if(iy > 0 && iy < mm) {
            for(jj = 0; jj < mp1*mp1; jj++) {
                Vt[ii][jj] = 0.0;
            }
        } else if(iy == 0) { // bottom side quad points only
            for(jj = 0; jj < mp1*mp1; jj++) {
                if(jj/mp1 !=  0) Vt[ii][jj] = 0.0;
            }
        } else {             // top side quad points only
            for(jj = 0; jj < mp1*mp1; jj++) {
                if(jj/mp1 != mm) Vt[ii][jj] = 0.0;
            }
        }
    }
}

Boundary::~Boundary() {
    VecDestroy(&ul);
    VecDestroy(&hl);
    VecDestroy(&ql);
    VecDestroy(&bl);
    VecDestroy(&qg);
    VecDestroy(&bg);

    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    delete[] Qa;
    delete[] Qb;
    delete[] UtQa;
    delete[] VtQb;
    delete U;
    delete V;
}

double UdotN(Geom* geom, int ei, int ii, double* ui, bool norm_horiz) {
    double** J = geom->J[ei][ii];
    double uMagInv, nMagInv, ni[2], uHat[2], nHat[2];

    if(norm_horiz) {
        ni[0] = J[0][1];
        ni[1] = J[1][1];
    } else {
        ni[0] = J[0][0];
        ni[1] = J[1][0];
    }

    uMagInv = 1.0 / sqrt(ui[0]*ui[0] + ui[1]*ui[1]);
    nMagInv = 1.0 / sqrt(ni[0]*ni[0] + ni[1]*ni[1]);

    uHat[0] = uMagInv * ui[0];
    uHat[1] = uMagInv * ui[1];
    nHat[0] = nMagInv * ni[0];
    nHat[1] = nMagInv * ni[1];

    return uHat[0]*nHat[0] + uHat[1]*nHat[1];
}

void Boundary::Interp2To0Bndry(int lev, Vec u, Vec h, bool upwind) {
    int ex, ey, ei, ii, jj, mm, mp1;
    int *inds_0;
    double hi, ui[2], uDotN;
    PetscScalar *uArray, *hArray, *qArray;

    mm = geom->quad->n;
    mp1 = mm+1;

    VecZeroEntries(ql);

    VecScatterBegin(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(ul, &uArray);
    VecGetArray(hl, &hArray);
    VecGetArray(ql, &qArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp1; ii++) {
                // bottom
                jj = ii;
                geom->interp2_g(ex, ey, jj%mp1, jj/mp1, hArray, &hi);
                hi *= 1.0/geom->thick[lev][inds_0[jj]];
                if(!upwind) {
                    qArray[jj] += 0.5*hi;
                } else {
                    geom->interp1_g(ex, ey, jj%mp1, jj/mp1, uArray, ui);
                    uDotN = UdotN(geom, ei, jj, ui, true);
                    qArray[jj] += (uDotN < 0.0) ? hi : 0.0;
                }

                // top
                jj = ii + mp1*mm;
                geom->interp2_g(ex, ey, jj%mp1, jj/mp1, hArray, &hi);
                hi *= 1.0/geom->thick[lev][inds_0[jj]];
                if(!upwind) {
                    qArray[jj] += 0.5*hi;
                } else {
                    geom->interp1_g(ex, ey, jj%mp1, jj/mp1, uArray, ui);
                    uDotN = UdotN(geom, ei, jj, ui, true);
                    qArray[jj] += (uDotN > 0.0) ? hi : 0.0;
                }

                // left
                jj = ii*mp1;
                geom->interp2_g(ex, ey, jj%mp1, jj/mp1, hArray, &hi);
                hi *= 1.0/geom->thick[lev][inds_0[jj]];
                if(!upwind) {
                    qArray[jj] += 0.5*hi;
                } else {
                    geom->interp1_g(ex, ey, jj%mp1, jj/mp1, uArray, ui);
                    uDotN = UdotN(geom, ei, jj, ui, false);
                    qArray[jj] += (uDotN < 0.0) ? hi : 0.0;
                }

                // right
                jj = ii*mp1 + mm;
                geom->interp2_g(ex, ey, jj%mp1, jj/mp1, hArray, &hi);
                hi *= 1.0/geom->thick[lev][inds_0[jj]];
                if(!upwind) {
                    qArray[jj] += 0.5*hi;
                } else {
                    geom->interp1_g(ex, ey, jj%mp1, jj/mp1, uArray, ui);
                    uDotN = UdotN(geom, ei, jj, ui, false);
                    qArray[jj] += (uDotN > 0.0) ? hi : 0.0;
                }
            }
        }
    }
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(hl, &hArray);
    VecRestoreArray(ql, &qArray);

    VecZeroEntries(qg);
    VecScatterBegin(topo->gtol_0, ql, qg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_0, ql, qg, ADD_VALUES, SCATTER_REVERSE);

    VecZeroEntries(ql);
    VecScatterBegin(topo->gtol_0, qg, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qg, ql, INSERT_VALUES, SCATTER_FORWARD);
}

void matvec(double** A, double* x, double* b, int ni, int nj) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        b[ii] = 0.0;
        for(jj = 0; jj < nj; jj++) {
            b[ii] += A[ii][jj] * x[jj];
        }
    }
}

void Boundary::_assembleGrad(int lev) {
    int ex, ey, ei, ii, kk, mm, mp1;
    int *inds_0, *inds_x, *inds_y;
    double **J, tang, norm, det;
    PetscScalar *qArray, *bArray;

    mm = geom->quad->n;
    mp1 = mm+1;

    VecZeroEntries(bl);
    VecGetArray(ql, &qArray);
    VecGetArray(bl, &bArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            inds_0 = topo->elInds0_l(ex, ey);
            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);

            for(ii = 0; ii < mp1; ii++) {
                // bottom
                kk = ii;
                det = geom->det[ei][kk];
                J = geom->J[ei][kk];
                // dot the jacobian onto the global tangent vector
                tang =  sqrt(J[0][0]*J[0][0] + J[1][0]*J[1][0]);
                norm = +sqrt(J[0][1]*J[0][1] + J[1][1]*J[1][1]);
                // layer thickness in jacobian cancels with the thickness inverse in the test function
                Qb[kk] = qArray[inds_0[kk]] * node->q->w[ii] * tang * norm / det;

                // top
                kk = ii + mp1*mm;
                det = geom->det[ei][kk];
                J = geom->J[ei][kk];
                // dot the jacobian onto the global tangent vector
                tang =  sqrt(J[0][0]*J[0][0] + J[1][0]*J[1][0]);
                norm = -sqrt(J[0][1]*J[0][1] + J[1][1]*J[1][1]);
                // layer thickness in jacobian cancels with the thickness inverse in the test function
                Qb[kk] = qArray[inds_0[kk]] * node->q->w[ii] * tang * norm / det;

                // left
                kk = mp1*ii;
                det = geom->det[ei][kk];
                J = geom->J[ei][kk];
                // dot the jacobian onto the global tangent vector
                tang =  sqrt(J[0][1]*J[0][1] + J[1][1]*J[1][1]);
                norm = +sqrt(J[0][0]*J[0][0] + J[1][0]*J[1][0]);
                // layer thickness in jacobian cancels with the thickness inverse in the test function
                Qa[kk] = qArray[inds_0[kk]] * node->q->w[ii] * tang * norm / det;

                // right
                kk = mp1*ii + mm;
                det = geom->det[ei][kk];
                J = geom->J[ei][kk];
                // dot the jacobian onto the global tangent vector
                tang =  sqrt(J[0][1]*J[0][1] + J[1][1]*J[1][1]);
                norm = -sqrt(J[0][0]*J[0][0] + J[1][0]*J[1][0]);
                // layer thickness in jacobian cancels with the thickness inverse in the test function
                Qa[kk] = qArray[inds_0[kk]] * node->q->w[ii] * tang * norm / det;
            }
            matvec(Ut, Qa, UtQa, U->nDofsJ, U->nDofsI);
            matvec(Vt, Qb, VtQb, U->nDofsJ, U->nDofsI);

            for(ii = 0; ii < mp1*mm; ii++) {
                bArray[inds_x[ii]] += UtQa[ii];
                bArray[inds_y[ii]] += VtQb[ii];
            }
        }
    }
    VecRestoreArray(ql, &qArray);
    VecRestoreArray(bl, &bArray);

    VecScatterBegin(topo->gtol_1, bl, bg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, bl, bg, ADD_VALUES, SCATTER_REVERSE);
}

void Boundary::AssembleGrad(int lev, Vec u, Vec h, bool upwind) {
    Interp2To0Bndry(lev, u, h, upwind);
    _assembleGrad(lev);
}
