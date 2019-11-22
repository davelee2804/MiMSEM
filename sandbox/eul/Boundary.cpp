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
    topo = _topo;
    geom = _geom;
    node = _node;
    edge = _edge;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &bl);
}

Boundary::~Boundary() {
    VecDestroy(&ul);
    VecDestroy(&hl);
    VecDestroy(&ql);
    VecDestroy(&bl);
}

double UdotN(Geom* geom, int ei, int ii, double* ui, bool norm_horiz) {
    double** J = geom->J[ei][ii];
    double uMagInv, nMagInv, ni[2], uHat[2], nHat[2];

    if(norm_horiz) {
        ni[0] = J[1][0];
        ni[1] = J[1][1];
    } else {
        ni[0] = J[0][0];
        ni[1] = J[0][1];
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
}

void matvec(int n, int m, double** A, double* x, double* b) {
    int ii, jj;

    for(ii = 0; ii < n; ii++) {
        b[ii] = 0.0;
        for(jj = 0; jj < m; jj++) {
            b[ii] += A[ii][jj]*x[jj];
        }
    }
}

void Boundary::_assembleGrad(int lev, Vec b) {
    int ex, ey, ei, ii, kk, mm, mp1;
    int *inds_0, *inds_x, *inds_y;
    double _S[99], _N[99], _E[99], _W[99], ES[99], EN[99], EE[99], EW[99], **J, tang;
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
                J = geom->J[ei][kk];
                // dot the jacobian onto the global tangent vector
                tang = sqrt(J[0][0]*J[0][0] + J[0][1]*J[0][1]);
                _S[ii] = node->q->w[ii] * tang * qArray[inds_0[kk]] * geom->thick[lev][inds_0[kk]];

                // top
                kk = ii + mp1*mm;
                J = geom->J[ei][kk];
                // dot the jacobian onto the global tangent vector
                tang = sqrt(J[0][0]*J[0][0] + J[0][1]*J[0][1]);
                _N[ii] = node->q->w[ii] * tang * qArray[inds_0[kk]] * geom->thick[lev][inds_0[kk]];

                // left
                kk = mp1*ii;
                J = geom->J[ei][kk];
                // dot the jacobian onto the global tangent vector
                tang = sqrt(J[1][0]*J[1][0] + J[1][1]*J[1][1]);
                _E[ii] = node->q->w[ii] * tang * qArray[inds_0[kk]] * geom->thick[lev][inds_0[kk]];

                // right
                kk = mp1*ii + mm;
                J = geom->J[ei][kk];
                // dot the jacobian onto the global tangent vector
                tang = sqrt(J[1][0]*J[1][0] + J[1][1]*J[1][1]);
                _W[ii] = node->q->w[ii] * tang * qArray[inds_0[kk]] * geom->thick[lev][inds_0[kk]];
            }

            matvec(mm, mp1, edge->ejxi_t, _S, ES);
            matvec(mm, mp1, edge->ejxi_t, _N, EN);
            matvec(mm, mp1, edge->ejxi_t, _E, EE);
            matvec(mm, mp1, edge->ejxi_t, _W, EW);

            // jump condition
            for(ii = 0; ii < mm; ii++) {
                // bottom
                kk = ii;
                bArray[inds_y[kk]] += ES[ii];

                // top
                kk = mm*mm+ii;
                bArray[inds_y[kk]] -= EN[ii];

                // left
                kk = mp1*ii;
                bArray[inds_x[kk]] += EE[ii];

                // right
                kk = mp1*ii + mm;
                bArray[inds_x[kk]] -= EW[ii];
            }
        }
    }
    VecRestoreArray(ql, &qArray);
    VecRestoreArray(bl, &bArray);

    VecScatterBegin(topo->gtol_1, bl, b, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, bl, b, INSERT_VALUES, SCATTER_REVERSE);
}

void Boundary::_assembleConv(int lev, Vec u, Vec b) {
    int ex, ey, ei, ii, kk, mm, mp1;
    int *inds_0, *inds_x, *inds_y;
    double _S[99], _N[99], _E[99], _W[99], ES[99], EN[99], EE[99], EW[99], **J, det_n;
    PetscScalar *qArray, *bArray, *uArray;

    mm = geom->quad->n;
    mp1 = mm+1;

    VecZeroEntries(bl);

    VecGetArray(ql, &qArray);
    VecGetArray(ul, &uArray); // assume this has already been scattered when ql was computed
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
                J = geom->J[ei][kk];
                det_n = sqrt(J[0][0] + J[1][0]);
                _S[ii] = node->q->w[ii] * det_n * qArray[inds_0[kk]] * geom->thick[lev][inds_0[kk]];

                // top
                kk = ii + mp1*mm;
                J = geom->J[ei][kk];
                det_n = sqrt(J[0][0] + J[1][0]);
                _N[ii] = node->q->w[ii] * det_n * qArray[inds_0[kk]] * geom->thick[lev][inds_0[kk]];

                // left
                kk = mp1*ii;
                J = geom->J[ei][kk];
                det_n = sqrt(J[0][1] + J[1][1]);
                _E[ii] = node->q->w[ii] * det_n * qArray[inds_0[kk]] * geom->thick[lev][inds_0[kk]];

                // right
                kk = mp1*ii + mm;
                J = geom->J[ei][kk];
                det_n = sqrt(J[0][1] + J[1][1]);
                _W[ii] = node->q->w[ii] * det_n * qArray[inds_0[kk]] * geom->thick[lev][inds_0[kk]];
            }

            matvec(mm, mp1, edge->ejxi_t, _S, ES);
            matvec(mm, mp1, edge->ejxi_t, _N, EN);
            matvec(mm, mp1, edge->ejxi_t, _E, EE);
            matvec(mm, mp1, edge->ejxi_t, _W, EW);

            // jump condition
            for(ii = 0; ii < mm; ii++) {
                // bottom
                kk = ii;
                bArray[inds_y[kk]] += ES[ii];

                // top
                kk = mm*mm+ii;
                bArray[inds_y[kk]] -= EN[ii];

                // left
                kk = mp1*ii;
                bArray[inds_x[kk]] += EE[ii];

                // right
                kk = mp1*ii + mm;
                bArray[inds_x[kk]] -= EW[ii];
            }
        }
    }
    VecRestoreArray(ql, &qArray);
    VecRestoreArray(bl, &bArray);

    VecScatterBegin(topo->gtol_1, bl, b, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, bl, b, INSERT_VALUES, SCATTER_REVERSE);
}

void Boundary::AssembleGrad(int lev, Vec u, Vec h, Vec b, bool upwind) {
    Interp2To0Bndry(lev, u, h, upwind);
    _assembleGrad(lev, b);
}
