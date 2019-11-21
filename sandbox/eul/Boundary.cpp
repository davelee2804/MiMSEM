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

    EQ = Alloc2D(node->n, node->q->n+1);

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &ql);
}

Boundary::~Boundary() {
    Free2D(node->n, EQ);

    VecDestroy(&ul);
    VecDestroy(&hl);
    VecDestroy(&ql);
}

void Boundary::Interp2FormTo0FormBndry(int lev, Vec u, Vec h, bool upwind) {
    int ex, ey, ii, jj[2], kk, mm, mp1;
    int *inds_0, *inds_2, *inds_x, *inds_y;
    double hi;
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
            inds_0 = topo->elInds0_l(ex, ey);
            inds_x = topo->elInds1x_l(ex, ey);
            inds_y = topo->elInds1y_l(ex, ey);
            inds_2 = topo->elInds2_l(ex, ey);

            for(ii = 0; ii < mm; ii++) {
                // bottom
                jj[0] = ii;
                jj[1] = jj[0] + 1;
                for(kk = 0; kk < 2; kk++) {
                    geom->interp2_g(ex, ey, jj[kk]%mp1, jj[kk]/mp1, hArray, &hi);
                    hi *= 1.0/geom->thick[lev][inds_0[jj[kk]]];
                    if(!upwind) {
                        qArray[jj[kk]] += 0.5*hi;
                    } else {
                        qArray[jj[kk]] += (uArray[inds_y[ii]] < 0.0) ? hi : 0.0;
                    }
                }

                // top
                jj[0] = ii + mp1*mm;
                jj[1] = jj[0] + 1;
                for(kk = 0; kk < 2; kk++) {
                    geom->interp2_g(ex, ey, jj[kk]%mp1, jj[kk]/mp1, hArray, &hi);
                    hi *= 1.0/geom->thick[lev][inds_0[jj[kk]]];
                    if(!upwind) {
                        qArray[jj[kk]] += 0.5*hi;
                    } else {
                        qArray[jj[kk]] += (uArray[inds_y[ii + mm*mm]] > 0.0) ? hi : 0.0;
                    }
                }

                // left
                jj[0] = ii*mp1;
                jj[1] = jj[0] + mp1;
                for(kk = 0; kk < 2; kk++) {
                    geom->interp2_g(ex, ey, jj[kk]%mp1, jj[kk]/mp1, hArray, &hi);
                    hi *= 1.0/geom->thick[lev][inds_0[jj[kk]]];
                    if(!upwind) {
                        qArray[jj[kk]] += 0.5*hi;
                    } else {
                        qArray[jj[kk]] += (uArray[inds_x[ii*mp1]] < 0.0) ? hi : 0.0;
                    }
                }

                // right
                jj[0] = ii*mp1 + mm;
                jj[1] = jj[0] + mp1;
                for(kk = 0; kk < 2; kk++) {
                    geom->interp2_g(ex, ey, jj[kk]%mp1, jj[kk]/mp1, hArray, &hi);
                    hi *= 1.0/geom->thick[lev][inds_0[jj[kk]]];
                    if(!upwind) {
                        qArray[jj[kk]] += 0.5*hi;
                    } else {
                        qArray[jj[kk]] += (uArray[inds_x[ii*mp1+mm]] > 0.0) ? hi : 0.0;
                    }
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
    double _S[99], _N[99], _E[99], _W[99], ES[99], EN[99], EE[99], EW[99], **J, det_n;
    PetscScalar *qArray, *uArray;

    mm = geom->quad->n;
    mp1 = mm+1;

    VecZeroEntries(ul);

    VecGetArray(ql, &qArray);
    VecGetArray(ul, &uArray);
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
                uArray[inds_y[kk]] += ES[ii];

                // top
                kk = mm*mm+ii;
                uArray[inds_y[kk]] -= EN[ii];

                // left
                kk = mp1*ii;
                uArray[inds_x[kk]] += EE[ii];

                // right
                kk = mp1*ii + mm;
                uArray[inds_x[kk]] -= EW[ii];
            }
        }
    }
    VecRestoreArray(ql, &qArray);
    VecRestoreArray(ul, &uArray);

    VecScatterBegin(topo->gtol_1, ul, b, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, ul, b, INSERT_VALUES, SCATTER_REVERSE);
}

void Boundary::AssembleGrad(int lev, Vec u, Vec h, Vec b, bool upwind) {
    Interp2FormTo0FormBndry(lev, u, h, upwind);
    _assembleGrad(lev, b);
}
