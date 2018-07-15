#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <petscvec.h>

#include "Basis.h"
#include "Topo.h"
#include "Geom.h"

using namespace std;
using std::string;

#define RAD_SPHERE 6371220.0
//#define RAD_SPHERE 1.0

Geom::Geom(Topo* _topo, int _nk, double _lx) {
    double dx;
    int ii, jj;
    ifstream file;
    string line;

    topo = _topo;
    nk   = _nk;
    lx   = _lx;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    x = new double[topo->elOrd*topo->nElsX];

    // initialise the geometry
    dx = lx/topo->nElsX;
    for(ii = 0; ii < topo->nElsX; ii++) {
        for(jj = 0; jj < topo->elOrd; jj++) {
            x[ii*topo->elOrd+jj] = ii*dx + dx*0.5*(quad->x[jj] + 1.0);
        }
    }

    det = new double*[topo->nElsX];
    for(ii = 0; ii < topo->nElsX; ii++) {
        det[ii] = new double[(quad->n+1)];
    }
    initJacobians();

    // initialise vertical level arrays
    topog = new double[topo->n0];
    levs = new double*[nk+1];
    for(ii = 0; ii < nk + 1; ii++) {
        levs[ii] = new double[topo->n0];
    }
    thick = new double*[nk];
    for(ii = 0; ii < nk; ii++) {
        thick[ii] = new double[topo->n0];
    }
}

Geom::~Geom() {
    int ii;

    delete[] x;

    for(ii = 0; ii < topo->nElsX; ii++) {
        delete[] det[ii];
    }
    delete[] det;

    delete edge;
    delete node;
    delete quad;

    // free the topography
    delete[] topog;
    for(ii = 0; ii < nk + 1; ii++) {
        delete[] levs[ii];
    }
    delete[] levs;
    for(ii = 0; ii < nk; ii++) {
        delete[] thick[ii];
    }
    delete[] thick;
}


double Geom::jacDet(int ex, int px) {
    return det[ex][px];
}

void Geom::interp0(int ex, int px, double* vec, double* val) {
    int* inds0 = topo->elInds0(ex);

    // assumes diagonal mass matrix for 0 forms
    val[0] = vec[inds0[px]];
}

void Geom::interp1(int ex, int px, double* vec, double* val) {
    int jj, np1;
    int *inds1;

    inds1 = topo->elInds1(ex);

    np1 = topo->elOrd + 1;

    val[0] = 0.0;
    for(jj = 0; jj < np1; jj++) {
        val[0] += vec[inds1[jj]]*node->ljxi[px][jj];
    }
}

void Geom::interp2(int ex, int px, double* vec, double* val) {
    int jj;
    int* inds2 = topo->elInds2(ex);
    double dj = det[ex][px];

    val[0] = 0.0;
    for(jj = 0; jj < topo->elOrd; jj++) {
        val[0] += vec[inds2[jj]]*edge->ejxi[px][jj];
    }
    val[0] /= dj;
}

void Geom::write0(Vec *q, char* fieldname, int tstep) {
    int ex, ii, jj, mp1, kk;
    int* inds0;
    char filename[100];
    ofstream file;
    PetscScalar *qArray, *qxArray;
    Vec qx;

    mp1 = quad->n + 1;

    sprintf(filename, "%s_%.4u.dat", fieldname, tstep);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &qx);
    for(kk = 0; kk < nk; kk++) {
        VecZeroEntries(qx);
        VecGetArray(q[kk], &qArray);
        VecGetArray(qx, &qxArray);
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0(ex);
            for(ii = 0; ii < mp1; ii++) {
                jj = inds0[ii];
                qxArray[jj]  = qArray[jj];
                // assume piecewise constant in the vertical, so rescale by
                // the vertical determinant inverse
                qxArray[jj] *= 2.0/thick[kk][jj];
            }
        }
        VecRestoreArray(q[kk], &qArray);
        VecRestoreArray(qx, &qxArray);

        file.open(filename, ios::out | ios::app);
        VecGetArray(qx, &qxArray);
        for(ii = 0; ii < topo->n0; ii++) {
            file << qxArray[ii] << endl;
        }
        VecRestoreArray(qx, &qxArray);
        file.close();
    }
    VecDestroy(&qx);
}

void Geom::write1(Vec *q, char* fieldname, int tstep) {
    int ex, ii, jj, mp1, kk;
    int* inds0;
    char filename[100];
    ofstream file;
    PetscScalar *qArray, *qxArray;
    Vec qx;

    mp1 = quad->n + 1;

    sprintf(filename, "%s_%.4u.dat", fieldname, tstep);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &qx);
    for(kk = 0; kk < nk; kk++) {
        VecZeroEntries(qx);
        VecGetArray(q[kk], &qArray);
        VecGetArray(qx, &qxArray);
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0(ex);
            for(ii = 0; ii < mp1; ii++) {
                jj = inds0[ii];
                qxArray[jj]  = qArray[jj];
                // assume piecewise constant in the vertical, so rescale by
                // the vertical determinant inverse
                qxArray[jj] *= 2.0/thick[kk][jj];
            }
        }
        VecRestoreArray(q[kk], &qArray);
        VecRestoreArray(qx, &qxArray);

        file.open(filename, ios::out | ios::app);
        VecGetArray(qx, &qxArray);
        for(ii = 0; ii < topo->n0; ii++) {
            file << qxArray[ii] << endl;
        }
        VecRestoreArray(qx, &qxArray);
        file.close();
    }
    VecDestroy(&qx);
}

// interpolate 2 form field to quadrature points
void Geom::write2(Vec *h, char* fieldname, int tstep, bool const_vert) {
    int ex, ii, kk, mp1;
    int nv = (const_vert) ? nk : nk + 1;
    int *inds0;
    char filename[100];
    double val;
    ofstream file;
    Vec hx;
    PetscScalar *hxArray, *hArray;

    mp1 = quad->n + 1;

    sprintf(filename, "%s_%.4u.dat", fieldname, tstep);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &hx);
    for(kk = 0; kk < nv; kk++) {
        VecZeroEntries(hx);
        VecGetArray(h[kk], &hArray);
        VecGetArray(hx, &hxArray);
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0(ex);

            // loop over quadrature points
            for(ii = 0; ii < mp1; ii++) {
                interp2(ex, ii, hArray, &val);

                hxArray[inds0[ii]] = val;
                // assume piecewise constant in the vertical, so rescale by
                // the vertical determinant inverse
                if(const_vert) {
                    hxArray[inds0[ii]] *= 2.0/thick[kk][inds0[ii]];
                }
            }
        }
        VecRestoreArray(h[kk], &hArray);
        VecRestoreArray(hx, &hxArray);

        file.open(filename, ios::out | ios::app);
        VecGetArray(hx, &hxArray);
        for(ii = 0; ii < topo->n2; ii++) {
            file << hxArray[ii] << endl;
        }
        VecRestoreArray(hx, &hxArray);
        file.close();
    }
    VecDestroy(&hx);
}

void Geom::writeSerial(Vec* vecs, char* fieldname, int tstep, int nv) {
    int ex, ii, jj, kk, mp1;
    int *inds0;
    char filename[100];
    double val;
    ofstream file;
    Vec hx;
    PetscScalar *hxArray, *hArray;

    mp1 = quad->n + 1;

    sprintf(filename, "%s_%.4u.dat", fieldname, tstep);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &hx);
    for(kk = 0; kk < nv; kk++) {
        VecZeroEntries(hx);
        VecGetArray(hx, &hxArray);
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0(ex);

            // loop over quadrature points
            VecGetArray(vecs[ex], &hArray);
            for(ii = 0; ii < mp1; ii++) {
                val = 0.0;
                for(jj = 0; jj < edge->n; jj++) {
                    val += edge->ejxi[ii][jj]*hArray[kk*edge->n+jj];
                }

                hxArray[inds0[ii]] = val;
                // assume piecewise constant in the vertical, so rescale by
                // the vertical determinant inverse
                hxArray[inds0[ii]] *= 2.0/thick[kk][inds0[ii]];
            }
            VecRestoreArray(vecs[ex], &hArray);
        }
        VecRestoreArray(hx, &hxArray);

        file.open(filename, ios::out | ios::app);
        VecGetArray(hx, &hxArray);
        for(ii = 0; ii < topo->n2; ii++) {
            file << hxArray[ii] << endl;
        }
        VecRestoreArray(hx, &hxArray);
        file.close();
    }
    VecDestroy(&hx);
}

void Geom::initJacobians() {
    int ex, mp1, ii;

    mp1 = quad->n + 1;

    for(ex = 0; ex < topo->nElsX; ex++) {
        for(ii = 0; ii < mp1; ii++) {
            det[ex][ii] = (lx/topo->nElsX)/2.0;
        }
    }
}

void Geom::initTopog(TopogFunc* ft, LevelFunc* fl) {
    int ii, jj;
    double zo;
    double max_height = (fl) ? fl(x[0], nk) : 1.0;//TODO: assumes x[0] is at the sea surface, fix later

    for(ii = 0; ii < topo->n0; ii++) {
        topog[ii] = ft(x[ii]);
    }

    for(ii = 0; ii < nk + 1; ii++) {
        for(jj = 0; jj < topo->n0; jj++) {
            if(fl) {
                zo = fl(x[jj], ii);
                levs[ii][jj] = (max_height - topog[jj])*zo/max_height + topog[jj];
            }
            else {
                levs[ii][jj] = 2.0;
            }
if(!jj) cout << ii << "\tlevel: " << levs[ii][jj] << endl;
        }
    }
    for(ii = 0; ii < nk; ii++) {
        for(jj = 0; jj < topo->n0; jj++) {
            if(fl) {
                thick[ii][jj] = levs[ii+1][jj] - levs[ii][jj];
            }
            else {
                thick[ii][jj] = 2.0;//TODO check this
            }
if(!jj) cout << ii << "\tthick:  " << thick[ii][jj] << endl;
        }
    }
}
