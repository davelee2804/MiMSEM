#include <iostream>
#include <cmath>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"

using namespace std;

// Outer product of 0-form in x and 1-form in y (columns)
// evaluated at the Gauss Lobatto quadrature points (rows)
// n: basis function order
// m: quadrature point order
M1x_j_xy_i::M1x_j_xy_i(LagrangeNode* _node, LagrangeEdge* _edge) {
    int ii, jj;
    int mi, nj, nn, np1, mp1;
    double li, ei;

    node = _node;
    edge = _edge;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
    nDofsI = mi;
    nDofsJ = nj;

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            li = node->ljxi[ii%mp1][jj%np1];
            ei = edge->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = li*ei;
        }
    }
}

M1x_j_xy_i::~M1x_j_xy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// Outer product of 1-form in x and 0-form in y (columns)
// evaluated at the Gauss Lobatto quadrature points (rows)
// n: basis function order
// m: quadrature point order
M1y_j_xy_i::M1y_j_xy_i(LagrangeNode* _node, LagrangeEdge* _edge) {
    int ii, jj;
    int mi, nj, nn, np1, mp1;
    double li, ei;

    node = _node;
    edge = _edge;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
    nDofsI = mi;
    nDofsJ = nj;

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            li = node->ljxi[ii/mp1][jj/nn];
            ei = edge->ejxi[ii%mp1][jj%nn];
            A[ii][jj] = ei*li;
        }
    }
}

M1y_j_xy_i::~M1y_j_xy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// Outer product of 1-form in x and 1-form in y
// evaluated at the Gauss Lobatto quadrature points
// n: basis function order
// m: quadrature point order
M2_j_xy_i::M2_j_xy_i(LagrangeEdge* _edge) {
    int ii, jj, nn, mp1, mi, nj;
    double ei, ej;

    edge = _edge;

    mp1 = edge->l->q->n + 1;
    mi = mp1*mp1;
    nn = edge->l->n;
    nj = nn*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
    nDofsI = mi;
    nDofsJ = nj;

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            ei = edge->ejxi[ii%mp1][jj%nn];
            ej = edge->ejxi[ii/mp1][jj/nn];
            A[ii][jj] = ei*ej;
        }        
    }
}

M2_j_xy_i::~M2_j_xy_i() {
    int ii, mp1, mi;

    mp1 = edge->l->q->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// 0 form basis function terms (j) evaluated at the
// quadrature points (i)
M0_j_xy_i::M0_j_xy_i(LagrangeNode* _node) {
    int ii, jj, np1, mp1, mi, nj;
    double li, lj;

    node = _node;

    mp1 = node->q->n + 1;
    np1 = node->n + 1;
    mi = mp1*mp1;
    nj = np1*np1;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
    nDofsI = mi;
    nDofsJ = nj;

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            li = node->ljxi[ii%mp1][jj%np1];
            lj = node->ljxi[ii/mp1][jj/np1];
            A[ii][jj] = li*lj;
        }
    }
}

M0_j_xy_i::~M0_j_xy_i() {
    int ii, mp1, mi;

    mp1 = node->q->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// Quadrature weights diagonal matrix
Wii::Wii(GaussLobatto* _quad, Geom* _geom) {
    int ii, jj, mp1, mi;

    quad = _quad;
    geom = _geom;

    mp1 = quad->n + 1;
    mi = mp1*mp1;

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[mi];
        for(jj = 0; jj < mi; jj ++) {
            A[ii][jj] = 0.0;
        }
    }
    nDofsI = mi;
    nDofsJ = mi;

    assemble();
}

void Wii::assemble() {
    int ii, mp1, mi;

    mp1 = quad->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        A[ii][ii] = quad->w[ii%mp1]*quad->w[ii/mp1];
    }
}

Wii::~Wii() {
    int ii, mp1, mi;

    mp1 = quad->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        delete[] A[ii];
    }
    delete[] A;

    delete[] J[0];
    delete[] J[1];
    delete[] J;
}
