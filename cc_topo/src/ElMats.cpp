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

// As above but with 1 forms interpolated to quadrature points
// (normal) velocity interpolated to quadrature points
M1x_j_Cxy_i::M1x_j_Cxy_i(LagrangeNode* _node, LagrangeEdge* _edge, Geom* _geom) {
    int ii;
    int mi, nj, nn, np1, mp1;

    node = _node;
    edge = _edge;
    geom = _geom;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    nDofsI = mi;
    nDofsJ = nj;
}

void M1x_j_Cxy_i::assemble(int ex, int ey, double* cx, double* cy) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1;
    double li, ei, ckx, cky, jac, uz, um, fac;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ckx = 0.0;
        cky = 0.0;
        for(kk = 0; kk < nj; kk++) {
            ckx += cx[kk]*node->ljxi[ii%mp1][kk%np1]*edge->ejxi[ii/mp1][kk/np1];
            cky += cy[kk]*edge->ejxi[ii%mp1][kk%nn]*node->ljxi[ii/mp1][kk/nn];
        }
        jac = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
        uz = (J[0][0]*ckx + J[0][1]*cky)/jac;
        um = (J[1][0]*ckx + J[1][1]*cky)/jac;
        fac = 0.5*(uz*J[0][0] + um*J[1][0]);

        for(jj = 0; jj < nj; jj++) {
            li = node->ljxi[ii%mp1][jj%np1];
            ei = edge->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = fac*li*ei;
        }
    }
}

M1x_j_Cxy_i::~M1x_j_Cxy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;

    delete[] J[0];
    delete[] J[1];
    delete[] J;
}

// As above but with 1 forms interpolated to quadrature points
// (normal) velocity interpolated to quadrature points
M1y_j_Cxy_i::M1y_j_Cxy_i(LagrangeNode* _node, LagrangeEdge* _edge, Geom* _geom) {
    int ii;
    int mi, nj, nn, np1, mp1;

    node = _node;
    edge = _edge;
    geom = _geom;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    nDofsI = mi;
    nDofsJ = nj;
}

void M1y_j_Cxy_i::assemble(int ex, int ey, double* cx, double* cy) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1;
    double li, ei, ckx, cky, jac, uz, um, fac;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ckx = 0.0;
        cky = 0.0;
        for(kk = 0; kk < nj; kk++) {
            ckx += cx[kk]*node->ljxi[ii%mp1][kk%np1]*edge->ejxi[ii/mp1][kk/np1];
            cky += cy[kk]*edge->ejxi[ii%mp1][kk%nn]*node->ljxi[ii/mp1][kk/nn];
        }
        jac = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
        uz = (J[0][0]*ckx + J[0][1]*cky)/jac;
        um = (J[1][0]*ckx + J[1][1]*cky)/jac;
        fac = 0.5*(uz*J[0][1] + um*J[1][1]);

        for(jj = 0; jj < nj; jj++) {
            ei = edge->ejxi[ii%mp1][jj%nn];
            li = node->ljxi[ii/mp1][jj/nn];
            A[ii][jj] = fac*ei*li;
        }
    }
}

M1y_j_Cxy_i::~M1y_j_Cxy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;

    delete[] J[0];
    delete[] J[1];
    delete[] J;
}

// As above but with 1 forms cross product interpolated to quadrature points
// (tangent) velocity interpolated to quadrature points
M1x_j_Exy_i::M1x_j_Exy_i(LagrangeNode* _node, LagrangeEdge* _edge) {
    int ii;
    int mi, nj, nn, np1, mp1;

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
}

M1x_j_Exy_i::~M1x_j_Exy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

void M1x_j_Exy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1;
    double li, ei, ck;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < nj; kk++) {
            ck += c[kk]*edge->ejxi[ii%mp1][kk%nn]*node->ljxi[ii/mp1][kk/nn];
        }

        for(jj = 0; jj < nj; jj++) {
            li = node->ljxi[ii%mp1][jj%np1];
            ei = edge->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = ck*li*ei;
        }
    }
}

// As above but with 1 forms (cross product) interpolated to quadrature points
// (tangent) velocity interpolated to quadrature points
M1y_j_Exy_i::M1y_j_Exy_i(LagrangeNode* _node, LagrangeEdge* _edge) {
    int ii;
    int mi, nj, nn, np1, mp1;

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
}

M1y_j_Exy_i::~M1y_j_Exy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

void M1y_j_Exy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1;
    double li, ei, ck;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk= 0; kk < nj; kk++) {
            ck += c[kk]*node->ljxi[ii%mp1][kk%np1]*edge->ejxi[ii/mp1][kk/np1];
        }

        for(jj = 0; jj < nj; jj++) {
            ei = edge->ejxi[ii%mp1][jj%nn];
            li = node->ljxi[ii/mp1][jj/nn];
            A[ii][jj] = ck*ei*li;
        }
    }
}

// As above but with 0 forms interpolated to quadrature points
// potential vorticity interpolated to quadrature points
M1x_j_Dxy_i::M1x_j_Dxy_i(LagrangeNode* _node, LagrangeEdge* _edge) {
    int ii;
    int mi, nj, nn, np1, mp1;

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
}

M1x_j_Dxy_i::~M1x_j_Dxy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

void M1x_j_Dxy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1, n2;
    double li, ei, ck;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;
    n2 = np1*np1;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*node->ljxi[ii%mp1][kk%np1]*node->ljxi[ii/mp1][kk/np1];
        }

        for(jj = 0; jj < nj; jj++) {
            li = node->ljxi[ii%mp1][jj%np1];
            ei = edge->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = ck*li*ei;
        }
    }
}

// As above but with 0 forms interpolated to quadrature points
// potential vorticity interpolated to quadrature points
M1y_j_Dxy_i::M1y_j_Dxy_i(LagrangeNode* _node, LagrangeEdge* _edge) {
    int ii;
    int mi, nj, nn, np1, mp1;

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
}

M1y_j_Dxy_i::~M1y_j_Dxy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

void M1y_j_Dxy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1, n2;
    double li, ei, ck;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;
    n2 = np1*np1;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*node->ljxi[ii%mp1][kk%np1]*node->ljxi[ii/mp1][kk/np1];
        }

        for(jj = 0; jj < nj; jj++) {
            ei = edge->ejxi[ii%mp1][jj%nn];
            li = node->ljxi[ii/mp1][jj/nn];
            A[ii][jj] = ck*ei*li;
        }
    }
}

// thickness interpolated to quadrature points for diagnosis of hu
M1x_j_Fxy_i::M1x_j_Fxy_i(LagrangeNode* _node, LagrangeEdge* _edge, Geom* _geom) {
    int ii;
    int mi, nj, nn, np1, mp1;

    node = _node;
    edge = _edge;
    geom = _geom;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    nDofsI = mi;
    nDofsJ = nj;
}

M1x_j_Fxy_i::~M1x_j_Fxy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;

    delete[] J[0];
    delete[] J[1];
    delete[] J;
}

void M1x_j_Fxy_i::assemble(int ex, int ey, double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1, n2;
    double li, ei, ck, jac, jacInv;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;
    n2 = nn*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*edge->ejxi[ii%mp1][kk%nn]*edge->ejxi[ii/mp1][kk/nn];
        }
        jac = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
        jacInv = 1.0/jac;

        for(jj = 0; jj < nj; jj++) {
            li = node->ljxi[ii%mp1][jj%np1];
            ei = edge->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = jacInv*ck*ei*li;
            //A[ii][jj] = ck*ei*li;
        }
    }
}

// thickness interpolated to quadrature points for diagnosis of hv
M1y_j_Fxy_i::M1y_j_Fxy_i(LagrangeNode* _node, LagrangeEdge* _edge, Geom* _geom) {
    int ii;
    int mi, nj, nn, np1, mp1;

    node = _node;
    edge = _edge;
    geom = _geom;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    nDofsI = mi;
    nDofsJ = nj;
}

M1y_j_Fxy_i::~M1y_j_Fxy_i() {
    int mp1 = node->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;

    delete[] J[0];
    delete[] J[1];
    delete[] J;
}

void M1y_j_Fxy_i::assemble(int ex, int ey, double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1, n2;
    double li, ei, ck, jac, jacInv;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;
    n2 = nn*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*edge->ejxi[ii%mp1][kk%nn]*edge->ejxi[ii/mp1][kk/nn];
        }
        jac = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
        jacInv = 1.0/jac;

        for(jj = 0; jj < nj; jj++) {
            ei = edge->ejxi[ii%mp1][jj%nn];
            li = node->ljxi[ii/mp1][jj/nn];
            A[ii][jj] = jacInv*ck*ei*li;
            //A[ii][jj] = ck*ei*li;
        }
    }
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

// left hand side of the potential vorticty equation
// assembles the nonlinear term hq
M0_j_Cxy_i::M0_j_Cxy_i(LagrangeNode* _node, LagrangeEdge* _edge) {
    int ii, nn, np1, mp1, mi, nj;

    node = _node;
    edge = _edge;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*np1;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
    nDofsI = mi;
    nDofsJ = nj;
}

M0_j_Cxy_i::~M0_j_Cxy_i() {
    int ii, mp1, mi;

    mp1 = edge->l->q->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

void M0_j_Cxy_i::assemble(double* c) {
    int ii, jj, kk, nn, n2, np1, mp1, mi, nj;
    double ck, li, lj;

    nn = node->n;
    np1 = nn + 1;
    mp1 = node->q->n + 1;
    mi = mp1*mp1;
    nj = np1*np1;
    n2 = nn*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*edge->ejxi[ii%mp1][kk%nn]*edge->ejxi[ii/mp1][kk/nn];
        }

        for(jj = 0; jj < nj; jj++) {
            li = node->ljxi[ii%mp1][jj%np1];
            lj = node->ljxi[ii/mp1][jj/np1];
            A[ii][jj] = ck*li*lj;
        }
    }
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
}

void Wii::assemble(int ex, int ey) {
    int ii, mp1, mi;
    double jac;

    mp1 = quad->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        jac = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
        A[ii][ii] = jac*quad->w[ii%mp1]*quad->w[ii/mp1];
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

// jacobian determinant inverse at quadrature point element matrix
// for integrating 2 forms
JacM2::JacM2(GaussLobatto* _quad, Geom* _geom) {
    int ii, jj, mp1, mi;

    quad = _quad;
    geom = _geom;

    mp1 = quad->n + 1;
    mi = mp1*mp1;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[mi];
        for(jj = 0; jj < mi; jj++) {
            A[ii][jj] = 0.0;
        }
    }

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    nDofsI = mi;
    nDofsJ = mi;
}

void JacM2::assemble(int ex, int ey) {
    int ii, mp1, mi;
    double jac;

    mp1 = quad->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        jac = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
        A[ii][ii] = 1.0/jac;
    }
}

JacM2::~JacM2() {
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

// jacobian determinant inverse at quadrature point element matrix
// for integrating 1 forms (piola transform)
JacM1::JacM1(GaussLobatto* _quad, Geom* _geom) {
    int ii, jj, mp1, mi;

    quad = _quad;
    geom = _geom;

    mp1 = quad->n + 1;
    mi = mp1*mp1;

    Aaa = new double*[mi];
    Aab = new double*[mi];
    Aba = new double*[mi];
    Abb = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        Aaa[ii] = new double[mi];
        Aab[ii] = new double[mi];
        Aba[ii] = new double[mi];
        Abb[ii] = new double[mi];
        for(jj = 0; jj < mi; jj++) {
            Aaa[ii][jj] = 0.0;
            Aab[ii][jj] = 0.0;
            Aba[ii][jj] = 0.0;
            Abb[ii][jj] = 0.0;
        }
    }

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    nDofsI = mi;
    nDofsJ = mi;
}

void JacM1::assemble(int ex, int ey) {
    int ii, mp1, mi;
    double jac, jacInv;

    mp1 = quad->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        jac = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
        jacInv = 1.0/jac;
        Aaa[ii][ii] = J[0][0]*jacInv;
        Aab[ii][ii] = J[0][1]*jacInv;
        Aba[ii][ii] = J[1][0]*jacInv;
        Abb[ii][ii] = J[1][1]*jacInv;
    }
}

JacM1::~JacM1() {
    int ii, mp1, mi;

    mp1 = quad->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        delete[] Aaa[ii];
        delete[] Aab[ii];
        delete[] Aba[ii];
        delete[] Abb[ii];
    }
    delete[] Aaa;
    delete[] Aab;
    delete[] Aba;
    delete[] Abb;

    delete[] J[0];
    delete[] J[1];
    delete[] J;
}
