#include "Basis.h"
#include "ElMats.h"

void mult(int ni, int nj, int nk, double** A, double** B, double** C) {
    int ii, jj, kk;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii][jj] = 0.0;
            for(kk = 0; kk < nk; kk++) {
                C[ii][jj] += A[ii][kk]*B[kk][jj];
            }
        }
    }
}

void tran(int ni, int nj, double**A, double** B) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            B[jj][ii] = A[ii][jj];
        }
    }
}

// Outer product of 0-form in x and 1-form in y (columns)
// evaluated at the Gauss Lobatto quadrature points (rows)
// n: basis function order
// m: quadrature point order
M1x_j_xy_i::M1x_j_xy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii, jj;
    int mi, nj, nn, np1, mp1;
    double li, ei;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            li = l->ljxi[ii%mp1][jj%np1];
            ei = e->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = li*ei;
        }
    }
}

M1x_j_xy_i::~M1x_j_xy_i() {
    int mp1 = l->q->n + 1;
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
M1y_j_xy_i::M1y_j_xy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii, jj;
    int mi, nj, nn, np1, mp1;
    double li, ei;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            li = l->ljxi[ii/mp1][jj/nn];
            ei = e->ejxi[ii%mp1][jj%nn];
            A[ii][jj] = ei*li;
        }
    }
}

M1y_j_xy_i::~M1y_j_xy_i() {
    int mp1 = l->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// As above but with 1 forms interpolated to quadrature points
// (normal) velocity interpolated to quadrature points
M1x_j_Cxy_i::M1x_j_Cxy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

void M1x_j_Cxy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1;
    double li, ei, ck;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < nj; kk++) {
            ck += c[kk]*l->ljxi[ii%mp1][kk%np1]*e->ejxi[ii/mp1][kk/np1];
        }

        for(jj = 0; jj < nj; jj++) {
            li = l->ljxi[ii%mp1][jj%np1];
            ei = e->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = ck*li*ei;
        }
    }
}

M1x_j_Cxy_i::~M1x_j_Cxy_i() {
    int mp1 = l->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// As above but with 1 forms interpolated to quadrature points
// (normal) velocity interpolated to quadrature points
M1y_j_Cxy_i::M1y_j_Cxy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

void M1y_j_Cxy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1;
    double li, ei, ck;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < nj; kk++) {
            ck += c[kk]*e->ejxi[ii%mp1][kk%nn]*l->ljxi[ii/mp1][kk/nn];
        }

        for(jj = 0; jj < nj; jj++) {
            ei = e->ejxi[ii%mp1][jj%nn];
            li = l->ljxi[ii/mp1][jj/nn];
            A[ii][jj] = ck*ei*li;
        }
    }
}

M1y_j_Cxy_i::~M1y_j_Cxy_i() {
    int mp1 = l->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// As above but with 1 forms cross product interpolated to quadrature points
// (tangent) velocity interpolated to quadrature points
M1x_j_Exy_i::M1x_j_Exy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

M1x_j_Exy_i::~M1x_j_Exy_i() {
    int mp1 = l->q->n + 1;
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

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < nj; kk++) {
            ck += c[kk]*e->ejxi[ii%mp1][kk%nn]*l->ljxi[ii/mp1][kk/nn];
        }

        for(jj = 0; jj < nj; jj++) {
            li = l->ljxi[ii%mp1][jj%np1];
            ei = e->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = ck*li*ei;
        }
    }
}

// As above but with 1 forms (cross product) interpolated to quadrature points
// (tangent) velocity interpolated to quadrature points
M1y_j_Exy_i::M1y_j_Exy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

M1y_j_Exy_i::~M1y_j_Exy_i() {
    int mp1 = l->q->n + 1;
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

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk= 0; kk < nj; kk++) {
            ck += c[kk]*l->ljxi[ii%mp1][kk%np1]*e->ejxi[ii/mp1][kk/np1];
        }

        for(jj = 0; jj < nj; jj++) {
            ei = e->ejxi[ii%mp1][jj%nn];
            li = l->ljxi[ii/mp1][jj/nn];
            A[ii][jj] = ck*ei*li;
        }
    }
}

// As above but with 0 forms interpolated to quadrature points
// potential vorticity interpolated to quadrature points
M1x_j_Dxy_i::M1x_j_Dxy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

M1x_j_Dxy_i::~M1x_j_Dxy_i() {
    int mp1 = l->q->n + 1;
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

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;
    n2 = np1*np1;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*l->ljxi[ii%mp1][kk%np1]*l->ljxi[ii/mp1][kk/np1];
        }

        for(jj = 0; jj < nj; jj++) {
            li = l->ljxi[ii%mp1][jj%np1];
            ei = e->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = ck*li*ei;
        }
    }
}

// As above but with 0 forms interpolated to quadrature points
// potential vorticity interpolated to quadrature points
M1y_j_Dxy_i::M1y_j_Dxy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

M1y_j_Dxy_i::~M1y_j_Dxy_i() {
    int mp1 = l->q->n + 1;
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

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;
    n2 = np1*np1;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*l->ljxi[ii%mp1][kk%np1]*l->ljxi[ii/mp1][kk/np1];
        }

        for(jj = 0; jj < nj; jj++) {
            ei = e->ejxi[ii%mp1][jj%nn];
            li = l->ljxi[ii/mp1][jj/nn];
			A[ii][jj] = ck*ei*li;
        }
    }
}

// thickness interpolated to quadrature points
// for diagnosis of hu
M1x_j_Fxy_i::M1x_j_Fxy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

M1x_j_Fxy_i::~M1x_j_Fxy_i() {
    int mp1 = l->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

void M1x_j_Fxy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1, n2;
    double li, ei, ck;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;
    n2 = nn*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*e->ejxi[ii%mp1][kk%nn]*e->ejxi[ii/mp1][kk/nn];
        }

        for(jj = 0; jj < nj; jj++) {
            li = l->ljxi[ii%mp1][jj%np1];
            ei = e->ejxi[ii/mp1][jj/np1];
            A[ii][jj] = ck*ei*li;
        }
    }
}

// thickness interpolated to quadrature points
// for diagnosis of hv
M1y_j_Fxy_i::M1y_j_Fxy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

M1y_j_Fxy_i::~M1y_j_Fxy_i() {
    int mp1 = l->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

void M1y_j_Fxy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1, n2;
    double li, ei, ck;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;
    n2 = nn*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*e->ejxi[ii%mp1][kk%nn]*e->ejxi[ii/mp1][kk/nn];
        }

        for(jj = 0; jj < nj; jj++) {
            ei = e->ejxi[ii%mp1][jj%nn];
            li = l->ljxi[ii/mp1][jj/nn];
            A[ii][jj] = ck*ei*li;
        }
    }
}

// Outer product of 1-form in x and 1-form in y
// evaluated at the Gauss Lobatto quadrature points
// n: basis function order
// m: quadrature point order
M2_j_xy_i::M2_j_xy_i(LagrangeEdge* _e) {
    int ii, jj, nn, mp1, mi, nj;
    double ei, ej;

    e = _e;

    mp1 = e->l->q->n + 1;
    mi = mp1*mp1;
    nn = e->l->n;
    nj = nn*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            ei = e->ejxi[ii%mp1][jj%nn];
            ej = e->ejxi[ii/mp1][jj/nn];
            A[ii][jj] = ei*ej;
        }        
    }
}

M2_j_xy_i::~M2_j_xy_i() {
    int ii, mp1, mi;

    mp1 = e->l->q->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// 0 form basis function terms (j) evaluated at the
// quadrature points (i)
M0_j_xy_i::M0_j_xy_i(LagrangeNode* _l) {
    int ii, jj, np1, mp1, mi, nj;
    double li, lj;

    l = _l;

    mp1 = l->q->n + 1;
    np1 = l->n + 1;
    mi = mp1*mp1;
    nj = np1*np1;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            li = l->ljxi[ii%mp1][jj%np1];
            lj = l->ljxi[ii/mp1][jj/np1];
            A[ii][jj] = li*lj;
        }
    }
}

M0_j_xy_i::~M0_j_xy_i() {
    int ii, mp1, mi;

    mp1 = l->q->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// left hand side of the potential vorticty equation
// assembles the nonlinear term hq
M0_j_Cxy_i::M0_j_Cxy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii, nn, np1, mp1, mi, nj;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*np1;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

M0_j_Cxy_i::~M0_j_Cxy_i() {
    int ii, mp1, mi;

    mp1 = e->l->q->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

void M0_j_Cxy_i::assemble(double* c) {
    int ii, jj, kk, nn, n2, np1, mp1, mi, nj;
    double ck, li, lj;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*np1;
    n2 = nn*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < n2; kk++) {
            ck += c[kk]*e->ejxi[ii%mp1][kk%nn]*e->ejxi[ii/mp1][kk/nn];
        }

        for(jj = 0; jj < nj; jj++) {
            li = l->ljxi[ii%mp1][jj%np1];
            lj = l->ljxi[ii/mp1][jj/np1];
            A[ii][jj] = ck*li*lj;
        }
    }
}

// Quadrature weights diagonal matrix
Wii::Wii(GaussLobatto* _q) {
    int ii, jj, mp1, mi;

    q = _q;

    mp1 = q->n + 1;
    mi = mp1*mp1;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[mi];
        for(jj = 0; jj < mi; jj ++) {
            A[ii][jj] = 0.0;
        }
    }

    for(ii = 0; ii < mi; ii++) {
        A[ii][ii] = q->x[ii%mp1]*q->x[ii/mp1];
    }
}

Wii::~Wii() {
    int ii, mp1, mi;

    mp1 = q->n + 1;
    mi = mp1*mp1;

    for(ii = 0; ii < mi; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}
