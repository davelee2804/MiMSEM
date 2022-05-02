#include <cmath>

#include "LinAlg.h"

// Allocate 2D array
double** Alloc2D(int ni, int nj) {
    int ii, jj;
    double** A;

    A = new double*[ni];
    for(ii = 0; ii < ni; ii++) {
        A[ii] = new double[nj];
        for(jj = 0; jj < nj; jj++) {
            A[ii][jj] = 0.0;
        }
    }

    return A;
}

// Free 2D array
void Free2D(int ni, double** A) {
    int ii;

    for(ii = 0; ii < ni; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// Flatten a 2D array into a 1D array
double* Flat2D(int ni, int nj, double** A) {
    int ii, jj, kk;
    double* Aflat = new double[ni*nj];

    kk = 0;
    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            Aflat[kk] = A[ii][jj];
            kk++;
        }
    }

    return Aflat;
}

// Flatten a 2D array into a 1D array supplied
void Flat2D_IP(int ni, int nj, double** A, double* Aflat) {
    int ii, jj, kk;

    kk = 0;
    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            Aflat[kk] = A[ii][jj];
            kk++;
        }
    }
}

// Multiply two matrices into a thrid (allocated)
double** Mult(int ni, int nj, int nk, double** A, double** B) {
    int ii, jj, kk;
    double** C;

    C = new double*[ni];
    for(ii = 0; ii < ni; ii++) {
        C[ii] = new double[nj];
    }

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii][jj] = 0.0;
            for(kk = 0; kk < nk; kk++) {
                C[ii][jj] += A[ii][kk]*B[kk][jj];
            }
        }
    }

    return C;
}

// Multiply two matrices into a thrid (supplied)
void Mult_IP(int ni, int nj, int nk, double** A, double** B, double** C) {
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

// Multiply full by diagonal matrix
void Mult_FD_IP(int ni, int nj, double** A, double* B, double** C) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii][jj] = A[ii][jj]*B[jj];
        }
    }
}


// Matrix transpose into a new matrix
double** Tran(int ni, int nj, double** A) {
    int ii, jj;
    double** B;

    B = new double*[nj];
    for(jj = 0; jj < nj; jj++) {
        B[jj] = new double[ni];
    }

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            B[jj][ii] = A[ii][jj];
        }
    }

    return B;
}

// Matrix transpose into a supplied matrix
void Tran_IP(int ni, int nj, double** A, double** B) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            B[jj][ii] = A[ii][jj];
        }
    }
}

void Ax_b(int ni, int nj, double** A, double* x, double* b) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        b[ii] = 0.0;
        for(jj = 0; jj < nj; jj++) {
            b[ii] += A[ii][jj]*x[jj];
        }
    }
}

double ArcLen(double* a, double* b, double rad) {
    double cx, cy, cz, c;

    cx = b[0] - a[0];
    cy = b[1] - a[1];
    cz = b[2] - a[2];

    c = sqrt(cx*cx + cy*cy + cz*cz);

    return rad*2.0*asin(c/(2.0*rad));
}

bool ArcInt(double rad, double* ai, double* af, double* bi, double* bf, double* xo) {
    double l_ai_af, l_ai_o, l_af_o, l_bi_bf, l_bi_o, l_bf_o, rad2;
    double va[3], vb[3], vl[3], vr[3];

    // Construct a vector normal to the plane of the great circle made by each of the two arcs
    va[0] = ai[1]*af[2] - af[1]*ai[2];
    va[1] = af[0]*ai[2] - ai[0]*af[2];
    va[2] = ai[0]*af[1] - af[0]*ai[1];

    vb[0] = bi[1]*bf[2] - bf[1]*bi[2];
    vb[1] = bf[0]*bi[2] - bi[0]*bf[2];
    vb[2] = bi[0]*bf[1] - bf[0]*bi[1];

    // Find the coordinate of the intersecting planes as the cross product of the vectors defining 
    // the planes of the great circles
    vl[0] = va[1]*vb[2] - vb[1]*va[2];
    vl[1] = vb[0]*va[2] - va[0]*vb[2];
    vl[2] = va[0]*vb[1] - vb[0]*va[1];

    // Normalize
    rad2 = sqrt(vl[0]*vl[0] + vl[1]*vl[1] + vl[2]*vl[2]);
    vl[0] = rad*vl[0]/rad2;
    vl[1] = rad*vl[1]/rad2;
    vl[2] = rad*vl[2]/rad2;
    vr[0] = -vl[0];
    vr[1] = -vl[1];
    vr[2] = -vl[2];

    // Check that the intersection point lies within the limits of the two arcs
    l_ai_af = ArcLen(ai, af, rad);
    l_ai_o  = ArcLen(ai, vl, rad);
    l_af_o  = ArcLen(af, vl, rad);

    l_bi_bf = ArcLen(bi, bf, rad);
    l_bi_o  = ArcLen(bi, vl, rad);
    l_bf_o  = ArcLen(bf, vl, rad);

    if(fabs(l_ai_af - l_ai_o - l_af_o) < 1.0e-6 && fabs(l_bi_bf - l_bi_o - l_bf_o) < 1.0e-6) {
        xo[0] = vl[0];
        xo[1] = vl[1];
        xo[2] = vl[2];
        return true;
    }

    // Check the antipodal intersection point
    l_ai_o  = ArcLen(ai, vr, rad);
    l_af_o  = ArcLen(af, vr, rad);

    l_bi_o  = ArcLen(bi, vr, rad);
    l_bf_o  = ArcLen(bf, vr, rad);

    if (fabs(l_ai_af - l_ai_o - l_af_o) < 1.0e-6 && fabs(l_bi_bf - l_bi_o - l_bf_o) < 1.0e-6) {
        xo[0] = vr[0];
        xo[1] = vr[1];
        xo[2] = vr[2];
        return true;
    }

    return false;
}
