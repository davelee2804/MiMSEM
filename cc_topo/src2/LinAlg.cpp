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
