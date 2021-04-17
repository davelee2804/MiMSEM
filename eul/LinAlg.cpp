#include <cmath>
#include <iostream>

//#include <cblas.h>

#include "LinAlg.h"

using namespace std;

// Allocate 2D array
double* Alloc2D(int ni, int nj) {
/*
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
*/
    return new double[ni*nj];
}

// Free 2D array
void Free2D(int ni, double* A) {
/*
    int ii;

    for(ii = 0; ii < ni; ii++) {
        delete[] A[ii];
    }
    delete[] A;
*/
    delete[] A;
}

// Flatten a 2D array into a 1D array
double* Flat2D(int ni, int nj, double* A) {
/*
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
*/
    int ii;
    double* Aflat = new double[ni*nj];
    for(ii = 0; ii < ni*nj; ii++) Aflat[ii] = A[ii];
    return Aflat;
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

// Multiply two matrices into a third (supplied)
void Mult_IP(int ni, int nj, int nk, double* A, double* B, double* C) {
    int ii, jj, kk;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii*nj+jj] = 0.0;
            for(kk = 0; kk < nk; kk++) {
                C[ii*nj+jj] += A[ii*nk+kk]*B[kk*nj+jj];
            }
        }
    }
/*
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, nk, 1.0, A, nk, B, nj, 0.0, C, nj);
*/
}

// Multiply diagonal by full matrix
void Mult_DF_IP(int ni, int nj, int nk, double* A, double* B, double* C) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii*nj+jj] = A[ii]*B[ii*nj+jj];
        }
    }
}

// Multiply full by diagonal matrix
void Mult_FD_IP(int ni, int nj, int nk, double* A, double* B, double* C) {
/*
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii][jj] = A[ii][jj]*B[jj];
        }
    }
*/
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii*nj+jj] = A[ii*nj+jj]*B[jj];
        }
    }
}

// Matrix transpose into a new matrix
double* Tran(int ni, int nj, double* A) {
    int ii, jj;
    double* B;

    B = new double[ni*nj];

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            B[jj*ni+ii] = A[ii*nj+jj];
        }
    }

    return B;
}

// Matrix transpose into a supplied matrix
void Tran_IP(int ni, int nj, double* A, double* B) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            B[jj*ni+ii] = A[ii*nj+jj];
        }
    }
}

// Matrix vector multiplication
void Ax_b(int ni, int nj, double* A, double* x, double* b) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        b[ii] = 0.0;
        for(jj = 0; jj < nj; jj++) {
            b[ii] += A[ii*nj+jj]*x[jj];
        }
    }
/*
    cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, nj, 1.0, A, ni, x, 1, 0.0, b, 1);
*/
}

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
}

#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}
// Matrix inverse into supplied matrix
int Inv( double* A, double* Ainv, int n ) {
/*
    int i;
    double _A[200];
    int ierr;
    int ipiv[99];
    int lwork = 200;
    double work[200];

    for(i = 0; i < n*n; i++) {
        _A[i] = A[i];
    }

    dgetrf_(&n, &n, _A, &n, ipiv, &ierr);
    dgetri_(&n, _A, &n, ipiv, work, &lwork, &ierr);

    for(i = 0; i < n*n; i++) {
        Ainv[i] = _A[i];
    }
    return ierr;
*/
    int error = 0;
    int *indxc, *indxr, *ipiv;
    int i, j, k, l, irow = 0, icol = 0, ll;
    double big, dum, pivinv, temp;

    indxc = new int[n]; indxr = new int[n]; ipiv  = new int[n];

    for( i = 0; i < n*n; i++ ) { Ainv[i] = A[i]; }
    for( j = 0; j< n; j++ ) { ipiv[j] = 0; }
    for( i = 0; i < n; i++ ) {
        big = 0.0;
        for( j = 0; j < n; j++ ) {
            if( ipiv[j] != 1 ) {
                for( k = 0; k < n; k++ ) {
                    if( ipiv[k] == 0 ) {
                        if( fabs(Ainv[j*n+k]) >= big ) {
                            big = fabs(Ainv[j*n+k]);
                            irow = j;
                            icol = k;
                        }
                    }
                    else if( ipiv[k] > 1 ) { error = 1; }
                }
            }
        }
        ++(ipiv[icol]);
        if( irow != icol ) {
            for( l = 0; l < n; l++ ) {
                //SWAP( Ainv[irow][l], Ainv[icol][l] );
                temp = Ainv[irow*n+l];
                Ainv[irow*n+l] = Ainv[icol*n+l];
                Ainv[icol*n+l] = temp;
            }
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if( fabs(Ainv[icol*n+icol]) < 1.0e-12 ) { error = 2; }
        pivinv = 1.0/Ainv[icol*n+icol];
        Ainv[icol*n+icol] = 1.0;
        for( l = 0; l < n; l++ ) { Ainv[icol*n+l] *= pivinv; }
        for( ll = 0; ll < n; ll++ ) {
            if( ll != icol ) {
                dum = Ainv[ll*n+icol];
                Ainv[ll*n+icol] = 0.0;
                for( l = 0; l < n; l++ ) { Ainv[ll*n+l] -= Ainv[icol*n+l]*dum; }
            }
        }
    }

    for( l = n-1; l >= 0; l-- ) {
        if( indxr[l] != indxc[l] ) {
            for( k = 0; k < n; k++ ) {
                //SWAP( Ainv[k][indxr[l]], Ainv[k][indxc[l]] );
                temp = Ainv[k*n+indxr[l]];
                Ainv[k*n+indxr[l]] = Ainv[k*n+indxc[l]];
                Ainv[k*n+indxc[l]] = temp;
            }
        }
    }
    delete[] indxc; delete[] indxr; delete[] ipiv;

    return error;
}
