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

// Sum of two matrix products into a new matrix (supplied)
// For application of the Piola transform for H(div) elements
void MultVec_IP(int ni, int nj, int nk, double** A1, double** B1, double** A2, double** B2, double** C) {
    int ii, jj, kk;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii][jj] = 0.0;
            for(kk = 0; kk < nk; kk++) {
                C[ii][jj] += A1[ii][kk]*B1[kk][jj] + A2[ii][kk]*B2[kk][jj];
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

#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}
// Matrix inverse into supplied matrix
int Inv( double** A, double** Ainv, int n ) {
    int error = 0;
    int *indxc, *indxr, *ipiv;
    int i, j, k, l, irow = 0, icol = 0, ll;
    double big, dum, pivinv, temp;

    indxc = new int[n]; indxr = new int[n]; ipiv  = new int[n];

    for( i = 0; i < n*n; i++ ) { Ainv[i/n][i%n] = A[i/n][i%n]; }
    for( j = 0; j< n; j++ ) { ipiv[j] = 0; }
    for( i = 0; i < n; i++ ) {
        big = 0.0;
        for( j = 0; j < n; j++ ) {
            if( ipiv[j] != 1 ) {
                for( k = 0; k < n; k++ ) {
                    if( ipiv[k] == 0 ) {
                        if( fabs(Ainv[j][k]) >= big ) {
                            big = fabs(Ainv[j][k]);
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
                temp = Ainv[irow][l];
                Ainv[irow][l] = Ainv[icol][l];
                Ainv[icol][l] = temp;
            }
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if( fabs(Ainv[icol][icol]) < 1.0e-12 ) { error = 2; }
        pivinv = 1.0/Ainv[icol][icol];
        Ainv[icol][icol] = 1.0;
        for( l = 0; l < n; l++ ) { Ainv[icol][l] *= pivinv; }
        for( ll = 0; ll < n; ll++ ) {
            if( ll != icol ) {
                dum = Ainv[ll][icol];
                Ainv[ll][icol] = 0.0;
                for( l = 0; l < n; l++ ) { Ainv[ll][l] -= Ainv[icol][l]*dum; }
            }
        }
    }

    for( l = n-1; l >= 0; l-- ) {
        if( indxr[l] != indxc[l] ) {
            for( k = 0; k < n; k++ ) {
                //SWAP( Ainv[k][indxr[l]], Ainv[k][indxc[l]] );
                temp = Ainv[k][indxr[l]];
                Ainv[k][indxr[l]] = Ainv[k][indxc[l]];
                Ainv[k][indxc[l]] = temp;
            }
        }
    }
    delete[] indxc; delete[] indxr; delete[] ipiv;

    return error;
}
