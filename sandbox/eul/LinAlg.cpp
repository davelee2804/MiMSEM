#include <cmath>
#include <iostream>

#include "LinAlg.h"

using namespace std;

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

// Multiply two matrices into a third (supplied)
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

// Multiply diagonal by full matrix
void Mult_DF_IP(int ni, int nj, int nk, double** A, double** B, double** C) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii][jj] = A[ii][ii]*B[ii][jj];
        }
    }
}

// Multiply full by diagonal matrix
void Mult_FD_IP(int ni, int nj, int nk, double** A, double** B, double** C) {
    int ii, jj;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii][jj] = A[ii][jj]*B[jj][jj];
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

double Determinant(double **a,int n)
{
   int i,j,j1,j2;
   double det = 0;
   double **m = 0;

   if (n < 1) { /* Error */

   } else if (n == 1) { /* Shouldn't get used */
      det = a[0][0];
   } else if (n == 2) {
      det = a[0][0] * a[1][1] - a[1][0] * a[0][1];
   } else {
      det = 0;
      for (j1=0;j1<n;j1++) {
         m = new double*[n-1];
         for (i=0;i<n-1;i++)
            m[i] = new double[n-1];
         for (i=1;i<n;i++) {
            j2 = 0;
            for (j=0;j<n;j++) {
               if (j == j1)
                  continue;
               m[i-1][j2] = a[i][j];
               j2++;
            }
         }
         det += pow(-1.0,j1+2.0) * a[0][j1] * Determinant(m,n-1);
         for (i=0;i<n-1;i++)
            delete[] m[i];
         delete[] m;
      }
   }
   return(det);
}

void CoFactor(double **a,int n,double **b)
{
   int i,j,ii,jj,i1,j1;
   double det;
   double **c;

   c = new double*[n-1];
   for (i=0;i<n-1;i++)
     c[i] = new double[n-1];

   for (j=0;j<n;j++) {
      for (i=0;i<n;i++) {

         /* Form the adjoint a_ij */
         i1 = 0;
         for (ii=0;ii<n;ii++) {
            if (ii == i)
               continue;
            j1 = 0;
            for (jj=0;jj<n;jj++) {
               if (jj == j)
                  continue;
               c[i1][j1] = a[ii][jj];
               j1++;
            }
            i1++;
         }

         /* Calculate the determinate */
         det = Determinant(c,n-1);

         /* Fill in the elements of the cofactor */
         b[i][j] = pow(-1.0,i+j+2.0) * det;
      }
   }
   for (i=0;i<n-1;i++)
      delete[] c[i];
   delete[] c;
}

void Transpose(double **a,int n)
{
   int i,j;
   double tmp;

   for (i=1;i<n;i++) {
      for (j=0;j<i;j++) {
         tmp = a[i][j];
         a[i][j] = a[j][i];
        a[j][i] = tmp;
      }
   }
}

void Inverse(double** A, double** Ainv, int n) {
   int ii, jj;
   double det = Determinant(A,n);

   CoFactor(A,n,Ainv);
   Transpose(Ainv,n);

   for(ii = 0; ii < n; ii++)
      for(jj = 0; jj < n; jj++)
         Ainv[ii][jj] /= det;
}
