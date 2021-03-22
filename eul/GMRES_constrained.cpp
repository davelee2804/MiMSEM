#include <iostream>
#include <fstream>

#include <lapacke.h>

#include <mpi.h>
#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "GMRES_constrained.h"

#define MAX_SIZE 40

using namespace std;

double** _Alloc2D(int ni, int nj) {
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
void _Free2D(int ni, double** A) {
    int ii;

    for(ii = 0; ii < ni; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// old matrix is size [k+1,k], new matrix is size [k+2,k+1]
double** _Resize(int kk, double** H) {
    int ii, jj;
    double** Hnew;

    Hnew = _Alloc2D(kk+2,kk+1);
    for(ii = 0; ii < kk+2; ii++) {
        for(jj = 0; jj < kk+1; jj++) {
            Hnew[ii][jj] = 0.0;
        }
    }

    for(ii = 0; ii < kk+1; ii++) {
        for(jj = 0; jj < kk; jj++) {
            Hnew[ii][jj] = H[ii][jj];
        }
    }

    _Free2D(kk+1, H);

    return Hnew;
}

GMRES_constrained::GMRES_constrained(Topo* _topo, double _tol) {
    tol = _tol;
    topo = _topo;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Q = new Vec[MAX_SIZE];
    for(int ii = 0; ii < MAX_SIZE; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Q[ii]);
    }
}

GMRES_constrained::~GMRES_constrained() {
    for(int ii = 0; ii < MAX_SIZE; ii++) {
        VecDestroy(&Q[ii]);
    }
    delete[] Q;
}

// modified GMRES solve, for which the Arnoldi iteration is further constrained 
// such that each Krylov vector is orthogonal both to the previous Krylov vectors
// and an additional vector 'c'
// A: linear operator (in)
// b: rhs vector (in)
// c: constraint vector (in)
// x: solution vector including initial guess (in/out)
void GMRES_constrained::solve(Mat A, Vec b, Vec c, Vec x) {
    int kk, jj, ll, done = 0;
    double beta, norm, dot1, dot2, e_1[MAX_SIZE], res[MAX_SIZE];

    // first krylov vector
    MatMult(A, x, Q[0]);
    VecAYPX(Q[0], -1.0, b);
    VecNorm(Q[0], NORM_2, &beta);
    VecScale(Q[0], 1.0/beta);

    Hess = _Alloc2D(2, 1);

    for(kk = 1; kk < MAX_SIZE; kk++) {
       MatMult(A, Q[kk-1], Q[kk]);
       // dimension of the Hessenberg matrix is [k+1,k]
       for(jj = 0; jj < kk; jj++) {
           VecDot(Q[jj], Q[kk], &dot1); // orthogonality w.r.t. previous vector
           VecDot(Q[jj], c,     &dot2); // orthogonality w.r.t. constraint vector 
           Hess[jj][kk-1] = dot1 + dot2;
           VecAXPY(Q[kk], -1.0*Hess[jj][kk-1], Q[jj]);
       }
       VecNorm(Q[kk], NORM_2, &Hess[kk][kk-1]);

       if(kk < 2) continue;

       // solve for the coefficients
       for(kk = 0; kk < MAX_SIZE; kk++) e_1[kk] = 0.0;
       e_1[0] = beta;
       LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', kk+1, kk, 1, *Hess, kk, e_1, 1);

       // test the residual
       for(jj = 0; jj < kk+1; jj++) {
          res[jj] = 0.0;
          for(ll = 0; ll < kk; ll++) {
             res[jj] += Hess[jj][ll]*e_1[ll];
          }
       }
       res[0] -= beta;
       norm = 0.0;
       for(jj = 0; jj < kk+1; jj++) {
           norm += res[jj]*res[jj];
       }
       norm = sqrt(norm);
       if(norm < tol) done = 1;
       MPI_Bcast(&done, 1, MPI_INT, 0, MPI_COMM_WORLD);
       if(!rank) cout << kk << "\t|residual|: " << norm << endl;

       if(done) break;

       Hess = _Resize(kk, Hess);
    }

    // build the solution
    for(jj = 0; jj < kk+1; jj++) {
        VecAXPY(x, e_1[jj], Q[jj]);
    }

    _Free2D(kk+1, Hess);
}
