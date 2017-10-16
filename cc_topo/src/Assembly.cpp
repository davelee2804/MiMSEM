#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>

#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"

void Free2D(int ni, double** A) {
    int ii;

    for(ii = 0; ii < ni; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// mass matrix for the 1 form vector (x-normal degrees of
// freedom first then y-normal degrees of freedom)
Umat::Umat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    l = _l;
    e = _e;

    assemble();
}

void Umat::assemble() {
    int ex, ey, lSize, ii, jj, kk;
    int *inds_x, *inds_y;
    double *UtQUflat, *VtQVflat;

    Wii* Q = new Wii(l->q);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double** Ut = tran(U->nDofsI, U->nDofsJ, U->A);
    double** Vt = tran(V->nDofsI, V->nDofsJ, V->A);
    double** UtQ = mult(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Q->A);
    double** VtQ = mult(V->nDofsJ, Q->nDofsJ, V->nDofsI, Vt, Q->A);
    double** UtQU = mult(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQ, U->A);
    double** VtQV = mult(V->nDofsJ, V->nDofsJ, Q->nDofsJ, VtQ, V->A);

    UtQUflat = new double[U->nDofsJ*U->nDofsJ];
    VtQVflat = new double[V->nDofsJ*V->nDofsJ];
    kk = 0;
    for(ii = 0; ii < U->nDofsJ; ii++) {
        for(jj = 0; jj < U->nDofsJ; jj++) {
            UtQUflat[kk] = UtQU[ii][jj];
            VtQVflat[kk] = VtQV[ii][jj];
            kk++;
        }
    }

    lSize = topo->n1x + topo->n1y;

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, lSize, lSize, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map1, topo->map1);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);
            // TODO: incorporate the jacobian transformation for each element
            MatSetValues(M, U->nDofsJ, inds_x, U->nDofsJ, inds_x, UtQUflat, ADD_VALUES);
            MatSetValues(M, V->nDofsJ, inds_y, V->nDofsJ, inds_y, VtQVflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    Free2D(U->nDofsJ, UtQ);
    Free2D(V->nDofsJ, VtQ);
    Free2D(U->nDofsJ, UtQU);
    Free2D(V->nDofsJ, VtQV);
    delete Q;
	delete U;
    delete V;
    delete[] UtQUflat;
    delete[] VtQVflat;
}

Umat::~Umat() {
    MatDestroy(&M);
}

// 2 form mass matrix
Wmat::Wmat(Topo* _topo, LagrangeEdge* _e) {
    topo = _topo;
    e = _e;

    assemble();
}

void Wmat::assemble() {
    int ii, jj, kk, ex, ey;
    int* inds;
    double* WtQWflat;

    Wii* Q = new Wii(e->l->q);
    M2_j_xy_i* W = new M2_j_xy_i(e);
    double** Wt = tran(W->nDofsI, W->nDofsJ, W->A);
    double** WtQ = mult(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q->A);
    double** WtQW = mult(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A);

    WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    kk = 0;
    for(ii = 0; ii < W->nDofsJ; ii++) {
        for(jj = 0; jj < W->nDofsJ; jj++) {
            WtQWflat[kk] = WtQW[ii][jj];
            kk++;
        }
    }

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2, topo->n2, topo->nDofs2G, topo->nDofs2G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map2, topo->map2);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds2_g(ex, ey);
            // TODO: incorporate the jacobian transformation for each element
            MatSetValues(M, W->nDofsJ, inds, W->nDofsJ, inds, WtQWflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete W;
    delete[] WtQWflat;
}

Wmat::~Wmat() {
    MatDestroy(&M);
}

// 0 form mass matrix
Pmat::Pmat(Topo* _topo, LagrangeNode* _l) {
    topo = _topo;
    l = _l;

    assemble();
}

void Pmat::assemble() {
    int ii, jj, kk, ex, ey;
    int* inds;
    double* PtQPflat;

    Wii* Q = new Wii(l->q);
    M0_j_xy_i* P = new M0_j_xy_i(l);
    double** Pt = tran(P->nDofsI, P->nDofsJ, P->A);
    double** PtQ = mult(P->nDofsJ, Q->nDofsJ, P->nDofsI, Pt, Q->A);
    double** PtQP = mult(P->nDofsJ, P->nDofsJ, Q->nDofsJ, PtQ, P->A);

    PtQPflat = new double[P->nDofsJ*P->nDofsJ];
    kk = 0;
    for(ii = 0; ii < P->nDofsJ; ii++) {
        for(jj = 0; jj < P->nDofsJ; jj++) {
            PtQPflat[kk] = PtQP[ii][jj];
            kk++;
        }
    }

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0, topo->n0, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 2*P->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map0, topo->map0);
    MatZeroEntries(M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds = topo->elInds0_g(ex, ey);
            // TODO: incorporate the jacobian transformation for each element
            MatSetValues(M, P->nDofsJ, inds, P->nDofsJ, inds, PtQPflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    Free2D(P->nDofsJ, PtQP);
    delete P;
    delete[] PtQPflat;
}

Pmat::~Pmat() {
    MatDestroy(&M);
}

// 1 form mass matrix with 2 forms interpolated to quadrature points
Uhmat::Uhmat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    int lSize;

    topo = _topo;
    l = _l;
    e = _e;

    Wii* Q = new Wii(l->q);
    M1x_j_xy_i* U = new M1x_j_xy_i(l, e);
    M1y_j_xy_i* V = new M1y_j_xy_i(l, e);
    double** Ut = tran(U->nDofsI, U->nDofsJ, U->A);
    double** Vt = tran(V->nDofsI, V->nDofsJ, V->A);
    UtQ = mult(U->nDofsJ, Q->nDofsJ, U->nDofsI, Ut, Q->A);
    VtQ = mult(V->nDofsJ, Q->nDofsJ, V->nDofsI, Vt, Q->A);
    UtQUflat = new double[U->nDofsJ*U->nDofsJ];
    VtQVflat = new double[V->nDofsJ*V->nDofsJ];
    ck = new double[l->n*l->n];

    Uh = new M1x_j_Fxy_i(l, e);
    Vh = new M1y_j_Fxy_i(l, e);

    lSize = topo->n1x + topo->n1y;

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, lSize, lSize, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*U->nDofsJ, PETSC_NULL, 2*U->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map1, topo->map1);

    Free2D(U->nDofsJ, Ut);
    Free2D(V->nDofsJ, Vt);
    delete Q;
    delete U;
    delete V;
}

void Uhmat::assemble(Vec h2) {
    int ex, ey, ii, jj, kk, n2;
    int *inds_x, *inds_y, *inds2;
    double **UtQU, **VtQV;
    PetscScalar* h2Array;

    n2 = topo->elOrd*topo->elOrd;

    MatZeroEntries(M);

    VecGetArray(h2, &h2Array);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // TODO: incorporate the jacobian transformation for each element
            inds2 = topo->elInds2_l(ex, ey);
            for(kk = 0; kk < n2; kk++) {
                ck[kk] = h2Array[inds2[kk]];
            }
            Uh->assemble(ck);
            Vh->assemble(ck);
            UtQU = mult(Uh->nDofsJ, Uh->nDofsJ, Uh->nDofsI, UtQ, Uh->A);
            VtQV = mult(Vh->nDofsJ, Vh->nDofsJ, Vh->nDofsI, VtQ, Vh->A);

            kk = 0;
            for(ii = 0; ii < Uh->nDofsJ; ii++) {
                for(jj = 0; jj < Uh->nDofsJ; jj++) {
                    UtQUflat[kk] = UtQU[ii][jj];
                    VtQVflat[kk] = VtQV[ii][jj];
                    kk++;
                }
            }

            inds_x = topo->elInds1x_g(ex, ey);
            inds_y = topo->elInds1y_g(ex, ey);

            MatSetValues(M, Uh->nDofsJ, inds_x, Uh->nDofsJ, inds_x, UtQUflat, ADD_VALUES);
            MatSetValues(M, Vh->nDofsJ, inds_y, Vh->nDofsJ, inds_y, VtQVflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(Uh->nDofsJ, UtQU);
    Free2D(Vh->nDofsJ, VtQV);
}

Uhmat::~Uhmat() {
    delete[] UtQUflat;
    delete[] VtQVflat;
    delete[] ck;

    Free2D(Uh->nDofsJ, UtQ);
    Free2D(Vh->nDofsJ, VtQ);

	delete Uh;
    delete Vh;

    MatDestroy(&M);
}

// Assembly of the diagonal 0 form mass matrix as a vector.
// Assumes inexact integration and a diagonal mass matrix for the 
// 0 form function space (ie: quadrature and basis functions are 
// the same order)
Pvec::Pvec(Topo* _topo, LagrangeNode* _l) {
    topo = _topo;
    l = _l;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, topo->nDofs0G, &v);
    VecSetLocalToGlobalMapping(v, topo->map0);

    entries = new PetscScalar[(l->n+1)*(l->n+1)];

    assemble();
}

void Pvec::assemble() {
    int ii, ex, ey, np1, np12;
    int *inds_x;
    double* weights = l->q->w;

    VecZeroEntries(v);

    np1 = l->n + 1;
    np12 = np1*np1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // TODO: incorporate the jacobian transformation for each element
            for(ii = 0; ii < np12; ii++) {
                // weight at quadrature point
                entries[ii] = weights[ii%np1]*weights[ii/np1];
            }
            inds_x = topo->elInds0_g(ex, ey);
            VecSetValues(v, np12, inds_x, entries, ADD_VALUES);
        }
    }
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}

Pvec::~Pvec() {
    delete[] entries;
    VecDestroy(&v);
}

// Assembly of the diagonal 0 form mass matrix as a vector 
// with 2 form vector interpolated onto quadrature points.
// Assumes inexact integration and a diagonal mass matrix for the 
// 0 form function space (ie: quadrature and basis functions are 
// the same order)
Phvec::Phvec(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e) {
    topo = _topo;
    l = _l;
    e = _e;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, topo->nDofs0G, &v);
    VecSetLocalToGlobalMapping(v, topo->map0);

    ck = new double[(l->n)*(l->n)];
    entries = new PetscScalar[(l->n+1)*(l->n+1)];
}

void Phvec::assemble(Vec h2) {
    int ii, kk, ex, ey, np1, np12, n2;
    int *inds2, *inds_x;
    double wt, hq;
    PetscScalar* h2Array;
    double* weights = l->q->w;
    double** ejxi = e->ejxi;

    VecGetArray(h2, &h2Array);
    VecZeroEntries(v);

    n2 = (l->n)*(l->n);
    np1 = l->n + 1;
    np12 = np1*np1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // TODO: incorporate the jacobian transformation for each element
            inds2 = topo->elInds2_l(ex, ey);
            for(kk = 0; kk < n2; kk++) {
                ck[kk] = h2Array[inds2[kk]];
            }

            for(ii = 0; ii < np12; ii++) {
                // weight at quaadrature point
                wt = weights[ii%np1]*weights[ii/np1];

                // interpolate 2 form at quadrature point
                hq = 0.0;
                for(kk = 0; kk < n2; kk++) {
                    hq += ck[kk]*ejxi[ii][kk];
                }

                entries[ii] = wt*hq;
            }

            inds_x = topo->elInds0_g(ex, ey);
            VecSetValues(v, np12, inds_x, entries, ADD_VALUES);
        }
    }
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}

Phvec::~Phvec() {
    delete[] ck;
    delete[] entries;
    VecDestroy(&v);
}

// Assumes quadrature points and 0 forms are the same (for now)
WtQmat::WtQmat(Topo* _topo, LagrangeEdge* _e) {
    topo = _topo;
    e = _e;

    assemble();
}

void WtQmat::assemble() {
    int ex, ey, ii, jj, kk;
    int *inds_2, *inds_0;
    double* WtQflat;

    M2_j_xy_i* W = new M2_j_xy_i(e);
    Wii* Q = new Wii(e->l->q);
    double** Wt = tran(W->nDofsI, W->nDofsJ, W->A);
    double** WtQ = mult(W->nDofsJ, Q->nDofsJ, Q->nDofsI, Wt, Q->A);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n2, topo->n0, topo->nDofs2G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*W->nDofsJ, PETSC_NULL, 2*W->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map2, topo->map0);
    MatZeroEntries(M);

    WtQflat = new double[W->nDofsJ*Q->nDofsJ];
    kk = 0;
    for(ii = 0; ii < W->nDofsJ; ii++) {
        for(jj = 0; jj < Q->nDofsJ; jj++) {
            WtQflat[kk] = WtQ[ii][jj];
            kk++;
        }
    }

    // TODO: incorportate jacobian tranformation for each element
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_2 = topo->elInds2_g(ex, ey);
            inds_0 = topo->elInds0_g(ex, ey);

            MatSetValues(M, W->nDofsJ, inds_2, Q->nDofsJ, inds_0, WtQflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    delete[] WtQflat;
    delete W;
    delete Q;
}

WtQmat::~WtQmat() {
    MatDestroy(&M);
}

// Assumes quadrature points and 0 forms are the same (for now)
PtQmat::PtQmat(Topo* _topo, LagrangeNode* _l) {
    topo = _topo;
    l = _l;

    assemble();
}

void PtQmat::assemble() {
    int ex, ey, ii, jj, kk;
    int *inds_0;
    double* PtQflat;

    M0_j_xy_i* P = new M0_j_xy_i(l);
    Wii* Q = new Wii(l->q);
    double** Pt = tran(P->nDofsI, P->nDofsJ, P->A);
    double** PtQ = mult(P->nDofsJ, Q->nDofsJ, Q->nDofsI, Pt, Q->A);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, topo->n0, topo->n0, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 4*P->nDofsJ, PETSC_NULL, 2*P->nDofsJ, PETSC_NULL);
    MatSetLocalToGlobalMapping(M, topo->map0, topo->map0);
    MatZeroEntries(M);

    PtQflat = new double[P->nDofsJ*Q->nDofsJ];
    kk = 0;
    for(ii = 0; ii < P->nDofsJ; ii++) {
        for(jj = 0; jj < Q->nDofsJ; jj++) {
            PtQflat[kk] = PtQ[ii][jj];
            kk++;
        }
    }

    // TODO: incorportate jacobian tranformation for each element
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds_0 = topo->elInds0_g(ex, ey);

            MatSetValues(M, P->nDofsJ, inds_0, Q->nDofsJ, inds_0, PtQflat, ADD_VALUES);
        }
    }

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    Free2D(P->nDofsJ, Pt);
    Free2D(P->nDofsJ, PtQ);
    delete[] PtQflat;
    delete P;
    delete Q;
}

PtQmat::~PtQmat() {
    MatDestroy(&M);
}

////s UtQmat:
////def __init__(self,topo,quad):
////	topo_q = Topo(topo.nx,topo.ny,quad.n)
////	maps,nnz = self.genMap(topo,topo_q)
////	self.assemble(topo,topo_q,maps,nnz)

////def assemble(self,topo,topo_q,maps,nnz):
////	Q = Wii(topo_q.n).A
////	U = M1x_j_xy_i(topo.n,topo_q.n).A
////	V = M1y_j_xy_i(topo.n,topo_q.n).A
////	Ut = U.transpose()
////	Vt = V.transpose()
////	UtQ = mult(Ut,Q)
////	VtQ = mult(Vt,Q)

////	np1 = topo.n+1
////	mp1 = topo_q.n+1
////	nrl = topo.n*np1  # number of rows in local matrix, (u or v)
////	ncl = mp1*mp1     # number of columns in local matrix (quad pts)
////	shift1Forms = (topo.n*topo.nx)*(topo.n*topo.ny)
////	shift0Forms = (topo_q.n*topo_q.nx)*(topo_q.n*topo_q.ny)
////	rows = np.zeros(nnz,dtype=np.int32)
////	cols = np.zeros(nnz,dtype=np.int32)
////	vals = np.zeros(nnz,dtype=np.float64)

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds0 = topo_q.localToGlobal0(ex,ey)
////			inds1 = topo.localToGlobal1x(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds1[jj/ncl]
////				col = inds0[jj%ncl]
////				ii = maps[row][col]
////				if ii == -1:
////					print 'ERROR! assembly'
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + UtQ[jj/ncl][jj%ncl]

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds0 = topo_q.localToGlobal0(ex,ey) + shift0Forms
////			inds1 = topo.localToGlobal1y(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds1[jj/ncl]
////				col = inds0[jj%ncl]
////				ii = maps[row][col]
////				if ii == -1:
////					print 'ERROR! assembly'
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + VtQ[jj/ncl][jj%ncl]

////	nr = 2*topo.nx*topo.ny*topo.n*topo.n
////	nc = 2*topo.nx*topo.ny*topo_q.n*topo_q.n
////	self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

////def genMap(self,topo,topo_q):
////	np1 = topo.n+1
////	mp1 = topo_q.n+1
////	nrl = np1*topo.n
////	ncl = mp1*mp1
////	nr = 2*topo.nx*topo.ny*topo.n*topo.n
////	nc = 2*topo.nx*topo.ny*topo_q.n*topo_q.n
////	maps = -1*np.ones((nr,nc),dtype=np.int32)
////	shift1Forms = (topo.n*topo.nx)*(topo.n*topo.ny)
////	shift0Forms = (topo_q.n*topo_q.nx)*(topo_q.n*topo_q.ny)
////	ii = 0
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds0 = topo_q.localToGlobal0(ex,ey)
////			inds1 = topo.localToGlobal1x(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds1[jj/ncl]
////				col = inds0[jj%ncl]
////				if maps[row][col] == -1:
////					maps[row][col] = ii
////					ii = ii + 1

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds0 = topo_q.localToGlobal0(ex,ey) + shift0Forms
////			inds1 = topo.localToGlobal1y(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds1[jj/ncl]
////				col = inds0[jj%ncl]
////				if maps[row][col] == -1:
////					maps[row][col] = ii
////					ii = ii + 1

////	return maps, ii

////s UtQPmat:
////def __init__(self,topo,quad,u):
////	M1x = M1x_j_Exy_i(topo.n,quad.n)
////	M1y = M1y_j_Exy_i(topo.n,quad.n)
////	Q = Wii(quad.n).A
////	W = M0_j_xy_i(topo.n,quad.n).A
////	QW = mult(Q,W)
////	
////	maps,nnz = self.genMap(topo)
////	shift = (topo.n*topo.nx)*(topo.n*topo.ny)
////	rows = np.zeros(nnz,dtype=np.int32)
////	cols = np.zeros(nnz,dtype=np.int32)
////	vals = np.zeros(nnz,dtype=np.float64)

////	np1 = topo.n+1
////	nrl = topo.n*np1
////	#ncl = topo.n*topo.n
////	ncl = np1*np1
////	c = np.zeros((nrl),dtype=np.float64)

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1x(ex,ey)
////			inds0 = topo.localToGlobal0(ex,ey)
////			inds1y = topo.localToGlobal1y(ex,ey)

////			for kk in np.arange(nrl):
////				c[kk] = +1.0*u[inds1y[kk]]

////			U = M1x.assemble(c)
////			Ut = U.transpose()
////			UtQW = mult(Ut,QW)

////			for jj in np.arange(nrl*ncl):
////				row = inds1[jj/ncl]
////				col = inds0[jj%ncl]
////				ii = maps[row][col]
////				if ii == -1:
////					print 'ERROR! assembly'
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + UtQW[jj/ncl][jj%ncl]
////	
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1y(ex,ey)
////			inds0 = topo.localToGlobal0(ex,ey)
////			inds1x = topo.localToGlobal1x(ex,ey)

////			for kk in np.arange(nrl):
////				c[kk] = -1.0*u[inds1x[kk]]

////			V = M1y.assemble(c)
////			Vt = V.transpose()
////			VtQW = mult(Vt,QW)

////			for jj in np.arange(nrl*ncl):
////				row = inds1[jj/ncl]
////				col = inds0[jj%ncl]
////				ii = maps[row][col]
////				if ii == -1:
////					print 'ERROR! assembly'
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + VtQW[jj/ncl][jj%ncl]

////	nr = 2*topo.nx*topo.ny*topo.n*topo.n
////	nc = topo.nx*topo.ny*topo.n*topo.n
////	self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

////def genMap(self,topo):
////	np1 = topo.n+1
////	nrl = topo.n*np1
////	#ncl = topo.n*topo.n
////	ncl = np1*np1
////	nr = 2*topo.nx*topo.ny*topo.n*topo.n
////	nc = topo.nx*topo.ny*topo.n*topo.n
////	maps = -1*np.ones((nr,nc),dtype=np.int32)
////	shift = (topo.n*topo.nx)*(topo.n*topo.ny)
////	ii = 0
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1x(ex,ey)
////			inds0 = topo.localToGlobal0(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds1[jj/ncl]
////				col = inds0[jj%ncl]
////				if maps[row][col] == -1:
////					maps[row][col] = ii;
////					ii = ii + 1

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1y(ex,ey)
////			inds0 = topo.localToGlobal0(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds1[jj/ncl]
////				col = inds0[jj%ncl]
////				if maps[row][col] == -1:
////					maps[row][col] = ii;
////					ii = ii + 1

////	return maps, ii

////s WtQUmat:
////def __init__(self,topo,quad):
////	self.topo = topo
////	self.quad = quad
////	Q = Wii(quad.n).A
////	self.M1x = M1x_j_Cxy_i(topo.n,quad.n)
////	self.M1y = M1y_j_Cxy_i(topo.n,quad.n)
////	W = M2_j_xy_i(topo.n,quad.n).A
////	Wt = W.transpose()
////	self.WtQ = mult(Wt,Q)
////	
////	self.maps,self.nnz = self.genMap(topo)

////def assemble(self,u):
////	topo = self.topo
////	maps = self.maps
////	nnz = self.nnz
////	rows = np.zeros(nnz,dtype=np.int32)
////	cols = np.zeros(nnz,dtype=np.int32)
////	vals = np.zeros(nnz,dtype=np.float64)

////	np1 = topo.n+1
////	nrl = topo.n*topo.n
////	ncl = topo.n*np1
////	shift = topo.nx*topo.ny*topo.n*topo.n

////	cj = np.zeros((ncl),dtype=np.float64)

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1x(ex,ey)
////			inds2 = topo.localToGlobal2(ex,ey)

////			for kk in np.arange(ncl):
////				cj[kk] = u[inds1[kk]]

////			U = self.M1x.assemble(cj)
////			WtQU = mult(self.WtQ,U)

////			for jj in np.arange(nrl*ncl):
////				row = inds2[jj/ncl]
////				col = inds1[jj%ncl]
////				ii = maps[row][col]
////				if ii == -1:
////					print 'ERROR! assembly'
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + WtQU[jj/ncl][jj%ncl]
////	
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1y(ex,ey)
////			inds2 = topo.localToGlobal2(ex,ey)

////			for kk in np.arange(ncl):
////				cj[kk] = u[inds1[kk]]

////			V = self.M1y.assemble(cj)
////			WtQV = mult(self.WtQ,V)

////			for jj in np.arange(nrl*ncl):
////				row = inds2[jj/ncl]
////				col = inds1[jj%ncl]
////				ii = maps[row][col]
////				if ii == -1:
////					print 'ERROR! assembly'
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + WtQV[jj/ncl][jj%ncl]

////	nr = topo.nx*topo.ny*topo.n*topo.n
////	nc = 2*topo.nx*topo.ny*topo.n*topo.n
////	self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

////	return self.M

////def genMap(self,topo):
////	np1 = topo.n+1
////	nrl = topo.n*topo.n
////	ncl = topo.n*np1
////	nr = topo.nx*topo.ny*topo.n*topo.n
////	nc = 2*topo.nx*topo.ny*topo.n*topo.n
////	maps = -1*np.ones((nr,nc),dtype=np.int32)
////	ii = 0
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1x(ex,ey)
////			inds2 = topo.localToGlobal2(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds2[jj/ncl]
////				col = inds1[jj%ncl]
////				if maps[row][col] == -1:
////					maps[row][col] = ii;
////					ii = ii + 1

////	shift = topo.nx*topo.ny*topo.n*topo.n
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1y(ex,ey)
////			inds2 = topo.localToGlobal2(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds2[jj/ncl]
////				col = inds1[jj%ncl]
////				if maps[row][col] == -1:
////					maps[row][col] = ii;
////					ii = ii + 1

////	return maps, ii

////oject the potential vorticity gradient velocity product onto the 0 forms
////s PtQUmat:
////def __init__(self,topo,quad):
////	self.topo = topo
////	Q = Wii(quad.n).A
////	self.M1x = M1x_j_Exy_i(topo.n,quad.n)
////	self.M1y = M1y_j_Exy_i(topo.n,quad.n)
////	P = M0_j_xy_i(topo.n,quad.n).A
////	Pt = P.transpose()
////	self.PtQ = mult(Pt,Q)
////	
////	self.maps,self.nnz = self.genMap(topo)

////def assemble(self,dq):
////	topo = self.topo
////	maps = self.maps
////	nnz = self.nnz

////	rows = np.zeros(nnz,dtype=np.int32)
////	cols = np.zeros(nnz,dtype=np.int32)
////	vals = np.zeros(nnz,dtype=np.float64)

////	np1 = topo.n+1
////	nrl = np1*np1
////	ncl = topo.n*np1
////	shift = topo.nx*topo.ny*topo.n*topo.n

////	cj = np.zeros((ncl),dtype=np.float64)

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1x = topo.localToGlobal1x(ex,ey)
////			inds1y = topo.localToGlobal1y(ex,ey)
////			inds0 = topo.localToGlobal0(ex,ey)

////			for kk in np.arange(ncl):
////				cj[kk] = dq[inds1y[kk]]

////			U = self.M1x.assemble(-1.0*cj)
////			PtQU = mult(self.PtQ,U)

////			for jj in np.arange(nrl*ncl):
////				row = inds0[jj/ncl]
////				col = inds1x[jj%ncl]
////				ii = maps[row][col]
////				if ii == -1:
////					print 'ERROR! assembly'
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + PtQU[jj/ncl][jj%ncl]
////	
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1x = topo.localToGlobal1x(ex,ey)
////			inds1y = topo.localToGlobal1y(ex,ey)
////			inds0 = topo.localToGlobal0(ex,ey)

////			for kk in np.arange(ncl):
////				cj[kk] = dq[inds1x[kk]]

////			V = self.M1y.assemble(cj)
////			PtQV = mult(self.PtQ,V)

////			for jj in np.arange(nrl*ncl):
////				row = inds0[jj/ncl]
////				col = inds1y[jj%ncl]
////				ii = maps[row][col]
////				if ii == -1:
////					print 'ERROR! assembly'
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + PtQV[jj/ncl][jj%ncl]

////	nr = topo.nx*topo.ny*topo.n*topo.n
////	nc = 2*topo.nx*topo.ny*topo.n*topo.n
////	self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

////	return self.M

////def genMap(self,topo):
////	np1 = topo.n+1
////	nrl = np1*np1
////	ncl = topo.n*np1
////	nr = topo.nx*topo.ny*topo.n*topo.n
////	nc = 2*topo.nx*topo.ny*topo.n*topo.n
////	maps = -1*np.ones((nr,nc),dtype=np.int32)
////	ii = 0
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1x(ex,ey)
////			inds0 = topo.localToGlobal0(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds0[jj/ncl]
////				col = inds1[jj%ncl]
////				if maps[row][col] == -1:
////					maps[row][col] = ii;
////					ii = ii + 1

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1 = topo.localToGlobal1y(ex,ey)
////			inds0 = topo.localToGlobal0(ex,ey)
////			for jj in np.arange(nrl*ncl):
////				row = inds0[jj/ncl]
////				col = inds1[jj%ncl]
////				if maps[row][col] == -1:
////					maps[row][col] = ii;
////					ii = ii + 1

////	return maps, ii

////form mass matrix with 0 form interpolated to quadrature points
////or rotational term in the momentum equation)
////s RotationalMat:
////def __init__(self,topo,quad):
////	self.topo = topo
////	self.quad = quad

////	Q = Wii(quad.n).A
////	U = M1x_j_xy_i(topo.n,quad.n).A
////	V = M1y_j_xy_i(topo.n,quad.n).A
////	Ut = U.transpose()
////	Vt = V.transpose()
////	self.UtQ = mult(Ut,Q)
////	self.VtQ = mult(Vt,Q)

////	self.M1x = M1x_j_Dxy_i(topo.n,quad.n)
////	self.M1y = M1y_j_Dxy_i(topo.n,quad.n)

////	self.maps,self.nnz = self.genMap(topo)
////
////def assemble(self,w):
////	topo = self.topo
////	maps = self.maps
////	nnz = self.nnz

////	shift = (topo.n*topo.nx)*(topo.n*topo.ny)
////	rows = np.zeros(nnz,dtype=np.int32)
////	cols = np.zeros(nnz,dtype=np.int32)
////	vals = np.zeros(nnz,dtype=np.float64)

////	n = topo.n
////	np1 = topo.n+1
////	nrl = topo.n*np1
////	ncl = topo.n*np1
////	n2 = np1*np1

////	cj = np.zeros((n2),dtype=np.float64)

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds0 = topo.localToGlobal0(ex,ey)
////			inds1x = topo.localToGlobal1x(ex,ey)
////			inds1y = topo.localToGlobal1y(ex,ey)

////			for j in np.arange(n2):
////				cj[j] = w[inds0[j]]

////			# TODO: should u be interpolated onto the quadrature
////			# points from the U^T matrix or the W matrix??
////			V = self.M1y.assemble(-1.0*cj)
////			UtQV = mult(self.UtQ,V)

////			for jj in np.arange(nrl*ncl):
////				row = inds1x[jj/ncl]
////				col = inds1y[jj%ncl]
////				ii = maps[row][col]
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + UtQV[jj/ncl][jj%ncl]
////	
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds0 = topo.localToGlobal0(ex,ey)
////			inds1y = topo.localToGlobal1y(ex,ey)
////			inds1x = topo.localToGlobal1x(ex,ey)
////			inds2 = topo.localToGlobal2(ex,ey)

////			for j in np.arange(n2):
////				cj[j] = w[inds0[j]]

////			# TODO: should u be interpolated onto the quadrature
////			# points from the U^T matrix or the W matrix??
////			U = self.M1x.assemble(cj)
////			VtQU = mult(self.VtQ,U)

////			for jj in np.arange(nrl*ncl):
////				row = inds1y[jj/ncl]
////				col = inds1x[jj%ncl]
////				ii = maps[row][col]
////				rows[ii] = row
////				cols[ii] = col
////				vals[ii] = vals[ii] + VtQU[jj/ncl][jj%ncl]

////	nr = 2*topo.nx*topo.ny*topo.n*topo.n
////	nc = 2*topo.nx*topo.ny*topo.n*topo.n
////	self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

////	return self.M

////def genMap(self,topo):
////	np1 = topo.n+1
////	ne = np1*topo.n
////	nr = topo.nx*topo.ny*2*topo.n*topo.n
////	nc = topo.nx*topo.ny*2*topo.n*topo.n
////	maps = -1*np.ones((nr,nc),dtype=np.int32)
////	shift = (topo.n*topo.nx)*(topo.n*topo.ny)
////	ii = 0
////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1x = topo.localToGlobal1x(ex,ey)
////			inds1y = topo.localToGlobal1y(ex,ey)
////			for jj in np.arange(ne*ne):
////				row = inds1x[jj/ne]
////				col = inds1y[jj%ne]
////				if maps[row][col] == -1:
////					maps[row][col] = ii
////					ii = ii + 1

////	for ey in np.arange(topo.ny):
////		for ex in np.arange(topo.nx):
////			inds1y = topo.localToGlobal1y(ex,ey)
////			inds1x = topo.localToGlobal1x(ex,ey)
////			for jj in np.arange(ne*ne):
////				row = inds1y[jj/ne]
////				col = inds1x[jj%ne]
////				if maps[row][col] == -1:
////					maps[row][col] = ii
////					ii = ii + 1

////	return maps, ii
