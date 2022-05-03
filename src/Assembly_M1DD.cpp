#include <iostream>
#include <fstream>

#include <mpi.h>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscksp.h>
#include <petscsnes.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "Assembly_M1DD.h"

using namespace std;

Assembly_M1DD::Assembly_M1DD(Topo* _topo, Geom* _geom) {
    topo = _topo;
    geom = _geom;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    quad = new GaussLobatto(geom->quad->n);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    Q = new Wii(node->q, geom);
    U = new M1x_j_xy_i(node, edge);
    V = new M1y_j_xy_i(node, edge);
    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    // allocate the matrices
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_intl_locl, topo->dd_n_intl_locl, 4*U->nDofsJ, PETSC_NULL, &Mii);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_intl_locl, topo->dd_n_skel_locl, 4*U->nDofsJ, PETSC_NULL, &Mis);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_skel_locl, topo->dd_n_intl_locl, 4*U->nDofsJ, PETSC_NULL, &Msi);
    MatCreate(MPI_COMM_WORLD, &Mss);
    MatSetSizes(Mss, topo->dd_n_skel_locl, topo->dd_n_skel_locl, topo->dd_n_skel_glob, topo->dd_n_skel_glob);
    MatSetType(Mss, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Mss, 4*U->nDofsJ, PETSC_NULL, 4*U->nDofsJ, PETSC_NULL);
    MatCreate(MPI_COMM_WORLD, &Ss);
    MatSetSizes(Ss, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(Ss, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Ss, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);

    MatCreate(MPI_COMM_WORLD, &Sg);
    MatSetSizes(Sg, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(Sg, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Sg, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);

    // and the vectors
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_intl_locl, &b_intl);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_intl_locl, &x_intl);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_intl_locl, &t_intl);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_skel_locl, &b_skel);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_skel_locl, &x_skel);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_skel_locl, &t_skel);
    VecCreateMPI(MPI_COMM_WORLD, topo->dd_n_skel_locl, topo->dd_n_skel_glob, &b_skel_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->dd_n_skel_locl, topo->dd_n_skel_glob, &x_skel_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->dd_n_skel_locl, topo->dd_n_skel_glob, &t_skel_g);
}

Assembly_M1DD::~Assembly_M1DD() {
    MatDestroy(&Mii);
    MatDestroy(&Mis);
    MatDestroy(&Msi);
    MatDestroy(&Mss);
    if(Ss_l)                    MatDestroy(&Ss_l);
    MatDestroy(&Ss);
    MatDestroy(&Sg);
    VecDestroy(&b_intl);
    VecDestroy(&x_intl);
    VecDestroy(&t_intl);
    VecDestroy(&b_skel);
    VecDestroy(&x_skel);
    VecDestroy(&t_skel);
    VecDestroy(&b_skel_g);
    VecDestroy(&x_skel_g);
    VecDestroy(&t_skel_g);
    delete U;
    delete V;
    delete Q;
    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    delete edge;
    delete node;
    delete quad;
}

void Assembly_M1DD::assemble_mat() {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_intl_x, *inds_intl_y, *inds_skel_x, *inds_skel_y, *inds_skel_x_g, *inds_skel_y_g;
    double det, **J;
    double** UtQaa = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** UtQab = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** VtQba = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** VtQbb = Alloc2D(U->nDofsJ, Q->nDofsJ);
    double** UtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** UtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** VtQU = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** VtQV = Alloc2D(U->nDofsJ, U->nDofsJ);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Qab = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Qbb = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double* UtQUflat = new double[U->nDofsJ*U->nDofsJ];

    MatZeroEntries(Mii);
    MatZeroEntries(Mis);
    MatZeroEntries(Msi);
    MatZeroEntries(Mss);

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii][ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]/det;
                Qab[ii][ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]/det;
                Qbb[ii][ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]/det;
            }

            inds_intl_x = topo->elInds_intl_x_l(ex, ey);
            inds_intl_y = topo->elInds_intl_y_l(ex, ey);
            inds_skel_x = topo->elInds_skel_x_l(ex, ey);
            inds_skel_y = topo->elInds_skel_y_l(ex, ey);
            inds_skel_x_g = topo->elInds_skel_x_g(ex, ey);
            inds_skel_y_g = topo->elInds_skel_y_g(ex, ey);


            Mult_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qaa, UtQaa);
            Mult_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Ut, Qab, UtQab);
            Mult_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qab, VtQba);
            Mult_IP(U->nDofsJ, Q->nDofsI, Q->nDofsJ, Vt, Qbb, VtQbb);

            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQaa, U->A, UtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, UtQab, V->A, UtQV);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQba, U->A, VtQU);
            Mult_IP(U->nDofsJ, U->nDofsJ, Q->nDofsJ, VtQbb, V->A, VtQV);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQU, UtQUflat);
            MatSetValues(Mii, U->nDofsJ, inds_intl_x, U->nDofsJ, inds_intl_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mis, U->nDofsJ, inds_intl_x, U->nDofsJ, inds_skel_x, UtQUflat, ADD_VALUES);
            MatSetValues(Msi, U->nDofsJ, inds_skel_x, U->nDofsJ, inds_intl_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mss, U->nDofsJ, inds_skel_x_g, U->nDofsJ, inds_skel_x_g, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(Mii, U->nDofsJ, inds_intl_x, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mis, U->nDofsJ, inds_intl_x, U->nDofsJ, inds_skel_y, UtQUflat, ADD_VALUES);
            MatSetValues(Msi, U->nDofsJ, inds_skel_x, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mss, U->nDofsJ, inds_skel_x_g, U->nDofsJ, inds_skel_y_g, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(Mii, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_intl_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mis, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_skel_x, UtQUflat, ADD_VALUES);
            MatSetValues(Msi, U->nDofsJ, inds_skel_y, U->nDofsJ, inds_intl_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mss, U->nDofsJ, inds_skel_y_g, U->nDofsJ, inds_skel_x_g, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(Mii, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mis, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_skel_y, UtQUflat, ADD_VALUES);
            MatSetValues(Msi, U->nDofsJ, inds_skel_y, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mss, U->nDofsJ, inds_skel_y_g, U->nDofsJ, inds_skel_y_g, UtQUflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(Mii, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Mis, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Msi, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Mss, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mii, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mis, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Msi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mss, MAT_FINAL_ASSEMBLY);

    Free2D(U->nDofsJ, UtQaa);
    Free2D(U->nDofsJ, UtQab);
    Free2D(U->nDofsJ, VtQba);
    Free2D(U->nDofsJ, VtQbb);
    Free2D(U->nDofsJ, UtQU);
    Free2D(U->nDofsJ, UtQV);
    Free2D(U->nDofsJ, VtQU);
    Free2D(U->nDofsJ, VtQV);
    Free2D(Q->nDofsI, Qaa);
    Free2D(Q->nDofsI, Qab);
    Free2D(Q->nDofsI, Qbb);
    delete[] UtQUflat;
}

void Assembly_M1DD::assemble_rhs_hu(Vec rho, Vec vel) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_intl_x, *inds_intl_y;
    int *inds_skel_x, *inds_skel_y;
    double det, **J;
    PetscScalar *intlArray, *skelArray, *velArray, *rhoArray;
    double _u[2], _r, Qaa[99], Qab[99], Qba[99], Qbb[99], rhs[99];

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    VecZeroEntries(b_intl);
    VecZeroEntries(b_skel);
    VecZeroEntries(b_skel_g);

    VecGetArray(b_intl, &intlArray);
    VecGetArray(b_skel, &skelArray);
    VecGetArray(vel, &velArray);
    VecGetArray(rho, &rhoArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                J = geom->J[ei][ii];

                Qaa[ii] = (J[0][0]*J[0][0] + J[1][0]*J[1][0])*Q->A[ii][ii]/det;
                Qab[ii] = (J[0][0]*J[0][1] + J[1][0]*J[1][1])*Q->A[ii][ii]/det;
                Qbb[ii] = (J[0][1]*J[0][1] + J[1][1]*J[1][1])*Q->A[ii][ii]/det;
                Qba[ii] = Qab[ii];

                // multiply by the velocity vector
                geom->interp1_l(ex, ey, ii%mp1, ii/mp1, velArray, _u);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rhoArray, &_r);

                Qaa[ii] *= (_u[0] * _r);
                Qba[ii] *= (_u[0] * _r);
                Qab[ii] *= (_u[1] * _r);
                Qbb[ii] *= (_u[1] * _r);
            }

            inds_intl_x = topo->elInds_intl_x_l(ex, ey);
            inds_intl_y = topo->elInds_intl_y_l(ex, ey);
            inds_skel_x = topo->elInds_skel_x_l(ex, ey);
            inds_skel_y = topo->elInds_skel_y_l(ex, ey);

            Ax_b(U->nDofsJ, Q->nDofsI, Ut, Qaa, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                if(inds_intl_x[ii] != -1) intlArray[inds_intl_x[ii]] += rhs[ii];
                if(inds_skel_x[ii] != -1) skelArray[inds_skel_x[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Ut, Qab, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                if(inds_intl_x[ii] != -1) intlArray[inds_intl_x[ii]] += rhs[ii];
                if(inds_skel_x[ii] != -1) skelArray[inds_skel_x[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Vt, Qba, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                if(inds_intl_y[ii] != -1) intlArray[inds_intl_y[ii]] += rhs[ii];
                if(inds_skel_y[ii] != -1) skelArray[inds_skel_y[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Vt, Qbb, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                if(inds_intl_y[ii] != -1) intlArray[inds_intl_y[ii]] += rhs[ii];
                if(inds_skel_y[ii] != -1) skelArray[inds_skel_y[ii]] += rhs[ii];
            }
        }
    }
    VecRestoreArray(b_intl, &intlArray);
    VecRestoreArray(b_skel, &skelArray);
    VecRestoreArray(vel, &velArray);
    VecRestoreArray(rho, &rhoArray);

    // scatter the skeleton vector to global indices
    VecScatterBegin(topo->gtol_skel, b_skel, b_skel_g, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_skel, b_skel, b_skel_g, ADD_VALUES, SCATTER_REVERSE);
}

void Assembly_M1DD::pack_schur_skel() {
    int ii, jj, ri, nCols, cols2[999];
    const int *cols;
    const double *vals;

    for(ii = 0; ii < topo->dd_n_skel_locl; ii++) {
        MatGetRow(Ss_l, ii, &nCols, &cols, &vals);
        ri = topo->dd_skel_locl_glob_map[ii];
        for(jj = 0; jj < nCols; jj++) {
            cols2[jj] = topo->dd_skel_locl_glob_map[cols[jj]];
        }
        MatSetValues(Ss, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(Ss_l, ii, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(Ss, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Ss, MAT_FINAL_ASSEMBLY);
}
