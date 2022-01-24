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
#include "M1DDSolve.h"

using namespace std;

#define PC_DD 1

#ifdef PC_DD
M1DDSolve::M1DDSolve(Topo* _topo, Geom* _geom) {
    PC pc;
    int size;
    M1x_j_xy_i* U;

    topo = _topo;
    geom = _geom;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    quad = new GaussLobatto(geom->quad->n);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // 1 form mass matrix
    M1 = new Umat(topo, geom, node, edge);
    EtoF = new E21mat(topo);

    // initialize the linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp, "sw_");
    KSPSetFromOptions(ksp);

    Q = new Wii(node->q, geom);
    U = new M1x_j_xy_i(node, edge);
    V = new M1y_j_xy_i(node, edge);
    Ut = Alloc2D(U->nDofsJ, U->nDofsI);
    Vt = Alloc2D(U->nDofsJ, U->nDofsI);
    Tran_IP(U->nDofsI, U->nDofsJ, U->A, Ut);
    Tran_IP(U->nDofsI, U->nDofsJ, V->A, Vt);

    // allocate the matrices
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_intl_locl, topo->dd_n_intl_locl, 4*U->nDofsJ, PETSC_NULL, &Mii);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_intl_locl, topo->dd_n_dual_locl, 4*U->nDofsJ, PETSC_NULL, &Mid);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_intl_locl, topo->dd_n_skel_locl, 4*U->nDofsJ, PETSC_NULL, &Mis);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_dual_locl, topo->dd_n_intl_locl, 4*U->nDofsJ, PETSC_NULL, &Mdi);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_dual_locl, topo->dd_n_dual_locl, 4*U->nDofsJ, PETSC_NULL, &Mdd);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_dual_locl, topo->dd_n_skel_locl, 4*U->nDofsJ, PETSC_NULL, &Mds);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_skel_locl, topo->dd_n_intl_locl, 4*U->nDofsJ, PETSC_NULL, &Msi);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_skel_locl, topo->dd_n_dual_locl, 4*U->nDofsJ, PETSC_NULL, &Msd);
    MatCreate(MPI_COMM_WORLD, &Mss);
    MatSetSizes(Mss, topo->dd_n_skel_locl, topo->dd_n_skel_locl, topo->dd_n_skel_glob, topo->dd_n_skel_glob);
    MatSetType(Mss, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Mss, 4*U->nDofsJ, PETSC_NULL, 4*U->nDofsJ, PETSC_NULL);

    // and the vectors
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_intl_locl, &b_intl);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_intl_locl, &x_intl);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_dual_locl, &b_dual);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_dual_locl, &x_dual);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_skel_locl, &b_skel);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_skel_locl, &x_skel);
    VecCreateMPI(MPI_COMM_WORLD, topo->dd_n_skel_locl, topo->dd_n_skel_glob, &b_skel_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->dd_n_skel_locl, topo->dd_n_skel_glob, &x_skel_g);
}

M1DDSolve::~M1DDSolve() {
    KSPDestroy(&ksp);
    MatDestroy(&Mii);
    MatDestroy(&Mid);
    MatDestroy(&Mis);
    MatDestroy(&Mdi);
    MatDestroy(&Mdd);
    MatDestroy(&Mds);
    MatDestroy(&Msi);
    MatDestroy(&Msd);
    MatDestroy(&Mss);
    VecDestroy(&b_intl);
    VecDestroy(&x_intl);
    VecDestroy(&b_dual);
    VecDestroy(&x_dual);
    VecDestroy(&b_skel);
    VecDestroy(&x_skel);
    delete M1;
    delete EtoF;
    delete U;
    delete V;
    delete Q;
    Free2D(U->nDofsJ, Ut);
    Free2D(U->nDofsJ, Vt);
    delete edge;
    delete node;
    delete quad;
}

void M1DDSolve::assemble_mat() {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_intl_x, *inds_intl_y, *inds_dual_x, *inds_dual_y, *inds_skel_x, *inds_skel_y, *inds_skel_x_g, *inds_skel_y_g;
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
    MatZeroEntries(Mid);
    MatZeroEntries(Mis);
    MatZeroEntries(Mdi);
    MatZeroEntries(Mdd);
    MatZeroEntries(Mds);
    MatZeroEntries(Msi);
    MatZeroEntries(Msd);
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
            inds_dual_x = topo->elInds_dual_x_l(ex, ey);
            inds_dual_y = topo->elInds_dual_y_l(ex, ey);
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
            MatSetValues(Mid, U->nDofsJ, inds_intl_x, U->nDofsJ, inds_dual_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mis, U->nDofsJ, inds_intl_x, U->nDofsJ, inds_skel_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mdi, U->nDofsJ, inds_dual_x, U->nDofsJ, inds_intl_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mdd, U->nDofsJ, inds_dual_x, U->nDofsJ, inds_dual_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mds, U->nDofsJ, inds_dual_x, U->nDofsJ, inds_skel_x, UtQUflat, ADD_VALUES);
            MatSetValues(Msi, U->nDofsJ, inds_skel_x, U->nDofsJ, inds_intl_x, UtQUflat, ADD_VALUES);
            MatSetValues(Msd, U->nDofsJ, inds_skel_x, U->nDofsJ, inds_dual_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mss, U->nDofsJ, inds_skel_x_g, U->nDofsJ, inds_skel_x_g, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, UtQV, UtQUflat);
            MatSetValues(Mii, U->nDofsJ, inds_intl_x, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mid, U->nDofsJ, inds_intl_x, U->nDofsJ, inds_dual_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mis, U->nDofsJ, inds_intl_x, U->nDofsJ, inds_skel_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mdi, U->nDofsJ, inds_dual_x, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mdd, U->nDofsJ, inds_dual_x, U->nDofsJ, inds_dual_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mds, U->nDofsJ, inds_dual_x, U->nDofsJ, inds_skel_y, UtQUflat, ADD_VALUES);
            MatSetValues(Msi, U->nDofsJ, inds_skel_x, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Msd, U->nDofsJ, inds_skel_x, U->nDofsJ, inds_dual_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mss, U->nDofsJ, inds_skel_x_g, U->nDofsJ, inds_skel_y_g, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQU, UtQUflat);
            MatSetValues(Mii, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_intl_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mid, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_dual_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mis, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_skel_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mdi, U->nDofsJ, inds_dual_y, U->nDofsJ, inds_intl_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mdd, U->nDofsJ, inds_dual_y, U->nDofsJ, inds_dual_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mds, U->nDofsJ, inds_dual_y, U->nDofsJ, inds_skel_x, UtQUflat, ADD_VALUES);
            MatSetValues(Msi, U->nDofsJ, inds_skel_y, U->nDofsJ, inds_intl_x, UtQUflat, ADD_VALUES);
            MatSetValues(Msd, U->nDofsJ, inds_skel_y, U->nDofsJ, inds_dual_x, UtQUflat, ADD_VALUES);
            MatSetValues(Mss, U->nDofsJ, inds_skel_y_g, U->nDofsJ, inds_skel_x_g, UtQUflat, ADD_VALUES);

            Flat2D_IP(U->nDofsJ, U->nDofsJ, VtQV, UtQUflat);
            MatSetValues(Mii, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mid, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_dual_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mis, U->nDofsJ, inds_intl_y, U->nDofsJ, inds_skel_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mdi, U->nDofsJ, inds_dual_y, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mdd, U->nDofsJ, inds_dual_y, U->nDofsJ, inds_dual_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mds, U->nDofsJ, inds_dual_y, U->nDofsJ, inds_skel_y, UtQUflat, ADD_VALUES);
            MatSetValues(Msi, U->nDofsJ, inds_skel_y, U->nDofsJ, inds_intl_y, UtQUflat, ADD_VALUES);
            MatSetValues(Msd, U->nDofsJ, inds_skel_y, U->nDofsJ, inds_dual_y, UtQUflat, ADD_VALUES);
            MatSetValues(Mss, U->nDofsJ, inds_skel_y_g, U->nDofsJ, inds_skel_y_g, UtQUflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(Mii, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Mid, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Mis, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Mdi, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Mdd, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Mds, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Msi, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Msd, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Mss, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mii, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mid, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mis, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mdi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mdd, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mds, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Msi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Msd, MAT_FINAL_ASSEMBLY);
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

void M1DDSolve::assemble_rhs_hu(Vec rho, Vec vel) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_intl_x, *inds_intl_y;
    int *inds_dual_x, *inds_dual_y;
    int *inds_skel_x, *inds_skel_y;
    double det, **J;
    PetscScalar *intlArray, *dualArray, *skelArray, *velArray, *rhoArray;
    double _u[2], _r, Qaa[99], Qab[99], Qba[99], Qbb[99], rhs[99];

    mp1 = node->q->n + 1;
    mp12 = mp1*mp1;

    VecZeroEntries(b_intl);
    VecZeroEntries(b_dual);
    VecZeroEntries(b_skel);

    VecGetArray(b_intl, &intlArray);
    VecGetArray(b_dual, &dualArray);
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
            inds_dual_x = topo->elInds_dual_x_l(ex, ey);
            inds_dual_y = topo->elInds_dual_y_l(ex, ey);
            inds_skel_x = topo->elInds_skel_x_l(ex, ey);
            inds_skel_y = topo->elInds_skel_y_l(ex, ey);

            Ax_b(U->nDofsJ, Q->nDofsI, Ut, Qaa, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                if(inds_intl_x[ii] != -1) intlArray[inds_intl_x[ii]] += rhs[ii];
                if(inds_dual_x[ii] != -1) dualArray[inds_dual_x[ii]] += rhs[ii];
                if(inds_skel_x[ii] != -1) skelArray[inds_skel_x[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Ut, Qab, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                if(inds_intl_x[ii] != -1) intlArray[inds_intl_x[ii]] += rhs[ii];
                if(inds_dual_x[ii] != -1) dualArray[inds_dual_x[ii]] += rhs[ii];
                if(inds_skel_x[ii] != -1) skelArray[inds_skel_x[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Vt, Qba, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                if(inds_intl_y[ii] != -1) intlArray[inds_intl_y[ii]] += rhs[ii];
                if(inds_dual_y[ii] != -1) dualArray[inds_dual_y[ii]] += rhs[ii];
                if(inds_skel_y[ii] != -1) skelArray[inds_skel_y[ii]] += rhs[ii];
            }

            Ax_b(U->nDofsJ, Q->nDofsI, Vt, Qbb, rhs);
            for(ii = 0; ii < U->nDofsJ; ii++) {
                if(inds_intl_y[ii] != -1) intlArray[inds_intl_y[ii]] += rhs[ii];
                if(inds_dual_y[ii] != -1) dualArray[inds_dual_y[ii]] += rhs[ii];
                if(inds_skel_y[ii] != -1) skelArray[inds_skel_y[ii]] += rhs[ii];
            }
        }
    }
    VecRestoreArray(b_intl, &intlArray);
    VecRestoreArray(b_dual, &dualArray);
    VecRestoreArray(b_skel, &skelArray);
    VecRestoreArray(vel, &velArray);
    VecRestoreArray(rho, &rhoArray);

    // TODO: scatter the skeleton vector to global indices
}

// upwinded test function matrix
void M1DDSolve::err1(Vec ug, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds0;
    double det, wd, l_inf;
    double un[2], dun[1], ua[2], dua[1];
    double local_1[2], global_1[2], local_2[2], global_2[2], local_i[2], global_i[2]; // first entry is the error, the second is the norm
    PetscScalar *array_1, *array_2;
    Vec ul, dug, dul;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &dul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dug);

    VecScatterBegin(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    MatMult(EtoF->E21, ug, dug);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local_1[0] = local_1[1] = 0.0;
    local_2[0] = local_2[1] = 0.0;
    local_i[0] = local_i[1] = 0.0;

    VecGetArray(ul, &array_1);
    VecGetArray(dug, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, un);
                ua[0] = fu(geom->x[inds0[ii]]);
                ua[1] = fv(geom->x[inds0[ii]]);

                det = geom->det[ei][ii];
                wd = det*quad->w[ii%mp1]*quad->w[ii/mp1];

                local_1[0] += wd*(fabs(un[0] - ua[0]) + fabs(un[1] - ua[1]));
                local_1[1] += wd*(fabs(ua[0]) + fabs(ua[1]));

                local_2[0] += wd*((un[0] - ua[0])*(un[0] - ua[0]) + (un[1] - ua[1])*(un[1] - ua[1]));
                local_2[1] += wd*(ua[0]*ua[0] + ua[1]*ua[1]);

                l_inf = wd*(fabs(un[0] - ua[0]) + fabs(un[1] - ua[1]));
                if(fabs(l_inf) > local_i[0]) {
                    local_i[0] = fabs(l_inf);
                    local_i[1] = wd*(fabs(ua[0]) + fabs(ua[1]));
                }
 
                if(fp != NULL) {
                    geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, dun);
                    dua[0] = fp(geom->x[inds0[ii]]);

                    local_2[0] += wd*(dun[0] - dua[0])*(dun[0] - dua[0]);
                    local_2[1] += wd*dua[0]*dua[0];
                }
            }
        }
    }
    VecRestoreArray(ul, &array_1);
    VecRestoreArray(dug, &array_2);

    MPI_Allreduce(local_1, global_1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_2, global_2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_i, global_i, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    VecDestroy(&ul);
    VecDestroy(&dul);
    VecDestroy(&dug);

    norms[0] = global_1[0]/global_1[1];
    norms[1] = sqrt(global_2[0]/global_2[1]);
    norms[2] = global_i[0]/global_i[1];
}
#endif
