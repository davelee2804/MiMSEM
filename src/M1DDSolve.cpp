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
    int ii, jj;
    double val;
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
    M2 = new Wmat(topo, geom, edge);
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
    MatCreateSeqDense(MPI_COMM_SELF, topo->dd_n_intl_locl, topo->dd_n_intl_locl, PETSC_NULL, &Mii_inv);
    MatCreateSeqDense(MPI_COMM_SELF, topo->dd_n_intl_locl+topo->dd_n_dual_locl, 
                                     topo->dd_n_intl_locl+topo->dd_n_dual_locl, PETSC_NULL, &Midid_inv);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_intl_locl+topo->dd_n_dual_locl, 
                                   topo->dd_n_skel_locl, 4*U->nDofsJ, PETSC_NULL, &Mid_s);
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_intl_locl, topo->dd_n_dual_locl+topo->dd_n_skel_locl, 
                                   4*U->nDofsJ, PETSC_NULL, &Mi_ds);
    MatCreate(MPI_COMM_WORLD, &Ss);
    MatSetSizes(Ss, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(Ss, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Ss, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);

    MatCreate(MPI_COMM_WORLD, &Sg);
    MatSetSizes(Sg, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(Sg, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Sg, 8*U->nDofsJ, PETSC_NULL, 8*U->nDofsJ, PETSC_NULL);

    // M0R_dual = [ 0 R_dual^{(i)T}]
    MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_dual_locl, topo->dd_n_intl_locl+topo->dd_n_dual_locl, 1, PETSC_NULL, &M0Rdual);
    MatZeroEntries(M0Rdual);
    val = 1.0;
    for(ii = 0; ii < topo->dd_n_dual_locl; ii++) {
        jj = ii + topo->dd_n_intl_locl;
        MatSetValues(M0Rdual, 1, &ii, 1, &jj, &val, INSERT_VALUES);
    }
    MatAssemblyBegin(M0Rdual, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M0Rdual, MAT_FINAL_ASSEMBLY);
    MatTranspose(M0Rdual, MAT_INITIAL_MATRIX, &M0Rdual_T);

    MatCreate(MPI_COMM_WORLD, &Phi);
    MatSetSizes(Phi, topo->dd_n_dual_locl+topo->dd_n_skel_locl, topo->dd_n_skel_locl,
              size*(topo->dd_n_dual_locl)+topo->dd_n_skel_glob, topo->dd_n_skel_glob);
    MatSetType(Phi, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Phi, topo->dd_n_skel_locl, PETSC_NULL, 0, PETSC_NULL);

    //MatCreateSeqAIJ(MPI_COMM_SELF, topo->dd_n_dual_locl, topo->dd_n_dual_locl, topo->dd_n_dual_locl, PETSC_NULL, &Mdd_inv);

    Mid_s_T                 = PETSC_NULL;
    Midid_inv_Mid_s         = PETSC_NULL;
    Ss_l                    = PETSC_NULL;
    M0Rdual_Midid_inv_Mid_s = PETSC_NULL;
    Midid_inv_M0Rdual_T     = PETSC_NULL;
    Mii_inv_Mid             = PETSC_NULL;
    Sd_inv                  = PETSC_NULL;
    ksp_ss                  = PETSC_NULL;

    // and the vectors
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_intl_locl, &b_intl);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_intl_locl, &x_intl);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_intl_locl, &t_intl);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_dual_locl, &b_dual);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_dual_locl, &x_dual);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_dual_locl, &t_dual);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_skel_locl, &b_skel);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_skel_locl, &x_skel);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_skel_locl, &t_skel);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_dual_locl+topo->dd_n_skel_locl, &b_dual_skel);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_dual_locl+topo->dd_n_skel_locl, &x_dual_skel);
    VecCreateSeq(MPI_COMM_SELF, topo->dd_n_dual_locl+topo->dd_n_skel_locl, &t_dual_skel);
    VecCreateMPI(MPI_COMM_WORLD, topo->dd_n_skel_locl, topo->dd_n_skel_glob, &b_skel_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->dd_n_skel_locl, topo->dd_n_skel_glob, &x_skel_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->dd_n_skel_locl, topo->dd_n_skel_glob, &t_skel_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->dd_n_dual_locl+topo->dd_n_skel_locl, 
                                 size*(topo->dd_n_dual_locl)+topo->dd_n_skel_glob, &b_dual_skel_g);
    VecDuplicate(b_dual_skel_g, &x_dual_skel_g);
    VecDuplicate(b_dual_skel_g, &t_dual_skel_g);
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
    MatDestroy(&Midid_inv);
    MatDestroy(&Mid_s);
    if(Midid_inv_Mid_s)         MatDestroy(&Midid_inv_Mid_s);
    if(Ss_l)                    MatDestroy(&Ss_l);
    if(M0Rdual_Midid_inv_Mid_s) MatDestroy(&M0Rdual_Midid_inv_Mid_s);
    if(Midid_inv_M0Rdual_T)     MatDestroy(&Midid_inv_M0Rdual_T);
    MatDestroy(&Mii_inv);
    if(Mid_s_T)                 MatDestroy(&Mid_s_T);
    if(Mii_inv_Mid)             MatDestroy(&Mii_inv_Mid);
    if(Sd_inv)                  MatDestroy(&Sd_inv);
    if(ksp_ss)                  KSPDestroy(&ksp_ss);
    MatDestroy(&M0Rdual);
    MatDestroy(&M0Rdual_T);
    MatDestroy(&Ss);
    MatDestroy(&Sg);
    MatDestroy(&Phi);
    //MatDestroy(&Mdd_inv);
    VecDestroy(&b_intl);
    VecDestroy(&x_intl);
    VecDestroy(&t_intl);
    VecDestroy(&b_dual);
    VecDestroy(&x_dual);
    VecDestroy(&t_dual);
    VecDestroy(&b_skel);
    VecDestroy(&x_skel);
    VecDestroy(&t_skel);
    VecDestroy(&b_dual_skel);
    VecDestroy(&x_dual_skel);
    VecDestroy(&t_dual_skel);
    VecDestroy(&b_skel_g);
    VecDestroy(&x_skel_g);
    VecDestroy(&t_skel_g);
    VecDestroy(&b_dual_skel_g);
    VecDestroy(&x_dual_skel_g);
    VecDestroy(&t_dual_skel_g);
    delete M1;
    delete M2;
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
    VecZeroEntries(b_skel_g);

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

    // scatter the skeleton vector to global indices
    VecScatterBegin(topo->gtol_skel, b_skel, b_skel_g, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_skel, b_skel, b_skel_g, ADD_VALUES, SCATTER_REVERSE);
}

void M1DDSolve::pack_intl_dual_sq() {
    int ii, jj, ri, nCols, cols2[999];
    const int *cols;
    const double *vals;

    MatZeroEntries(Midid_inv);

    for(ii = 0; ii < topo->dd_n_intl_locl; ii++) {
        MatGetRow(Mii, ii, &nCols, &cols, &vals);
        MatSetValues(Midid_inv, 1, &ii, nCols, cols, vals, INSERT_VALUES);
        MatRestoreRow(Mii, ii, &nCols, &cols, &vals);

        MatGetRow(Mid, ii, &nCols, &cols, &vals);
        for(jj = 0; jj < nCols; jj++) {
            cols2[jj] = cols[jj] + topo->dd_n_intl_locl;
        }
        MatSetValues(Midid_inv, 1, &ii, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Mid, ii, &nCols, &cols, &vals);
    }
    for(ii = 0; ii < topo->dd_n_dual_locl; ii++) {
        ri = ii + topo->dd_n_intl_locl;

        MatGetRow(Mdi, ii, &nCols, &cols, &vals);
        MatSetValues(Midid_inv, 1, &ri, nCols, cols, vals, INSERT_VALUES);
        MatRestoreRow(Mdi, ii, &nCols, &cols, &vals);

        MatGetRow(Mdd, ii, &nCols, &cols, &vals);
        for(jj = 0; jj < nCols; jj++) {
            cols2[jj] = cols[jj] + topo->dd_n_intl_locl;
        }
        MatSetValues(Midid_inv, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Mdd, ii, &nCols, &cols, &vals);
    }

    MatAssemblyBegin(Midid_inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Midid_inv, MAT_FINAL_ASSEMBLY);
    MatLUFactor(Midid_inv, PETSC_NULL, PETSC_NULL, PETSC_NULL);
}

void M1DDSolve::pack_intl_dual_skel() {
    int ii, jj, ri, nCols, cols2[999];
    const int *cols;
    const double *vals;

    MatZeroEntries(Mid_s);
    for(ii = 0; ii < topo->dd_n_intl_locl; ii++) {
        MatGetRow(Mis, ii, &nCols, &cols, &vals);
        MatSetValues(Mid_s, 1, &ii, nCols, cols, vals, INSERT_VALUES);
        MatRestoreRow(Mis, ii, &nCols, &cols, &vals);
    }
    for(ii = 0; ii < topo->dd_n_dual_locl; ii++) {
        ri = ii + topo->dd_n_intl_locl;

        MatGetRow(Mds, ii, &nCols, &cols, &vals);
        MatSetValues(Mid_s, 1, &ri, nCols, cols, vals, INSERT_VALUES);
        MatRestoreRow(Mds, ii, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(Mid_s, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Mid_s, MAT_FINAL_ASSEMBLY);

    MatZeroEntries(Mi_ds);
    for(ii = 0; ii < topo->dd_n_intl_locl; ii++) {
        MatGetRow(Mid, ii, &nCols, &cols, &vals);
        MatSetValues(Mi_ds, 1, &ii, nCols, cols, vals, INSERT_VALUES);
        MatRestoreRow(Mid, ii, &nCols, &cols, &vals);

        MatGetRow(Mis, ii, &nCols, &cols, &vals);
        for(jj = 0; jj < nCols; jj++) {
            cols2[jj] = cols[jj] + topo->dd_n_dual_locl;
        }
        MatSetValues(Mi_ds, 1, &ii, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Mis, ii, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(Mi_ds, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Mi_ds, MAT_FINAL_ASSEMBLY);

    // compute the transpose
    if(!Mid_s_T) {
        MatTranspose(Mid_s, MAT_INITIAL_MATRIX, &Mid_s_T);
    } else {
        MatTranspose(Mid_s, MAT_REUSE_MATRIX, &Mid_s_T);
    }
}

// extract [dual dual] blocl from [intl dual; intl dual]^{-1}
void M1DDSolve::pack_dual_dual_inv() {
    int ii, jj, ri, nCols, nCols2, cols2[999];
    double vals2[999];
    const int *cols;
    const double *vals;

    MatZeroEntries(Mdd_inv);
    for(ii = 0; ii < topo->dd_n_dual_locl; ii++) {
        ri = ii + topo->dd_n_intl_locl;
        MatGetRow(Midid_inv, ri, &nCols, &cols, &vals);
        nCols2 = 0;
        for(jj = 0; jj < nCols; jj++) {
            if(cols[jj] > topo->dd_n_intl_locl) {
                cols2[nCols2] = cols[jj] - topo->dd_n_intl_locl;
                vals2[nCols2] = vals[jj];
                nCols2++;
            }
        }
        if(nCols2) {
            MatSetValues(Mdd_inv, 1, &ii, nCols2, cols2, vals2, INSERT_VALUES);
        }
        MatRestoreRow(Midid_inv, ri, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(Mdd_inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Mdd_inv, MAT_FINAL_ASSEMBLY);
}

// Phi shape: [n_dual + n_skel] X [n_skel]
void M1DDSolve::pack_phi() {
    int ii, jj, ri, cj, nCols, cols2[999];
    const int *cols;
    const double *vals, one = 1.0;

    MatZeroEntries(Phi);
    for(ii = 0; ii < topo->dd_n_dual_locl; ii++) {
        MatGetRow(Midid_inv_Mid_s, ii, &nCols, &cols, &vals);
	ri = ii + rank*(topo->dd_n_dual_locl + topo->dd_n_skel_locl);
        for(jj = 0; jj < nCols; jj++) {
            cols2[jj] = topo->dd_skel_locl_glob_map[cols[jj]];
        }
        MatSetValues(Phi, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(Midid_inv_Mid_s, ii, &nCols, &cols, &vals);
    }
    for(ii = 0; ii < topo->dd_n_skel_locl; ii++) {
        ri = rank*(topo->dd_n_dual_locl + topo->dd_n_skel_locl) + topo->dd_n_dual_locl + ii;
        cj = rank*(topo->dd_n_dual_locl + topo->dd_n_skel_locl) + ii;
        MatSetValues(Phi, 1, &ri, 1, &cj, &one, ADD_VALUES);
    }
    MatAssemblyBegin(Phi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Phi, MAT_FINAL_ASSEMBLY);
}

void M1DDSolve::pack_schur_skel() {
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

void M1DDSolve::pack_dual_skel_g(Vec dual, Vec skel_g, Vec dual_skel_g) {
    int ii, jj;
    PetscScalar *skel_array, *dual_array, *dual_skel_array;

    VecGetArray(b_dual, &dual_array);
    VecGetArray(b_skel_g, &skel_array);
    VecGetArray(b_dual_skel_g, &dual_skel_array);
    for(ii = 0; ii < topo->dd_n_dual_locl; ii++) {
        dual_skel_array[ii] = dual_array[ii];
    }
    for(ii = 0; ii < topo->dd_n_skel_locl; ii++) {
        jj = ii + topo->dd_n_dual_locl;
        dual_skel_array[jj] = skel_array[ii];
    }
    VecRestoreArray(dual_skel_g, &dual_skel_array);
    VecRestoreArray(skel_g, &skel_array);
    VecRestoreArray(dual, &dual_array);
}

void M1DDSolve::scat_dual_skel_l2g(Vec dual, Vec skel, Vec skel_g, Vec dual_skel_l, Vec dual_skel_g) {
    int ii;
    PetscScalar *dsArray, *sArray, *dArray;
 
    VecGetArray(skel, &sArray);
    VecGetArray(dual_skel_l, &dsArray);
    for(ii = 0; ii < topo->dd_n_skel_locl; ii++) {
        sArray[ii] = dsArray[ii+topo->dd_n_dual_locl];
    }
    VecRestoreArray(dual_skel_l, &dsArray);
    VecRestoreArray(skel, &sArray);
    VecZeroEntries(skel_g);
    VecScatterBegin(topo->gtol_skel, skel, skel_g, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_skel, skel, skel_g, ADD_VALUES, SCATTER_REVERSE);
    VecGetArray(dual, &dArray);
    VecGetArray(dual_skel_g, &dsArray);
    for(ii = 0; ii < topo->dd_n_dual_locl; ii++) {
        dsArray[ii] = dArray[ii];
    }
    VecRestoreArray(dual_skel_g, &dsArray);
    VecRestoreArray(dual, &dArray);
}

void M1DDSolve::setup_matrices() {
    assemble_mat();
    // construct the internal-dual (local) dof schur complement inverse
    MatCopy(Mii, Mii_inv, DIFFERENT_NONZERO_PATTERN);
    MatLUFactor(Mii_inv, PETSC_NULL, PETSC_NULL, PETSC_NULL);
    if(Sd_inv) {
        MatMatMult(Mii_inv, Mid,         MAT_REUSE_MATRIX, PETSC_DEFAULT, &Mii_inv_Mid);
        MatMatMult(Mdi,     Mii_inv_Mid, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Sd_inv);
    } else {
        MatMatMult(Mii_inv, Mid,         MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mii_inv_Mid);
        MatMatMult(Mdi,     Mii_inv_Mid, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Sd_inv);
    }
    MatAYPX(Sd_inv, -1.0, Mdd, DIFFERENT_NONZERO_PATTERN);
    MatLUFactor(Sd_inv, PETSC_NULL, PETSC_NULL, PETSC_NULL);
    // construct the skeleton (global) dof schur complement
    {
        Mat Sd_inv_Mdi;
        Mat Sd_inv_Mdi_Mii_inv;                 // x2
        Mat Sd_inv_Mdi_Mii_inv_Mis;
        Mat A10;
        Mat Mid_Sd_inv_Mdi_Mii_inv;
        Mat Mii_inv_Mid_Sd_inv_Mdi_Mii_inv;
        Mat Mii_inv_Mid_Sd_inv_Mdi_Mii_inv_Mis;
        Mat Sd_inv_Mds;                         // x2
        Mat Mid_Sd_inv_Mds;
        Mat Mii_inv_Mid_Sd_inv_Mds;
        Mat A01;
        Mat A11;

        MatMatMult(Sd_inv, Mdi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Sd_inv_Mdi);
        MatMatMult(Sd_inv_Mdi, Mii_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Sd_inv_Mdi_Mii_inv);
        MatMatMult(Sd_inv_Mdi_Mii_inv, Mis, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Sd_inv_Mdi_Mii_inv_Mis);
        MatMatMult(Msd, Sd_inv_Mdi_Mii_inv_Mis, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A10);

        MatMatMult(Mid, Sd_inv_Mdi_Mii_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mid_Sd_inv_Mdi_Mii_inv);
        MatMatMult(Mii_inv, Mid_Sd_inv_Mdi_Mii_inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mii_inv_Mid_Sd_inv_Mdi_Mii_inv);
        MatAXPY(Mii_inv_Mid_Sd_inv_Mdi_Mii_inv_Mis, +1.0, Mii_inv, DIFFERENT_NONZERO_PATTERN);
        MatMatMult(Mii_inv_Mid_Sd_inv_Mdi_Mii_inv, Mis, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mii_inv_Mid_Sd_inv_Mdi_Mii_inv_Mis);
        if(Ss_l) {
            MatZeroEntries(Ss_l);
            MatMatMult(Msi, Mii_inv_Mid_Sd_inv_Mdi_Mii_inv_Mis, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Ss_l);
        } else {
            MatMatMult(Msi, Mii_inv_Mid_Sd_inv_Mdi_Mii_inv_Mis, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Ss_l);
        }

        MatMatMult(Sd_inv, Mds, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Sd_inv_Mds);
        MatMatMult(Mid, Sd_inv_Mds, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mid_Sd_inv_Mds);
        MatMatMult(Mii_inv, Mid_Sd_inv_Mds, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mii_inv_Mid_Sd_inv_Mds);
        MatMatMult(Mdi, Mii_inv_Mid_Sd_inv_Mds, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A01);

        MatMatMult(Mds, Sd_inv_Mds, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A11);

	MatAYPX(Ss_l, -1.0, A01, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(Ss_l, +1.0, A10, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(Ss_l, -1.0, A11, DIFFERENT_NONZERO_PATTERN);
        
        MatDestroy(&Sd_inv_Mdi);
        MatDestroy(&Sd_inv_Mdi_Mii_inv);
        MatDestroy(&Sd_inv_Mdi_Mii_inv_Mis);
        MatDestroy(&A10);
        MatDestroy(&Mid_Sd_inv_Mdi_Mii_inv);
        MatDestroy(&Mii_inv_Mid_Sd_inv_Mdi_Mii_inv);
        MatDestroy(&Mii_inv_Mid_Sd_inv_Mdi_Mii_inv_Mis);
        MatDestroy(&Sd_inv_Mds);
        MatDestroy(&Mid_Sd_inv_Mds);
        MatDestroy(&Mii_inv_Mid_Sd_inv_Mds);
        MatDestroy(&A01);
        MatDestroy(&A11);
    }
    MatZeroEntries(Ss);
    pack_schur_skel();
    MatAXPY(Ss, +1.0, Mss, DIFFERENT_NONZERO_PATTERN);
    
/*
    pack_intl_dual_sq();
    pack_intl_dual_skel();
    if(Midid_inv_Mid_s) {
        MatMatMult(Midid_inv, Mid_s,           MAT_REUSE_MATRIX, PETSC_DEFAULT, &Midid_inv_Mid_s);
        MatMatMult(Mid_s_T,   Midid_inv_Mid_s, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Ss_l);
        MatMatMult(M0Rdual,   Midid_inv_Mid_s, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M0Rdual_Midid_inv_Mid_s);
    } else {
        MatMatMult(Midid_inv, Mid_s,           MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Midid_inv_Mid_s);
        MatMatMult(Mid_s_T,   Midid_inv_Mid_s, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Ss_l);
        MatMatMult(M0Rdual,   Midid_inv_Mid_s, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M0Rdual_Midid_inv_Mid_s);
    }
    pack_dual_dual_inv();

    // S_{ss}
    MatZeroEntries(Ss);
    pack_schur_skel();
    MatAYPX(Ss, -1.0, Mss, DIFFERENT_NONZERO_PATTERN);

    // [Aid]^{-1}[0 Rd]^T
    if(Midid_inv_M0Rdual_T) {
        MatMatMult(Midid_inv, M0Rdual_T, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Midid_inv_M0Rdual_T);
    } else {
        MatMatMult(Midid_inv, M0Rdual_T, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Midid_inv_M0Rdual_T);
    }

    pack_phi();
*/
    if(!ksp_ss) {
        PC pc;
        KSPCreate(MPI_COMM_WORLD, &ksp_ss);
        KSPSetOperators(ksp_ss, Ss, Ss);
        KSPSetTolerances(ksp_ss, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
        KSPSetType(ksp_ss, KSPGMRES);
        KSPGetPC(ksp_ss, &pc);
        PCSetType(pc, PCBJACOBI);
        PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
        KSPSetOptionsPrefix(ksp_ss, "dd_");
        KSPSetFromOptions(ksp_ss);
    }
}

void M1DDSolve::solve_F(Vec h, Vec ul, bool do_rhs) {
    if(do_rhs) {
        assemble_rhs_hu(h, ul);
        pack_dual_skel_g(b_dual, b_skel_g, b_dual_skel_g);
    }

    // boundary dof solve
    // global component
    MatMult(Mii_inv, b_intl, t_intl);
    MatMultTranspose(Mi_ds, t_intl, t_dual_skel);
    // scatter the skeleton dofs
    scat_dual_skel_l2g(t_dual, t_skel, t_skel_g, t_dual_skel, t_dual_skel_g);
    VecAYPX(t_dual_skel_g, -1.0, b_dual_skel_g);
    // global part of the solve, [Phi][S_s]^{-1}[Phi]^T
    MatMultTranspose(Phi, t_dual_skel_g, t_skel_g);
    KSPSolve(ksp_ss, t_skel_g, x_skel_g);
    MatMult(Phi, x_skel_g, x_dual_skel_g);
    // local component
    //MatMult(Mdd_inv, b_dual, t_dual);
    MatMult(Sd_inv, b_dual, t_dual);
    pack_dual_skel_g(t_dual, t_skel_g, t_dual_skel_g);
    VecAXPY(x_dual_skel_g, +1.0, t_dual_skel_g);

    // internal dof solve
}

void M1DDSolve::init1(Vec u, ICfunc* func_x, ICfunc* func_y) {
    int ex, ey, ii, mp1, mp12;
    int *inds0, *loc02;
    UtQmat* UQ = new UtQmat(topo, geom, node, edge);
    PetscScalar *bArray;
    Vec bl, bg, UQb;
    IS isl, isg;
    VecScatter scat;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, 2*geom->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, 2*geom->n0l, 2*geom->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &UQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[2*inds0[ii]+0] = func_x(geom->x[inds0[ii]]);
                bArray[2*inds0[ii]+1] = func_y(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);

    // create a new vec scatter object to handle vector quantity on nodes
    loc02 = new int[2*geom->n0];
    for(ii = 0; ii < geom->n0; ii++) {
        loc02[2*ii+0] = 2*geom->loc0[ii]+0;
        loc02[2*ii+1] = 2*geom->loc0[ii]+1;
    }
    ISCreateStride(MPI_COMM_WORLD, 2*geom->n0, 0, 1, &isl);
    ISCreateGeneral(MPI_COMM_WORLD, 2*geom->n0, loc02, PETSC_COPY_VALUES, &isg);
    VecScatterCreate(bg, isg, bl, isl, &scat);
    VecScatterBegin(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(UQ->M, bg, UQb);
    KSPSolve(ksp, UQb, u);

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&UQb);
    ISDestroy(&isl);
    ISDestroy(&isg);
    VecScatterDestroy(&scat);
    delete UQ;
    delete[] loc02;
}

void M1DDSolve::init2(Vec h, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    PetscScalar *bArray;
    KSP ksp2;
    Vec bl, bg, WQb;
    WtQmat* WQ = new WtQmat(topo, geom, edge);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, geom->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &WQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);
    VecScatterBegin(geom->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(geom->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(WQ->M, bg, WQb);

    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPSetOptionsPrefix(ksp2, "init2_");
    KSPSetFromOptions(ksp2);
    KSPSolve(ksp2, WQb, h);

    delete WQ;
    KSPDestroy(&ksp2);
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
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
