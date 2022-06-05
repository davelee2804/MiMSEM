#include <iostream>
#include <fstream>

#include <mpi.h>
#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscksp.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "L2Vecs.h"
#include "ElMats.h"
#include "VertOps.h"
#include "Assembly.h"
#include "HorizSolve.h"
#include "VertSolve.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
//#define RAYLEIGH (1.0/120.0)
#define RAYLEIGH (4.0/120.0)

using namespace std;

VertSolve::VertSolve(Topo* _topo, Geom* _geom, double _dt) {
    int ii, elOrd2;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    vo = new VertOps(topo, geom);

    elOrd2 = topo->elOrd * topo->elOrd;

    gv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &gv[ii]);
    }
    zv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &zv[ii]);
    }
    initGZ();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &_Phi_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_tmpA1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_tmpA2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &_tmpB1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &_tmpB2);

    _PCz = NULL;
    pc_A_rt = NULL;
    _V0_invV0_rt = NULL;
    ksp_pi = ksp_rho = NULL;
    pc_V01VBA = NULL;
    G_rho = G_rt = NULL;
    pc_VB_rt_invVB_pi = NULL;
    UdotGRAD = NULL;
    M_u_inv = NULL;
    N_pi = NULL;

    KSPCreate(MPI_COMM_SELF, &ksp_w);
    KSPSetOperators(ksp_w, vo->VA, vo->VA);
    KSPGetPC(ksp_w, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(ksp_w, "ksp_w_");
    KSPSetFromOptions(ksp_w);

    theta_h = new L2Vecs(geom->nk+1, topo, geom);
    exner_h = new L2Vecs(geom->nk+0, topo, geom);
    horiz = new HorizSolve(topo, geom);

    step = 0;
}

void VertSolve::initGZ() {
    int ex, ey, ei, ii, kk, n2, mp12;
    int* inds0;
    int inds2k[99], inds0k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double* Q0 = new double[Q->nDofsI];
    double* Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double* WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    Vec gz;
    Mat GRAD, BQ;
    PetscScalar* zArray;

    n2   = topo->elOrd*topo->elOrd;
    mp12 = (quad->n + 1)*(quad->n + 1);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*mp12, &gz);

    MatCreate(MPI_COMM_SELF, &BQ);
    MatSetType(BQ, MATSEQAIJ);
    MatSetSizes(BQ, (geom->nk+0)*n2, (geom->nk+1)*mp12, (geom->nk+0)*n2, (geom->nk+1)*mp12);
    MatSeqAIJSetPreallocation(BQ, 2*mp12, PETSC_NULL);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = geom->elInds0_l(ex, ey);

            MatZeroEntries(BQ);
            for(kk = 0; kk < geom->nk; kk++) {
                for(ii = 0; ii < mp12; ii++) {
                    Q0[ii]  = Q->A[ii]*SCALE;
                    // for linear field we multiply by the vertical jacobian determinant when
                    // integrating, and do no other trasformations for the basis functions
                    Q0[ii] *= 0.5;
                }
                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);

                for(ii = 0; ii < W->nDofsJ; ii++) {
                    inds2k[ii] = ii + kk*W->nDofsJ;
                }

                // assemble the first basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+0)*mp12;
                }
                MatSetValues(BQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQ, ADD_VALUES);
                // assemble the second basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+1)*mp12;
                }
                MatSetValues(BQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQ, ADD_VALUES);
            }
            MatAssemblyBegin(BQ, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(BQ, MAT_FINAL_ASSEMBLY);

            if(!ei) {
                MatMatMult(vo->V01, BQ, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GRAD);
            } else {
                MatMatMult(vo->V01, BQ, MAT_REUSE_MATRIX, PETSC_DEFAULT, &GRAD);
            }

            VecZeroEntries(gz);
            VecGetArray(gz, &zArray);
            for(kk = 0; kk < geom->nk+1; kk++) {
                for(ii = 0; ii < mp12; ii++) {
                    zArray[kk*mp12+ii] = GRAVITY*geom->levs[kk][inds0[ii]];
                }
            }
            VecRestoreArray(gz, &zArray);
            MatMult(GRAD, gz, gv[ei]);
            MatMult(BQ,   gz, zv[ei]);
        }
    }

    VecDestroy(&gz);
    MatDestroy(&GRAD);
    MatDestroy(&BQ);
    delete[] Q0;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    delete W;
    delete Q;
}

VertSolve::~VertSolve() {
    VecDestroy(&_Phi_z);
    VecDestroy(&_tmpA1);
    VecDestroy(&_tmpA2);
    VecDestroy(&_tmpB1);
    VecDestroy(&_tmpB2);

    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
        VecDestroy(&zv[ii]);
    }
    delete[] gv;
    delete[] zv;

    delete vo;

    delete edge;
    delete node;
    delete quad;

    if(_PCz)    MatDestroy(&_PCz);
    if(ksp_w)   KSPDestroy(&ksp_w);
    if(ksp_pi)  KSPDestroy(&ksp_pi);
    if(ksp_rho) KSPDestroy(&ksp_rho);

    delete theta_h;
    delete exner_h;
    delete horiz;

    if(G_rt) {
        MatDestroy(&VAB);
        MatDestroy(&pc_V0_invV0_rt_DT);
        MatDestroy(&pc_V0_invV0_rt);
        MatDestroy(&pc_V0_invDTV1);
        MatDestroy(&pc_VB_rt_invVB_pi);
        MatDestroy(&pc_DTV1);
        MatDestroy(&pc_DV0_invV0_rt);
        MatDestroy(&_V0_invV0_rt);
        MatDestroy(&G_pi_N_pi_inv);
        MatDestroy(&Q_rt_rho_M_rho_inv);
        MatDestroy(&Q_rt_rho);
        MatDestroy(&D_rt_M_u_inv);
        MatDestroy(&D_rt);
        MatDestroy(&D_rho);
        MatDestroy(&N_pi_inv);
        MatDestroy(&N_rt);
        MatDestroy(&M_rt);
        MatDestroy(&M_u_inv);
        MatDestroy(&M_rho_inv);
        MatDestroy(&G_pi);
        MatDestroy(&G_rt);
    }
}

double VertSolve::MaxNorm(Vec dx, Vec x, double max_norm) {
    double norm_dx, norm_x, new_max_norm;

    VecNorm(dx, NORM_2, &norm_dx);
    VecNorm(x, NORM_2, &norm_x);
    new_max_norm = (norm_dx/norm_x > max_norm) ? norm_dx/norm_x : max_norm;
    return new_max_norm;
}

void VertSolve::diagnose_F_z(int ex, int ey, Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec _F) {
    MatReuse reuse = (!_V0_invV0_rt) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    VecZeroEntries(_F);

    vo->AssembleLinearInv(ex, ey, vo->VA_inv);

    vo->AssembleLinearWithRT(ex, ey, rho1, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &_V0_invV0_rt);

    MatMult(_V0_invV0_rt, velz1, _tmpA1);
    VecAXPY(_F, 1.0/3.0, _tmpA1);

    MatMult(_V0_invV0_rt, velz2, _tmpA1);
    VecAXPY(_F, 1.0/6.0, _tmpA1);

    vo->AssembleLinearWithRT(ex, ey, rho2, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &_V0_invV0_rt);

    MatMult(_V0_invV0_rt, velz1, _tmpA1);
    VecAXPY(_F, 1.0/6.0, _tmpA1);

    MatMult(_V0_invV0_rt, velz2, _tmpA1);
    VecAXPY(_F, 1.0/3.0, _tmpA1);
}

void VertSolve::diagnose_Phi_z(int ex, int ey, Vec velz1, Vec velz2, Vec Phi) {
    int ei = ey*topo->nElsX + ex;

    VecZeroEntries(Phi);
    VecZeroEntries(_tmpB2);

    // kinetic energy term
    MatZeroEntries(vo->VBA);
    vo->AssembleConLinWithW(ex, ey, velz1, vo->VBA);

    MatMult(vo->VBA, velz1, _tmpB1);
    VecAXPY(Phi, 1.0/6.0, _tmpB1);
    
    MatMult(vo->VBA, velz2, _tmpB1);
    VecAXPY(Phi, 1.0/6.0, _tmpB1);

    MatZeroEntries(vo->VBA);
    vo->AssembleConLinWithW(ex, ey, velz2, vo->VBA);

    MatMult(vo->VBA, velz2, _tmpB1);
    VecAXPY(Phi, 1.0/6.0, _tmpB1);

    // potential energy term
    VecAXPY(Phi, 1.0, zv[ei]);
}

/* All vectors, rho, rt and theta are VERTICAL vectors */
void VertSolve::diagTheta2(Vec* rho, Vec* rt, Vec* theta) {
    int ex, ey, n2, ei;
    Vec frt;
    PC pc;
    KSP kspColA2;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &frt);

    KSPCreate(MPI_COMM_SELF, &kspColA2);
    KSPSetOperators(kspColA2, vo->VA2, vo->VA2);
    KSPGetPC(kspColA2, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColA2, "kspColA2_");
    KSPSetFromOptions(kspColA2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            vo->AssembleLinCon2(ex, ey, vo->VAB2);
            MatMult(vo->VAB2, rt[ei], frt);

            vo->AssembleLinearWithRho2(ex, ey, rho[ei], vo->VA2);
            KSPSolve(kspColA2, frt, theta[ei]);
        }
    }
    VecDestroy(&frt);
    KSPDestroy(&kspColA2);
}

void VertSolve::diagTheta_up(Vec* rho, Vec* rt, Vec* theta, Vec* ul) {
    int ex, ey, n2, ei;
    Vec frt;
    PC pc;
    KSP kspColA2;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &frt);

    KSPCreate(MPI_COMM_SELF, &kspColA2);
    KSPSetOperators(kspColA2, vo->VA2, vo->VA2);
    KSPGetPC(kspColA2, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColA2, "kspColA2_");
    KSPSetFromOptions(kspColA2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            vo->AssembleLinCon2_up(ex, ey, vo->VAB2, dt, ul);
            MatMult(vo->VAB2, rt[ei], frt);

            vo->AssembleLinearWithRho2_up(ex, ey, rho[ei], vo->VA2, dt, ul);
            KSPSolve(kspColA2, frt, theta[ei]);
        }
    }
    VecDestroy(&frt);
    KSPDestroy(&kspColA2);
}

void VertSolve::assemble_residual(int ex, int ey, Vec theta, Vec Pi, 
                                  Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fw, Vec _F, Vec _G) 
{
    double dot = 0.0;

    // diagnose the hamiltonian derivatives
    diagnose_F_z(ex, ey, velz1, velz2, rho1, rho2, _F);
    diagnose_Phi_z(ex, ey, velz1, velz2, _Phi_z);

    // assemble the momentum equation residual
    vo->AssembleLinear(ex, ey, vo->VA);
    MatMult(vo->VA, velz2, fw);

    MatMult(vo->VA, velz1, _tmpA1);
    VecAXPY(fw, -1.0, _tmpA1);

    MatMult(vo->V01, _Phi_z, _tmpA1);
    VecAXPY(fw, +dt, _tmpA1); // bernoulli function term

    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, Pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMult(vo->VA, _tmpA2, _tmpA1);
    VecAXPY(fw, +dt, _tmpA1); // pressure gradient term

    // compute the kinetic to internal energy power
    VecDot(_F, _tmpA1, &dot);
    k2i_z += dot / SCALE;

    // update the temperature equation flux
    MatMult(vo->VA, _F, _tmpA1); // includes theta
    MatMult(vo->VA_inv, _tmpA1, _G);

    // add the rayleigh friction
#ifdef RAYLEIGH
    vo->AssembleRayleigh(ex, ey, vo->VA);
    MatMult(vo->VA, velz2, _tmpA1);
    VecAXPY(fw, 0.5*dt*RAYLEIGH, _tmpA1);
    MatMult(vo->VA, velz1, _tmpA1);
    VecAXPY(fw, 0.5*dt*RAYLEIGH, _tmpA1);
#endif
}

void VertSolve::solve_schur_column_3(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                   Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi, int itt) 
{
    int n2 = topo->elOrd*topo->elOrd;
    MatReuse reuse = (!G_rt) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    MatReuse reuse_2 = (!pc_VB_rt_invVB_pi) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    if(!M_u_inv) {
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk-1)*n2, (geom->nk-1)*n2, n2, NULL, &M_u_inv);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &M_rho_inv);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &M_rt);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &N_pi_inv);
    }

    // assemble the operators for the coupled system
#ifdef RAYLEIGH
    vo->AssembleLinearWithRayleighInv(ex, ey, 0.5*dt*RAYLEIGH, M_u_inv);
#else
    vo->AssembleLinearInv(ex, ey, M_u_inv);
#endif
    vo->AssembleConst(ex, ey, M_rt);
    vo->AssembleConstInv(ex, ey, M_rho_inv);
    vo->Assemble_EOS_BlockInv(ex, ey, pi, NULL, N_pi_inv);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, reuse, &VAB);
    MatAssemblyBegin(VAB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (VAB, MAT_FINAL_ASSEMBLY);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(VAB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatAssemblyBegin(pc_V0_invV0_rt_DT, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_V0_invV0_rt_DT, MAT_FINAL_ASSEMBLY);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, 0.5*dt);
    MatAssemblyBegin(G_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (G_rt, MAT_FINAL_ASSEMBLY);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    MatAssemblyBegin(pc_DTV1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_DTV1, MAT_FINAL_ASSEMBLY);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    MatAssemblyBegin(pc_V0_invDTV1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_V0_invDTV1, MAT_FINAL_ASSEMBLY);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);
    MatAssemblyBegin(G_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (G_pi, MAT_FINAL_ASSEMBLY);

    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatAssemblyBegin(pc_V0_invV0_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_V0_invV0_rt, MAT_FINAL_ASSEMBLY);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatAssemblyBegin(pc_DV0_invV0_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_DV0_invV0_rt, MAT_FINAL_ASSEMBLY);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);
    MatAssemblyBegin(D_rho, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (D_rho, MAT_FINAL_ASSEMBLY);

    vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    MatMatMult(vo->VB, vo->V10, reuse, PETSC_DEFAULT, &D_rt);
    MatScale(D_rt, 0.5*dt);
    MatAssemblyBegin(D_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (D_rt, MAT_FINAL_ASSEMBLY);

    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse_2, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatAssemblyBegin(pc_VB_rt_invVB_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_VB_rt_invVB_pi, MAT_FINAL_ASSEMBLY);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse_2, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
    MatAssemblyBegin(N_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (N_rt, MAT_FINAL_ASSEMBLY);

    vo->AssembleConstWithTheta(ex, ey, theta, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    MatAssemblyBegin(pc_DTV1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_DTV1, MAT_FINAL_ASSEMBLY);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    MatAssemblyBegin(pc_V0_invDTV1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_V0_invDTV1, MAT_FINAL_ASSEMBLY);
vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    MatMatMult(vo->VBA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &Q_rt_rho);
    MatScale(Q_rt_rho, 0.5*dt);
    MatAssemblyBegin(Q_rt_rho, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Q_rt_rho, MAT_FINAL_ASSEMBLY);

    MatMatMult(Q_rt_rho, M_rho_inv, reuse_2, PETSC_DEFAULT, &Q_rt_rho_M_rho_inv);
    MatAssemblyBegin(Q_rt_rho_M_rho_inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Q_rt_rho_M_rho_inv, MAT_FINAL_ASSEMBLY);

    MatMatMult(G_pi, N_pi_inv, reuse_2, PETSC_DEFAULT, &G_pi_N_pi_inv);
    MatAssemblyBegin(G_pi_N_pi_inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (G_pi_N_pi_inv, MAT_FINAL_ASSEMBLY);
    //MatMatMult(G_pi_N_pi_inv, N_rt, reuse, PETSC_DEFAULT, &G_pi_N_pi_inv_N_rt);
    MatMatMult(G_pi_N_pi_inv, N_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &G_pi_N_pi_inv_N_rt);
    MatAYPX(G_pi_N_pi_inv_N_rt, -1.0, G_rt, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(G_pi_N_pi_inv_N_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (G_pi_N_pi_inv_N_rt, MAT_FINAL_ASSEMBLY);

    // assemble the schur complement operator
    //MatMatMult(Q_rt_rho_M_rho_inv, D_rho, reuse, PETSC_DEFAULT, &Q_rt_rho_M_rho_inv_D_rho);
    MatMatMult(Q_rt_rho_M_rho_inv, D_rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Q_rt_rho_M_rho_inv_D_rho);
    MatAYPX(Q_rt_rho_M_rho_inv_D_rho, -1.0, D_rt, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(Q_rt_rho_M_rho_inv_D_rho, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Q_rt_rho_M_rho_inv_D_rho, MAT_FINAL_ASSEMBLY);
    MatMatMult(Q_rt_rho_M_rho_inv_D_rho, M_u_inv, reuse_2, PETSC_DEFAULT, &D_rt_M_u_inv);
    MatAssemblyBegin(D_rt_M_u_inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (D_rt_M_u_inv, MAT_FINAL_ASSEMBLY);
    //MatMatMult(D_rt_M_u_inv, G_pi_N_pi_inv_N_rt, reuse, PETSC_DEFAULT, &L_rt_rt);
    MatMatMult(D_rt_M_u_inv, G_pi_N_pi_inv_N_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L_rt_rt);
    MatAYPX(L_rt_rt, -1.0, M_rt, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(L_rt_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (L_rt_rt, MAT_FINAL_ASSEMBLY);

    // update the residuals
    MatMult(Q_rt_rho_M_rho_inv, F_rho, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);

    MatMult(G_pi_N_pi_inv, F_pi, _tmpA1);
    VecAXPY(F_u, -1.0, _tmpA1);

    MatMult(D_rt_M_u_inv, F_u, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);

    // schur complement solve
    //if(reuse == MAT_INITIAL_MATRIX) {
        PC pc;

        KSPCreate(MPI_COMM_SELF, &ksp_pi);
        KSPSetOperators(ksp_pi, L_rt_rt, L_rt_rt);
        KSPGetPC(ksp_pi, &pc);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(ksp_pi, "ksp_pi_");
        KSPSetFromOptions(ksp_pi);
    //}
    VecScale(F_rt, -1.0);
    KSPSolve(ksp_pi, F_rt, d_rt);

    // back substitute for updates
    MatMult(G_pi_N_pi_inv_N_rt, d_rt, _tmpA1);
    VecAXPY(F_u, 1.0, _tmpA1);
    VecScale(F_u, -1.0);
    MatMult(M_u_inv, F_u, d_u);

    MatMult(N_rt, d_rt, _tmpB1);
    VecAXPY(F_pi, 1.0, _tmpB1);
    VecScale(F_pi, -1.0);
    MatMult(N_pi_inv, F_pi, d_pi);

    MatMult(D_rho, d_u, _tmpB1);
    VecAXPY(F_rho, 1.0, _tmpB1);
    VecScale(F_rho, -1.0);
    MatMult(M_rho_inv, F_rho, d_rho);

MatDestroy(&G_pi_N_pi_inv_N_rt);
MatDestroy(&Q_rt_rho_M_rho_inv_D_rho);
MatDestroy(&L_rt_rt);
KSPDestroy(&ksp_pi);
}

void VertSolve::solve_schur(
L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, 
L2Vecs* udwdx, double del2_x, Umat* M1, Wmat* M2, E21mat* EtoF, KSP ksp_x,
L2Vecs* F_rho_o, L2Vecs* F_rt_o)
{
    bool done = false;
    int ex, ey, elOrd2, itt = 0;
    double norm_x, max_norm_w, max_norm_exner, max_norm_rho, max_norm_rt;
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* velz_h = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_j = new L2Vecs(geom->nk+1, topo, geom);
    Vec F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, F_z, G_z, dF_z, dG_z;
    Vec h_tmp_1, h_tmp_2, u_tmp_1, u_tmp_2;

    elOrd2 = topo->elOrd*topo->elOrd;
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &d_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp_2);

    velz_i->HorizToVert();
    rho_i->HorizToVert();
    rt_i->HorizToVert();
    exner_i->HorizToVert();

    velz_j->CopyFromVert(velz_i->vz);
    rho_j->CopyFromVert(rho_i->vz);
    rt_j->CopyFromVert(rt_i->vz);
    exner_j->CopyFromVert(exner_i->vz);

    // diagnose the potential temperature
    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();
    theta_j->CopyFromVert(theta_i->vz);
    theta_j->VertToHoriz();

    exner_h->CopyFromHoriz(exner_i->vh);
    exner_h->HorizToVert();

    velz_h->CopyFromVert(velz_i->vz);
    rho_h->CopyFromVert(rho_i->vz);
    rt_h->CopyFromVert(rt_i->vz);

    F_rho_o->HorizToVert();
    F_rt_o->HorizToVert();

    do {
        k2i_z = 0.0;
        max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // assemble the residual vectors
            assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                              rt_i->vz[ii], rt_j->vz[ii], F_w, F_z, G_z);

            if(udwdx) VecAXPY(F_w, dt, udwdx->vz[ii]);
            //VecAXPY(F_w, 0.5*dt, d4w_i->vz[ii]);
            //VecAXPY(F_w, 0.5*dt, d4w_j->vz[ii]);
            //VecAXPY(F_w, dt, d4w_i->vz[ii]);
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);
            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii] );

            // add the horizontal forcing
            VecAXPY(dF_z, dt, F_rho_o->vz[ii]);
            VecAXPY(dG_z, dt, F_rt_o->vz[ii] );

            MatMult(vo->VB, dF_z, F_rho);
            MatMult(vo->VB, dG_z, F_rt);

            solve_schur_column_3(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii], 
                               F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, itt);

            VecAXPY(velz_j->vz[ii],  1.0, d_w);
            VecAXPY(rho_j->vz[ii],   1.0, d_rho);
            VecAXPY(rt_j->vz[ii],    1.0, d_rt);
            VecAXPY(exner_j->vz[ii], 1.0, d_exner);

            max_norm_exner = MaxNorm(d_exner, exner_j->vz[ii], max_norm_exner);
            max_norm_w     = MaxNorm(d_w,     velz_j->vz[ii],  max_norm_w    );
            max_norm_rho   = MaxNorm(d_rho,   rho_j->vz[ii],   max_norm_rho  );
            max_norm_rt    = MaxNorm(d_rt,    rt_j->vz[ii],    max_norm_rt   );

            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            VecZeroEntries(velz_h->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_i->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_j->vz[ii]);

            VecZeroEntries(rho_h->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_i->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_j->vz[ii]);

            VecZeroEntries(rt_h->vz[ii]);
            VecAXPY(rt_h->vz[ii], 0.5, rt_i->vz[ii]);
            VecAXPY(rt_h->vz[ii], 0.5, rt_j->vz[ii]);
        }

        diagTheta2(rho_j->vz, rt_j->vz, theta_j->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(theta_h->vz[ii]);
            VecAXPY(theta_h->vz[ii], 0.5, theta_j->vz[ii]);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
        }
        theta_h->VertToHoriz();
        theta_j->VertToHoriz();

        MPI_Allreduce(&max_norm_exner, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_exner = norm_x;
        MPI_Allreduce(&max_norm_w,     &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w     = norm_x;
        MPI_Allreduce(&max_norm_rho,   &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho   = norm_x;
        MPI_Allreduce(&max_norm_rt,    &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt    = norm_x;

        itt++;

        if(max_norm_exner < 1.0e-8 && max_norm_w < 1.0e-8 && max_norm_rho < 1.0e-8 && max_norm_rt < 1.0e-8) done = true;
        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                 "\t|d_w|/|w|: "          << max_norm_w     <<
                                 "\t|d_rho|/|rho|: "      << max_norm_rho   <<
                                 "\t|d_rt|/|rt|: "        << max_norm_rt    << endl;
    } while(!done);

    velz_i->CopyFromVert(velz_j->vz);
    rho_i->CopyFromVert(rho_j->vz);
    rt_i->CopyFromVert(rt_j->vz);
    exner_i->CopyFromVert(exner_j->vz);

    velz_i->VertToHoriz();
    rho_i->VertToHoriz();
    rt_i->VertToHoriz();
    exner_i->VertToHoriz();

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete theta_i;
    delete theta_j;
    delete velz_h;
    delete rho_h;
    delete rt_h;
    VecDestroy(&F_w);
    VecDestroy(&F_rho);
    VecDestroy(&F_rt);
    VecDestroy(&F_exner);
    VecDestroy(&d_w);
    VecDestroy(&d_rho);
    VecDestroy(&d_rt);
    VecDestroy(&d_exner);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);

    VecDestroy(&u_tmp_1);
    VecDestroy(&u_tmp_2);
    VecDestroy(&h_tmp_1);
    VecDestroy(&h_tmp_2);
}

// compute the contribution of the vorticity vector to the vertical momentum equation
void VertSolve::AssembleVertMomVort(Vec* ul, L2Vecs* velz, KSP ksp1, Umat* M1, Wmat* M2, E21mat* EtoF, WtQdUdz_mat* Rz, L2Vecs* uuz) {
    int kk;
    Vec _ul, ug, tmp, tmp1, dwdx;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &_ul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dwdx);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ug);

    for(kk = 0; kk < geom->nk-1; kk++) {
        VecZeroEntries(uuz->vh[kk]);
    }

    velz->VertToHoriz();
    for(kk = 0; kk < geom->nk-1; kk++) {
        // take the horizontal gradient of the vertical velocity
        // both matrices are piecewise linear in the vertical. as such they
        // should both be scaled by J_z = dz/2, however since the layer thicknesses
        // are constant for now we just omit these. These will need to be updated
        // for variable thickness layers.
        M1->assemble(kk, SCALE, false);
        M2->assemble(kk, SCALE, false);
        MatMult(M2->M, velz->vh[kk], tmp);
        MatMult(EtoF->E12, tmp, tmp1);
        KSPSolve(ksp1, tmp1, dwdx); // horizontal gradient of vertical velocity

        VecZeroEntries(_ul);
        VecAXPY(_ul, 0.5, ul[kk+0]);
        VecAXPY(_ul, 0.5, ul[kk+1]);
        Rz->assemble(_ul, SCALE);
        MatMult(Rz->M, dwdx, uuz->vh[kk]); // horizontal advection of vertical velocity
    }
    uuz->HorizToVert();

    VecDestroy(&_ul);
    VecDestroy(&ug);
    VecDestroy(&tmp);
    VecDestroy(&tmp1);
    VecDestroy(&dwdx);
}

// incorportate the horizontal divergence terms into the solve
void VertSolve::solve_schur_2(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, 
                              L2Vecs* udwdx, Vec* velx1, Vec* velx2, Vec* u1l, Vec* u2l, bool hs_forcing) {
    bool done = false;
    int ex, ey, elOrd2, itt = 0;
    double norm_x, max_norm_w, max_norm_exner, max_norm_rho, max_norm_rt;
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* velz_h = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_j = new L2Vecs(geom->nk+1, topo, geom);
    Vec F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, F_z, G_z, dF_z, dG_z, d_theta;
    L2Vecs* dFx = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* dGx = new L2Vecs(geom->nk, topo, geom);

    step++;

    elOrd2 = topo->elOrd*topo->elOrd;
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &d_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*elOrd2, &d_theta);

    velz_i->HorizToVert();
    rho_i->HorizToVert();
    rt_i->HorizToVert();
    exner_i->HorizToVert();

    velz_j->CopyFromVert(velz_i->vz);
    rho_j->CopyFromVert(rho_i->vz);
    rt_j->CopyFromVert(rt_i->vz);
    exner_j->CopyFromVert(exner_i->vz);

    // diagnose the potential temperature
    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();
    theta_j->CopyFromVert(theta_i->vz);
    theta_j->VertToHoriz();

    exner_h->CopyFromHoriz(exner_i->vh);
    exner_h->HorizToVert();

    velz_h->CopyFromVert(velz_i->vz);
    rho_h->CopyFromVert(rho_i->vz);
    rt_h->CopyFromVert(rt_i->vz);

    do {
        k2i_z = 0.0;
        max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        rho_j->VertToHoriz();
        horiz->advection_rhs(velx1, velx2, rho_i->vh, rho_j->vh, theta_h, dFx, dGx, u1l, u2l);

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // assemble the residual vectors
            assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                              rt_i->vz[ii], rt_j->vz[ii], F_w, F_z, G_z);

            if(udwdx) VecAXPY(F_w, dt, udwdx->vz[ii]);
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);
            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii] );

            // add the horizontal forcing
            VecAXPY(dF_z, dt, dFx->vz[ii]);
            VecAXPY(dG_z, dt, dGx->vz[ii]);

            MatMult(vo->VB, dF_z, F_rho);
            MatMult(vo->VB, dG_z, F_rt);

            if(hs_forcing) {
                vo->AssembleTempForcing_HS(ex, ey, exner_h->vz[ii], theta_h->vz[ii], rho_h->vz[ii], dG_z);
                VecAXPY(F_rt, dt, dG_z);
            }

            solve_schur_column_3(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii], 
                               F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, itt);

            VecAXPY(velz_j->vz[ii],  1.0, d_w);
            VecAXPY(rho_j->vz[ii],   1.0, d_rho);
            VecAXPY(rt_j->vz[ii],    1.0, d_rt);
            VecAXPY(exner_j->vz[ii], 1.0, d_exner);

            max_norm_exner = MaxNorm(d_exner, exner_j->vz[ii], max_norm_exner);
            max_norm_w     = MaxNorm(d_w,     velz_j->vz[ii],  max_norm_w    );
            max_norm_rho   = MaxNorm(d_rho,   rho_j->vz[ii],   max_norm_rho  );
            max_norm_rt    = MaxNorm(d_rt,    rt_j->vz[ii],    max_norm_rt   );

            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            VecZeroEntries(velz_h->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_i->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_j->vz[ii]);

            VecZeroEntries(rho_h->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_i->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_j->vz[ii]);

            VecZeroEntries(rt_h->vz[ii]);
            VecAXPY(rt_h->vz[ii], 0.5, rt_i->vz[ii]);
            VecAXPY(rt_h->vz[ii], 0.5, rt_j->vz[ii]);
        }

        diagTheta2(rho_j->vz, rt_j->vz, theta_j->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(theta_h->vz[ii]);
            VecAXPY(theta_h->vz[ii], 0.5, theta_j->vz[ii]);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
        }
        theta_h->VertToHoriz();
        theta_j->VertToHoriz();

        MPI_Allreduce(&max_norm_exner, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_exner = norm_x;
        MPI_Allreduce(&max_norm_w,     &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w     = norm_x;
        MPI_Allreduce(&max_norm_rho,   &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho   = norm_x;
        MPI_Allreduce(&max_norm_rt,    &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt    = norm_x;

        itt++;

        if(max_norm_exner < 1.0e-12 && max_norm_rho < 1.0e-12 && max_norm_rt < 1.0e-12) done = true;
        if(!rank) cout << "\t" << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                 "\t|d_w|/|w|: "          << max_norm_w     <<
                                 "\t|d_rho|/|rho|: "      << max_norm_rho   <<
                                 "\t|d_rt|/|rt|: "        << max_norm_rt    << endl;
    } while(!done);

    velz_i->CopyFromVert(velz_j->vz);
    rho_i->CopyFromVert(rho_j->vz);
    rt_i->CopyFromVert(rt_j->vz);
    exner_i->CopyFromVert(exner_j->vz);

    velz_i->VertToHoriz();
    rho_i->VertToHoriz();
    rt_i->VertToHoriz();
    exner_i->VertToHoriz();

    theta_h->VertToHoriz();
    exner_h->VertToHoriz();

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete theta_i;
    delete theta_j;
    delete velz_h;
    delete rho_h;
    delete rt_h;
    delete dFx;
    delete dGx;
    VecDestroy(&F_w);
    VecDestroy(&F_rho);
    VecDestroy(&F_rt);
    VecDestroy(&F_exner);
    VecDestroy(&d_w);
    VecDestroy(&d_rho);
    VecDestroy(&d_rt);
    VecDestroy(&d_exner);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
    VecDestroy(&d_theta);
}

void VertSolve::assemble_operators(int ex, int ey, Vec theta, Vec rho, Vec rt, Vec pi, Vec velz) {
    MatReuse reuse = (!G_rt) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    // assemble the operators for the coupled system
    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, reuse, &VAB);
    MatAssemblyBegin(VAB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (VAB, MAT_FINAL_ASSEMBLY);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(VAB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatAssemblyBegin(pc_V0_invV0_rt_DT, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_V0_invV0_rt_DT, MAT_FINAL_ASSEMBLY);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, 0.5*dt);
    MatAssemblyBegin(G_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (G_rt, MAT_FINAL_ASSEMBLY);

    //vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    MatAssemblyBegin(pc_DTV1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_DTV1, MAT_FINAL_ASSEMBLY);
    //vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    MatAssemblyBegin(pc_V0_invDTV1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_V0_invDTV1, MAT_FINAL_ASSEMBLY);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);
    MatAssemblyBegin(G_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (G_pi, MAT_FINAL_ASSEMBLY);

    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    //vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatAssemblyBegin(pc_V0_invV0_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_V0_invV0_rt, MAT_FINAL_ASSEMBLY);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatAssemblyBegin(pc_DV0_invV0_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_DV0_invV0_rt, MAT_FINAL_ASSEMBLY);
    //vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);
    MatAssemblyBegin(D_rho, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (D_rho, MAT_FINAL_ASSEMBLY);

    vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    MatMatMult(vo->VB, vo->V10, reuse, PETSC_DEFAULT, &D_rt);
    MatScale(D_rt, 0.5*dt);
    MatAssemblyBegin(D_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (D_rt, MAT_FINAL_ASSEMBLY);

    vo->AssembleConstWithTheta(ex, ey, theta, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    MatAssemblyBegin(pc_DTV1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_DTV1, MAT_FINAL_ASSEMBLY);
    //vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    MatAssemblyBegin(pc_V0_invDTV1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_V0_invDTV1, MAT_FINAL_ASSEMBLY);
vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    MatMatMult(vo->VBA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &Q_rt_rho);
    MatScale(Q_rt_rho, 0.5*dt);
    MatAssemblyBegin(Q_rt_rho, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (Q_rt_rho, MAT_FINAL_ASSEMBLY);

    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatAssemblyBegin(pc_VB_rt_invVB_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_VB_rt_invVB_pi, MAT_FINAL_ASSEMBLY);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
    MatAssemblyBegin(N_rt, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (N_rt, MAT_FINAL_ASSEMBLY);

    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, pi, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatAssemblyBegin(pc_VB_rt_invVB_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pc_VB_rt_invVB_pi, MAT_FINAL_ASSEMBLY);
    if(!N_pi)
        MatMatMult(vo->VB, pc_VB_rt_invVB_pi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &N_pi);
    else
        MatMatMult(vo->VB, pc_VB_rt_invVB_pi, MAT_REUSE_MATRIX, PETSC_DEFAULT, &N_pi);
    MatAssemblyBegin(N_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (N_pi, MAT_FINAL_ASSEMBLY);
}

void VertSolve::solve_schur_vert(L2Vecs* velz_i, L2Vecs* velz_j, L2Vecs* velz_h, L2Vecs* rho_i, L2Vecs* rho_j, L2Vecs* rho_h, 
                                 L2Vecs* rt_i, L2Vecs* rt_j, L2Vecs* rt_h, L2Vecs* exner_i, L2Vecs* exner_j, L2Vecs* _exner_h, 
                                 L2Vecs* theta_i, L2Vecs* _theta_h, L2Vecs* udwdx, Vec* velx1, Vec* velx2, Vec* u1l, Vec* u2l, 
                                 bool hs_forcing) {
    bool done = false;
    int ex, ey, elOrd2, itt = 0;
    double norm_x, max_norm_w, max_norm_exner, max_norm_rho, max_norm_rt;
    Vec F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, F_z, G_z, dF_z, dG_z;
    L2Vecs* dFx = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* dGx = new L2Vecs(geom->nk, topo, geom);
    //L2Vecs* velz_p  = new L2Vecs(geom->nk-1, topo, geom);
    //L2Vecs* rho_p   = new L2Vecs(geom->nk, topo, geom);
    //L2Vecs* rt_p    = new L2Vecs(geom->nk, topo, geom);
    //L2Vecs* exner_p = new L2Vecs(geom->nk, topo, geom);

    elOrd2 = topo->elOrd*topo->elOrd;
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &d_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);

    diagTheta2(rho_j->vz, rt_j->vz, _theta_h->vz);
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecScale(_theta_h->vz[ii], 0.5);
        VecAXPY(_theta_h->vz[ii], 0.5, theta_i->vz[ii]);
    }
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecZeroEntries(rho_h->vz[ii]);
	VecAXPY(rho_h->vz[ii], 0.5, rho_i->vz[ii]);
	VecAXPY(rho_h->vz[ii], 0.5, rho_j->vz[ii]);
        VecZeroEntries(rt_h->vz[ii]);
	VecAXPY(rt_h->vz[ii], 0.5, rt_i->vz[ii]);
	VecAXPY(rt_h->vz[ii], 0.5, rt_j->vz[ii]);
        VecZeroEntries(velz_h->vz[ii]);
	VecAXPY(velz_h->vz[ii], 0.5, velz_i->vz[ii]);
	VecAXPY(velz_h->vz[ii], 0.5, velz_j->vz[ii]);
        VecZeroEntries(_exner_h->vz[ii]);
	VecAXPY(_exner_h->vz[ii], 0.5, exner_i->vz[ii]);
	VecAXPY(_exner_h->vz[ii], 0.5, exner_j->vz[ii]);
    }
    rho_h->VertToHoriz();
    rt_h->VertToHoriz();
    velz_h->VertToHoriz();
    _exner_h->VertToHoriz();
    _theta_h->VertToHoriz();

    //velz_p->CopyFromVert(velz_j->vz);
    //rho_p->CopyFromVert(rho_j->vz);
    //rt_p->CopyFromVert(rt_j->vz);
    //exner_p->CopyFromVert(exner_j->vz);

    do {
        k2i_z = 0.0;
        max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        //if(!itt) {
            rho_j->VertToHoriz();
            horiz->advection_rhs(velx1, velx2, rho_i->vh, rho_j->vh, _theta_h, dFx, dGx, u1l, u2l);
        //}

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // assemble the residual vectors
            assemble_residual(ex, ey, _theta_h->vz[ii], _exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                              rt_i->vz[ii], rt_j->vz[ii], F_w, F_z, G_z);
            //assemble_residual(ex, ey, _theta_h->vz[ii], _exner_h->vz[ii], velz_i->vz[ii], velz_p->vz[ii], rho_i->vz[ii], rho_p->vz[ii], 
              //                rt_i->vz[ii], rt_p->vz[ii], F_w, F_z, G_z);

            if(udwdx) VecAXPY(F_w, dt, udwdx->vz[ii]);
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);
            //vo->Assemble_EOS_Residual(ex, ey, rt_p->vz[ii], exner_p->vz[ii], F_exner);
            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            //VecAYPX(dF_z, dt, rho_p->vz[ii]);
            //VecAYPX(dG_z, dt, rt_p->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii] );

            // add the horizontal forcing
            VecAXPY(dF_z, dt, dFx->vz[ii]);
            VecAXPY(dG_z, dt, dGx->vz[ii]);

            MatMult(vo->VB, dF_z, F_rho);
            MatMult(vo->VB, dG_z, F_rt);

            if(hs_forcing) {
                vo->AssembleTempForcing_HS(ex, ey, _exner_h->vz[ii], _theta_h->vz[ii], rho_h->vz[ii], dG_z);
                VecAXPY(F_rt, dt, dG_z);
            }
/*{
double norm[4];
VecNorm(F_rho,NORM_2,&norm[0]);
VecNorm(F_rt,NORM_2,&norm[1]);
VecNorm(F_exner,NORM_2,&norm[2]);
VecNorm(F_w,NORM_2,&norm[3]);
if(!rank && !ii)cout<<ii<<"\t|F_rho|: "<<norm[0]<<"\t"
                 <<"\t|F_rt|: "<<norm[1]<<"\t"
                 <<"\t|F_pi|: "<<norm[2]<<"\t"
                 <<"\t|F_w|: "<<norm[3]<<"\n";
}*/


            solve_schur_column_3(ex, ey, _theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], _exner_h->vz[ii], 
                               F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, itt);

            VecAXPY(velz_j->vz[ii],  1.0, d_w);
            VecAXPY(rho_j->vz[ii],   1.0, d_rho);
            VecAXPY(rt_j->vz[ii],    1.0, d_rt);
            VecAXPY(exner_j->vz[ii], 1.0, d_exner);
            //VecAXPY(velz_p->vz[ii],  1.0, d_w);
            //VecAXPY(rho_p->vz[ii],   1.0, d_rho);
            //VecAXPY(rt_p->vz[ii],    1.0, d_rt);
            //VecAXPY(exner_p->vz[ii], 1.0, d_exner);

            max_norm_exner = MaxNorm(d_exner, exner_j->vz[ii], max_norm_exner);
            max_norm_w     = MaxNorm(d_w,     velz_j->vz[ii],  max_norm_w    );
            max_norm_rho   = MaxNorm(d_rho,   rho_j->vz[ii],   max_norm_rho  );
            max_norm_rt    = MaxNorm(d_rt,    rt_j->vz[ii],    max_norm_rt   );
            //max_norm_exner = MaxNorm(d_exner, exner_p->vz[ii], max_norm_exner);
            //max_norm_w     = MaxNorm(d_w,     velz_p->vz[ii],  max_norm_w    );
            //max_norm_rho   = MaxNorm(d_rho,   rho_p->vz[ii],   max_norm_rho  );
            //max_norm_rt    = MaxNorm(d_rt,    rt_p->vz[ii],    max_norm_rt   );

            VecZeroEntries(_exner_h->vz[ii]);
            VecAXPY(_exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(_exner_h->vz[ii], 0.5, exner_j->vz[ii]);
            //VecAXPY(_exner_h->vz[ii], 0.5, exner_p->vz[ii]);

            VecZeroEntries(velz_h->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_i->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_j->vz[ii]);
            //VecAXPY(velz_h->vz[ii], 0.5, velz_p->vz[ii]);

            VecZeroEntries(rho_h->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_i->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_j->vz[ii]);
            //VecAXPY(rho_h->vz[ii], 0.5, rho_p->vz[ii]);

            VecZeroEntries(rt_h->vz[ii]);
            VecAXPY(rt_h->vz[ii], 0.5, rt_i->vz[ii]);
            VecAXPY(rt_h->vz[ii], 0.5, rt_j->vz[ii]);
            //VecAXPY(rt_h->vz[ii], 0.5, rt_p->vz[ii]);
        }

        diagTheta2(rho_j->vz, rt_j->vz, _theta_h->vz);
        //diagTheta2(rho_p->vz, rt_p->vz, _theta_h->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecScale(_theta_h->vz[ii], 0.5);
            VecAXPY(_theta_h->vz[ii], 0.5, theta_i->vz[ii]);
        }
        _theta_h->VertToHoriz();

        MPI_Allreduce(&max_norm_exner, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_exner = norm_x;
        MPI_Allreduce(&max_norm_w,     &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w     = norm_x;
        MPI_Allreduce(&max_norm_rho,   &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho   = norm_x;
        MPI_Allreduce(&max_norm_rt,    &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt    = norm_x;

        itt++;

        if(max_norm_exner < 1.0e-12 && max_norm_rho < 1.0e-12 && max_norm_rt < 1.0e-12) done = true;
        if(!rank /*&& done*/) cout << "\t" << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                 "\t|d_w|/|w|: "          << max_norm_w     <<
                                 "\t|d_rho|/|rho|: "      << max_norm_rho   <<
                                 "\t|d_rt|/|rt|: "        << max_norm_rt    << endl;
    } while(!done);
/*
    if(update) {
        velz_j->CopyFromVert(velz_p->vz);
        rho_j->CopyFromVert(rho_p->vz);
        rt_j->CopyFromVert(rt_p->vz);
        exner_j->CopyFromVert(exner_p->vz);
*/
//        velz_j->VertToHoriz();
//        rho_j->VertToHoriz();
//        rt_j->VertToHoriz();
//        exner_j->VertToHoriz();
//    }

    velz_h->VertToHoriz();
    rho_h->VertToHoriz();
    rt_h->VertToHoriz();
    _exner_h->VertToHoriz();

    delete dFx;
    delete dGx;
    //delete velz_p;
    //delete rho_p;
    //delete rt_p;
    //delete exner_p;
    VecDestroy(&F_w);
    VecDestroy(&F_rho);
    VecDestroy(&F_rt);
    VecDestroy(&F_exner);
    VecDestroy(&d_w);
    VecDestroy(&d_rho);
    VecDestroy(&d_rt);
    VecDestroy(&d_exner);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
}
