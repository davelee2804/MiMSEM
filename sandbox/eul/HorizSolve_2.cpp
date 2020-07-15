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
#include "Boundary.h"
#include "HorizSolve_2.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
//#define RAYLEIGH 0.2

using namespace std;

HorizSolve::HorizSolve(Topo* _topo, Geom* _geom, double _dt) {
    int ii, elOrd2;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    do_visc = true;
    del2 = viscosity();
    step = 0;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    vo = new VertOps(topo, geom);

    // 0 form lumped mass matrix (vector)
    m0 = new Pvec(topo, geom, node);
    mh0 = new Phvec(topo, geom, node);

    // 1 form mass matrix
    M1 = new Umat(topo, geom, node, edge);

    // 2 form mass matrix
    M2 = new Wmat(topo, geom, edge);

    // incidence matrices
    NtoE = new E10mat(topo);
    EtoF = new E21mat(topo);

    // rotational operator
    R = new RotMat(topo, geom, node, edge);

    // mass flux operator
    F = new Uhmat(topo, geom, node, edge);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    // additional vorticity operator
    M1t = new Ut_mat(topo, geom, node, edge);
    Rh = new UtQWmat(topo, geom, node, edge);

    // potential temperature projection operator
    T = new Whmat(topo, geom, edge);

    // equation or state right hand side vector
    eos = new EoSvec(topo, geom, edge);

    // derivative of the equation of state (for the Theta preconditioner operator)
    eos_mat = new EoSmat(topo, geom, edge);

    // additional operators for the preconditioner
    M2inv = new WmatInv(topo, geom, edge);
    M2_rho_inv = new WhmatInv(topo, geom, edge);
    M2_pi_inv = new N_rt_Inv(topo, geom, edge);
    M2_rt_inv = new N_rt_Inv(topo, geom, edge);

    APV = new PtQUt_mat(topo, geom, node, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

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

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp1);
    KSPSetOperators(ksp1, M1->M, M1->M);
    KSPSetTolerances(ksp1, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1, KSPGMRES);
    KSPGetPC(ksp1, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp1, "ksp1_");
    KSPSetFromOptions(ksp1);

    // initialize the 2 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPGetPC(ksp2, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, elOrd2, NULL);
    KSPSetOptionsPrefix(ksp2, "ksp2_");
    KSPSetFromOptions(ksp2);

    _PCx = NULL;

    MatCreate(MPI_COMM_WORLD, &pcx_M1_inv);
    MatSetSizes(pcx_M1_inv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(pcx_M1_inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(pcx_M1_inv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatZeroEntries(pcx_M1_inv);

    MatCreate(MPI_COMM_WORLD, &pcx_M0_inv);
    MatSetSizes(pcx_M0_inv, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(pcx_M0_inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(pcx_M0_inv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatZeroEntries(pcx_M0_inv);

    bndry = new Boundary(topo, geom, node, edge);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double HorizSolve::viscosity() {
    double ae = 4.0*M_PI*RAD_EARTH*RAD_EARTH;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void HorizSolve::coriolis() {
    int ii, kk;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fxl, fxg, PtQfxg;

    // initialise the coriolis vector (local and global)
    fg = new Vec[geom->nk];
    fl = new Vec[geom->nk];

    // evaluate the coriolis term at nodes
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < topo->n0; ii++) {
        fArray[ii] = 2.0*OMEGA*sin(geom->s[ii][1]);
    }
    VecRestoreArray(fxl, &fArray);

    // scatter array to global vector
    VecScatterBegin(topo->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);

    // project vector onto 0 forms
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PtQfxg);
    VecZeroEntries(PtQfxg);
    MatMult(PtQ->M, fxg, PtQfxg);
    // diagonal mass matrix as vector
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg[kk]);
        m0->assemble(kk, 1.0);
        VecPointwiseDivide(fg[kk], PtQfxg, m0->vg);
        VecZeroEntries(fl[kk]);
        VecScatterBegin(topo->gtol_0, fg[kk], fl[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_0, fg[kk], fl[kk], INSERT_VALUES, SCATTER_FORWARD);
    }
    
    delete PtQ;
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

void HorizSolve::initGZ() {
    int ex, ey, ei, ii, kk, n2, mp12;
    int* inds0;
    int inds2k[99], inds0k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double* WtQflat = new double[W->nDofsJ*Q->nDofsJ];
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
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
            inds0 = topo->elInds0_l(ex, ey);
            Q->assemble(ex, ey);

            MatZeroEntries(BQ);
            for(kk = 0; kk < geom->nk; kk++) {
                for(ii = 0; ii < mp12; ii++) {
                    Q0[ii][ii]  = Q->A[ii][ii]*SCALE;
                    // for linear field we multiply by the vertical jacobian determinant when
                    // integrating, and do no other trasformations for the basis functions
                    Q0[ii][ii] *= 0.5;
                }
                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
                Flat2D_IP(W->nDofsJ, Q->nDofsJ, WtQ, WtQflat);

                for(ii = 0; ii < W->nDofsJ; ii++) {
                    inds2k[ii] = ii + kk*W->nDofsJ;
                }

                // assemble the first basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+0)*mp12;
                }
                MatSetValues(BQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQflat, ADD_VALUES);
                // assemble the second basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+1)*mp12;
                }
                MatSetValues(BQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQflat, ADD_VALUES);
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
    delete[] WtQflat;
    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    delete W;
    delete Q;
}

HorizSolve::~HorizSolve() {
    int ii;

    MatDestroy(&pcx_M0_inv);
    MatDestroy(&pcx_M1_inv);
    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);

    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&fg[ii]);
        VecDestroy(&fl[ii]);
    }
    delete[] fg;
    delete[] fl;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
        VecDestroy(&zv[ii]);
    }
    delete[] gv;
    delete[] zv;

    if(_PCx) {
        MatDestroy(&_PCx);
        KSPDestroy(&ksp_exner);
    }

    delete m0;
    delete mh0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;
    delete Rh;
    delete T;
    delete eos;
    delete eos_mat;

    delete M2inv;
    delete M2_pi_inv;
    delete M2_rt_inv;
    delete M2_rho_inv;
    delete APV;
    delete bndry;

    delete edge;
    delete node;
    delete quad;

    delete vo;
}

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void HorizSolve::grad(bool assemble, Vec phi, Vec* u, int lev, Vec bndry_vel) {
    Vec Mphi, dMphi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, u);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mphi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMphi);

    if(assemble) {
        M1->assemble(lev, SCALE, true);
        M2->assemble(lev, SCALE, true);
    }

    MatMult(M2->M, phi, Mphi);
    MatMult(EtoF->E12, Mphi, dMphi);
    if(bndry_vel) {
        bndry->AssembleGrad(lev, bndry_vel, phi, false);
        VecAXPY(dMphi, 1.0, bndry->bg);
    }
    KSPSolve(ksp1, dMphi, *u);

    VecDestroy(&Mphi);
    VecDestroy(&dMphi);
}

/*
Take the weak form curl of a 1 form vector field as a 1 form vector field
*/
void HorizSolve::curl(bool assemble, Vec u, Vec* w, int lev, bool add_f) {
    Vec Mu, dMu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &dMu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu);

    if(assemble) {
        m0->assemble(lev, SCALE);
        M1->assemble(lev, SCALE, true);
    }
    MatMult(M1->M, u, Mu);
    MatMult(NtoE->E01, Mu, dMu);
    VecPointwiseDivide(*w, dMu, m0->vg);

    // add the coliolis term
    if(add_f) {
        VecAYPX(*w, 1.0, fg[lev]);
    }
    VecDestroy(&Mu);
    VecDestroy(&dMu);
}

void HorizSolve::laplacian(bool assemble, Vec ui, Vec* ddu, int lev) {
    Vec Du, Cu, RCu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &RCu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Du);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, ui, Du);

    // grad (weak form)
    grad(assemble, Du, ddu, lev, NULL);

    /*** rotational component ***/
    // curl (weak form)
    curl(assemble, ui, &Cu, lev, false);

    // rot (strong form)
    MatMult(NtoE->E10, Cu, RCu);

    // add rotational and divergent components
    VecAXPY(*ddu, +1.0, RCu);
    VecScale(*ddu, del2);

    VecDestroy(&Cu);
    VecDestroy(&RCu);
    VecDestroy(&Du);
}

void HorizSolve::diagnose_F(int level, Vec u1, Vec u2, Vec h1, Vec h2, Vec _F) {
    Vec hu, b, h1l, h2l;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &h1l);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &h2l);

    VecScatterBegin(topo->gtol_2, h1, h1l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h1, h1l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, h2, h2l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h2, h2l, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);
    VecZeroEntries(_F);
    VecZeroEntries(hu);

    // assemble the nonlinear rhs mass matrix (note that hl is a local vector)
    F->assemble(h1l, level, true, SCALE);

    MatMult(F->M, u1, b);
    VecAXPY(hu, 1.0/3.0, b);

    MatMult(F->M, u2, b);
    VecAXPY(hu, 1.0/6.0, b);

    F->assemble(h2l, level, true, SCALE);

    MatMult(F->M, u1, b);
    VecAXPY(hu, 1.0/6.0, b);

    MatMult(F->M, u2, b);
    VecAXPY(hu, 1.0/3.0, b);

    // solve the linear system
    M1->assemble(level, SCALE, true);
    KSPSolve(ksp1, hu, _F);

    VecDestroy(&hu);
    VecDestroy(&b);
    VecDestroy(&h1l);
    VecDestroy(&h2l);
}

void HorizSolve::diagnose_Phi(int level, Vec u1, Vec u2, Vec u1l, Vec u2l, Vec* Phi) {
    Vec b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Phi);
    VecZeroEntries(*Phi);

    // u^2 terms (0.5 factor incorportated into the matrix assembly)
    K->assemble(u1l, level, SCALE);

    MatMult(K->M, u1, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    MatMult(K->M, u2, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    K->assemble(u2l, level, SCALE);

    MatMult(K->M, u2, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    VecDestroy(&b);
}

void HorizSolve::diagnose_wxu(int level, Vec u1, Vec u2, Vec* wxu) {
    Vec w1, w2, wl, uh;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, wxu);

    // assume the vertex and volume mass matrices have already been assembled
    // TODO compute once for half step
    curl(false, u1, &w1, level, true);
    curl(false, u2, &w2, level, true);
    VecAXPY(w1, 1.0, w2);
    VecScale(w1, 0.5);

    VecScatterBegin(topo->gtol_0, w1, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, w1, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(uh);
    VecAXPY(uh, 0.5, u1);
    VecAXPY(uh, 0.5, u2);

    R->assemble(wl, level, SCALE);
    MatMult(R->M, uh, *wxu);

    VecDestroy(&w1);
    VecDestroy(&w2);
    VecDestroy(&wl);
    VecDestroy(&uh);
}

/* All vectors, rho, rt and theta are VERTICAL vectors */
void HorizSolve::diagTheta2(Vec* rho, Vec* rt, Vec* theta) {
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

// compute the vorticity components dudz, dvdz
void HorizSolve::diagHorizVort(Vec* velx, Vec* dudz) {
    int ii;
    Vec* Mu = new Vec[geom->nk];
    Vec  du;
    PC pc;
    KSP ksp1_t;

    KSPCreate(MPI_COMM_WORLD, &ksp1_t);
    KSPSetOperators(ksp1_t, M1t->M, M1t->M);
    KSPSetTolerances(ksp1_t, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1_t, KSPGMRES);
    KSPGetPC(ksp1_t, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp1_t, "ksp1_t_");
    KSPSetFromOptions(ksp1_t);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu[ii]);
        M1->assemble(ii, SCALE, true);
        MatMult(M1->M, velx[ii], Mu[ii]);
    }

    for(ii = 0; ii < geom->nk-1; ii++) {
        VecZeroEntries(du);
        VecAXPY(du, +1.0, Mu[ii+1]);
        VecAXPY(du, -1.0, Mu[ii+0]);
        M1t->assemble(ii, SCALE);
        KSPSolve(ksp1_t, du, dudz[ii]);
    }

    VecDestroy(&du);
    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&Mu[ii]);
    }
    delete[] Mu;
    KSPDestroy(&ksp1_t);
}

void HorizSolve::assemble_residual(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                                Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec fu, Vec _F, Vec _G, Vec uil, Vec ujl, Vec uhl)
{
    Vec Phi, dPi, wxu, wxz, utmp, d2u, d4u;
    Vec theta_h, dp, dudz_h, velz_h, dudz_l;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &wxz);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &velz_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dudz_l);

    m0->assemble(level, SCALE);
    M1->assemble(level, SCALE, true);
    M2->assemble(level, SCALE, true);

    // assume theta is 0.5*(theta_i + theta_j)
    VecZeroEntries(theta_h);
    VecAXPY(theta_h, 0.5, theta[level+0]);
    VecAXPY(theta_h, 0.5, theta[level+1]);

    VecZeroEntries(fu);

    // assemble in the skew-symmetric parts of the vector
    diagnose_F(level, velx1, velx2, rho1, rho2, _F);
    diagnose_Phi(level, velx1, velx2, uil, ujl, &Phi);
    grad(false, Pi, &dPi, level, NULL/*velx2 do boundary!*/);
    diagnose_wxu(level, velx1, velx2, &wxu);

    MatMult(EtoF->E12, Phi, fu);
    VecAXPY(fu, 1.0, wxu);

    // add the pressure gradient force
    F->assemble_up(theta_h, level, SCALE, uhl);
    MatMult(F->M, dPi, dp);
    VecAXPY(fu, 1.0, dp);

    // diagnose the temperature flux (assume the H(div) mass matrix has
    // already been assembled at this level
    MatMult(F->M, _F, utmp);
    KSPSolve(ksp1, utmp, _G);

    // second voritcity term
    VecZeroEntries(utmp);
    if(level > 0) {
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, dudz1[level-1]);
        VecAXPY(dudz_h, 0.5, dudz2[level-1]);
        VecScatterBegin(topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);

        VecZeroEntries(velz_h);
        VecAXPY(velz_h, 0.5, velz1[level-1]);
        VecAXPY(velz_h, 0.5, velz2[level-1]);

        Rh->assemble(dudz_l, SCALE);
        MatMult(Rh->M, velz_h, dp);
        VecAXPY(utmp, 0.5, dp);
    }
    if(level < geom->nk-1) {
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, dudz1[level+0]);
        VecAXPY(dudz_h, 0.5, dudz2[level+0]);
        VecScatterBegin(topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);

        VecZeroEntries(velz_h);
        VecAXPY(velz_h, 0.5, velz1[level+0]);
        VecAXPY(velz_h, 0.5, velz2[level+0]);

        Rh->assemble(dudz_l, SCALE);
        MatMult(Rh->M, velz_h, dp);
        VecAXPY(utmp, 0.5, dp);
    }
    VecAXPY(fu, 1.0, utmp);
    VecScale(fu, dt);

    // assemble the mass matrix terms
    MatMult(M1->M, velx2, utmp);
    VecAXPY(fu, +1.0, utmp);
#ifdef RAYLEIGH
    if(level == geom->nk-1) VecAXPY(fu, +0.5*dt*RAYLEIGH, utmp);
#endif
    MatMult(M1->M, velx1, utmp);
    VecAXPY(fu, -1.0, utmp);
#ifdef RAYLEIGH
    if(level == geom->nk-1) VecAXPY(fu, +0.5*dt*RAYLEIGH, utmp);
#endif

    if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, velx1);
        VecAXPY(utmp, 0.5, velx2);
        laplacian(false, utmp, &d2u, level);
        laplacian(false, d2u, &d4u, level);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    // clean up
    VecDestroy(&utmp);
    VecDestroy(&Phi);
    VecDestroy(&dPi);
    VecDestroy(&wxu);
    VecDestroy(&theta_h);
    VecDestroy(&dp);
    VecDestroy(&wxz);
    VecDestroy(&dudz_h);
    VecDestroy(&velz_h);
    VecDestroy(&dudz_l);
}

void HorizSolve::coriolisMatInv(Mat A, Mat* Ainv) {
    int mi, mf, ci, nCols1, nCols2;
    const int *cols1, *cols2;
    const double *vals1;
    const double *vals2;
    double D[2][2], Dinv[2][2], detInv;
    double valsInv[4];
    int rows[2];

    D[0][0] = D[0][1] = D[1][0] = D[1][1] = 0.0;

    MatCreate(MPI_COMM_WORLD, Ainv);
    MatSetSizes(*Ainv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(*Ainv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(*Ainv, 2, PETSC_NULL, 2, PETSC_NULL);
    MatZeroEntries(*Ainv);

    MatGetOwnershipRange(A, &mi, &mf);
    for(int mm = mi; mm < mf; mm += 2) {
        rows[0] = mm+0;
        rows[1] = mm+1;

        MatGetRow(A, mm+0, &nCols1, &cols1, &vals1);
        for(ci = 0; ci < nCols1; ci++) {
            if(cols1[ci] == mm+0) {
                D[0][0] = vals1[ci+0];
                D[0][1] = vals1[ci+1];
                break;
            }
        }
        MatRestoreRow(A, mm+0, &nCols1, &cols1, &vals1);

        MatGetRow(A, mm+1, &nCols2, &cols2, &vals2);
        for(ci = 0; ci < nCols2; ci++) {
            if(cols2[ci] == mm+1) {
                D[1][0] = vals2[ci-1];
                D[1][1] = vals2[ci+0];
                break;
            }
        }
        MatRestoreRow(A, mm+1, &nCols2, &cols2, &vals2);

        detInv = 1.0/(D[0][0]*D[1][1] - D[0][1]*D[1][0]);

        Dinv[0][0] = +detInv*D[1][1];
        Dinv[1][1] = +detInv*D[0][0];
        Dinv[0][1] = -detInv*D[0][1];
        Dinv[1][0] = -detInv*D[1][0];

        valsInv[0] = Dinv[0][0];
        valsInv[1] = Dinv[0][1];
        valsInv[2] = Dinv[1][0];
        valsInv[3] = Dinv[1][1];

        MatSetValues(*Ainv, 2, rows, 2, rows, valsInv, INSERT_VALUES);
    }
    MatAssemblyBegin(*Ainv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  *Ainv, MAT_FINAL_ASSEMBLY);
}

void HorizSolve::assemble_biharmonic(int lev, MatReuse reuse, Mat* BVISC) {
    Vec m0_inv, ones_0;
    Mat LAP_1, VISC, VISC2;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &m0_inv);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &ones_0);

    VecSet(ones_0, 1.0);
    VecPointwiseDivide(m0_inv, ones_0, m0->vg);
    MatDiagonalSet(pcx_M0_inv, m0_inv, INSERT_VALUES);

    MatMatMult(pcx_M1invD12, pcx_M2D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &LAP_1);

    MatMatMult(NtoE->E01, M1->M, reuse, PETSC_DEFAULT, &pcx_CTM1);
    MatMatMult(pcx_M0_inv, pcx_CTM1, reuse, PETSC_DEFAULT, &pcx_M0_invCTM1);
    MatMatMult(NtoE->E10, pcx_M0_invCTM1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC);

    MatAssemblyBegin(LAP_1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  LAP_1, MAT_FINAL_ASSEMBLY);
    MatAXPY(VISC, 1.0, LAP_1, DIFFERENT_NONZERO_PATTERN);
    MatScale(VISC, del2);
    MatMatMult(VISC, VISC, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC2);
    MatMatMult(M1->M, VISC2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, BVISC);  
    MatScale(*BVISC, 0.5*dt);

    if(lev == geom->nk-1) MatScale(*BVISC, 4.0);

    VecDestroy(&m0_inv);
    VecDestroy(&ones_0);
    MatDestroy(&LAP_1);
    MatDestroy(&VISC);
    MatDestroy(&VISC2);
}

void HorizSolve::assemble_biharmonic_temp(int lev, Vec rho, MatReuse reuse, Mat* BVISC) {
    M2_rho_inv->assemble(rho, lev, SCALE);
    MatMatMult(M2_rho_inv->M, M2->M, reuse, PETSC_DEFAULT, &pcx_M2_invM2);

    MatMatMult(pcx_M1invD12M2, pcx_M2_invM2, reuse, PETSC_DEFAULT, &pcx_M1_invDT_M2M2_invM2);

    F->assemble(rho, lev, true, SCALE);
    MatMatMult(F->M, pcx_M1_invDT_M2M2_invM2, reuse, PETSC_DEFAULT, &pcx_M1_rhoM1_invDT_M2M2_invM2);

    MatMatMult(pcx_M2DM1_inv, pcx_M1_rhoM1_invDT_M2M2_invM2, reuse, PETSC_DEFAULT, &pcx_M2_LAP_Theta);

    MatMatMult(EtoF->E12, pcx_M2_LAP_Theta, reuse, PETSC_DEFAULT, &pcx_DT_LAP_Theta);
    MatMatMult(pcx_M2DM1_inv, pcx_DT_LAP_Theta, MAT_INITIAL_MATRIX, PETSC_DEFAULT, BVISC);

    MatAYPX(*BVISC, 0.5*dt*del2*del2, M2->M, DIFFERENT_NONZERO_PATTERN);

    if(lev == geom->nk-1) MatScale(*BVISC, 4.0);
}

void HorizSolve::assemble_rho_correction(int lev, Vec rho, Vec exner, Vec theta_k, MatReuse reuse, Vec diag_g, Vec ones_g, Mat* Au, Vec uhl) {
    MatGetDiagonal(F->M, diag_g);
    VecPointwiseDivide(diag_g, ones_g, diag_g);
    MatDiagonalSet(pcx_M1_inv, diag_g, INSERT_VALUES);

    F->assemble_up(theta_k, lev, SCALE, uhl);
    MatMatMult(F->M, pcx_M1_inv, reuse, PETSC_DEFAULT, &pcx_M1_exner_M1_inv);
    MatMatMult(pcx_M1_exner_M1_inv, EtoF->E12, reuse, PETSC_DEFAULT, &pcx_DTM2_exnerM2_rho_invM2);
    T->assemble(exner, lev, SCALE, true);
    MatMatMult(pcx_DTM2_exnerM2_rho_invM2, T->M, reuse, PETSC_DEFAULT, &pcx_Au_2);
    MatScale(pcx_Au_2, 0.5*dt*RD/CV);

    F->assemble(exner, lev, true, SCALE);
    MatMatMult(pcx_M1_exner_M1_inv, F->M, reuse, PETSC_DEFAULT, &pcx_M1_thetaM1_rho_invM1);
    MatMatMult(pcx_M1_thetaM1_rho_invM1, pcx_M1_inv, reuse, PETSC_DEFAULT, &pcx_M1_thetaM1_rho_invM1M1_rho_inv);
    MatMatMult(pcx_M1_thetaM1_rho_invM1M1_rho_inv, EtoF->E12, reuse, PETSC_DEFAULT, &pcx_M1_thetaM1_rho_invM1M1_rho_inv_DT);
    T->assemble(rho, lev, SCALE, true);
    MatMatMult(pcx_M1_thetaM1_rho_invM1M1_rho_inv_DT, T->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, Au);
    MatAYPX(*Au, -0.5*dt*RD/CV, pcx_Au_2, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(*Au, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  *Au, MAT_FINAL_ASSEMBLY);
}

void HorizSolve::assemble_and_update(int lev, Vec* theta, Vec velx, Vec rho, Vec rt, Vec exner, 
                                     Vec F_u, Vec F_rho, Vec F_rt, Vec F_exner, bool eos_update, bool neg_scale, L2Vecs* velz_i, L2Vecs* velz_j, Vec uhl)
{
    MatReuse reuse = (!_PCx) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    bool set_ksp = (!_PCx) ? true : false;
    Vec wg, wl, theta_k, diag_g, ones_g, h_tmp, velz_h_l;
    Mat Mu_inv, Mu_prime, M1_OP;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &velz_h_l);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &diag_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ones_g);

    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE, true);
    M2->assemble(lev, SCALE, true);
    M2inv->assemble(lev, SCALE);

    MatGetDiagonal(M1->M, diag_g);
    VecSet(ones_g, 1.0);
    VecPointwiseDivide(diag_g, ones_g, diag_g);
    MatDiagonalSet(pcx_M1_inv, diag_g, INSERT_VALUES);

    // [u,u] block
    curl(false, velx, &wg, lev, true);
    VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    R->assemble(wl, lev, SCALE);
    MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);
#ifdef RAYLEIGH
    if(lev == geom->nk-1) MatAXPY(R->M, 0.5*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
#endif
    VecDestroy(&wg);
    if(velz_i && velz_j) {
        VecZeroEntries(velz_h_l);
        if(lev > 0) {
            VecAXPY(velz_h_l, 1.0/(geom->thick[lev-1][0]+geom->thick[lev+0][0]), velz_i->vl[lev-1]);
            VecAXPY(velz_h_l, 1.0/(geom->thick[lev-1][0]+geom->thick[lev+0][0]), velz_j->vl[lev-1]);
        }
        if(lev < geom->nk-1) {
            VecAXPY(velz_h_l, 1.0/(geom->thick[lev+0][0]+geom->thick[lev+1][0]), velz_i->vl[lev+0]);
            VecAXPY(velz_h_l, 1.0/(geom->thick[lev+0][0]+geom->thick[lev+1][0]), velz_j->vl[lev+0]);
        }
        F->assemble(velz_h_l, lev, false, SCALE);
        MatAXPY(R->M, 0.5*dt, F->M, DIFFERENT_NONZERO_PATTERN);
    }

    // [u,exner] block
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta[lev+0]);
    VecAXPY(theta_k, 0.5, theta[lev+1]);
    F->assemble_up(theta_k, lev, SCALE, uhl);
    MatMatMult(pcx_M1_inv, EtoF->E12, reuse, PETSC_DEFAULT, &pcx_M1invD12);
    MatMatMult(pcx_M1invD12, M2->M, reuse, PETSC_DEFAULT, &pcx_M1invD12M2);
    MatMatMult(F->M, pcx_M1invD12M2, reuse, PETSC_DEFAULT, &pcx_G);
    MatScale(pcx_G, 0.5*dt);

    MatMatMult(M2->M, EtoF->E21, reuse, PETSC_DEFAULT, &pcx_M2D);
    MatMatMult(pcx_M2D, pcx_M1_inv, reuse, PETSC_DEFAULT, &pcx_M2DM1_inv);
    // [rt,u] block
    F->assemble(rt, lev, true, SCALE);
    MatMatMult(pcx_M2DM1_inv, F->M, reuse, PETSC_DEFAULT, &pcx_D);
    MatScale(pcx_D, 0.5*dt);
    // [rho,u] block
    F->assemble(rho, lev, true, SCALE);
    MatMatMult(pcx_M2DM1_inv, F->M, reuse, PETSC_DEFAULT, &pcx_D_rho);
    MatScale(pcx_D_rho, 0.5*dt);

    // biharmonic viscosity operator
    if(do_visc) assemble_biharmonic(lev, reuse, &M1_OP);

    // density corrections
    assemble_rho_correction(lev, rho, exner, theta_k, reuse, diag_g, ones_g, &pcx_Au, uhl);

    MatMatMult(pcx_Au, M2inv->M, reuse, PETSC_DEFAULT, &pcx_Au_M2_inv);
    MatMatMult(pcx_Au_M2_inv, pcx_D_rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mu_prime); // invalid read on the second pass
    MatAYPX(Mu_prime, 1.0, R->M, DIFFERENT_NONZERO_PATTERN);
    if(do_visc) { 
        MatAXPY(M1_OP, 1.0, Mu_prime, DIFFERENT_NONZERO_PATTERN);
        coriolisMatInv(M1_OP, &Mu_inv);
    } else {
        coriolisMatInv(Mu_prime, &Mu_inv);
    }
    MatMult(pcx_Au_M2_inv, F_rho, diag_g);
    VecAXPY(F_u, -1.0, diag_g);

    MatDestroy(&pcx_Au);

    // build the preconditioner
    MatMatMult(pcx_D, Mu_inv, reuse, PETSC_DEFAULT, &pcx_D_Mu_inv);
    MatMatMult(pcx_D_Mu_inv, pcx_G, reuse, PETSC_DEFAULT, &pcx_LAP);

    M2_rt_inv->assemble(rt, lev, SCALE, true);
    MatScale(M2_rt_inv->M, -1.0*CV/RD);

    if(do_visc) {   // temperature viscosity (biharmonic)
        assemble_biharmonic_temp(lev, rho, reuse, &pcx_LAP2_Theta);
        MatMatMult(pcx_LAP2_Theta, M2_rt_inv->M, reuse, PETSC_DEFAULT, &pcx_M2N_rt_inv);
    } else {
        MatMatMult(M2->M, M2_rt_inv->M, reuse, PETSC_DEFAULT, &pcx_M2N_rt_inv);
    }
    M2_pi_inv->assemble(exner, lev, SCALE, false);
    MatMatMult(pcx_M2N_rt_inv, M2_pi_inv->M, reuse, PETSC_DEFAULT, &pcx_M2N_rt_invN_pi);

    MatAssemblyBegin(pcx_M2N_rt_invN_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pcx_M2N_rt_invN_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(pcx_LAP, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pcx_LAP, MAT_FINAL_ASSEMBLY);

    if(!_PCx) MatMatMult(pcx_D, pcx_G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &_PCx);
    MatZeroEntries(_PCx);
    MatAXPY(_PCx, -1.0, pcx_LAP, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(_PCx, -1.0, pcx_M2N_rt_invN_pi, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(_PCx, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (_PCx, MAT_FINAL_ASSEMBLY);

    // update the rhs
    if(eos_update) {
        MatMult(pcx_M2N_rt_inv, F_exner, h_tmp);
        VecAXPY(F_rt, -1.0, h_tmp);
    }
    MatMult(pcx_D_Mu_inv, F_u, h_tmp);
    VecAXPY(F_rt, -1.0, h_tmp);
    if(neg_scale) {
        VecScale(F_rt, -1.0);
    }

    if(set_ksp) {
        int size;
        PC pc;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        KSPCreate(MPI_COMM_WORLD, &ksp_exner);
        KSPSetOperators(ksp_exner, _PCx, _PCx);
        KSPSetTolerances(ksp_exner, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
        KSPSetType(ksp_exner, KSPGMRES);
        KSPGetPC(ksp_exner, &pc);
        PCSetType(pc, PCBJACOBI);
        PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
        KSPSetOptionsPrefix(ksp_exner, "ksp_exner_x_");
        KSPSetFromOptions(ksp_exner);
    }

    VecDestroy(&wl);
    VecDestroy(&theta_k);
    VecDestroy(&h_tmp);
    VecDestroy(&diag_g);
    VecDestroy(&ones_g);
    VecDestroy(&velz_h_l);
    MatDestroy(&Mu_inv);
    MatDestroy(&Mu_prime);
    if(do_visc) MatDestroy(&M1_OP);
    if(do_visc) MatDestroy(&pcx_LAP2_Theta);
}

void HorizSolve::set_deltas(int lev, Vec* theta, Vec velx, Vec rho, Vec rt, Vec exner, 
                            Vec F_u, Vec F_rho, Vec F_exner, Vec du, Vec drho, Vec drt, Vec dexner, 
                            bool do_rt, bool neg_scale, L2Vecs* velz_i, L2Vecs* velz_j, Vec uhl) {
    int size;
    MatReuse reuse = MAT_REUSE_MATRIX;
    Vec wg, wl, theta_k, diag_g, ones_g, h_tmp, velz_h;
    Mat Mu_prime, M1_OP;
    PC pc;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &velz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &diag_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ones_g);

    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE, true);
    M2->assemble(lev, SCALE, true);
    M2inv->assemble(lev, SCALE);

    MatGetDiagonal(M1->M, diag_g);
    VecSet(ones_g, 1.0);
    VecPointwiseDivide(diag_g, ones_g, diag_g);
    MatDiagonalSet(pcx_M1_inv, diag_g, INSERT_VALUES);

    // [u,u] block
    curl(false, velx, &wg, lev, true);
    VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    R->assemble(wl, lev, SCALE);
    MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);
#ifdef RAYLEIGH
    if(lev == geom->nk-1) MatAXPY(R->M, 0.5*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
#endif
    if(velz_i && velz_j) {
        VecZeroEntries(velz_h);
        if(lev > 0) {
            VecAXPY(velz_h, 1.0/(geom->thick[lev-1][0]+geom->thick[lev+0][0]), velz_i->vl[lev-1]);
            VecAXPY(velz_h, 1.0/(geom->thick[lev-1][0]+geom->thick[lev+0][0]), velz_j->vl[lev-1]);
        }
        if(lev < geom->nk-1) {
            VecAXPY(velz_h, 1.0/(geom->thick[lev+0][0]+geom->thick[lev+1][0]), velz_i->vl[lev+0]);
            VecAXPY(velz_h, 1.0/(geom->thick[lev+0][0]+geom->thick[lev+1][0]), velz_j->vl[lev+0]);
        }
        F->assemble(velz_h, lev, false, SCALE);
        MatAXPY(R->M, 0.5*dt, F->M, DIFFERENT_NONZERO_PATTERN);
    }

    // [u,exner] block
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta[lev+0]);
    VecAXPY(theta_k, 0.5, theta[lev+1]);
    F->assemble_up(theta_k, lev, SCALE, uhl);
    MatMatMult(pcx_M1_inv, EtoF->E12, reuse, PETSC_DEFAULT, &pcx_M1invD12);
    MatMatMult(pcx_M1invD12, M2->M, reuse, PETSC_DEFAULT, &pcx_M1invD12M2);
    MatMatMult(F->M, pcx_M1invD12M2, reuse, PETSC_DEFAULT, &pcx_G);
    MatScale(pcx_G, 0.5*dt);

    // [rho,u] block
    MatMatMult(M2->M, EtoF->E21, reuse, PETSC_DEFAULT, &pcx_M2D);
    MatMatMult(pcx_M2D, pcx_M1_inv, reuse, PETSC_DEFAULT, &pcx_M2DM1_inv);
    F->assemble(rho, lev, true, SCALE);
    MatMatMult(pcx_M2DM1_inv, F->M, reuse, PETSC_DEFAULT, &pcx_D_rho);
    MatScale(pcx_D_rho, 0.5*dt);

    // biharmonic viscosity operator
    if(do_visc) assemble_biharmonic(lev, reuse, &M1_OP);

    // density corrections
    assemble_rho_correction(lev, rho, exner, theta_k, reuse, diag_g, ones_g, &pcx_Au, uhl);

    MatMatMult(pcx_Au, M2inv->M, reuse, PETSC_DEFAULT, &pcx_Au_M2_inv);
    MatMatMult(pcx_Au_M2_inv, pcx_D_rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mu_prime); // invalid read on the second pass
    MatAYPX(Mu_prime, 1.0, R->M, DIFFERENT_NONZERO_PATTERN);
    MatDestroy(&pcx_Au);

    // velocity update
    MatMult(pcx_G, dexner, ones_g);
    VecAXPY(F_u, 1.0, ones_g);

    // actual solve for delta u update improves convergence
    KSPCreate(MPI_COMM_WORLD, &ksp_u);
    if(do_visc) {
        MatAXPY(M1_OP, 1.0, Mu_prime, DIFFERENT_NONZERO_PATTERN);
        KSPSetOperators(ksp_u, M1_OP, M1_OP);
    } else {
        KSPSetOperators(ksp_u, Mu_prime, Mu_prime);
    }
    KSPSetTolerances(ksp_u, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_u, KSPGMRES);
    KSPGetPC(ksp_u, &pc);
    PCSetType(pc, PCBJACOBI);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
//    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp_u, "ksp_u_");
    KSPSetFromOptions(ksp_u);

    KSPSolve(ksp_u, F_u, du);
    VecScale(du, -1.0);

    // density weighted potential temperature update
    if(do_rt) {
        M2_pi_inv->assemble(exner, lev, SCALE, false);
        M2_rt_inv->assemble(rt, lev, SCALE, true);
        MatScale(M2_rt_inv->M, -1.0*CV/RD);
        MatMult(M2_pi_inv->M, dexner, h_tmp);
        VecAXPY(F_exner, 1.0, h_tmp);
        MatMult(M2_rt_inv->M, F_exner, drt);
        if(neg_scale) {
            VecScale(drt, -1.0);
        }
    }

    // density update
    MatMult(pcx_D_rho, du, h_tmp);
    VecAXPY(F_rho, 1.0, h_tmp);
    //MatMult(M2inv->M, F_rho, drho);
    if(neg_scale) {
        MatMult(M2inv->M, F_rho, drho);
        VecScale(drho, -1.0);
    }

    VecDestroy(&wl);
    VecDestroy(&wg);
    VecDestroy(&theta_k);
    VecDestroy(&h_tmp);
    VecDestroy(&diag_g);
    VecDestroy(&ones_g);
    VecDestroy(&velz_h);
    MatDestroy(&Mu_prime);
    KSPDestroy(&ksp_u);
    if(do_visc) MatDestroy(&M1_OP);
}

/***************************************************************************************/

// update F_u (for density residual), F_rt (for exner residual and horiztonal velocity residual), 
// and assemble the laplacian term of the preconditioner
void HorizSolve::update_residuals(int lev, Vec* theta, Vec velx, Vec rho, Vec rt, Vec exner, Vec F_u, Vec F_rho, Vec F_rt, Vec F_exner,
                                  L2Vecs* velz_i, L2Vecs* velz_j, Vec uhl) {
    Vec wg, wl, theta_k, diag_g, ones_g, h_tmp, velz_h;
    Mat Mu_inv, Mu_prime, M1_OP;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &velz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &diag_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ones_g);

    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE, true);
    M2->assemble(lev, SCALE, true);
    M2inv->assemble(lev, SCALE);

    MatGetDiagonal(M1->M, diag_g);
    VecSet(ones_g, 1.0);
    VecPointwiseDivide(diag_g, ones_g, diag_g);
    MatDiagonalSet(pcx_M1_inv, diag_g, INSERT_VALUES);

    // [u,u] block
    curl(false, velx, &wg, lev, true);
    VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    R->assemble(wl, lev, SCALE);
    MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);
#ifdef RAYLEIGH
    if(lev == geom->nk-1) MatAXPY(R->M, 0.5*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
#endif
    VecDestroy(&wg);
    if(velz_i && velz_j) {
        VecZeroEntries(velz_h);
        if(lev > 0) {
            VecAXPY(velz_h, 1.0/(geom->thick[lev-1][0]+geom->thick[lev+0][0]), velz_i->vl[lev-1]);
            VecAXPY(velz_h, 1.0/(geom->thick[lev-1][0]+geom->thick[lev+0][0]), velz_j->vl[lev-1]);
        }
        if(lev < geom->nk-1) {
            VecAXPY(velz_h, 1.0/(geom->thick[lev+0][0]+geom->thick[lev+1][0]), velz_i->vl[lev+0]);
            VecAXPY(velz_h, 1.0/(geom->thick[lev+0][0]+geom->thick[lev+1][0]), velz_j->vl[lev+0]);
        }
        F->assemble(velz_h, lev, false, SCALE);
        MatAXPY(R->M, 0.5*dt, F->M, DIFFERENT_NONZERO_PATTERN);
    }

    // [u,exner] block
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta[lev+0]);
    VecAXPY(theta_k, 0.5, theta[lev+1]);
    F->assemble_up(theta_k, lev, SCALE, uhl);
    MatMatMult(pcx_M1_inv, EtoF->E12, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M1invD12);
    MatMatMult(pcx_M1invD12, M2->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M1invD12M2);
    MatMatMult(F->M, pcx_M1invD12M2, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_G);
    MatScale(pcx_G, 0.5*dt);

    MatMatMult(M2->M, EtoF->E21, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M2D);
    MatMatMult(pcx_M2D, pcx_M1_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M2DM1_inv);
    // [rt,u] block
    F->assemble(rt, lev, true, SCALE);
    MatMatMult(pcx_M2DM1_inv, F->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_D);
    MatScale(pcx_D, 0.5*dt);
    // [rho,u] block
    F->assemble(rho, lev, true, SCALE);
    MatMatMult(pcx_M2DM1_inv, F->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_D_rho);
    MatScale(pcx_D_rho, 0.5*dt);

    // biharmonic viscosity operator
    if(do_visc) assemble_biharmonic(lev, MAT_REUSE_MATRIX, &M1_OP);

    // density corrections
    assemble_rho_correction(lev, rho, exner, theta_k, MAT_REUSE_MATRIX, diag_g, ones_g, &pcx_Au, uhl);

    MatMatMult(pcx_Au, M2inv->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_Au_M2_inv);
    MatMatMult(pcx_Au_M2_inv, pcx_D_rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mu_prime); // invalid read on the second pass
    MatAYPX(Mu_prime, 1.0, R->M, DIFFERENT_NONZERO_PATTERN);
    if(do_visc) { 
        MatAXPY(M1_OP, 1.0, Mu_prime, DIFFERENT_NONZERO_PATTERN);
        coriolisMatInv(M1_OP, &Mu_inv);
    } else {
        coriolisMatInv(Mu_prime, &Mu_inv);
    }
    MatMult(pcx_Au_M2_inv, F_rho, diag_g);
    VecAXPY(F_u, -1.0, diag_g);

    MatDestroy(&pcx_Au);

    // build the preconditioner
    MatMatMult(pcx_D, Mu_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_D_Mu_inv);
    MatMatMult(pcx_D_Mu_inv, pcx_G, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_LAP);
    MatAssemblyBegin(pcx_LAP, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pcx_LAP, MAT_FINAL_ASSEMBLY);

    M2_rt_inv->assemble(rt, lev, SCALE, true);
    MatScale(M2_rt_inv->M, -1.0*CV/RD);

    if(do_visc) {   // temperature viscosity (biharmonic)
        assemble_biharmonic_temp(lev, rho, MAT_REUSE_MATRIX, &pcx_LAP2_Theta);
        MatMatMult(pcx_LAP2_Theta, M2_rt_inv->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M2N_rt_inv);
    } else {
        MatMatMult(M2->M, M2_rt_inv->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M2N_rt_inv);
    }

    MatZeroEntries(_PCx);
    MatAXPY(_PCx, -1.0, pcx_LAP, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(_PCx, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (_PCx, MAT_FINAL_ASSEMBLY);

    // update the rhs
    MatMult(pcx_M2N_rt_inv, F_exner, h_tmp);
    VecAXPY(F_rt, -1.0, h_tmp);
    MatMult(pcx_D_Mu_inv, F_u, h_tmp);
    VecAXPY(F_rt, -1.0, h_tmp);

    VecDestroy(&wl);
    VecDestroy(&theta_k);
    VecDestroy(&h_tmp);
    VecDestroy(&diag_g);
    VecDestroy(&ones_g);
    VecDestroy(&velz_h);
    MatDestroy(&Mu_inv);
    MatDestroy(&Mu_prime);
    if(do_visc) MatDestroy(&M1_OP);
    if(do_visc) MatDestroy(&pcx_LAP2_Theta);
}
