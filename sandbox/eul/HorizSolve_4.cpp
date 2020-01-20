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
#include "HorizSolve_4.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
//#define RAYLEIGH 0.4

using namespace std;

HorizSolve::HorizSolve(Topo* _topo, Geom* _geom, double _dt) {
    int ii, elOrd2;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    //do_visc = true;
    do_visc = false;
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
    N2_pi_inv = new N_rt_Inv(topo, geom, edge);
    N2_rt = new N_rt_Inv(topo, geom, edge);

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
    G_rt = NULL;

    MatCreate(MPI_COMM_WORLD, &M1_inv);
    MatSetSizes(M1_inv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M1_inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M1_inv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatZeroEntries(M1_inv);

    MatCreate(MPI_COMM_WORLD, &M0_inv);
    MatSetSizes(M0_inv, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M0_inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M0_inv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatZeroEntries(M0_inv);
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

    MatDestroy(&M0_inv);
    MatDestroy(&M1_inv);
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
    delete N2_pi_inv;
    delete N2_rt;
    delete M2_rho_inv;

    delete edge;
    delete node;
    delete quad;

    delete vo;
}

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void HorizSolve::grad(bool assemble, Vec phi, Vec* u, int lev) {
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
    grad(assemble, Du, ddu, lev);

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
                                Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec fu, Vec _F, Vec _G, Vec uil, Vec ujl, Vec grad_pi)
{
    Vec Phi, dPi, wxu, wxz, utmp, d2u, d4u;
    Vec theta_h, dp, dudz_h, velz_h;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &wxz);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &velz_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dudz_h);

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
    grad(false, Pi, &dPi, level);
    diagnose_wxu(level, velx1, velx2, &wxu);

    MatMult(EtoF->E12, Phi, fu);
    VecAXPY(fu, 1.0, wxu);

    // add the pressure gradient force
    F->assemble(theta_h, level, false, SCALE);
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

        VecZeroEntries(velz_h);
        VecAXPY(velz_h, 0.5, velz1[level-1]);
        VecAXPY(velz_h, 0.5, velz2[level-1]);

        Rh->assemble(dudz_h, SCALE);
        MatMult(Rh->M, velz_h, dp);
        VecAXPY(utmp, 0.5, dp);
    }
    if(level < geom->nk-1) {
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, dudz1[level+0]);
        VecAXPY(dudz_h, 0.5, dudz2[level+0]);

        VecZeroEntries(velz_h);
        VecAXPY(velz_h, 0.5, velz1[level+0]);
        VecAXPY(velz_h, 0.5, velz2[level+0]);

        Rh->assemble(dudz_h, SCALE);
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

    //if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, velx1);
        VecAXPY(utmp, 0.5, velx2);
        laplacian(false, utmp, &d2u, level);
        laplacian(false, d2u, &d4u, level);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    //}

    VecScatterBegin(topo->gtol_1, dPi, grad_pi, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dPi, grad_pi, INSERT_VALUES, SCATTER_FORWARD);

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
    Vec ones;
    Mat LAP_1, VISC, VISC2;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &ones);

    VecSet(ones, 1.0);
    VecPointwiseDivide(ones, ones, m0->vg);
    MatDiagonalSet(M0_inv, ones, INSERT_VALUES);

    MatMatMult(M1invDT, M2D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &LAP_1);

    MatMatMult(NtoE->E01, M1->M, reuse, PETSC_DEFAULT, &CTM1);
    MatMatMult(M0_inv, CTM1, reuse, PETSC_DEFAULT, &M0_invCTM1);
    MatMatMult(NtoE->E10, M0_invCTM1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC);

    MatAssemblyBegin(LAP_1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  LAP_1, MAT_FINAL_ASSEMBLY);
    MatAXPY(VISC, 1.0, LAP_1, DIFFERENT_NONZERO_PATTERN);
    MatScale(VISC, del2);
    MatMatMult(VISC, VISC, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC2);
    MatMatMult(M1->M, VISC2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, BVISC);  
    MatScale(*BVISC, 0.5*dt);

    VecDestroy(&ones);
    MatDestroy(&LAP_1);
    MatDestroy(&VISC);
    MatDestroy(&VISC2);
}

void HorizSolve::assemble_biharmonic_temp(int lev, Vec rho, MatReuse reuse, Mat* BVISC) {
    M2_rho_inv->assemble(rho, lev, SCALE);
    MatMatMult(M2_rho_inv->M, M2->M, reuse, PETSC_DEFAULT, &M2_invM2);

    MatMatMult(M1invDTM2, M2_invM2, reuse, PETSC_DEFAULT, &M1_invDT_M2M2_invM2);

    F->assemble(rho, lev, true, SCALE);
    MatMatMult(F->M, M1_invDT_M2M2_invM2, reuse, PETSC_DEFAULT, &M1_rhoM1_invDT_M2M2_invM2);

    MatMatMult(M2DM1_inv, M1_rhoM1_invDT_M2M2_invM2, reuse, PETSC_DEFAULT, &M2_LAP_Theta);

    MatMatMult(EtoF->E12, M2_LAP_Theta, reuse, PETSC_DEFAULT, &DT_LAP_Theta);
    MatMatMult(M2DM1_inv, DT_LAP_Theta, MAT_INITIAL_MATRIX, PETSC_DEFAULT, BVISC);

    MatAYPX(*BVISC, 0.5*dt*del2*del2, M2->M, DIFFERENT_NONZERO_PATTERN);
}

void HorizSolve::solve_schur_level(int lev, Vec* theta, Vec velx_l, Vec velx_g, Vec rho, Vec rt, Vec pi, 
                                   Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi, Vec dpil)
{
    MatReuse reuse = (!G_rt) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    Vec wl, wg, theta_k, ones, diag, tmp_h, tmp_u_1, tmp_u_2;
    PC pc;

    if(!rank) cout << "solving schur complement for level: " << lev << endl;

    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE, true);
    M2->assemble(lev, SCALE, true);
    M2inv->assemble(lev, SCALE);

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta[lev+0]);
    VecAXPY(theta_k, 0.5, theta[lev+1]);

    // assemble operators
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &diag);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ones);
    VecSet(ones, 1.0);
    MatGetDiagonal(M1->M, diag);
    VecPointwiseDivide(diag, ones, diag);
    MatDiagonalSet(M1_inv, diag, INSERT_VALUES);
    VecDestroy(&diag);
    VecDestroy(&ones);

    // M_u
    curl(false, velx_g, &wg, lev, true);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecDestroy(&wg);
    R->assemble(wl, lev, SCALE);
    VecDestroy(&wl);
    MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);
#ifdef RAYLEIGH
    if(lev == geom->nk-1) MatAXPY(R->M, 0.500*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
    if(lev == geom->nk-2) MatAXPY(R->M, 0.250*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
    if(lev == geom->nk-3) MatAXPY(R->M, 0.125*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
#endif

    // G_rt
    K->assemble(dpil, lev, SCALE); // 0.5 factor included here
    MatTranspose(K->M, reuse, &KT);
    M2_rho_inv->assemble(rho, lev, SCALE);
    MatMatMult(KT, M2_rho_inv->M, reuse, PETSC_DEFAULT, &KTM2_inv);
    MatMatMult(KTM2_inv, M2->M, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, dt);

    // G_pi
    F->assemble(theta_k, lev, false, SCALE);
    MatMatMult(M1_inv, EtoF->E12, reuse, PETSC_DEFAULT, &M1invDT);
    MatMatMult(M1invDT, M2->M, reuse, PETSC_DEFAULT, &M1invDTM2);
    MatMatMult(F->M, M1invDTM2, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);

    // D_rho
    MatMatMult(M2->M, EtoF->E21, reuse, PETSC_DEFAULT, &M2D);
    MatMatMult(M2D, M1_inv, reuse, PETSC_DEFAULT, &M2DM1_inv);
    F->assemble(rho, lev, true, SCALE);
    MatMatMult(M2DM1_inv, F->M, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);

    // D_rt
    T->assemble(rho, lev, SCALE, true);
    MatMatMult(T->M, EtoF->E21, reuse, PETSC_DEFAULT, &D_rt);
    MatScale(D_rt, 0.5*dt);

    // Q_rt_rho
    MatMatMult(M1_inv, EtoF->E12, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M1invDT);
    T->assemble(theta_k, lev, SCALE, false);
    MatMatMult(M1invDT, T->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M1invDTM2);
    K->assemble(velx_l, lev, SCALE); // 0.5 factor included here
    MatMatMult(K->M, M1invDTM2, reuse, PETSC_DEFAULT, &Q_rt_rho);
    MatScale(Q_rt_rho, dt);

    // N_rt
    N2_rt->assemble(rt, lev, SCALE, false);
    MatScale(N2_rt->M, -1.0*RD/CV);

    // N_pi (inverse)
    N2_pi_inv->assemble(pi, lev, SCALE, true);

    // M_u (inverse)
    if(do_visc) { 
        assemble_biharmonic(lev, reuse, &M_u);
        MatAXPY(M_u, 1.0, R->M, DIFFERENT_NONZERO_PATTERN);
    } else {
        MatDuplicate(R->M, MAT_COPY_VALUES, &M_u);
    }
    coriolisMatInv(M_u, &M_u_inv);

    // M_rt
    if(do_visc) {
        assemble_biharmonic_temp(lev, rho, reuse, &_PCx);
        MatAXPY(_PCx, 1.0, M2->M, DIFFERENT_NONZERO_PATTERN);
    }

    // assemble the secondary operators
    MatMatMult(D_rho, M_u_inv, reuse, PETSC_DEFAULT, &D_rho_M_u_inv);
    MatMatMult(D_rt,  M_u_inv, reuse, PETSC_DEFAULT, &D_rt_M_u_inv );

    MatMatMult(D_rho_M_u_inv, G_rt, reuse, PETSC_DEFAULT, &L_rho_rt);
    MatMatMult(D_rho_M_u_inv, G_pi, reuse, PETSC_DEFAULT, &L_rho_pi);
    if(do_visc) {
        MatMatMult(D_rt_M_u_inv,  G_rt, reuse, PETSC_DEFAULT, &L_rt_rt );
    } else {
        MatMatMult(D_rt_M_u_inv,  G_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L_rt_rt );
    }
    MatMatMult(D_rt_M_u_inv,  G_pi, reuse, PETSC_DEFAULT, &L_rt_pi );

    if(do_visc) {
        MatAXPY(_PCx, -1.0, L_rt_rt, DIFFERENT_NONZERO_PATTERN);
    } else {
        MatAYPX(L_rt_rt, -1.0, M2->M, DIFFERENT_NONZERO_PATTERN);
    }

    MatMatMult(L_rho_pi, N2_pi_inv->M, reuse, PETSC_DEFAULT, &L_rho_pi_N_pi_inv);
    MatMatMult(L_rt_pi,  N2_pi_inv->M, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv );
    MatMatMult(L_rt_pi_N_pi_inv,  N2_rt->M, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv_N_rt );
    MatMatMult(L_rho_pi_N_pi_inv, N2_rt->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L_rho_pi_N_pi_inv_N_rt); // invalid read if reused matrix

    MatMatMult(Q_rt_rho, M2inv->M, reuse, PETSC_DEFAULT, &Q_rt_rho_M_rho_inv);

    MatAXPY(L_rho_pi_N_pi_inv_N_rt, -1.0, L_rho_rt, DIFFERENT_NONZERO_PATTERN); // seg fault for reused matrix

    if(do_visc) {
        MatMatMult(Q_rt_rho_M_rho_inv, L_rho_pi_N_pi_inv_N_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &TEMP1);
        MatAYPX(TEMP1, -1.0, L_rt_pi_N_pi_inv_N_rt, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(_PCx, +1.0, TEMP1, DIFFERENT_NONZERO_PATTERN);
    } else {
        MatMatMult(Q_rt_rho_M_rho_inv, L_rho_pi_N_pi_inv_N_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &_PCx);
        MatAYPX(_PCx, -1.0, L_rt_pi_N_pi_inv_N_rt, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(_PCx, +1.0, L_rt_rt, DIFFERENT_NONZERO_PATTERN);
    }

    KSPCreate(MPI_COMM_WORLD, &ksp_rt);
    KSPSetOperators(ksp_rt, _PCx, _PCx);
    KSPSetTolerances(ksp_rt, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_rt, KSPGMRES);
    KSPGetPC(ksp_rt, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 6*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp_rt, "ksp_rt_");
    KSPSetFromOptions(ksp_rt);

    KSPCreate(MPI_COMM_WORLD, &ksp_u);
    KSPSetOperators(ksp_u, M_u, M_u);
    KSPSetTolerances(ksp_u, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_u, KSPGMRES);
    KSPGetPC(ksp_u, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 6*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp_u, "ksp_u_");
    KSPSetFromOptions(ksp_u);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp_u_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp_u_2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp_h);

    // update the residuals
    MatMult(D_rho_M_u_inv, F_u, tmp_h);
    VecAXPY(F_rho, -1.0, tmp_h);         // F_{rho}'
    MatMult(D_rt_M_u_inv,  F_u, tmp_h);
    VecAXPY(F_rt,  -1.0, tmp_h);         // F_{rt}'

    MatMult(L_rho_pi_N_pi_inv, F_pi, tmp_h);
    VecAXPY(F_rho, +1.0, tmp_h);         // F_{rho}''
    MatMult(L_rt_pi_N_pi_inv,  F_pi, tmp_h);
    VecAXPY(F_rt,  +1.0, tmp_h);         // F_{rt}''

    MatMult(Q_rt_rho_M_rho_inv, F_rho, tmp_h);
    VecAYPX(F_rt, -1.0, tmp_h);          // F_{rt}'''

    KSPSolve(ksp_rt, F_rt, d_rt);

    // back substitute
    MatMult(N2_rt->M, d_rt, tmp_h);
    VecAXPY(F_pi, 1.0, tmp_h);
    MatMult(N2_pi_inv->M, F_pi, d_pi);
    VecScale(d_pi, -1.0);

    MatMult(L_rho_pi_N_pi_inv_N_rt, d_rt, tmp_h);
    VecAXPY(F_rho, 1.0, tmp_h);
    MatMult(M2inv->M, F_rho, d_rho);
    VecScale(d_rho, -1.0);

    MatMult(G_rt, d_rt, tmp_u_1);
    MatMult(G_pi, d_pi, tmp_u_2);
    VecAXPY(F_u, 1.0, tmp_u_1);
    VecAXPY(F_u, 1.0, tmp_u_2);
    VecScale(F_u, -1.0);
    KSPSolve(ksp_u, F_u, d_u);

    VecDestroy(&theta_k);
    VecDestroy(&tmp_h);
    VecDestroy(&tmp_u_1);
    VecDestroy(&tmp_u_2);
    MatDestroy(&M_u);
    MatDestroy(&L_rho_pi_N_pi_inv_N_rt);
    if(do_visc) MatDestroy(&TEMP1);
    else        MatDestroy(&L_rt_rt);
    MatDestroy(&_PCx);
    KSPDestroy(&ksp_rt);
    KSPDestroy(&ksp_u);
}

void HorizSolve::assemble_and_update(int lev, Vec* theta, Vec velx_l, Vec velx_g, Vec rho, Vec rt, Vec pi, 
                                     Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec dpil)
{
    MatReuse reuse = (!G_rt) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    Vec wl, wg, theta_k, ones, diag, tmp_h;

    if(!rank) cout << "solving schur complement for level: " << lev << endl;

    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE, true);
    M2->assemble(lev, SCALE, true);
    M2inv->assemble(lev, SCALE);

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta[lev+0]);
    VecAXPY(theta_k, 0.5, theta[lev+1]);

    // assemble operators
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &diag);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ones);
    VecSet(ones, 1.0);
    MatGetDiagonal(M1->M, diag);
    VecPointwiseDivide(diag, ones, diag);
    MatDiagonalSet(M1_inv, diag, INSERT_VALUES);
    VecDestroy(&diag);
    VecDestroy(&ones);

    // M_u
    curl(false, velx_g, &wg, lev, true);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecDestroy(&wg);
    R->assemble(wl, lev, SCALE);
    VecDestroy(&wl);
    MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);
#ifdef RAYLEIGH
    if(lev == geom->nk-1) MatAXPY(R->M, 0.500*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
    if(lev == geom->nk-2) MatAXPY(R->M, 0.250*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
    if(lev == geom->nk-3) MatAXPY(R->M, 0.125*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
#endif

    // G_rt
    K->assemble(dpil, lev, SCALE); // 0.5 factor included here
    MatTranspose(K->M, reuse, &KT);
    M2_rho_inv->assemble(rho, lev, SCALE);
    MatMatMult(KT, M2_rho_inv->M, reuse, PETSC_DEFAULT, &KTM2_inv);
    MatMatMult(KTM2_inv, M2->M, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, dt);

    // G_pi
    F->assemble(theta_k, lev, false, SCALE);
    MatMatMult(M1_inv, EtoF->E12, reuse, PETSC_DEFAULT, &M1invDT);
    MatMatMult(M1invDT, M2->M, reuse, PETSC_DEFAULT, &M1invDTM2);
    MatMatMult(F->M, M1invDTM2, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);

    // D_rho
    MatMatMult(M2->M, EtoF->E21, reuse, PETSC_DEFAULT, &M2D);
    MatMatMult(M2D, M1_inv, reuse, PETSC_DEFAULT, &M2DM1_inv);
    F->assemble(rho, lev, true, SCALE);
    MatMatMult(M2DM1_inv, F->M, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);

    // D_rt
    T->assemble(rho, lev, SCALE, true);
    MatMatMult(T->M, EtoF->E21, reuse, PETSC_DEFAULT, &D_rt);
    MatScale(D_rt, 0.5*dt);

    // Q_rt_rho
    MatMatMult(M1_inv, EtoF->E12, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M1invDT);
    T->assemble(theta_k, lev, SCALE, false);
    MatMatMult(M1invDT, T->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M1invDTM2);
    K->assemble(velx_l, lev, SCALE); // 0.5 factor included here
    MatMatMult(K->M, M1invDTM2, reuse, PETSC_DEFAULT, &Q_rt_rho);
    MatScale(Q_rt_rho, dt);

    // N_rt
    N2_rt->assemble(rt, lev, SCALE, false);
    MatScale(N2_rt->M, -1.0*RD/CV);

    // N_pi (inverse)
    N2_pi_inv->assemble(pi, lev, SCALE, true);

    // M_u (inverse)
    if(do_visc) { 
        assemble_biharmonic(lev, reuse, &M_u);
        MatAXPY(M_u, 1.0, R->M, DIFFERENT_NONZERO_PATTERN);
    } else {
        MatDuplicate(R->M, MAT_COPY_VALUES, &M_u);
    }
    coriolisMatInv(M_u, &M_u_inv);

    // M_rt
    if(do_visc) {
        assemble_biharmonic_temp(lev, rho, reuse, &_PCx); // mass matrix already included from vertical assembly
        //MatAXPY(_PCx, 1.0, M2->M, DIFFERENT_NONZERO_PATTERN);
    }

    // assemble the secondary operators
    MatMatMult(D_rho, M_u_inv, reuse, PETSC_DEFAULT, &D_rho_M_u_inv);
    MatMatMult(D_rt,  M_u_inv, reuse, PETSC_DEFAULT, &D_rt_M_u_inv );

    MatMatMult(D_rho_M_u_inv, G_rt, reuse, PETSC_DEFAULT, &L_rho_rt);
    MatMatMult(D_rho_M_u_inv, G_pi, reuse, PETSC_DEFAULT, &L_rho_pi);
    if(do_visc) {
        MatMatMult(D_rt_M_u_inv,  G_rt, reuse, PETSC_DEFAULT, &L_rt_rt );
    } else {
        MatMatMult(D_rt_M_u_inv,  G_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L_rt_rt );
    }
    MatMatMult(D_rt_M_u_inv,  G_pi, reuse, PETSC_DEFAULT, &L_rt_pi );

    if(do_visc) {
        MatAXPY(_PCx, -1.0, L_rt_rt, DIFFERENT_NONZERO_PATTERN);
    //} else { // mass matrix already included in vertical assembly
    //    MatAYPX(L_rt_rt, -1.0, M2->M, DIFFERENT_NONZERO_PATTERN);
    }

    MatMatMult(L_rho_pi, N2_pi_inv->M, reuse, PETSC_DEFAULT, &L_rho_pi_N_pi_inv);
    MatMatMult(L_rt_pi,  N2_pi_inv->M, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv );
    MatMatMult(L_rt_pi_N_pi_inv,  N2_rt->M, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv_N_rt );
    MatMatMult(L_rho_pi_N_pi_inv, N2_rt->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L_rho_pi_N_pi_inv_N_rt); // invalid read if reused matrix

    MatMatMult(Q_rt_rho, M2inv->M, reuse, PETSC_DEFAULT, &Q_rt_rho_M_rho_inv);

    MatAXPY(L_rho_pi_N_pi_inv_N_rt, -1.0, L_rho_rt, DIFFERENT_NONZERO_PATTERN); // seg fault for reused matrix

    if(do_visc) {
        MatMatMult(Q_rt_rho_M_rho_inv, L_rho_pi_N_pi_inv_N_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &TEMP1);
        MatAYPX(TEMP1, -1.0, L_rt_pi_N_pi_inv_N_rt, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(_PCx, +1.0, TEMP1, DIFFERENT_NONZERO_PATTERN);
    } else {
        MatMatMult(Q_rt_rho_M_rho_inv, L_rho_pi_N_pi_inv_N_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &_PCx);
        MatAYPX(_PCx, -1.0, L_rt_pi_N_pi_inv_N_rt, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(_PCx, +1.0, L_rt_rt, DIFFERENT_NONZERO_PATTERN);
    }

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp_h);

    // update the residuals
    MatMult(D_rho_M_u_inv, F_u, tmp_h);
    VecAXPY(F_rho, -1.0, tmp_h);         // F_{rho}'
    MatMult(D_rt_M_u_inv,  F_u, tmp_h);
    VecAXPY(F_rt,  -1.0, tmp_h);         // F_{rt}'

    MatMult(L_rho_pi_N_pi_inv, F_pi, tmp_h);
    VecAXPY(F_rho, +1.0, tmp_h);         // F_{rho}''
    MatMult(L_rt_pi_N_pi_inv,  F_pi, tmp_h);
    VecAXPY(F_rt,  +1.0, tmp_h);         // F_{rt}''

    //MatMult(Q_rt_rho_M_rho_inv, F_rho, tmp_h); // residial update done in vertical assembly
    //VecAYPX(F_rt, -1.0, tmp_h);          // F_{rt}'''

    VecDestroy(&theta_k);
    VecDestroy(&tmp_h);
    MatDestroy(&M_u);
    MatDestroy(&L_rho_pi_N_pi_inv_N_rt);
    if(do_visc) MatDestroy(&TEMP1);
    else        MatDestroy(&L_rt_rt);
    //MatDestroy(&_PCx); // don't destroy the preconditioner here, this gets assembled into the schur matrix outside
}

void HorizSolve::update_deltas(int lev, Vec* theta, Vec velx_l, Vec velx_g, Vec rho, Vec rt, Vec pi, 
                               Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi, Vec dpil)
{
    MatReuse reuse = MAT_REUSE_MATRIX;
    Vec wl, wg, theta_k, ones, diag, tmp_h, tmp_u_1, tmp_u_2;
    PC pc;

    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE, true);
    M2->assemble(lev, SCALE, true);
    M2inv->assemble(lev, SCALE);

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta[lev+0]);
    VecAXPY(theta_k, 0.5, theta[lev+1]);

    // assemble operators
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &diag);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ones);
    VecSet(ones, 1.0);
    MatGetDiagonal(M1->M, diag);
    VecPointwiseDivide(diag, ones, diag);
    MatDiagonalSet(M1_inv, diag, INSERT_VALUES);
    VecDestroy(&diag);
    VecDestroy(&ones);

    // M_u
    curl(false, velx_g, &wg, lev, true);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecDestroy(&wg);
    R->assemble(wl, lev, SCALE);
    VecDestroy(&wl);
    MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);
#ifdef RAYLEIGH
    if(lev == geom->nk-1) MatAXPY(R->M, 0.500*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
    if(lev == geom->nk-2) MatAXPY(R->M, 0.250*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
    if(lev == geom->nk-3) MatAXPY(R->M, 0.125*dt*RAYLEIGH, M1->M, DIFFERENT_NONZERO_PATTERN);
#endif

    // G_rt
    K->assemble(dpil, lev, SCALE); // 0.5 factor included here
    MatTranspose(K->M, reuse, &KT);
    M2_rho_inv->assemble(rho, lev, SCALE);
    MatMatMult(KT, M2_rho_inv->M, reuse, PETSC_DEFAULT, &KTM2_inv);
    MatMatMult(KTM2_inv, M2->M, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, dt);

    // G_pi
    F->assemble(theta_k, lev, false, SCALE);
    MatMatMult(M1_inv, EtoF->E12, reuse, PETSC_DEFAULT, &M1invDT);
    MatMatMult(M1invDT, M2->M, reuse, PETSC_DEFAULT, &M1invDTM2);
    MatMatMult(F->M, M1invDTM2, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);

    // D_rho
    MatMatMult(M2->M, EtoF->E21, reuse, PETSC_DEFAULT, &M2D);
    MatMatMult(M2D, M1_inv, reuse, PETSC_DEFAULT, &M2DM1_inv);
    F->assemble(rho, lev, true, SCALE);
    MatMatMult(M2DM1_inv, F->M, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);

    // N_rt
    N2_rt->assemble(rt, lev, SCALE, false);
    MatScale(N2_rt->M, -1.0*RD/CV);

    // N_pi (inverse)
    N2_pi_inv->assemble(pi, lev, SCALE, true);

    // M_u (inverse)
    if(do_visc) { 
        assemble_biharmonic(lev, reuse, &M_u);
        MatAXPY(M_u, 1.0, R->M, DIFFERENT_NONZERO_PATTERN);
    } else {
        MatDuplicate(R->M, MAT_COPY_VALUES, &M_u);
    }
    coriolisMatInv(M_u, &M_u_inv);

    // assemble the secondary operators
    MatMatMult(D_rho, M_u_inv, reuse, PETSC_DEFAULT, &D_rho_M_u_inv);

    MatMatMult(D_rho_M_u_inv, G_rt, reuse, PETSC_DEFAULT, &L_rho_rt);
    MatMatMult(D_rho_M_u_inv, G_pi, reuse, PETSC_DEFAULT, &L_rho_pi);

    MatMatMult(L_rho_pi, N2_pi_inv->M, reuse, PETSC_DEFAULT, &L_rho_pi_N_pi_inv);
    MatMatMult(L_rho_pi_N_pi_inv, N2_rt->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L_rho_pi_N_pi_inv_N_rt); // invalid read if reused matrix

    MatAXPY(L_rho_pi_N_pi_inv_N_rt, -1.0, L_rho_rt, DIFFERENT_NONZERO_PATTERN); // seg fault for reused matrix

    KSPCreate(MPI_COMM_WORLD, &ksp_u);
    KSPSetOperators(ksp_u, M_u, M_u);
    KSPSetTolerances(ksp_u, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_u, KSPGMRES);
    KSPGetPC(ksp_u, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 6*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp_u, "ksp_u_");
    KSPSetFromOptions(ksp_u);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp_u_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp_u_2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp_h);

    // back substitute (exner pressure done in the vertical update
    MatMult(L_rho_pi_N_pi_inv_N_rt, d_rt, tmp_h);
    //VecAXPY(F_rho, 1.0, tmp_h);
    //MatMult(M2inv->M, F_rho, d_rho);
    //VecScale(d_rho, -1.0);
    VecAXPY(d_rho, -1.0, tmp_h); // F_rho term done in vertical update

    MatMult(G_rt, d_rt, tmp_u_1);
    MatMult(G_pi, d_pi, tmp_u_2);
    VecAXPY(F_u, 1.0, tmp_u_1);
    VecAXPY(F_u, 1.0, tmp_u_2);
    VecScale(F_u, -1.0);
    KSPSolve(ksp_u, F_u, d_u);

    VecDestroy(&theta_k);
    VecDestroy(&tmp_h);
    VecDestroy(&tmp_u_1);
    VecDestroy(&tmp_u_2);
    MatDestroy(&M_u);
    MatDestroy(&L_rho_pi_N_pi_inv_N_rt);
    KSPDestroy(&ksp_u);
}

double HorizSolve::MaxNorm(Vec dx, Vec x, double max_norm) {
    double norm_dx, norm_x, new_max_norm;

    VecNorm(dx, NORM_2, &norm_dx);
    VecNorm(x, NORM_2, &norm_x);
    new_max_norm = (norm_dx/norm_x > max_norm) ? norm_dx/norm_x : max_norm;
    return new_max_norm;
}

void HorizSolve::solve_schur(Vec* velx, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i) {
    bool done = false;
    int lev, ii, itt = 0;
    double max_norm_u, max_norm_exner, max_norm_rho, max_norm_rt, norm_x;
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* F_exner = new L2Vecs(geom->nk, topo, geom);
    L1Vecs* velx_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* velx_j  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_j  = new L1Vecs(geom->nk, topo, geom);
    Vec fu, frho, frt, fexner, du, drho, drt, dexner, _F, _G, dF, dG, dtheta, u_tmp_1, u_tmp_2, grad_pi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &frho);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &frt);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fexner);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &drho);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &drt);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dexner);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dG);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_2);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &grad_pi);

    rho_j->CopyFromHoriz(rho_i->vh);
    rt_j->CopyFromHoriz(rt_i->vh);
    exner_j->CopyFromHoriz(exner_i->vh);
    exner_h->CopyFromHoriz(exner_i->vh);
    velz_j->CopyFromHoriz(velz_i->vh);

    velx_i->CopyFrom(velx);
    velx_i->UpdateLocal();
    velx_j->CopyFrom(velx);
    velx_j->UpdateLocal();

    // diagnose the potential temperature
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rt_i->UpdateLocal();
    rt_i->HorizToVert();
    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();

    // diagnose the vorticity terms
    diagHorizVort(velx_i->vh, dudz_i->vh);
    dudz_i->UpdateLocal();
    dudz_j->CopyFrom(dudz_i->vh);
    dudz_j->UpdateLocal();

    rho_j->UpdateLocal();
    rho_j->HorizToVert();
    rt_j->UpdateLocal();
    rt_j->HorizToVert();

    do {
        max_norm_u = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        // exner pressure residual
        exner_j->UpdateLocal();
        exner_j->HorizToVert();
        for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            vo->Assemble_EOS_Residual(ii%topo->nElsX, ii/topo->nElsX, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
        }
        F_exner->VertToHoriz();
        F_exner->UpdateGlobal();

        for(lev = 0; lev < geom->nk; lev++) {
            // velocity residual
            assemble_residual(lev, theta_h->vl, dudz_i->vl, dudz_j->vl, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
                              velx_i->vh[lev], velx_j->vh[lev], rho_i->vh[lev], rho_j->vh[lev], fu, _F, _G, velx_i->vl[lev], velx_j->vl[lev], grad_pi);

            M2->assemble(lev, SCALE, true);

            // density residual
            MatMult(EtoF->E21, _F, dF);
            VecAYPX(dF, dt, rho_j->vh[lev]);
            VecAXPY(dF, -1.0, rho_i->vh[lev]);
            MatMult(M2->M, dF, frho);

            // density weighted potential temperature residual
            MatMult(EtoF->E21, _G, dG);
            VecAYPX(dG, dt, rt_j->vh[lev]);
            VecAXPY(dG, -1.0, rt_i->vh[lev]);
            MatMult(M2->M, dG, frt);

            // add in the viscous term
            //if(do_visc) {
                M1->assemble(lev, SCALE, true);
                VecZeroEntries(dF);
                VecAXPY(dF, 0.5, theta_h->vh[lev+0]);
                VecAXPY(dF, 0.5, theta_h->vh[lev+1]);

                grad(false, dF, &dtheta, lev);
                F->assemble(rho_j->vl[lev], lev, true, SCALE);
                MatMult(F->M, dtheta, u_tmp_1);
                VecDestroy(&dtheta);

                KSPSolve(ksp1, u_tmp_1, u_tmp_2);
                MatMult(EtoF->E21, u_tmp_2, dG);

                grad(false, dG, &dtheta, lev);
                MatMult(EtoF->E21, dtheta, dG);
                VecDestroy(&dtheta);
                MatMult(M2->M, dG, dF);
                VecAXPY(frt, dt*del2*del2, dF);
            //}

            // delta updates  - velx is a global vector, while theta and exner are local vectors
            solve_schur_level(lev, theta_h->vl, velx_j->vl[lev], velx_j->vh[lev], rho_j->vl[lev], rt_j->vl[lev], exner_j->vl[lev], 
                              fu, frho, frt, F_exner->vh[lev], du, drho, drt, dexner, grad_pi);

            VecAXPY(velx_j->vh[lev], 1.0, du);
            VecAXPY(rho_j->vh[lev], 1.0, drho);
            VecAXPY(rt_j->vh[lev], 1.0, drt);
            VecAXPY(exner_j->vh[lev], 1.0, dexner);

            max_norm_exner = MaxNorm(dexner, exner_j->vh[lev], max_norm_exner);
            max_norm_u     = MaxNorm(du,     velx_j->vh[lev],  max_norm_u    );
            max_norm_rho   = MaxNorm(drho,   rho_j->vh[lev],   max_norm_rho  );
            max_norm_rt    = MaxNorm(drt,    rt_j->vh[lev],    max_norm_rt   );
        }

        MPI_Allreduce(&max_norm_exner, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_exner = norm_x;
        MPI_Allreduce(&max_norm_u,     &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_u     = norm_x;
        MPI_Allreduce(&max_norm_rho,   &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho   = norm_x;
        MPI_Allreduce(&max_norm_rt,    &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt    = norm_x;

        itt++;

        if(max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_rho < 1.0e-8 && max_norm_rt < 1.0e-8) done = true;
        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                 "\t|d_u|/|u|: "          << max_norm_u     <<
                                 "\t|d_rho|/|rho|: "      << max_norm_rho   <<
                                 "\t|d_rt|/|rt|: "        << max_norm_rt    << endl;

        diagHorizVort(velx_j->vh, dudz_j->vh);
        dudz_j->UpdateLocal();

        // diagnose the potential temperature (at the half step)
        velx_j->UpdateLocal();
        rho_j->UpdateLocal();
        rho_j->HorizToVert();
        rt_j->UpdateLocal();
        rt_j->HorizToVert();
        diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        theta_h->VertToHoriz();
        for(lev = 0; lev < geom->nk; lev++) {
            VecScale(theta_h->vl[lev], 0.5);
            VecAXPY(theta_h->vl[lev], 0.5, theta_i->vl[lev]);

            VecZeroEntries(exner_h->vh[lev]);
            VecAXPY(exner_h->vh[lev], 0.5, exner_i->vh[lev]);
            VecAXPY(exner_h->vh[lev], 0.5, exner_j->vh[lev]);
        }
    } while(!done);

    // update the input/output fields
    velx_j->CopyTo(velx);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_h->vh);

    delete velx_i;
    delete velx_j;
    delete dudz_i;
    delete dudz_j;
    VecDestroy(&fu);
    VecDestroy(&frho);
    VecDestroy(&frt);
    VecDestroy(&fexner);
    VecDestroy(&du);
    VecDestroy(&drho);
    VecDestroy(&drt);
    VecDestroy(&dexner);
    VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&dF);
    VecDestroy(&dG);
    VecDestroy(&u_tmp_1);
    VecDestroy(&u_tmp_2);
    VecDestroy(&grad_pi);
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete theta_i;
    delete theta_h;
    delete velz_j;
    delete F_exner;
}
