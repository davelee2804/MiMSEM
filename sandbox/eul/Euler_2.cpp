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
#include "VertSolve.h"
#include "Euler_2.h"

#define RAD_EARTH 6371220.0
//#define RAD_EARTH (6371220.0/125.0)
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
//#define OMEGA 0.0
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
//#define RAYLEIGH 0.2
#define RAYLEIGH 0.0
//#define WITH_UDWDX

using namespace std;

Euler::Euler(Topo* _topo, Geom* _geom, double _dt) {
    int ii, n2;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    do_visc = true;
    del2 = viscosity();
    step = 0;
    firstStep = true;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // 0 form lumped mass matrix (vector)
    m0 = new Pvec(topo, geom, node);

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
    Rz = new WtQdUdz_mat(topo, geom, node, edge);

    // potential temperature projection operator
    T = new Whmat(topo, geom, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

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
    PCBJacobiSetTotalBlocks(pc, topo->elOrd*topo->elOrd, NULL);
    KSPSetOptionsPrefix(ksp2, "ksp2_");
    KSPSetFromOptions(ksp2);

    Kh = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Kh[ii]);
    }
    gv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd*topo->elOrd, &gv[ii]);
    }
    zv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*topo->elOrd*topo->elOrd, &zv[ii]);
    }
    uz = new Vec[geom->nk-1];
    uuz = new L2Vecs(geom->nk-1, topo, geom);
    for(ii = 0; ii < geom->nk - 1; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uz[ii]);
    }

    // initialise the single column mass matrices and solvers
    n2 = topo->elOrd*topo->elOrd;

    // implicit vertical solver run over a half time step
    vert = new VertSolve(topo, geom, 0.5*dt);

    MatCreate(MPI_COMM_SELF, &VA);
    MatSetType(VA, MATSEQAIJ);
    MatSetSizes(VA, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VA, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VB);
    MatSetType(VB, MATSEQAIJ);
    MatSetSizes(VB, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(VB, n2, PETSC_NULL);

    KSPCreate(MPI_COMM_SELF, &kspColA);
    KSPSetOperators(kspColA, VA, VA);
    KSPGetPC(kspColA, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColA, "kspColA_");
    KSPSetFromOptions(kspColA);

    KSPCreate(MPI_COMM_SELF, &kspColA2);
    KSPSetOperators(kspColA2, vert->vo->VA2, vert->vo->VA2);
    KSPGetPC(kspColA2, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColA2, "kspColA2_");
    KSPSetFromOptions(kspColA2);

    KSPCreate(MPI_COMM_WORLD, &kspE);
    KSPSetOperators(kspE, T->M, T->M);
    KSPSetTolerances(kspE, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspE, KSPGMRES);
    KSPGetPC(kspE, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, n2, NULL);
    KSPSetOptionsPrefix(kspE, "exner_");
    KSPSetFromOptions(kspE);

    Q = new Wii(node->q, geom);
    W = new M2_j_xy_i(edge);
    Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    initGZ();

    exner_prev = new L2Vecs(geom->nk, topo, geom);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double Euler::viscosity() {
    double ae = 4.0*M_PI*RAD_EARTH*RAD_EARTH;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void Euler::coriolis() {
    int ii, kk;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fl, fxl, fxg, PtQfxg;

    // initialise the coriolis vector (local and global)
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    fg = new Vec[geom->nk];

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
        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg[kk]);
        m0->assemble(kk, 1.0);
        VecPointwiseDivide(fg[kk], PtQfxg, m0->vg);
    }
    
    delete PtQ;
    VecDestroy(&fl);
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

void Euler::initGZ() {
    int ex, ey, ei, ii, kk, n2, mp12;
    int* inds0;
    int inds2k[99], inds0k[99];
    double* WtQflat = new double[W->nDofsJ*Q->nDofsJ];
    Vec gz;
    Mat GRAD, BQ;
    PetscScalar* zArray;
    MatReuse reuse;

    n2   = topo->elOrd*topo->elOrd;
    mp12 = (quad->n + 1)*(quad->n + 1);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*mp12, &gz);

    MatCreate(MPI_COMM_SELF, &BQ);
    MatSetType(BQ, MATSEQAIJ);
    MatSetSizes(BQ, (geom->nk+0)*n2, (geom->nk+1)*mp12, (geom->nk+0)*n2, (geom->nk+1)*mp12);
    MatSeqAIJSetPreallocation(BQ, 2*mp12, PETSC_NULL);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);
            Q->assemble(ex, ey);

            reuse = (!ei) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

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

            MatMatMult(vert->vo->V01, BQ, reuse, PETSC_DEFAULT, &GRAD);

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
}

Euler::~Euler() {
    int ii;

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    delete Q;
    delete W;

    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);
    KSPDestroy(&kspE);
    KSPDestroy(&kspColA);
    KSPDestroy(&kspColA2);

    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&fg[ii]);
        VecDestroy(&Kh[ii]);
    }
    delete[] fg;
    delete[] Kh;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
        VecDestroy(&zv[ii]);
    }
    delete[] gv;
    delete[] zv;
    for(ii = 0; ii < geom->nk-1; ii++) {
        VecDestroy(&uz[ii]);
    }
    delete[] uz;
    delete uuz;

    MatDestroy(&VA);
    MatDestroy(&VB);

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;
    delete T;
    delete M1t;
    delete Rh;
    delete Rz;

    delete edge;
    delete node;
    delete quad;

    delete vert;

    delete exner_prev;
}

/*
*/
void Euler::AssembleKEVecs(Vec* velx, Vec* velz) {
    int kk;
    Vec velx_l;

    // assemble the horiztonal operators
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &velx_l);
    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(velx_l);
        VecScatterBegin(topo->gtol_1, velx[kk], velx_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, velx[kk], velx_l, INSERT_VALUES, SCATTER_FORWARD);
        K->assemble(velx_l, kk, SCALE);
        VecZeroEntries(Kh[kk]);
        MatMult(K->M, velx[kk], Kh[kk]);
    }
    VecDestroy(&velx_l);
}

/*
compute the right hand side for the momentum equation for a given level
note that the vertical velocity, uv, is stored as a different vector for 
each element
*/
void Euler::horizMomRHS(Vec uh, Vec* theta_l, Vec exner, int lev, Vec Fu, Vec Flux, Vec uzb, Vec uzt, Vec velz_b, Vec velz_t) {
    double dot;
    Vec wl, wi, Ru, Ku, Mh, d2u, d4u, theta_k, dExner, dp;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);

    // assemble the mass matrices for use in the weak form grad, curl and laplacian operators
    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE, true);
    M2->assemble(lev, SCALE, true);

    curl(false, uh, &wi, lev, true);
    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(Fu);
    R->assemble(wl, lev, SCALE);
    MatMult(R->M, uh, Ru);
    MatMult(EtoF->E12, Kh[lev], Fu);
    VecAXPY(Fu, 1.0, Ru);

    // add the thermodynamic term (theta is in the same space as the vertical velocity)
    // project theta onto 1 forms
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta_l[lev+0]);
    VecAXPY(theta_k, 0.5, theta_l[lev+1]);

    grad(false, exner, &dExner, lev);
    F->assemble(theta_k, lev, false, SCALE);
    MatMult(F->M, dExner, dp);
    VecAXPY(Fu, 1.0, dp);
    VecDestroy(&dExner);

    // second voritcity term
    if(velz_b) {
        Rh->assemble(uzb, SCALE);
        MatMult(Rh->M, velz_b, Ru);
        VecAXPY(Fu, 0.5, Ru);
    }
    if(velz_t) {
        Rh->assemble(uzt, SCALE);
        MatMult(Rh->M, velz_t, Ru);
        VecAXPY(Fu, 0.5, Ru);
    }

    // evaluate the kinetic to internal energy exchange diagnostic (for this layer)
    VecScale(dp, 1.0/SCALE);
    VecDot(Flux, dp, &dot);
    k2i += dot;

    // add in the biharmonic viscosity
    if(do_visc) {
        laplacian(false, uh, &d2u, lev);
        laplacian(false, d2u, &d4u, lev);
        VecZeroEntries(d2u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(Fu, 1.0, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    VecDestroy(&wl);
    VecDestroy(&wi);
    VecDestroy(&Ru);
    VecDestroy(&Ku);
    VecDestroy(&Mh);
    VecDestroy(&dp);
    VecDestroy(&theta_k);
}

// pi is a local vector
void Euler::massRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* Flux) {
    int kk;
    Vec pu, Fh;

    // compute the horiztonal mass fluxes
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &pu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fh);

    for(kk = 0; kk < geom->nk; kk++) {
        F->assemble(pi[kk], kk, true, SCALE);
        M1->assemble(kk, SCALE, true);
        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fh);
        MatMult(EtoF->E21, Fh, Fp[kk]);

        VecZeroEntries(Flux[kk]);
        VecCopy(Fh, Flux[kk]);
    }

    VecDestroy(&pu);
    VecDestroy(&Fh);
}

void Euler::tempRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* rho_l, Vec* exner) {
    int kk;
    double dot;
    Vec pu, Fh, dF, theta_l, theta_g, dTheta, rho_dTheta_1, rho_dTheta_2, d2Theta, d3Theta;

    // compute the horiztonal mass fluxes
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_l);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &pu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    if(rho_l) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_g);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rho_dTheta_1);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rho_dTheta_2);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &d2Theta);
    }

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(theta_l);
        VecAXPY(theta_l, 0.5, pi[kk+0]);
        VecAXPY(theta_l, 0.5, pi[kk+1]);
        F->assemble(theta_l, kk, false, SCALE);

        M1->assemble(kk, SCALE, true);
        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fh);
        MatMult(EtoF->E21, Fh, Fp[kk]);

        // update the internal to kinetic energy flux diagnostic
        M2->assemble(kk, SCALE, true);
        MatMult(M2->M, Fp[kk], dF);
        VecScale(dF, 1.0/SCALE);
        VecDot(exner[kk], dF, &dot);
        i2k += dot;

        // apply horiztonal viscosity
        if(rho_l) {
            VecScatterBegin(topo->gtol_2, theta_l, theta_g, INSERT_VALUES, SCATTER_REVERSE);
            VecScatterEnd(  topo->gtol_2, theta_l, theta_g, INSERT_VALUES, SCATTER_REVERSE);

            M2->assemble(kk, SCALE, false);
            grad(false, theta_g, &dTheta, kk);
            F->assemble(rho_l[kk], kk, true, SCALE);
            MatMult(F->M, dTheta, rho_dTheta_1);

            KSPSolve(ksp1, rho_dTheta_1, rho_dTheta_2);
            MatMult(EtoF->E21, rho_dTheta_2, d2Theta);

            M2->assemble(kk, SCALE, true);
            grad(false, d2Theta, &d3Theta, kk);
            MatMult(EtoF->E21, d3Theta, d2Theta);
            VecAXPY(Fp[kk], del2*del2, d2Theta);

            VecDestroy(&dTheta);
            VecDestroy(&d3Theta);
        }
    }

    VecDestroy(&pu);
    VecDestroy(&Fh);
    VecDestroy(&dF);
    VecDestroy(&theta_l);
    if(rho_l) {
        VecDestroy(&theta_g);
        VecDestroy(&rho_dTheta_1);
        VecDestroy(&rho_dTheta_2);
        VecDestroy(&d2Theta);
    }
}

/* All vectors, rho, rt and theta are VERTICAL vectors */
void Euler::diagTheta2(Vec* rho, Vec* rt, Vec* theta) {
    int ex, ey, n2, ei;
    Vec frt;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &frt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            vert->vo->AssembleLinCon2(ex, ey, vert->vo->VAB2);
            MatMult(vert->vo->VAB2, rt[ei], frt);

            vert->vo->AssembleLinearWithRho2(ex, ey, rho[ei], vert->vo->VA2);
            KSPSolve(kspColA2, frt, theta[ei]);
        }
    }
    VecDestroy(&frt);
}

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void Euler::grad(bool assemble, Vec phi, Vec* u, int lev) {
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
void Euler::curl(bool assemble, Vec u, Vec* w, int lev, bool add_f) {
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

void Euler::laplacian(bool assemble, Vec ui, Vec* ddu, int lev) {
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

// rho and rt are local vectors, and exner is a global vector
void Euler::HorizRHS(Vec* velx, L2Vecs* rho, L2Vecs* rt, Vec* exner, Vec* Fu, Vec* Fp, Vec* Ft, Vec* velz) {
    int kk;
    L2Vecs* theta = new L2Vecs(geom->nk+1, topo, geom);
    Vec* Flux;
    Vec uzt, uzb, velz_t, velz_b;

    k2i = i2k = 0.0;

    Flux = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Flux[kk]);
    }
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &uzt);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &uzb);

    // set the top and bottom potential temperature bcs
    rho->HorizToVert();
    rt->HorizToVert();
    diagTheta2(rho->vz, rt->vz, theta->vz);
    theta->VertToHoriz();

    massRHS(velx, rho->vl, Fp, Flux);
    tempRHS(Flux, theta->vl, Ft, rho->vl, exner);

    HorizVort(velx);
    for(kk = 0; kk < geom->nk; kk++) {
        velz_t = velz_b = NULL;
        if(kk > 0) {
            VecScatterBegin(topo->gtol_1, uz[kk-1], uzb, INSERT_VALUES, SCATTER_FORWARD);
            VecScatterEnd(  topo->gtol_1, uz[kk-1], uzb, INSERT_VALUES, SCATTER_FORWARD);
            velz_b = velz[kk-1];
        }
        if(kk < geom->nk - 1) {
            VecScatterBegin(topo->gtol_1, uz[kk+0], uzt, INSERT_VALUES, SCATTER_FORWARD);
            VecScatterEnd(  topo->gtol_1, uz[kk+0], uzt, INSERT_VALUES, SCATTER_FORWARD);
            velz_t = velz[kk+0];
        }
        horizMomRHS(velx[kk], theta->vl, exner[kk], kk, Fu[kk], Flux[kk], uzb, uzt, velz_b, velz_t);
    }

    delete theta;
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Flux[kk]);
    }
    delete[] Flux;
    VecDestroy(&uzt);
    VecDestroy(&uzb);
}

// rt and Ft and local vectors
void Euler::SolveExner(Vec* rt, Vec* Ft, Vec* exner_i, Vec* exner_f, double _dt) {
    int ii;
    Vec rt_sum, rhs;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &rt_sum);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);

    for(ii = 0; ii < geom->nk; ii++) {
        VecCopy(Ft[ii], rt_sum);
        VecScale(rt_sum, -_dt*RD/CV);
        VecAXPY(rt_sum, 1.0, rt[ii]);
        T->assemble(rt_sum, ii, SCALE, true);
        MatMult(T->M, exner_i[ii], rhs);
        
        T->assemble(rt[ii], ii, SCALE, true);
        KSPSolve(kspE, rhs, exner_f[ii]);
    }
    VecDestroy(&rt_sum);
    VecDestroy(&rhs);
}

void Euler::VertToHoriz2(int ex, int ey, int ki, int kf, Vec pv, Vec* ph) {
    int ii, kk, n2;
    int* inds2 = topo->elInds2_l(ex, ey);
    PetscScalar *hArray, *vArray;

    n2 = topo->elOrd*topo->elOrd;

    VecGetArray(pv, &vArray);
    for(kk = ki; kk < kf; kk++) {
        VecGetArray(ph[kk], &hArray);
        for(ii = 0; ii < n2; ii++) {
            hArray[inds2[ii]] += vArray[(kk-ki)*n2+ii];
        }
        VecRestoreArray(ph[kk], &hArray);
    }
    VecRestoreArray(pv, &vArray);
}

void Euler::HorizToVert2(int ex, int ey, Vec* ph, Vec pv) {
    int ii, kk, n2;
    int* inds2 = topo->elInds2_l(ex, ey);
    PetscScalar *hArray, *vArray;

    n2 = topo->elOrd*topo->elOrd;

    VecGetArray(pv, &vArray);
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(ph[kk], &hArray);
        for(ii = 0; ii < n2; ii++) {
            vArray[kk*n2+ii] += hArray[inds2[ii]];
        }
        VecRestoreArray(ph[kk], &hArray);
    }
    VecRestoreArray(pv, &vArray);
}

void Euler::init0(Vec* q, ICfunc3D* func) {
    int ex, ey, ii, kk, mp1, mp12;
    int* inds0;
    PtQmat* PQ = new PtQmat(topo, geom, node);
    PetscScalar *bArray;
    Vec bl, bg, PQb;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bg);
        VecGetArray(bl, &bArray);

        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                inds0 = topo->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    bArray[inds0[ii]] = func(geom->x[inds0[ii]], kk);
                }
            }
        }
        VecRestoreArray(bl, &bArray);
        VecScatterBegin(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

        m0->assemble(kk, 1.0);
        MatMult(PQ->M, bg, PQb);
        VecPointwiseDivide(q[kk], PQb, m0->vg);
    }

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void Euler::init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y) {
    int ex, ey, ii, kk, mp1, mp12;
    int *inds0, *loc02;
    UtQmat* UQ = new UtQmat(topo, geom, node, edge);
    PetscScalar *bArray;
    Vec bl, bg, UQb;
    IS isl, isg;
    VecScatter scat;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    loc02 = new int[2*topo->n0];
    VecCreateSeq(MPI_COMM_SELF, 2*topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, 2*topo->n0l, 2*topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &UQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bg);
        VecGetArray(bl, &bArray);

        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                inds0 = topo->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    bArray[2*inds0[ii]+0] = func_x(geom->x[inds0[ii]], kk);
                    bArray[2*inds0[ii]+1] = func_y(geom->x[inds0[ii]], kk);
                }
            }
        }
        VecRestoreArray(bl, &bArray);

        // create a new vec scatter object to handle vector quantity on nodes
        for(ii = 0; ii < topo->n0; ii++) {
            loc02[2*ii+0] = 2*topo->loc0[ii]+0;
            loc02[2*ii+1] = 2*topo->loc0[ii]+1;
        }
        ISCreateStride(MPI_COMM_WORLD, 2*topo->n0, 0, 1, &isl);
        ISCreateGeneral(MPI_COMM_WORLD, 2*topo->n0, loc02, PETSC_COPY_VALUES, &isg);
        VecScatterCreate(bg, isg, bl, isl, &scat);
        VecScatterBegin(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

        M1->assemble(kk, SCALE, true);
        MatMult(UQ->M, bg, UQb);
        VecScale(UQb, SCALE);
        KSPSolve(ksp1, UQb, u[kk]);

        ISDestroy(&isl);
        ISDestroy(&isg);
        VecScatterDestroy(&scat);
    }

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&UQb);
    delete UQ;
    delete[] loc02;
}

void Euler::init2(Vec* h, ICfunc3D* func) {
    int ex, ey, ii, kk, mp1, mp12, *inds0;
    PetscScalar *bArray;
    Vec bl, bg, WQb;
    WtQmat* WQ = new WtQmat(topo, geom, edge);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &WQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bl);
        VecZeroEntries(bg);
        VecGetArray(bl, &bArray);

        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                inds0 = topo->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    bArray[inds0[ii]] = func(geom->x[inds0[ii]], kk);
                }
            }
        }
        VecRestoreArray(bl, &bArray);
        VecScatterBegin(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

        MatMult(WQ->M, bg, WQb);
        VecScale(WQb, SCALE);          // have to rescale the M2 operator as the metric terms scale
        M2->assemble(kk, SCALE, true); // this down to machine precision, so rescale the rhs as well
        KSPSolve(ksp2, WQb, h[kk]);
    }

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void Euler::initTheta(Vec theta, ICfunc3D* func) {
    int ex, ey, ii, mp1, mp12, *inds0;
    PetscScalar *bArray;
    Vec bl, bg, WQb;
    WtQmat* WQ = new WtQmat(topo, geom, edge);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &WQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]], 0);
            }
        }
    }
    VecRestoreArray(bl, &bArray);
    VecScatterBegin(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    M2->assemble(0, SCALE, false);
    MatMult(WQ->M, bg, WQb);
    VecScale(WQb, SCALE);
    KSPSolve(ksp2, WQb, theta);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

double Euler::int2(Vec ug) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, uq, local, global;
    PetscScalar *array_2;
    Vec ul;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &ul);
    VecScatterBegin(topo->gtol_2, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ul, &array_2);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, &uq);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*uq;
            }
        }
    }
    VecRestoreArray(ul, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&ul);

    return global;
}

void Euler::diagnostics(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner) {
    char filename[80];
    ofstream file;
    int kk, ex, ey, ei, n2;
    double keh, kev, pe, ie, k2p, p2k, mass;
    double dot, loc1, loc2, loc3;
    Vec hu, M2Pi, w2, gRho, gi, zi;
    Mat BA;
    L2Vecs* l2_rho = new L2Vecs(geom->nk, topo, geom);

    n2 = topo->elOrd*topo->elOrd;

    keh = kev = pe = ie = k2p = p2k = 0.0;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &w2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &gRho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &gi);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &zi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &M2Pi);

    MatCreate(MPI_COMM_SELF, &BA);
    MatSetType(BA, MATSEQAIJ);
    MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(BA, 2*n2, PETSC_NULL);

    l2_rho->CopyFromHoriz(rho);
    l2_rho->UpdateLocal();
    l2_rho->HorizToVert();

    // horiztonal kinetic energy
    for(kk = 0; kk < geom->nk; kk++) {
        F->assemble(l2_rho->vl[kk], kk, true, SCALE);
        MatMult(F->M, velx[kk], hu);
        VecScale(hu, 1.0/SCALE);
        VecDot(hu, velx[kk], &dot);
        keh += 0.5*dot;
    }

    // vertical kinetic energy and kinetic to potential exchange
    loc1 = loc2 = loc3 = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            MatZeroEntries(BA);
            vert->vo->AssembleConLinWithW(ex, ey, velz[ei], BA);
            MatMult(BA, velz[ei], w2);
            VecScale(w2, 1.0/SCALE);
            VecDot(l2_rho->vz[ei], w2, &dot);
            loc1 += 0.5*dot;

            vert->vo->AssembleLinearWithRT(ex, ey, l2_rho->vz[ei], VA, true);
            MatMult(VA, velz[ei], zi);
            vert->vo->AssembleLinearInv(ex, ey, VA);
            MatMult(VA, zi, gi);
            VecDot(gi, gv[ei], &dot);
            loc2 += dot/SCALE;

            MatMult(vert->vo->V10, gi, w2);
            VecDot(w2, zv[ei], &dot);
            loc3 += dot/SCALE;
        }
    }
    MPI_Allreduce(&loc1, &kev, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&loc2, &k2p, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&loc3, &p2k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // internal energy
    for(kk = 0; kk < geom->nk; kk++) {
        M2->assemble(kk, SCALE, true);
        MatMult(M2->M, exner[kk], M2Pi);
        VecScale(M2Pi, 1.0/SCALE);
        VecDot(rt[kk], M2Pi, &dot);
        ie += (CV/CP)*dot;
    }

    // potential energy
    loc1 = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            VecDot(zv[ei], l2_rho->vz[ei], &dot);
            loc1 += dot/SCALE;
        }
    }
    MPI_Allreduce(&loc1, &pe,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // mass
    mass = 0.0;
    for(kk = 0; kk < geom->nk; kk++) {
        mass += int2(rho[kk]);
    }

    if(!rank) {
        sprintf(filename, "output/energetics.dat");
        file.open(filename, ios::out | ios::app);
        file.precision(16);
        file << keh << "\t";
        file << kev << "\t";
        file << pe  << "\t";
        file << ie  << "\t";
        file << k2p << "\t";
        file << p2k << "\t";
        file << k2i << "\t";
        file << i2k << "\t";
        file << k2i_z << "\t";
        file << i2k_z << "\t";
        file << mass << "\t";
        file << endl;
        file.close();
    }

    VecDestroy(&M2Pi);
    VecDestroy(&hu);
    VecDestroy(&w2);
    VecDestroy(&gRho);
    VecDestroy(&gi);
    VecDestroy(&zi);
    MatDestroy(&BA);
    delete l2_rho;
}

void Euler::DiagExner(Vec* rtz, L2Vecs* exner) {
    int ex, ey, ei;
    int n2 = topo->elOrd*topo->elOrd;
    Vec eos_rhs;

    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &eos_rhs);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            vert->vo->Assemble_EOS_RHS(ex, ey, rtz[ei], eos_rhs, CP*pow(RD/P0, RD/CV), RD/CV);
            vert->vo->AssembleConstInv(ex, ey, VB);
            MatMult(VB, eos_rhs, exner->vz[ei]);
        }
    }

    VecDestroy(&eos_rhs);

    exner->VertToHoriz();
    exner->UpdateGlobal();
}

void Euler::VertSolve_Explicit(Vec* velz, Vec* rho, Vec* rt, Vec* exner, Vec* velz_n, Vec* rho_n, Vec* rt_n, Vec* exner_n) {
    int ex, ey, ei, n2;
    Mat V0_rt, V0_inv, V0_theta, V10_w, AB;
    Mat DTV10_w           = NULL;
    Mat DTV1              = NULL;
    Mat V0_invDTV1        = NULL;
    Mat GRAD              = NULL;
    Mat V0_invV0_rt       = NULL;
    Mat VR;
    Vec rhs, tmp;
    Vec wRho, wRhoTheta, Ftemp, frt;
    L2Vecs* l2_theta = new L2Vecs(geom->nk+1, topo, geom);
    VertOps* vo = vert->vo;
    MatReuse reuse = MAT_INITIAL_MATRIX;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &rhs);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &tmp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &wRho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &wRhoTheta);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Ftemp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &frt);

    // initialise matrices
    MatCreate(MPI_COMM_SELF, &V0_rt);
    MatSetType(V0_rt, MATSEQAIJ);
    MatSetSizes(V0_rt, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(V0_rt, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &V0_inv);
    MatSetType(V0_inv, MATSEQAIJ);
    MatSetSizes(V0_inv, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(V0_inv, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &V0_theta);
    MatSetType(V0_theta, MATSEQAIJ);
    MatSetSizes(V0_theta, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(V0_theta, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &V10_w);
    MatSetType(V10_w, MATSEQAIJ);
    MatSetSizes(V10_w, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(V10_w, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &AB);
    MatSetType(AB, MATSEQAIJ);
    MatSetSizes(AB, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(AB, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VR);
    MatSetType(VR, MATSEQAIJ);
    MatSetSizes(VR, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VR, 2*n2, PETSC_NULL);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            // assemble the vertical velocity rhs (except the exner term)
            vo->AssembleLinear(ex, ey, VA);
            MatMult(VA, velz_n[ei], rhs);
            VecAXPY(rhs, -0.5*dt, gv[ei]); // subtract the +ve gravity
#ifdef WITH_UDWDX
            VecAXPY(rhs, -0.5*dt, uuz->vz[ei]); // subtract the horizontal advection of vertical velocity
#endif
            vo->AssembleRayleigh(ex, ey, VR);

            // assemble the operators
            vo->AssembleConLinWithW(ex, ey, velz_n[ei], V10_w);
            MatMatMult(vo->V01, V10_w, reuse, PETSC_DEFAULT, &DTV10_w);

            vo->AssembleConst(ex, ey, VB);
            MatMatMult(vo->V01, VB, reuse, PETSC_DEFAULT, &DTV1);

            vo->AssembleLinearInv(ex, ey, V0_inv);
            MatMatMult(V0_inv, DTV1, reuse, PETSC_DEFAULT, &V0_invDTV1);

            vo->AssembleLinCon2(ex, ey, vo->VAB2);
            MatMult(vo->VAB2, rt_n[ei], frt);
            vo->AssembleLinearWithRho2(ex, ey, rho_n[ei], vo->VA2);
            KSPSolve(kspColA2, frt, l2_theta->vz[ei]);

            vo->AssembleLinearWithTheta(ex, ey, l2_theta->vz[ei], V0_theta);
            MatMatMult(V0_theta, V0_invDTV1, reuse, PETSC_DEFAULT, &GRAD);

            // add the exner pressure at the previous time level to the rhs
            MatMult(GRAD, exner_n[ei], tmp);
            VecAXPY(rhs, -0.5*dt, tmp);

            MatMult(DTV10_w, velz_n[ei], tmp);
            VecAXPY(rhs, -0.25*dt, tmp); // 0.5 for the nonlinear term and 0.5 for the time step

            MatMult(VR, velz_n[ei], tmp);
            VecAXPY(rhs, -0.5*dt*RAYLEIGH, tmp);

            vo->AssembleLinear(ex, ey, VA);
            KSPSolve(kspColA, rhs, velz[ei]);

            // update the density
            vo->AssembleLinearWithRT(ex, ey, rho_n[ei], V0_rt, true);
            MatMatMult(V0_inv, V0_rt, reuse, PETSC_DEFAULT, &V0_invV0_rt);
            MatMult(V0_invV0_rt, velz_n[ei], wRho);

            MatMult(vo->V10, wRho, Ftemp);
            VecCopy(rho_n[ei], rho[ei]);
            VecAXPY(rho[ei], -0.5*dt, Ftemp);

            // update the density weighted potential temperature
            MatZeroEntries(V0_invV0_rt);
            MatMatMult(V0_inv, V0_theta, MAT_REUSE_MATRIX, PETSC_DEFAULT, &V0_invV0_rt);
            MatMult(V0_invV0_rt, wRho, wRhoTheta);

            MatMult(vo->V10, wRhoTheta, Ftemp);
            VecCopy(rt_n[ei], rt[ei]);
            VecAXPY(rt[ei], -0.5*dt, Ftemp);

            // update the exner pressure
            vert->vo->Assemble_EOS_RHS(ex, ey, rt[ei], Ftemp, CP*pow(RD/P0, RD/CV), RD/CV);
            vert->vo->AssembleConstInv(ex, ey, vo->VB_inv);
            MatMult(vo->VB_inv, Ftemp, exner[ei]);

            reuse = MAT_REUSE_MATRIX;
        }
    }

    // update the initial solutions
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            VecCopy(velz[ei] , velz_n[ei] );
            VecCopy(exner[ei], exner_n[ei]);
            VecCopy(rt[ei]   , rt_n[ei]   );
            VecCopy(rho[ei]  , rho_n[ei]  );
        }
    }

    // deallocate
    VecDestroy(&rhs);
    VecDestroy(&tmp);
    VecDestroy(&wRho);
    VecDestroy(&wRhoTheta);
    VecDestroy(&Ftemp);
    VecDestroy(&frt);
    MatDestroy(&V0_rt            );
    MatDestroy(&V0_inv           );
    MatDestroy(&V0_theta         );
    MatDestroy(&V10_w            );
    MatDestroy(&DTV10_w          );
    MatDestroy(&DTV1             );
    MatDestroy(&V0_invDTV1       );
    MatDestroy(&GRAD             );
    MatDestroy(&V0_invV0_rt      );
    MatDestroy(&AB               );
    MatDestroy(&VR);
    delete l2_theta;
}

Vec* CreateHorizVecs(Topo* topo, Geom* geom) {
    Vec* vecs = new Vec[geom->nk];

    for(int ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &vecs[ii]);
    }
    return vecs;
}

void DestroyHorizVecs(Vec* vecs, Geom* geom) {
    for(int ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&vecs[ii]);
    }
    delete[] vecs;
}

void Euler::StrangCarryover(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int     ii;
    char    fieldname[100];
    Vec     wi, bu, xu;
    L2Vecs* velz_1  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_0   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_1   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_2   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_3   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_4   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_5   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_0    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_1    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_2    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_3    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_4    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_5    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_i = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* Fp      = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* Ft      = new L2Vecs(geom->nk, topo, geom);
    Vec*    Fu      = CreateHorizVecs(topo, geom);
    Vec*    velx_2  = CreateHorizVecs(topo, geom);
    Vec*    velx_3  = CreateHorizVecs(topo, geom);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &xu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);

    // 1.  explicit vertical solve using data from the end of the previous horizontal step
    if(!rank) cout << "vertical half step (1)..............." << endl;

    rho_0->CopyFromHoriz(rho);
    rho_0->UpdateLocal();
    rho_0->HorizToVert();
    rt_0->CopyFromHoriz(rt);
    rt_0->UpdateLocal();
    rt_0->HorizToVert();
    exner_i->CopyFromHoriz(exner);
    exner_i->UpdateLocal();
    exner_i->HorizToVert();

    exner_prev->CopyFromVert(exner_i->vz);
    exner_prev->VertToHoriz();
    exner_prev->UpdateGlobal();

    velz_1->CopyFromVert(velz);    velz_1->VertToHoriz();  velz_1->UpdateGlobal();
    rho_1->CopyFromHoriz(rho);     rho_1->UpdateLocal();   rho_1->HorizToVert();
    rt_1->CopyFromHoriz(rt);       rt_1->UpdateLocal();    rt_1->HorizToVert();

//    if(firstStep) {
//        if(!rank) cout << "first step, coupled vertical solve...\n";
        vert->solve_schur(velz_1, rho_1, rt_1, exner_i);
        //vert->solve_coupled(velz_1, rho_1, rt_1, exner_i);
//    } else {
        // pass in the current time level as well for q^{1} = q^{n} + F(q^{prev})
        // use the same horizontal kinetic energy as the previous horizontal solve, so don't assemble here
#ifdef WITH_UDWDX
//        velz_1->CopyFromVert(velz);
//        AssembleVertMomVort(velx, velz_1);
#endif
//        VertSolve_Explicit(velz_1->vz, rho_1->vz, rt_1->vz, exner_i->vz, velz, rho_0->vz, rt_0->vz, exner_prev->vz);
//    }
    velz_1->VertToHoriz();
    velz_1->UpdateGlobal();
    rho_1->VertToHoriz();
    rho_1->UpdateGlobal();
    rt_1->VertToHoriz();
    rt_1->UpdateGlobal();
    exner_i->VertToHoriz();
    exner_i->UpdateGlobal();
    exner_prev->VertToHoriz();
    exner_prev->UpdateGlobal();

    // 2.1 first horiztonal substep
    if(!rank) cout << "horiztonal step (1).................." << endl;
    AssembleKEVecs(velx, velz_1->vz);
    HorizRHS(velx, rho_1, rt_1, exner_i->vh, Fu, Fp->vh, Ft->vh, velz_1->vh);
    for(ii = 0; ii < geom->nk; ii++) {
        // momentum
        M1->assemble(ii, SCALE, true);
        MatMult(M1->M, velx[ii], bu);
        VecAXPY(bu, -dt, Fu[ii]);
        KSPSolve(ksp1, bu, velx_2[ii]);

        // continuity
        VecCopy(rho_1->vh[ii], rho_2->vh[ii]);
        VecAXPY(rho_2->vh[ii], -dt, Fp->vh[ii]);

        // internal energy
        VecCopy(rt_1->vh[ii], rt_2->vh[ii]);
        VecAXPY(rt_2->vh[ii], -dt, Ft->vh[ii]);
    }
    rho_2->UpdateLocal();
    rt_2->UpdateLocal();
    rt_2->HorizToVert();

    // 2.2 second horiztonal step
    if(!rank) cout << "horiztonal step (2).................." << endl;
    AssembleKEVecs(velx_2, velz_1->vz);
    DiagExner(rt_2->vz, exner_i);
    HorizRHS(velx_2, rho_2, rt_2, exner_i->vh, Fu, Fp->vh, Ft->vh, velz_1->vh);
    for(ii = 0; ii < geom->nk; ii++) {
        // momentum
        VecZeroEntries(xu);
        VecAXPY(xu, 0.75, velx[ii]);
        VecAXPY(xu, 0.25, velx_2[ii]);
        M1->assemble(ii, SCALE, true);
        MatMult(M1->M, xu, bu);
        VecAXPY(bu, -0.25*dt, Fu[ii]);
        KSPSolve(ksp1, bu, velx_3[ii]);

        // continuity
        VecZeroEntries(rho_3->vh[ii]);
        VecAXPY(rho_3->vh[ii], 0.75, rho_1->vh[ii]);
        VecAXPY(rho_3->vh[ii], 0.25, rho_2->vh[ii]);
        VecAXPY(rho_3->vh[ii], -0.25*dt, Fp->vh[ii]);

        // internal energy
        VecZeroEntries(rt_3->vh[ii]);
        VecAXPY(rt_3->vh[ii], 0.75, rt_1->vh[ii]);
        VecAXPY(rt_3->vh[ii], 0.25, rt_2->vh[ii]);
        VecAXPY(rt_3->vh[ii], -0.25*dt, Ft->vh[ii]);
    }
    rho_3->UpdateLocal();
    rt_3->UpdateLocal();
    rt_3->HorizToVert();

    // 2.3 third horiztonal step
    if(!rank) cout << "horiztonal step (3).................." << endl;
    AssembleKEVecs(velx_3, velz_1->vz);
    DiagExner(rt_3->vz, exner_i);
    HorizRHS(velx_3, rho_3, rt_3, exner_i->vh, Fu, Fp->vh, Ft->vh, velz_1->vh);
    for(ii = 0; ii < geom->nk; ii++) {
        // momentum
        VecZeroEntries(xu);
        VecAXPY(xu, 1.0/3.0, velx[ii]);
        VecAXPY(xu, 2.0/3.0, velx_3[ii]);
        M1->assemble(ii, SCALE, true);
        MatMult(M1->M, xu, bu);
        VecAXPY(bu, (-2.0/3.0)*dt, Fu[ii]);
        KSPSolve(ksp1, bu, velx[ii]);

        // continuity
        VecZeroEntries(rho_4->vh[ii]);
        VecAXPY(rho_4->vh[ii], 1.0/3.0, rho_1->vh[ii]);
        VecAXPY(rho_4->vh[ii], 2.0/3.0, rho_3->vh[ii]);
        VecAXPY(rho_4->vh[ii], (-2.0/3.0)*dt, Fp->vh[ii]);

        // internal energy
        VecZeroEntries(rt_4->vh[ii]);
        VecAXPY(rt_4->vh[ii], 1.0/3.0, rt_1->vh[ii]);
        VecAXPY(rt_4->vh[ii], 2.0/3.0, rt_3->vh[ii]);
        VecAXPY(rt_4->vh[ii], (-2.0/3.0)*dt, Ft->vh[ii]);
    }
    rho_4->UpdateLocal();
    rho_4->HorizToVert();
    rt_4->UpdateLocal();
    rt_4->HorizToVert();

    // carry over the vertical fields to the next time level
    DiagExner(rt_4->vz, exner_i);
    exner_prev->CopyFromVert(exner_i->vz);

    // 3.  second vertical half step
    if(!rank) cout << "vertical half step (2)..............." << endl;
    AssembleKEVecs(velx, velz_1->vz);
    velz_1->CopyToVert(velz);
    rho_5->CopyFromVert(rho_4->vz);
    rt_5->CopyFromVert(rt_4->vz);
#ifdef WITH_UDWDX
    AssembleVertMomVort(velx, velz_1);
#endif
    velz_1->VertToHoriz();  velz_1->UpdateGlobal();
    rho_5->VertToHoriz();   rho_5->UpdateGlobal();
    rt_5->VertToHoriz();    rt_5->UpdateGlobal();
    DiagExner(rt_5->vz, exner_i);
    exner_i->VertToHoriz(); exner_i->UpdateGlobal();

    vert->solve_schur(velz_1, rho_5, rt_5, exner_i);
    //vert->solve_coupled(velz_1, rho_5, rt_5, exner_i);
    velz_1->CopyToVert(velz);

    // update input vectors
    rho_5->VertToHoriz();
    rho_5->UpdateGlobal();
    rho_5->CopyToHoriz(rho);
    rt_5->VertToHoriz();
    rt_5->UpdateGlobal();
    rt_5->CopyToHoriz(rt);
    exner_i->VertToHoriz();
    exner_i->UpdateGlobal();
    exner_i->CopyToHoriz(exner);

    firstStep = false;

    diagnostics(velx, velz, rho, rt, exner);

    // write output
    if(save) {
        step++;

        L2Vecs* l2Theta = new L2Vecs(geom->nk+1, topo, geom);
        diagTheta2(rho_5->vz, rt_5->vz, l2Theta->vz);
        l2Theta->VertToHoriz();
        l2Theta->UpdateGlobal();
        for(ii = 0; ii < geom->nk+1; ii++) {
            sprintf(fieldname, "theta");
            geom->write2(l2Theta->vh[ii], fieldname, step, ii, false);
        }
        delete l2Theta;

        for(ii = 0; ii < geom->nk; ii++) {
            curl(true, velx[ii], &wi, ii, false);

            sprintf(fieldname, "vorticity");
            geom->write0(wi, fieldname, step, ii);
            sprintf(fieldname, "velocity_h");
            geom->write1(velx[ii], fieldname, step, ii);
            sprintf(fieldname, "density");
            geom->write2(rho[ii], fieldname, step, ii, true);
            sprintf(fieldname, "rhoTheta");
            geom->write2(rt[ii], fieldname, step, ii, true);
            sprintf(fieldname, "exner");
            geom->write2(exner[ii], fieldname, step, ii, true);

            VecDestroy(&wi);
        }
        sprintf(fieldname, "velocity_z");
        geom->writeVertToHoriz(velz, fieldname, step, geom->nk-1);
    }

    VecDestroy(&xu);
    VecDestroy(&bu);
    delete velz_1;
    delete rho_0;
    delete rho_1;
    delete rho_2;
    delete rho_3;
    delete rho_4;
    delete rho_5;
    delete rt_0;
    delete rt_1;
    delete rt_2;
    delete rt_3;
    delete rt_4;
    delete rt_5;
    delete exner_i;
    delete Fp;
    delete Ft;
    DestroyHorizVecs(Fu, geom);
    DestroyHorizVecs(velx_2, geom);
    DestroyHorizVecs(velx_3, geom);
}

// compute the vorticity components dudz, dvdz
void Euler::HorizVort(Vec* velx) {
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
        KSPSolve(ksp1_t, du, uz[ii]);
    }

    VecDestroy(&du);
    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&Mu[ii]);
    }
    delete[] Mu;
    KSPDestroy(&ksp1_t);
}

// compute the contribution of the vorticity vector to the vertical momentum equation
void Euler::AssembleVertMomVort(Vec* velx, L2Vecs* velz) {
    int kk;
    Vec ul, ug, tmp, tmp1, dwdx;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dwdx);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ug);

    for(kk = 0; kk < geom->nk-1; kk++) {
        VecZeroEntries(uuz->vh[kk]);
    }

    velz->VertToHoriz();
    velz->UpdateGlobal();
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

        VecZeroEntries(ug);
        VecAXPY(ug, 0.5, velx[kk+0]);
        VecAXPY(ug, 0.5, velx[kk+1]);
        VecScatterBegin(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
        
        Rz->assemble(ul, SCALE);
        MatMult(Rz->M, dwdx, uuz->vh[kk]); // horizontal advection of vertical velocity
    }
    uuz->UpdateLocal();
    uuz->HorizToVert();

    VecDestroy(&ul);
    VecDestroy(&ug);
    VecDestroy(&tmp);
    VecDestroy(&tmp1);
    VecDestroy(&dwdx);
}
