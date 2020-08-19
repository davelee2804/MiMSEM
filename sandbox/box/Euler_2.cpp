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

#define _LX (1000.0)
#define GRAVITY 9.80616
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
#define WITH_UDWDX

using namespace std;

Euler::Euler(Topo* _topo, Geom* _geom, double _dt) {
    int ii, n2, size;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    do_visc = true;
    del2 = viscosity();
    step = 0;
    firstStep = true;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp1);
    KSPSetOperators(ksp1, M1->M, M1->M);
    KSPSetTolerances(ksp1, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    //KSPSetType(ksp1, KSPGMRES);
    KSPSetType(ksp1, KSPCG);
    KSPGetPC(ksp1, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1, "ksp1_");
    KSPSetFromOptions(ksp1);

    // initialize the 2 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPGetPC(ksp2, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp2, "ksp2_");
    KSPSetFromOptions(ksp2);

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp1o);
    KSPSetOperators(ksp1o, M1->Mo, M1->Mo);
    KSPSetTolerances(ksp1o, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    //KSPSetType(ksp1o, KSPGMRES);
    KSPSetType(ksp1o, KSPCG);
    KSPGetPC(ksp1o, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1o, "ksp1o_");
    KSPSetFromOptions(ksp1o);

    // initialize the 2 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp2o);
    KSPSetOperators(ksp2o, M2->Mo, M2->Mo);
    KSPSetTolerances(ksp2o, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2o, KSPGMRES);
    KSPGetPC(ksp2o, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp2o, "ksp2_");
    KSPSetFromOptions(ksp2o);

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
    uzl = new Vec[geom->nk-1];
    for(ii = 0; ii < geom->nk - 1; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uz[ii]);
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &uzl[ii]);
    }
    ul = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul[ii]);
    }
#ifdef WITH_UDWDX
    uuz = new L2Vecs(geom->nk-1, topo, geom);
#else
    uuz = NULL;
#endif

    // initialise the single column mass matrices and solvers
    n2 = topo->elOrd*topo->elOrd;

    // implicit vertical solver
    vert = new VertSolve(topo, geom, dt);

    MatCreate(MPI_COMM_SELF, &VA);
    MatSetType(VA, MATSEQAIJ);
    MatSetSizes(VA, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VA, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VB);
    MatSetType(VB, MATSEQAIJ);
    MatSetSizes(VB, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(VB, n2, PETSC_NULL);

    KSPCreate(MPI_COMM_SELF, &kspColA2);
    KSPSetOperators(kspColA2, vert->vo->VA2, vert->vo->VA2);
    KSPGetPC(kspColA2, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColA2, "kspColA2_");
    KSPSetFromOptions(kspColA2);

    Q = new Wii(node->q, geom);
    W = new M2_j_xy_i(edge);
    Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    initGZ();

    KT = NULL;
    M2inv = new WmatInv(topo, geom, edge);

    M1u = new Uvec(topo, geom, node, edge);
    M2h = new Wvec(topo, geom, edge);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double Euler::viscosity() {
    double ae = _LX*_LX;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    del4 *= 2.0;

    return -sqrt(del4);
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
    KSPDestroy(&ksp1o);
    KSPDestroy(&ksp2o);
    KSPDestroy(&kspColA2);

    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&Kh[ii]);
        VecDestroy(&ul[ii]);
    }
    delete[] Kh;
    delete[] ul;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
        VecDestroy(&zv[ii]);
    }
    delete[] gv;
    delete[] zv;
    for(ii = 0; ii < geom->nk-1; ii++) {
        VecDestroy(&uz[ii]);
        VecDestroy(&uzl[ii]);
    }
    delete[] uz;
    delete[] uzl;
#ifdef WITH_UDWDX
    delete uuz;
#endif

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

    delete M2inv;
    if(KT) MatDestroy(&KT);

    delete edge;
    delete node;
    delete quad;

    delete vert;

    delete M1u;
    delete M2h;
}

/*
*/
void Euler::AssembleKEVecs(Vec* velx) {
    // assemble the horiztonal operators
    for(int kk = 0; kk < geom->nk; kk++) {
        K->assemble(ul[kk], kk, SCALE);
        VecZeroEntries(Kh[kk]);
        MatMult(K->M, velx[kk], Kh[kk]);
    }
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
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_k);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);

    curl(uh, &wi, lev);
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

    grad(ksp1, M2->M, exner, &dExner, lev);
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
        laplacian(uh, &d2u, lev);
        laplacian(d2u, &d4u, lev);
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
        //F->assemble(pi[kk], kk, true, SCALE);
        //MatMult(F->M, uh[kk], pu);
        //KSPSolve(ksp1, pu, Fh);
        M1u->assemble_hu(kk, SCALE, true, ul[kk], pi[kk]);
        KSPSolve(ksp1, M1u->vg, Fh);

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
    Vec pu, Fh, dF, theta_g, dTheta, rho_dTheta_1, rho_dTheta_2, d2Theta, d3Theta;

    // compute the horiztonal mass fluxes
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
        VecZeroEntries(theta_g);
        VecAXPY(theta_g, 0.5, pi[kk+0]);
        VecAXPY(theta_g, 0.5, pi[kk+1]);
        F->assemble(theta_g, kk, false, SCALE);

        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fh);
        MatMult(EtoF->E21, Fh, Fp[kk]);

        // update the internal to kinetic energy flux diagnostic
        MatMult(M2->M, Fp[kk], dF);
        VecScale(dF, 1.0/SCALE);
        VecDot(exner[kk], dF, &dot);
        i2k += dot;

        // apply horiztonal viscosity
        if(rho_l) {
            grad(ksp1o, M2->Mo, theta_g, &dTheta, kk);
            F->assemble(rho_l[kk], kk, true, SCALE);
            MatMult(F->M, dTheta, rho_dTheta_1);

            KSPSolve(ksp1, rho_dTheta_1, rho_dTheta_2);
            MatMult(EtoF->E21, rho_dTheta_2, d2Theta);

            grad(ksp1, M2->M, d2Theta, &d3Theta, kk);
            MatMult(EtoF->E21, d3Theta, d2Theta);
            VecAXPY(Fp[kk], del2*del2, d2Theta);
            VecDestroy(&d3Theta);
            VecDestroy(&dTheta);
        }
    }

    VecDestroy(&pu);
    VecDestroy(&Fh);
    VecDestroy(&dF);
    if(rho_l) {
        VecDestroy(&theta_g);
        VecDestroy(&rho_dTheta_1);
        VecDestroy(&rho_dTheta_2);
        VecDestroy(&d2Theta);
    }
}

/* All vectors, rho, rt and theta are VERTICAL vectors */
void Euler::diagTheta(Vec* rho, Vec* rt, Vec* theta) {
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
void Euler::grad(KSP _ksp, Mat _M2, Vec phi, Vec* u, int lev) {
    Vec Mphi, dMphi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, u);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mphi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMphi);

    MatMult(_M2, phi, Mphi);
    MatMult(EtoF->E12, Mphi, dMphi);
    KSPSolve(_ksp, dMphi, *u);

    VecDestroy(&Mphi);
    VecDestroy(&dMphi);
}

/*
Take the weak form curl of a 1 form vector field as a 1 form vector field
*/
void Euler::curl(Vec u, Vec* w, int lev) {
    Vec Mu, dMu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &dMu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu);

    MatMult(M1->M, u, Mu);
    MatMult(NtoE->E01, Mu, dMu);
    VecPointwiseDivide(*w, dMu, m0->vg);

    VecDestroy(&Mu);
    VecDestroy(&dMu);
}

void Euler::laplacian(Vec ui, Vec* ddu, int lev) {
    Vec Du, Cu, RCu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &RCu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Du);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, ui, Du);

    // grad (weak form)
    grad(ksp1, M2->M, Du, ddu, lev);

    /*** rotational component ***/
    // curl (weak form)
    curl(ui, &Cu, lev);

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

    // set the top and bottom potential temperature bcs
    rho->HorizToVert();
    rt->HorizToVert();
    diagTheta(rho->vz, rt->vz, theta->vz);
    theta->VertToHoriz();

    massRHS(velx, rho->vh, Fp, Flux);
    tempRHS(Flux, theta->vh, Ft, rho->vh, exner);

    HorizVort(velx);
    for(kk = 0; kk < geom->nk; kk++) {
        velz_t = velz_b = NULL;
        uzb = uzt = NULL;
        if(kk > 0) {
            uzb = uzl[kk-1];
            velz_b = velz[kk-1];
        }
        if(kk < geom->nk - 1) {
            uzt = uzl[kk+0];
            velz_t = velz[kk+0];
        }
        horizMomRHS(velx[kk], theta->vh, exner[kk], kk, Fu[kk], Flux[kk], uzb, uzt, velz_b, velz_t);
    }

    delete theta;
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Flux[kk]);
    }
    delete[] Flux;
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

        MatMult(PQ->M, bg, PQb);
        VecPointwiseDivide(q[kk], PQb, m0->vg1);
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

    MatMult(WQ->M, bg, WQb);
    VecScale(WQb, SCALE);
    KSPSolve(ksp2o, WQb, theta);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

double Euler::int2(Vec ug) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, uq, local, global;
    PetscScalar *array_2;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ug, &array_2);
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
    VecRestoreArray(ug, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
    l2_rho->HorizToVert();

    // horiztonal kinetic energy
    for(kk = 0; kk < geom->nk; kk++) {
        F->assemble(l2_rho->vh[kk], kk, true, SCALE);
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

    k2i_z = vert->k2i_z;

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

void Euler::Trapazoidal(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    char    fieldname[100];
    Vec     wi, bu, xu;
    Vec*    Fu_0    = CreateHorizVecs(topo, geom);
    L2Vecs* Fp_0    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* Ft_0    = new L2Vecs(geom->nk, topo, geom);
    Vec*    Fu_1    = CreateHorizVecs(topo, geom);
    L2Vecs* Fp_1    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* Ft_1    = new L2Vecs(geom->nk, topo, geom);
    Vec*    Fu_2    = CreateHorizVecs(topo, geom);
    L2Vecs* Fp_2    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* Ft_2    = new L2Vecs(geom->nk, topo, geom);
    Vec*    velx_0  = CreateHorizVecs(topo, geom);
    L2Vecs* velz_0  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_0   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_0    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_0 = new L2Vecs(geom->nk, topo, geom);
    Vec*    velx_1  = CreateHorizVecs(topo, geom);
    L2Vecs* rho_1   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_1    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_1 = new L2Vecs(geom->nk, topo, geom);
    Vec*    velx_2  = CreateHorizVecs(topo, geom);
    L2Vecs* rho_2   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_2    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_2 = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_3   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_3    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_3 = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* velz_h  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_h   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_rho_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_rt_h  = new L2Vecs(geom->nk, topo, geom);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &xu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);

    // 0.  Copy initial fields 
    for(int kk = 0; kk < geom->nk; kk++) VecCopy(velx[kk], velx_0[kk]);
    velz_0->CopyFromVert(velz);    velz_0->VertToHoriz();
    rho_0->CopyFromHoriz(rho);     rho_0->HorizToVert();
    rt_0->CopyFromHoriz(rt);       rt_0->HorizToVert();
    exner_0->CopyFromHoriz(exner); exner_0->HorizToVert();

    AssembleVertMomVort(velz_0);

    // 1.  Explicit horizontal solve
    if(!rank) cout << "horiztonal step (1).................." << endl;
    AssembleKEVecs(velx_0);
    HorizRHS(velx_0, rho_0, rt_0, exner_0->vh, Fu_0, Fp_0->vh, Ft_0->vh, velz_0->vh);
    for(int kk = 0; kk < geom->nk; kk++) {
        // momentum
        MatMult(M1->M, velx_0[kk], bu);
        VecAXPY(bu, -dt, Fu_0[kk]);

        KSPSolve(ksp1, bu, velx_1[kk]);
        VecScatterBegin(topo->gtol_1, velx_1[kk], ul[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, velx_1[kk], ul[kk], INSERT_VALUES, SCATTER_FORWARD);

        // continuity
        VecCopy(rho_0->vh[kk], rho_1->vh[kk]);
        VecAXPY(rho_1->vh[kk], -dt, Fp_0->vh[kk]);

        // internal energy
        VecCopy(rt_0->vh[kk], rt_1->vh[kk]);
        VecAXPY(rt_1->vh[kk], -dt, Ft_0->vh[kk]);
    }
    rho_1->HorizToVert();
    rt_1->HorizToVert();
    DiagExner(rt_1->vz, exner_1);

    // 2.1 Explicit horiztonal solve
    if(!rank) cout << "horiztonal step (2).................." << endl;
    AssembleKEVecs(velx_1);
    HorizRHS(velx_1, rho_1, rt_1, exner_1->vh, Fu_1, Fp_1->vh, Ft_1->vh, velz_0->vh);
    for(int kk = 0; kk < geom->nk; kk++) {
        // momentum
        MatMult(M1->M, velx_0[kk], bu);
        VecAXPY(bu, -0.5*dt, Fu_0[kk]);
        VecAXPY(bu, -0.5*dt, Fu_1[kk]);

        KSPSolve(ksp1, bu, velx_2[kk]);
        VecScatterBegin(topo->gtol_1, velx_2[kk], ul[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, velx_2[kk], ul[kk], INSERT_VALUES, SCATTER_FORWARD);

        // continuity
        VecCopy(rho_0->vh[kk], rho_2->vh[kk]);
        VecAXPY(rho_2->vh[kk], -0.5*dt, Fp_0->vh[kk]);
        VecAXPY(rho_2->vh[kk], -0.5*dt, Fp_1->vh[kk]);

        // internal energy
        VecCopy(rt_0->vh[kk],  rt_2->vh[kk]);
        VecAXPY(rt_2->vh[kk],  -0.5*dt, Ft_0->vh[kk]);
        VecAXPY(rt_2->vh[kk],  -0.5*dt, Ft_1->vh[kk]);

        // horizontal forcings for vertical solve
        VecZeroEntries(F_rho_h->vh[kk]);
        VecAXPY(F_rho_h->vh[kk], 0.5, Fp_0->vh[kk]);
        VecAXPY(F_rho_h->vh[kk], 0.5, Fp_1->vh[kk]);
        VecZeroEntries(F_rt_h->vh[kk]);
        VecAXPY(F_rt_h->vh[kk], 0.5, Ft_0->vh[kk]);
        VecAXPY(F_rt_h->vh[kk], 0.5, Ft_1->vh[kk]);
    }

    // 2.2 Implicit vertical solve
    //AssembleVertMomVort(velz_0);

    velz_h->CopyFromHoriz(velz_0->vh);
    rho_h->CopyFromHoriz(rho_0->vh);
    rt_h->CopyFromHoriz(rt_0->vh);
    exner_h->CopyFromHoriz(exner_0->vh);

    if(!rank) cout << "  vertical step (2).................." << endl;
    vert->solve_schur(velz_h, rho_h, rt_h, exner_h, uuz, M2->Mo, EtoF, ksp1o, F_rho_h, F_rt_h, ul, Rz);
    rho_2->CopyFromHoriz(rho_h->vh);     rho_2->HorizToVert();
    rt_2->CopyFromHoriz(rt_h->vh);       rt_2->HorizToVert();
    exner_2->CopyFromHoriz(exner_h->vh); exner_2->HorizToVert();

    // 3.2 Explicit horiztonal solve
    if(!rank) cout << "horiztonal step (3).................." << endl;
    AssembleKEVecs(velx_2);
    HorizRHS(velx_2, rho_2, rt_2, exner_2->vh, Fu_2, Fp_2->vh, Ft_2->vh, velz_h->vh);
    for(int kk = 0; kk < geom->nk; kk++) {
        // momentum
        MatMult(M1->M, velx_0[kk], bu);
        VecAXPY(bu, -0.5*dt, Fu_0[kk]);
        VecAXPY(bu, -0.5*dt, Fu_2[kk]);

        KSPSolve(ksp1, bu, velx[kk]);
        VecScatterBegin(topo->gtol_1, velx[kk], ul[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, velx[kk], ul[kk], INSERT_VALUES, SCATTER_FORWARD);

        // continuity
        VecCopy(rho_0->vh[kk], rho_3->vh[kk]);
        VecAXPY(rho_3->vh[kk], -0.5*dt, Fp_0->vh[kk]);
        VecAXPY(rho_3->vh[kk], -0.5*dt, Fp_2->vh[kk]);

        // internal energy
        VecCopy(rt_0->vh[kk],  rt_3->vh[kk]);
        VecAXPY(rt_3->vh[kk],  -0.5*dt, Ft_0->vh[kk]);
        VecAXPY(rt_3->vh[kk],  -0.5*dt, Ft_2->vh[kk]);

        // horizontal forcings for vertical solve
        VecZeroEntries(F_rho_h->vh[kk]);
        VecAXPY(F_rho_h->vh[kk], 0.5, Fp_0->vh[kk]);
        VecAXPY(F_rho_h->vh[kk], 0.5, Fp_2->vh[kk]);
        VecZeroEntries(F_rt_h->vh[kk]);
        VecAXPY(F_rt_h->vh[kk], 0.5, Ft_0->vh[kk]);
        VecAXPY(F_rt_h->vh[kk], 0.5, Ft_2->vh[kk]);
    }

    // 3.1 Implicit vertical solve
    //AssembleVertMomVort(velz_0);

    velz_h->CopyFromHoriz(velz_0->vh);
    rho_h->CopyFromHoriz(rho_0->vh);
    rt_h->CopyFromHoriz(rt_0->vh);
    exner_h->CopyFromHoriz(exner_0->vh);

    if(!rank) cout << "  vertical step (3).................." << endl;
    vert->solve_schur(velz_h, rho_h, rt_h, exner_h, uuz, M2->Mo, EtoF, ksp1o, F_rho_h, F_rt_h, ul, Rz);
    
    // update input vectors
    velz_h->CopyToVert(velz);
    rho_h->CopyToHoriz(rho);
    rt_h->CopyToHoriz(rt);
    exner_h->CopyToHoriz(exner);

    diagnostics(velx, velz, rho, rt, exner);

    // write output
    if(save) {
        step++;

        L2Vecs* l2Theta = new L2Vecs(geom->nk+1, topo, geom);
        diagTheta(rho_h->vz, rt_h->vz, l2Theta->vz);
        l2Theta->VertToHoriz();
        for(int kk = 0; kk < geom->nk+1; kk++) {
            sprintf(fieldname, "theta");
            geom->write2(l2Theta->vh[kk], fieldname, step, kk, false);
        }
        delete l2Theta;

        for(int kk = 0; kk < geom->nk; kk++) {
            curl(velx[kk], &wi, kk);

            sprintf(fieldname, "vorticity");
            geom->write0(wi, fieldname, step, kk);
            sprintf(fieldname, "velocity_h");
            geom->write1(velx[kk], fieldname, step, kk);
            sprintf(fieldname, "density");
            geom->write2(rho[kk], fieldname, step, kk, true);
            sprintf(fieldname, "rhoTheta");
            geom->write2(rt[kk], fieldname, step, kk, true);
            sprintf(fieldname, "exner");
            geom->write2(exner[kk], fieldname, step, kk, true);

            VecDestroy(&wi);
        }
        sprintf(fieldname, "velocity_z");
        geom->writeVertToHoriz(velz, fieldname, step, geom->nk-1);
    }

    VecDestroy(&xu);
    VecDestroy(&bu);
    DestroyHorizVecs(Fu_0, geom);
    delete Fp_0;
    delete Ft_0;
    DestroyHorizVecs(Fu_1, geom);
    delete Fp_1;
    delete Ft_1;
    DestroyHorizVecs(Fu_2, geom);
    delete Fp_2;
    delete Ft_2;
    DestroyHorizVecs(velx_0, geom);
    delete velz_0;
    delete rho_0;
    delete rt_0;
    delete exner_0;
    DestroyHorizVecs(velx_1, geom);
    delete rho_1;
    delete rt_1;
    delete exner_1;
    DestroyHorizVecs(velx_2, geom);
    delete rho_2;
    delete rt_2;
    delete exner_2;
    delete rho_3;
    delete rt_3;
    delete exner_3;
    delete velz_h;
    delete rho_h;
    delete rt_h;
    delete exner_h;
    delete F_rho_h;
    delete F_rt_h;
}

// compute the vorticity components dudz, dvdz
void Euler::HorizVort(Vec* velx) {
    int ii, size;
    Vec* Mu = new Vec[geom->nk];
    Vec  du;
    PC pc;
    KSP ksp1_t;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    KSPCreate(MPI_COMM_WORLD, &ksp1_t);
    KSPSetOperators(ksp1_t, M1t->M, M1t->M);
    KSPSetTolerances(ksp1_t, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1_t, KSPGMRES);
    KSPGetPC(ksp1_t, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1_t, "ksp1_t_");
    KSPSetFromOptions(ksp1_t);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu[ii]);
        MatMult(M1->M, velx[ii], Mu[ii]);
    }

    for(ii = 0; ii < geom->nk-1; ii++) {
        VecZeroEntries(du);
        VecAXPY(du, +1.0, Mu[ii+1]);
        VecAXPY(du, -1.0, Mu[ii+0]);
        KSPSolve(ksp1_t, du, uz[ii]);
        VecScatterBegin(topo->gtol_1, uz[ii], uzl[ii], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, uz[ii], uzl[ii], INSERT_VALUES, SCATTER_FORWARD);
    }

    VecDestroy(&du);
    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&Mu[ii]);
    }
    delete[] Mu;
    KSPDestroy(&ksp1_t);
}

// compute the contribution of the vorticity vector to the vertical momentum equation
void Euler::AssembleVertMomVort(L2Vecs* velz) {
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
        MatMult(M2->Mo, velz->vh[kk], tmp);
        MatMult(EtoF->E12, tmp, tmp1);
        KSPSolve(ksp1o, tmp1, dwdx); // horizontal gradient of vertical velocity

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
