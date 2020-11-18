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

#define _LX (1000.0)
#define OMEGA 7.29212e-5
#define SCALE 1.0e+8

using namespace std;

HorizSolve::HorizSolve(Topo* _topo, Geom* _geom, double _dt) {
    int ii;
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

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp1);
    KSPSetOperators(ksp1, M1->M, M1->M);
    KSPSetTolerances(ksp1, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1, KSPGMRES);
    KSPGetPC(ksp1, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1, "ksp1_");
    KSPSetFromOptions(ksp1);

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp1o);
    KSPSetOperators(ksp1o, M1->Mo, M1->Mo);
    KSPSetTolerances(ksp1o, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1o, KSPGMRES);
    KSPGetPC(ksp1o, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1o, "ksp1o_");
    KSPSetFromOptions(ksp1o);

    Fk = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fk[ii]);
    }

    M1u = new Uvec(topo, geom, node, edge);
    M2h = new Wvec(topo, geom, edge);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double HorizSolve::viscosity() {
    double ae = _LX*_LX;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    del4 *= 2.0;

    return -sqrt(del4);
}

HorizSolve::~HorizSolve() {
    int ii;

    KSPDestroy(&ksp1);
    KSPDestroy(&ksp1o);

    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&Fk[ii]);
    }
    delete[] Fk;

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;
    delete M1t;
    delete Rh;

    delete M1u;
    delete M2h;

    delete edge;
    delete node;
    delete quad;
}

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void HorizSolve::grad(KSP _ksp, Mat _M2, Vec phi, Vec* u) {
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
void HorizSolve::curl(Vec u, Vec* w) {
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

void HorizSolve::laplacian(Vec ui, Vec* ddu) {
    Vec Du, Cu, RCu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &RCu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Du);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, ui, Du);

    // grad (weak form)
    grad(ksp1, M2->M, Du, ddu);

    /*** rotational component ***/
    // curl (weak form)
    curl(ui, &Cu);

    // rot (strong form)
    MatMult(NtoE->E10, Cu, RCu);

    // add rotational and divergent components
    VecAXPY(*ddu, +1.0, RCu);
    VecScale(*ddu, del2);

    VecDestroy(&Cu);
    VecDestroy(&RCu);
    VecDestroy(&Du);
}

void HorizSolve::diagnose_fluxes(int level, Vec u1, Vec u2, Vec h1l, Vec h2l, Vec* theta_l, Vec _F, Vec _G, Vec u1l, Vec u2l) {
    Vec hu, b, tmp2l, tmp1l;

    //VecCreateSeq(MPI_COMM_SELF, topo->n2, &tmp2l);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp2l);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &tmp1l);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);
    VecZeroEntries(_F);
    VecZeroEntries(_G);
    VecZeroEntries(hu);

    // assemble the nonlinear rhs mass matrix (note that hl is a local vector)
/*
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
*/
    M1u->assemble_hu(level, SCALE, true, u1l, h1l);
    VecAXPY(hu, 1.0/3.0, M1u->vg);
    M1u->assemble_hu(level, SCALE, true, u1l, h2l);
    VecAXPY(hu, 1.0/6.0, M1u->vg);
    M1u->assemble_hu(level, SCALE, true, u2l, h1l);
    VecAXPY(hu, 1.0/6.0, M1u->vg);
    M1u->assemble_hu(level, SCALE, true, u2l, h2l);
    VecAXPY(hu, 1.0/3.0, M1u->vg);

    // solve the linear system
    KSPSolve(ksp1, hu, _F);

    // diagnose the corresponding temperature flux
    VecZeroEntries(tmp2l);
    VecAXPY(tmp2l, 0.5, theta_l[level+0]);
    VecAXPY(tmp2l, 0.5, theta_l[level+1]);
    F->assemble(tmp2l, level, false, SCALE);
    MatMult(F->M, _F, hu);
    KSPSolve(ksp1, hu, _G);

    VecDestroy(&hu);
    VecDestroy(&b);
    VecDestroy(&tmp2l);
    VecDestroy(&tmp1l);
}

void HorizSolve::advection_rhs(Vec* u1, Vec* u2, Vec* h1l, Vec* h2l, L2Vecs* theta, L2Vecs* dF, L2Vecs* dG, Vec* u1l, Vec* u2l, bool do_temp_visc) {
    //Vec _F, _G, tmp1, rho_dTheta_1, rho_dTheta_2, tmp2, dTheta, d3Theta;
    Vec _G, tmp1, rho_dTheta_1, rho_dTheta_2, tmp2, dTheta, d3Theta;

    //VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rho_dTheta_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rho_dTheta_2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp2);

    for(int kk = 0; kk < geom->nk; kk++) {
        //diagnose_fluxes(kk, u1[kk], u2[kk], h1l[kk], h2l[kk], theta->vh, _F, _G, u1l[kk], u2l[kk]);
        diagnose_fluxes(kk, u1[kk], u2[kk], h1l[kk], h2l[kk], theta->vh, Fk[kk], _G, u1l[kk], u2l[kk]);

        if(do_temp_visc) {
            VecZeroEntries(tmp2);
            VecAXPY(tmp2, 0.5, theta->vh[kk+0]);
            VecAXPY(tmp2, 0.5, theta->vh[kk+1]);
            grad(ksp1o, M2->Mo, tmp2, &dTheta);

            //VecZeroEntries(tmp2l);
            //VecAXPY(tmp2l, 0.5, h1l[kk]);
            //VecAXPY(tmp2l, 0.5, h2l[kk]);
            //F->assemble(tmp2l, kk, true, SCALE);
            VecZeroEntries(tmp2);
            VecAXPY(tmp2, 0.5, h1l[kk]);
            VecAXPY(tmp2, 0.5, h2l[kk]);
            F->assemble(tmp2, kk, true, SCALE);
            MatMult(F->M, dTheta, rho_dTheta_1);

            KSPSolve(ksp1, rho_dTheta_1, rho_dTheta_2);
            MatMult(EtoF->E21, rho_dTheta_2, tmp2);

            grad(ksp1, M2->M, tmp2, &d3Theta);

            VecAXPY(_G, del2*del2, d3Theta);

            VecDestroy(&d3Theta);
            VecDestroy(&dTheta);
        }
        //MatMult(EtoF->E21, _F, dF->vh[kk]);
        MatMult(EtoF->E21, Fk[kk], dF->vh[kk]);
        MatMult(EtoF->E21, _G, dG->vh[kk]);
    }
    dF->HorizToVert();
    dG->HorizToVert();

    //VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&tmp1);
    VecDestroy(&rho_dTheta_1);
    VecDestroy(&rho_dTheta_2);
    VecDestroy(&tmp2);
}

void HorizSolve::diagnose_Phi(int level, Vec u1, Vec u2, Vec u1l, Vec u2l, Vec* Phi) {
    Vec b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Phi);
    VecZeroEntries(*Phi);

    // u^2 terms (0.5 factor incorportated into the matrix assembly)
/*
    K->assemble(u1l, level, SCALE);

    MatMult(K->M, u1, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    MatMult(K->M, u2, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    K->assemble(u2l, level, SCALE);

    MatMult(K->M, u2, b);
    VecAXPY(*Phi, 1.0/3.0, b);
*/
    M2h->assemble_K(level, SCALE, true, u1l, u1l);
    VecAXPY(*Phi, 1.0/3.0, M2h->vg);
    M2h->assemble_K(level, SCALE, true, u1l, u2l);
    VecAXPY(*Phi, 1.0/3.0, M2h->vg);
    M2h->assemble_K(level, SCALE, true, u2l, u2l);
    VecAXPY(*Phi, 1.0/3.0, M2h->vg);

    VecDestroy(&b);
}

void HorizSolve::diagnose_wxu(int level, Vec u1, Vec u2, Vec* wxu) {
    Vec w1, w2, wl, uh;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, wxu);

    // assume the vertex and volume mass matrices have already been assembled
    // TODO compute once for half step
    curl(u1, &w1);
    curl(u2, &w2);
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

void HorizSolve::momentum_rhs(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                              Vec velx1, Vec velx2, Vec uil, Vec ujl, Vec fu)
{
    double k2i_l;
    Vec Phi, dPi, wxu, wxz, utmp, d2u, d4u;
    Vec theta_h, dp, dudz_h, velz_h;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &wxz);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &velz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dudz_h);

    // assume theta is 0.5*(theta_i + theta_j)
    VecZeroEntries(theta_h);
    VecAXPY(theta_h, 0.5, theta[level+0]);
    VecAXPY(theta_h, 0.5, theta[level+1]);

    VecZeroEntries(fu);

    // assemble in the skew-symmetric parts of the vector
    diagnose_Phi(level, velx1, velx2, uil, ujl, &Phi);
    grad(ksp1, M2->M, Pi, &dPi);
    diagnose_wxu(level, velx1, velx2, &wxu);

    MatMult(EtoF->E12, Phi, fu);
    VecAXPY(fu, 1.0, wxu);

    // add the pressure gradient force
    F->assemble(theta_h, level, false, SCALE);
    MatMult(F->M, dPi, dp);
    VecAXPY(fu, 1.0, dp);

    // update the horizontal kinetic to internal energy exchange
    if(level == 0) k2i = 0.0;
    VecDot(Fk[level], dp, &k2i_l);
    k2i_l /= SCALE;
    k2i   += k2i_l;

    // second voritcity term
    if(level > 0) {
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, dudz1[level-1]);
        VecAXPY(dudz_h, 0.5, dudz2[level-1]);

        VecZeroEntries(velz_h);
        VecAXPY(velz_h, 0.5, velz1[level-1]);
        VecAXPY(velz_h, 0.5, velz2[level-1]);

        Rh->assemble(dudz_h, SCALE);
        MatMult(Rh->M, velz_h, dp);
        VecAXPY(fu, 0.5, dp);
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
        VecAXPY(fu, 0.5, dp);
    }

    if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, velx1);
        VecAXPY(utmp, 0.5, velx2);
        laplacian(utmp, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, 1.0, d2u);
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
        KSPSolve(ksp1_t, du, dudz[ii]);
    }

    VecDestroy(&du);
    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&Mu[ii]);
    }
    delete[] Mu;
    KSPDestroy(&ksp1_t);
}
