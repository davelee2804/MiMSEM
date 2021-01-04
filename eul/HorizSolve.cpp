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

#define RAD_EARTH 6371220.0
#define RD 287.0
#define CV 717.5
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
    m0h = new Phvec(topo, geom, node);

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

    m1 = new Uvec(topo, geom, node, edge);
    m2 = new Wvec(topo, geom, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

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

    Fk = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fk[ii]);
    }

    // for the implicit solve
    MatCreate(MPI_COMM_WORLD, &M1_inv);
    MatSetSizes(M1_inv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M1_inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M1_inv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatZeroEntries(M1_inv);

    T = new Whmat(topo, geom, edge);
    M2inv = new WmatInv(topo, geom, edge);
    M2_rho_inv = new WhmatInv(topo, geom, edge);
    N2_rt = new N_rt_Inv(topo, geom, edge);
    N2_pi_inv = new N_rt_Inv(topo, geom, edge);
    G_rt = NULL;
    DIV = GRAD = NULL;
    kspColA2 = NULL;
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double HorizSolve::viscosity() {
    double ae = 4.0*M_PI*RAD_EARTH*RAD_EARTH;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

//del4 *= 2.0;
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

HorizSolve::~HorizSolve() {
    int ii;

    KSPDestroy(&ksp1);

    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&fg[ii]);
        VecDestroy(&fl[ii]);
        VecDestroy(&Fk[ii]);
    }
    delete[] fg;
    delete[] fl;
    delete[] Fk;

    delete m0;
    delete m0h;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;
    delete M1t;
    delete Rh;

    delete m1;
    delete m2;

    delete M2inv;
    delete M2_rho_inv;
    if(kspColA2) KSPDestroy(&kspColA2);
    MatDestroy(&M1_inv);
    //if(DIV)  MatDestroy(&DIV);
    //if(GRAD) MatDestroy(&GRAD);

    delete edge;
    delete node;
    delete quad;
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
    //m2->assemble(lev, SCALE, true, phi);
    //MatMult(EtoF->E12, m2->vg, dMphi);
    KSPSolve(ksp1, dMphi, *u);

    VecDestroy(&Mphi);
    VecDestroy(&dMphi);
}

/*
Take the weak form curl of a 1 form vector field as a 1 form vector field
*/
void HorizSolve::curl(bool assemble, Vec u, Vec* w, int lev, bool add_f, Vec ul) {
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
    //m1->assemble(lev, SCALE, true, ul);
    //MatMult(NtoE->E01, m1->vg, dMu);
    VecPointwiseDivide(*w, dMu, m0->vg);

    // add the coliolis term
    if(add_f) {
        VecAYPX(*w, 1.0, fg[lev]);
    }
    VecDestroy(&Mu);
    VecDestroy(&dMu);
}

void HorizSolve::laplacian(bool assemble, Vec ui, Vec* ddu, int lev, Vec ul) {
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
    curl(assemble, ui, &Cu, lev, false, ul);

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
    VecZeroEntries(m1->vl);
    VecZeroEntries(m1->vg);
    m1->assemble_hu(level, SCALE, u1l, h1l, false, 1.0/3.0);
    m1->assemble_hu(level, SCALE, u1l, h2l, false, 1.0/6.0);
    m1->assemble_hu(level, SCALE, u2l, h1l, false, 1.0/6.0);
    m1->assemble_hu(level, SCALE, u2l, h2l, false, 1.0/3.0);
    VecScatterBegin(topo->gtol_1, m1->vl, m1->vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, m1->vl, m1->vg, ADD_VALUES, SCATTER_REVERSE);
    VecAXPY(hu, 1.0, m1->vg);

    // solve the linear system
    M1->assemble(level, SCALE, true);
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
    Vec _G, tmp1, rho_dTheta_1, rho_dTheta_2, tmp2, dTheta, d3Theta;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rho_dTheta_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rho_dTheta_2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp2);

    for(int kk = 0; kk < geom->nk; kk++) {
        diagnose_fluxes(kk, u1[kk], u2[kk], h1l[kk], h2l[kk], theta->vh, Fk[kk], _G, u1l[kk], u2l[kk]);

        if(do_temp_visc) {
            VecZeroEntries(tmp2);
            VecAXPY(tmp2, 0.5, theta->vh[kk+0]);
            VecAXPY(tmp2, 0.5, theta->vh[kk+1]);
            M2->assemble(kk, SCALE, false);
            grad(false, tmp2, &dTheta, kk);

            VecZeroEntries(tmp2);
            VecAXPY(tmp2, 0.5, h1l[kk]);
            VecAXPY(tmp2, 0.5, h2l[kk]);
            F->assemble(tmp2, kk, true, SCALE);
            MatMult(F->M, dTheta, rho_dTheta_1);

            KSPSolve(ksp1, rho_dTheta_1, rho_dTheta_2);
            MatMult(EtoF->E21, rho_dTheta_2, tmp2);

            M2->assemble(kk, SCALE, true);
            grad(false, tmp2, &d3Theta, kk);

            VecAXPY(_G, del2*del2, d3Theta);

            VecDestroy(&d3Theta);
            VecDestroy(&dTheta);
        }
        MatMult(EtoF->E21, Fk[kk], dF->vh[kk]);
        MatMult(EtoF->E21, _G, dG->vh[kk]);
    }
    dF->HorizToVert();
    dG->HorizToVert();

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
    K->assemble(u1l, level, SCALE);

    MatMult(K->M, u1, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    MatMult(K->M, u2, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    K->assemble(u2l, level, SCALE);

    MatMult(K->M, u2, b);
    VecAXPY(*Phi, 1.0/3.0, b);
/*
    m2->assemble_K(level, SCALE, u1l, u1l);
    VecAXPY(*Phi, 1.0/3.0, m2->vg);
    m2->assemble_K(level, SCALE, u1l, u2l);
    VecAXPY(*Phi, 1.0/3.0, m2->vg);
    m2->assemble_K(level, SCALE, u2l, u2l);
    VecAXPY(*Phi, 1.0/3.0, m2->vg);
*/
    VecDestroy(&b);
}

void HorizSolve::diagnose_q(int level, bool do_assemble, Vec rho, Vec vel, Vec* qi, Vec ul) {
    Vec rhs, tmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, qi);

    //if(do_assemble) M1->assemble(level, SCALE, true);
    //MatMult(M1->M, vel, tmp1);
    //MatMult(NtoE->E01, tmp1, rhs);
    m1->assemble(level, SCALE, true, ul);
    MatMult(NtoE->E01, m1->vg, rhs);

    if(do_assemble) m0->assemble(level, SCALE);
    VecPointwiseMult(tmp, m0->vg, fg[level]);
    VecAXPY(rhs, 1.0, tmp);

    m0h->assemble(rho, level, SCALE);
    VecPointwiseDivide(*qi, rhs, m0h->vg);

    VecDestroy(&rhs);
    VecDestroy(&tmp);
}

void HorizSolve::momentum_rhs(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                              Vec velx1, Vec velx2, Vec uil, Vec ujl, Vec rho1, Vec rho2, Vec fu, Vec* Fz)
{
    double k2i_l;
    Vec Phi, dPi, utmp, d2u, d4u;
    Vec theta_h, dp, dudz_h, velz_h;
    Vec qi, qj, qh, ql;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &qh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &velz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
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
    diagnose_Phi(level, velx1, velx2, uil, ujl, &Phi);
    grad(false, Pi, &dPi, level);

    MatMult(EtoF->E12, Phi, fu);

    diagnose_q(level, false, rho1, velx1, &qi, uil);
    diagnose_q(level, false, rho2, velx2, &qj, ujl);
    VecZeroEntries(qh);
    VecAXPY(qh, 0.5, qi);
    VecAXPY(qh, 0.5, qj);
    VecScatterBegin(topo->gtol_0, qh, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qh, ql, INSERT_VALUES, SCATTER_FORWARD);
    R->assemble(ql, level, SCALE);

    VecZeroEntries(dp);
/*
    F->assemble(rho1, level, true, SCALE);
    MatMult(F->M, velx1, utmp);
    VecAXPY(dp, 1.0/3.0, utmp);
    MatMult(F->M, velx2, utmp);
    VecAXPY(dp, 1.0/6.0, utmp);
    F->assemble(rho2, level, true, SCALE);
    MatMult(F->M, velx1, utmp);
    VecAXPY(dp, 1.0/6.0, utmp);
    MatMult(F->M, velx2, utmp);
    VecAXPY(dp, 1.0/3.0, utmp);
*/
    VecZeroEntries(m1->vl);
    VecZeroEntries(m1->vg);
    m1->assemble_hu(level, SCALE, uil, rho1, false, 1.0/3.0);
    m1->assemble_hu(level, SCALE, ujl, rho1, false, 1.0/6.0);
    m1->assemble_hu(level, SCALE, uil, rho2, false, 1.0/6.0);
    m1->assemble_hu(level, SCALE, ujl, rho2, false, 1.0/3.0);
    VecScatterBegin(topo->gtol_1, m1->vl, m1->vg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, m1->vl, m1->vg, ADD_VALUES, SCATTER_REVERSE);
    VecAXPY(dp, 1.0, m1->vg);

    KSPSolve(ksp1, dp, utmp);
    MatMult(R->M, utmp, dp);
    VecAXPY(fu, 1.0, dp);

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

        Rh->assemble(dudz_h, SCALE);
        if(Fz) {
            MatMult(Rh->M, Fz[level-1], dp);
        } else {
            VecZeroEntries(velz_h);
            VecAXPY(velz_h, 0.5, velz1[level-1]);
            VecAXPY(velz_h, 0.5, velz2[level-1]);

            MatMult(Rh->M, velz_h, dp);
        }
        VecAXPY(fu, 0.5, dp);
    }
    if(level < geom->nk-1) {
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, dudz1[level+0]);
        VecAXPY(dudz_h, 0.5, dudz2[level+0]);

        Rh->assemble(dudz_h, SCALE);
        if(Fz) {
            MatMult(Rh->M, Fz[level+0], dp);
        } else {
            VecZeroEntries(velz_h);
            VecAXPY(velz_h, 0.5, velz1[level+0]);
            VecAXPY(velz_h, 0.5, velz2[level+0]);

            MatMult(Rh->M, velz_h, dp);
        }
        VecAXPY(fu, 0.5, dp);
    }

    if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, velx1);
        VecAXPY(utmp, 0.5, velx2);
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, uil);
        VecAXPY(dudz_h, 0.5, ujl);
        laplacian(false, utmp, &d2u, level, dudz_h);
        laplacian(false, d2u, &d4u, level, dudz_h);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, 1.0, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    // clean up
    VecDestroy(&utmp);
    VecDestroy(&Phi);
    VecDestroy(&dPi);
    VecDestroy(&theta_h);
    VecDestroy(&dp);
    VecDestroy(&dudz_h);
    VecDestroy(&velz_h);
    VecDestroy(&qi);
    VecDestroy(&qj);
    VecDestroy(&qh);
    VecDestroy(&ql);
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
    velx_j->CopyFrom(velx);
    velx_i->UpdateLocal();
    for(lev = 0; lev < geom->nk; lev++) VecCopy(velx_i->vl[lev], velx_j->vl[lev]);

    // diagnose the potential temperature
    rho_i->HorizToVert();
    rt_i->HorizToVert();
    diagTheta2(rho_i->vz, rt_i->vz, theta_i);
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();

    // diagnose the vorticity terms
    diagHorizVort(velx_i->vh, dudz_i->vh);
    dudz_j->CopyFrom(dudz_i->vh);
    dudz_i->UpdateLocal();
    for(lev = 0; lev < geom->nk; lev++) VecCopy(dudz_i->vl[lev], dudz_j->vl[lev]);

    rho_j->HorizToVert();
    rt_j->HorizToVert();

    do {
        max_norm_u = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        // exner pressure residual
        exner_j->HorizToVert();
        for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            vo->Assemble_EOS_Residual(ii%topo->nElsX, ii/topo->nElsX, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
        }
        F_exner->VertToHoriz();

        for(lev = 0; lev < geom->nk; lev++) {
            // velocity residual
            assemble_residual(lev, theta_h->vh, dudz_i->vl, dudz_j->vl, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
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
            if(do_visc) {
                M1->assemble(lev, SCALE, true);
                VecZeroEntries(dF);
                VecAXPY(dF, 0.5, theta_h->vh[lev+0]);
                VecAXPY(dF, 0.5, theta_h->vh[lev+1]);

                grad(false, dF, &dtheta, lev);
                F->assemble(rho_j->vh[lev], lev, true, SCALE);
                MatMult(F->M, dtheta, u_tmp_1);
                VecDestroy(&dtheta);

                KSPSolve(ksp1, u_tmp_1, u_tmp_2);
                MatMult(EtoF->E21, u_tmp_2, dG);

                grad(false, dG, &dtheta, lev);
                MatMult(EtoF->E21, dtheta, dG);
                VecDestroy(&dtheta);
                MatMult(M2->M, dG, dF);
                VecAXPY(frt, dt*del2*del2, dF);
            }

            // delta updates  - velx is a global vector, while theta and exner are local vectors
            solve_schur_level(lev, theta_h->vh, velx_j->vl[lev], velx_j->vh[lev], rho_j->vh[lev], rt_j->vh[lev], exner_j->vh[lev], 
                              fu, frho, frt, F_exner->vh[lev], du, drho, drt, dexner, grad_pi);

            VecAXPY(velx_j->vh[lev], 1.0, du);
            VecAXPY(rho_j->vh[lev], 1.0, drho);
            VecAXPY(rt_j->vh[lev], 1.0, drt);
            VecAXPY(exner_j->vh[lev], 1.0, dexner);
            velx_j->UpdateLocal();

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

        if(max_norm_exner < 1.0e-12 && max_norm_u < 1.0e-12 && max_norm_rho < 1.0e-12 && max_norm_rt < 1.0e-12) done = true;
        if(!rank) cout << itt << ":\t|d_exner|/|exner|: "  << max_norm_exner << 
                                  "\t|d_u|/|u|: "          << max_norm_u     <<
                                  "\t|d_rho|/|rho|: "      << max_norm_rho   <<
                                  "\t|d_rt|/|rt|: "        << max_norm_rt    << endl;

        diagHorizVort(velx_j->vh, dudz_j->vh);
        dudz_j->UpdateLocal();

        // diagnose the potential temperature (at the half step)
        rho_j->HorizToVert();
        rt_j->HorizToVert();
        diagTheta2(rho_j->vz, rt_j->vz, theta_h);
        for(lev = 0; lev < geom->nk; lev++) {
            VecScale(theta_h->vh[lev], 0.5);
            VecAXPY(theta_h->vh[lev], 0.5, theta_i->vh[lev]);

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

void HorizSolve::solve_schur_level(int lev, Vec* theta, Vec velx_l, Vec velx_g, Vec rho, Vec rt, Vec pi, 
                                   Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi, Vec dpil)
{
    MatReuse reuse = (!G_rt) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    Vec wl, wg, theta_k, ones, diag, tmp_h, tmp_u;
    PC pc;

    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE, true);
    M2->assemble(lev, SCALE, true);
    M2inv->assemble(lev, SCALE);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_k);
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
    curl(false, velx_g, &wg, lev, true, velx_l);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecDestroy(&wg);
    R->assemble(wl, lev, SCALE);
    VecDestroy(&wl);
    MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);

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
    T->assemble(rt, lev, SCALE, true);
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
    coriolisMatInv(R->M, &M_u_inv, reuse);

    // assemble the secondary operators
    MatMatMult(Q_rt_rho, M2inv->M, reuse, PETSC_DEFAULT, &Q_rt_rho_M_rho_inv);
//    if(!DIV) {
        MatProductCreate(Q_rt_rho_M_rho_inv, D_rho, PETSC_NULL, &DIV);
        MatProductSetType(DIV, MATPRODUCT_AB);
        MatProductSetAlgorithm(DIV, "default");
        MatProductSetFill(DIV, PETSC_DEFAULT);
        MatProductSetFromOptions(DIV);
        MatProductSymbolic(DIV);
//    }
    MatMatMult(Q_rt_rho_M_rho_inv, D_rho, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DIV);
    MatAYPX(DIV, -1.0, D_rt, DIFFERENT_NONZERO_PATTERN);

    MatMatMult(G_pi, N2_pi_inv->M, reuse, PETSC_DEFAULT, &G_pi_C_pi_inv);
//    if(!GRAD) {
        MatProductCreate(G_pi_C_pi_inv, N2_rt->M, PETSC_NULL, &GRAD);
        MatProductSetType(GRAD, MATPRODUCT_AB);
        MatProductSetAlgorithm(GRAD, "default");
        MatProductSetFill(GRAD, PETSC_DEFAULT);
        MatProductSetFromOptions(GRAD);
        MatProductSymbolic(GRAD);
//    }
    MatMatMult(G_pi_C_pi_inv, N2_rt->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &GRAD);
    MatAYPX(GRAD, -1.0, G_rt, DIFFERENT_NONZERO_PATTERN);

    MatMatMult(DIV, M_u_inv, reuse, PETSC_DEFAULT, &D_M_u_inv);
    MatMatMult(D_M_u_inv, GRAD, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &_PCx);
    MatAYPX(_PCx, -1.0, M2->M, DIFFERENT_NONZERO_PATTERN);

    // assign the linear solvers
    KSPCreate(MPI_COMM_WORLD, &ksp_rt);
    KSPSetOperators(ksp_rt, _PCx, _PCx);
    KSPSetTolerances(ksp_rt, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_rt, KSPGMRES);
    KSPGetPC(ksp_rt, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp_rt, "ksp_rt_");
    KSPSetFromOptions(ksp_rt);

    KSPCreate(MPI_COMM_WORLD, &ksp_u);
    KSPSetOperators(ksp_u, R->M, R->M);
    KSPSetTolerances(ksp_u, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_u, KSPGMRES);
    KSPGetPC(ksp_u, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp_u, "ksp_u_");
    KSPSetFromOptions(ksp_u);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp_u);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp_h);

    // update the residuals
    MatMult(G_pi_C_pi_inv, F_pi, tmp_u);
    VecAYPX(F_u, -1.0, tmp_u);                  // F_{u}'
    MatMult(Q_rt_rho_M_rho_inv, F_rho, tmp_h);
    VecAYPX(F_rt, -1.0, tmp_h);                 // F_{rt}'

    MatMult(D_M_u_inv, F_u, tmp_h);
    VecAXPY(tmp_h, -1.0, F_rt);

    KSPSolve(ksp_rt, tmp_h, d_rt);

    // back substitute
    MatMult(N2_rt->M, d_rt, tmp_h);
    VecAXPY(F_pi, 1.0, tmp_h);
    MatMult(N2_pi_inv->M, F_pi, d_pi);
    VecScale(d_pi, -1.0);

    MatMult(GRAD, d_rt, tmp_u);
    VecAXPY(tmp_u, +1.0, F_u);
    VecScale(tmp_u, -1.0);
    KSPSolve(ksp_u, tmp_u, d_u);

    MatMult(D_rho, d_u, tmp_h);
    VecAXPY(tmp_h, +1.0, F_rho);
    MatMult(M2inv->M, tmp_h, d_rho);
    VecScale(d_rho, -1.0);

    VecDestroy(&theta_k);
    VecDestroy(&tmp_h);
    VecDestroy(&tmp_u);
    MatDestroy(&_PCx);
    KSPDestroy(&ksp_rt);
    KSPDestroy(&ksp_u);
MatDestroy(&DIV);
MatDestroy(&GRAD);
}

void HorizSolve::coriolisMatInv(Mat A, Mat* Ainv, MatReuse reuse) {
    int mi, mf, ci, nCols1, nCols2;
    const int *cols1, *cols2;
    const double *vals1;
    const double *vals2;
    double D[2][2], Dinv[2][2], detInv;
    double valsInv[4];
    int rows[2];

    D[0][0] = D[0][1] = D[1][0] = D[1][1] = 0.0;

    if(reuse == MAT_INITIAL_MATRIX) {
        MatCreate(MPI_COMM_WORLD, Ainv);
        MatSetSizes(*Ainv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
        MatSetType(*Ainv, MATMPIAIJ);
        MatMPIAIJSetPreallocation(*Ainv, 2, PETSC_NULL, 2, PETSC_NULL);
    }
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

void HorizSolve::assemble_residual(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                                Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec fu, Vec _F, Vec _G, Vec uil, Vec ujl, Vec grad_pi)
{
    Vec Phi, dPi, wxz, utmp, d2u, d4u;
    Vec theta_h, dp, dudz_h, velz_h;
    Vec qi, qj, qh, ql;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &wxz);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &velz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dudz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &qh);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);

    m0->assemble(level, SCALE);
    M1->assemble(level, SCALE, true);
    M2->assemble(level, SCALE, true);

    // assume theta is 0.5*(theta_i + theta_j)
    VecZeroEntries(theta_h);
    VecAXPY(theta_h, 0.5, theta[level+0]);
    VecAXPY(theta_h, 0.5, theta[level+1]);

    VecZeroEntries(fu);

    // assemble in the skew-symmetric parts of the vector
    diagnose_fluxes(level, velx1, velx2, rho1, rho2, theta, _F, _G, uil, ujl);

    diagnose_Phi(level, velx1, velx2, uil, ujl, &Phi);
    grad(false, Pi, &dPi, level);
    diagnose_q(level, false, rho1, velx1, &qi, uil);
    diagnose_q(level, false, rho2, velx2, &qj, ujl);
    VecZeroEntries(qh);
    VecAXPY(qh, 0.5, qi);
    VecAXPY(qh, 0.5, qj);
    VecScatterBegin(topo->gtol_0, qh, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qh, ql, INSERT_VALUES, SCATTER_FORWARD);
    R->assemble(ql, level, SCALE);
    MatMult(R->M, _F, dp);

    MatMult(EtoF->E12, Phi, fu);
    VecAXPY(fu, 1.0, dp);

    // add the pressure gradient force
    F->assemble(theta_h, level, false, SCALE);
    MatMult(F->M, dPi, dp);
    VecAXPY(fu, 1.0, dp);

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
    MatMult(M1->M, velx1, utmp);
    VecAXPY(fu, -1.0, utmp);

    if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, velx1);
        VecAXPY(utmp, 0.5, velx2);
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, uil);
        VecAXPY(dudz_h, 0.5, ujl);
        laplacian(false, utmp, &d2u, level, dudz_h);
        laplacian(false, d2u, &d4u, level, dudz_h);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    VecScatterBegin(topo->gtol_1, dPi, grad_pi, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dPi, grad_pi, INSERT_VALUES, SCATTER_FORWARD);

    // clean up
    VecDestroy(&utmp);
    VecDestroy(&Phi);
    VecDestroy(&dPi);
    VecDestroy(&theta_h);
    VecDestroy(&dp);
    VecDestroy(&wxz);
    VecDestroy(&dudz_h);
    VecDestroy(&velz_h);
    VecDestroy(&qi);
    VecDestroy(&qj);
    VecDestroy(&qh);
    VecDestroy(&ql);
}

void HorizSolve::diagTheta2(Vec* rho, Vec* rt, L2Vecs* theta) {
    int ex, ey, n2, ei;
    Vec frt;
    PC pc;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &frt);

    if(!kspColA2) {
        KSPCreate(MPI_COMM_SELF, &kspColA2);
        KSPSetOperators(kspColA2, vo->VA2, vo->VA2);
        KSPGetPC(kspColA2, &pc);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(kspColA2, "kspColA2_");
        KSPSetFromOptions(kspColA2);
    }

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            vo->AssembleLinCon2(ex, ey, vo->VAB2);
            MatMult(vo->VAB2, rt[ei], frt);

            vo->AssembleLinearWithRho2(ex, ey, rho[ei], vo->VA2);
            KSPSolve(kspColA2, frt, theta->vz[ei]);
        }
    }
    VecDestroy(&frt);
    theta->VertToHoriz();
}

double HorizSolve::MaxNorm(Vec dx, Vec x, double max_norm) {
    double norm_dx, norm_x, new_max_norm;

    VecNorm(dx, NORM_2, &norm_dx);
    VecNorm(x, NORM_2, &norm_x);
    new_max_norm = (norm_dx/norm_x > max_norm) ? norm_dx/norm_x : max_norm;
    return new_max_norm;
}
