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
#define OMEGA 7.29212e-5
#define SCALE 1.0e+8

using namespace std;

HorizSolve::HorizSolve(Topo* _topo, Geom* _geom) {
    int ii;
    PC pc;

    topo = _topo;
    geom = _geom;

    do_visc = true;
    do_temp_visc = false;
    del2 = viscosity();
    step = 0;

    quad = new GaussLobatto(geom->quad->n);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // 0 form lumped mass matrix (vector)
    M0 = new Pmat(topo, geom, node);

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

    // initialize the 0 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp0);
    KSPSetOperators(ksp0, M0->M, M0->M);
    KSPSetTolerances(ksp0, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp0, KSPGMRES);
    KSPGetPC(ksp0, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp0, "ksp0_");
    KSPSetFromOptions(ksp0);

    // coriolis vector (projected onto 0 forms)
    coriolis();

    Fk = new Vec[geom->nk];
    Gk = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fk[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Gk[ii]);
    }

    M2h_mat = new Whmat(topo, geom, edge);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double HorizSolve::viscosity() {
    double ae = 4.0*M_PI*RAD_EARTH*RAD_EARTH;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

//del4 *= 4.0;
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
    VecCreateSeq(MPI_COMM_SELF, geom->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < geom->n0; ii++) {
        fArray[ii] = 2.0*OMEGA*sin(geom->s[ii][1]);
    }
    VecRestoreArray(fxl, &fArray);

    // scatter array to global vector
    VecScatterBegin(geom->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  geom->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);

    // project vector onto 0 forms
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PtQfxg);
    VecZeroEntries(PtQfxg);
    MatMult(PtQ->M, fxg, PtQfxg);
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg[kk]);
	M0->assemble(kk, 1.0);
	KSPSolve(ksp0, PtQfxg, fg[kk]);
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
    KSPDestroy(&ksp1);
    KSPDestroy(&ksp0);

    for(int ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&fg[ii]);
        VecDestroy(&fl[ii]);
        VecDestroy(&Fk[ii]);
        VecDestroy(&Gk[ii]);
    }
    delete[] fg;
    delete[] fl;
    delete[] Fk;
    delete[] Gk;

    delete M0;
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

    delete M2h_mat;

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
void HorizSolve::curl(bool assemble, Vec u, Vec* w, int lev, bool add_f) {
    Vec Mu, dMu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &dMu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu);

    if(assemble) {
        M0->assemble(lev, SCALE);
        M1->assemble(lev, SCALE, true);
    }
    MatMult(M1->M, u, Mu);
    MatMult(NtoE->E01, Mu, dMu);
    KSPSolve(ksp0, dMu, *w);

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

void HorizSolve::diagnose_fluxes(int level, Vec u1, Vec u2, Vec h1l, Vec h2l, Vec* theta_l, Vec _F, Vec _G, Vec u1l, Vec u2l, bool theta_in_Wt) {
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
    if(theta_in_Wt) {
        VecZeroEntries(tmp2l);
        VecAXPY(tmp2l, 0.5, theta_l[level+0]);
        VecAXPY(tmp2l, 0.5, theta_l[level+1]);
        F->assemble(tmp2l, level, false, SCALE);
    } else {
        F->assemble(theta_l[level], level, true, SCALE);
    }
    MatMult(F->M, _F, hu);
    KSPSolve(ksp1, hu, _G);

    VecDestroy(&hu);
    VecDestroy(&b);
    VecDestroy(&tmp2l);
    VecDestroy(&tmp1l);
}

void HorizSolve::advection_rhs(Vec* u1, Vec* u2, Vec* h1l, Vec* h2l, L2Vecs* theta, L2Vecs* dF, L2Vecs* dG, Vec* u1l, Vec* u2l) {
    Vec tmp1, rho_dTheta_1, rho_dTheta_2, tmp2, dTheta, d3Theta;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rho_dTheta_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rho_dTheta_2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp2);

    for(int kk = 0; kk < geom->nk; kk++) {
        diagnose_fluxes(kk, u1[kk], u2[kk], h1l[kk], h2l[kk], theta->vh, Fk[kk], Gk[kk], u1l[kk], u2l[kk], true);

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

            VecAXPY(Gk[kk], del2*del2, d3Theta);

            VecDestroy(&d3Theta);
            VecDestroy(&dTheta);
        }
        MatMult(EtoF->E21, Fk[kk], dF->vh[kk]);
        MatMult(EtoF->E21, Gk[kk], dG->vh[kk]);
    }
    dF->HorizToVert();
    dG->HorizToVert();

    VecDestroy(&tmp1);
    VecDestroy(&rho_dTheta_1);
    VecDestroy(&rho_dTheta_2);
    VecDestroy(&tmp2);
}

// include the entropy conserving corrections for the temperature flux
// note that - the resulting RHS terms include the integral over the L2 test functions
//           - here \theta is in L2 for entropy conservation
void HorizSolve::advection_rhs_ec(Vec* u1, Vec* u2, Vec* h1l, Vec* h2l, Vec* theta, L2Vecs* dF, L2Vecs* dG, Vec* u1l, Vec* u2l) {
    Vec dTheta, dTheta_l, dFk, dGk;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dFk);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dGk);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dTheta_l);

    for(int kk = 0; kk < geom->nk; kk++) {
        diagnose_fluxes(kk, u1[kk], u2[kk], h1l[kk], h2l[kk], theta, Fk[kk], Gk[kk], u1l[kk], u2l[kk], false);

        M2->assemble(kk, SCALE, true);

        MatMult(EtoF->E21, Fk[kk], dFk);
	MatMult(M2->M, dFk, dF->vh[kk]);

        MatMult(EtoF->E21, Gk[kk], dGk);
	MatMult(M2->M, dGk, dG->vh[kk]);
	VecScale(dG->vh[kk], 0.5);

        M2h_mat->assemble(theta[kk], kk, SCALE, true);
	MatMult(M2h_mat->M, dFk, dGk);
	VecAXPY(dG->vh[kk], 0.5, dGk);

        grad(false, theta[kk], &dTheta, kk);
        VecScatterBegin(topo->gtol_1, dTheta, dTheta_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, dTheta, dTheta_l, INSERT_VALUES, SCATTER_FORWARD);
        VecDestroy(&dTheta);
	K->assemble(dTheta_l, kk, SCALE); // inc. 0.5 factor
        MatMult(K->M, Fk[kk], dGk);
	VecAXPY(dG->vh[kk], 1.0, dGk);
    }
    dF->HorizToVert();
    dG->HorizToVert();

    VecDestroy(&dFk);
    VecDestroy(&dGk);
    VecDestroy(&dTheta_l);
}

void HorizSolve::diagnose_Phi(int level, Vec u1, Vec u2, Vec u1l, Vec u2l, Vec* velz1, Vec* velz2, Vec* Phi) {
    Vec b, _velz1, _velz2;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Phi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &_velz1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &_velz2);
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

    // vertical terms
    VecZeroEntries(_velz1);
    VecZeroEntries(_velz2);
    if(level > 0) {
        VecAXPY(_velz1, 0.5, velz1[level-1]);
        VecAXPY(_velz2, 0.5, velz2[level-1]);
    }
    if(level < geom->nk-1) {
        VecAXPY(_velz1, 0.5, velz1[level+0]);
        VecAXPY(_velz2, 0.5, velz2[level+0]);
    }
    M2h_mat->assemble(_velz1, level, SCALE, false);
    MatMult(M2h_mat->M, _velz1, b);
    VecAXPY(*Phi, 1.0/6.0, b);
    MatMult(M2h_mat->M, _velz2, b);
    VecAXPY(*Phi, 1.0/6.0, b);
    M2h_mat->assemble(_velz2, level, SCALE, false);
    MatMult(M2h_mat->M, _velz2, b);
    VecAXPY(*Phi, 1.0/6.0, b);

    VecDestroy(&b);
    VecDestroy(&_velz1);
    VecDestroy(&_velz2);
}

void HorizSolve::diagnose_q(int level, Vec rho, Vec ul, Vec* qi) {
    Vec rhs, tmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, qi);

    m1->assemble(level, SCALE, true, ul);
    MatMult(NtoE->E01, m1->vg, rhs);

    M0->assemble(level, SCALE);
    MatMult(M0->M, fg[level], tmp);
    VecAXPY(rhs, 1.0, tmp);

    M0->assemble_h(level, SCALE, rho);
    KSPSolve(ksp0, rhs, *qi);

    VecDestroy(&rhs);
    VecDestroy(&tmp);
}

void HorizSolve::momentum_rhs(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                              Vec velx1, Vec velx2, Vec uil, Vec ujl, Vec rho1, Vec rho2, Vec fu, Vec Fx, Vec* Fz, Vec* dwdx1, Vec* dwdx2)
{
    double k2i_l;
    Vec Phi, dPi, utmp, d2u, d4u;
    Vec theta_h, dp, dudz_h, velz_h;
    Vec qh, ql;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &velz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dudz_h);

    M1->assemble(level, SCALE, true);
    M2->assemble(level, SCALE, true);

    // assume theta is 0.5*(theta_i + theta_j)
    VecZeroEntries(theta_h);
    VecAXPY(theta_h, 0.5, theta[level+0]);
    VecAXPY(theta_h, 0.5, theta[level+1]);

    VecZeroEntries(fu);

    // assemble in the skew-symmetric parts of the vector
    diagnose_Phi(level, velx1, velx2, uil, ujl, velz1, velz2, &Phi);
    grad(false, Pi, &dPi, level);

    MatMult(EtoF->E12, Phi, fu);

    VecZeroEntries(dudz_h);
    VecAXPY(dudz_h, 0.5, uil);
    VecAXPY(dudz_h, 0.5, ujl);
    VecZeroEntries(velz_h);
    VecAXPY(velz_h, 0.5, rho1);
    VecAXPY(velz_h, 0.5, rho2);
    diagnose_q(level, velz_h, dudz_h, &qh);
    VecScatterBegin(topo->gtol_0, qh, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qh, ql, INSERT_VALUES, SCATTER_FORWARD);
    R->assemble(ql, level, SCALE);
    if(!Fx) {
        VecZeroEntries(dp);
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
    } else {
        MatMult(R->M, Fx, dp);
    }
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
        if(dwdx1) {
            VecAXPY(dudz_h, -0.5, dwdx1[level-1]);
            VecAXPY(dudz_h, -0.5, dwdx2[level-1]);
        }

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
        if(dwdx1) {
            VecAXPY(dudz_h, -0.5, dwdx1[level+0]);
            VecAXPY(dudz_h, -0.5, dwdx2[level+0]);
        }

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
        M0->assemble(level, SCALE);
        laplacian(false, utmp, &d2u, level);
        laplacian(false, d2u, &d4u, level);
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
    VecDestroy(&qh);
    VecDestroy(&ql);
}

// entropy conserving - theta is in L2
void HorizSolve::momentum_rhs_ec(int level, Vec theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                              Vec velx1, Vec velx2, Vec uil, Vec ujl, Vec rho1, Vec rho2, Vec fu, Vec Fx, Vec* Fz, Vec* dwdx1, Vec* dwdx2)
{
    double k2i_l;
    Vec Phi, dPi, dTheta, utmp, d2u, d4u;
    Vec dp, dudz_h, velz_h, qh, ql;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &velz_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dudz_h);

    M1->assemble(level, SCALE, true);
    M2->assemble(level, SCALE, true);

    VecZeroEntries(fu);

    // assemble in the skew-symmetric parts of the vector
    diagnose_Phi(level, velx1, velx2, uil, ujl, velz1, velz2, &Phi);
    grad(false, Pi, &dPi, level);
    grad(false, theta, &dTheta, level);

    MatMult(EtoF->E12, Phi, fu);

    VecZeroEntries(dudz_h);
    VecAXPY(dudz_h, 0.5, uil);
    VecAXPY(dudz_h, 0.5, ujl);
    VecZeroEntries(velz_h);
    VecAXPY(velz_h, 0.5, rho1);
    VecAXPY(velz_h, 0.5, rho2);
    diagnose_q(level, velz_h, dudz_h, &qh);
    VecScatterBegin(topo->gtol_0, qh, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qh, ql, INSERT_VALUES, SCATTER_FORWARD);
    R->assemble(ql, level, SCALE);
    if(!Fx) {
        VecZeroEntries(dp);
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
    } else {
        MatMult(R->M, Fx, dp);
    }
    VecAXPY(fu, 1.0, dp);

    // add the pressure gradient force
    F->assemble(theta, level, true, SCALE);
    MatMult(F->M, dPi, dp);
    VecAXPY(fu, +0.5, dp);
    F->assemble(Pi, level, true, SCALE);
    MatMult(F->M, dTheta, dp);
    VecAXPY(fu, -0.5, dp);
    // this term is technically part of the potential
    M2h_mat->assemble(Pi, level, SCALE, true);
    MatMult(M2h_mat->M, theta, velz_h);
    MatMult(EtoF->E12, velz_h, dp);
    VecAXPY(fu, +0.5, dp);

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
        if(dwdx1) {
            VecAXPY(dudz_h, -0.5, dwdx1[level-1]);
            VecAXPY(dudz_h, -0.5, dwdx2[level-1]);
        }

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
        if(dwdx1) {
            VecAXPY(dudz_h, -0.5, dwdx1[level+0]);
            VecAXPY(dudz_h, -0.5, dwdx2[level+0]);
        }

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
        M0->assemble(level, SCALE);
        laplacian(false, utmp, &d2u, level);
        laplacian(false, d2u, &d4u, level);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, 1.0, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    // clean up
    VecDestroy(&utmp);
    VecDestroy(&Phi);
    VecDestroy(&dPi);
    VecDestroy(&dTheta);
    VecDestroy(&dp);
    VecDestroy(&dudz_h);
    VecDestroy(&velz_h);
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

void HorizSolve::diagVertVort(Vec* velz, Vec* rho, Vec* dwdx) {
    Vec rho_h, rhs, dwdx_g;
    PC pc;
    KSP ksp1_t;

    KSPCreate(MPI_COMM_WORLD, &ksp1_t);
    KSPSetOperators(ksp1_t, F->M, F->M);
    KSPSetTolerances(ksp1_t, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1_t, KSPGMRES);
    KSPGetPC(ksp1_t, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1_t, "ksp1_t_");
    KSPSetFromOptions(ksp1_t);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dwdx_g);

    M2->assemble(0, SCALE, true);
    for(int ii = 0; ii < geom->nk-1; ii++) {
        VecZeroEntries(rho_h);
        VecAXPY(rho_h, 0.5, rho[ii+0]);
        VecAXPY(rho_h, 0.5, rho[ii+1]);
        F->assemble(rho_h, 0, false, SCALE);

        MatMult(M2->M, velz[ii], rho_h);
        MatMult(EtoF->E12, rho_h, rhs);
        KSPSolve(ksp1_t, rhs, dwdx_g);

        VecScatterBegin(topo->gtol_1, dwdx_g, dwdx[ii], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, dwdx_g, dwdx[ii], INSERT_VALUES, SCATTER_FORWARD);
    }

    VecDestroy(&rho_h);
    VecDestroy(&rhs);
    VecDestroy(&dwdx_g);
    KSPDestroy(&ksp1_t);
}
