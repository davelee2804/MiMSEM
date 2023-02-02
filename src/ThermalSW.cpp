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
#include "ThermalSW.h"

#define RAD_EARTH 6371220.0
#define UP_SUPG 1
#define H_MEAN 1.0e+4
//#define ROS_ALPHA (1.0 + 0.5*sqrt(2.0))
#define ROS_ALPHA (0.5)
#define DO_THERMAL 1

using namespace std;

ThermalSW::ThermalSW(Topo* _topo, Geom* _geom) {
    PC pc;
    int ii, jj;
    int dof_proc;
    int* loc = new int[_topo->n1+2*_topo->n2];
    IS is_g, is_l;
    Vec xl, xg;

    topo = _topo;
    geom = _geom;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    omega = 7.292e-5;
    step = 0;
    first_step = true;
    del2 = viscosity();

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

    // adjoint differential operators (curl and grad)
    MatMatMult(NtoE->E01, M1->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E01M1);
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E12M2);

    // rotational operator
    R = new RotMat(topo, geom, node, edge);
    R_up = new RotMat_up(topo, geom, node, edge);

    M1h = new Uhmat(topo, geom, node, edge);
    M2h = new Whmat(topo, geom, edge);
    M0h = new Phmat(topo, geom, node);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    // variable buoyancy upwinding
    //M2tau = new W_tau_mat(topo, geom, edge);

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

    KSPCreate(MPI_COMM_WORLD, &ksp0);
    KSPSetOperators(ksp0, M0->M, M0->M);
    KSPSetTolerances(ksp0, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp0, KSPGMRES);
    KSPGetPC(ksp0, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp0, "ksp_0_");
    KSPSetFromOptions(ksp0);

    KSPCreate(MPI_COMM_WORLD, &ksp0h);
    KSPSetOperators(ksp0h, M0h->M, M0h->M);
    KSPSetTolerances(ksp0h, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp0h, KSPGMRES);
    KSPGetPC(ksp0h, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp0h, "ksp_0_");
    KSPSetFromOptions(ksp0h);

    // coriolis vector (projected onto 0 forms)
    coriolis();

    A = NULL;
    B = NULL;
    kspA = NULL;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &si);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &sj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_prev);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_prev);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &s_prev);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Phi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &T);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ds);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &wi);

    // create the [u,h,s] -> [x] vec scatter
    for(ii = 0; ii < topo->n1; ii++) {
        dof_proc = topo->loc1[ii] / topo->n1l;
        loc[ii] = dof_proc * (topo->n1l + 2*topo->n2l) + topo->loc1[ii] % topo->n1l;
    }
    for(ii = 0; ii < 2*topo->n2; ii++) {
        jj = ii + topo->n1;
        loc[jj] = rank*(topo->n1l + 2*topo->n2l) + ii + topo->n1l;
    }

    ISCreateStride(MPI_COMM_SELF, topo->n1+2*topo->n2, 0, 1, &is_l);
    ISCreateGeneral(MPI_COMM_WORLD, topo->n1+2*topo->n2, loc, PETSC_COPY_VALUES, &is_g);

    VecCreateSeq(MPI_COMM_SELF, topo->n1+2*topo->n2, &xl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+2*topo->n2l, topo->nDofs1G+2*topo->nDofs2G, &xg);

    VecScatterCreate(xg, is_g, xl, is_l, &gtol_x);

    delete[] loc;
    ISDestroy(&is_l);
    ISDestroy(&is_g);
    VecDestroy(&xl);
    VecDestroy(&xg);

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &uil);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ujl);

    KSPCreate(MPI_COMM_WORLD, &ksp1h);
    KSPSetOperators(ksp1h, M1h->M, M1h->M);
    KSPSetTolerances(ksp1h, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1h, KSPGMRES);
    KSPGetPC(ksp1h, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1h, "Fonh_");
    KSPSetFromOptions(ksp1h);

    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPSetOptionsPrefix(ksp2, "ksp2_");
    KSPSetFromOptions(ksp2);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double ThermalSW::viscosity() {
    double ae = 4.0*M_PI*RAD_EARTH*RAD_EARTH;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

void ThermalSW::grad(Vec phi, Vec* _u) {
    Vec Mphi, dMphi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, _u);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mphi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMphi);

    MatMult(M2->M, phi, Mphi);
    MatMult(EtoF->E12, Mphi, dMphi);
    KSPSolve(ksp, dMphi, *_u);

    VecDestroy(&Mphi);
    VecDestroy(&dMphi);
}

void ThermalSW::laplacian(Vec _u, Vec* ddu) {
    Vec Du, RCu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &RCu);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, _u, Du);
    // grad (weak form)
    grad(Du, ddu);

    /*** rotational component ***/
    // curl (weak form)
    curl(_u);
    // rot (strong form)
    MatMult(NtoE->E10, wi, RCu);

    // add rotational and divergent components
    VecAXPY(*ddu, +1.0, RCu);
    VecScale(*ddu, del2);

    VecDestroy(&Du);
    VecDestroy(&RCu);
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void ThermalSW::coriolis() {
    int ii, size;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fxl, fxg, PtQfxg;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // initialise the coriolis vector (local and global)
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg);

    // evaluate the coriolis term at nodes
    VecCreateSeq(MPI_COMM_SELF, geom->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < geom->n0; ii++) {
        fArray[ii] = 2.0*omega*sin(geom->s[ii][1]);
    }
    VecRestoreArray(fxl, &fArray);

    // scatter array to global vector
    VecScatterBegin(geom->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  geom->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);

    // project vector onto 0 forms
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PtQfxg);
    VecZeroEntries(PtQfxg);
    MatMult(PtQ->M, fxg, PtQfxg);
    // diagonal mass matrix as vector
    KSPSolve(ksp0, PtQfxg, fg);
    
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    VecScatterBegin(topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);

    delete PtQ;
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

// derive vorticity (global vector) as \omega = curl u
void ThermalSW::curl(Vec u) {
    Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);

    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    // diagonal mass matrix as vector
    KSPSolve(ksp0, du, wi);

    VecDestroy(&du);
}

// dH/du = hu = F
void ThermalSW::diagnose_F() {
    Vec hu, b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);
    VecZeroEntries(F);
    VecZeroEntries(hu);

    // assemble the nonlinear rhs matrix
    M1h->assemble(hi);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/3.0, b);
    //VecAXPY(hu, 1.0/2.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/6.0, b);

    M1h->assemble(hj);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/6.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/3.0, b);
    //VecAXPY(hu, 1.0/2.0, b);

    // solve the linear system
    KSPSolve(ksp, hu, F);

    VecDestroy(&hu);
    VecDestroy(&b);
}

void ThermalSW::diagnose_F_up(double _dt) {
    Vec hu, b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);
    VecZeroEntries(F);
    VecZeroEntries(hu);

    // assemble the nonlinear rhs mass matrix (note that hl is a local vector)
/*
    M1->assemble_up(_dt, ul, hi);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/3.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/6.0, b);

    M1->assemble_up(_dt, ul, hj);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/6.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/3.0, b);

    // solve the linear system
    M1->assemble_up(_dt, ul, NULL);
    KSPSolve(ksp, hu, *F);
    M1->assemble();
*/
    M1->assemble_up(_dt, uil, hi);
    MatMult(M1h->M, ui, b);
    M1->assemble_up(_dt, uil, NULL);
    KSPSolve(ksp, b, hu);
    VecAXPY(F, 0.5, hu);

    M1->assemble_up(_dt, ujl, hj);
    MatMult(M1h->M, uj, b);
    M1->assemble_up(_dt, ujl, NULL);
    KSPSolve(ksp, b, hu);
    VecAXPY(F, 0.5, hu);

    M1->assemble();

    VecDestroy(&hu);
    VecDestroy(&b);
}

// dH/dh = (1/2)u^2 + gh = \Phi
// dH/dh = (1/2)u^2 + gh = \Phi
// note: \Phi is in integral form here
//          \int_{\Omega} \gamma_h,\Phi_h d\Omega
void ThermalSW::diagnose_Phi() {
    Vec b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecZeroEntries(Phi);

    // u^2 terms (0.5 factor incorportated into the matrix assembly)
    K->assemble(uil);

    MatMult(K->M, ui, b);
    VecAXPY(Phi, 1.0/3.0, b);
    //VecAXPY(Phi, 1.0/2.0, b);

    MatMult(K->M, uj, b);
    VecAXPY(Phi, 1.0/3.0, b);

    K->assemble(ujl);

    MatMult(K->M, uj, b);
    VecAXPY(Phi, 1.0/3.0, b);
    //VecAXPY(Phi, 1.0/2.0, b);

    // sh terms
#ifdef DO_THERMAL
    M2h->assemble(hi);

    MatMult(M2h->M, si, b);
    VecAXPY(Phi, 1.0/3.0, b);
    //VecAXPY(Phi, 1.0/2.0, b);

    MatMult(M2h->M, sj, b);
    VecAXPY(Phi, 1.0/6.0, b);

    M2h->assemble(hj);

    MatMult(M2h->M, si, b);
    VecAXPY(Phi, 1.0/6.0, b);

    MatMult(M2h->M, sj, b);
    VecAXPY(Phi, 1.0/3.0, b);
    //VecAXPY(Phi, 1.0/2.0, b);
#else
    MatMult(M2->M, hi, b);
    VecAXPY(Phi, 0.5*9.80616, b);
    MatMult(M2->M, hj, b);
    VecAXPY(Phi, 0.5*9.80616, b);
#endif

    VecDestroy(&b);
}

void ThermalSW::diagnose_T() {
    Vec b, rhs;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);
    VecZeroEntries(T);

    M2h->assemble(hi);

    MatMult(M2h->M, hi, b);
    VecAXPY(rhs, 1.0/6.0, b);
    //VecAXPY(rhs, 1.0/4.0, b);

    MatMult(M2h->M, hj, b);
    VecAXPY(rhs, 1.0/6.0, b);

    M2h->assemble(hj);

    MatMult(M2h->M, hj, b);
    VecAXPY(rhs, 1.0/6.0, b);
    //VecAXPY(rhs, 1.0/4.0, b);

    KSPSolve(ksp2, rhs, T);

    VecDestroy(&b);
    VecDestroy(&rhs);
}

void ThermalSW::diagnose_q(Vec _ug, Vec _h, Vec* qi) {
    Vec rhs, tmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, qi);

    MatMult(M0->M, fg, rhs);
    MatMult(E01M1, _ug, tmp);
    VecAXPY(rhs, 1.0, tmp);
    M0h->assemble(_h);
    KSPSolve(ksp0h, rhs, *qi);

    VecDestroy(&rhs);
    VecDestroy(&tmp);
}

void ThermalSW::diagnose_ds(bool do_damping, double _dt) {
    Vec hh, sh, tmp1, tmp2, rhs, uhl;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &sh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rhs);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &uhl);

    VecZeroEntries(hh);
    VecAXPY(hh, 0.5, hi);
    VecAXPY(hh, 0.5, hj);
    M1h->assemble(hh);

    VecZeroEntries(sh);
    VecAXPY(sh, 0.5, si);
    VecAXPY(sh, 0.5, sj);

    // buoyancy damping
    /*
    MatMult(EtoF->E21, F, hh);
    VecAXPY(hh, +1.0/_dt, hj);
    VecAXPY(hh, -1.0/_dt, hi);
    VecScale(hh, -tau);
    VecAXPY(hh, 0.5, si);
    VecAXPY(hh, 0.5, sj);
    MatMult(M2->M, hh, tmp);
    MatMult(EtoF->E12, tmp, rhs);
    */
    if(do_damping) {
/*
        VecZeroEntries(uhl);
        VecAXPY(uhl, 0.5, uil);
        VecAXPY(uhl, 0.5, ujl);
        MatMult(EtoF->E21, F, tmp1);
        //VecAXPY(tmp1, +1.0/_dt, hj);
        //VecAXPY(tmp1, -1.0/_dt, hi);
        //M2tau->assemble(hh, sh, uhl, tmp1, _dt, 0.025);
        //M2tau->assemble(hh, sh, uhl, tmp1, _dt, 0.25);
        M2tau->assemble(hh, sh, uhl, tmp1, _dt, 0.5);
        MatMult(M2tau->M, tmp1, tmp2);
        KSPSolve(ksp2, tmp2, tmp1);
        VecAYPX(tmp1, +1.0, sh);     // generates entropy
        //VecAYPX(tmp1, -1.0, sh);     // dissipates entropy
        MatMult(M2->M, tmp1, tmp2);
*/
    } else {
        MatMult(M2->M, sh, tmp2);
    }
    MatMult(EtoF->E12, tmp2, rhs);
    KSPSolve(ksp1h, rhs, ds);

    VecDestroy(&hh);
    VecDestroy(&sh);
    VecDestroy(&tmp1);
    VecDestroy(&tmp2);
    VecDestroy(&rhs);
    VecDestroy(&uhl);
}

void ThermalSW::rhs_u(Vec uo, double _dt) {
    Vec tmp, tmph, qi, qj, qil, qjl, dqil, dqjl, dqg;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmph);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &qil);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &qjl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dqg);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dqil);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dqjl);

    VecZeroEntries(fu);
    MatMult(M1->M, uo, fu);

    MatMult(EtoF->E12, Phi, tmp);
    VecAXPY(fu, -_dt, tmp);

    diagnose_q(ui, hi, &qi);
    diagnose_q(uj, hj, &qj);
    VecScatterBegin(topo->gtol_0, qi, qil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qi, qil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_0, qj, qjl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qj, qjl, INSERT_VALUES, SCATTER_FORWARD);
    MatMult(NtoE->E10, qi, dqg);
    VecScatterBegin(topo->gtol_1, dqg, dqil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dqg, dqil, INSERT_VALUES, SCATTER_FORWARD);
    MatMult(NtoE->E10, qj, dqg);
    VecScatterBegin(topo->gtol_1, dqg, dqjl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dqg, dqjl, INSERT_VALUES, SCATTER_FORWARD);

    R_up->assemble_apvm(qil, uil, dqil, 0.5, _dt, ds, hi);
    MatMult(R_up->M, F, tmp);
    VecAXPY(fu, -0.5*_dt, tmp);
    R_up->assemble_apvm(qjl, ujl, dqjl, 0.5, _dt, ds, hj);
    MatMult(R_up->M, F, tmp);
    VecAXPY(fu, -0.5*_dt, tmp);
    /*
    VecZeroEntries(tmp);
    VecAXPY(tmp, 0.5, ui);
    VecAXPY(tmp, 0.5, uj);
    VecZeroEntries(tmph);
    VecAXPY(tmph, 0.5, hi);
    VecAXPY(tmph, 0.5, hj);
    diagnose_q(tmp, tmph, &qi);
    VecScatterBegin(topo->gtol_0, qi, qil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qi, qil, INSERT_VALUES, SCATTER_FORWARD);
    //R->assemble(qil);
    //MatMult(R->M, F, tmp);
    //VecAXPY(fu, -_dt, tmp);
    VecZeroEntries(dqil);
    VecAXPY(dqil, 0.5, uil);
    VecAXPY(dqil, 0.5, ujl);
    MatMult(NtoE->E10, qi, dqg);
    VecScatterBegin(topo->gtol_1, dqg, dqjl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dqg, dqjl, INSERT_VALUES, SCATTER_FORWARD);
    R_up->assemble_apvm(qil, dqil, dqjl, 0.5, _dt);
    MatMult(R->M, F, tmp);
    VecAXPY(fu, -_dt, tmp);
    */
#ifdef DO_THERMAL
    VecScatterBegin(topo->gtol_1, ds, dqil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ds, dqil, INSERT_VALUES, SCATTER_FORWARD);
    K->assemble(dqil);
    MatMultTranspose(K->M, T, tmp);
    VecAXPY(fu, +2.0*_dt, tmp); // K includes 0.5 factor
#endif

    VecDestroy(&tmp);
    VecDestroy(&tmph);
    VecDestroy(&qi);
    VecDestroy(&qj);
    VecDestroy(&qil);
    VecDestroy(&qjl);
    VecDestroy(&dqg);
    VecDestroy(&dqil);
    VecDestroy(&dqjl);
}

void ThermalSW::rhs_h(Vec ho, double _dt, bool mul_mat) {
    Vec tmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp);

    MatMult(EtoF->E21, F, tmp);
    VecAYPX(tmp, -_dt, ho);
    if(mul_mat) {
        MatMult(M2->M, tmp, fh);
    } else {
        VecCopy(tmp, fh);
    }

    VecDestroy(&tmp);
}

void ThermalSW::rhs_s(Vec so, double _dt) {
    Vec tmp, dsl;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dsl);

    MatMult(M2->M, so, fs);

#ifdef DO_THERMAL
    VecScatterBegin(topo->gtol_1, ds, dsl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ds, dsl, INSERT_VALUES, SCATTER_FORWARD);
    K->assemble(dsl);
    MatMult(K->M, F, tmp);
    VecAXPY(fs, -2.0*_dt, tmp); // K includes 0.5 factor
#endif

    VecDestroy(&tmp);
    VecDestroy(&dsl);
}

void ThermalSW::unpack(Vec x, Vec u, Vec h, Vec s) {
    Vec xl, ul;
    PetscScalar *xArray, *uArray, *hArray, *sArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, topo->n1 + 2*topo->n2, &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);

    VecScatterBegin(gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(xl, &xArray);
    VecGetArray(ul, &uArray);
    VecGetArray(h, &hArray);
    VecGetArray(s, &sArray);
    for(ii = 0; ii < topo->n1; ii++) {
        uArray[ii] = xArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        hArray[ii] = xArray[ii+topo->n1];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        sArray[ii] = xArray[ii+topo->n1+topo->n2];
    }
    VecRestoreArray(xl, &xArray);
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(h, &hArray);
    VecRestoreArray(s, &sArray);

    VecScatterBegin(topo->gtol_1, ul, u, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, ul, u, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&xl);
    VecDestroy(&ul);
}

void ThermalSW::repack(Vec x, Vec u, Vec h, Vec s) {
    Vec xl, ul;
    PetscScalar *xArray, *uArray, *hArray, *sArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, topo->n1 + 2*topo->n2, &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);

    VecScatterBegin(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(xl, &xArray);
    VecGetArray(ul, &uArray);
    VecGetArray(h, &hArray);
    VecGetArray(s, &sArray);
    for(ii = 0; ii < topo->n1; ii++) {
        xArray[ii] = uArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        xArray[ii+topo->n1] = hArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        xArray[ii+topo->n1+topo->n2] = sArray[ii];
    }
    VecRestoreArray(xl, &xArray);
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(h, &hArray);
    VecRestoreArray(s, &sArray);

    VecScatterBegin(gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&xl);
    VecDestroy(&ul);
}

void ThermalSW::assemble_operator(double _dt) {
    int n2 = (topo->elOrd+1)*(topo->elOrd+1);
    int mm, mi, mf, ri, ci, dof_proc;
    int nCols;
    const int *cols;
    const double* vals;
    int cols2[9999];
    Mat Muh, Mhu;

    if(!A) {
        MatCreate(MPI_COMM_WORLD, &A);
        MatSetSizes(A, topo->n1l+2*topo->n2l, topo->n1l+2*topo->n2l, topo->nDofs1G+2*topo->nDofs2G, topo->nDofs1G+2*topo->nDofs2G);
        MatSetType(A, MATMPIAIJ);
        MatMPIAIJSetPreallocation(A, 16*n2, PETSC_NULL, 16*n2, PETSC_NULL);

        KSPCreate(MPI_COMM_WORLD, &kspA);
        KSPSetOperators(kspA, A, A);
        KSPSetTolerances(kspA, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
        KSPSetOptionsPrefix(kspA, "A_");
        KSPSetFromOptions(kspA);
    }

    R->assemble(fl);
    MatAXPY(M1->M, ROS_ALPHA*_dt, R->M, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(M1->M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M1->M, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(M1->M, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(M1->M, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + 2*topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + 2*topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(M1->M, mm, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(M1->M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M1->M, MAT_FINAL_ASSEMBLY);

    // [u,h] block
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Muh);
    MatScale(Muh, ROS_ALPHA*_dt*9.80616);
    MatAssemblyBegin(Muh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Muh, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Muh, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Muh, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + 2*topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + 2*topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Muh, mm, &nCols, &cols, &vals);
    }

    // [u,s] block
#ifdef DO_THERMAL
    MatScale(Muh, -0.5*H_MEAN/9.80616);
    MatAssemblyBegin(Muh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Muh, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Muh, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Muh, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + 2*topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + 2*topo->n2l) + cols[ci] % topo->n2l + topo->n1l + topo->n2l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Muh, mm, &nCols, &cols, &vals);
    }
#endif

    // [h,u] block
    MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mhu);
    MatScale(Mhu, ROS_ALPHA*_dt*H_MEAN);
    MatAssemblyBegin(Mhu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Mhu, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Mhu, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Mhu, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + 2*topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + 2*topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Mhu, mm, &nCols, &cols, &vals);
    }

    // [h,h] block
    MatGetOwnershipRange(M2->M, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(M2->M, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + 2*topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + 2*topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(M2->M, mm, &nCols, &cols, &vals);
    }

    // [s,s] block
    MatGetOwnershipRange(M2->M, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(M2->M, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + 2*topo->n2l) + mm % topo->n2l + topo->n1l + topo->n2l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + 2*topo->n2l) + cols[ci] % topo->n2l + topo->n1l + topo->n2l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(M2->M, mm, &nCols, &cols, &vals);
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);

    MatDestroy(&Muh);
    MatDestroy(&Mhu);

    M1->assemble();
}

void ThermalSW::solve(double _dt, bool save, int nits) {
    int it = 0;
    double norm = 1.0e+9, norm_dx, norm_x;
    Vec x, f, dx, tmph, tmpu;

    dt = _dt;

    if(!A) assemble_operator(dt);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+2*topo->n2l, topo->nDofs1G+2*topo->nDofs2G, &x);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+2*topo->n2l, topo->nDofs1G+2*topo->nDofs2G, &f);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+2*topo->n2l, topo->nDofs1G+2*topo->nDofs2G, &dx);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmph);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmpu);

    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(uil, ujl);

    // solution vector
    VecCopy(ui, uj);
    VecCopy(hi, hj);
    VecCopy(si, sj);
    repack(x, ui, hi, si);

    do {
        //diagnose_F();
        diagnose_Phi();
#ifdef DO_THERMAL
        diagnose_T();
        diagnose_ds(false, _dt);
#endif

        diagnose_F();
        rhs_h(hi, _dt, true);
        //diagnose_F_up(-0.005*_dt);
        rhs_u(ui, _dt);
        rhs_s(si, _dt);

	MatMult(M1->M, uj, tmpu);
	VecAYPX(fu, -1.0, tmpu);

	MatMult(M2->M, hj, tmph);
	VecAYPX(fh, -1.0, tmph);

	MatMult(M2->M, sj, tmph);
	VecAYPX(fs, -1.0, tmph);

	repack(f, fu, fh, fs);
        KSPSolve(kspA, f, dx);
        VecAXPY(x, -1.0, dx);
        unpack(x, uj, hj, sj);
        VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
        VecNorm(x, NORM_2, &norm_x);
        VecNorm(dx, NORM_2, &norm_dx);
        norm = norm_dx/norm_x;
        if(!rank) {
            cout << scientific;
            cout << "iteration: " << it << "\t|x|: " << norm_x << "\t|dx|: " << norm_dx << "\t|dx|/|x|: " << norm << endl; 
        }
        it++;
    //} while(norm > 1.0e-14 and it < nits);
    } while(norm > 1.0e-16 and it < nits);

    VecCopy(uj, ui);
    VecCopy(hj, hi);
    VecCopy(sj, si);
    VecCopy(ujl, uil);

    if(save) {
        char fieldname[20];

        step++;
        curl(ui);
	MatMult(EtoF->E21, F, tmph);

        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(ui, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hi, fieldname, step);
        sprintf(fieldname, "buoyancy");
        geom->write2(si, fieldname, step);
        sprintf(fieldname, "divergence");
        geom->write2(tmph, fieldname, step);
    }

    VecDestroy(&x);
    VecDestroy(&f);
    VecDestroy(&dx);
    VecDestroy(&tmph);
    VecDestroy(&tmpu);
}

void ThermalSW::solve_rk2(double _dt, bool save) {
    VecCopy(ui, uj);
    VecCopy(hi, hj);
    VecCopy(si, sj);
    VecCopy(uil, ujl);

    VecCopy(ui, u_prev);
    VecCopy(hi, h_prev);
    VecCopy(si, s_prev);

    // stage 1
    diagnose_F();
    diagnose_Phi();
    diagnose_T();
    //diagnose_ds(2.0/_dt, _dt);
    //diagnose_ds(-2.0/_dt, _dt);
    diagnose_ds(0.0/_dt, _dt);

    if(first_step) {
        rhs_u(ui, _dt);
        rhs_h(hi, _dt, false);
        rhs_s(si, _dt);
	first_step = false;
    } else {
        rhs_u(u_prev, 2.0*_dt);
        rhs_h(h_prev, 2.0*_dt, false);
        rhs_s(s_prev, 2.0*_dt);
    }

    KSPSolve(ksp, fu, uj);
    VecCopy(fh, hj);
    KSPSolve(ksp2, fs, sj);

    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    // stage 2
    diagnose_F();
    diagnose_Phi();
    diagnose_T();
    //diagnose_ds(2.0/_dt, _dt);
    //diagnose_ds(-2.0/_dt, _dt);
    diagnose_ds(0.0/_dt, _dt);

    rhs_u(ui, _dt);
    rhs_h(hi, _dt, false);
    rhs_s(si, _dt);

    KSPSolve(ksp, fu, uj);
    VecCopy(fh, hj);
    KSPSolve(ksp2, fs, sj);

    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    VecCopy(uj, ui);
    VecCopy(hj, hi);
    VecCopy(sj, si);
    VecCopy(ujl, uil);

    if(save) {
        char fieldname[20];

        step++;
        curl(ui);

        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(ui, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hi, fieldname, step);
        sprintf(fieldname, "buoyancy");
        geom->write2(si, fieldname, step);
    }
}

ThermalSW::~ThermalSW() {
    KSPDestroy(&ksp);
    KSPDestroy(&ksp0);
    KSPDestroy(&ksp2);
    KSPDestroy(&ksp0h);
    KSPDestroy(&ksp1h);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);
    VecDestroy(&fl);
    VecScatterDestroy(&gtol_x);
    VecDestroy(&ui);
    VecDestroy(&hi);
    VecDestroy(&si);
    VecDestroy(&uj);
    VecDestroy(&hj);
    VecDestroy(&sj);
    VecDestroy(&fu);
    VecDestroy(&fh);
    VecDestroy(&fs);
    VecDestroy(&u_prev);
    VecDestroy(&h_prev);
    VecDestroy(&s_prev);
    VecDestroy(&uil);
    VecDestroy(&ujl);
    VecDestroy(&F);
    VecDestroy(&Phi);
    VecDestroy(&T);
    VecDestroy(&ds);
    VecDestroy(&wi);
    if(A) { 
        MatDestroy(&A);
        KSPDestroy(&kspA);
    }

    delete M0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete R_up;
    delete M1h;
    delete M2h;
    delete M0h;
    delete K;
    //delete M2tau;

    delete edge;
    delete node;
    delete quad;
}

void ThermalSW::init0(Vec q, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int* inds0;
    PtQmat* PQ = new PtQmat(topo, geom, node);
    PetscScalar *bArray;
    Vec bl, bg, PQb;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, geom->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PQb);
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
    VecScatterEnd(  geom->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(PQ->M, bg, PQb);
    KSPSolve(ksp0, PQb, q);

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void ThermalSW::init1(Vec u, ICfunc* func_x, ICfunc* func_y) {
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

void ThermalSW::init2(Vec h, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    PetscScalar *bArray;
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

    KSPSolve(ksp2, WQb, h);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void ThermalSW::err0(Vec ug, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds0;
    double det, wd, l_inf;
    double un[1], dun[2], ua[1], dua[2];
    double local_1[2], global_1[2], local_2[2], global_2[2], local_i[2], global_i[2]; // first entry is the error, the second is the norm
    PetscScalar *array_0, *array_1;
    Vec ul, dug, dul;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dug);

    VecScatterBegin(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    MatMult(NtoE->E10, ug, dug);
    VecScatterBegin(topo->gtol_1, dug, dul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, dug, dul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local_1[0] = local_1[1] = 0.0;
    local_2[0] = local_2[1] = 0.0;
    local_i[0] = local_i[1] = 0.0;

    VecGetArray(ul, &array_0);
    VecGetArray(dul, &array_1);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
                geom->interp0(ex, ey, ii%mp1, ii/mp1, array_0, un);
                ua[0] = fw(geom->x[inds0[ii]]);

                det = geom->det[ei][ii];
                wd = det*quad->w[ii%mp1]*quad->w[ii/mp1];

                local_1[0] += wd*fabs(un[0] - ua[0]);
                local_1[1] += wd*fabs(ua[0]);

                local_2[0] += wd*(un[0] - ua[0])*(un[0] - ua[0]);
                local_2[1] += wd*ua[0]*ua[0];

                l_inf = wd*fabs(un[0] - ua[0]);
                if(fabs(l_inf) > local_i[0]) {
                    local_i[0] = fabs(l_inf);
                    local_i[1] = fabs(wd*fabs(ua[0]));
                }

                if(fu != NULL && fv != NULL) {
                    geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, dun);
                    dua[0] = fu(geom->x[inds0[ii]]);
                    dua[1] = fv(geom->x[inds0[ii]]);

                    local_2[0] += wd*((dun[0] - dua[0])*(dun[0] - dua[0]) + (dun[1] - dua[1])*(dun[1] - dua[1]));
                    local_2[1] += wd*(dua[0]*dua[0] + dua[1]*dua[1]);
                }
            }
        }
    }
    VecRestoreArray(ul, &array_0);
    VecRestoreArray(dul, &array_1);

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

void ThermalSW::err1(Vec ug, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms) {
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

void ThermalSW::err2(Vec ug, ICfunc* fu, double* norms) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds0;
    double det, wd, l_inf;
    double un[1], ua[1];
    double local_1[2], global_1[2], local_2[2], global_2[2], local_i[2], global_i[2]; // first entry is the error, the second is the norm
    PetscScalar *array_2;
    Vec ul;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &ul);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local_1[0] = local_1[1] = 0.0;
    local_2[0] = local_2[1] = 0.0;
    local_i[0] = local_i[1] = 0.0;

    VecGetArray(ug, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
if(fabs(geom->s[inds0[ii]][1]) > 0.45*M_PI) continue;
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, un);
                ua[0] = fu(geom->x[inds0[ii]]);

                det = geom->det[ei][ii];
                wd = det*quad->w[ii%mp1]*quad->w[ii/mp1];

                local_1[0] += wd*fabs(un[0] - ua[0]);
                local_1[1] += wd*fabs(ua[0]);

                local_2[0] += wd*(un[0] - ua[0])*(un[0] - ua[0]);
                local_2[1] += wd*ua[0]*ua[0];

                l_inf = wd*fabs(un[0] - ua[0]);
                if(fabs(l_inf) > local_i[0]) {
                    local_i[0] = fabs(l_inf);
                    local_i[1] = fabs(wd*fabs(ua[0]));
                }

            }
        }
    }
    VecRestoreArray(ug, &array_2);

    MPI_Allreduce(local_1, global_1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_2, global_2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_i, global_i, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    VecDestroy(&ul);

    norms[0] = global_1[0]/global_1[1];
    norms[1] = sqrt(global_2[0]/global_2[1]);
    norms[2] = global_i[0]/global_i[1];
}

double ThermalSW::int2(Vec ug) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, uq, local, global;
    PetscScalar *array_2;
    Vec ul;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &ul);

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

    VecDestroy(&ul);

    return global;
}

double ThermalSW::intE(Vec ul, Vec hg, Vec sg) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, hq, sq, uq[2], local, global;
    PetscScalar *array_1, *array_2, *array_3;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(ul, &array_1);
    VecGetArray(hg, &array_2);
    VecGetArray(sg, &array_3);

    local = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, uq);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, &hq);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_3, &sq);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*0.5*(sq*hq*hq + hq*(uq[0]*uq[0] + uq[1]*uq[1]));
            }
        }
    }
    VecRestoreArray(ul, &array_1);
    VecRestoreArray(hg, &array_2);
    VecRestoreArray(sg, &array_3);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global;
}

double ThermalSW::intK(Vec dqg, Vec dsg) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, uq1[2], uq2[2], local, global;
    PetscScalar *array_1, *array_2;
    Vec dql, dsl;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dql);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dsl);

    VecScatterBegin(topo->gtol_1, dqg, dql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dqg, dql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, dsg, dsl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dsg, dsl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(dql, &array_1);
    VecGetArray(dsl, &array_2);

    local = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, uq1);
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_2, uq2);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*0.5*(uq1[0]*uq2[0] + uq1[1]*uq2[1]);
            }
        }
    }
    VecRestoreArray(dql, &array_1);
    VecRestoreArray(dsl, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&dql);
    VecDestroy(&dsl);

    return global;
}

void ThermalSW::writeConservation(double time, double mass0, double vort0, double ener0, double enst0, double buoy0) {
    double mass, vort, ener, enst, buoy, enst_rhs;
    char filename[50];
    ofstream file;
    Vec qi, v0, htmp, dq, utmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &v0);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dq);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);

    curl(ui);

    diagnose_q(ui, hi, &qi);
    MatMult(M0h->M, qi, v0);
    VecDot(qi, v0, &enst);

    MatMult(M2->M, hi, htmp);
    //VecSum(htmp, &mass);
    mass = int2(hi);

    VecDot(htmp, si, &buoy);

    MatMult(M0->M, wi, v0);
    VecSum(v0, &vort);

    //K->assemble(uil);
    //MatMult(K->M, ui, htmp);
    //VecDot(hi, htmp, &ener);
    //ener += 0.5*buoy;
    ener = intE(uil, hi, si);

    // potential enstrophy growth term
    diagnose_ds(false, 0.0);
    MatMult(NtoE->E10, qi, dq);
    MatMult(M1->M, dq, utmp);
    enst_rhs = intK(dq, ds);

    if(!rank) {
        cout << "conservation of mass:      " << (mass - mass0)/mass0 << endl;
        cout << "conservation of vorticity: " << (vort - vort0) << endl;
        cout << "conservation of energy:    " << (ener - ener0)/ener0 << endl;
        cout << "conservation of enstrophy: " << (enst - enst0)/enst0 << endl;
        cout << "conservation of buoyancy:  " << (buoy - buoy0)/buoy0 << endl;

        sprintf(filename, "output/conservation.dat");
        file.open(filename, ios::out | ios::app);
        // write time in days
        file << scientific;
        file << time/60.0/60.0/24.0 << "\t" << (mass-mass0)/mass0 << "\t" << (vort-vort0) << "\t" 
                                            << (ener-ener0)/ener0 << "\t" << (enst-enst0)/enst0 << "\t"
					    << (buoy-buoy0)/buoy0 << "\t" << enst_rhs << "\t"
					    << endl;
        file.close();
    }
    VecDestroy(&qi);
    VecDestroy(&v0);
    VecDestroy(&htmp);
    VecDestroy(&dq);
    VecDestroy(&utmp);
}

////////////////////////////////////////////////////////////////
void ThermalSW::diagnose_F_inst(Vec _u, Vec _h, Vec _F) {
    Vec b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);
    VecZeroEntries(_F);
    M1h->assemble(_h);
    MatMult(M1h->M, _u, b);
    KSPSolve(ksp, b, _F);
    VecDestroy(&b);
}

void ThermalSW::diagnose_Phi_inst(Vec _ul, Vec _ug, Vec _h, Vec _s, Vec _Phi) {
    Vec b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecZeroEntries(Phi);

    K->assemble(_ul); // includes the 0.5 factor
    MatMult(K->M, _ug, b);
    VecAXPY(_Phi, 1.0, b);

    M2h->assemble(_h);
    MatMult(M2h->M, _s, b);
    VecAXPY(_Phi, 1.0, b);

    VecDestroy(&b);
}

void ThermalSW::diagnose_T_inst(Vec _h, Vec _T) {
    Vec b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecZeroEntries(_T);
    //M2h->assemble(_h); already assembled the Phi diagnostic routine
    MatMult(M2h->M, _h, b);
    VecScale(b, 0.5);
    KSPSolve(ksp2, b, _T);
    VecDestroy(&b);
}

void ThermalSW::diagnose_ds_inst(bool upwind, double _dt, Vec _hi, Vec _hj, Vec _s, Vec _F, Vec _ul, Vec _ds) {
    Vec tmp1, tmp2, rhs, dsl;
    //double tau = -0.5*10.0*_dt/10000.0;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rhs);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dsl);

    if(upwind) {
/*
        MatMult(EtoF->E21, _F, tmp1);
        //VecCopy(tmp1, tmp2);
        VecAXPY(tmp1, +1.0/_dt, _hj);
        VecAXPY(tmp1, -1.0/_dt, _hi);
        VecCopy(tmp1, tmp2);

        //VecAYPX(tmp1, tau, _s);
        //MatMult(M2->M, tmp1, tmp2);
        //MatMult(EtoF->E12, tmp2, rhs);

        M2tau->assemble(_hi, _s, _ul, tmp2, _dt, 0.5);
        MatMult(M2tau->M, tmp1, tmp2);
        KSPSolve(ksp2, tmp2, tmp1);
        VecAYPX(tmp1, +1.0, _s);
*/
    } else {
        VecCopy(_s, tmp1);
    }
    MatMult(M2->M, tmp1, tmp2);
    MatMult(EtoF->E12, tmp2, rhs);

    //M1h->assemble(_hi); assume this already assembled in the mass flux routine
    KSPSolve(ksp1h, rhs, _ds);

    VecScatterBegin(topo->gtol_1, _ds, dsl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, _ds, dsl, INSERT_VALUES, SCATTER_FORWARD);
    K->assemble(dsl);

    VecDestroy(&tmp1);
    VecDestroy(&tmp2);
    VecDestroy(&rhs);
    VecDestroy(&dsl);
}

void ThermalSW::rhs_u_inst(Vec _u, Vec _h, Vec _F, Vec _Phi, Vec _T, Vec _ds, Vec _q, Vec _ql, Vec _qil, Vec _qjl, Vec _ul, Vec _fu, double _dt) {
    Vec tmp, dql, dqg, d2u, d4u;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dqg);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dql);

    MatMult(EtoF->E12, _Phi, _fu);
/*
    MatMult(NtoE->E10, _q, dqg);
    VecScatterBegin(topo->gtol_1, dqg, dql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dqg, dql, INSERT_VALUES, SCATTER_FORWARD);

    R_up->assemble_supg(_ql, _ul, dql, 0.5, -_dt, _qjl);
    MatMult(R_up->M, _F, tmp);
    VecAXPY(_fu, 0.5, tmp);
    R_up->assemble_supg(_ql, _ul, dql, 0.5, +_dt, _qil);
    MatMult(R_up->M, _F, tmp);
    VecAXPY(_fu, 0.5, tmp);
*/
    R->assemble(_ql);
    MatMult(R->M, _F, tmp);
    VecAXPY(_fu, 1.0, tmp);

    //VecScatterBegin(topo->gtol_1, _ds, dql, INSERT_VALUES, SCATTER_FORWARD);
    //VecScatterEnd(  topo->gtol_1, _ds, dql, INSERT_VALUES, SCATTER_FORWARD);
    //K->assemble(dql); // assembling is ds diagnostic routine
    MatMultTranspose(K->M, _T, tmp);
    VecAXPY(_fu, -2.0, tmp); // K includes 0.5 factor

    laplacian(_u, &d2u);
    laplacian(d2u, &d4u);
    MatMult(M1->M, d4u, d2u);
    VecAXPY(fu, 1.0, d2u);
    VecDestroy(&d2u);
    VecDestroy(&d4u);

    VecDestroy(&tmp);
    VecDestroy(&dqg);
    VecDestroy(&dql);
}

void ThermalSW::rhs_h_inst(Vec _F, Vec _fh, bool mul_mat) {
    Vec tmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp);

    MatMult(EtoF->E21, _F, tmp);
    if(mul_mat) {
        MatMult(M2->M, tmp, _fh);
    } else {
        VecCopy(tmp, _fh);
    }

    VecDestroy(&tmp);
}

void ThermalSW::rhs_s_inst(Vec _F, Vec _s, Vec _fs) {
    Vec ds, dds, Mdds;

    // assume K(ds) has already been assembled in the momentum eqn rhs function
    MatMult(K->M, _F, _fs);
    VecScale(_fs, 2.0);

    if(_s) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dds);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mdds);

        grad(_s, &ds);
        MatMult(EtoF->E21, ds, dds);
	VecDestroy(&ds);

	grad(dds, &ds);
        MatMult(EtoF->E21, ds, dds);
	VecDestroy(&ds);

	MatMult(M2->M, dds, Mdds);
	VecAXPY(_fs, del2*del2, Mdds);

	VecDestroy(&dds);
	VecDestroy(&Mdds);
    }
}

void ThermalSW::solve_ssp_rk2(double _dt, bool save) {
    Vec _fu1, _fu2, _fh1, _fh2, _fs1, _fs2, _q1, _q2, _q1l, _q2l, _ql, tmp1, tmp2;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_fu1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_fu2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &_fh1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &_fh2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &_fs1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &_fs2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp2);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &_ql);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &_q1l);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &_q2l);

    VecCopy(ui, uj);
    VecCopy(hi, hj);
    VecCopy(si, sj);
    VecCopy(uil, ujl);

    // first step
    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    diagnose_q(ui, hi, &_q1);
    VecScatterBegin(topo->gtol_0, _q1, _q1l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, _q1, _q1l, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(_q1l, _ql);
    VecCopy(_q1l, _q2l);

    diagnose_F_inst(ui, hi, F);
    diagnose_Phi_inst(uil, ui, hi, si, Phi);
    diagnose_T_inst(hi, T);
    diagnose_ds_inst(true, _dt, hi, hj, si, F, uil, ds);

    rhs_u_inst(ui, hi, F, Phi, T, ds, _q1, _ql, _q1l, _q2l, uil, _fu1, _dt);
    rhs_h_inst(F, _fh1, false);
    rhs_s_inst(F, si, _fs1);

    MatMult(M1->M, ui, tmp1);
    VecAXPY(tmp1, -_dt, _fu1);
    KSPSolve(ksp, tmp1, uj);

    VecCopy(hi, hj);
    VecAXPY(hj, -_dt, _fh1);

    MatMult(M2->M, si, tmp2);
    VecAXPY(tmp2, -_dt, _fs1);
    KSPSolve(ksp2, tmp2, sj);

    // second step
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    diagnose_q(uj, hj, &_q2);
    VecScatterBegin(topo->gtol_0, _q2, _q2l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, _q2, _q2l, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(_q2l, _ql);

    diagnose_F_inst(uj, hj, F);
    diagnose_Phi_inst(ujl, uj, hj, sj, Phi);
    diagnose_T_inst(hj, T);
    diagnose_ds_inst(true, _dt, hi, hj, sj, F, ujl, ds);

    rhs_u_inst(uj, hj, F, Phi, T, ds, _q2, _ql, _q1l, _q2l, ujl, _fu2, _dt);
    rhs_h_inst(F, _fh2, false);
    rhs_s_inst(F, sj, _fs2);

    MatMult(M1->M, ui, tmp1);
    VecAXPY(tmp1, -0.5*_dt, _fu1);
    VecAXPY(tmp1, -0.5*_dt, _fu2);
    KSPSolve(ksp, tmp1, uj);

    VecCopy(hi, hj);
    VecAXPY(hj, -0.5*_dt, _fh1);
    VecAXPY(hj, -0.5*_dt, _fh2);

    MatMult(M2->M, si, tmp2);
    VecAXPY(tmp2, -0.5*_dt, _fs1);
    VecAXPY(tmp2, -0.5*_dt, _fs2);
    KSPSolve(ksp2, tmp2, sj);

    // udpate
    VecCopy(uj, ui);
    VecCopy(hj, hi);
    VecCopy(sj, si);
    VecCopy(ujl, uil);

    if(save) {
        char fieldname[20];

        step++;
        curl(ui);
	MatMult(EtoF->E21, F, tmp2);

        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(ui, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hi, fieldname, step);
        sprintf(fieldname, "buoyancy");
        geom->write2(si, fieldname, step);
        sprintf(fieldname, "divergence");
        geom->write2(tmp2, fieldname, step);
    }

    VecDestroy(&_fu1);
    VecDestroy(&_fu2);
    VecDestroy(&_fh1);
    VecDestroy(&_fh2);
    VecDestroy(&_fs1);
    VecDestroy(&_fs2);
    VecDestroy(&_q1);
    VecDestroy(&_q2);
    VecDestroy(&_ql);
    VecDestroy(&_q1l);
    VecDestroy(&_q2l);
    VecDestroy(&tmp1);
    VecDestroy(&tmp2);
}
