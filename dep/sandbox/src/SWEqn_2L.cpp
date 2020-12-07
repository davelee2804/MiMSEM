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
#include "SWEqn_2L.h"

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0

using namespace std;

SWEqn_2L::SWEqn_2L(Topo* _topo, Geom* _geom) {
    PC pc;
    int ii, jj;
    int dof_proc;
    int* loc = new int[2*(_topo->n1+_topo->n2)];
    IS is_g, is_l;
    Vec xl, xg;

    topo = _topo;
    geom = _geom;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    grav    = 9.80616*(RAD_SPHERE/RAD_EARTH);
    rho_t   = 0.600192080827;//0.661798009439;//0.131703004382;
    rho_b   = 1.09857215148;//0.749152837846;
    H_t     = 10000.0;//8000.0;//20000.0;
    H_b     = 2000.0;//10000.0;
    omega   = 7.292e-5;
    del2    = viscosity();
    do_visc = true;
    step    = 0;

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

    // adjoint differential operators (curl and grad)
    MatMatMult(NtoE->E01, M1->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E01M1);
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E12M2);

    // rotational operator
    R = new RotMat(topo, geom, node, edge);

    // mass flux operator
    M1h = new Uhmat(topo, geom, node, edge);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

    A = NULL;

    // initialize the linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp, "sw_");
    KSPSetFromOptions(ksp);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_ti);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_ti);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_bi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_bi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_bj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_bj);

    // create the [u_t,h_t,u_b,h_b] -> [x] vec scatter
    for(ii = 0; ii < topo->n1; ii++) {
        dof_proc = topo->loc1[ii] / topo->n1l;
        loc[ii] = dof_proc * 2*(topo->n1l + topo->n2l) + topo->loc1[ii] % topo->n1l;
    }
    for(ii = 0; ii < topo->n2; ii++) {
        jj = ii + topo->n1;
        dof_proc = topo->loc2[ii] / topo->n2l;
        loc[jj] = dof_proc * 2*(topo->n1l + topo->n2l) + topo->loc2[ii] % topo->n2l + topo->n1l;
    }
    for(ii = 0; ii < topo->n1; ii++) {
        jj = ii + topo->n1 + topo->n2;
        dof_proc = topo->loc1[ii] / topo->n1l;
        loc[jj] = dof_proc * 2*(topo->n1l + topo->n2l) + topo->loc1[ii] % topo->n1l + topo->n1l + topo->n2l;
    }
    for(ii = 0; ii < topo->n2; ii++) {
        jj = ii + 2*topo->n1 + topo->n2;
        dof_proc = topo->loc2[ii] / topo->n2l;
        loc[jj] = dof_proc * 2*(topo->n1l + topo->n2l) + topo->loc2[ii] % topo->n2l + 2*topo->n1l + topo->n2l;
    }

    ISCreateStride(MPI_COMM_SELF, 2*(topo->n1+topo->n2), 0, 1, &is_l);
    ISCreateGeneral(MPI_COMM_WORLD, 2*(topo->n1+topo->n2), loc, PETSC_COPY_VALUES, &is_g);

    VecCreateSeq(MPI_COMM_SELF, 2*(topo->n1+topo->n2), &xl);
    VecCreateMPI(MPI_COMM_WORLD, 2*(topo->n1l+topo->n2l), 2*(topo->nDofs1G+topo->nDofs2G), &xg);

    VecScatterCreate(xg, is_g, xl, is_l, &gtol_x);

    delete[] loc;
    ISDestroy(&is_l);
    ISDestroy(&is_g);
    VecDestroy(&xl);
    VecDestroy(&xg);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double SWEqn_2L::viscosity() {
    double ae = 4.0*M_PI*RAD_SPHERE*RAD_SPHERE;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void SWEqn_2L::coriolis() {
    int ii;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fl, fxl, fxg, PtQfxg;

    // initialise the coriolis vector (local and global)
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg);

    // evaluate the coriolis term at nodes
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < topo->n0; ii++) {
        fArray[ii] = 2.0*omega*sin(geom->s[ii][1]);
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
    VecPointwiseDivide(fg, PtQfxg, m0->vg);
    
    delete PtQ;
    VecDestroy(&fl);
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

// derive vorticity (global vector) as \omega = curl u
// assumes diagonal 0 form mass matrix
void SWEqn_2L::curl(Vec u, Vec *w) {
    Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);

    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    // diagonal mass matrix as vector
    VecPointwiseDivide(*w, du, m0->vg);

    VecDestroy(&du);
}

// dH/du = hu = F
void SWEqn_2L::diagnose_F(Vec ui, Vec uj, Vec hi, Vec hj, Vec* F) {
    Vec hu, b, hil, hjl;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hil);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hjl);

    VecScatterBegin(topo->gtol_2, hi, hil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, hi, hil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, hj, hjl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, hj, hjl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);
    VecZeroEntries(*F);
    VecZeroEntries(hu);

    // assemble the nonlinear rhs mass matrix (note that hl is a local vector)
    M1h->assemble(hil);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/3.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/6.0, b);

    M1h->assemble(hjl);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/6.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/3.0, b);

    // solve the linear system
    KSPSolve(ksp, hu, *F);

    VecDestroy(&hu);
    VecDestroy(&b);
    VecDestroy(&hil);
    VecDestroy(&hjl);
}

// dH/dh = (1/2)u^2 + gh = \Phi
// note: \Phi is in integral form here
//          \int_{\Omega} \gamma_h,\Phi_h d\Omega
void SWEqn_2L::diagnose_Phi(Vec ui, Vec uj, Vec h1i, Vec h1j, Vec h2i, Vec h2j, double grav_this, double grav_other, Vec* Phi) {
    Vec uil, ujl, b, hSum;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &uil);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ujl);

    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hSum);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Phi);
    VecZeroEntries(*Phi);

    // u^2 terms (0.5 factor incorportated into the matrix assembly)
    K->assemble(uil);

    MatMult(K->M, ui, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    MatMult(K->M, uj, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    K->assemble(ujl);

    MatMult(K->M, uj, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    // gh terms
    VecZeroEntries(hSum);
    VecAXPY(hSum, 0.5*grav_this,  h1i);
    VecAXPY(hSum, 0.5*grav_this,  h1j);
    VecAXPY(hSum, 0.5*grav_other, h2i);
    VecAXPY(hSum, 0.5*grav_other, h2j);

    MatMult(M2->M, hSum, b);
    VecAXPY(*Phi, 1.0, b);

    VecDestroy(&uil);
    VecDestroy(&ujl);
    VecDestroy(&b);
    VecDestroy(&hSum);
}

void SWEqn_2L::diagnose_wxu(Vec ui, Vec uj, Vec* wxu) {
    Vec wi, wj, wl, uh;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, wxu);

    curl(ui, &wi);
    curl(uj, &wj);
    VecAXPY(wi, 1.0, wj);
    VecAYPX(wi, 0.5, fg);

    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(uh);
    VecAXPY(uh, 0.5, ui);
    VecAXPY(uh, 0.5, uj);

    R->assemble(wl);
    MatMult(R->M, uh, *wxu);

    VecDestroy(&wi);
    VecDestroy(&wj);
    VecDestroy(&wl);
    VecDestroy(&uh);
}

void SWEqn_2L::laplacian(Vec u, Vec* ddu) {
    Vec Du, Cu, RCu, GDu, MDu, dMDu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, ddu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &RCu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &GDu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMDu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &MDu);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, u, Du);

    // grad (weak form)
    MatMult(M2->M, Du, MDu);
    MatMult(EtoF->E12, MDu, dMDu);
    KSPSolve(ksp, dMDu, GDu);

    /*** rotational component ***/
    // curl (weak form)
    curl(u, &Cu);

    // rot (strong form)
    MatMult(NtoE->E10, Cu, RCu);

    // add rotational and divergent components
    VecCopy(GDu, *ddu);
    VecAXPY(*ddu, +1.0, RCu);

    VecScale(*ddu, del2);

    VecDestroy(&Cu);
    VecDestroy(&RCu);
    VecDestroy(&GDu);
    VecDestroy(&dMDu);
    VecDestroy(&Du);
    VecDestroy(&MDu);
}

void SWEqn_2L::unpack(Vec x, Vec u_t, Vec h_t, Vec u_b, Vec h_b) {
    Vec xl, ul, hl;
    PetscScalar *xArray, *uArray, *hArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, 2*(topo->n1 + topo->n2), &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);

    VecScatterBegin(gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(xl, &xArray);
    VecGetArray(ul, &uArray);
    VecGetArray(hl, &hArray);
    // top level
    for(ii = 0; ii < topo->n1; ii++) {
        uArray[ii] = xArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        hArray[ii] = xArray[ii+topo->n1];
    }
    VecScatterBegin(topo->gtol_1, ul, u_t, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, ul, u_t, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterBegin(topo->gtol_2, hl, h_t, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_2, hl, h_t, INSERT_VALUES, SCATTER_REVERSE);

    // bottom level
    for(ii = 0; ii < topo->n1; ii++) {
        uArray[ii] = xArray[ii+topo->n1+topo->n2];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        hArray[ii] = xArray[ii+2*topo->n1+topo->n2];
    }
    VecScatterBegin(topo->gtol_1, ul, u_b, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, ul, u_b, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterBegin(topo->gtol_2, hl, h_b, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_2, hl, h_b, INSERT_VALUES, SCATTER_REVERSE);

    VecRestoreArray(xl, &xArray);
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(hl, &hArray);

    VecDestroy(&xl);
    VecDestroy(&ul);
    VecDestroy(&hl);
}

void SWEqn_2L::repack(Vec x, Vec u_t, Vec h_t, Vec u_b, Vec h_b) {
    Vec xl, ul, hl;
    PetscScalar *xArray, *uArray, *hArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, 2*(topo->n1 + topo->n2), &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);

    VecGetArray(xl, &xArray);

    // top level
    VecScatterBegin(topo->gtol_1, u_t, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, u_t, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, h_t, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h_t, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecGetArray(ul, &uArray);
    VecGetArray(hl, &hArray);

    for(ii = 0; ii < topo->n1; ii++) {
        xArray[ii] = uArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        xArray[ii+topo->n1] = hArray[ii];
    }
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(hl, &hArray);

    // bottom level
    VecScatterBegin(topo->gtol_1, u_b, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, u_b, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, h_b, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h_b, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecGetArray(ul, &uArray);
    VecGetArray(hl, &hArray);

    for(ii = 0; ii < topo->n1; ii++) {
        xArray[ii+topo->n1+topo->n2] = uArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        xArray[ii+2*topo->n1+topo->n2] = hArray[ii];
    }
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(hl, &hArray);

    VecRestoreArray(xl, &xArray);
    VecScatterBegin(gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&xl);
    VecDestroy(&ul);
    VecDestroy(&hl);
}

void SWEqn_2L::assemble_residual(Vec x, Vec f) {
    Vec F, Phi, wxu, fu_t, fh_t, fu_b, fh_b, utmp, htmp1, htmp2, d2u, d4u, fs;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu_t);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh_t);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu_b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh_b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp2);
    VecCreateMPI(MPI_COMM_WORLD, 2*(topo->n1l+topo->n2l), 2*(topo->nDofs1G+topo->nDofs2G), &fs);

    VecZeroEntries(fu_t);
    VecZeroEntries(fh_t);
    VecZeroEntries(fu_b);
    VecZeroEntries(fh_b);

    unpack(x, u_tj, h_tj, u_bj, h_bj);

    // assemble in the skew-symmetric parts of the vector

    // top level
    diagnose_F(u_ti, u_tj, h_ti, h_tj, &F);
    diagnose_Phi(u_ti, u_tj, h_ti, h_tj, h_bi, h_bj, grav, grav, &Phi);
    diagnose_wxu(u_ti, u_tj, &wxu);

    // -- momentum terms
    MatMult(EtoF->E12, Phi, fu_t);
    VecAXPY(fu_t, 1.0, wxu);

    // -- continuity term
    MatMult(EtoF->E21, F, htmp1);
    MatMult(M2->M, htmp1, htmp2);
    VecAXPY(fh_t, 1.0, htmp2);

    VecDestroy(&F);
    VecDestroy(&Phi);
    VecDestroy(&wxu);

    // bottom level
    diagnose_F(u_bi, u_bj, h_bi, h_bj, &F);
    diagnose_Phi(u_bi, u_bj, h_bi, h_bj, h_ti, h_tj, grav, (rho_t/rho_b)*grav, &Phi);
    diagnose_wxu(u_bi, u_bj, &wxu);

    // -- momentum terms
    MatMult(EtoF->E12, Phi, fu_b);
    VecAXPY(fu_b, 1.0, wxu);

    // -- continuity term
    MatMult(EtoF->E21, F, htmp1);
    MatMult(M2->M, htmp1, htmp2);
    VecAXPY(fh_b, 1.0, htmp2);

    VecDestroy(&F);
    VecDestroy(&Phi);
    VecDestroy(&wxu);

    repack(fs, fu_t, fh_t, fu_b, fh_b);

    // assemble the mass matrix terms
    VecZeroEntries(fu_t);
    VecZeroEntries(fh_t);
    VecZeroEntries(fu_b);
    VecZeroEntries(fh_b);

    // top level
    MatMult(M1->M, u_tj, fu_t);
    MatMult(M1->M, u_ti, utmp);
    VecAXPY(fu_t, -1.0, utmp);

    if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, u_ti);
        VecAXPY(utmp, 0.5, u_tj);
        laplacian(utmp, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu_t, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    MatMult(M2->M, h_tj, fh_t);
    MatMult(M2->M, h_ti, htmp1);
    VecAXPY(fh_t, -1.0, htmp1);

    // bottom level
    MatMult(M1->M, u_bj, fu_b);
    MatMult(M1->M, u_bi, utmp);
    VecAXPY(fu_b, -1.0, utmp);

    if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, u_bi);
        VecAXPY(utmp, 0.5, u_bj);
        laplacian(utmp, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu_b, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    MatMult(M2->M, h_bj, fh_b);
    MatMult(M2->M, h_bi, htmp1);
    VecAXPY(fh_b, -1.0, htmp1);

    repack(f, fu_t, fh_t, fu_b, fh_b);
    VecAXPY(f, dt, fs);

    // clean up
    VecDestroy(&fu_t);
    VecDestroy(&fh_t);
    VecDestroy(&fu_b);
    VecDestroy(&fh_b);
    VecDestroy(&utmp);
    VecDestroy(&htmp1);
    VecDestroy(&htmp2);
    VecDestroy(&fs);
}

void SWEqn_2L::assemble_operator() {
    int n2 = (topo->elOrd+1)*(topo->elOrd+1);
    int mm, mi, mf, ri, ci, dof_proc;
    int nCols;
    const int *cols;
    const double* vals;
    double levVals[9999];
    int cols2[9999];
    Vec fl;
    Mat Muu, Muh, Mhu, Mhh;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    VecScatterBegin(topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);

    MatCreate(MPI_COMM_WORLD, &Muu);
    MatSetSizes(Muu, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(Muu, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Muu, 8*n2, PETSC_NULL, 8*n2, PETSC_NULL);

    MatCreate(MPI_COMM_WORLD, &Mhh);
    MatSetSizes(Mhh, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(Mhh, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Mhh, 8*n2, PETSC_NULL, 8*n2, PETSC_NULL);

    MatCreate(MPI_COMM_WORLD, &A);
    MatSetSizes(A, 2*(topo->n1l+topo->n2l), 2*(topo->n1l+topo->n2l), 2*(topo->nDofs1G+topo->nDofs2G), 2*(topo->nDofs1G+topo->nDofs2G));
    MatSetType(A, MATMPIAIJ);
    MatMPIAIJSetPreallocation(A, 16*n2, PETSC_NULL, 16*n2, PETSC_NULL);

    // [u,u] blocks
    MatCopy(M1->M, Muu, DIFFERENT_NONZERO_PATTERN);
    R->assemble(fl);
    MatAXPY(Muu, 0.5*dt, R->M, DIFFERENT_NONZERO_PATTERN);

    MatGetOwnershipRange(Muu, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Muu, mm, &nCols, &cols, &vals);

        // top level
        dof_proc = mm / topo->n1l;
        ri = dof_proc * 2*(topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * 2*(topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        // bottom level
        ri += topo->n1l + topo->n2l;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] += topo->n1l + topo->n2l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);

        MatRestoreRow(Muu, mm, &nCols, &cols, &vals);
    }

    // [u,h] blocks
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Muh);
    MatScale(Muh, 0.5*dt);
    MatGetOwnershipRange(Muh, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Muh, mm, &nCols, &cols, &vals);

        // [u_t,h_t]
        dof_proc = mm / topo->n1l;
        ri = dof_proc * 2*(topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * 2*(topo->n1l + topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
            levVals[ci] = grav * vals[ci];
        }
        MatSetValues(A, 1, &ri, nCols, cols2, levVals, INSERT_VALUES);
        // [u_t,h_b]
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] += topo->n1l + topo->n2l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, levVals, INSERT_VALUES);
        // [u_b,h_t]
        ri += topo->n1l + topo->n2l;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] -= topo->n1l + topo->n2l;
            levVals[ci] = (rho_t/rho_b) * grav * vals[ci];
        }
        MatSetValues(A, 1, &ri, nCols, cols2, levVals, INSERT_VALUES);
        // [u_b,h_b]
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] += topo->n1l + topo->n2l;
            levVals[ci] = grav * vals[ci];
        }
        MatSetValues(A, 1, &ri, nCols, cols2, levVals, INSERT_VALUES);

        MatRestoreRow(Muh, mm, &nCols, &cols, &vals);
    }

    // [h,u] block
    MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mhu);
    MatScale(Mhu, 0.5*dt);
    MatGetOwnershipRange(Mhu, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Mhu, mm, &nCols, &cols, &vals);

        // [h_t,u_t]
        dof_proc = mm / topo->n2l;
        ri = dof_proc * 2*(topo->n1l + topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * 2*(topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
            levVals[ci] = H_t * vals[ci];
        }
        MatSetValues(A, 1, &ri, nCols, cols2, levVals, INSERT_VALUES);

        // [h_b,u_b]
        ri += topo->n1l + topo->n2l;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] += topo->n1l + topo->n2l;
            levVals[ci] = H_b * vals[ci];
        }
        MatSetValues(A, 1, &ri, nCols, cols2, levVals, INSERT_VALUES);

        MatRestoreRow(Mhu, mm, &nCols, &cols, &vals);
    }

    // [h,h] block
    MatCopy(M2->M, Mhh, DIFFERENT_NONZERO_PATTERN);
    MatGetOwnershipRange(Mhh, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Mhh, mm, &nCols, &cols, &vals);

        // [h_t,h_t]
        dof_proc = mm / topo->n2l;
        ri = dof_proc * 2*(topo->n1l + topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * 2*(topo->n1l + topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);

        // [h_b,h_b]
        ri += topo->n1l + topo->n2l;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] += topo->n1l + topo->n2l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);

        MatRestoreRow(Mhh, mm, &nCols, &cols, &vals);
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);

    VecDestroy(&fl);
    MatDestroy(&Muu);
    MatDestroy(&Muh);
    MatDestroy(&Mhu);
    MatDestroy(&Mhh);
}

void SWEqn_2L::solve(Vec u_tn, Vec u_bn, Vec h_tn, Vec h_bn, double _dt, bool save) {
    bool done = false;
    int it = 0;
    double norm = 1.0e+9, norm_dx, norm_x;
    Vec x, f, dx;
    KSP kspA;

    dt = _dt;

    if(!A) assemble_operator();

    VecCreateMPI(MPI_COMM_WORLD, 2*(topo->n1l+topo->n2l), 2*(topo->nDofs1G+topo->nDofs2G), &x);
    VecCreateMPI(MPI_COMM_WORLD, 2*(topo->n1l+topo->n2l), 2*(topo->nDofs1G+topo->nDofs2G), &f);
    VecCreateMPI(MPI_COMM_WORLD, 2*(topo->n1l+topo->n2l), 2*(topo->nDofs1G+topo->nDofs2G), &dx);

    // solution vector
    VecCopy(u_tn, u_ti);
    VecCopy(h_tn, h_ti);
    VecCopy(u_tn, u_tj);
    VecCopy(h_tn, h_tj);

    VecCopy(u_bn, u_bi);
    VecCopy(h_bn, h_bi);
    VecCopy(u_bn, u_bj);
    VecCopy(h_bn, h_bj);

    repack(x, u_tn, h_tn, u_bn, h_bn);

    KSPCreate(MPI_COMM_WORLD, &kspA);
    KSPSetOperators(kspA, A, A);
    KSPSetTolerances(kspA, 1.0e-14, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetOptionsPrefix(kspA, "A_");
    KSPSetFromOptions(kspA);

    do {
        assemble_residual(x, f);
        VecScale(f, -1.0);
        KSPSolve(kspA, f, dx);
        VecAXPY(x, +1.0, dx);
        VecNorm(x, NORM_2, &norm_x);
        VecNorm(dx, NORM_2, &norm_dx);
        norm = norm_dx/norm_x;

        if(!rank) {
            cout << scientific;
            cout << it << "\t|dx|/|x|: " << norm << endl; 
        }
        it++;

        if(norm < 1.0e-14) done = true;
    } while(!done);

    unpack(x, u_tn, h_tn, u_bn, h_bn);

    if(save) {
        Vec w_t, w_b, w_h, u_h;
        char fieldname[20];

        step++;
        curl(u_tn, &w_t);
        curl(u_bn, &w_b);

        sprintf(fieldname, "vorticity_t");
        geom->write0(w_t, fieldname, step);
        sprintf(fieldname, "velocity_t");
        geom->write1(u_tn, fieldname, step);
        sprintf(fieldname, "pressure_t");
        geom->write2(h_tn, fieldname, step);

        sprintf(fieldname, "vorticity_b");
        geom->write0(w_b, fieldname, step);
        sprintf(fieldname, "velocity_b");
        geom->write1(u_bn, fieldname, step);
        sprintf(fieldname, "pressure_b");
        geom->write2(h_bn, fieldname, step);

        VecDestroy(&w_t);
        VecDestroy(&w_b);

        // write out the baroclinic fields
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_h);
        VecZeroEntries(u_h);
        VecAXPY(u_h, +0.5, u_tn);
        VecAXPY(u_h, -0.5, u_bn);
        curl(u_h, &w_h);

        sprintf(fieldname, "vorticity_h");
        geom->write0(w_h, fieldname, step);
        sprintf(fieldname, "velocity_h");
        geom->write1(u_h, fieldname, step);

        VecDestroy(&w_h);
        VecDestroy(&u_h);
    }

    VecDestroy(&x);
    VecDestroy(&f);
    VecDestroy(&dx);
    KSPDestroy(&kspA);
}

SWEqn_2L::~SWEqn_2L() {
    KSPDestroy(&ksp);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);
    VecScatterDestroy(&gtol_x);
    VecDestroy(&u_ti);
    VecDestroy(&h_ti);
    VecDestroy(&u_tj);
    VecDestroy(&h_tj);
    VecDestroy(&u_bi);
    VecDestroy(&h_bi);
    VecDestroy(&u_bj);
    VecDestroy(&h_bj);
    MatDestroy(&A);

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete M1h;
    delete K;

    delete edge;
    delete node;
    delete quad;
}

void SWEqn_2L::init0(Vec q, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int* inds0;
    PtQmat* PQ = new PtQmat(topo, geom, node);
    PetscScalar *bArray;
    Vec bl, bg, PQb;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);
    VecScatterBegin(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(PQ->M, bg, PQb);
    VecPointwiseDivide(q, PQb, m0->vg);

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void SWEqn_2L::init1(Vec u, ICfunc* func_x, ICfunc* func_y) {
    int ex, ey, ii, mp1, mp12;
    int *inds0, *loc02;
    UtQmat* UQ = new UtQmat(topo, geom, node, edge);
    PetscScalar *bArray;
    Vec bl, bg, UQb;
    IS isl, isg;
    VecScatter scat;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, 2*topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, 2*topo->n0l, 2*topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &UQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[2*inds0[ii]+0] = func_x(geom->x[inds0[ii]]);
                bArray[2*inds0[ii]+1] = func_y(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);

    // create a new vec scatter object to handle vector quantity on nodes
    loc02 = new int[2*topo->n0];
    for(ii = 0; ii < topo->n0; ii++) {
        loc02[2*ii+0] = 2*topo->loc0[ii]+0;
        loc02[2*ii+1] = 2*topo->loc0[ii]+1;
    }
    ISCreateStride(MPI_COMM_WORLD, 2*topo->n0, 0, 1, &isl);
    ISCreateGeneral(MPI_COMM_WORLD, 2*topo->n0, loc02, PETSC_COPY_VALUES, &isg);
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

void SWEqn_2L::init2(Vec h, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    PetscScalar *bArray;
    KSP ksp2;
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
                bArray[inds0[ii]] = func(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);
    VecScatterBegin(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

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

void SWEqn_2L::err0(Vec ug, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms) {
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

void SWEqn_2L::err1(Vec ug, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms) {
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
    VecScatterBegin(topo->gtol_2, dug, dul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, dug, dul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local_1[0] = local_1[1] = 0.0;
    local_2[0] = local_2[1] = 0.0;
    local_i[0] = local_i[1] = 0.0;

    VecGetArray(ul, &array_1);
    VecGetArray(dul, &array_2);

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
    VecRestoreArray(dul, &array_2);

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

void SWEqn_2L::err2(Vec ug, ICfunc* fu, double* norms) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds0;
    double det, wd, l_inf;
    double un[1], ua[1];
    double local_1[2], global_1[2], local_2[2], global_2[2], local_i[2], global_i[2]; // first entry is the error, the second is the norm
    PetscScalar *array_2;
    Vec ul;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &ul);
    VecScatterBegin(topo->gtol_2, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local_1[0] = local_1[1] = 0.0;
    local_2[0] = local_2[1] = 0.0;
    local_i[0] = local_i[1] = 0.0;

    VecGetArray(ul, &array_2);

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
    VecRestoreArray(ul, &array_2);

    MPI_Allreduce(local_1, global_1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_2, global_2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_i, global_i, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    VecDestroy(&ul);

    norms[0] = global_1[0]/global_1[1];
    norms[1] = sqrt(global_2[0]/global_2[1]);
    norms[2] = global_i[0]/global_i[1];
}

double SWEqn_2L::int0(Vec ug) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, uq, local, global;
    PetscScalar *array_0;
    Vec ul;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ul);
    VecScatterBegin(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ul, &array_0);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp0(ex, ey, ii%mp1, ii/mp1, array_0, &uq);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*uq;
            }
        }
    }
    VecRestoreArray(ul, &array_0);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&ul);

    return global;
}

double SWEqn_2L::int2(Vec ug) {
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

double SWEqn_2L::intE(double gravity, Vec ug, Vec hg) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, hq, local, global;
    double uq[2];
    PetscScalar *array_1, *array_2;
    Vec ul, hl;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecScatterBegin(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecScatterBegin(topo->gtol_2, hg, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, hg, hl, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ul, &array_1);
    VecGetArray(hl, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, uq);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, &hq);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*(gravity*hq*hq + 0.5*hq*(uq[0]*uq[0] + uq[1]*uq[1]));
            }
        }
    }
    VecRestoreArray(ul, &array_1);
    VecRestoreArray(hl, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&ul);
    VecDestroy(&hl);

    return global;
}

void SWEqn_2L::writeConservation(double time, Vec u1, Vec u2, Vec h1, Vec h2, double mass0, double vort0, double ener0) {
    double mass, vort, ener;
    char filename[50];
    ofstream file;
    Vec w1, w2;

    curl(u1, &w1);
    curl(u2, &w2);

    mass = int2(h1) + int2(h2);
    vort = int0(w1) + int0(w2);
    ener = intE(grav, u1, h1) + intE((rho_t/rho_b)*grav, u2, h2);

    if(!rank) {
        cout << "conservation of mass:      " << (mass - mass0)/mass0 << endl;
        cout << "conservation of vorticity: " << (vort - vort0) << endl;
        cout << "conservation of energy:    " << (ener - ener0)/ener0 << endl;

        sprintf(filename, "output/conservation.dat");
        file.open(filename, ios::out | ios::app);
        // write time in days
        file << time/60.0/60.0/24.0 << "\t" << (mass-mass0)/mass0 << "\t" << (vort-vort0) << "\t" << (ener-ener0)/ener0 << endl;
        file.close();
    }
    VecDestroy(&w1);
    VecDestroy(&w2);
} 

void SWEqn_2L::solve_explicit(Vec u_tn, Vec u_bn, Vec h_tn, Vec h_bn, double _dt, bool save) {
    double gPrime = (rho_t/rho_b)*grav;
    Vec F_t1, F_t2, F_b1, F_b2;
    Vec wxu_t1, wxu_t2, wxu_b1, wxu_b2;
    Vec Phi_t1, Phi_t2, Phi_b1, Phi_b2;
    Vec bu, tu, th, d2u, d4u;

    dt = _dt;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &th);

    VecCopy(u_tn, u_ti);
    VecCopy(u_bn, u_bi);
    VecCopy(h_tn, h_ti);
    VecCopy(h_bn, h_bi);

    // first step
    // ...momentum
    diagnose_wxu(u_ti, u_ti, &wxu_t1);
    diagnose_wxu(u_bi, u_bi, &wxu_b1);
    diagnose_Phi(u_ti, u_ti, h_ti, h_ti, h_bi, h_bi, grav, grav,   &Phi_t1);
    diagnose_Phi(u_bi, u_bi, h_bi, h_bi, h_ti, h_ti, grav, gPrime, &Phi_b1);

    // ......top layer
    MatMult(M1->M, u_tn, bu);
    VecAXPY(bu, -dt, wxu_t1);
    MatMult(EtoF->E12, Phi_t1, tu);
    VecAXPY(bu, -dt, tu);
    if(do_visc) {
        laplacian(u_tn, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(bu, -dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
    VecZeroEntries(u_tj);
    KSPSolve(ksp, bu, u_tj);

    // ......bottom layer
    MatMult(M1->M, u_bn, bu);
    VecAXPY(bu, -dt, wxu_b1);
    MatMult(EtoF->E12, Phi_b1, tu);
    VecAXPY(bu, -dt, tu);
    if(do_visc) {
        laplacian(u_bn, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(bu, -dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
    VecZeroEntries(u_bj);
    KSPSolve(ksp, bu, u_bj);

    // ...continuity
    // ......top layer
    diagnose_F(u_ti, u_ti, h_ti, h_ti, &F_t1);
    MatMult(EtoF->E21, F_t1, h_tj);
    VecAYPX(h_tj, -dt, h_tn);

    // ......bottom layer
    diagnose_F(u_bi, u_bi, h_bi, h_bi, &F_b1);
    MatMult(EtoF->E21, F_b1, h_bj);
    VecAYPX(h_bj, -dt, h_bn);

    // second step
    // ...momentum
    diagnose_wxu(u_tj, u_tj, &wxu_t2);
    diagnose_wxu(u_bj, u_bj, &wxu_b2);
    diagnose_Phi(u_tj, u_tj, h_tj, h_tj, h_bj, h_bj, grav, grav,   &Phi_t2);
    diagnose_Phi(u_bj, u_bj, h_bj, h_bj, h_tj, h_tj, grav, gPrime, &Phi_b2);

    // ......top layer
    MatMult(M1->M, u_tn, bu);
    VecAXPY(bu, -0.5*dt, wxu_t1);
    VecAXPY(bu, -0.5*dt, wxu_t2);
    MatMult(EtoF->E12, Phi_t1, tu);
    VecAXPY(bu, -0.5*dt, tu);
    MatMult(EtoF->E12, Phi_t2, tu);
    VecAXPY(bu, -0.5*dt, tu);
    if(do_visc) {
        laplacian(u_tj, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(bu, -dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
    VecZeroEntries(u_tn);
    KSPSolve(ksp, bu, u_tn);

    // ......bottom layer
    MatMult(M1->M, u_bn, bu);
    VecAXPY(bu, -0.5*dt, wxu_b1);
    VecAXPY(bu, -0.5*dt, wxu_b2);
    MatMult(EtoF->E12, Phi_b1, tu);
    VecAXPY(bu, -0.5*dt, tu);
    MatMult(EtoF->E12, Phi_b2, tu);
    VecAXPY(bu, -0.5*dt, tu);
    if(do_visc) {
        laplacian(u_bj, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(bu, -dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
    VecZeroEntries(u_bn);
    KSPSolve(ksp, bu, u_bn);

    // ...continuity
    // ......top layer
    diagnose_F(u_tj, u_tj, h_tj, h_tj, &F_t2);
    VecAXPY(F_t2, 1.0, F_t1);
    MatMult(EtoF->E21, F_t2, th);
    VecAXPY(h_tn, -0.5*dt, th);

    // ......bottom layer
    diagnose_F(u_bj, u_bj, h_bj, h_bj, &F_b2);
    VecAXPY(F_b2, 1.0, F_b1);
    MatMult(EtoF->E21, F_b2, th);
    VecAXPY(h_bn, -0.5*dt, th);

    if(save) {
        Vec w_t, w_b, w_h, u_h;
        char fieldname[20];

        step++;
        curl(u_tn, &w_t);
        curl(u_bn, &w_b);

        sprintf(fieldname, "vorticity_t");
        geom->write0(w_t, fieldname, step);
        sprintf(fieldname, "velocity_t");
        geom->write1(u_tn, fieldname, step);
        sprintf(fieldname, "pressure_t");
        geom->write2(h_tn, fieldname, step);

        sprintf(fieldname, "vorticity_b");
        geom->write0(w_b, fieldname, step);
        sprintf(fieldname, "velocity_b");
        geom->write1(u_bn, fieldname, step);
        sprintf(fieldname, "pressure_b");
        geom->write2(h_bn, fieldname, step);

        VecDestroy(&w_t);
        VecDestroy(&w_b);

        // write out the baroclinic fields
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_h);
        VecZeroEntries(u_h);
        VecAXPY(u_h, +0.5, u_tn);
        VecAXPY(u_h, -0.5, u_bn);
        curl(u_h, &w_h);

        sprintf(fieldname, "vorticity_h");
        geom->write0(w_h, fieldname, step);
        sprintf(fieldname, "velocity_h");
        geom->write1(u_h, fieldname, step);

        VecDestroy(&w_h);
        VecDestroy(&u_h);
    }

    VecDestroy(&bu);
    VecDestroy(&tu);
    VecDestroy(&th);
    VecDestroy(&Phi_t1);
    VecDestroy(&Phi_b1);
    VecDestroy(&Phi_t2);
    VecDestroy(&Phi_b2);
    VecDestroy(&wxu_t1);
    VecDestroy(&wxu_b1);
    VecDestroy(&wxu_t2);
    VecDestroy(&wxu_b2);
    VecDestroy(&F_t1);
    VecDestroy(&F_b1);
    VecDestroy(&F_t2);
    VecDestroy(&F_b2);
}
