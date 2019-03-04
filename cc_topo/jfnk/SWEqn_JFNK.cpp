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
#include "SWEqn_JFNK.h"

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
//#define RAD_SPHERE 1.0
#define W2_ALPHA (0.25*M_PI)

#define WEAK_FORM_H
//#define RIGHT

using namespace std;

/*
use as:

matrix free:
    mpirun -np 6 ./mimsem 0 -snes_mf -snes_type newtontr -snes_stol 1.0e-4

preconditioned:
    mpirun -np 6 ./mimsem 0 -snes_mf_operator -snes_type newtontr -snes_stol 1.0e-8 \
        -ksp_rtol 1.0e-7 -ksp_converged_reason -ksp_monitor

fieldsplit:
    mpirun -np 6 ./mimsem -snes_monitor -ksp_monitor_true_residual -ksp_converged_reason \
        -snes_mf_operator -snes_type newtontr -snes_stol 1.0e-12 -ksp_rtol 1.0e-12 \
        -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_type fgmres
*/

SWEqn::SWEqn(Topo* _topo, Geom* _geom) {
    PC pc;

    topo = _topo;
    geom = _geom;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    grav = 9.80616*(RAD_SPHERE/RAD_EARTH);
    omega = 7.292e-5;
    del2 = viscosity();
    do_visc = true;
    u_only = false;
    step = 0;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // 0 form lumped mass matrix (vector)
    m0 = new Pvec(topo, geom, node);
    m0h = new Ph_vec(topo, geom, node);

    // 1 form mass matrix
    M1 = new Umat(topo, geom, node, edge);

    // 2 form mass matrix
    M2 = new Wmat(topo, geom, edge);

    U0 = new U0mat(topo, geom, node, edge);

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

    // create the [u,h] -> [x] vec scatter
    {
        int ii, jj;
        int dof_proc;
        int* loc = new int[topo->n1+topo->n2];
        IS is_g, is_l;
        Vec xl, xg;

        for(ii = 0; ii < topo->n1; ii++) {
            dof_proc = topo->loc1[ii] / topo->n1l;
            loc[ii] = dof_proc * (topo->n1l + topo->n2l) + topo->loc1[ii] % topo->n1l;
        }
        for(ii = 0; ii < topo->n2; ii++) {
            jj = ii + topo->n1;
            dof_proc = topo->loc2[ii] / topo->n2l;
            loc[jj] = dof_proc * (topo->n1l + topo->n2l) + topo->loc2[ii] % topo->n2l + topo->n1l;
        }

        ISCreateStride(MPI_COMM_SELF, topo->n1+topo->n2, 0, 1, &is_l);
        ISCreateGeneral(MPI_COMM_WORLD, topo->n1+topo->n2, loc, PETSC_COPY_VALUES, &is_g);

        VecCreateSeq(MPI_COMM_SELF, topo->n1+topo->n2, &xl);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &xg);

        VecScatterCreate(xg, is_g, xl, is_l, &gtol_x);

        // create the u and h index sets on this processor for later use by the fieldsplit preconditioner
        ISCreateStride(MPI_COMM_WORLD, topo->n1l, rank * (topo->n1l + topo->n2l), 1, &is_u);
        ISCreateStride(MPI_COMM_WORLD, topo->n2l, rank * (topo->n1l + topo->n2l) + topo->n1l, 1, &is_h);

        delete[] loc;
        ISDestroy(&is_l);
        ISDestroy(&is_g);
        VecDestroy(&xl);
        VecDestroy(&xg);
    }

    // schur complement matrix (for the fieldsplit preconditioner)
    MatCreate(MPI_COMM_WORLD, &SC);
    MatSetSizes(SC, topo->n2l, topo->n2l, topo->nDofs2G, topo->nDofs2G);
    MatSetType(SC, MATMPIAIJ);
    MatMPIAIJSetPreallocation(SC, 9*edge->n*edge->n, PETSC_NULL, 9*edge->n*edge->n, PETSC_NULL);
    MatSetOptionsPrefix(SC, "SC_");
    MatZeroEntries(SC);

    precon_assembled = false;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double SWEqn::viscosity() {
    double ae = 4.0*M_PI*RAD_SPHERE*RAD_SPHERE;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void SWEqn::coriolis() {
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
#ifdef W2_ALPHA
        fArray[ii] = 2.0*omega*( -cos(geom->s[ii][0])*cos(geom->s[ii][1])*sin(W2_ALPHA) + sin(geom->s[ii][1])*cos(W2_ALPHA) );
#else
        fArray[ii] = 2.0*omega*sin(geom->s[ii][1]);
#endif
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
void SWEqn::curl(Vec u, Vec *w) {
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
void SWEqn::diagnose_F(Vec* F) {
    double scale = 1.0;
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
    M1h->assemble(hil, scale);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/3.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/6.0, b);

    M1h->assemble(hjl, scale);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/6.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/3.0, b);

    // solve the linear system
    M1->assemble(scale);
    KSPSolve(ksp, hu, *F);
    M1->assemble(1.0);

/*
    int rr, ri, rj, cc, ncols;
    const int* cols;
    const PetscScalar* vals;
    double valInv;
    Mat Uinv;
    MatCreate(MPI_COMM_WORLD, &Uinv);
    MatSetSizes(Uinv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(Uinv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Uinv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatZeroEntries(Uinv);
    MatGetOwnershipRange(M1->M, &ri, &rj);
    for(rr = ri; rr < rj; rr++) {
        MatGetRow(M1->M, rr, &ncols, &cols, &vals);
        for(cc = 0; cc < ncols; cc++) {
            if(cols[cc] == rr) {
                valInv = 1.0/vals[cc];
                MatSetValues(Uinv, 1, &rr, 1, &rr, &valInv, ADD_VALUES);
            }
        }
        MatRestoreRow(M1->M, rr, &ncols, &cols, &vals);
    }
    MatAssemblyBegin(Uinv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Uinv, MAT_FINAL_ASSEMBLY);
    MatMult(Uinv, hu, *F);
    MatDestroy(&Uinv);
*/

    VecDestroy(&hu);
    VecDestroy(&b);
    VecDestroy(&hil);
    VecDestroy(&hjl);
}

// dH/dh = (1/2)u^2 + gh = \Phi
// note: \Phi is in integral form here
//          \int_{\Omega} \gamma_h,\Phi_h d\Omega
void SWEqn::diagnose_Phi(Vec* Phi) {
    Vec uil, ujl, b;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &uil);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ujl);

    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
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
    MatMult(M2->M, hi, b);
    VecAXPY(*Phi, grav/2.0, b);

    MatMult(M2->M, hj, b);
    VecAXPY(*Phi, grav/2.0, b);

    VecDestroy(&uil);
    VecDestroy(&ujl);
    VecDestroy(&b);
}

void SWEqn::diagnose_wxu(Vec* wxu) {
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

void SWEqn::laplacian(Vec u, Vec* ddu) {
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

void SWEqn::unpack(Vec x, Vec u, Vec h) {
    Vec xl, ul, hl;
    PetscScalar *xArray, *uArray, *hArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, topo->n1 + topo->n2, &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);

    VecScatterBegin(gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(xl, &xArray);
    VecGetArray(ul, &uArray);
    VecGetArray(hl, &hArray);
    for(ii = 0; ii < topo->n1; ii++) {
        uArray[ii] = xArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        hArray[ii] = xArray[ii+topo->n1];
    }
    VecRestoreArray(xl, &xArray);
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(hl, &hArray);

    VecScatterBegin(topo->gtol_1, ul, u, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, ul, u, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterBegin(topo->gtol_2, hl, h, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_2, hl, h, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&xl);
    VecDestroy(&ul);
    VecDestroy(&hl);
}

void SWEqn::repack(Vec x, Vec u, Vec h) {
    Vec xl, ul, hl;
    PetscScalar *xArray, *uArray, *hArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, topo->n1 + topo->n2, &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);

    VecScatterBegin(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(xl, &xArray);
    VecGetArray(ul, &uArray);
    VecGetArray(hl, &hArray);
    for(ii = 0; ii < topo->n1; ii++) {
        xArray[ii] = uArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        xArray[ii+topo->n1] = hArray[ii];
    }
    VecRestoreArray(xl, &xArray);
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(hl, &hArray);

    VecScatterBegin(gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&xl);
    VecDestroy(&ul);
    VecDestroy(&hl);
}

void SWEqn::jfnk_vector(Vec x, Vec f) {
    UtQh_vec* bndry = new UtQh_vec(topo, geom, node, edge);
    Vec F, Phi, wxu, fu, fh, utmp, htmp1, htmp2, d2u, d4u;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp2);

    unpack(x, uj, hj);

    // momentum equation
    VecZeroEntries(fu);

    diagnose_F(&F);
    diagnose_Phi(&Phi);
    diagnose_wxu(&wxu);

    MatMult(M1->M, uj, fu);
    VecAXPY(fu, dt, wxu);

    MatMult(EtoF->E12, Phi, utmp);
    VecAXPY(fu, dt, utmp);

    if(do_visc) {
        laplacian(uj, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    // assemble the contour integral of the (left+right)/2 depth field
    // around each element
    //bndry->assemble(hi);
    //VecAXPY(fu, +0.5*dt*grav, bndry->ug);
    //bndry->assemble(hj);
    //VecAXPY(fu, +0.5*dt*grav, bndry->ug);

    // continuity equation
    VecZeroEntries(fh);

#ifdef WEAK_FORM_H
    MatMult(M2->M, hj, fh);
#else
    VecCopy(hj, fh);
#endif
    MatMult(EtoF->E21, F, htmp1);
#ifdef WEAK_FORM_H
    MatMult(M2->M, htmp1, htmp2);
    VecAXPY(fh, dt, htmp2);
#else
    VecAXPY(fh, dt, htmp1);
#endif

    repack(f, fu, fh);

    // clean up
    VecDestroy(&fu);
    VecDestroy(&fh);
    VecDestroy(&utmp);
    VecDestroy(&htmp1);
    VecDestroy(&htmp2);
    VecDestroy(&F);
    VecDestroy(&Phi);
    VecDestroy(&wxu);

    delete bndry;
}

void SWEqn::jfnk_vector_u(Vec x, Vec f) {
    Vec F, Phi, wxu, fu, fh, utmp, htmp, d2u, d4u;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);

    VecCopy(x, uj);

    diagnose_F(&F);
    diagnose_Phi(&Phi);
    diagnose_wxu(&wxu);

    MatMult(EtoF->E21, F, htmp);
    VecCopy(hi, hj);
    VecAXPY(hj, -dt, htmp);

    // momentum equation
    VecZeroEntries(fu);
    MatMult(M1->M, uj, fu);
    VecAXPY(fu, dt, wxu);
    MatMult(EtoF->E12, Phi, utmp);
    VecAXPY(fu, dt, utmp);

    if(do_visc) {
        laplacian(uj, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
    VecCopy(fu, f);

    // clean up
    VecDestroy(&fu);
    VecDestroy(&fh);
    VecDestroy(&utmp);
    VecDestroy(&htmp);
    VecDestroy(&F);
    VecDestroy(&Phi);
    VecDestroy(&wxu);
}

/* 
  left preconditioning (strong form continuity equation):

    [  U   GW ] = [  U         0      ][ I  U^{-1}GW ]
    [ G^T   0 ]   [ G^T  -G^TU^{-1}GW ][ 0      I    ]

  right preconditioning (strong form continuity equation):

    [  U   GW ] = [     I      0 ][ U       GW      ]
    [ G^T   0 ]   [ G^TU^{-1}  I ][ 0  -G^TU^{-1}GW ]
*/
void SWEqn::jfnk_precon(Mat P) {
    int ri, rj, rr, cc, ncols, row_proc, col_proc;
    int pRow, pCols[99];
    const int* cols;
    const PetscScalar* vals;
    double valInv;
    Mat M2D, GM2, M2DUinv, M2DUinvGM2;
    Mat Uinv;

    if(precon_assembled) return;

    // assemble the approximate H(div) matrix inverse
    MatCreate(MPI_COMM_WORLD, &Uinv);
    MatSetSizes(Uinv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(Uinv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Uinv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatZeroEntries(Uinv);
    MatGetOwnershipRange(M1->M, &ri, &rj);
    for(rr = ri; rr < rj; rr++) {
        MatGetRow(M1->M, rr, &ncols, &cols, &vals);
        for(cc = 0; cc < ncols; cc++) {
            if(cols[cc] == rr) {
                valInv = 1.0/vals[cc];
                MatSetValues(Uinv, 1, &rr, 1, &rr, &valInv, ADD_VALUES);
            }
        }
        MatRestoreRow(M1->M, rr, &ncols, &cols, &vals);
    }
    MatAssemblyBegin(Uinv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Uinv, MAT_FINAL_ASSEMBLY);

#ifdef WEAK_FORM_H
    MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M2D);
#else
    MatConvert(EtoF->E21, MATSAME, MAT_INITIAL_MATRIX, &M2D);
#endif
    MatScale(M2D, dt*10000.0);
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GM2);
    MatScale(GM2, dt*grav);
    MatMatMult(M2D, Uinv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M2DUinv);
    MatMatMult(M2DUinv, GM2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M2DUinvGM2);
    MatZeroEntries(SC);
    MatCopy(M2DUinvGM2, SC, DIFFERENT_NONZERO_PATTERN);
    MatScale(SC, -1.0);

    MatAssemblyBegin(SC, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  SC, MAT_FINAL_ASSEMBLY);

    MatZeroEntries(P);

    // [u,u] block
    MatGetOwnershipRange(M1->M, &ri, &rj);
    for(rr = ri; rr < rj; rr++) {
        MatGetRow(M1->M, rr, &ncols, &cols, &vals);

        row_proc = rr / topo->n1l;
        pRow = row_proc * (topo->n1l + topo->n2l) + rr % topo->n1l;

        for(cc = 0; cc < ncols; cc++) {
            col_proc = cols[cc] / topo->n1l;
            pCols[cc] = col_proc * (topo->n1l + topo->n2l) + cols[cc] % topo->n1l;
        }

        MatSetValues(P, 1, &pRow, ncols, pCols, vals, ADD_VALUES);
        MatRestoreRow(M1->M, rr, &ncols, &cols, &vals);
    }

#ifdef RIGHT
    // [u,h] block
    MatGetOwnershipRange(GM2, &ri, &rj);
    for(rr = ri; rr < rj; rr++) {
        MatGetRow(GM2, rr, &ncols, &cols, &vals);

        row_proc = rr / topo->n1l;
        pRow = row_proc * (topo->n1l + topo->n2l) + rr % topo->n1l;

        for(cc = 0; cc < ncols; cc++) {
            col_proc = cols[cc] / topo->n2l;
            pCols[cc] = col_proc * (topo->n1l + topo->n2l) + cols[cc] % topo->n2l + topo->n1l;
        }

        MatSetValues(P, 1, &pRow, ncols, pCols, vals, ADD_VALUES);
        MatRestoreRow(GM2, rr, &ncols, &cols, &vals);
    }
#else
    // [h,u] block
    MatGetOwnershipRange(M2D, &ri, &rj);
    for(rr = ri; rr < rj; rr++) {
        MatGetRow(M2D, rr, &ncols, &cols, &vals);

        row_proc = rr / topo->n2l;
        pRow = row_proc * (topo->n1l + topo->n2l) + rr % topo->n2l + topo->n1l;

        for(cc = 0; cc < ncols; cc++) {
            col_proc = cols[cc] / topo->n1l;
            pCols[cc] = col_proc * (topo->n1l + topo->n2l) + cols[cc] % topo->n1l;
        }

        MatSetValues(P, 1, &pRow, ncols, pCols, vals, ADD_VALUES);
        MatRestoreRow(M2D, rr, &ncols, &cols, &vals);
    }
#endif

    // [h,h] block
    MatGetOwnershipRange(SC, &ri, &rj);
    for(rr = ri; rr < rj; rr++) {
        MatGetRow(SC, rr, &ncols, &cols, &vals);

        row_proc = rr / topo->n2l;
        pRow = row_proc * (topo->n1l + topo->n2l) + rr % topo->n2l + topo->n1l;

        for(cc = 0; cc < ncols; cc++) {
            col_proc = cols[cc] / topo->n2l;
            pCols[cc] = col_proc * (topo->n1l + topo->n2l) + cols[cc] % topo->n2l + topo->n1l;
        }
        MatSetValues(P, 1, &pRow, ncols, pCols, vals, ADD_VALUES);
        MatRestoreRow(SC, rr, &ncols, &cols, &vals);
    }

    //precon_assembled = true;

    MatDestroy(&M2D);
    MatDestroy(&GM2);
    MatDestroy(&M2DUinv);
    MatDestroy(&M2DUinvGM2);
    MatDestroy(&Uinv);
}

void SWEqn::jfnk_precon_u(Mat P) {
    int ri, rj, rr, cc, ncols;
    const int* cols;
    const PetscScalar* vals;
    double valInv;
    Mat M2D, GM2, M2DUinv, M2DUinvGM2;
    Mat Uinv;

    if(precon_assembled) return;

    // assemble the approximate H(div) matrix inverse
    MatCreate(MPI_COMM_WORLD, &Uinv);
    MatSetSizes(Uinv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(Uinv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Uinv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatZeroEntries(Uinv);
    MatGetOwnershipRange(M1->M, &ri, &rj);
    for(rr = ri; rr < rj; rr++) {
        MatGetRow(M1->M, rr, &ncols, &cols, &vals);
        for(cc = 0; cc < ncols; cc++) {
            if(cols[cc] == rr) {
                valInv = 1.0/vals[cc];
                MatSetValues(Uinv, 1, &rr, 1, &rr, &valInv, ADD_VALUES);
            }
        }
        MatRestoreRow(M1->M, rr, &ncols, &cols, &vals);
    }
    MatAssemblyBegin(Uinv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Uinv, MAT_FINAL_ASSEMBLY);

#ifdef WEAK_FORM_H
    MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M2D);
#else
    MatConvert(EtoF->E21, MATSAME, MAT_INITIAL_MATRIX, &M2D);
#endif
    MatScale(M2D, dt*10000.0);
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GM2);
    MatScale(GM2, dt*grav);
    MatMatMult(M2D, Uinv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M2DUinv);
    MatMatMult(M2DUinv, GM2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M2DUinvGM2);
    MatZeroEntries(SC);
    MatCopy(M2DUinvGM2, SC, DIFFERENT_NONZERO_PATTERN);
    MatScale(SC, -1.0);

    MatAssemblyBegin(SC, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  SC, MAT_FINAL_ASSEMBLY);

    MatZeroEntries(P);

    MatCopy(M1->M, P, DIFFERENT_NONZERO_PATTERN);

    //precon_assembled = true;

    MatDestroy(&M2D);
    MatDestroy(&GM2);
    MatDestroy(&M2DUinv);
    MatDestroy(&M2DUinvGM2);
    MatDestroy(&Uinv);
}

int _snes_function(SNES snes, Vec x, Vec f, void* ctx) {
    SWEqn* sw = (SWEqn*)ctx;

    if(sw->u_only) {
        sw->jfnk_vector_u(x, f);
    } else {
        sw->jfnk_vector(x, f);
    }

    return 0;
}

int _snes_jacobian(SNES snes, Vec x, Mat J, Mat P, void* ctx) {
    SWEqn* sw = (SWEqn*)ctx;

    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  J, MAT_FINAL_ASSEMBLY);

    if(sw->u_only) {
        sw->jfnk_precon_u(P);
    } else {
        sw->jfnk_precon(P);
    }

    MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  P, MAT_FINAL_ASSEMBLY);

    return 0;
}

void SWEqn::solve(Vec un, Vec hn, double _dt, bool save) {
    int ii, its;
    int n_null = 2;
    double norm_u, norm_h, norm_u0, norm_h0;
    Vec x, f, b, bu, bh;
    Vec* Zi;
    MatNullSpace null;
    Mat J, P;
    SNES snes;
    KSP kspFromSnes;
    PC pcFromSnes;
    SNESConvergedReason reason;

    dt = _dt;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &bh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &x);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &f);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &b);

    MatCreate(MPI_COMM_WORLD, &J);
    MatSetSizes(J, topo->n1l+topo->n2l, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, topo->nDofs1G+topo->nDofs2G);
    MatSetType(J, MATMPIAIJ);
    MatMPIAIJSetPreallocation(J, 2*quad->n*(quad->n+1), PETSC_NULL, 2*quad->n*(quad->n+1), PETSC_NULL);
    MatZeroEntries(J);

    MatCreate(MPI_COMM_WORLD, &P);
    MatSetSizes(P, topo->n1l+topo->n2l, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, topo->nDofs1G+topo->nDofs2G);
    MatSetType(P, MATMPIAIJ);
    MatMPIAIJSetPreallocation(P, 8*quad->n*(quad->n+1), PETSC_NULL, 8*quad->n*(quad->n+1), PETSC_NULL);
    MatSetOptionsPrefix(P, "precon_");
    MatZeroEntries(P);

    // solution vector
    VecCopy(un, ui);
    VecCopy(hn, hi);
    VecCopy(un, uj);
    VecCopy(hn, hj);
    repack(x, un, hn);

    // rhs vector
    MatMult(M1->M, ui, bu);
#ifdef WEAK_FORM_H
    MatMult(M2->M, hi, bh);
#else
    VecCopy(hi, bh);
#endif
    repack(b, bu, bh);

    // setup solver
    SNESCreate(MPI_COMM_WORLD, &snes);
    SNESSetFunction(snes, f,    _snes_function, (void*)this);
    SNESSetJacobian(snes, J, P, _snes_jacobian, (void*)this);
    SNESSetType(snes, SNESNEWTONTR);
#ifdef RIGHT
    SNESSetNPCSide(snes, PC_RIGHT);
#else
    SNESSetNPCSide(snes, PC_LEFT);
#endif
    SNESSetFromOptions(snes);

    // create the null space
    Zi = diagnose_null_space_vecs(ui, hi, n_null);
    MatNullSpaceCreate(MPI_COMM_WORLD, PETSC_FALSE, n_null, Zi, &null);
    MatSetNullSpace(J, null);

    // field split preconditioning
    SNESGetKSP(snes, &kspFromSnes);
    KSPGetPC(kspFromSnes, &pcFromSnes);
    //PCFieldSplitSetIS(pcFromSnes, "u", is_u);
    //PCFieldSplitSetIS(pcFromSnes, "h", is_h);

    //PCFieldSplitSetSchurPre(pcFromSnes, PC_FIELDSPLIT_SCHUR_PRE_USER, SC);
    SNESSolve(snes, b, x);

    unpack(x, un, hn);
    VecCopy(un, uj);
    VecCopy(hn, hj);

    VecAXPY(uj, -1.0, ui);
    VecAXPY(hj, -1.0, hi);
    VecNorm(uj, NORM_2, &norm_u);
    VecNorm(hj, NORM_2, &norm_h);
    VecNorm(ui, NORM_2, &norm_u0);
    VecNorm(hi, NORM_2, &norm_h0);
    SNESGetNumberFunctionEvals(snes, &its);
    SNESGetConvergedReason(snes, &reason);
    if(!rank) {
        cout << scientific;
        cout << "SNES converged as " << SNESConvergedReasons[reason] << " iteration: " << its << 
                "\t|u_j-u_i|/|u_j|: " << norm_u/norm_u0 << "\t|h_j-h_i|/|h_j|: " << norm_h/norm_h0 << endl;
    }

    if(save) {
        Vec wi;
        char fieldname[20];

        step++;
        curl(un, &wi);

        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(un, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hn, fieldname, step);

        VecDestroy(&wi);
    }

    VecDestroy(&x);
    VecDestroy(&f);
    VecDestroy(&b);
    VecDestroy(&bu);
    VecDestroy(&bh);
    MatDestroy(&J);
    MatDestroy(&P);
    SNESDestroy(&snes);
    MatNullSpaceDestroy(&null);
    for(ii = 0; ii < n_null; ii++) {
        VecDestroy(&Zi[ii]);
    }
    delete[] Zi;
}

void SWEqn::solve_u(Vec un, Vec hn, double _dt, bool save) {
    int its;
    double norm_u, norm_h, norm_u0, norm_h0;
    Vec x, f, b;
    Mat J, P;
    SNES snes;
    SNESConvergedReason reason;

    dt = _dt;
    u_only = true;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &x);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &f);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);

    MatCreate(MPI_COMM_WORLD, &J);
    MatSetSizes(J, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(J, MATMPIAIJ);
    MatMPIAIJSetPreallocation(J, 0, PETSC_NULL, 0, PETSC_NULL);
    MatZeroEntries(J);

    MatCreate(MPI_COMM_WORLD, &P);
    MatSetSizes(P, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(P, MATMPIAIJ);
    MatMPIAIJSetPreallocation(P, 8*quad->n*(quad->n+1), PETSC_NULL, 8*quad->n*(quad->n+1), PETSC_NULL);
    MatSetOptionsPrefix(P, "precon_");
    MatZeroEntries(P);

    // solution vector
    VecCopy(un, ui);
    VecCopy(hn, hi);
    VecCopy(un, uj);
    VecCopy(hn, hj);
    VecCopy(un, x);

    // rhs vector
    MatMult(M1->M, ui, b);

    // setup solver
    SNESCreate(MPI_COMM_WORLD, &snes);
    SNESSetFunction(snes, f,    _snes_function, (void*)this);
    SNESSetJacobian(snes, J, P, _snes_jacobian, (void*)this);
    SNESSetType(snes, SNESNEWTONTR);
#ifdef RIGHT
    SNESSetNPCSide(snes, PC_RIGHT);
#else
    SNESSetNPCSide(snes, PC_LEFT);
#endif
    SNESSetFromOptions(snes);

    SNESSolve(snes, b, x);

    VecCopy(x, un);
    VecCopy(un, uj);
    VecCopy(hn, hj);

    VecAXPY(uj, -1.0, ui);
    VecAXPY(hj, -1.0, hi);
    VecNorm(uj, NORM_2, &norm_u);
    VecNorm(hj, NORM_2, &norm_h);
    VecNorm(ui, NORM_2, &norm_u0);
    VecNorm(hi, NORM_2, &norm_h0);
    SNESGetNumberFunctionEvals(snes, &its);
    SNESGetConvergedReason(snes, &reason);
    if(!rank) {
        cout << scientific;
        cout << "SNES converged as " << SNESConvergedReasons[reason] << " iteration: " << its << 
                "\t|u_j-u_i|/|u_j|: " << norm_u/norm_u0 << "\t|h_j-h_i|/|h_j|: " << norm_h/norm_h0 << endl;
    }

    if(save) {
        Vec wi;
        char fieldname[20];

        step++;
        curl(un, &wi);

        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(un, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hn, fieldname, step);

        VecDestroy(&wi);
    }

    VecDestroy(&x);
    VecDestroy(&f);
    VecDestroy(&b);
    MatDestroy(&J);
    MatDestroy(&P);
    SNESDestroy(&snes);
}

void SWEqn::solve_explicit(Vec un, Vec hn, double _dt, bool save) {
    Vec F1, F2, Phi1, Phi2, wxu1, wxu2, bu, tu, uh, hh, d2u, d4u;

    dt = _dt;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hh);

    // first step
    VecCopy(un, ui);
    VecCopy(un, uj);
    VecCopy(hn, hi);
    VecCopy(hn, hj);

    // ...momentum
    diagnose_wxu(&wxu1);
    diagnose_Phi(&Phi1);
    VecZeroEntries(bu);
    MatMult(M1->M, un, bu);
    VecAXPY(bu, -dt, wxu1);
    MatMult(EtoF->E12, Phi1, tu);
    VecAXPY(bu, -dt, tu);
    if(do_visc) {
        laplacian(uj, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(bu, -dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
    VecZeroEntries(uh);
    KSPSolve(ksp, bu, uh);

    // ...continuity
    diagnose_F(&F1);
    MatMult(EtoF->E21, F1, hh);
    VecAYPX(hh, -dt, hn);

    // second step
    VecCopy(uh, ui);
    VecCopy(uh, uj);
    VecCopy(hh, hi);
    VecCopy(hh, hj);

    // ...momentum
    diagnose_wxu(&wxu2);
    diagnose_Phi(&Phi2);
    MatMult(M1->M, un, bu);
    VecAXPY(bu, -0.5*dt, wxu1);
    VecAXPY(bu, -0.5*dt, wxu2);
    MatMult(EtoF->E12, Phi1, tu);
    VecAXPY(bu, -0.5*dt, tu);
    MatMult(EtoF->E12, Phi2, tu);
    VecAXPY(bu, -0.5*dt, tu);
    if(do_visc) {
        laplacian(uj, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(bu, -dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
    VecZeroEntries(un);
    KSPSolve(ksp, bu, un);

    // ...continuity
    diagnose_F(&F2);
    VecAXPY(F2, 1.0, F1);
    MatMult(EtoF->E21, F2, hh);
    VecAYPX(hh, -0.5*dt, hn);
    VecCopy(hh, hn);

    if(save) {
        Vec wi;
        char fieldname[20];

        step++;
        curl(un, &wi);

        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(un, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hn, fieldname, step);

        VecDestroy(&wi);
    }

    VecDestroy(&bu);
    VecDestroy(&tu);
    VecDestroy(&uh);
    VecDestroy(&Phi1);
    VecDestroy(&Phi2);
    VecDestroy(&wxu1);
    VecDestroy(&wxu2);
    VecDestroy(&hh);
    VecDestroy(&F1);
    VecDestroy(&F2);
}

void SWEqn::diagnose_q(Vec u, Vec h, Vec* q) {
    Vec hl, du, mf;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &mf);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, q);
    VecZeroEntries(*q);

    VecScatterBegin(topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);

    m0h->assemble(hl);
    MatMult(E01M1, u, du);
    VecPointwiseMult(mf, m0->vg, fg);
    VecAXPY(du, 1.0, mf);
    VecPointwiseDivide(*q, du, m0h->vg);

    VecDestroy(&hl);
    VecDestroy(&du);
    VecDestroy(&mf);
}

Vec* SWEqn::diagnose_null_space_vecs(Vec u, Vec h, int n) {
    int ii;
    double norm;
    Vec q, qn, dq, q2, tmp2;
    KSP ksp2;
    Vec* Zi = new Vec[n];
    WtQmat* WQ = new WtQmat(topo, geom, edge);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &qn);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dq);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &q2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp2);
    VecZeroEntries(qn);

    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);

    diagnose_q(u, h, &q);
    VecCopy(q, qn);
    for(ii = 0; ii < n; ii++) {
      VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &Zi[ii]);

      // velocity section of the nth null space vector
      VecZeroEntries(dq);
      MatMult(NtoE->E10, qn, dq);
      VecScale(dq, -1.0);

      // pressure section of the nth null space vector
      VecPointwiseMult(qn, qn, q);
      MatMult(WQ->M, qn, tmp2);
      VecZeroEntries(q2);
      KSPSolve(ksp2, tmp2, q2);
      VecScale(q2, 1.0/(n + 2.0) - 1.0);

      repack(Zi[ii], dq, q2);

      VecNorm(Zi[ii], NORM_2, &norm);
      VecScale(Zi[ii], 1.0/norm);
    }

    delete WQ;
    VecDestroy(&q);
    VecDestroy(&qn);
    VecDestroy(&dq);
    VecDestroy(&q2);
    VecDestroy(&tmp2);
    KSPDestroy(&ksp2);

    return Zi;
}

SWEqn::~SWEqn() {
    ISDestroy(&is_u);
    ISDestroy(&is_h);
    MatDestroy(&SC);
    KSPDestroy(&ksp);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);
    VecScatterDestroy(&gtol_x);
    VecDestroy(&ui);
    VecDestroy(&hi);
    VecDestroy(&uj);
    VecDestroy(&hj);

    delete m0;
    delete m0h;
    delete M1;
    delete M2;
    delete U0;

    delete NtoE;
    delete EtoF;

    delete R;
    delete M1h;
    delete K;

    delete edge;
    delete node;
    delete quad;
}

void SWEqn::init0(Vec q, ICfunc* func) {
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

void SWEqn::init1(Vec u, ICfunc* func_x, ICfunc* func_y) {
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

void SWEqn::init2(Vec h, ICfunc* func) {
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

void SWEqn::err0(Vec ug, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms) {
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

void SWEqn::err1(Vec ug, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms) {
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

void SWEqn::err2(Vec ug, ICfunc* fu, double* norms) {
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

double SWEqn::int0(Vec ug) {
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

double SWEqn::int2(Vec ug) {
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

double SWEqn::intE(Vec ug, Vec hg) {
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

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*(grav*hq*hq + 0.5*hq*(uq[0]*uq[0] + uq[1]*uq[1]));
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

void SWEqn::writeConservation(double time, Vec u, Vec h, double mass0, double vort0, double ener0) {
    int rank;
    double mass, vort, ener;
    char filename[50];
    ofstream file;
    Vec wi;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    curl(u, &wi);

    mass = int2(h);
    vort = int0(wi);
    ener = intE(u, h);

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
    VecDestroy(&wi);
} 
