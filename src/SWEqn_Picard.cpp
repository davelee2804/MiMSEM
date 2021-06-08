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
#include "SWEqn_Picard.h"

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
//#define RAD_SPHERE 1.0
//#define W2_ALPHA (0.25*M_PI)
#define UP_VORT
#define H_MEAN 1.0e+4

using namespace std;

SWEqn::SWEqn(Topo* _topo, Geom* _geom) {
    PC pc;
    int ii, jj, size;
    int dof_proc;
    int* loc = new int[_topo->n1+_topo->n2];
    IS is_g, is_l;
    Vec xl, xg;

    topo = _topo;
    geom = _geom;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    grav = 9.80616*(RAD_SPHERE/RAD_EARTH);
    omega = 7.292e-5;
    del2 = viscosity();
    do_visc = true;
    step = 0;

    quad = new GaussLobatto(geom->quad->n);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // 0 form lumped mass matrix (vector)
    //m0 = new Pvec(topo, geom, node);
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

    // mass flux operator
    M1h = new Uhmat(topo, geom, node, edge);

    M0h = new Phmat(topo, geom, node);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

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
    DM1inv = NULL;
    ksp_helm = NULL;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);

    // create the [u,h] -> [x] vec scatter
    for(ii = 0; ii < topo->n1; ii++) {
        dof_proc = topo->loc1[ii] / topo->n1l;
        loc[ii] = dof_proc * (topo->n1l + topo->n2l) + topo->loc1[ii] % topo->n1l;
    }
    for(ii = 0; ii < topo->n2; ii++) {
        jj = ii + topo->n1;
        loc[jj] = rank*(topo->n1l + topo->n2l) + ii + topo->n1l;
    }

    ISCreateStride(MPI_COMM_SELF, topo->n1+topo->n2, 0, 1, &is_l);
    ISCreateGeneral(MPI_COMM_WORLD, topo->n1+topo->n2, loc, PETSC_COPY_VALUES, &is_g);

    VecCreateSeq(MPI_COMM_SELF, topo->n1+topo->n2, &xl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &xg);

    VecScatterCreate(xg, is_g, xl, is_l, &gtol_x);

    delete[] loc;
    ISDestroy(&is_l);
    ISDestroy(&is_g);
    VecDestroy(&xl);
    VecDestroy(&xg);

    topog = NULL;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &uil);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ujl);

    u_prev = NULL;
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
#ifdef W2_ALPHA
        fArray[ii] = 2.0*omega*( -cos(geom->s[ii][0])*cos(geom->s[ii][1])*sin(W2_ALPHA) + sin(geom->s[ii][1])*cos(W2_ALPHA) );
#else
        fArray[ii] = 2.0*omega*sin(geom->s[ii][1]);
#endif
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
    //VecPointwiseDivide(fg, PtQfxg, m0->vg);
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
void SWEqn::curl(Vec u, Vec *w) {
    Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);

    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    // diagonal mass matrix as vector
    //VecPointwiseDivide(*w, du, m0->vg);
    KSPSolve(ksp0, du, *w);

    VecDestroy(&du);
}

// dH/du = hu = F
void SWEqn::diagnose_F(Vec* F) {
    Vec hu, b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);
    VecZeroEntries(*F);
    VecZeroEntries(hu);

    // assemble the nonlinear rhs mass matrix (note that hl is a local vector)
    M1h->assemble(hi);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/3.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/6.0, b);

    M1h->assemble(hj);

    MatMult(M1h->M, ui, b);
    VecAXPY(hu, 1.0/6.0, b);

    MatMult(M1h->M, uj, b);
    VecAXPY(hu, 1.0/3.0, b);

    // solve the linear system
    KSPSolve(ksp, hu, *F);

    VecDestroy(&hu);
    VecDestroy(&b);
}

// dH/dh = (1/2)u^2 + gh = \Phi
// note: \Phi is in integral form here
//          \int_{\Omega} \gamma_h,\Phi_h d\Omega
void SWEqn::diagnose_Phi(Vec* Phi) {
    Vec b;

    //VecCreateSeq(MPI_COMM_SELF, topo->n1, &uil);
    //VecCreateSeq(MPI_COMM_SELF, topo->n1, &ujl);

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

    if(topog) {
        MatMult(M2->M, topog, b);
        VecAXPY(*Phi, grav, b);
    }

    VecDestroy(&b);
}

void SWEqn::diagnose_q(double _dt, Vec _ug, Vec _ul, Vec _h, Vec* qi) {
    Vec rhs, tmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, qi);

    MatMult(M0->M, fg, rhs);
    MatMult(E01M1, _ug, tmp);
    VecAXPY(rhs, 1.0, tmp);
#ifdef UP_VORT
    M0h->assemble_up(_ul, _h, _dt);
#else
    M0h->assemble(_h);
#endif
    KSPSolve(ksp0h, rhs, *qi);

    VecDestroy(&rhs);
    VecDestroy(&tmp);
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
    Vec xl, ul;
    PetscScalar *xArray, *uArray, *hArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, topo->n1 + topo->n2, &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);

    VecScatterBegin(gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(xl, &xArray);
    VecGetArray(ul, &uArray);
    VecGetArray(h, &hArray);
    for(ii = 0; ii < topo->n1; ii++) {
        uArray[ii] = xArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        hArray[ii] = xArray[ii+topo->n1];
    }
    VecRestoreArray(xl, &xArray);
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(h, &hArray);

    VecScatterBegin(topo->gtol_1, ul, u, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, ul, u, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&xl);
    VecDestroy(&ul);
}

void SWEqn::repack(Vec x, Vec u, Vec h) {
    Vec xl, ul;
    PetscScalar *xArray, *uArray, *hArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, topo->n1 + topo->n2, &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecScatterBegin(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(xl, &xArray);
    VecGetArray(ul, &uArray);
    VecGetArray(h, &hArray);
    for(ii = 0; ii < topo->n1; ii++) {
        xArray[ii] = uArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        xArray[ii+topo->n1] = hArray[ii];
    }
    VecRestoreArray(xl, &xArray);
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(h, &hArray);

    VecScatterBegin(gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&xl);
    VecDestroy(&ul);
}

void SWEqn::assemble_residual(Vec x, Vec f) {
    Vec F, Phi, fu, fh, utmp, htmp1, htmp2, d2u, d4u, fs, qi, qj, ql;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &fs);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);

    VecZeroEntries(fu);
    VecZeroEntries(fh);

    unpack(x, uj, hj);

    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    // assemble in the skew-symmetric parts of the vector
    diagnose_F(&F);
    diagnose_Phi(&Phi);

    // momentum terms
    MatMult(EtoF->E12, Phi, fu);

    // upwinded convective term
    diagnose_q(dt, ui, uil, hi, &qi);
    VecScatterBegin(topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble(ql, uil, dt);
    MatMult(R_up->M, F, utmp);
#else
    R->assemble(ql);
    MatMult(R->M, F, utmp);
#endif
    VecAXPY(fu, 0.5, utmp);

    diagnose_q(dt, uj, ujl, hj, &qj);
    VecScatterBegin(topo->gtol_0, qj, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qj, ql, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble(ql, ujl, dt);
    MatMult(R_up->M, F, utmp);
#else
    R->assemble(ql);
    MatMult(R->M, F, utmp);
#endif
    VecAXPY(fu, 0.5, utmp);
    //

    // continuity term
    MatMult(EtoF->E21, F, htmp1);
    MatMult(M2->M, htmp1, htmp2);
    VecAXPY(fh, 1.0, htmp2);
    repack(fs, fu, fh);

    // assemble the mass matrix terms
    VecZeroEntries(fu);
    VecZeroEntries(fh);

    MatMult(M1->M, uj, fu);
    MatMult(M1->M, ui, utmp);
    VecAXPY(fu, -1.0, utmp);

    if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, ui);
        VecAXPY(utmp, 0.5, uj);
        laplacian(utmp, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    MatMult(M2->M, hj, fh);
    MatMult(M2->M, hi, htmp1);
    VecAXPY(fh, -1.0, htmp1);

    repack(f, fu, fh);
    VecAXPY(f, dt, fs);

    // clean up
    VecDestroy(&fu);
    VecDestroy(&fh);
    VecDestroy(&utmp);
    VecDestroy(&htmp1);
    VecDestroy(&htmp2);
    VecDestroy(&F);
    VecDestroy(&Phi);
    VecDestroy(&fs);
    VecDestroy(&qi);
    VecDestroy(&qj);
    VecDestroy(&ql);
}

void SWEqn::assemble_operator(double dt) {
    int n2 = (topo->elOrd+1)*(topo->elOrd+1);
    int mm, mi, mf, ri, ci, dof_proc;
    int nCols;
    const int *cols;
    const double* vals;
    int cols2[9999];
    Mat Muh, Mhu;

    if(A) return;

    MatCreate(MPI_COMM_WORLD, &A);
    MatSetSizes(A, topo->n1l+topo->n2l, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, topo->nDofs1G+topo->nDofs2G);
    MatSetType(A, MATMPIAIJ);
    MatMPIAIJSetPreallocation(A, 16*n2, PETSC_NULL, 16*n2, PETSC_NULL);

    R->assemble(fl);
    MatAXPY(M1->M, 0.5*dt, R->M, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(M1->M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M1->M, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(M1->M, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(M1->M, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(M1->M, mm, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(M1->M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M1->M, MAT_FINAL_ASSEMBLY);

    // [u,h] block
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Muh);
    MatScale(Muh, 0.5*dt*grav);
    MatAssemblyBegin(Muh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Muh, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Muh, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Muh, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Muh, mm, &nCols, &cols, &vals);
    }

    // [h,u] block
    MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mhu);
    MatScale(Mhu, 0.5*dt*H_MEAN);
    MatAssemblyBegin(Mhu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Mhu, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Mhu, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Mhu, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Mhu, mm, &nCols, &cols, &vals);
    }

    // [h,h] block
    MatGetOwnershipRange(M2->M, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(M2->M, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(M2->M, mm, &nCols, &cols, &vals);
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);

    MatDestroy(&Muh);
    MatDestroy(&Mhu);

    MatDestroy(&M1->M);
    M1->assemble();
/*
    int n2 = (topo->elOrd+1)*(topo->elOrd+1);
    int mm, mi, mf, ri, ci, dof_proc;
    int nCols;
    const int *cols;
    const double* vals;
    int cols2[9999];
    Vec fl;
    Mat Muu, Muh, Mhu, Mhh;

    if(A) return;

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
    MatSetSizes(A, topo->n1l+topo->n2l, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, topo->nDofs1G+topo->nDofs2G);
    MatSetType(A, MATMPIAIJ);
    MatMPIAIJSetPreallocation(A, 16*n2, PETSC_NULL, 16*n2, PETSC_NULL);

    // [u,u] block // TODO: incorporate viscosity? time step = 1.0*dt?
    MatAssemblyBegin(Muu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Muu, MAT_FINAL_ASSEMBLY);
    MatCopy(M1->M, Muu, DIFFERENT_NONZERO_PATTERN);
    R->assemble(fl);
    MatAXPY(Muu, 0.5*dt, R->M, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(Muu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Muu, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Muu, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Muu, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Muu, mm, &nCols, &cols, &vals);
    }

    // [u,h] block
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Muh);
    MatScale(Muh, 0.5*dt*grav);
    MatAssemblyBegin(Muh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Muh, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Muh, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Muh, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Muh, mm, &nCols, &cols, &vals);
    }

    // [h,u] block
    MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mhu);
    MatScale(Mhu, 0.5*dt*H_MEAN);
    MatAssemblyBegin(Mhu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Mhu, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Mhu, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Mhu, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Mhu, mm, &nCols, &cols, &vals);
    }

    // [h,h] block
    MatAssemblyBegin(Mhh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Mhh, MAT_FINAL_ASSEMBLY);
    MatCopy(M2->M, Mhh, DIFFERENT_NONZERO_PATTERN);

    MatGetOwnershipRange(Mhh, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Mhh, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
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
*/
}

void SWEqn::assemble_operator_schur(double imp_dt) {
    int size;
    Mat M1inv;
    Mat M2D;
    PC pc;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    R->assemble(fl);
    MatAXPY(M1->M, imp_dt, R->M, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(M1->M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M1->M, MAT_FINAL_ASSEMBLY);
    coriolisMatInv(M1->M, &M1inv);

    MatDestroy(&M1->M);
    M1->assemble();

    MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M2D);
    MatMatMult(M2D, M1inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DM1inv);
    MatMatMult(DM1inv, E12M2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A);
    MatAYPX(A, -imp_dt*imp_dt*grav*H_MEAN, M2->M, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);

    KSPCreate(MPI_COMM_WORLD, &ksp_helm);
    KSPSetOperators(ksp_helm, A, A);
    KSPSetTolerances(ksp_helm, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_helm, KSPGMRES);
    KSPGetPC(ksp_helm, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp_helm, "ksp_helm_");
    KSPSetFromOptions(ksp_helm);

    MatDestroy(&M1inv);
    MatDestroy(&M2D);
}

void SWEqn::solve_schur(Vec Fu, Vec Fh, Vec _u, Vec _h, double imp_dt) {
    Vec rhs_h, rhs_u;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rhs_u);

    // pressure solve
    MatMult(DM1inv, Fu, rhs_h);
    VecScale(rhs_h, imp_dt*H_MEAN);
    VecAXPY(rhs_h, 1.0, Fh);
    KSPSolve(ksp_helm, rhs_h, _h);

    // velocity solve
    MatMult(E12M2, _h, rhs_u);
    VecAYPX(rhs_u, -imp_dt*grav, Fu);

    R->assemble(fl);
    MatAXPY(M1->M, imp_dt, R->M, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(M1->M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M1->M, MAT_FINAL_ASSEMBLY);

    KSPSolve(ksp, rhs_u, _u);

    MatDestroy(&M1->M);
    M1->assemble();

    VecDestroy(&rhs_h);
    VecDestroy(&rhs_u);
}

void SWEqn::solve(Vec un, Vec hn, double _dt, bool save) {
    int it = 0;
    double norm = 1.0e+9, norm_dx, norm_x;
    Vec x, f, dx;
    KSP kspA;

    dt = _dt;

    if(!A) assemble_operator(dt);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &x);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &f);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &dx);

    // solution vector
    VecCopy(un, ui);
    VecCopy(hn, hi);
    VecCopy(un, uj);
    VecCopy(hn, hj);
    repack(x, un, hn);

    unpack(x, un, hn);
    VecCopy(un, uj);
    VecCopy(hn, hj);

    KSPCreate(MPI_COMM_WORLD, &kspA);
    KSPSetOperators(kspA, A, A);
    KSPSetTolerances(kspA, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
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
            cout << it << "\t|x|: " << norm_x << "\t|dx|: " << norm_dx << "\t|dx|/|x|: " << norm << endl; 
        }
        it++;
    } while(norm > 1.0e-14);

    unpack(x, un, hn);

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
    VecDestroy(&dx);
    KSPDestroy(&kspA);
}

SWEqn::~SWEqn() {
    KSPDestroy(&ksp);
    KSPDestroy(&ksp0);
    KSPDestroy(&ksp0h);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);
    VecDestroy(&fl);
    VecScatterDestroy(&gtol_x);
    VecDestroy(&ui);
    VecDestroy(&hi);
    VecDestroy(&uj);
    VecDestroy(&hj);
    VecDestroy(&uil);
    VecDestroy(&ujl);
    if(u_prev) VecDestroy(&u_prev);
    if(topog) VecDestroy(&topog);
    if(A) MatDestroy(&A);
    if(DM1inv) MatDestroy(&DM1inv);
    if(ksp_helm) KSPDestroy(&ksp_helm);

    //delete m0;
    delete M0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete M1h;
    delete M0h;
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
    //VecPointwiseDivide(q, PQb, m0->vg);
    KSPSolve(ksp0, PQb, q);

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

void SWEqn::init2(Vec h, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    PetscScalar *bArray;
    KSP ksp2;
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

void SWEqn::err2(Vec ug, ICfunc* fu, double* norms) {
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

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ul, &array_1);
    VecGetArray(hg, &array_2);

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
    VecRestoreArray(hg, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&ul);
    VecDestroy(&hl);

    return global;
}

void SWEqn::writeConservation(double time, Vec u, Vec h, double mass0, double vort0, double ener0) {
    double mass, vort, ener;
    char filename[50];
    ofstream file;
    Vec wi;

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

void SWEqn::solve_imex(Vec un, Vec hn, double _dt, bool save) {
    int size;
    double en_con_err, upwind_tau = 120.0;
    char filename[50];
    ofstream file;
    Vec Fi, Fj, Phi, qi, qj, ql, ul, utmp, htmp, fu, up;
    Mat KT, KTD;
    KSP ksp_f;
    PC pc;

    dt = _dt;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Phi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &up);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fj);

    // 1. compute provisional velocity
    VecCopy(un, ui);
    VecCopy(un, uj);
    VecCopy(hn, hi);
    VecCopy(hn, hj);
    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);

    // F^n
    //M1h->assemble(hi);
    //MatMult(M1h->M, ui, utmp);
    K->assemble(uil);
    MatTranspose(K->M, MAT_INITIAL_MATRIX, &KT);
    MatMult(KT, hi, utmp);
    VecScale(utmp, 2.0);
    KSPSolve(ksp, utmp, Fi);

    // Phi^n
    MatMult(K->M, ui, htmp);
    MatMult(M2->M, hi, Phi);
    VecAYPX(Phi, grav, htmp);
    MatMult(EtoF->E12, Phi, fu);

    // q^n
    diagnose_q(upwind_tau, ui, uil, hi, &qi);
    VecScatterBegin(topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble(ql, uil, upwind_tau);
    MatMult(R_up->M, Fi, utmp);
#else
    R->assemble(ql);
    MatMult(R->M, Fi, utmp);
#endif
    VecAXPY(fu, 1.0, utmp);

    if(u_prev) {
        MatMult(M1->M, u_prev, utmp);
        VecAYPX(fu, -2.0*_dt, utmp);
    } else {
        MatMult(M1->M, ui, utmp);
        VecAYPX(fu, -_dt, utmp);
    }
    KSPSolve(ksp, fu, up);

    if(!u_prev) VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_prev);
    VecCopy(ui, u_prev);

    VecScatterBegin(topo->gtol_1, up, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, up, ujl, INSERT_VALUES, SCATTER_FORWARD);

    // 2. compute the semi-implict pressure advection 
    //M1h->assemble(hi);
    //MatMult(M1h->M, ui, fu);
    //MatMult(M1h->M, up, utmp);
    //VecAYPX(fu, 2.0, utmp);
    //VecScale(fu, 1.0/6.0);
    MatMult(KT, hi, utmp);
    K->assemble(ujl);
    MatTranspose(K->M, MAT_REUSE_MATRIX, &KT);
    MatMult(KT, hi, fu);
    VecAXPY(fu, 2.0, utmp);
    VecScale(fu, 1.0/3.0);
    KSPSolve(ksp, fu, Fi);

    VecZeroEntries(ul);
    VecAXPY(ul, 1.0/3.0, uil);
    VecAXPY(ul, 2.0/3.0, ujl);
    K->assemble(ul);
    MatTranspose(K->M, MAT_REUSE_MATRIX, &KT);
    MatMatMult(KT, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &KTD);

    MatMult(KTD, Fi, fu);
    MatMult(KT, hi, utmp);
    VecAYPX(fu, -_dt, utmp);
    MatAYPX(KTD, _dt, M1->M, DIFFERENT_NONZERO_PATTERN);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    KSPCreate(MPI_COMM_WORLD, &ksp_f);
    KSPSetOperators(ksp_f, KTD, KTD);
    KSPSetTolerances(ksp_f, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_f, KSPGMRES);
    KSPGetPC(ksp_f, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp_f, "adv_f_");
    KSPSetFromOptions(ksp_f);
    KSPSolve(ksp_f, fu, Fj);

    VecAXPY(Fj, 1.0, Fi);
    MatMult(EtoF->E21, Fj, hj);
    VecAYPX(hj, -_dt, hi);

    // 3. compute final velocity
    VecDestroy(&Phi);
    diagnose_Phi(&Phi);

    MatMult(EtoF->E12, Phi, fu);

#ifdef UP_VORT
    R_up->assemble(ql, uil, upwind_tau);
    MatMult(R_up->M, Fj, utmp);
#else
    R->assemble(qil);
    MatMult(R->M, Fj, utmp);
#endif
    VecAXPY(fu, 0.5, utmp);

    diagnose_q(upwind_tau, uj, ujl, hj, &qj);
    VecScatterBegin(topo->gtol_0, qj, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qj, ql, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble(ql, ujl, upwind_tau);
    MatMult(R_up->M, Fj, utmp);
#else
    R->assemble(ql);
    MatMult(R->M, Fj, utmp);
#endif
    VecAXPY(fu, 0.5, utmp);

    MatMult(M1->M, ui, utmp);
    VecAYPX(fu, -_dt, utmp);
    KSPSolve(ksp, fu, uj);

    VecCopy(uj, un);
    VecCopy(hj, hn);

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

    // write the energy conservation error
    VecAYPX(up, -1.0, uj);
    MatMult(M1->M, Fj, utmp);
    VecDot(Fj, up, &en_con_err);
    sprintf(filename, "output/conservation_err.dat");
    if(!rank) {
        file.open(filename, ios::out | ios::app);
        file << en_con_err << endl;
        file.close();
    }

    VecDestroy(&qi);
    VecDestroy(&qj);
    VecDestroy(&ql);
    VecDestroy(&ul);
    VecDestroy(&Phi);
    VecDestroy(&htmp);
    VecDestroy(&Fi);
    VecDestroy(&Fj);
    VecDestroy(&fu);
    VecDestroy(&utmp);
    VecDestroy(&up);
    MatDestroy(&KT);
    MatDestroy(&KTD);
    KSPDestroy(&ksp_f);
}

void SWEqn::rosenbrock_residuals(Vec _u, Vec _h, Vec _ul, Vec fu, Vec fh) {
    double upwind_tau = 120.0;
    Vec utmp, _F, qi, ql, htmp, _Phi;
    Mat KT;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &_Phi);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);

    K->assemble(_ul);
    MatTranspose(K->M, MAT_INITIAL_MATRIX, &KT);
    MatMult(KT, _h, utmp);
    VecScale(utmp, 2.0);
    KSPSolve(ksp, utmp, _F);

    MatMult(K->M, _u, htmp);
    MatMult(M2->M, _h, _Phi);
    VecAYPX(_Phi, grav, htmp);
    MatMult(EtoF->E12, _Phi, fu);

    diagnose_q(upwind_tau, _u, _ul, _h, &qi);
    VecScatterBegin(topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble(ql, _ul, upwind_tau);
    MatMult(R_up->M, _F, utmp);
#else
    R->assemble(ql);
    MatMult(R->M, _F, utmp);
#endif
    VecAXPY(fu, 1.0, utmp);

    MatMult(EtoF->E21, _F, htmp);
    MatMult(M2->M, htmp, fh);

    VecDestroy(&utmp);
    VecDestroy(&htmp);
    VecDestroy(&_F);
    VecDestroy(&_Phi);
    VecDestroy(&ql);
    MatDestroy(&KT);
}

void SWEqn::rosenbrock_solve(Vec _ui, Vec _uil, Vec _hi, Vec _uj, Vec _hj) {
    Vec fu, fh, du1, dh1, du2, dh2, _ul, utmp, htmp;
    double alpha = 1.0 + 0.5*sqrt(2.0);

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &_ul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dh1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dh2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);

    if(!A) assemble_operator_schur(alpha*dt);

    rosenbrock_residuals(_ui, _hi, _uil, fu, fh);
    VecScale(fu, -1.0*dt);
    VecScale(fh, -1.0*dt);
    solve_schur(fu, fh, du1, dh1, alpha*dt);

    VecCopy(_ui, _uj);
    VecCopy(_hi, _hj);
    VecAXPY(_uj, 1.0, du1);
    VecAXPY(_hj, 1.0, dh1);

    VecScatterBegin(topo->gtol_1, _uj, _ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, _uj, _ul, INSERT_VALUES, SCATTER_FORWARD);

    rosenbrock_residuals(_uj, _hj, _ul, fu, fh);
    VecScale(fu, -1.0*dt);
    VecScale(fh, -1.0*dt);
    MatMult(M1->M, du1, utmp);
    MatMult(M2->M, dh1, htmp);
    VecAXPY(fu, -2.0, utmp);
    VecAXPY(fh, -2.0, htmp);
    solve_schur(fu, fh, du2, dh2, alpha*dt);

/*
    VecCopy(_ui, _uj);
    VecCopy(_hi, _hj);
    VecAXPY(_uj, 1.5, du1);
    VecAXPY(_hj, 1.5, dh1);
    VecAXPY(_uj, 0.5, du2);
    VecAXPY(_hj, 0.5, dh2);
*/
    VecCopy(du1, utmp);
    VecCopy(dh1, htmp);
    VecAXPY(utmp, +1.0, _ui);
    VecAXPY(htmp, +1.0, _hi);
    VecCopy(du2, _uj);
    VecCopy(dh2, _hj);
    VecAXPY(_uj, +3.0, utmp);
    VecAXPY(_hj, +3.0, htmp);
    VecAXPY(_uj, -2.0, _ui);
    VecAXPY(_hj, -2.0, _hi);

    VecDestroy(&_ul);
    VecDestroy(&du1);
    VecDestroy(&dh1);
    VecDestroy(&du2);
    VecDestroy(&dh2);
    VecDestroy(&fu);
    VecDestroy(&fh);
    VecDestroy(&utmp);
    VecDestroy(&htmp);
}

void SWEqn::solve_implicit(Vec un, Vec hn, double _dt, bool save) {
    double upwind_tau = 120.0;
    Vec _F, Phi, qi, qj, ql, utmp, fu, fh;

    dt = _dt;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh);

    VecCopy(un, ui);
    VecCopy(hn, hi);
    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);

    rosenbrock_solve(ui, uil, hi, uj, hj);
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    // compute final velocity
    diagnose_Phi(&Phi);
    diagnose_F(&_F);
    MatMult(EtoF->E12, Phi, fu);

    diagnose_q(upwind_tau, ui, uil, hi, &qi);
    VecScatterBegin(topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble(ql, uil, upwind_tau);
    MatMult(R_up->M, _F, utmp);
#else
    R->assemble(ql);
    MatMult(R->M, _F, utmp);
#endif
    VecAXPY(fu, 0.5, utmp);

    diagnose_q(upwind_tau, uj, ujl, hj, &qj);
    VecScatterBegin(topo->gtol_0, qj, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qj, ql, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble(ql, ujl, upwind_tau);
    MatMult(R_up->M, _F, utmp);
#else
    R->assemble(ql);
    MatMult(R->M, _F, utmp);
#endif
    VecAXPY(fu, 0.5, utmp);

    MatMult(M1->M, ui, utmp);
    VecAYPX(fu, -_dt, utmp);
    KSPSolve(ksp, fu, uj);

    MatMult(EtoF->E21, _F, hj);
    VecAYPX(hj, -_dt, hi);

    VecCopy(uj, un);
    VecCopy(hj, hn);

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

    VecDestroy(&ql);
    VecDestroy(&qi);
    VecDestroy(&qj);
    VecDestroy(&_F);
    VecDestroy(&Phi);
    VecDestroy(&utmp);
    VecDestroy(&fu);
    VecDestroy(&fh);
}

void SWEqn::solve_rosenbrock(Vec un, Vec hn, double _dt, bool save) {
    Vec fu, fh, du1, dh1, du2, dh2, utmp, htmp, Phi, _F, qi, qj, ql;
    double upwind_tau = 120.0;
    double alpha = 1.0 + 0.5*sqrt(2.0);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dh1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dh2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);

    if(!A) assemble_operator_schur(alpha*dt);

    VecCopy(un, ui);
    VecCopy(hn, hi);
    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);

    rosenbrock_residuals(ui, hi, uil, fu, fh);
    VecScale(fu, dt);
    VecScale(fh, dt);
    solve_schur(fu, fh, du1, dh1, alpha*dt);

    VecCopy(ui, uj);
    VecCopy(hi, hj);
    VecAXPY(uj, 1.0, du1);
    VecAXPY(hj, 1.0, dh1);
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    rosenbrock_residuals(uj, hj, ujl, fu, fh);
    VecScale(fu, dt);
    VecScale(fh, dt);
    MatMult(M1->M, du1, utmp);
    MatMult(M2->M, dh1, htmp);
    VecAXPY(fu, -2.0, utmp);
    VecAXPY(fh, -2.0, htmp);
    solve_schur(fu, fh, du2, dh2, alpha*dt);

    VecCopy(ui, uj);
    VecCopy(hi, hj);
    VecAXPY(uj, 1.5, du1);
    VecAXPY(hj, 1.5, dh1);
    VecAXPY(uj, 0.5, du2);
    VecAXPY(hj, 0.5, dh2);

    // compute the provisional state; tilde{u,h}_2 = {u,h}_2 - 3{u,h}_1 + 2{u,h}_0
    VecCopy(du1, utmp);
    VecCopy(dh1, htmp);
    VecAXPY(utmp, +1.0, ui);
    VecAXPY(htmp, +1.0, hi);
    VecCopy(du2, uj);
    VecCopy(dh2, hj);
    VecAXPY(uj, +3.0, utmp);
    VecAXPY(hj, +3.0, htmp);
    VecAXPY(uj, -2.0, ui);
    VecAXPY(hj, -2.0, hi);

    // energetically balanced solve
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    diagnose_Phi(&Phi);
    diagnose_F(&_F);
    MatMult(EtoF->E12, Phi, fu);

    diagnose_q(upwind_tau, ui, uil, hi, &qi);
    VecScatterBegin(topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qi, ql, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble(ql, uil, upwind_tau);
    MatMult(R_up->M, _F, utmp);
#else
    R->assemble(ql);
    MatMult(R->M, _F, utmp);
#endif
    VecAXPY(fu, 0.5, utmp);

    diagnose_q(upwind_tau, uj, ujl, hj, &qj);
    VecScatterBegin(topo->gtol_0, qj, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qj, ql, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble(ql, ujl, upwind_tau);
    MatMult(R_up->M, _F, utmp);
#else
    R->assemble(ql);
    MatMult(R->M, _F, utmp);
#endif
    VecAXPY(fu, 0.5, utmp);

    MatMult(M1->M, ui, utmp);
    VecAYPX(fu, -_dt, utmp);
    KSPSolve(ksp, fu, uj);

    MatMult(EtoF->E21, _F, hj);
    VecAYPX(hj, -_dt, hi);

    VecCopy(uj, un);
    VecCopy(hj, hn);

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

    VecDestroy(&du1);
    VecDestroy(&dh1);
    VecDestroy(&du2);
    VecDestroy(&dh2);
    VecDestroy(&fu);
    VecDestroy(&fh);
    VecDestroy(&utmp);
    VecDestroy(&htmp);
    //VecDestroy(&Phi);
    //VecDestroy(&_F);
    //VecDestroy(&qi);
    //VecDestroy(&qj);
    VecDestroy(&ql);
}

void SWEqn::coriolisMatInv(Mat A, Mat* Ainv) {
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

