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
#include "ThermalShallowWater.h"

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
//#define W2_ALPHA (0.25*M_PI)
#define UP_VORT

using namespace std;

ThermalShallowWater::ThermalShallowWater(Topo* _topo, Geom* _geom) {
    Vec diag, ones;
    PC pc;
    int size;

    topo = _topo;
    geom = _geom;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    grav = 9.80616*(RAD_SPHERE/RAD_EARTH);
    omega = 7.292e-5;
    del2 = viscosity();
    do_visc = true;
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

    // adjoint differential operators (curl and grad)
    MatMatMult(NtoE->E01, M1->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E01M1);
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E12M2);

    // rotational operator
    R = new RotMat(topo, geom, node, edge);

    // mass flux operator
    M1h = new Uhmat(topo, geom, node, edge);

    WQ = new WtQmat(topo, geom, edge);
    MatTranspose(WQ->M, MAT_INITIAL_MATRIX, &WQT);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    M2h = new Whmat(topo, geom, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

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

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &si);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &sj);

#ifdef UP_VORT
    P_up = new P_up_mat(topo, geom, node);
    R_up = new RotMat_up(topo, geom, node, edge);
    KSPCreate(MPI_COMM_WORLD, &ksp_p);
    KSPSetOperators(ksp_p, P_up->M, P_up->M);
    KSPSetTolerances(ksp_p, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_p, KSPGMRES);
    KSPGetPC(ksp_p, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp_p, "p_up_");
    KSPSetFromOptions(ksp_p);
#endif
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &uil);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ujl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dhl);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &sil);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &sjl);

    G_s = M1sT = NULL;

    MatCreate(MPI_COMM_WORLD, &M1inv);
    MatSetSizes(M1inv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M1inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M1inv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatZeroEntries(M1inv);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &diag);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ones);
    VecSet(ones, 1.0);
    MatGetDiagonal(M1->M, diag);
    VecPointwiseDivide(diag, ones, diag);
    MatDiagonalSet(M1inv, diag, INSERT_VALUES);
    VecDestroy(&diag);
    VecDestroy(&ones);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double ThermalShallowWater::viscosity() {
    double ae = 4.0*M_PI*RAD_SPHERE*RAD_SPHERE;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void ThermalShallowWater::coriolis() {
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
void ThermalShallowWater::curl(Vec u, Vec *w, bool add_coriolis) {
    Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);

    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    if(add_coriolis) {
        Vec tmp;
        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &tmp);
        VecPointwiseMult(tmp, m0->vg, fg);
        VecAXPY(du, 1.0, tmp);
        VecDestroy(&tmp);
    }
    // diagonal mass matrix as vector
    VecPointwiseDivide(*w, du, m0->vg);

    VecDestroy(&du);
}

// dH/du = hu = F
void ThermalShallowWater::diagnose_F(Vec* F) {
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

// dH/dh = (1/2)(u^2 + S) = \Phi, S = hs
// note: \Phi is in integral form here
//          \int_{\Omega} \gamma_h,\Phi_h d\Omega
void ThermalShallowWater::diagnose_Phi(Vec* Phi) {
    Vec b;

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
    MatMult(M2->M, si, b);
    VecAXPY(*Phi, 1.0/4.0, b);

    MatMult(M2->M, sj, b);
    VecAXPY(*Phi, 1.0/4.0, b);

    VecDestroy(&b);
}

void ThermalShallowWater::diagnose_q(Vec* qi, Vec* qj) {
    Vec rhs, tmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, qi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, qj);

    // first time level
    VecPointwiseMult(rhs, m0->vg, fg);
    MatMult(E01M1, ui, tmp);
    VecAXPY(rhs, 1.0, tmp);
#ifdef UP_VORT
    P_up->assemble_h(uil, hi, dt);
#else
    P_up->assemble_h(NULL, hi, dt);
#endif
    KSPSolve(ksp_p, rhs, *qi);

    // second time level
    VecPointwiseMult(rhs, m0->vg, fg);
    MatMult(E01M1, uj, tmp);
    VecAXPY(rhs, 1.0, tmp);
#ifdef UP_VORT
    P_up->assemble_h(ujl, hj, dt);
#else
    P_up->assemble_h(NULL, hj, dt);
#endif
    KSPSolve(ksp_p, rhs, *qj);

    VecDestroy(&rhs);
    VecDestroy(&tmp);
}

void ThermalShallowWater::diagnose_s(Vec* _si, Vec* _sj) {
    Vec rhs;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, _si);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, _sj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &rhs);

    // first time level
    MatMult(WQT, si, rhs);
#ifdef UP_VORT
    P_up->assemble_h(uil, hi, dt);
#else
    P_up->assemble_h(NULL, hi, dt);
#endif
    KSPSolve(ksp_p, rhs, *_si);

    // second time level
    MatMult(WQT, sj, rhs);
#ifdef UP_VORT
    P_up->assemble_h(ujl, hj, dt);
#else
    P_up->assemble_h(NULL, hj, dt);
#endif
    KSPSolve(ksp_p, rhs, *_sj);

    VecDestroy(&rhs);
}

void ThermalShallowWater::laplacian(Vec u, Vec* ddu) {
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
    curl(u, &Cu, false);

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

void ThermalShallowWater::assemble_residual(Vec fu, Vec fh, Vec fs) {
    MatReuse reuse = (!M1sT) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    Vec F, Phi, utmp1, utmp2, htmp1, htmp2, d2u, d4u, qi, qj, qil, qjl, _si, _sj;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp2);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &qil);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &qjl);

    VecZeroEntries(fu);
    VecZeroEntries(fh);
    VecZeroEntries(fs);

    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    // assemble in the skew-symmetric parts of the vector
    diagnose_F(&F);
    diagnose_Phi(&Phi);

    // momentum terms
    MatMult(EtoF->E12, Phi, fu);

    // upwinded convective and buoyancy terms
    diagnose_q(&qi, &qj);
    diagnose_s(&_si, &_sj);

    VecScatterBegin(topo->gtol_0, qi, qil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qi, qil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_0, qj, qjl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qj, qjl, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    R_up->assemble_2ndOrd(qil, uil, qjl, ujl, dt);
    MatMult(R_up->M, F, utmp1);
#else
    VecAXPY(qil, 1.0, qjl);
    VecScale(qil, 0.5);
    R->assemble(qil);
    MatMult(R->M, F, utmp1);
#endif
    VecAXPY(fu, 1.0, utmp1);

    // pressure gradient term
    VecCopy(hi, htmp1);
    VecAXPY(htmp1, 1.0, hj);
    VecScale(htmp1, 0.25); // dH/dS = h/2
    MatMult(M2->M, htmp1, htmp2);
    MatMult(EtoF->E12, htmp2, utmp1);
    KSPSolve(ksp, utmp1, utmp2);
    // scatter the pressure gradient forward for use in the preconditioner
    VecScatterBegin(topo->gtol_1, utmp2, dhl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, utmp2, dhl, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_0, _si, sil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, _si, sil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_0, _sj, sjl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, _sj, sjl, INSERT_VALUES, SCATTER_FORWARD);
#ifdef UP_VORT
    M1h->assemble_0_up(sil, uil, sjl, ujl, dt);
#else    
    VecAXPY(sil, 1.0, sjl);
    VecScale(sil, 0.5);
    M1h->assemble_0(ql);
#endif
    MatMult(M1h->M, utmp2, utmp1);
    MatTranspose(M1h->M, reuse, &M1sT);
    VecAXPY(fu, 1.0, utmp1);

    VecScale(fu, dt);
    VecCopy(uj, utmp1);
    VecAXPY(utmp1, -1.0, ui);
    MatMult(M1->M, utmp1, utmp2);
    VecAXPY(fu, +1.0, utmp2);

    if(do_visc) {
        VecZeroEntries(utmp1);
        VecAXPY(utmp1, 0.5, ui);
        VecAXPY(utmp1, 0.5, uj);
        laplacian(utmp1, &d2u);
        laplacian(d2u, &d4u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    // continuity equation
    MatMult(EtoF->E21, F, htmp1);
    VecScale(htmp1, dt);
    VecAXPY(htmp1, +1.0, hj);
    VecAXPY(htmp1, -1.0, hi);
    MatMult(M2->M, htmp1, fh);

    // buoyancy equation
    MatMult(M1sT, F, utmp1);
    KSPSolve(ksp, utmp1, utmp2); // temperature flux
    MatMult(EtoF->E21, utmp2, htmp1);
    VecScale(htmp1, dt);
    VecAXPY(htmp1, +1.0, sj);
    VecAXPY(htmp1, -1.0, si);
    MatMult(M2->M, htmp1, fs);

    // clean up
    VecDestroy(&utmp1);
    VecDestroy(&utmp2);
    VecDestroy(&htmp1);
    VecDestroy(&htmp2);
    VecDestroy(&F);
    VecDestroy(&Phi);
    VecDestroy(&qi);
    VecDestroy(&qj);
    VecDestroy(&qil);
    VecDestroy(&qjl);
    VecDestroy(&_si);
    VecDestroy(&_sj);
}

void ThermalShallowWater::solve_schur(Vec fu, Vec fh, Vec fs, Vec du, Vec dh, Vec ds) {
    MatReuse reuse = (!G_s) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    Vec wg, wl;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &wg);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);

    // M_u
    curl(uj, &wg, true);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    R->assemble(wl);
    MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);

    // G_s: 0.5(\nabla  + (\nabla h)/h)
    K->assemble(dhl); // includes the 0.5 factor
    MatTranspose(K->M, reuse, &KT);
    M2h->assemble_inverse(hj);
    MatMatMult(KT, M2h->M, reuse, PETSC_DEFAULT, &KTM2inv);
    MatMatMult(KTM2inv, M2->M, reuse, PETSC_DEFAULT, &G_s);
    MatAXPY(G_s, 0.5, E12M2, DIFFERENT_NONZERO_PATTERN);
    MatScale(G_s, 0.5*dt);
    
    // D_h: \nabla\cdot(h\cdot)
    MatMatMult(M2->M, EtoF->E21, reuse, PETSC_DEFAULT, &M2D);
    MatMatMult(M2D, M1inv, reuse, PETSC_DEFAULT, &M2DM1inv);
    M1h->assemble(hj);
    MatMatMult(M2DM1inv, M1h->M, reuse, PETSC_DEFAULT, &D_h);
    MatScale(D_h, 0.5*dt);

    // D_s: s\nabla\cdot
    M1h->assemble(sj);
    MatMatMult(M1h->M, EtoF->E21, reuse, PETSC_DEFAULT, &D_s);
    MatScale(D_s, 0.5*dt);

    // Q  : u\cdot\nabla _s (_s = s/h)
    K->assemble(ujl); // 0.5 factor included here
    MatMatMult(K->M, EtoF->E12, reuse, PETSC_DEFAULT, &KDT);
    MatMatMult(KDT, M2->Minv, reuse, PETSC_DEFAULT, &KDTM2inv);
    //M1h->assemble_0(sjl);
    //MatMatMult(KDTM2inv, M1h->M, reuse, PETSC_DEFAULT, &Q);
    MatScale(Q, dt);


    VecDestroy(&wl);
    VecDestroy(&wg);
}

void ThermalShallowWater::solve(Vec un, Vec hn, Vec sn, double _dt, bool save) {
    int it = 0;
    double norm = 1.0e+9, norm_dx, norm_x;
    Vec x, f, dx;
    //KSP kspA;

    dt = _dt;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &x);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &f);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &dx);

    // solution vector
    VecCopy(un, ui);
    VecCopy(hn, hi);
    VecCopy(un, uj);
    VecCopy(hn, hj);

    VecCopy(un, uj);
    VecCopy(hn, hj);

    //KSPCreate(MPI_COMM_WORLD, &kspA);
    //KSPSetOperators(kspA, A, A);
    //KSPSetTolerances(kspA, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    //KSPSetOptionsPrefix(kspA, "A_");
    //KSPSetFromOptions(kspA);

    do {
        //assemble_residual(x, f);
        //VecScale(f, -1.0);
        //KSPSolve(kspA, f, dx);
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

    if(save) {
        Vec wi;
        char fieldname[20];

        step++;
        curl(un, &wi, false);

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
    //KSPDestroy(&kspA);
}

ThermalShallowWater::~ThermalShallowWater() {
#ifdef UP_VORT
    delete P_up;
    KSPDestroy(&ksp_p);
#endif
    KSPDestroy(&ksp);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    MatDestroy(&WQT);
    MatDestroy(&M1sT);
    MatDestroy(&KT);
    MatDestroy(&M1inv);
    VecDestroy(&fg);
    VecDestroy(&ui);
    VecDestroy(&hi);
    VecDestroy(&si);
    VecDestroy(&uj);
    VecDestroy(&hj);
    VecDestroy(&sj);
    VecDestroy(&uil);
    VecDestroy(&ujl);
    VecDestroy(&dhl);
    VecDestroy(&sil);
    VecDestroy(&sjl);

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete M1h;
    delete K;
    delete WQ;
    delete M2h;

    delete edge;
    delete node;
    delete quad;
}

void ThermalShallowWater::init0(Vec q, ICfunc* func) {
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

void ThermalShallowWater::init1(Vec u, ICfunc* func_x, ICfunc* func_y) {
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

void ThermalShallowWater::init2(Vec h, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    PetscScalar *bArray;
    KSP ksp2;
    Vec bl, bg, WQb;

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

    KSPDestroy(&ksp2);
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void ThermalShallowWater::err0(Vec ug, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms) {
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

void ThermalShallowWater::err1(Vec ug, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms) {
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

void ThermalShallowWater::err2(Vec ug, ICfunc* fu, double* norms) {
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

double ThermalShallowWater::int0(Vec ug) {
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

double ThermalShallowWater::int2(Vec ug) {
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

double ThermalShallowWater::intE(Vec ug, Vec hg) {
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

void ThermalShallowWater::writeConservation(double time, Vec u, Vec h, double mass0, double vort0, double ener0) {
    double mass, vort, ener;
    char filename[50];
    ofstream file;
    Vec wi;

    curl(u, &wi, false);

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
