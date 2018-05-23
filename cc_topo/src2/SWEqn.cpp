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
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
//#define RAD_SPHERE 1.0
//#define W2_ALPHA (0.25*M_PI)

using namespace std;

SWEqn::SWEqn(Topo* _topo, Geom* _geom) {
    PC pc;

    topo = _topo;
    geom = _geom;

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
    F = new Uhmat(topo, geom, node, edge);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

    // initialize the linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp,"sw_");
    KSPSetFromOptions(ksp);
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
    VecScatterEnd(topo->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);

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

// derive vorticity (global vector) as \omega = curl u + f
// assumes diagonal 0 form mass matrix
void SWEqn::diagnose_w(Vec u, Vec *w, bool add_f) {
    Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);

    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    // diagonal mass matrix as vector
    VecPointwiseDivide(*w, du, m0->vg);
    // add the (0 form) coriolis vector
    if(add_f) {
        VecAYPX(*w, 1.0, fg);
    }

    VecDestroy(&du);
}

void SWEqn::diagnose_F(Vec u, Vec hl, Vec* hu) {
    Vec Fu;

    // assemble the nonlinear rhs mass matrix (note that hl is a local vector)
    F->assemble(hl);

    // get the rhs vector
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fu);
    VecZeroEntries(Fu);
    MatMult(F->M, u, Fu);

    // solve the linear system
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, hu);
    VecZeroEntries(*hu);
    KSPSolve(ksp, Fu, *hu);
    VecDestroy(&Fu);
}

void SWEqn::_massEqn(Vec hi, Vec uj, Vec hj, Vec hf, double dt) {
    Vec hl, hu, Fj, dF;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecScatterBegin(topo->gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);

    F->assemble(hl);
    MatMult(F->M, uj, hu);
    KSPSolve(ksp, hu, Fj);
    MatMult(EtoF->E21, Fj, dF);
    VecCopy(hi, hf);
    VecAXPY(hf, -dt, dF);

    VecDestroy(&hl);
    VecDestroy(&hu);
    VecDestroy(&Fj);
    VecDestroy(&dF);
}

void SWEqn::_momentumEqn(Vec ui, Vec uj, Vec hj, Vec uf, double dt) {
    Vec wj, wl, ul, Ru, Ku, Mh, dh, bu;

    diagnose_w(uj, &wj, true);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);

    VecScatterBegin(topo->gtol_0, wj, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wj, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, uj, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, uj, ul, INSERT_VALUES, SCATTER_FORWARD);

    R->assemble(wl);
    K->assemble(ul);

    MatMult(R->M, uj, Ru);
    MatMult(K->M, uj, Ku);
    MatMult(M2->M, hj, Mh);
    VecAYPX(Mh, grav, Ku);
    MatMult(EtoF->E12, Mh, dh);
    VecAXPY(dh, 1.0, Ru);

    MatMult(M1->M, ui, bu);
    VecAXPY(bu, -dt, dh);
    KSPSolve(ksp, bu, uf);

    VecDestroy(&wj);
    VecDestroy(&wl);
    VecDestroy(&ul);
    VecDestroy(&bu);
    VecDestroy(&dh);
    VecDestroy(&Ru);
    VecDestroy(&Ku);
    VecDestroy(&Mh);
}

// RK2 time integrator
void SWEqn::solve_RK2(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save) {
    int rank;
    char fieldname[20];
    Vec wf, uh, hh;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hh);

    /*** half step ***/
    if(!rank) cout << "half step..." << endl;

    _massEqn(hi, ui, hi, hh, 0.5*dt);
    _momentumEqn(ui, ui, hi, uh, 0.5*dt);

    /*** full step ***/
    if(!rank) cout << "full step..." << endl;

    _massEqn(hi, uh, hh, hf, dt);
    _momentumEqn(ui, uh, hh, uf, dt);

    if(!rank) cout << "...done." << endl;

    // write fields
    if(save) {
        step++;
        diagnose_w(uf, &wf, false);

        sprintf(fieldname, "vorticity");
        geom->write0(wf, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(uf, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hf, fieldname, step);

        VecDestroy(&wf);
    }

    VecDestroy(&uh);
    VecDestroy(&hh);
}

void SWEqn::boundaryInt(Vec ui, Vec hi, Vec Ku, Vec* bi) {
    int mp1 = quad->n + 1;
    double kph, scale;
    PetscScalar *kArray, *fArray;
    Vec fi, fl, ki, kl;
    KSP ksp2;
    Krhs* Fk = new Krhs(topo, geom, node, edge);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &kl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &ki);

    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPSetOptionsPrefix(ksp2,"sw_2_");
    KSPSetFromOptions(ksp2);

    KSPSolve(ksp2, Ku, ki);
    VecAXPY(ki, grav, hi);

    VecScatterBegin(topo->gtol_2, ki, kl, ADD_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, ki, kl, ADD_VALUES, SCATTER_FORWARD);

    VecGetArray(kl, &kArray);
    VecGetArray(fl, &fArray);

    for(int ey = 0; ey < topo->nElsX; ey++) {
        for(int ex = 0; ex < topo->nElsX; ex++) {
            int* inds0 = topo->elInds0_l(ex, ey);
            for(int qi = 0; qi < mp1*mp1; qi++) {
                geom->interp2_g(ex, ey, qi%mp1, qi/mp1, kArray, &kph);
                if(qi == 0 || qi == mp1 - 1 || qi == mp1*(mp1-1) || qi == mp1*mp1-1) {
                    scale = 0.25;
                }
                else if(qi/mp1 == 0 || qi/mp1 == mp1-1 || qi%mp1 == 0 || qi%mp1 == mp1-1) {
                    scale = 0.5;
                }
                else {
                    scale = 0.0;
                }
                fArray[inds0[qi]] = scale*kph;
            }
        }
    }

    VecRestoreArray(kl, &kArray);
    VecRestoreArray(fl, &fArray);

    VecScatterBegin(topo->gtol_0, fl, fi, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, fl, fi, ADD_VALUES, SCATTER_REVERSE);

    VecZeroEntries(fl);

    VecScatterBegin(topo->gtol_0, fi, fl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, fi, fl, INSERT_VALUES, SCATTER_FORWARD);

    Fk->assemble(fl, bi);

    VecDestroy(&kl);
    VecDestroy(&fl);
    VecDestroy(&ki);
    VecDestroy(&fi);
    KSPDestroy(&ksp2);
    delete Fk;
}

void SWEqn::_massTend(Vec ui, Vec hi, Vec *Fh) {
    Vec hl, hu, Fi;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecScatterBegin(topo->gtol_2, hi, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, hi, hl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fi);

    F->assemble(hl);
    MatMult(F->M, ui, hu);
    KSPSolve(ksp, hu, Fi);
    MatMult(EtoF->E21, Fi, *Fh);

    VecDestroy(&hl);
    VecDestroy(&hu);
    VecDestroy(&Fi);
}

void SWEqn::_momentumTend(Vec ui, Vec hi, Vec *Fu) {
    Vec wl, ul, wi, Ru, Ku, Mh, d2u, d4u;//, bi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, Fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);

    diagnose_w(ui, &wi, true);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);

    R->assemble(wl);
    K->assemble(ul);

    MatMult(R->M, ui, Ru);
    MatMult(K->M, ui, Ku);
    MatMult(M2->M, hi, Mh);
    VecAYPX(Mh, grav, Ku);
    MatMult(EtoF->E12, Mh, *Fu);
    VecAXPY(*Fu, 1.0, Ru);

    // add in the biharmonic voscosity
    if(do_visc) {
        laplacian(ui, &d2u);
        laplacian(d2u, &d4u);
        VecAXPY(*Fu, 1.0, d4u);
    }

    // TODO: add in the bernoulli function flux terms
    //boundaryInt(ui, hi, Ku, &bi);
    //VecAXPY(bi, 1.0, *Fu);
    //VecDestroy(&bi);
 
    VecDestroy(&wl);
    VecDestroy(&ul);
    VecDestroy(&wi);
    VecDestroy(&Ru);
    VecDestroy(&Ku);
    VecDestroy(&Mh);
    if(do_visc) {
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
}

// RK2 time integrator (stiffly stable scheme)
void SWEqn::solve_RK2_SS(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save) {
    int rank;
    char fieldname[20];
    Vec wf, Ui, Uj, Hi, Hj, Mu, bu, uj, hj;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);

    MatMult(M1->M, ui, Mu);

    // first step
    if(!rank) cout << "first step...." << endl;

    _momentumTend(ui, hi, &Ui);
    _massTend(ui, hi, &Hi);

    VecCopy(Mu, bu);
    VecAXPY(bu, -dt, Ui);
    KSPSolve(ksp, bu, uj);

    VecCopy(hi, hj);
    VecAXPY(hj, -dt, Hi);

    // second step
    if(!rank) cout << "second step..." << endl;

    _momentumTend(uj, hj, &Uj);
    _massTend(uj, hj, &Hj);

    VecCopy(Mu, bu);
    VecAXPY(bu, -0.5*dt, Ui);
    VecAXPY(bu, -0.5*dt, Uj);
    KSPSolve(ksp, bu, uf);

    VecCopy(hi, hf);
    VecAXPY(hf, -0.5*dt, Hi);
    VecAXPY(hf, -0.5*dt, Hj);

    if(!rank) cout << "...done." << endl;

    // write fields
    if(save) {
        step++;
        diagnose_w(uf, &wf, false);

        sprintf(fieldname, "vorticity");
        geom->write0(wf, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(uf, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hf, fieldname, step);

        VecDestroy(&wf);
    }

    VecDestroy(&Mu);
    VecDestroy(&bu);
    VecDestroy(&uj);
    VecDestroy(&hj);
    VecDestroy(&Ui);
    VecDestroy(&Uj);
    VecDestroy(&Hi);
    VecDestroy(&Hj);
}

void SWEqn::_massEuler(Vec ui, Vec hi, Vec uj, Vec hj, Vec hf, KSP ksp2, double dt) {
    Vec hl, hui, huj, Fi, dF, WdF, b;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecScatterBegin(topo->gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &huj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &WdF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);

    F->assemble(hl);
    MatMult(F->M, ui, hui);
    MatMult(F->M, uj, huj);
    VecAXPY(hui, 1.0, huj);
    KSPSolve(ksp, hui, Fi);

    MatMult(EtoF->E21, Fi, dF);
    MatMult(M2->M, dF, WdF);

    MatMult(M2->M, hi, b);
    VecAXPY(b, -0.5*dt, WdF);

    KSPSolve(ksp2, b, hf);

    VecDestroy(&hl);
    VecDestroy(&hui);
    VecDestroy(&huj);
    VecDestroy(&Fi);
    VecDestroy(&dF);
    VecDestroy(&WdF);
    VecDestroy(&b);
}

void SWEqn::_momentumEuler(Vec ui, Vec hi, Vec uj, Vec hj, Vec uf, double dt) {
    Vec Ru, Ku, Mhi, Mhj, bu, bF;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mhi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mhj);

    VecZeroEntries(bu);
    VecZeroEntries(bF);
    VecZeroEntries(Ru);
    VecZeroEntries(Ku);
    VecZeroEntries(Mhi);
    VecZeroEntries(Mhj);

    MatMult(R->M, ui, Ru);
    MatMult(K->M, ui, Ku);
    MatMult(M2->M, hi, Mhi);
    MatMult(M2->M, hj, Mhj);

    VecAXPY(Mhi, 1.0, Mhj);
    VecScale(Mhi, 0.5*grav);
    VecAXPY(Mhi, 1.0, Ku);
    MatMult(EtoF->E12, Mhi, bF);
    VecAXPY(bF, 1.0, Ru);

    MatMult(M1->M, ui, bu);
    VecAXPY(bu, -dt, bF);

    VecZeroEntries(uf);
    KSPSolve(ksp, bu, uf);

    VecDestroy(&bu);
    VecDestroy(&bF);
    VecDestroy(&Ru);
    VecDestroy(&Ku);
    VecDestroy(&Mhi);
    VecDestroy(&Mhj);
}

void SWEqn::solve_EEC(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save) {
    bool done;
    int iter, rank;
    char fieldname[20];
    PetscScalar unorm, hnorm;
    Vec wi, wl, ul, uj, hj, du, dh;
    KSP ksp2;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dh);

    // initialize the linear solver for 2 forms
    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPSetOptionsPrefix(ksp2,"sw_2_");
    KSPSetFromOptions(ksp2);

    // assemble the convective and kinetic energy terms at the current time level
    diagnose_w(ui, &wi, true);

    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);

    R->assemble(wl);
    K->assemble(ul);

    VecCopy(ui, uj);
    VecCopy(hi, hj);

    done = false;
    iter = 0;
    unorm = 1.0e+9;
    hnorm = 1.0e+9;
    do {
        _massEuler(ui, hi, uj, hj, hf, ksp2, dt);
        //_momentumEuler(ui, hi, uj, hj, uf, dt);
        _momentumEuler(ui, hi, uj, hf, uf, dt);

        VecZeroEntries(du);
        VecAXPY(du, +1.0, uf);
        VecAXPY(du, -1.0, uj);
        VecNorm(du, NORM_2, &unorm);
        unorm /= topo->nDofs1G;
        VecCopy(uf, uj);

        VecZeroEntries(dh);
        VecAXPY(dh, +1.0, hf);
        VecAXPY(dh, -1.0, hj);
        VecNorm(dh, NORM_2, &hnorm);
        hnorm /= topo->nDofs2G;
        VecCopy(hf, hj);

        if(!rank) cout << iter << "\t|u|: " << unorm << "\t|h|: " << hnorm << endl;
        iter++;

        if(iter > 100) done = true;
        //if(unorm < 1.0e-3 && hnorm < 1.0e-3) done = true;
        if(unorm < 1.0e-6 && hnorm < 2.0e-2) done = true;
    } while(!done);

    // write fields
    if(save) {
        step++;
        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(uf, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hf, fieldname, step);
    }

    VecDestroy(&wi);
    VecDestroy(&wl);
    VecDestroy(&ul);
    VecDestroy(&uj);
    VecDestroy(&du);
    VecDestroy(&hj);
    VecDestroy(&dh);
    KSPDestroy(&ksp2);
}

void SWEqn::laplacian(Vec ui, Vec* ddu) {
    Vec Du, Cu, RCu, GDu, MDu, dMDu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, ddu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &RCu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &GDu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMDu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &MDu);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, ui, Du);

    // grad (weak form)
    MatMult(M2->M, Du, MDu);
    MatMult(EtoF->E12, MDu, dMDu);
    KSPSolve(ksp, dMDu, GDu);

    /*** rotational component ***/
    // curl (weak form)
    diagnose_w(ui, &Cu, false);

    // rot (strong form)
    MatMult(NtoE->E10, Cu, RCu);

    // add rotational and divergent components
    VecCopy(GDu, *ddu);
    VecAXPY(*ddu, +1.0, RCu); // TODO: check sign here

    VecScale(*ddu, del2);

    VecDestroy(&Cu);
    VecDestroy(&RCu);
    VecDestroy(&GDu);
    VecDestroy(&dMDu);
    VecDestroy(&Du);
    VecDestroy(&MDu);
}

SWEqn::~SWEqn() {
    KSPDestroy(&ksp);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
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
    VecScatterEnd(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

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
    KSPSetOptionsPrefix(ksp2,"init2_");
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

void SWEqn::writeConservation(double time, Vec ui, Vec hi, double mass0, double vort0, double ener0) {
    int rank;
    double mass, vort, ener;
    char filename[50];
    ofstream file;
    Vec wi;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    diagnose_w(ui, &wi, false);

    mass = int2(hi);
    vort = int0(wi);
    ener = intE(ui, hi);

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
