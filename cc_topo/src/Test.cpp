#include <iostream>

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
#include "Test.h"

using namespace std;

Test::Test(SWEqn* _sw) {
    sw = _sw;
}

Test::~Test() {
}

void Test::vorticity(ICfunc* fu, ICfunc* fv) {
    Vec w, u;
    char filename[20];

    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &u);

    sw->init1(u, fu, fv);
    sw->diagnose_w(u, &w);

    sprintf(filename,"test_w");
    sw->geom->write0(w, filename, 0);

    VecDestroy(&w);
    VecDestroy(&u);
}

void Test::gradient(ICfunc* fh) {
    Vec u, h, M2h, dM2h;
    KSP ksp;
    PC pc;
    char filename[20];

    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &u);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &dM2h);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &h);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &M2h);

    sw->init2(h, fh);
    MatMult(sw->M2->M, h, M2h);
    MatMult(sw->EtoF->E12, M2h, dM2h);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, sw->M1->M, sw->M1->M);
    KSPSetTolerances(ksp, 1.0e-12, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*sw->topo->elOrd*(sw->topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp,"test_grad_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, dM2h, u);

    sprintf(filename,"test_grad");
    sw->geom->write1(u, filename, 0);

    KSPDestroy(&ksp);
    VecDestroy(&u);
    VecDestroy(&dM2h);
    VecDestroy(&h);
    VecDestroy(&M2h);
}

void Test::convection(ICfunc* fu, ICfunc* fv) {
    Vec w, wl, u, Ru, c;
    KSP ksp;
    PC pc;
    char filename[20];

    VecCreateSeq(MPI_COMM_SELF, sw->topo->n0, &wl);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &u);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &c);

    sw->init1(u, fu, fv);
    sw->diagnose_w(u, &w);
    
    VecScatterBegin(sw->topo->gtol_0, w, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(sw->topo->gtol_0, w, wl, INSERT_VALUES, SCATTER_FORWARD);

    sw->R->assemble(wl);

    MatMult(sw->R->M, u, Ru);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, sw->M1->M, sw->M1->M);
    KSPSetTolerances(ksp, 1.0e-12, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*sw->topo->elOrd*(sw->topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp,"test_conv_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, Ru, c);

    sprintf(filename,"test_conv");
    sw->geom->write1(c, filename, 0);

    VecDestroy(&w);
    VecDestroy(&wl);
    VecDestroy(&u);
    VecDestroy(&Ru);
    VecDestroy(&c);
}

void Test::massFlux(ICfunc* fu, ICfunc* fv, ICfunc* fh) {
    Vec u, F, h, hl;
    KSP ksp;
    PC pc;
    char filename[20];

    VecCreateSeq(MPI_COMM_SELF, sw->topo->n2, &hl);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &u);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &h);

    sw->init1(u, fu, fv);
    sw->init2(h, fh);

    VecScatterBegin(sw->topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(sw->topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, sw->M1->M, sw->M1->M);
    KSPSetTolerances(ksp, 1.0e-12, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*sw->topo->elOrd*(sw->topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp,"test_mf_");
    KSPSetFromOptions(ksp);

    sw->diagnose_F(u, hl, ksp, &F);

    sprintf(filename,"test_mf");
    sw->geom->write1(F, filename, 0);

    KSPDestroy(&ksp);
    VecDestroy(&hl);
    VecDestroy(&u);
    VecDestroy(&h);
    VecDestroy(&F);
}

void Test::kineticEnergy(ICfunc* fu, ICfunc* fv) {
    Vec ul, u, u2, k;
    KSP ksp;
    char filename[20];

    VecCreateSeq(MPI_COMM_SELF, sw->topo->n1, &ul);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &u);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &u2);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &k);

    sw->init1(u, fu, fv);

    VecScatterBegin(sw->topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(sw->topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(k);
    sw->K->assemble(ul);
    MatMult(sw->K->M, u, u2);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, sw->M2->M, sw->M2->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOptionsPrefix(ksp,"test_ke_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, u2, k);

    sprintf(filename,"test_ke");
    sw->geom->write2(k, filename, 0);

    KSPDestroy(&ksp);
    VecDestroy(&ul);
    VecDestroy(&u);
    VecDestroy(&u2);
    VecDestroy(&k);
}
