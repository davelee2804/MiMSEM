#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscksp.h>

#include "Basis.h"
#include "ElMats.h"
#include "Topo.h"
#include "Geom.h"
#include "Assembly.h"
#include "SWEqn.h"

SWEqn::SWEqn(Topo* _topo, Geom* _geom) {
    topo = _topo;
    geom = _geom;

    grav = 9.8;
    omega = 7.2921150e-5;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // 0 form lumped mass matrix (vector)
    m0 = new Pvec(topo, node);

    // 1 form mass matrix
    M1 = new Umat(topo, node, edge);

    // 2 form mass matrix
    M2 = new Wmat(topo, edge);

    // incidence matrices
    NtoE = new E10mat(topo);
    EtoF = new E21mat(topo);

    // adjoint differential operators
    MatMatMult(NtoE->E01, M1->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E01M1);
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E12M2);

    // rotational operator
    R = new RotMat(topo, node, edge);

    // mass flux operator
    F = new Uhmat(topo, node, edge);

    // kinetic energy operator
    K = new WtQUmat(topo, node, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();
}

// project coriolis term onto 0 forms
void SWEqn::coriolis() {
    int ii;
    PtQmat* PtQ = new PtQmat(topo, node);
    PetscScalar* fVals = new PetscScalar[topo->n0];
    Vec fx, PtQfx;
    PetscScalar *aVals, *bVals;

    for(ii = 0; ii < topo->n0; ii++) {
        fVals[ii] = 2.0*omega*geom->x[ii][2];
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n0, topo->nDofs0G, &fx);
    VecSetLocalToGlobalMapping(fx, topo->map0);
    VecSetValues(fx, topo->n0, topo->loc0, fVals, INSERT_VALUES);
    VecAssemblyBegin(fx);
    VecAssemblyEnd(fx);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, topo->nDofs0G, &PtQfx);
    VecSetLocalToGlobalMapping(PtQfx, topo->map0);
    MatMult(PtQ->M, fx, PtQfx);

    VecGetArray(m0->v, &aVals);
    VecGetArray(PtQfx, &bVals);
    for(ii = 0; ii < topo->n0; ii++) {
        fVals[ii] = bVals[ii]/aVals[ii];
    }
    VecRestoreArray(m0->v, &aVals);
    VecRestoreArray(PtQfx, &bVals);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, topo->nDofs0G, &f);
    VecSetLocalToGlobalMapping(f, topo->map0);
    VecSetValues(f, topo->n0, topo->loc0, fVals, INSERT_VALUES);
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);

    delete PtQ;
    delete[] fVals;
    VecDestroy(&fx);
    VecDestroy(&PtQfx);
}

// derive vorticity
void SWEqn::diagnose_w(Vec u, Vec *w) {
    int ii;
    PetscScalar *duVals, *m0Vals, *wVals;
    Vec du;

    wVals = new PetscScalar[topo->n0];

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, topo->nDofs0G, &du);
    VecSetLocalToGlobalMapping(du, topo->map0);
    MatMult(E01M1, u, du);
    VecAYPX(du, 1.0, f);

    VecGetArray(du, &duVals);
    VecGetArray(m0->v, &m0Vals);

    for(ii = 0; ii < topo->n0; ii++) {
        wVals[ii] = duVals[ii]/m0Vals[ii];
    }
    VecRestoreArray(du, &duVals);
    VecRestoreArray(m0->v, &m0Vals);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, topo->nDofs0G, w);
    VecSetLocalToGlobalMapping(*w, topo->map0);
    VecSetValues(*w, topo->n0, topo->loc0, wVals, INSERT_VALUES);
    VecAssemblyBegin(*w);
    VecAssemblyEnd(*w);

    delete[] wVals;
    VecDestroy(&du);
}

void SWEqn::diagnose_F(Vec u, Vec h, Vec* hu) {
    Vec Fu;
    KSP ksp;

    F->assemble(h);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1, topo->nDofs1G, &Fu);
    VecSetLocalToGlobalMapping(Fu, topo->map1);
    MatMult(F->M, u, Fu);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1, topo->nDofs1G, hu);
    VecSetLocalToGlobalMapping(*hu, topo->map1);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPSetTolerances(ksp, 1.0e-12, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSolve(ksp, Fu, *hu);

    VecDestroy(&Fu);
    KSPDestroy(&ksp);
}

void SWEqn::solve(Vec u0, Vec h0, double dt) {
    Vec w0;
    Vec hu0;
    Vec uu;
    Vec hh;
    Vec dh;
    Vec Rv;
    Vec Mu;
    Vec u1;
    Vec h1;
    KSP ksp;

    // first step, momemtum equation
    diagnose_w(u0, &w0);

    K->assemble(u0);
    R->assemble(w0);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2, topo->nDofs2G, &uu);
    VecSetLocalToGlobalMapping(uu, topo->map2);
    MatMult(K->M, u0, uu);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2, topo->nDofs2G, &hh);
    VecSetLocalToGlobalMapping(hh, topo->map2);
    MatMult(M2->M, h0, hh);

    VecAYPX(hh, grav, uu);
    
    VecCreateMPI(MPI_COMM_WORLD, topo->n1, topo->nDofs1G, &dh);
    VecSetLocalToGlobalMapping(dh, topo->map1);
    MatMult(EtoF->E12, hh, dh);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1, topo->nDofs1G, &Rv);
    VecSetLocalToGlobalMapping(Rv, topo->map1);
    MatMult(R->M, u0, Rv);

    VecAYPX(dh, 1.0, Rv);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1, topo->nDofs1G, &Mu);
    VecSetLocalToGlobalMapping(Mu, topo->map1);
    MatMult(M1->M, u0, Mu);

    VecAYPX(dh, -dt, Mu);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPSetTolerances(ksp, 1.0e-12, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSolve(ksp, dh, u1);
    KSPDestroy(&ksp);

    // first step, mass equation
    diagnose_F(u0, h0, &hu0);
    
    VecCreateMPI(MPI_COMM_WORLD, topo->n2, topo->nDofs2G, &h1);
    VecSetLocalToGlobalMapping(h1, topo->map2);
    MatMult(EtoF->E21, hu0, h1);
    VecAYPX(h1, -dt, h0);

    // second step, momentum equation

    // second step, mass equation

    VecDestroy(&w0);
    VecDestroy(&hu0);
    VecDestroy(&uu);
    VecDestroy(&hh);
    VecDestroy(&dh);
    VecDestroy(&Rv);
    VecDestroy(&Mu);
    VecDestroy(&u1);
    VecDestroy(&h1);
}
    

SWEqn::~SWEqn() {
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&f);

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;

    delete quad;
    delete node;
    delete edge;
}
