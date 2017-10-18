#include <petsc.h>
#include <petscis.h>
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
    Vec vl, vg;

    topo = _topo;
    geom = _geom;

    grav = 9.8;
    omega = 7.2921150e-5;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // initialise the vec scatter objects for nodes/edges/faces
    VecCreateSeq(MPI_COMM_WORLD, topo->n0, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vg);
    VecScatterCreate(vg, topo->is_g_0, vl, topo->is_l_0, &gtol_0);
    VecDestroy(&vl);
    VecDestroy(&vg);

    VecCreateSeq(MPI_COMM_WORLD, topo->n1, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &vg);
    VecScatterCreate(vg, topo->is_g_1, vl, topo->is_l_1, &gtol_1);
    VecDestroy(&vl);
    VecDestroy(&vg);

    VecCreateSeq(MPI_COMM_WORLD, topo->n2, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &vg);
    VecScatterCreate(vg, topo->is_g_2, vl, topo->is_l_2, &gtol_2);
    VecDestroy(&vl);
    VecDestroy(&vg);

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
// assumes diagonal 0 form mass matrix
void SWEqn::coriolis() {
    PtQmat* PtQ = new PtQmat(topo, node);
    PetscScalar *fArray;
    Vec fxl, fxg, PtQfxg;

    // initialise the coriolis vector (local and global)
    VecCreateSeq(MPI_COMM_WORLD, topo->n0, &fl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg);

    // evaluate the coriolis term at nodes
    VecCreateSeq(MPI_COMM_WORLD, topo->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < topo->n0; ii++) {
        fArray[ii] = 2.0*omega*geom->x[ii][2];
    }
    VecRestoreArray(fxl, &fArray);

    // scatter array to global vector
    VecScatterBegin(gtol_0, fxl, fxg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(gtol_0, fxl, fxg, ADD_VALUES, SCATTER_REVERSE);

    // project vector onto 0 forms
    VecCreateSeq(MPI_COMM_WORLD, topo->n0, &PtQfxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PtQfxg);
    VecZeroEntries(PtQfxg);
    MatMult(PtQ->M, fxg, PtQfxg);
    // diagonal mass matrix as vector
    VecPointwiseDivide(fg, PtQfxg, m0->vg);
    
    // scatter to back to local vector
    VecScatterBegin(gtol_0, PtQfxg, PtQfxl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol_0, PtQfxg, PtQfxl, INSERT_VALUES, SCATTER_FORWARD);

    delete PtQ;
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

// derive vorticity (global vector) as \omega = curl u + f
// assumes diagonal 0 form mass matrix
void SWEqn::diagnose_w(Vec u, Vec *w) {
    Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);
    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    VecAYPX(du, 1.0, f);
    // diagonal mass matrix as vector
    VecPointwiseDivide(w, du, m0->vg);

    VecDestroy(&du);
}

void SWEqn::diagnose_F(Vec u, Vec h, Vec* hu) {
    Vec Fu;
    KSP ksp;

    // assemble the nonlinear rhs mass matrix
    F->assemble(h);

    // get the rhs vector
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fu);
    VecZeroEntries(Fu);
    MatMult(F->M, u, Fu);

    // solve the linear system
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, hu);

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

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &uu);
    VecZeroEntries(uu);
    MatMult(K->M, u0, uu);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hh);
    VecZeroEntries(hh);
    MatMult(M2->M, h0, hh);
    VecAYPX(hh, grav, uu);
    
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dh);
    VecZeroEntries(dh);
    MatMult(EtoF->E12, hh, dh);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Rv);
    VecZeroEntries(Rv);
    MatMult(R->M, u0, Rv);
    VecAYPX(dh, 1.0, Rv);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu);
    VecZeroEntries(Mu);
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
    
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h1);
    VecZeroEntries(h1);
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
    VecDestroy(&fl);
    VecDestroy(&fg);

    VecScatterDestroy(&gtol_0);
    VecScatterDestroy(&gtol_1);
    VecScatterDestroy(&gtol_2);

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
