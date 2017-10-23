#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscksp.h>

#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"

using namespace std;

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
    VecCreateMPI(MPI_COMM_WORLD, topo->n0, PETSC_DETERMINE, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vg);
    VecScatterCreate(vg, topo->is_g_0, vl, topo->is_l_0, &gtol_0);
    VecDestroy(&vl);
    VecDestroy(&vg);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1, PETSC_DETERMINE, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &vg);
    VecScatterCreate(vg, topo->is_g_1, vl, topo->is_l_1, &gtol_1);
    VecDestroy(&vl);
    VecDestroy(&vg);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2, PETSC_DETERMINE, &vl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &vg);
    VecScatterCreate(vg, topo->is_g_2, vl, topo->is_l_2, &gtol_2);
    VecDestroy(&vl);
    VecDestroy(&vg);

    // 0 form lumped mass matrix (vector)
    m0 = new Pvec(topo, geom, node);

    // 1 form mass matrix
    M1 = new Umat(topo, geom, node, edge);

    // 2 form mass matrix
    M2 = new Wmat(topo, geom, edge);

    // incidence matrices
    NtoE = new E10mat(topo);
    EtoF = new E21mat(topo);

    // adjoint differential operators
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
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void SWEqn::coriolis() {
    int ii;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fxl, fxg, PtQfxg;

    // initialise the coriolis vector (local and global)
    VecCreateMPI(MPI_COMM_WORLD, topo->n0, PETSC_DETERMINE, &fl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg);

    // evaluate the coriolis term at nodes
    VecCreateMPI(MPI_COMM_WORLD, topo->n0, PETSC_DETERMINE, &fxl);
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
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PtQfxg);
    VecZeroEntries(PtQfxg);
    MatMult(PtQ->M, fxg, PtQfxg);
    // diagonal mass matrix as vector
    VecPointwiseDivide(fg, PtQfxg, m0->vg);
    
    // scatter to back to local vector
    VecScatterBegin(gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);

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
    // diagonal mass matrix as vector
    VecPointwiseDivide(*w, du, m0->vg);
    // add the (0 form) coriolis vector
    VecAYPX(*w, 1.0, fg);

    VecDestroy(&du);
}

void SWEqn::diagnose_F(Vec u, Vec hl, KSP ksp, Vec* hu) {
    Vec Fu;

    // assemble the nonlinear rhs mass matrix (note that hl is a local vector)
    F->assemble(hl);

    // get the rhs vector
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fu);
    VecZeroEntries(Fu);
    MatMult(F->M, u, Fu);

    // solve the linear system
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, hu);
    KSPSolve(ksp, Fu, *hu);
    VecDestroy(&Fu);
}

void SWEqn::solve(Vec ui, Vec hi, Vec uf, Vec hf, double dt) {
    Vec wi, wj, uj, hj;
    Vec gh, uu, wv, hu;
    Vec Ui, Hi, Uj, Hj;
    Vec bu;
    Vec wl, ul, hl;
    KSP ksp;

    // initialize vectors
    VecCreateMPI(MPI_COMM_WORLD, topo->n0, PETSC_DETERMINE, &wl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1, PETSC_DETERMINE, &ul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2, PETSC_DETERMINE, &hl);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Hj);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &wv);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &gh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &uu);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);

    // initialize the linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPSetTolerances(ksp, 1.0e-12, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);

    /*** first step ***/

    // diagnose the initial vorticity
    diagnose_w(ui, &wi);

    // scatter initial vectors to local versions
    VecScatterBegin(gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(gtol_2, hi, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol_2, hi, hl, INSERT_VALUES, SCATTER_FORWARD);

    // momemtum equation
    K->assemble(ul);
    R->assemble(wl);

    VecZeroEntries(uu);
    MatMult(K->M, ui, uu);

    VecZeroEntries(gh);
    MatMult(M2->M, hi, gh);
    VecAYPX(gh, grav, uu);      // M2.(K + gh)
    
    VecZeroEntries(Ui);
    MatMult(EtoF->E12, gh, Ui); // E12.M2.(K + gh)

    VecZeroEntries(wv);
    MatMult(R->M, ui, wv);
    VecAYPX(Ui, 1.0, wv);       // M1.(w + f)Xu + E12.M2.(K + gh)

    VecZeroEntries(bu);
    MatMult(M1->M, ui, bu);     // M1.u
    VecAXPY(bu, -dt, Ui);       // M1.u - dt{ M1.(w + f)Xu + E12.M2.(K + gh) }

    // linear solve for uj
    VecZeroEntries(uj);
    KSPSolve(ksp, bu, uj);

    // mass equation
    diagnose_F(ui, hl, ksp, &hu);
    
    VecZeroEntries(Hi);
    MatMult(EtoF->E21, hu, Hi);

    VecZeroEntries(hj);
    VecAXPY(hj, 1.0, hi);
    VecAXPY(hj, -dt, Hi);

    VecDestroy(&hu);

    /*** second step ***/

    // diagnose the half step vorticity
    diagnose_w(uj, &wj);

    // scatter half step vectors to local versions
    VecScatterBegin(gtol_0, wj, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol_0, wj, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(gtol_1, uj, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol_1, uj, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);

    // momentum equation
    K->assemble(ul);
    R->assemble(wl);

    VecZeroEntries(uu);
    MatMult(K->M, uj, uu);

    VecZeroEntries(gh);
    MatMult(M2->M, hj, gh);
    VecAYPX(gh, grav, uu);      // M2.(K + gh)
    
    VecZeroEntries(Uj);
    MatMult(EtoF->E12, gh, Uj); // E12.M2.(K + gh)

    VecZeroEntries(wv);
    MatMult(R->M, uj, wv);
    VecAYPX(Uj, 1.0, wv);       // M1.(w + f)Xu + E12.M2.(K + gh)

    VecZeroEntries(bu);
    MatMult(M1->M, uj, bu);     // M1.u
    VecAXPY(bu, -0.5*dt, Ui);   // M1.u - dt{ M1.(w + f)Xu + E12.M2.(K + gh) }
    VecAXPY(bu, -0.5*dt, Uj);

    // linear solve for uf
    VecZeroEntries(uf);
    KSPSolve(ksp, bu, uf);

    // mass equation
    diagnose_F(uj, hl, ksp, &hu);
    
    VecZeroEntries(Hj);
    MatMult(EtoF->E21, hu, Hj);

    VecZeroEntries(hf);
    VecAXPY(hf, 1.0, hi);
    VecAXPY(hf, -0.5*dt, Hi);
    VecAXPY(hf, -0.5*dt, Hj);

    VecDestroy(&hu);

    // clean up
    VecDestroy(&wl);
    VecDestroy(&ul);
    VecDestroy(&hl);

    VecDestroy(&wi);
    VecDestroy(&wj);
    VecDestroy(&uj);
    VecDestroy(&hj);

    VecDestroy(&wv);
    VecDestroy(&uu);
    VecDestroy(&gh);

    VecDestroy(&Ui);
    VecDestroy(&Uj);
    VecDestroy(&Hi);
    VecDestroy(&Hj);

    VecDestroy(&bu);

    KSPDestroy(&ksp);
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

    delete edge;
    delete node;
    delete quad;
}
