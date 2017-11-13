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

using namespace std;
int step = 0;

SWEqn::SWEqn(Topo* _topo, Geom* _geom) {
    topo = _topo;
    geom = _geom;

    grav = 9.80616;
    omega = 7.292e-5;

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
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg);

    // evaluate the coriolis term at nodes
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < topo->n0; ii++) {
        fArray[ii] = 2.0*omega*geom->x[ii][2];
    }
    VecRestoreArray(fxl, &fArray);

    // scatter array to global vector
    VecScatterBegin(topo->gtol_0, fxl, fxg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, fxl, fxg, ADD_VALUES, SCATTER_REVERSE);

    // project vector onto 0 forms
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PtQfxg);
    VecZeroEntries(PtQfxg);
    MatMult(PtQ->M, fxg, PtQfxg);
    // diagonal mass matrix as vector
    VecPointwiseDivide(fg, PtQfxg, m0->vg);
    
    // scatter to back to local vector
    VecScatterBegin(topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);

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

void SWEqn::solve(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save) {
    int rank;
    char fieldname[20];
    Vec wi, wj, uj, hj;
    Vec gh, uu, wv, hu;
    Vec Ui, Hi, Uj, Hj;
    Vec bu;
    Vec wl, ul, hl;
    KSP ksp;
    PC pc;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // initialize vectors
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);

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
    KSPGetPC(ksp,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp,"sw_");
    KSPSetFromOptions(ksp);

    /*** first step ***/
    if(!rank) cout << "half step..." << endl;

    // diagnose the initial vorticity
    diagnose_w(ui, &wi);

    // scatter initial vectors to local versions
    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_2, hi, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, hi, hl, INSERT_VALUES, SCATTER_FORWARD);

    // momemtum equation
    if(!rank) cout << "\tmomentum eqn" << endl;
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
    if(!rank) cout << "\tcontinuity eqn" << endl;
    diagnose_F(ui, hl, ksp, &hu);
    
    VecZeroEntries(Hi);
    MatMult(EtoF->E21, hu, Hi);

    VecZeroEntries(hj);
    VecAXPY(hj, 1.0, hi);
    VecAXPY(hj, -dt, Hi);

    VecDestroy(&hu);

    /*** second step ***/
    if(!rank) cout << "full step..." << endl;

    // diagnose the half step vorticity
    diagnose_w(uj, &wj);

    // scatter half step vectors to local versions
    VecScatterBegin(topo->gtol_0, wj, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wj, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_1, uj, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, uj, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);

    // momentum equation
    if(!rank) cout << "\tmomentum eqn" << endl;
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
    if(!rank) cout << "\tcontinuity eqn" << endl;
    diagnose_F(uj, hl, ksp, &hu);
    
    VecZeroEntries(Hj);
    MatMult(EtoF->E21, hu, Hj);

    VecZeroEntries(hf);
    VecAXPY(hf, 1.0, hi);
    VecAXPY(hf, -0.5*dt, Hi);
    VecAXPY(hf, -0.5*dt, Hj);

    VecDestroy(&hu);

    if(!rank) cout << "...done." << endl;

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
    KSP ksp;
    PC pc;
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
                bArray[inds0[ii]] = func_x(geom->x[inds0[ii]]);
                bArray[inds0[ii]+topo->n0] = func_y(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);

    // create a new vec scatter object to handle vector quantity on nodes
    loc02 = new int[2*topo->n0];
    for(ii = 0; ii < topo->n0; ii++) {
        loc02[ii] = topo->loc0[ii];
        loc02[ii+topo->n0] = topo->loc0[ii] + topo->nDofs0G;
    }
    ISCreateStride(MPI_COMM_WORLD, 2*topo->n0, 0, 1, &isl);
    ISCreateGeneral(MPI_COMM_WORLD, 2*topo->n0, loc02, PETSC_COPY_VALUES, &isg);
    VecScatterCreate(bg, isg, bl, isl, &scat);
    VecScatterBegin(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(UQ->M, bg, UQb);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPGetPC(ksp,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOptionsPrefix(ksp,"init1_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, UQb, u);

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&UQb);
    KSPDestroy(&ksp);
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
    KSP ksp;
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

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M2->M, M2->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOptionsPrefix(ksp,"init2_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, WQb, h);

    delete WQ;
    KSPDestroy(&ksp);
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

/*
void SWEqn::init2(Vec h, ICfunc* func) {
    int ex, ey, ii, jj, mp1, mp12, nn, n2, err, rank;
    int *inds0, *inds2;
    PetscScalar* hArray;
    Vec hl;
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    Wii* Q = new Wii(edge->l->q, geom);
    JacM2* J = new JacM2(edge->l->q, geom);
    double** JW = Alloc2D(J->nDofsI, W->nDofsJ);
    double** JWt = Alloc2D(W->nDofsJ, J->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double** WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    double** WinvW = Alloc2D(W->nDofsJ, Q->nDofsJ);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;
    nn = topo->elOrd;
    n2 = nn*nn;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecGetArray(hl, &hArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            inds2 = topo->elInds2_l(ex, ey);

            Q->assemble(ex, ey);
            J->assemble(ex, ey);

            Mult_IP(J->nDofsI, W->nDofsJ, W->nDofsJ, J->A, W->A, JW);
            Tran_IP(J->nDofsI, W->nDofsJ, JW, JWt);
            Mult_IP(W->nDofsJ, Q->nDofsJ, Q->nDofsI, JWt, Q->A, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, JW, WtQW);

            //err = Inv(WtQW,WtQWinv,W->nDofsJ);
            //if( err > 0 ) {
            //    cout << rank << ": mat inv error (" << err << "): " << ex << "\t" << ey << endl;
            //} 
            Inverse(WtQW,WtQWinv,W->nDofsJ);

            Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsJ, WtQWinv, WtQ, WinvW);

            for(jj = 0; jj < n2; jj++) {
                hArray[inds2[jj]] = 0.0;
                for(ii = 0; ii < mp12; ii++) {
                    hArray[inds2[jj]] += WinvW[jj][ii]*func(geom->x[inds0[ii]]);
                }
            }
        }
    }
    VecRestoreArray(hl, &hArray);

    VecScatterBegin(topo->gtol_2, hl, h, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_2, hl, h, INSERT_VALUES, SCATTER_REVERSE);

    Free2D(J->nDofsI, JW);
    Free2D(W->nDofsJ, JWt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    Free2D(W->nDofsJ, WinvW);
    delete W;
    delete Q;
    delete J;
    VecDestroy(&hl);
}
*/
