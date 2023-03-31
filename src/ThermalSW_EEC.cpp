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
#include "ThermalSW_EEC.h"

#define RAD_EARTH 6371220.0
#define H_MEAN 1.0e+4
#define GRAVITY 9.80616
#define DO_THERMAL 1

using namespace std;

ThermalSW_EEC::ThermalSW_EEC(Topo* _topo, Geom* _geom) {
    PC pc;

    topo = _topo;
    geom = _geom;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    omega = 7.292e-5;
    step = 0;
    adv_S = adv_s = true;
    damp_div = false;

    quad = new GaussLobatto(geom->quad->n);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // 0 form lumped mass matrix (vector)
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

    M1h = new Uhmat(topo, geom, node, edge);
    M2h = new Whmat(topo, geom, edge);
    M0h = new Phmat(topo, geom, node);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    M2_ip = new W_IP_mat(topo, geom, edge);

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

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Si);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Sj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fS);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Phi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &si);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &sj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &wi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ds_on_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &S_on_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ds_on_h_l);

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &uil);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ujl);

    KSPCreate(MPI_COMM_WORLD, &ksp1h);
    KSPSetOperators(ksp1h, M1h->M, M1h->M);
    KSPSetTolerances(ksp1h, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1h, KSPGMRES);
    KSPGetPC(ksp1h, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1h, "Fonh_");
    KSPSetFromOptions(ksp1h);

    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPSetOptionsPrefix(ksp2, "ksp2_");
    KSPSetFromOptions(ksp2);

    KSPCreate(MPI_COMM_WORLD, &ksp2h);
    KSPSetOperators(ksp2h, M2h->M, M2h->M);
    KSPSetTolerances(ksp2h, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2h, KSPGMRES);
    KSPSetOptionsPrefix(ksp2h, "ksp2h_");
    KSPSetFromOptions(ksp2h);
}

void ThermalSW_EEC::grad(Vec phi, Vec* _u) {
    Vec dMphi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, _u);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMphi);

    MatMult(E12M2, phi, dMphi);
    KSPSolve(ksp, dMphi, *_u);

    VecDestroy(&dMphi);
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void ThermalSW_EEC::coriolis() {
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
        fArray[ii] = 2.0*omega*sin(geom->s[ii][1]);
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
    KSPSolve(ksp0, PtQfxg, fg);
    
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    VecScatterBegin(topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &M0fg);
    MatMult(M0->M, fg, M0fg);

    delete PtQ;
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

// derive vorticity (global vector) as \omega = curl u
void ThermalSW_EEC::curl(Vec u) {
    Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);

    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    // diagonal mass matrix as vector
    KSPSolve(ksp0, du, wi);

    VecDestroy(&du);
}

void ThermalSW_EEC::diagnose_q(Vec _u, Vec _h, Vec* qi) {
    Vec rhs;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, qi);

    MatMult(E01M1, _u, rhs);
    VecAXPY(rhs, 1.0, M0fg);
    M0h->assemble(_h);
    KSPSolve(ksp0h, rhs, *qi);

    VecDestroy(&rhs);
}

void ThermalSW_EEC::diagnose_s(Vec _h, Vec _S) {
    Vec rhs;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);

    M2h->assemble(_h);
    MatMult(M2->M, _S, rhs);
    KSPSolve(ksp2h, rhs, S_on_h);
    //M1h->assemble(S_on_h);
    //M1h->assemble(sj);
    
    VecDestroy(&rhs);
}

void ThermalSW_EEC::diagnose_ds(Vec _h, Vec _s) {
    Vec tmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp);

    //MatMult(E12M2, _s, tmp);
    MatMult(E12M2, S_on_h, tmp);
    M1h->assemble(_h);
    KSPSolve(ksp1h, tmp, ds_on_h);

    VecScatterBegin(topo->gtol_1, ds_on_h, ds_on_h_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ds_on_h, ds_on_h_l, INSERT_VALUES, SCATTER_FORWARD);
    K->assemble(ds_on_h_l);
    //M1h->assemble(S_on_h);
    M1h->assemble(sj);

    VecDestroy(&tmp);
}

void ThermalSW_EEC::diagnose_G(Vec _s) {
    Vec rhs;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &rhs);

    VecZeroEntries(G);
    //M1h->assemble(S_on_h);
    MatMult(M1h->M, F, rhs);
    KSPSolve(ksp, rhs, G);

    VecDestroy(&rhs);
}

ThermalSW_EEC::~ThermalSW_EEC() {
    KSPDestroy(&ksp);
    KSPDestroy(&ksp0);
    KSPDestroy(&ksp2);
    KSPDestroy(&ksp0h);
    KSPDestroy(&ksp1h);
    KSPDestroy(&ksp2h);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);
    VecDestroy(&M0fg);
    VecDestroy(&fl);
    VecDestroy(&ui);
    VecDestroy(&hi);
    VecDestroy(&Si);
    VecDestroy(&uj);
    VecDestroy(&hj);
    VecDestroy(&Sj);
    VecDestroy(&fu);
    VecDestroy(&fh);
    VecDestroy(&fS);
    VecDestroy(&fs);
    VecDestroy(&uil);
    VecDestroy(&ujl);
    VecDestroy(&F);
    VecDestroy(&Phi);
    VecDestroy(&G);
    VecDestroy(&si);
    VecDestroy(&sj);
    VecDestroy(&wi);
    VecDestroy(&ds_on_h);
    VecDestroy(&ds_on_h_l);
    VecDestroy(&S_on_h);

    delete M0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete R_up;
    delete M1h;
    delete M2h;
    delete M0h;
    delete K;
    delete M2_ip;

    delete edge;
    delete node;
    delete quad;
}

void ThermalSW_EEC::init0(Vec q, ICfunc* func) {
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
    KSPSolve(ksp0, PQb, q);

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void ThermalSW_EEC::init1(Vec u, ICfunc* func_x, ICfunc* func_y) {
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

void ThermalSW_EEC::init2(Vec h, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    PetscScalar *bArray;
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

    KSPSolve(ksp2, WQb, h);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void ThermalSW_EEC::err0(Vec ug, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms) {
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

void ThermalSW_EEC::err1(Vec ug, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_q;
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
            inds_q = geom->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, un);
                ua[0] = fu(geom->x[inds_q[ii]]);
                ua[1] = fv(geom->x[inds_q[ii]]);

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
                    dua[0] = fp(geom->x[inds_q[ii]]);

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

void ThermalSW_EEC::err2(Vec ug, ICfunc* fu, double* norms) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds_q;
    double det, wd, l_inf;
    double un[1], ua[1];
    double local_1[2], global_1[2], local_2[2], global_2[2], local_i[2], global_i[2]; // first entry is the error, the second is the norm
    PetscScalar *array_2;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local_1[0] = local_1[1] = 0.0;
    local_2[0] = local_2[1] = 0.0;
    local_i[0] = local_i[1] = 0.0;

    VecGetArray(ug, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds_q = geom->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
if(fabs(geom->s[inds_q[ii]][1]) > 0.45*M_PI) continue;
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, un);
                ua[0] = fu(geom->x[inds_q[ii]]);

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

    norms[0] = global_1[0]/global_1[1];
    norms[1] = sqrt(global_2[0]/global_2[1]);
    norms[2] = global_i[0]/global_i[1];
}

double ThermalSW_EEC::int2(Vec ug) {
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

double ThermalSW_EEC::intE(Vec ul, Vec hg, Vec Sg) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, hq, Sq, uq[2], local, global;
    PetscScalar *array_1, *array_2, *array_3;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecGetArray(ul, &array_1);
    VecGetArray(hg, &array_2);
    VecGetArray(Sg, &array_3);

    local = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, uq);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, &hq);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_3, &Sq);
#ifdef DO_THERMAL
                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*0.5*(Sq*hq + hq*(uq[0]*uq[0] + uq[1]*uq[1]));
#else
                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*0.5*(GRAVITY*hq*hq + hq*(uq[0]*uq[0] + uq[1]*uq[1]));
#endif
            }
        }
    }
    VecRestoreArray(ul, &array_1);
    VecRestoreArray(hg, &array_2);
    VecRestoreArray(Sg, &array_3);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global;
}

double ThermalSW_EEC::intK(Vec dqg, Vec dsg) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, uq1[2], uq2[2], local, global;
    PetscScalar *array_1, *array_2;
    Vec dql, dsl;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dql);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dsl);

    VecScatterBegin(topo->gtol_1, dqg, dql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dqg, dql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, dsg, dsl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dsg, dsl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(dql, &array_1);
    VecGetArray(dsl, &array_2);

    local = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, uq1);
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_2, uq2);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*0.5*(uq1[0]*uq2[0] + uq1[1]*uq2[1]);
            }
        }
    }
    VecRestoreArray(dql, &array_1);
    VecRestoreArray(dsl, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&dql);
    VecDestroy(&dsl);

    return global;
}

void ThermalSW_EEC::writeConservation(double time, double mass0, double vort0, double ener0, double enst0, double buoy0, double entr0) {
    double mass, vort, ener, enst, buoy, entr;
    char filename[50];
    ofstream file;
    Vec qi, v0, htmp, dq, utmp, htmp2;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &v0);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dq);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);

    curl(ui);

    diagnose_q(ui, hi, &qi);
    MatMult(M0h->M, qi, v0);
    VecDot(qi, v0, &enst);

    MatMult(M2->M, hi, htmp);
    mass = int2(hi);

    buoy = int2(Si);

    MatMult(M0->M, wi, v0);
    VecSum(v0, &vort);

    ener = intE(uil, hi, Si);

    M2h->assemble(hi);
    MatMult(M2->M, Si, htmp);
    KSPSolve(ksp2h, htmp, htmp2);
    MatMult(M2->M, htmp2, htmp);
    VecDot(htmp, Si, &entr);
    entr *= 0.5;

    if(!rank) {
        cout << "conservation of mass:      " << (mass - mass0)/mass0 << endl;
        cout << "conservation of vorticity: " << (vort - vort0) << endl;
        cout << "conservation of energy:    " << (ener - ener0)/ener0 << endl;
        cout << "conservation of enstrophy: " << (enst - enst0)/enst0 << endl;
        cout << "conservation of buoyancy:  " << (buoy - buoy0)/buoy0 << endl;
        cout << "conservation of entropy:   " << (entr - entr0)/entr0 << endl;

        sprintf(filename, "output/conservation.dat");
        file.open(filename, ios::out | ios::app);
        // write time in days
        file << scientific;
        file << time/60.0/60.0/24.0 << "\t" << (mass-mass0)/mass0 << "\t" << (vort-vort0) << "\t" 
                                            << (ener-ener0)/ener0 << "\t" << (enst-enst0)/enst0 << "\t"
					    << (buoy-buoy0)/buoy0 << "\t" << (entr-entr0)/entr0 << endl;
        file.close();
    }
    VecDestroy(&qi);
    VecDestroy(&v0);
    VecDestroy(&htmp);
    VecDestroy(&htmp2);
    VecDestroy(&dq);
    VecDestroy(&utmp);
}

void ThermalSW_EEC::solve_rk(double _dt, bool save) {
    Vec tmph, rhsh, tmpu, tmpu2;

    dt = _dt;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmph);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhsh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmpu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmpu2);

    VecScatterBegin(topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, ui, uil, INSERT_VALUES, SCATTER_FORWARD);
    VecCopy(uil, ujl);

    // solution vector
    VecCopy(ui, uj);
    VecCopy(hi, hj);
    VecCopy(Si, Sj);
    VecCopy(si, sj);

    // stage 1
    diagnose_F(uj, hj);
    diagnose_Phi(uj, hj, Sj, sj, ujl);
#ifdef DO_THERMAL
    if(adv_S) {
        diagnose_s(hj, Sj);
    }
    if(adv_s) {
        diagnose_ds(hj, sj);
    }
        diagnose_G(sj);
#endif

    rhs_u(uj, hj, Sj, sj, ujl, _dt);
    MatMult(M1->M, ui, tmpu);
    VecAXPY(tmpu, -_dt, fu);
    KSPSolve(ksp, tmpu, uj);
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    MatMult(EtoF->E21, F, hj);
    VecAYPX(hj, -_dt, hi);

#ifdef DO_THERMAL
    if(adv_S) {
        MatMult(EtoF->E21, G, Sj);
        VecAYPX(Sj, -_dt, Si);
    }
    if(adv_s) {
        MatMult(K->M, F, tmph); // inc. 0.5
        MatMult(M2->M, si, rhsh);
        VecAXPY(rhsh, -2.0*_dt, tmph);
        KSPSolve(ksp2, rhsh, sj);
    }
#endif

    // second stage
    diagnose_F(uj, hj);
    diagnose_Phi(uj, hj, Sj, sj, ujl);
#ifdef DO_THERMAL
    if(adv_S) {
        diagnose_s(hj, Sj);
    }
    if(adv_s) {
        diagnose_ds(hj, sj);
    }
        diagnose_G(sj);
#endif

    rhs_u(uj, hj, Sj, sj, ujl, _dt);
    VecZeroEntries(tmpu);
    VecAXPY(tmpu, 0.25, uj);
    VecAXPY(tmpu, 0.75, ui);
    MatMult(M1->M, tmpu, tmpu2);
    VecAXPY(tmpu2, -0.25*_dt, fu);
    KSPSolve(ksp, tmpu2, uj);
    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    MatMult(EtoF->E21, F, tmph);
    VecAYPX(tmph, -_dt, hj);
    VecAXPY(tmph, 3.0, hi);
    VecScale(tmph, 0.25);
    VecCopy(tmph, hj);

#ifdef DO_THERMAL
    if(adv_S) {
        MatMult(EtoF->E21, G, tmph);
        VecAYPX(tmph, -_dt, Sj);
        VecAXPY(tmph, 3.0, Si);
        VecScale(tmph, 0.25);
        VecCopy(tmph, Sj);
    }
    if(adv_s) {
        VecZeroEntries(tmph);
        VecAXPY(tmph, 0.75, si);
        VecAXPY(tmph, 0.25, sj);
        MatMult(M2->M, tmph, rhsh);
        MatMult(K->M, F, tmph); // inc. 0.5
        VecAXPY(rhsh, -0.5*_dt, tmph);
        KSPSolve(ksp2, rhsh, sj);
    }
#endif

    // third stage
    diagnose_F(uj, hj);
    diagnose_Phi(uj, hj, Sj, sj, ujl);
#ifdef DO_THERMAL
    if(adv_S) {
        diagnose_s(hj, Sj);
    }
    if(adv_s) {
        diagnose_ds(hj, sj);
    }
        diagnose_G(sj);
#endif

    rhs_u(uj, hj, Sj, sj, ujl, _dt);
    VecZeroEntries(tmpu);
    VecAXPY(tmpu, 1.0/3.0, ui);
    VecAXPY(tmpu, 2.0/3.0, uj);
    MatMult(M1->M, tmpu, tmpu2);
    VecAXPY(tmpu2, -2.0/3.0*_dt, fu);
    KSPSolve(ksp, tmpu2, uj);

    MatMult(EtoF->E21, F, tmph);
    VecAYPX(tmph, -_dt, hj);
    VecAYPX(tmph, 2.0, hi);
    VecScale(tmph, 1.0/3.0);
    VecCopy(tmph, hj);

#ifdef DO_THERMAL
    if(adv_S) {
        MatMult(EtoF->E21, G, tmph);
        VecAYPX(tmph, -_dt, Sj);
        VecAYPX(tmph, 2.0, Si);
        VecScale(tmph, 1.0/3.0);
        VecCopy(tmph, Sj);
    }
    if(adv_s) {
        VecZeroEntries(tmph);
        VecAXPY(tmph, 1.0/3.0, si);
        VecAXPY(tmph, 2.0/3.0, sj);
        MatMult(M2->M, tmph, rhsh);
        MatMult(K->M, F, tmph); // inc. 0.5
        VecAXPY(rhsh, -4.0/3.0*_dt, tmph);
        KSPSolve(ksp2, rhsh, sj);
    }
#endif

    VecCopy(uj, ui);
    VecCopy(hj, hi);
    VecCopy(Sj, Si);
    VecCopy(sj, si);
    VecCopy(ujl, uil);

    if(save) {
        char fieldname[20];

        step++;
        curl(ui);
	MatMult(EtoF->E21, F, tmph);

        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(ui, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hi, fieldname, step);
        sprintf(fieldname, "buoyancy");
        geom->write2(si, fieldname, step);
        sprintf(fieldname, "depth_buoyancy");
        geom->write2(Si, fieldname, step);
        sprintf(fieldname, "divergence");
        geom->write2(tmph, fieldname, step);
    }

    VecDestroy(&tmph);
    VecDestroy(&rhsh);
    VecDestroy(&tmpu);
    VecDestroy(&tmpu2);
}

void ThermalSW_EEC::diagnose_F(Vec _u, Vec _h) {
    Vec hu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecZeroEntries(F);
    VecZeroEntries(hu);

    M1h->assemble(_h);
    MatMult(M1h->M, _u, hu);
    KSPSolve(ksp, hu, F);

    VecDestroy(&hu);
}

void ThermalSW_EEC::diagnose_Phi(Vec _u, Vec _h, Vec _S, Vec _s, Vec _ul) {
    double fac = (adv_S && adv_s) ? 0.5 : 1.0;
    Vec b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecZeroEntries(Phi);

    // u^2 terms (0.5 factor incorportated into the matrix assembly)
    K->assemble(_ul);
    MatMult(K->M, _u, Phi);

    // S/2 terms
#ifdef DO_THERMAL
    if(adv_S) {
        MatMult(M2->M, _S, b);
        VecAXPY(Phi, 0.5*fac, b);
    }
    if(adv_s) {
        M2h->assemble(_s);
        MatMult(M2h->M, _h, b);
        VecAXPY(Phi, fac, b);
    }
#else
    MatMult(M2->M, _h, b);
    VecAXPY(Phi, GRAVITY, b);
#endif

    VecDestroy(&b);
}

void ThermalSW_EEC::rhs_u(Vec _u, Vec _h, Vec _S, Vec _s, Vec _ul, double _dt) {
    double fac = (adv_S && adv_s) ? 0.5 : 1.0;
    Vec tmp, qi, qil, dqil, dqg, dh, htmp, h2, tmpq, tmpq2;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &qil);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dqg);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dqil);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h2);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &tmpq);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &tmpq2);

    VecZeroEntries(fu);

    MatMult(EtoF->E12, Phi, tmp);
    VecAXPY(fu, 1.0, tmp);

    diagnose_q(_u, _h, &qi);
    VecScatterBegin(topo->gtol_0, qi, qil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qi, qil, INSERT_VALUES, SCATTER_FORWARD);
    MatMult(NtoE->E10, qi, dqg);
    VecScatterBegin(topo->gtol_1, dqg, dqil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dqg, dqil, INSERT_VALUES, SCATTER_FORWARD);

    R_up->assemble_supg(qil, _ul, dqil, 0.5, _dt, qil); // apvm
    MatMult(R_up->M, F, tmp);
    VecAXPY(fu, 1.0, tmp);

#ifdef DO_THERMAL
    if(adv_S) {
        grad(_h, &dh);
        //M1h->assemble(S_on_h);
        MatMult(M1h->M, dh, tmp);
        VecAXPY(fu, 0.5*fac, tmp);
        VecDestroy(&dh);
    }
    if(adv_s) {
        M2h->assemble(_h);
        MatMult(M2h->M, _h, htmp);
        KSPSolve(ksp2, htmp, h2);
        MatMultTranspose(K->M, h2, tmp); // inc. 0.5
        VecAXPY(fu, -fac, tmp);
    }
#endif

    if(damp_div) {
        MatMult(EtoF->E21, F, htmp);
        MatMult(M2_ip->M_QW, htmp, tmpq);
        MatMult(M2_ip->M_Q, tmpq, tmpq2);
        MatMult(M2_ip->M_WQ, tmpq2, htmp);
        MatMult(EtoF->E12, htmp, tmp);
        VecAXPY(fu, -5.0*9224893284.699825/H_MEAN, tmp);
    }

    VecDestroy(&tmp);
    VecDestroy(&qi);
    VecDestroy(&qil);
    VecDestroy(&dqg);
    VecDestroy(&dqil);
    VecDestroy(&htmp);
    VecDestroy(&h2);
    VecDestroy(&tmpq);
    VecDestroy(&tmpq2);
}
