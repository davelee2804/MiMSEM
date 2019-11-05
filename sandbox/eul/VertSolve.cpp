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
#include "L2Vecs.h"
#include "ElMats.h"
#include "VertOps.h"
#include "VertSolve.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
#define VERT_TOL 1.0e-8
#define HORIZ_TOL 1.0e-12
//#define RAYLEIGH 0.2
#define VISC 1

using namespace std;

VertSolve::VertSolve(Topo* _topo, Geom* _geom, double _dt) {
    int ii, elOrd2;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    step = 0;
    firstStep = true;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    vo = new VertOps(topo, geom);

    elOrd2 = topo->elOrd * topo->elOrd;

    gv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &gv[ii]);
    }
    zv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &zv[ii]);
    }
    initGZ();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &_Phi_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*elOrd2, &_theta_h);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_tmpA1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_tmpA2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &_tmpB1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &_tmpB2);

    _PCz = NULL;
    pc_A_rt = NULL;
    _V0_invV0_rt = NULL;

    viscosity();
}

void VertSolve::initGZ() {
    int ex, ey, ei, ii, kk, n2, mp12;
    int* inds0;
    int inds2k[99], inds0k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double* WtQflat = new double[W->nDofsJ*Q->nDofsJ];
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    Vec gz;
    Mat GRAD, BQ;
    PetscScalar* zArray;

    n2   = topo->elOrd*topo->elOrd;
    mp12 = (quad->n + 1)*(quad->n + 1);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*mp12, &gz);

    MatCreate(MPI_COMM_SELF, &BQ);
    MatSetType(BQ, MATSEQAIJ);
    MatSetSizes(BQ, (geom->nk+0)*n2, (geom->nk+1)*mp12, (geom->nk+0)*n2, (geom->nk+1)*mp12);
    MatSeqAIJSetPreallocation(BQ, 2*mp12, PETSC_NULL);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);
            Q->assemble(ex, ey);

            MatZeroEntries(BQ);
            for(kk = 0; kk < geom->nk; kk++) {
                for(ii = 0; ii < mp12; ii++) {
                    Q0[ii][ii]  = Q->A[ii][ii]*SCALE;
                    // for linear field we multiply by the vertical jacobian determinant when
                    // integrating, and do no other trasformations for the basis functions
                    Q0[ii][ii] *= 0.5;
                }
                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
                Flat2D_IP(W->nDofsJ, Q->nDofsJ, WtQ, WtQflat);

                for(ii = 0; ii < W->nDofsJ; ii++) {
                    inds2k[ii] = ii + kk*W->nDofsJ;
                }

                // assemble the first basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+0)*mp12;
                }
                MatSetValues(BQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQflat, ADD_VALUES);
                // assemble the second basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+1)*mp12;
                }
                MatSetValues(BQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQflat, ADD_VALUES);
            }
            MatAssemblyBegin(BQ, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(BQ, MAT_FINAL_ASSEMBLY);

            if(!ei) {
                MatMatMult(vo->V01, BQ, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GRAD);
            } else {
                MatMatMult(vo->V01, BQ, MAT_REUSE_MATRIX, PETSC_DEFAULT, &GRAD);
            }

            VecZeroEntries(gz);
            VecGetArray(gz, &zArray);
            for(kk = 0; kk < geom->nk+1; kk++) {
                for(ii = 0; ii < mp12; ii++) {
                    zArray[kk*mp12+ii] = GRAVITY*geom->levs[kk][inds0[ii]];
                }
            }
            VecRestoreArray(gz, &zArray);
            MatMult(GRAD, gz, gv[ei]);
            MatMult(BQ,   gz, zv[ei]);
        }
    }

    VecDestroy(&gz);
    MatDestroy(&GRAD);
    MatDestroy(&BQ);
    delete[] WtQflat;
    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    delete W;
    delete Q;
}

VertSolve::~VertSolve() {
    int ii;

    VecDestroy(&_Phi_z);
    VecDestroy(&_theta_h);
    VecDestroy(&_tmpA1);
    VecDestroy(&_tmpA2);
    VecDestroy(&_tmpB1);
    VecDestroy(&_tmpB2);

    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
        VecDestroy(&zv[ii]);
    }
    delete[] gv;
    delete[] zv;

    delete edge;
    delete node;
    delete quad;

    delete vo;

    if(_PCz) {
        MatDestroy(&_PCz);
        KSPDestroy(&ksp_exner);
    }
}

double VertSolve::MaxNorm(Vec dx, Vec x, double max_norm) {
    double norm_dx, norm_x, new_max_norm;

    VecNorm(dx, NORM_2, &norm_dx);
    VecNorm(x, NORM_2, &norm_x);
    new_max_norm = (norm_dx/norm_x > max_norm) ? norm_dx/norm_x : max_norm;
    return new_max_norm;
}

void VertSolve::solve_coupled(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i) {
    bool done = false;
    int ex, ey, elOrd2, itt = 0;
    int nDofsTotal = (4*geom->nk - 1)*vo->n2;
    double norm_x, max_norm_w, max_norm_exner, max_norm_rho, max_norm_rt;
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* F_z = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* G_z = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* dF_z = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* dG_z = new L2Vecs(geom->nk, topo, geom);
    Vec F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, F, dx;
    PC pc;
    Mat PC_coupled = NULL;
    KSP ksp_coupled = NULL;

    elOrd2 = topo->elOrd*topo->elOrd;
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &d_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_exner);
    VecCreateSeq(MPI_COMM_SELF, nDofsTotal, &F);
    VecCreateSeq(MPI_COMM_SELF, nDofsTotal, &dx);

    velz_i->UpdateLocal();
    velz_i->HorizToVert();
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rt_i->UpdateLocal();
    rt_i->HorizToVert();
    exner_i->UpdateLocal();
    exner_i->HorizToVert();

    velz_j->CopyFromVert(velz_i->vz);
    rho_j->CopyFromVert(rho_i->vz);
    rt_j->CopyFromVert(rt_i->vz);
    exner_j->CopyFromVert(exner_i->vz);

    // diagnose the potential temperature
    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();

    exner_h->CopyFromHoriz(exner_i->vh);
    exner_h->UpdateLocal();
    exner_h->HorizToVert();

    do {
        max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // implicit coupled solve
            assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], F_w, F_z->vz[ii], G_z->vz[ii]);

            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);

            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z->vz[ii], dF_z->vz[ii]);
            MatMult(vo->V10, G_z->vz[ii], dG_z->vz[ii]);

            MatMult(vo->VB, rho_j->vz[ii], F_rho);
            MatMult(vo->VB, rho_i->vz[ii], _tmpB1);
            VecAXPY(F_rho, -1.0, _tmpB1);
            MatMult(vo->VB, dF_z->vz[ii], _tmpB1);
            VecAXPY(F_rho, dt, _tmpB1);

            MatMult(vo->VB, rt_j->vz[ii], F_rt);
            MatMult(vo->VB, rt_i->vz[ii], _tmpB1);
            VecAXPY(F_rt, -1.0, _tmpB1);
            MatMult(vo->VB, dG_z->vz[ii], _tmpB1);
            VecAXPY(F_rt, dt, _tmpB1);

            repack_z(F, F_w, F_rho, F_rt, F_exner);
            VecScale(F, -1.0);

            assemble_operator(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_j->vz[ii], &PC_coupled);

            KSPCreate(MPI_COMM_SELF, &ksp_coupled);
            KSPSetOperators(ksp_coupled, PC_coupled, PC_coupled);
            KSPGetPC(ksp_coupled, &pc);
            PCSetType(pc, PCLU);
            KSPSetOptionsPrefix(ksp_coupled, "ksp_coupled_");
            KSPSetFromOptions(ksp_coupled);
            KSPSolve(ksp_coupled, F, dx);
            KSPDestroy(&ksp_coupled);

            unpack_z(dx, d_w, d_rho, d_rt, d_exner);
            VecAXPY(velz_j->vz[ii],  1.0, d_w);
            VecAXPY(rho_j->vz[ii],   1.0, d_rho);
            VecAXPY(rt_j->vz[ii],    1.0, d_rt);
            VecAXPY(exner_j->vz[ii], 1.0, d_exner);
      
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            max_norm_exner = MaxNorm(d_exner, exner_j->vz[ii], max_norm_exner);
            max_norm_w     = MaxNorm(d_w,     velz_j->vz[ii],  max_norm_w    );
            max_norm_rho   = MaxNorm(d_rho,   rho_j->vz[ii],   max_norm_rho  );
            max_norm_rt    = MaxNorm(d_rt,    rt_j->vz[ii],    max_norm_rt   );
        }

        diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        theta_h->VertToHoriz();
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(theta_h->vz[ii], 1.0, theta_i->vz[ii]);
            VecScale(theta_h->vz[ii], 0.5);
        }
        theta_h->VertToHoriz();

        MPI_Allreduce(&max_norm_exner, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_exner = norm_x;
        MPI_Allreduce(&max_norm_w,     &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w     = norm_x;
        MPI_Allreduce(&max_norm_rho,   &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho   = norm_x;
        MPI_Allreduce(&max_norm_rt,    &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt    = norm_x;

        itt++;

        if(max_norm_exner < 1.0e-6 && max_norm_w < 1.0e-8 && max_norm_rho < 1.0e-6 && max_norm_rt < 1.0e-6) done = true;
        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                 "\t|d_w|/|w|: "          << max_norm_w     <<
                                 "\t|d_rho|/|rho|: "      << max_norm_rho   <<
                                 "\t|d_rt|/|rt|: "        << max_norm_rt    << endl;
    } while(!done);

    velz_i->CopyFromVert(velz_j->vz);
    rho_i->CopyFromVert(rho_j->vz);
    rt_i->CopyFromVert(rt_j->vz);
    exner_i->CopyFromVert(exner_h->vz);

    velz_i->VertToHoriz();
    velz_i->UpdateGlobal();
    rho_i->VertToHoriz();
    rho_i->UpdateGlobal();
    rt_i->VertToHoriz();
    rt_i->UpdateGlobal();
    exner_i->VertToHoriz();
    exner_i->UpdateGlobal();

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete theta_i;
    delete theta_h;
    delete F_z;
    delete G_z;
    delete dF_z;
    delete dG_z;
    VecDestroy(&F_w);
    VecDestroy(&F_rho);
    VecDestroy(&F_rt);
    VecDestroy(&F_exner);
    VecDestroy(&d_w);
    VecDestroy(&d_rho);
    VecDestroy(&d_rt);
    VecDestroy(&d_exner);
    VecDestroy(&F);
    VecDestroy(&dx);
    MatDestroy(&PC_coupled);
    KSPDestroy(&ksp_coupled);
}

void VertSolve::solve_schur(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i) {
    bool done = false;
    int ex, ey, elOrd2, itt = 0;
    double norm_x, max_norm_w, max_norm_exner, max_norm_rho, max_norm_rt;
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    Vec F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, F_z, G_z, dF_z, dG_z;
    PC pc;

    elOrd2 = topo->elOrd*topo->elOrd;
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &d_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);

    velz_i->UpdateLocal();
    velz_i->HorizToVert();
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rt_i->UpdateLocal();
    rt_i->HorizToVert();
    exner_i->UpdateLocal();
    exner_i->HorizToVert();

    velz_j->CopyFromVert(velz_i->vz);
    rho_j->CopyFromVert(rho_i->vz);
    rt_j->CopyFromVert(rt_i->vz);
    exner_j->CopyFromVert(exner_i->vz);

    // diagnose the potential temperature
    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();

    exner_h->CopyFromHoriz(exner_i->vh);
    exner_h->UpdateLocal();
    exner_h->HorizToVert();

    do {
        max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // assemble the residual vectors
            assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], F_w, F_z, G_z);

            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);

            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii]);
            MatMult(vo->VB, dF_z, F_rho);
            MatMult(vo->VB, dG_z, F_rt);

            // solve schur complement 
            //assemble_operator_schur(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_j->vz[ii], 
            //                        F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner);
            //assemble_operator_schur(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], 
            //                        F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner);
            assemble_and_update(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], F_w, F_rho, F_rt, F_exner, true, true);
            MatScale(_PCz, -1.0);
            if(!itt) {
                KSPCreate(MPI_COMM_SELF, &ksp_exner);
                KSPSetOperators(ksp_exner, _PCz, _PCz);
                KSPGetPC(ksp_exner, &pc);
                PCSetType(pc, PCLU);
                KSPSetOptionsPrefix(ksp_exner, "ksp_exner_");
                KSPSetFromOptions(ksp_exner);
            }
            KSPSolve(ksp_exner, F_rt, d_exner);
            set_deltas(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], 
                       F_w, F_rho, F_exner, d_w, d_rho, d_rt, d_exner, false, true);

            VecAXPY(velz_j->vz[ii],  1.0, d_w);
            VecAXPY(rho_j->vz[ii],   1.0, d_rho);
            VecAXPY(rt_j->vz[ii],    1.0, d_rt);
            VecAXPY(exner_j->vz[ii], 1.0, d_exner);

            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            max_norm_exner = MaxNorm(d_exner, exner_j->vz[ii], max_norm_exner);
            max_norm_w     = MaxNorm(d_w,     velz_j->vz[ii],  max_norm_w    );
            max_norm_rho   = MaxNorm(d_rho,   rho_j->vz[ii],   max_norm_rho  );
            max_norm_rt    = MaxNorm(d_rt,    rt_j->vz[ii],    max_norm_rt   );
        }

        diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        theta_h->VertToHoriz();
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(theta_h->vz[ii], 1.0, theta_i->vz[ii]);
            VecScale(theta_h->vz[ii], 0.5);
        }
        theta_h->VertToHoriz();

        MPI_Allreduce(&max_norm_exner, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_exner = norm_x;
        MPI_Allreduce(&max_norm_w,     &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w     = norm_x;
        MPI_Allreduce(&max_norm_rho,   &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho   = norm_x;
        MPI_Allreduce(&max_norm_rt,    &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt    = norm_x;

        itt++;

        if(max_norm_exner < 1.0e-8 && max_norm_w < 1.0e-8 && max_norm_rho < 1.0e-8 && max_norm_rt < 1.0e-8) done = true;
        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                 "\t|d_w|/|w|: "          << max_norm_w     <<
                                 "\t|d_rho|/|rho|: "      << max_norm_rho   <<
                                 "\t|d_rt|/|rt|: "        << max_norm_rt    << endl;
    } while(!done);

    velz_i->CopyFromVert(velz_j->vz);
    rho_i->CopyFromVert(rho_j->vz);
    rt_i->CopyFromVert(rt_j->vz);
    exner_i->CopyFromVert(exner_h->vz);

    velz_i->VertToHoriz();
    velz_i->UpdateGlobal();
    rho_i->VertToHoriz();
    rho_i->UpdateGlobal();
    rt_i->VertToHoriz();
    rt_i->UpdateGlobal();
    exner_i->VertToHoriz();
    exner_i->UpdateGlobal();

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete theta_i;
    delete theta_h;
    VecDestroy(&F_w);
    VecDestroy(&F_rho);
    VecDestroy(&F_rt);
    VecDestroy(&F_exner);
    VecDestroy(&d_w);
    VecDestroy(&d_rho);
    VecDestroy(&d_rt);
    VecDestroy(&d_exner);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
}

void VertSolve::diagnose_F_z(int ex, int ey, Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec _F) {
    MatReuse reuse = (!_V0_invV0_rt) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    VecZeroEntries(_F);

    vo->AssembleLinearInv(ex, ey, vo->VA_inv);

    vo->AssembleLinearWithRT(ex, ey, rho1, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &_V0_invV0_rt);

    MatMult(_V0_invV0_rt, velz1, _tmpA1);
    VecAXPY(_F, 1.0/3.0, _tmpA1);

    MatMult(_V0_invV0_rt, velz2, _tmpA1);
    VecAXPY(_F, 1.0/6.0, _tmpA1);

    vo->AssembleLinearWithRT(ex, ey, rho2, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &_V0_invV0_rt);

    MatMult(_V0_invV0_rt, velz1, _tmpA1);
    VecAXPY(_F, 1.0/6.0, _tmpA1);

    MatMult(_V0_invV0_rt, velz2, _tmpA1);
    VecAXPY(_F, 1.0/3.0, _tmpA1);
}

void VertSolve::diagnose_Phi_z(int ex, int ey, Vec velz1, Vec velz2, Vec Phi) {
    int ei = ey*topo->nElsX + ex;
    double alpha = 0.3;

    VecZeroEntries(Phi);
    VecZeroEntries(_tmpB2);

    // kinetic energy term
    MatZeroEntries(vo->VBA);
    vo->AssembleConLinWithW(ex, ey, velz1, vo->VBA);

    MatMult(vo->VBA, velz1, _tmpB1);
    //VecAXPY(Phi, 1.0/6.0, _tmpB1);
    VecAXPY(_tmpB2, 1.0/6.0, _tmpB1);
    
    MatMult(vo->VBA, velz2, _tmpB1);
    //VecAXPY(Phi, 1.0/6.0, _tmpB1);
    VecAXPY(_tmpB2, 1.0/6.0, _tmpB1);

    MatZeroEntries(vo->VBA);
    vo->AssembleConLinWithW(ex, ey, velz2, vo->VBA);

    MatMult(vo->VBA, velz2, _tmpB1);
    //VecAXPY(Phi, 1.0/6.0, _tmpB1);
    VecAXPY(_tmpB2, 1.0/6.0, _tmpB1);

    VecAXPY(Phi, alpha, _tmpB2);

    // potential energy term
    VecAXPY(Phi, 1.0, zv[ei]);

    // kinetic energy at vertices
    VecZeroEntries(_tmpA1);
    vo->AssembleLinearWithW(ex, ey, velz1, vo->VA);
    MatMult(vo->VA, velz1, _tmpA2);
    VecAXPY(_tmpA1, 1.0/6.0, _tmpA2);
    MatMult(vo->VA, velz2, _tmpA2);
    VecAXPY(_tmpA1, 1.0/6.0, _tmpA2);
    vo->AssembleLinearWithW(ex, ey, velz2, vo->VA);
    MatMult(vo->VA, velz2, _tmpA2);
    VecAXPY(_tmpA1, 1.0/6.0, _tmpA2);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2);
    vo->AssembleConLin(ex, ey, vo->VBA);
    MatMult(vo->VBA, _tmpA2, _tmpB1);
    VecAXPY(Phi, 1.0-alpha, _tmpB1);
}

/* All vectors, rho, rt and theta are VERTICAL vectors */
void VertSolve::diagTheta2(Vec* rho, Vec* rt, Vec* theta) {
    int ex, ey, n2, ei;
    Vec frt;
    PC pc;
    KSP kspColA2;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &frt);

    KSPCreate(MPI_COMM_SELF, &kspColA2);
    KSPSetOperators(kspColA2, vo->VA2, vo->VA2);
    KSPGetPC(kspColA2, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColA2, "kspColA2_");
    KSPSetFromOptions(kspColA2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            vo->AssembleLinCon2(ex, ey, vo->VAB2);
            MatMult(vo->VAB2, rt[ei], frt);

            vo->AssembleLinearWithRho2(ex, ey, rho[ei], vo->VA2);
            KSPSolve(kspColA2, frt, theta[ei]);
        }
    }
    VecDestroy(&frt);
    KSPDestroy(&kspColA2);
}

void VertSolve::assemble_residual_z(int ex, int ey, Vec theta, Vec Pi, 
                                Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fw, Vec _F, Vec _G) 
{
    // diagnose the hamiltonian derivatives
    diagnose_F_z(ex, ey, velz1, velz2, rho1, rho2, _F);
    diagnose_Phi_z(ex, ey, velz1, velz2, _Phi_z);

    // assemble the momentum equation residual
    vo->AssembleLinear(ex, ey, vo->VA);
    MatMult(vo->VA, velz2, fw);

    MatMult(vo->VA, velz1, _tmpA1);
    VecAXPY(fw, -1.0, _tmpA1);

    MatMult(vo->V01, _Phi_z, _tmpA1);
    VecAXPY(fw, +dt, _tmpA1); // bernoulli function term

    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, Pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMult(vo->VA, _tmpA2, _tmpA1);
    VecAXPY(fw, +dt, _tmpA1); // pressure gradient term

    // update the temperature equation flux
    MatMult(vo->VA, _F, _tmpA1); // includes theta
    MatMult(vo->VA_inv, _tmpA1, _G);

    // add the rayleigh friction
#ifdef RAYLEIGH
    vo->AssembleRayleigh(ex, ey, vo->VA);
    MatMult(vo->VA, velz2, _tmpA1);
    VecAXPY(fw, 0.5*dt*RAYLEIGH, _tmpA1);
    MatMult(vo->VA, velz1, _tmpA1);
    VecAXPY(fw, 0.5*dt*RAYLEIGH, _tmpA1);
#endif

    // add the laplacian viscosity
#ifdef VISC
    VecZeroEntries(_tmpA1);
    VecAXPY(_tmpA1, 0.5*dt*visc, velz1);
    VecAXPY(_tmpA1, 0.5*dt*visc, velz2);
    MatMult(vo->V10, _tmpA1, _tmpB1);
    MatMult(vo->VB, _tmpB1, _tmpB2);
    MatMult(vo->V01, _tmpB2, _tmpA1);
    VecAXPY(fw, -1.0, _tmpA1);
#endif
}

void VertSolve::repack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner) {
    int ii, shift;
    PetscScalar *xArray, *uArray, *rhoArray, *rtArray, *eArray;

    VecGetArray(x,     &xArray  );
    VecGetArray(u,     &uArray  );
    VecGetArray(rho,   &rhoArray);
    VecGetArray(rt,    &rtArray );
    VecGetArray(exner, &eArray  );

    for(ii = 0; ii < vo->n2*(geom->nk-1); ii++) {
        xArray[ii] = uArray[ii];
    }
    shift = vo->n2*(geom->nk-1);
    for(ii = 0; ii < vo->n2*geom->nk; ii++) {
        xArray[shift+ii] = rhoArray[ii];
    }
    shift += vo->n2*geom->nk;
    for(ii = 0; ii < vo->n2*geom->nk; ii++) {
        xArray[shift+ii] = rtArray[ii];
    }
    shift += vo->n2*geom->nk;
    for(ii = 0; ii < vo->n2*geom->nk; ii++) {
        xArray[shift+ii] = eArray[ii];
    }

    VecRestoreArray(x,     &xArray  );
    VecRestoreArray(u,     &uArray  );
    VecRestoreArray(rho,   &rhoArray);
    VecRestoreArray(rt,    &rtArray );
    VecRestoreArray(exner, &eArray  );
}

void VertSolve::unpack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner) {
    int ii, shift;
    PetscScalar *xArray, *uArray, *rhoArray, *rtArray, *eArray;

    VecGetArray(x,     &xArray  );
    VecGetArray(u,     &uArray  );
    VecGetArray(rho,   &rhoArray);
    VecGetArray(rt,    &rtArray );
    VecGetArray(exner, &eArray  );

    for(ii = 0; ii < vo->n2*(geom->nk-1); ii++) {
        uArray[ii] = xArray[ii];
    }
    shift = vo->n2*(geom->nk-1);
    for(ii = 0; ii < vo->n2*geom->nk; ii++) {
        rhoArray[ii] = xArray[shift+ii];
    }
    shift += vo->n2*geom->nk;
    for(ii = 0; ii < vo->n2*geom->nk; ii++) {
        rtArray[ii] = xArray[shift+ii];
    }
    shift += vo->n2*geom->nk;
    for(ii = 0; ii < vo->n2*geom->nk; ii++) {
        eArray[ii] = xArray[shift+ii];
    }

    VecRestoreArray(x,     &xArray  );
    VecRestoreArray(u,     &uArray  );
    VecRestoreArray(rho,   &rhoArray);
    VecRestoreArray(rt,    &rtArray );
    VecRestoreArray(exner, &eArray  );
}

void VertSolve::assemble_operator(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, Mat* _PC) {
    int n2 = topo->elOrd*topo->elOrd;
    int nDofsW = (geom->nk-1)*n2;
    int nDofsRho = geom->nk*n2;
    int nDofsTotal = nDofsW + 3*nDofsRho;
    int mm, mi, mf, ri, ci;
    int nCols;
    const int *cols;
    const double* vals;
    int cols2[999];
    MatReuse reuse = (!*_PC) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    bool firstPass = false;

    if(!*_PC) {
        MatCreateSeqAIJ(MPI_COMM_SELF, nDofsTotal, nDofsTotal, 12*n2, NULL, _PC);
        firstPass = true;
    }
    MatZeroEntries(*_PC);

    // [u,u] block
    vo->AssembleLinear(ex, ey, vo->VA);
    MatGetOwnershipRange(vo->VA, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(vo->VA, mm, &nCols, &cols, &vals);
        ri = mm;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci];
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(vo->VA, mm, &nCols, &cols, &vals);
    }

    // [u,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &pc_GRAD);
    MatScale(pc_GRAD, 0.5*dt);
    MatGetOwnershipRange(pc_GRAD, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_GRAD, mm, &nCols, &cols, &vals);
        ri = mm;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + 2*nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_GRAD, mm, &nCols, &cols, &vals);
    }

    // [u,rho] block
/*
    vo->AssembleLinearWithRT(ex, ey, exner, vo->VA, true);
    vo->AssembleLinearWithRhoInv(ex, ey, rho, vo->VA_inv);
    MatMatMult(vo->VA, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(pc_V0_invV0_rt, pc_GRAD, reuse, PETSC_DEFAULT, &pc_A_u);
    MatScale(pc_A_u, RD/CV);
    MatGetOwnershipRange(pc_A_u, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_A_u, mm, &nCols, &cols, &vals);
        ri = mm;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_A_u, mm, &nCols, &cols, &vals);
    }
*/

    // [rho,u] block
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_V1DV0_invV0_rt);
    MatScale(pc_V1DV0_invV0_rt, 0.5*dt);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V1DV0_invV0_rt, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci];
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_V1DV0_invV0_rt, mm, &nCols, &cols, &vals);
    }

    // [rho,rho] block
    vo->AssembleLinConWithTheta(ex, ey, vo->VAB, velz);
    MatMatMult(vo->VA_inv, vo->VAB, reuse, PETSC_DEFAULT, &pc_V0_invV01);
    MatMatMult(vo->V10, pc_V0_invV01, reuse, PETSC_DEFAULT, &pc_DV0_invV01);
    MatMatMult(vo->VB, pc_DV0_invV01, reuse, PETSC_DEFAULT, &pc_V1DV0_invV01);
    MatAYPX(pc_V1DV0_invV01, 0.5*dt, vo->VB, DIFFERENT_NONZERO_PATTERN);
    MatGetOwnershipRange(pc_V1DV0_invV01, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V1DV0_invV01, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_V1DV0_invV01, mm, &nCols, &cols, &vals);
    }

    // [rt,u] block
    vo->AssembleLinearWithRT(ex, ey, rt, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V1DV0_invV0_rt);
    MatScale(pc_V1DV0_invV0_rt, 0.5*dt);
    MatGetOwnershipRange(pc_V1DV0_invV0_rt, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V1DV0_invV0_rt, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci];
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_V1DV0_invV0_rt, mm, &nCols, &cols, &vals);
    }

    // [rt,rt] block
    MatGetOwnershipRange(pc_V1DV0_invV01, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V1DV0_invV01, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_V1DV0_invV01, mm, &nCols, &cols, &vals);
    }

    // [exner,rt] block
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_VBVB_rt_invVB_pi);
    MatScale(pc_VBVB_rt_invVB_pi, -RD/CV);
    MatGetOwnershipRange(pc_VBVB_rt_invVB_pi, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_VBVB_rt_invVB_pi, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + 2*nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_VBVB_rt_invVB_pi, mm, &nCols, &cols, &vals);
    }

    // [exner,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VBVB_rt_invVB_pi);
    MatGetOwnershipRange(pc_VBVB_rt_invVB_pi, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_VBVB_rt_invVB_pi, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + 2*nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + 2*nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_VBVB_rt_invVB_pi, mm, &nCols, &cols, &vals);
    }

    MatAssemblyBegin(*_PC, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  *_PC, MAT_FINAL_ASSEMBLY);
}

void VertSolve::assemble_operator_schur(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, 
                                    Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner) {
    int n2 = topo->elOrd*topo->elOrd;
    bool build_ksp = (!_PCz) ? true : false;
    MatReuse reuse = (!_PCz) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    PC pc;

    // [u,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &pc_G);
    MatScale(pc_G, 0.5*dt);

    // [u,rho] block
    vo->AssembleLinearWithRhoInv(ex, ey, rho, vo->VA_inv);
    MatMatMult(vo->VA, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(pc_V0_invV0_rt, vo->V01, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &pc_A_u);
    MatScale(pc_A_u, 0.5*dt*RD/CV);

/*
    vo->AssembleLinearWithRT(ex, ey, exner, vo->VA, true);
    MatMatMult(pc_V0_invV0_rt, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_thetaV0_invV0_exner);
    MatMatMult(pc_V0_thetaV0_invV0_exner, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_V0_thetaV0_invV0_exnerV0_inv);
    MatMatMult(pc_V0_thetaV0_invV0_exnerV0_inv, vo->V01, reuse, PETSC_DEFAULT, &pc_V0_thetaV0_invV0_exnerV0_invDT);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(pc_V0_thetaV0_invV0_exnerV0_invDT, vo->VB, reuse, PETSC_DEFAULT, &pc_A_u_2);
    MatAXPY(pc_A_u, -0.5*dt*RD/CV, pc_A_u_2, DIFFERENT_NONZERO_PATTERN);
*/

    // [rho,u] block
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_D_rho);
    MatScale(pc_D_rho, 0.5*dt);

    // [rt,u] block
    vo->AssembleLinearWithRT(ex, ey, rt, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_D_rt);
    MatScale(pc_D_rt, 0.5*dt);

    // [exner,rt] block
    if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_N_rt_inv);
    vo->Assemble_EOS_BlockInv(ex, ey, rt, NULL, pc_N_rt_inv);
    MatScale(pc_N_rt_inv, -1.0*CV/RD);

    // [exner,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_N_exner);

    // 1. density corrections
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(pc_A_u, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_A_u_VB_inv);
    MatMatMult(pc_A_u_VB_inv, pc_D_rho, reuse, PETSC_DEFAULT, &pc_M_u);

    vo->AssembleLinear(ex, ey, vo->VA);
    MatAYPX(pc_M_u, -1.0, vo->VA, DIFFERENT_NONZERO_PATTERN);

    MatMult(pc_A_u_VB_inv, F_rho, _tmpA1);
    VecAXPY(F_w, -1.0, _tmpA1);
    
    // 2. density weighted potential temperature correction
    MatMatMult(vo->VB, pc_N_rt_inv, reuse, PETSC_DEFAULT, &pc_VB_N_rt_inv);
    MatMatMult(pc_VB_N_rt_inv, pc_N_exner, reuse, PETSC_DEFAULT, &pc_N_exner_2);
    MatMult(pc_VB_N_rt_inv, F_exner, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);

    // 3. schur complement solve for exner pressure
    if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk-1)*n2, (geom->nk-1)*n2, 1, NULL, &pc_M_u_inv);
#ifdef VISC
    MatMatMult(pc_DTV1, vo->V10, reuse, PETSC_DEFAULT, &pc_VISC);
    MatAYPX(pc_VISC, -0.5*dt*visc, pc_M_u, DIFFERENT_NONZERO_PATTERN);
    MatGetDiagonal(pc_VISC, _tmpA1);
#else
    MatGetDiagonal(pc_M_u, _tmpA1);
#endif
    VecSet(_tmpA2, 1.0);
    VecPointwiseDivide(_tmpA2, _tmpA2, _tmpA1);
    MatZeroEntries(pc_M_u_inv);
    MatDiagonalSet(pc_M_u_inv, _tmpA2, INSERT_VALUES);

    MatMatMult(pc_D_rt, pc_M_u_inv, reuse, PETSC_DEFAULT, &pc_D_rt_M_u_inv);
    MatMatMult(pc_D_rt_M_u_inv, pc_G, reuse, PETSC_DEFAULT, &_PCz);
    MatAXPY(_PCz, 1.0, pc_N_exner_2, DIFFERENT_NONZERO_PATTERN);
    MatScale(_PCz, -1.0);

    MatMult(pc_D_rt_M_u_inv, F_w, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);
    VecScale(F_rt, -1.0);

    if(build_ksp) {
        KSPCreate(MPI_COMM_SELF, &ksp_exner);
        KSPSetOperators(ksp_exner, _PCz, _PCz);
        KSPGetPC(ksp_exner, &pc);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(ksp_exner, "ksp_exner_");
        KSPSetFromOptions(ksp_exner);
    }
    KSPSolve(ksp_exner, F_rt, dexner);

    // 4. back substitution
    // -- velocity
    MatMult(pc_G, dexner, _tmpA1);
    VecAXPY(_tmpA1, 1.0, F_w);
    MatMult(pc_M_u_inv, _tmpA1, dw);
/*
    MatAssemblyBegin(pc_M_u, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  pc_M_u, MAT_FINAL_ASSEMBLY);
    if(build_ksp) {
        KSPCreate(MPI_COMM_SELF, &ksp_w);
        KSPSetOperators(ksp_w, pc_M_u, pc_M_u);
        KSPGetPC(ksp_w, &pc);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(ksp_w, "ksp_w_");
        KSPSetFromOptions(ksp_w);
    }
    KSPSolve(ksp_w, _tmpA1, dw);
*/
    VecScale(dw, -1.0);

    // -- density weighted potential temperature
    MatMult(pc_N_exner, dexner, _tmpB1);
    VecAXPY(_tmpB1, 1.0, F_exner);
    MatMult(pc_N_rt_inv, _tmpB1, drt);
    VecScale(drt, -1.0);

    // -- density
    MatMult(pc_D_rho, dw, _tmpB1);
    VecAXPY(_tmpB1, 1.0, F_rho);
    MatMult(vo->VB_inv, _tmpB1, drho);
    VecScale(drho, -1.0);
}

void VertSolve::assemble_and_update(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner, 
    bool eos_update, bool eos_update_mat) {
    int n2 = topo->elOrd*topo->elOrd;
    bool build_ksp = (!_PCz) ? true : false;
    MatReuse reuse = (!_PCz) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    // [u,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &pc_G);
    MatScale(pc_G, 0.5*dt);

    // [u,rho] block
    vo->AssembleLinearWithRhoInv(ex, ey, rho, vo->VA_inv);
    MatMatMult(vo->VA, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(pc_V0_invV0_rt, vo->V01, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &pc_A_u);
    MatScale(pc_A_u, 0.5*dt*RD/CV);

    // [rho,u] block
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_D_rho);
    MatScale(pc_D_rho, 0.5*dt);

    // [rt,u] block
    vo->AssembleLinearWithRT(ex, ey, rt, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_D_rt);
    MatScale(pc_D_rt, 0.5*dt);

    // [exner,rt] block
    if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_N_rt_inv);
    vo->Assemble_EOS_BlockInv(ex, ey, rt, NULL, pc_N_rt_inv);
    MatScale(pc_N_rt_inv, -1.0*CV/RD);

    // [exner,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_N_exner);

    // 1. density corrections
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(pc_A_u, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_A_u_VB_inv);
    MatMatMult(pc_A_u_VB_inv, pc_D_rho, reuse, PETSC_DEFAULT, &pc_M_u);

    vo->AssembleLinear(ex, ey, vo->VA);
    MatAYPX(pc_M_u, -1.0, vo->VA, DIFFERENT_NONZERO_PATTERN);
#ifdef RAYLEIGH
    vo->AssembleRayleigh(ex, ey, vo->VA);
    MatAXPY(pc_M_u, 0.5*dt*RAYLEIGH, vo->VA, DIFFERENT_NONZERO_PATTERN);
#endif

    MatMult(pc_A_u_VB_inv, F_rho, _tmpA1);
    VecAXPY(F_w, -1.0, _tmpA1);
    
    // 2. density weighted potential temperature correction
    if(eos_update_mat) {
        MatMatMult(vo->VB, pc_N_rt_inv, reuse, PETSC_DEFAULT, &pc_VB_N_rt_inv);
        MatMatMult(pc_VB_N_rt_inv, pc_N_exner, reuse, PETSC_DEFAULT, &pc_N_exner_2);
        if(eos_update) {
            MatMult(pc_VB_N_rt_inv, F_exner, _tmpB1);
            VecAXPY(F_rt, -1.0, _tmpB1);
        }
    }

    // 3. schur complement solve for exner pressure
    if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk-1)*n2, (geom->nk-1)*n2, 1, NULL, &pc_M_u_inv);
#ifdef VISC
    MatMatMult(pc_DTV1, vo->V10, reuse, PETSC_DEFAULT, &pc_VISC);
    MatAYPX(pc_VISC, -0.5*dt*visc, pc_M_u, DIFFERENT_NONZERO_PATTERN);
    MatGetDiagonal(pc_VISC, _tmpA1);
#else
    MatGetDiagonal(pc_M_u, _tmpA1);
#endif
    VecSet(_tmpA2, 1.0);
    VecPointwiseDivide(_tmpA2, _tmpA2, _tmpA1);
    MatZeroEntries(pc_M_u_inv);
    MatDiagonalSet(pc_M_u_inv, _tmpA2, INSERT_VALUES);

    MatMatMult(pc_D_rt, pc_M_u_inv, reuse, PETSC_DEFAULT, &pc_D_rt_M_u_inv);
    MatMatMult(pc_D_rt_M_u_inv, pc_G, reuse, PETSC_DEFAULT, &_PCz);
    MatMatMult(pc_D_rt_M_u_inv, pc_G, reuse, PETSC_DEFAULT, &pc_LAP); // TODO: optimize
    if(eos_update_mat) {
        MatAXPY(_PCz, 1.0, pc_N_exner_2, DIFFERENT_NONZERO_PATTERN);
    }
    MatScale(_PCz, -1.0);

    MatMult(pc_D_rt_M_u_inv, F_w, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);

    if(build_ksp) {
        PC pc;
        KSPCreate(MPI_COMM_SELF, &ksp_exner);
        KSPSetOperators(ksp_exner, _PCz, _PCz);
        KSPGetPC(ksp_exner, &pc);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(ksp_exner, "ksp_exner_z_");
        KSPSetFromOptions(ksp_exner);
    }
}

void VertSolve::set_deltas(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, 
                           Vec F_w, Vec F_rho, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner, 
                           bool add_delta, bool neg_scale) {
    // [u,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_G);
    MatScale(pc_G, 0.5*dt);

    // [u,rho] block
    vo->AssembleLinearWithRhoInv(ex, ey, rho, vo->VA_inv);
    MatMatMult(vo->VA, vo->VA_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(pc_V0_invV0_rt, vo->V01, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_A_u);
    MatScale(pc_A_u, 0.5*dt*RD/CV);

    // [rho,u] block
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_D_rho);
    MatScale(pc_D_rho, 0.5*dt);

    // [rt,u] block
    vo->AssembleLinearWithRT(ex, ey, rt, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_D_rt);
    MatScale(pc_D_rt, 0.5*dt);

    // [exner,rt] block
    vo->Assemble_EOS_BlockInv(ex, ey, rt, NULL, pc_N_rt_inv);
    MatScale(pc_N_rt_inv, -1.0*CV/RD);

    // [exner,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_N_exner);

    // 1. density corrections
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(pc_A_u, vo->VB_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_A_u_VB_inv);
    MatMatMult(pc_A_u_VB_inv, pc_D_rho, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_M_u);

    vo->AssembleLinear(ex, ey, vo->VA);
    MatAYPX(pc_M_u, -1.0, vo->VA, DIFFERENT_NONZERO_PATTERN);
#ifdef RAYLEIGH
    vo->AssembleRayleigh(ex, ey, vo->VA);
    MatAXPY(pc_M_u, 0.5*dt*RAYLEIGH, vo->VA, DIFFERENT_NONZERO_PATTERN);
#endif

    // 3. schur complement solve for exner pressure
#ifdef VISC
    MatMatMult(pc_DTV1, vo->V10, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VISC);
    MatAYPX(pc_VISC, -0.5*dt*visc, pc_M_u, DIFFERENT_NONZERO_PATTERN);
    MatGetDiagonal(pc_VISC, _tmpA1);
#else
    MatGetDiagonal(pc_M_u, _tmpA1);
#endif
    VecSet(_tmpA2, 1.0);
    VecPointwiseDivide(_tmpA2, _tmpA2, _tmpA1);
    MatZeroEntries(pc_M_u_inv);
    MatDiagonalSet(pc_M_u_inv, _tmpA2, INSERT_VALUES);

    // back substitution
    // -- velocity
    MatMult(pc_G, dexner, _tmpA1);
    VecAXPY(_tmpA1, 1.0, F_w);
    MatMult(pc_M_u_inv, _tmpA1, dw);
    VecScale(dw, -1.0);

    // -- density weighted potential temperature
    MatMult(pc_N_exner, dexner, _tmpB1);
    VecAXPY(_tmpB1, 1.0, F_exner);
    MatMult(pc_N_rt_inv, _tmpB1, _tmpB2);
    if(add_delta) VecAXPY(drt, 1.0, _tmpB2);
    else          VecCopy(_tmpB2, drt);
    if(neg_scale) VecScale(drt, -1.0);

    // -- density
    MatMult(pc_D_rho, dw, _tmpB1);
    VecAXPY(F_rho, 1.0, _tmpB1);
    if(neg_scale) {
        MatMult(vo->VB_inv, F_rho, drho);
        VecScale(drho, -1.0);
    }
/*
    MatMult(pc_D_rho, dw, _tmpB1);
    VecAXPY(_tmpB1, 1.0, F_rho);
    MatMult(vo->VB_inv, _tmpB1, _tmpB2);
    if(add_delta) VecAXPY(drho, 1.0, _tmpB2);
    else          VecCopy(_tmpB2, drho);
    if(neg_scale) VecScale(drho, -1.0);
*/
}

/****************************************************************************************************************************************/

void VertSolve::update_residuals(int ex, int ey, Vec theta, Vec rho, Vec rt, Vec exner, Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner) {
    // [u,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_G);
    MatScale(pc_G, 0.5*dt);

    // [u,rho] block
    vo->AssembleLinearWithRhoInv(ex, ey, rho, vo->VA_inv);
    MatMatMult(vo->VA, vo->VA_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(pc_V0_invV0_rt, vo->V01, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_A_u);
    MatScale(pc_A_u, 0.5*dt*RD/CV);

    // [rho,u] block
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_D_rho);
    MatScale(pc_D_rho, 0.5*dt);

    // [rt,u] block
    vo->AssembleLinearWithRT(ex, ey, rt, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_D_rt);
    MatScale(pc_D_rt, 0.5*dt);

    // [exner,rt] block
    vo->Assemble_EOS_BlockInv(ex, ey, rt, NULL, pc_N_rt_inv);
    MatScale(pc_N_rt_inv, -1.0*CV/RD);

    // [exner,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_N_exner);

    // 1. density corrections
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(pc_A_u, vo->VB_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_A_u_VB_inv);
    MatMatMult(pc_A_u_VB_inv, pc_D_rho, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_M_u);

    vo->AssembleLinear(ex, ey, vo->VA);
    MatAYPX(pc_M_u, -1.0, vo->VA, DIFFERENT_NONZERO_PATTERN);

    MatMult(pc_A_u_VB_inv, F_rho, _tmpA1);
    VecAXPY(F_w, -1.0, _tmpA1);
    
    // 2. density weighted potential temperature correction
    MatMatMult(vo->VB, pc_N_rt_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VB_N_rt_inv);
    MatMatMult(pc_VB_N_rt_inv, pc_N_exner, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_N_exner_2);
    MatMult(pc_VB_N_rt_inv, F_exner, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);

    // 3. schur complement solve for exner pressure
#ifdef VISC
    MatMatMult(pc_DTV1, vo->V10, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VISC);
    MatAYPX(pc_VISC, -0.5*dt*visc, pc_M_u, DIFFERENT_NONZERO_PATTERN);
    MatGetDiagonal(pc_VISC, _tmpA1);
#else
    MatGetDiagonal(pc_M_u, _tmpA1);
#endif
    VecSet(_tmpA2, 1.0);
    VecPointwiseDivide(_tmpA2, _tmpA2, _tmpA1);
    MatZeroEntries(pc_M_u_inv);
    MatDiagonalSet(pc_M_u_inv, _tmpA2, INSERT_VALUES);

    MatMatMult(pc_D_rt, pc_M_u_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_D_rt_M_u_inv);
    MatMatMult(pc_D_rt_M_u_inv, pc_G, MAT_REUSE_MATRIX, PETSC_DEFAULT, &_PCz);
    MatMult(pc_D_rt_M_u_inv, F_w, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);
}

void VertSolve::assemble_pc(int ex, int ey, Vec theta, Vec rho, Vec rt, Vec exner, bool eos_update) {
    // [u,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_G);
    MatScale(pc_G, 0.5*dt);

    // [u,rho] block
    vo->AssembleLinearWithRhoInv(ex, ey, rho, vo->VA_inv);
    MatMatMult(vo->VA, vo->VA_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(pc_V0_invV0_rt, vo->V01, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_A_u);
    MatScale(pc_A_u, 0.5*dt*RD/CV);

    // [rho,u] block
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_D_rho);
    MatScale(pc_D_rho, 0.5*dt);

    // [rt,u] block
    vo->AssembleLinearWithRT(ex, ey, rt, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_D_rt);
    MatScale(pc_D_rt, 0.5*dt);

    // [exner,rt] block
    vo->Assemble_EOS_BlockInv(ex, ey, rt, NULL, pc_N_rt_inv);
    MatScale(pc_N_rt_inv, -1.0*CV/RD);

    // [exner,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_N_exner);

    // 1. density corrections
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(pc_A_u, vo->VB_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_A_u_VB_inv);
    MatMatMult(pc_A_u_VB_inv, pc_D_rho, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_M_u);

    vo->AssembleLinear(ex, ey, vo->VA);
    MatAYPX(pc_M_u, -1.0, vo->VA, DIFFERENT_NONZERO_PATTERN);
#ifdef RAYLEIGH
    vo->AssembleRayleigh(ex, ey, vo->VA);
    MatAXPY(pc_M_u, 0.5*dt*RAYLEIGH, vo->VA, DIFFERENT_NONZERO_PATTERN);
#endif

    // 2. density weighted potential temperature correction
    if(eos_update) {
        MatMatMult(vo->VB, pc_N_rt_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VB_N_rt_inv);
        MatMatMult(pc_VB_N_rt_inv, pc_N_exner, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_N_exner_2);
    }

    // 3. schur complement solve for exner pressure
#ifdef VISC
    MatMatMult(pc_DTV1, vo->V10, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VISC);
    MatAYPX(pc_VISC, -0.5*dt*visc, pc_M_u, DIFFERENT_NONZERO_PATTERN);
    MatGetDiagonal(pc_VISC, _tmpA1);
#else
    MatGetDiagonal(pc_M_u, _tmpA1);
#endif
    VecSet(_tmpA2, 1.0);
    VecPointwiseDivide(_tmpA2, _tmpA2, _tmpA1);
    MatZeroEntries(pc_M_u_inv);
    MatDiagonalSet(pc_M_u_inv, _tmpA2, INSERT_VALUES);

    MatMatMult(pc_D_rt, pc_M_u_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_D_rt_M_u_inv);
    MatMatMult(pc_D_rt_M_u_inv, pc_G, MAT_REUSE_MATRIX, PETSC_DEFAULT, &_PCz);
    MatMatMult(pc_D_rt_M_u_inv, pc_G, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_LAP); // TODO: optimize
    if(eos_update) {
        MatAXPY(_PCz, 1.0, pc_N_exner_2, DIFFERENT_NONZERO_PATTERN);
    }
    MatScale(_PCz, -1.0);
}

void VertSolve::viscosity() {
    double dzMaxG, dzMax = 1.0e-6;

    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ii = 0; ii < topo->n0; ii++) {
            if(geom->thick[kk][ii] > dzMax) {
                dzMax = geom->thick[kk][ii];
            }
        }
    }
    MPI_Allreduce(&dzMax, &dzMaxG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    visc = 8.0*dzMaxG/M_PI;
}
