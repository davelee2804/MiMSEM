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
#include "Assembly.h"
#include "Schur.h"
#include "VertSolve_4.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
//#define RAYLEIGH 0.4
//#define VISC 0
//#define NEW_EOS 1

using namespace std;

VertSolve::VertSolve(Topo* _topo, Geom* _geom, double _dt) {
    int ii, elOrd2;
    PC pc;

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
    ksp_pi = ksp_rho = NULL;
    pc_V01VBA = NULL;
    G_rho = G_rt = NULL;
    UdotGRAD = NULL;
    M_u_inv = NULL;

    viscosity();

    KSPCreate(MPI_COMM_SELF, &ksp_w);
    KSPSetOperators(ksp_w, vo->VA, vo->VA);
    KSPGetPC(ksp_w, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(ksp_w, "ksp_w_");
    KSPSetFromOptions(ksp_w);
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

    delete vo;

    delete edge;
    delete node;
    delete quad;

    if(_PCz)    MatDestroy(&_PCz);
    if(ksp_w)   KSPDestroy(&ksp_w);
    if(ksp_pi)  KSPDestroy(&ksp_pi);
    if(ksp_rho) KSPDestroy(&ksp_rho);
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
    L2Vecs* theta_j = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* F_z = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* G_z = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* dF_z = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* dG_z = new L2Vecs(geom->nk, topo, geom);
    Vec F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, F, dx;
    PC pc;
    Mat PC_coupled = NULL;
    KSP ksp_coupled = NULL;
    L2Vecs* velz_h = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h = new L2Vecs(geom->nk, topo, geom);

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
    theta_j->CopyFromVert(theta_i->vz);
    theta_j->VertToHoriz();

    exner_h->CopyFromHoriz(exner_i->vh);
    exner_h->UpdateLocal();
    exner_h->HorizToVert();

    velz_h->CopyFromVert(velz_i->vz);
    rho_h->CopyFromVert(rho_i->vz);
    exner_h->CopyFromVert(exner_i->vz);

    do {
        max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // implicit coupled solve
            assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                              rt_i->vz[ii], rt_j->vz[ii], F_w, F_z->vz[ii], G_z->vz[ii]);
#ifdef NEW_EOS
            vo->Assemble_EOS_Residual_new(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);
#else
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);
#endif
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

            assemble_operator(ex, ey, theta_j->vz[ii], velz_j->vz[ii], rho_j->vz[ii], rt_j->vz[ii], exner_j->vz[ii], &PC_coupled);

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

            VecZeroEntries(velz_h->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_i->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_j->vz[ii]);
   
            VecZeroEntries(rho_h->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_i->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_j->vz[ii]);
   
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            max_norm_exner = MaxNorm(d_exner, exner_j->vz[ii], max_norm_exner);
            max_norm_w     = MaxNorm(d_w,     velz_j->vz[ii],  max_norm_w    );
            max_norm_rho   = MaxNorm(d_rho,   rho_j->vz[ii],   max_norm_rho  );
            max_norm_rt    = MaxNorm(d_rt,    rt_j->vz[ii],    max_norm_rt   );
        }

        diagTheta2(rho_j->vz, rt_j->vz, theta_j->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(theta_h->vz[ii]);
            VecAXPY(theta_h->vz[ii], 0.5, theta_j->vz[ii]);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
        }
        theta_h->VertToHoriz();
        theta_j->VertToHoriz();

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
    delete theta_j;
    delete theta_h;
    delete F_z;
    delete G_z;
    delete dF_z;
    delete dG_z;
    delete velz_h;
    delete rho_h;
    delete rt_h;
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
    double norm_x, max_norm_w, max_norm_exner, max_norm_rho, max_norm_rt, alpha = 1.0;
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_j = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    Vec F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, F_z, G_z, dF_z, dG_z;

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
    theta_j->CopyFromVert(theta_i->vz);
    theta_j->VertToHoriz();

    exner_h->CopyFromHoriz(exner_i->vh);
    exner_h->UpdateLocal();
    exner_h->HorizToVert();

    do {
        max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // assemble the residual vectors
            assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                              rt_i->vz[ii], rt_j->vz[ii], F_w, F_z, G_z);
#ifdef NEW_EOS
            vo->Assemble_EOS_Residual_new(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);
#else
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);
#endif

            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii]);
            MatMult(vo->VB, dF_z, F_rho);
            MatMult(vo->VB, dG_z, F_rt);

            solve_schur_column(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_h->vz[ii], 
                               F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner);

            alpha = LineSearch(velz_i->vz[ii],  velz_j->vz[ii],  d_w, 
                               rho_i->vz[ii],   rho_j->vz[ii],   d_rho, 
                               rt_i->vz[ii],    rt_j->vz[ii],    d_rt, 
                               exner_i->vz[ii], exner_j->vz[ii], d_exner, exner_h->vz[ii],
                               theta_i->vz[ii], theta_h->vz[ii], ii);

            VecScale(d_w,     alpha);
            VecScale(d_rho,   alpha);
            VecScale(d_rt,    alpha);
            VecScale(d_exner, alpha);

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

        diagTheta2(rho_j->vz, rt_j->vz, theta_j->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(theta_h->vz[ii]);
            VecAXPY(theta_h->vz[ii], 0.5, theta_j->vz[ii]);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
        }
        theta_h->VertToHoriz();
        theta_j->VertToHoriz();

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
    exner_i->CopyFromVert(exner_j->vz);

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
    delete theta_j;
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

    VecZeroEntries(Phi);
    VecZeroEntries(_tmpB2);

    // kinetic energy term
    MatZeroEntries(vo->VBA);
    vo->AssembleConLinWithW(ex, ey, velz1, vo->VBA);

    MatMult(vo->VBA, velz1, _tmpB1);
    VecAXPY(Phi, 1.0/6.0, _tmpB1);
    
    MatMult(vo->VBA, velz2, _tmpB1);
    VecAXPY(Phi, 1.0/6.0, _tmpB1);

    MatZeroEntries(vo->VBA);
    vo->AssembleConLinWithW(ex, ey, velz2, vo->VBA);

    MatMult(vo->VBA, velz2, _tmpB1);
    VecAXPY(Phi, 1.0/6.0, _tmpB1);

    // potential energy term
    VecAXPY(Phi, 1.0, zv[ei]);
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

void VertSolve::assemble_residual(int ex, int ey, Vec theta, Vec Pi, 
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

    if(!*_PC) MatCreateSeqAIJ(MPI_COMM_SELF, nDofsTotal, nDofsTotal, 12*n2, NULL, _PC);
    MatZeroEntries(*_PC);

    // [u,u] block
    vo->AssembleLinear(ex, ey, vo->VA);
#ifdef RAYLEIGH
    vo->AssembleRayleigh(ex, ey, vo->VA_inv);
    MatAXPY(vo->VA, 0.5*dt*RAYLEIGH, vo->VA_inv, DIFFERENT_NONZERO_PATTERN);
#endif
    vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    MatMatMult(vo->V01, vo->VBA, reuse, PETSC_DEFAULT, &pc_V01VBA);
    MatAYPX(pc_V01VBA, 0.5*dt*2.0, vo->VA, DIFFERENT_NONZERO_PATTERN);
    MatGetOwnershipRange(pc_V01VBA, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V01VBA, mm, &nCols, &cols, &vals);
        ri = mm;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci];
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(pc_V01VBA, mm, &nCols, &cols, &vals);
    }

    // [u,rt]
    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, exner, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &pc_A_u);
    MatScale(pc_A_u, 0.5*dt);
    MatGetOwnershipRange(pc_A_u, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_A_u, mm, &nCols, &cols, &vals);
        ri = mm;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(pc_A_u, mm, &nCols, &cols, &vals);
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
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(pc_GRAD, mm, &nCols, &cols, &vals);
    }

    // [rho,u] block
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_V1DV0_invV0_rt);
    MatScale(pc_V1DV0_invV0_rt, 0.5*dt);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V1DV0_invV0_rt, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci];
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(pc_V1DV0_invV0_rt, mm, &nCols, &cols, &vals);
    }

    // [rho,rho] block
    vo->AssembleConst(ex, ey, vo->VB);
    MatGetOwnershipRange(vo->VB, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(vo->VB, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(vo->VB, mm, &nCols, &cols, &vals);
    }

    // [rt,u] block
    vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    MatMatMult(vo->VB, vo->V10, reuse, PETSC_DEFAULT, &pc_V1D);
    MatScale(pc_V1D, 0.5*dt);
    MatGetOwnershipRange(pc_V1D, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V1D, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci];
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(pc_V1D, mm, &nCols, &cols, &vals);
    }

    // [rt,rho] block
    vo->AssembleConstWithTheta(ex, ey, theta, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_V1_invV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_V1_invV1, reuse, PETSC_DEFAULT, &pc_V01V1_invV1);
    vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    MatMatMult(vo->VBA, pc_V01V1_invV1, reuse, PETSC_DEFAULT, &pc_V1DV0_invV01V1_invV1);
    MatScale(pc_V1DV0_invV01V1_invV1, 0.5*dt);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V1DV0_invV01V1_invV1, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(pc_V1DV0_invV01V1_invV1, mm, &nCols, &cols, &vals);
    }

    // [rt,rt] block
    vo->AssembleConst(ex, ey, vo->VB);
    MatGetOwnershipRange(vo->VB, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(vo->VB, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
        MatRestoreRow(vo->VB, mm, &nCols, &cols, &vals);
    }

    // [exner,rt] block
#ifdef NEW_EOS
    vo->AssembleN_RT(ex, ey, rt, exner, vo->VB);
    MatGetOwnershipRange(vo->VB, &mi, &mf);
#else
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_VBVB_rt_invVB_pi);
    MatScale(pc_VBVB_rt_invVB_pi, -RD/CV);
    MatGetOwnershipRange(pc_VBVB_rt_invVB_pi, &mi, &mf);
#endif
    for(mm = mi; mm < mf; mm++) {
#ifdef NEW_EOS
        MatGetRow(vo->VB, mm, &nCols, &cols, &vals);
#else
        MatGetRow(pc_VBVB_rt_invVB_pi, mm, &nCols, &cols, &vals);
#endif
        ri = mm + nDofsW + 2*nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
#ifdef NEW_EOS
        MatRestoreRow(vo->VB, mm, &nCols, &cols, &vals);
#else
        MatRestoreRow(pc_VBVB_rt_invVB_pi, mm, &nCols, &cols, &vals);
#endif
    }

    // [exner,exner] block
#ifdef NEW_EOS
    vo->AssembleN_PiInv(ex, ey, rt, exner, vo->VB, false);
    MatGetOwnershipRange(vo->VB, &mi, &mf);
#else
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VBVB_rt_invVB_pi);
    MatGetOwnershipRange(pc_VBVB_rt_invVB_pi, &mi, &mf);
#endif
    for(mm = mi; mm < mf; mm++) {
#ifdef NEW_EOS
        MatGetRow(vo->VB, mm, &nCols, &cols, &vals);
#else
        MatGetRow(pc_VBVB_rt_invVB_pi, mm, &nCols, &cols, &vals);
#endif
        ri = mm + nDofsW + 2*nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + 2*nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, ADD_VALUES);
#ifdef NEW_EOS
        MatRestoreRow(vo->VB, mm, &nCols, &cols, &vals);
#else
        MatRestoreRow(pc_VBVB_rt_invVB_pi, mm, &nCols, &cols, &vals);
#endif
    }

    MatAssemblyBegin(*_PC, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  *_PC, MAT_FINAL_ASSEMBLY);
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

void VertSolve::solve_schur_column(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                   Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi) 
{
    int n2 = topo->elOrd*topo->elOrd;
    MatReuse reuse = (!_PCz) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    if(!_PCz) {
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk-1)*n2, (geom->nk-1)*n2, n2, NULL, &M_u_inv);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &M_rho_inv);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &M_rt);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &N_pi_inv);
#ifdef NEW_EOS
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &N_rt);
#endif
    }

    // assemble the operators for the coupled system
#ifdef RAYLEIGH
    vo->AssembleLinearWithRayleighInv(ex, ey, 0.5*dt*RAYLEIGH, M_u_inv);
#else
    vo->AssembleLinearInv(ex, ey, M_u_inv);
#endif
    vo->AssembleConst(ex, ey, M_rt);
    vo->AssembleConstInv(ex, ey, M_rho_inv);
#ifdef NEW_EOS
    vo->AssembleN_PiInv(ex, ey, rt, pi, N_pi_inv, true);
#else
    vo->Assemble_EOS_BlockInv(ex, ey, pi, NULL, N_pi_inv);
#endif
    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, 0.5*dt);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);

    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);

    vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    MatMatMult(vo->VB, vo->V10, reuse, PETSC_DEFAULT, &D_rt);
    MatScale(D_rt, 0.5*dt);
#ifdef NEW_EOS
    vo->AssembleN_RT(ex, ey, rt, pi, N_rt);
#else
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
#endif
    vo->AssembleConstWithTheta(ex, ey, theta, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    MatMatMult(vo->VBA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &Q_rt_rho);
    MatScale(Q_rt_rho, 0.5*dt);

    // assemble the secondary operators
    MatMatMult(D_rho, M_u_inv, reuse, PETSC_DEFAULT, &D_rho_M_u_inv);
    MatMatMult(D_rt,  M_u_inv, reuse, PETSC_DEFAULT, &D_rt_M_u_inv );

    MatMatMult(D_rho_M_u_inv, G_rt, reuse, PETSC_DEFAULT, &L_rho_rt);
    MatMatMult(D_rho_M_u_inv, G_pi, reuse, PETSC_DEFAULT, &L_rho_pi);
    MatMatMult(D_rt_M_u_inv,  G_rt, reuse, PETSC_DEFAULT, &L_rt_rt );
    MatMatMult(D_rt_M_u_inv,  G_pi, reuse, PETSC_DEFAULT, &L_rt_pi );

    MatAYPX(L_rt_rt, -1.0, M_rt, DIFFERENT_NONZERO_PATTERN);

    MatMatMult(L_rho_pi, N_pi_inv, reuse, PETSC_DEFAULT, &L_rho_pi_N_pi_inv);
    MatMatMult(L_rt_pi,  N_pi_inv, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv );
    MatMatMult(L_rho_pi_N_pi_inv, N_rt, reuse, PETSC_DEFAULT, &L_rho_pi_N_pi_inv_N_rt);
    MatMatMult(L_rt_pi_N_pi_inv,  N_rt, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv_N_rt );

    MatMatMult(Q_rt_rho, M_rho_inv, reuse, PETSC_DEFAULT, &Q_rt_rho_M_rho_inv);

    MatAXPY(L_rho_pi_N_pi_inv_N_rt, -1.0, L_rho_rt, DIFFERENT_NONZERO_PATTERN);
    MatMatMult(Q_rt_rho_M_rho_inv, L_rho_pi_N_pi_inv_N_rt, reuse, PETSC_DEFAULT, &_PCz);
    MatAYPX(_PCz, -1.0, L_rt_pi_N_pi_inv_N_rt, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(_PCz, +1.0, L_rt_rt, DIFFERENT_NONZERO_PATTERN);

    if(reuse == MAT_INITIAL_MATRIX) {
        PC pc;

        KSPCreate(MPI_COMM_SELF, &ksp_pi);
        KSPSetOperators(ksp_pi, _PCz, _PCz);
        KSPGetPC(ksp_pi, &pc);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(ksp_pi, "ksp_pi_");
        KSPSetFromOptions(ksp_pi);
    }

    // update the residuals
    MatMult(D_rho_M_u_inv, F_u, _tmpB1);
    VecAXPY(F_rho, -1.0, _tmpB1);         // F_{rho}'
    MatMult(D_rt_M_u_inv,  F_u, _tmpB1);
    VecAXPY(F_rt,  -1.0, _tmpB1);         // F_{rt}'

    MatMult(L_rho_pi_N_pi_inv, F_pi, _tmpB1);
    VecAXPY(F_rho, +1.0, _tmpB1);         // F_{rho}''
    MatMult(L_rt_pi_N_pi_inv,  F_pi, _tmpB1);
    VecAXPY(F_rt,  +1.0, _tmpB1);         // F_{rt}''

    MatMult(Q_rt_rho_M_rho_inv, F_rho, _tmpB1);
    VecAYPX(F_rt, -1.0, _tmpB1);          // F_{rt}'''

    KSPSolve(ksp_pi, F_rt, d_rt);

    // back substitute
    MatMult(N_rt, d_rt, _tmpB1);
    VecAXPY(F_pi, 1.0, _tmpB1);
    MatMult(N_pi_inv, F_pi, d_pi);
    VecScale(d_pi, -1.0);

    MatMult(L_rho_pi_N_pi_inv_N_rt, d_rt, _tmpB1);
    VecAXPY(F_rho, 1.0, _tmpB1);
    MatMult(M_rho_inv, F_rho, d_rho);
    VecScale(d_rho, -1.0);

    MatMult(G_rt, d_rt, _tmpA1);
    MatMult(G_pi, d_pi, _tmpA2);
    VecAXPY(F_u, 1.0, _tmpA1);
    VecAXPY(F_u, 1.0, _tmpA2);
    VecScale(F_u, -1.0);
#ifdef RAYLEIGH
    vo->AssembleLinear(ex, ey, vo->VA);
    vo->AssembleRayleigh(ex, ey, vo->VA_inv);
    MatAXPY(vo->VA, 0.5*dt*RAYLEIGH, vo->VA_inv, DIFFERENT_NONZERO_PATTERN);
    KSPSolve(ksp_w, F_u, d_u);
#else
    MatMult(M_u_inv, F_u, d_u);
#endif
}

void VertSolve::assemble_and_update(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                    Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Schur* schur)
{
    int n2 = topo->elOrd*topo->elOrd;
    MatReuse reuse = (!G_rt) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    if(!M_u_inv) {
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk-1)*n2, (geom->nk-1)*n2, n2, NULL, &M_u_inv);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &M_rho_inv);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &M_rt);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &N_pi_inv);
#ifdef NEW_EOS
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &N_rt);
#endif
    }

    // assemble the operators for the coupled system
#ifdef RAYLEIGH
    vo->AssembleLinearWithRayleighInv(ex, ey, 0.5*dt*RAYLEIGH, M_u_inv);
#else
    vo->AssembleLinearInv(ex, ey, M_u_inv);
#endif
    vo->AssembleConst(ex, ey, M_rt);
    vo->AssembleConstInv(ex, ey, M_rho_inv);
#ifdef NEW_EOS
    vo->AssembleN_PiInv(ex, ey, rt, pi, N_pi_inv, true);
#else
    vo->Assemble_EOS_BlockInv(ex, ey, pi, NULL, N_pi_inv);
#endif
    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, 0.5*dt);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);

    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);

    vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    MatMatMult(vo->VB, vo->V10, reuse, PETSC_DEFAULT, &D_rt);
    MatScale(D_rt, 0.5*dt);
#ifdef NEW_EOS
    vo->AssembleN_RT(ex, ey, rt, pi, N_rt);
#else
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
#endif
    vo->AssembleConstWithTheta(ex, ey, theta, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    MatMatMult(vo->VBA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &Q_rt_rho);
    MatScale(Q_rt_rho, 0.5*dt);
/*
    MatMatMult(vo->VB, M_rho_inv, reuse, PETSC_DEFAULT, &pc_VBtheta_VBinv);
    MatMult(vo->V10, velz, _tmpB1);
    vo->AssembleConstWithRho(ex, ey, _tmpB1, vo->VB);
    MatMatMult(pc_VBtheta_VBinv, vo->VB, reuse, PETSC_DEFAULT, &pc_VBtheta_VBinv_VBdu);
    MatAXPY(Q_rt_rho, 0.5*dt, pc_VBtheta_VBinv_VBdu, DIFFERENT_NONZERO_PATTERN);
*/
    // assemble the secondary operators
    MatMatMult(D_rho, M_u_inv, reuse, PETSC_DEFAULT, &D_rho_M_u_inv);
    MatMatMult(D_rt,  M_u_inv, reuse, PETSC_DEFAULT, &D_rt_M_u_inv );

    MatMatMult(D_rho_M_u_inv, G_rt, reuse, PETSC_DEFAULT, &L_rho_rt);
    MatMatMult(D_rho_M_u_inv, G_pi, reuse, PETSC_DEFAULT, &L_rho_pi);
    MatMatMult(D_rt_M_u_inv,  G_rt, reuse, PETSC_DEFAULT, &L_rt_rt );
    MatMatMult(D_rt_M_u_inv,  G_pi, reuse, PETSC_DEFAULT, &L_rt_pi );

    MatAYPX(L_rt_rt, -1.0, M_rt, DIFFERENT_NONZERO_PATTERN);

    MatMatMult(L_rho_pi, N_pi_inv, reuse, PETSC_DEFAULT, &L_rho_pi_N_pi_inv);
    MatMatMult(L_rt_pi,  N_pi_inv, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv );
    MatMatMult(L_rho_pi_N_pi_inv, N_rt, reuse, PETSC_DEFAULT, &L_rho_pi_N_pi_inv_N_rt);
    MatMatMult(L_rt_pi_N_pi_inv,  N_rt, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv_N_rt );

    MatMatMult(Q_rt_rho, M_rho_inv, reuse, PETSC_DEFAULT, &Q_rt_rho_M_rho_inv);

    MatAXPY(L_rho_pi_N_pi_inv_N_rt, -1.0, L_rho_rt, DIFFERENT_NONZERO_PATTERN);

    if(schur) {
        vo->AssembleConst(ex, ey, vo->VB);
        vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
        MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
        MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
        MatScale(N_rt, -1.0*RD/CV);

        vo->AssembleConstInv(ex, ey, vo->VB_inv);

        MatMatMult(L_rt_pi_N_pi_inv, N_rt, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv_N_rho);
        //MatMatMult(L_rt_pi_N_pi_inv_N_rho, vo->VB_inv, reuse, PETSC_DEFAULT, &L_rt_pi_N_pi_inv_N_rho_M_inv);
        //schur->AddFromVertMat(ey*topo->nElsX + ex, L_rt_pi_N_pi_inv_N_rho_M_inv, schur->Q);
        schur->AddFromVertMat(ey*topo->nElsX + ex, L_rt_pi_N_pi_inv_N_rho, schur->D);

        MatMatMult(L_rho_pi_N_pi_inv, N_rt, reuse, PETSC_DEFAULT, &L_rho_pi_N_pi_inv_N_rho);
        //schur->AddFromVertMat(ey*topo->nElsX + ex, L_rho_pi_N_pi_inv_N_rho, schur->P);

        //schur->AddFromVertMat(ey*topo->nElsX + ex, Q_rt_rho_M_rho_inv, schur->Q);
        schur->AddFromVertMat(ey*topo->nElsX + ex, Q_rt_rho, schur->D);
        schur->AddFromVertMat(ey*topo->nElsX + ex, L_rho_pi_N_pi_inv_N_rt, schur->G);
        schur->AddFromVertMat(ey*topo->nElsX + ex, L_rt_rt, schur->T);
        schur->AddFromVertMat(ey*topo->nElsX + ex, L_rt_pi_N_pi_inv_N_rt, schur->T);
    } else {
        MatMatMult(Q_rt_rho_M_rho_inv, L_rho_pi_N_pi_inv_N_rt, reuse, PETSC_DEFAULT, &_PCz);
        MatAYPX(_PCz, -1.0, L_rt_pi_N_pi_inv_N_rt, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(_PCz, +1.0, L_rt_rt, DIFFERENT_NONZERO_PATTERN);
    }

    // update the residuals
    MatMult(D_rho_M_u_inv, F_u, _tmpB1);
    VecAXPY(F_rho, -1.0, _tmpB1);         // F_{rho}'
    MatMult(D_rt_M_u_inv,  F_u, _tmpB1);
    VecAXPY(F_rt,  -1.0, _tmpB1);         // F_{rt}'

    MatMult(L_rho_pi_N_pi_inv, F_pi, _tmpB1);
    VecAXPY(F_rho, +1.0, _tmpB1);         // F_{rho}''
    MatMult(L_rt_pi_N_pi_inv,  F_pi, _tmpB1);
    VecAXPY(F_rt,  +1.0, _tmpB1);         // F_{rt}''

    if(!schur) {
        MatMult(Q_rt_rho_M_rho_inv, F_rho, _tmpB1);
        VecAXPY(F_rt, -1.0, _tmpB1);          // F_{rt}''' (-ve scaling done outside)
    }
}

void VertSolve::update_deltas(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                              Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi) 
{
    MatReuse reuse = MAT_REUSE_MATRIX;

    // assemble the operators for the coupled system
#ifdef RAYLEIGH
    vo->AssembleLinearWithRayleighInv(ex, ey, 0.5*dt*RAYLEIGH, M_u_inv);
#else
    vo->AssembleLinearInv(ex, ey, M_u_inv);
#endif
    vo->AssembleConstInv(ex, ey, M_rho_inv);
#ifdef NEW_EOS
    vo->AssembleN_PiInv(ex, ey, rt, pi, N_pi_inv, true);
#else
    vo->Assemble_EOS_BlockInv(ex, ey, pi, NULL, N_pi_inv);
#endif
    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, 0.5*dt);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);

    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);
#ifdef NEW_EOS
    vo->AssembleN_RT(ex, ey, rt, pi, N_rt);
#else
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
#endif
    vo->AssembleConstWithTheta(ex, ey, theta, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    MatMatMult(vo->VBA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &Q_rt_rho);
    MatScale(Q_rt_rho, 0.5*dt);

    // assemble the secondary operators
    MatMatMult(D_rho, M_u_inv, reuse, PETSC_DEFAULT, &D_rho_M_u_inv);

    MatMatMult(D_rho_M_u_inv, G_rt, reuse, PETSC_DEFAULT, &L_rho_rt);
    MatMatMult(D_rho_M_u_inv, G_pi, reuse, PETSC_DEFAULT, &L_rho_pi);

    MatMatMult(L_rho_pi_N_pi_inv, N_rt, reuse, PETSC_DEFAULT, &L_rho_pi_N_pi_inv_N_rt);
    MatAXPY(L_rho_pi_N_pi_inv_N_rt, -1.0, L_rho_rt, DIFFERENT_NONZERO_PATTERN);

    // back substitute
    MatMult(N_rt, d_rt, _tmpB1);
    VecAXPY(F_pi, 1.0, _tmpB1);
    MatMult(N_pi_inv, F_pi, d_pi);
    VecScale(d_pi, -1.0);

    MatMult(L_rho_pi_N_pi_inv_N_rt, d_rt, _tmpB1);
    VecAXPY(F_rho, 1.0, _tmpB1);
    MatMult(M_rho_inv, F_rho, d_rho);
    VecScale(d_rho, -1.0);

    MatMult(G_rt, d_rt, _tmpA1);
    MatMult(G_pi, d_pi, _tmpA2);
    VecAXPY(F_u, 1.0, _tmpA1);
    VecAXPY(F_u, 1.0, _tmpA2);
    VecScale(F_u, -1.0);
#ifdef RAYLEIGH
    vo->AssembleLinear(ex, ey, vo->VA);
    vo->AssembleRayleigh(ex, ey, vo->VA_inv);
    MatAXPY(vo->VA, 0.5*dt*RAYLEIGH, vo->VA_inv, DIFFERENT_NONZERO_PATTERN);
    KSPSolve(ksp_w, F_u, d_u);
#else
    MatMult(M_u_inv, F_u, d_u);
#endif
}

void VertSolve::update_delta_u(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                               Vec F_u, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi) 
{
    MatReuse reuse = MAT_REUSE_MATRIX;

    // assemble the operators for the coupled system
#ifdef RAYLEIGH
    vo->AssembleLinearWithRayleighInv(ex, ey, 0.5*dt*RAYLEIGH, M_u_inv);
#else
    vo->AssembleLinearInv(ex, ey, M_u_inv);
#endif
    vo->AssembleConstInv(ex, ey, M_rho_inv);
#ifdef NEW_EOS
    vo->AssembleN_PiInv(ex, ey, rt, pi, N_pi_inv, true);
#else
    vo->Assemble_EOS_BlockInv(ex, ey, pi, NULL, N_pi_inv);
#endif
    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, 0.5*dt);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);

#ifdef NEW_EOS
    vo->AssembleN_RT(ex, ey, rt, pi, N_rt);
#else
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
#endif

    // back substitute
/*
    MatMult(N_rt, d_rt, _tmpB1);
    VecAXPY(F_pi, 1.0, _tmpB1);
    MatMult(N_pi_inv, F_pi, d_pi);
    VecScale(d_pi, -1.0);
*/
    MatMult(G_rt, d_rt, _tmpA1);
    MatMult(G_pi, d_pi, _tmpA2);
    VecAXPY(F_u, 1.0, _tmpA1);
    VecAXPY(F_u, 1.0, _tmpA2);
    VecScale(F_u, -1.0);
/*
#ifdef RAYLEIGH
    vo->AssembleLinear(ex, ey, vo->VA);
    vo->AssembleRayleigh(ex, ey, vo->VA_inv);
    MatAXPY(vo->VA, 0.5*dt*RAYLEIGH, vo->VA_inv, DIFFERENT_NONZERO_PATTERN);
    KSPSolve(ksp_w, F_u, d_u);
#else
    MatMult(M_u_inv, F_u, d_u);
#endif
*/
    vo->AssembleLinear(ex, ey, vo->VA);
    vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    if(!pc_V01VBA) {
        PC pc;
        MatMatMult(vo->V01, vo->VBA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &pc_V01VBA);
        if(ksp_w) KSPDestroy(&ksp_w);
        KSPCreate(MPI_COMM_SELF, &ksp_w);
        //KSPSetOperators(ksp_w, pc_V01VBA, pc_V01VBA);
        KSPSetOperators(ksp_w, vo->VA, vo->VA);
        KSPGetPC(ksp_w, &pc);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(ksp_w, "ksp_w_");
        KSPSetFromOptions(ksp_w);
    } else {
        MatMatMult(vo->V01, vo->VBA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V01VBA);
    }
    MatAYPX(pc_V01VBA, 0.5*dt*2.0, vo->VA, DIFFERENT_NONZERO_PATTERN);
    KSPSolve(ksp_w, F_u, d_u);
}

double VertSolve::LineSearch(Vec velz_i,  Vec velz_j, Vec d_velz, 
                             Vec rho_i,   Vec rho_j,  Vec d_rho, 
                             Vec rt_i,    Vec rt_j,   Vec d_rt, 
                             Vec pi_i,    Vec pi_j,   Vec d_pi, Vec pi_h,
                             Vec theta_i, Vec theta_h, int ei) {
    bool   done  = false;
    int    ex, ey;
    int    n2    = topo->elOrd*topo->elOrd;
    double alpha = 1.0;
    double c1    = 1.0e-4;
    double dfd, fak, fakp1;
    Vec    u_tmp, h_tmp, f_tmp, F_z, G_z, velz_k, rho_k, rt_k, pi_k, frt, theta_k;
    PC     pc;
    KSP    kspCol;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &u_tmp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &f_tmp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &h_tmp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &velz_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &rho_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &rt_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &pi_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &frt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &theta_k);

    ex = ei%topo->nElsX;
    ey = ei/topo->nElsX;

/*
    vo->AssembleLinear(ex, ey, vo->VA);
#ifdef RAYLEIGH
    vo->AssembleRayleigh(ex, ey, vo->VA_inv);
    MatAXPY(vo->VA, 0.5*dt*RAYLEIGH, vo->VA_inv, DIFFERENT_NONZERO_PATTERN);
#endif
    MatMult(vo->VA, d_velz, f_tmp);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi_h, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv(ex, ey, rho_j, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &G_rt);

    MatMult(G_rt, d_rt, u_tmp);
    VecAXPY(f_tmp, 0.5*dt, u_tmp);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta_h, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &G_pi);

    MatMult(G_pi, d_pi, u_tmp);
    VecAXPY(f_tmp, 0.5*dt, u_tmp);
    VecNorm(f_tmp, NORM_2, &dfd);
*/
    assemble_and_update(ex, ey, theta_h, velz_j, rho_j, rt_j, pi_h, velz_k, rho_k, rt_k, pi_k, NULL);
    MatMult(_PCz, d_rt, h_tmp);
    VecNorm(h_tmp, NORM_2, &dfd);

    assemble_residual(ex, ey, theta_h, pi_h, velz_i, velz_j, rho_i, rho_j, rt_i, rt_j, f_tmp, F_z, G_z);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->V10, G_z, _tmpB2);
    MatMult(vo->VB, rt_j, h_tmp);
    MatMult(vo->VB, rt_i, _tmpB1);
    VecAXPY(h_tmp, -1.0, _tmpB1);
    MatMult(vo->VB, _tmpB2, _tmpB1);
    VecAXPY(h_tmp, dt, _tmpB1);

    VecNorm(h_tmp, NORM_2, &fak);

    KSPCreate(MPI_COMM_SELF, &kspCol);
    KSPSetOperators(kspCol, vo->VA2, vo->VA2);
    KSPGetPC(kspCol, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspCol, "kspCol_");
    KSPSetFromOptions(kspCol);

    do {
        VecWAXPY(velz_k, alpha, d_velz, velz_j);
        VecWAXPY(rho_k,  alpha, d_rho,  rho_j );
        VecWAXPY(rt_k,   alpha, d_rt,   rt_j  );
        VecWAXPY(pi_k,   alpha, d_pi,   pi_j  );
        VecAXPY(pi_k, 1.0, pi_i);
        VecScale(pi_k, 0.5);

        vo->AssembleLinCon2(ex, ey, vo->VAB2);
        MatMult(vo->VAB2, rt_k, frt);
        vo->AssembleLinearWithRho2(ex, ey, rho_k, vo->VA2);
        KSPSolve(kspCol, frt, theta_k);
        VecScale(theta_k, 0.5);
        VecAXPY(theta_k, 0.5, theta_i);

        assemble_residual(ex, ey, theta_k, pi_k, velz_i, velz_k, rho_i, rho_k, rt_i, rt_k, f_tmp, F_z, G_z);

        //VecNorm(f_tmp, NORM_2, &fakp1);

        vo->AssembleConst(ex, ey, vo->VB);
        MatMult(vo->V10, G_z, _tmpB2);
        MatMult(vo->VB, rt_j, h_tmp);
        MatMult(vo->VB, rt_i, _tmpB1);
        VecAXPY(h_tmp, -1.0, _tmpB1);
        MatMult(vo->VB, _tmpB2, _tmpB1);
        VecAXPY(h_tmp, dt, _tmpB1);

        VecNorm(h_tmp, NORM_2, &fakp1);

        if(fakp1 > fak + c1*alpha*dfd) alpha = 0.9*alpha;
        else                           done = true;

        if(alpha < 0.001) done = true;

        //if(!rank) cout << fakp1 << "\t" << fak + c1*alpha*dfd << "\t" << alpha << endl;
    } while(!done);
    if(!rank) cout << fakp1 << "\t" << fak + c1*alpha*dfd << "\t" << alpha << endl;

    VecDestroy(&u_tmp);
    VecDestroy(&h_tmp);
    VecDestroy(&f_tmp);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&velz_k);
    VecDestroy(&rho_k);
    VecDestroy(&rt_k);
    VecDestroy(&pi_k);
    VecDestroy(&frt);
    VecDestroy(&theta_k);
    KSPDestroy(&kspCol);

    return alpha;
}

void VertSolve::assemble_and_update_2(int ex, int ey, Vec velz, Vec rho, Vec rt, Vec pi, 
                                    Vec F_u, Vec F_rho)
{
    MatReuse reuse = (!G_rho) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    // assemble the operators for the coupled system
#ifdef RAYLEIGH
    vo->AssembleLinearWithRayleighInv(ex, ey, 0.5*dt*RAYLEIGH, M_u_inv);
#else
    vo->AssembleLinearInv(ex, ey, M_u_inv);
#endif
    vo->AssembleConst(ex, ey, M_rt);
    vo->AssembleConstInv(ex, ey, M_rho_inv);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv2(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &G_rho);
    MatScale(G_rho, -0.5*dt);

    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);

    // assemble the secondary operators
    MatMatMult(D_rho, M_u_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &D_rho_M_u_inv);
    MatMatMult(D_rho_M_u_inv, G_rho, reuse, PETSC_DEFAULT, &L_rho_rho);
    MatAYPX(L_rho_rho, -1.0, M_rt, DIFFERENT_NONZERO_PATTERN);

    // update the residuals
    MatMult(D_rho_M_u_inv, F_u, _tmpB1);
    VecAXPY(F_rho, -1.0, _tmpB1);         // F_{rho}'
}

void VertSolve::update_delta_u_2(int ex, int ey, Vec velz, Vec rho, Vec rt, Vec pi, 
                                    Vec F_u, Vec F_rho, Vec d_u, Vec d_rho)
{
    // assemble the operators for the coupled system
#ifdef RAYLEIGH
    vo->AssembleLinearWithRayleighInv(ex, ey, 0.5*dt*RAYLEIGH, M_u_inv);
#else
    vo->AssembleLinearInv(ex, ey, M_u_inv);
#endif
    vo->AssembleConst(ex, ey, M_rt);
    vo->AssembleConstInv(ex, ey, M_rho_inv);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv2(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &G_rho);
    MatScale(G_rho, -0.5*dt);

    // update the residuals
    MatMult(G_rho, d_rho, _tmpA1);
    VecAXPY(F_u, 1.0, _tmpA1);
    VecScale(F_u, -1.0);
    MatMult(M_u_inv, F_u, d_u);
}

void VertSolve::update_delta_pi(int ex, int ey, Vec rho, Vec rt, Vec pi, Vec F_pi, Vec d_rho, Vec d_rt, Vec d_pi) {
    MatReuse reuse = MAT_REUSE_MATRIX;

#ifdef NEW_EOS
    vo->AssembleN_PiInv(ex, ey, rt, pi, N_pi_inv, true);
    vo->AssembleN_RT(ex, ey, rt, pi, N_rt);
#else
    vo->Assemble_EOS_BlockInv(ex, ey, pi, NULL, N_pi_inv);
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
#endif
    MatMult(N_rt, d_rt, _tmpB1);
/*
#ifdef NEW_EOS
    vo->AssembleN_RT(ex, ey, rho, pi, N_rt);
#else
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
#endif
    MatMult(N_rt, d_rho, _tmpB2);

    VecAXPY(_tmpB1, 1.0, _tmpB2);
*/
    VecAXPY(F_pi, 1.0, _tmpB1);
    MatMult(N_pi_inv, F_pi, d_pi);
    VecScale(d_pi, -1.0);
}

void VertSolve::update_delta_pi_2(int ex, int ey, Vec rt, Vec pi, Vec F_pi, Vec d_rt, Vec d_pi) {
    MatReuse reuse = MAT_REUSE_MATRIX;

#ifdef NEW_EOS
    vo->AssembleN_PiInv(ex, ey, rt, pi, N_pi_inv, true);
    vo->AssembleN_RT(ex, ey, rt, pi, N_rt);
#else
    vo->Assemble_EOS_BlockInv(ex, ey, pi, NULL, N_pi_inv);
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
#endif
    MatMult(N_rt, d_rt, _tmpB1);
    VecAXPY(F_pi, 1.0, _tmpB1);
    MatMult(N_pi_inv, F_pi, d_pi);
    VecScale(d_pi, -1.0);
}

void VertSolve::assemble_M_rho(int ex, int ey, Vec velz, Schur* schur) {
    MatReuse reuse = (!UdotGRAD) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);

    vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);

    MatMatMult(vo->VBA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &UdotGRAD);
    MatAYPX(UdotGRAD, 0.5*dt, vo->VB, DIFFERENT_NONZERO_PATTERN);
    //MatAYPX(UdotGRAD, 0.0, vo->VB, DIFFERENT_NONZERO_PATTERN);
    schur->AddFromVertMat(ey*topo->nElsX + ex, UdotGRAD, schur->P);
}

void VertSolve::assemble_and_update_3(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                    Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec Du, Schur* schur, int itt)
{
    int n2 = topo->elOrd*topo->elOrd;
    int ei = ey * topo->nElsX + ex;
    MatReuse reuse = (!G_rt) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    if(!M_u_inv) {
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk-1)*n2, (geom->nk-1)*n2, n2, NULL, &M_u_inv);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &M_rho_inv);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &M_rt);
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &N_pi_inv);
#ifdef NEW_EOS
        MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk+0)*n2, (geom->nk+0)*n2, n2, NULL, &N_rt);
#endif
    }

    // assemble the operators for the coupled system
#ifdef RAYLEIGH
    vo->AssembleLinearWithRayleighInv(ex, ey, 0.5*dt*RAYLEIGH, M_u_inv);
#else
    vo->AssembleLinearInv(ex, ey, M_u_inv);
#endif
    vo->AssembleConst(ex, ey, M_rt);
    vo->AssembleConstInv(ex, ey, M_rho_inv);
#ifdef NEW_EOS
    vo->AssembleN_PiInv(ex, ey, rt, pi, N_pi_inv, true);
#else
    vo->Assemble_EOS_BlockInv(ex, ey, pi, NULL, N_pi_inv);
#endif
    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, 0.5*dt);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);

    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);

    vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    MatMatMult(vo->VB, vo->V10, reuse, PETSC_DEFAULT, &D_rt);
    MatScale(D_rt, 0.5*dt);
#ifdef NEW_EOS
    vo->AssembleN_RT(ex, ey, rt, pi, N_rt);
#else
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
#endif
    vo->AssembleConstWithTheta(ex, ey, theta, vo->VB);
    MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    MatMatMult(vo->VBA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &Q_rt_rho);
    MatScale(Q_rt_rho, 0.5*dt);

    MatMatMult(Q_rt_rho, M_rho_inv, reuse, PETSC_DEFAULT, &Q_rt_rho_M_rho_inv);

    MatMatMult(G_pi, N_pi_inv, reuse, PETSC_DEFAULT, &G_pi_N_pi_inv);
    MatMatMult(G_pi_N_pi_inv, N_rt, reuse, PETSC_DEFAULT, &G_pi_N_pi_inv_N_rt);
    MatAYPX(G_pi_N_pi_inv_N_rt, -1.0, G_rt, DIFFERENT_NONZERO_PATTERN);

    MatMatMult(D_rho, M_u_inv, reuse, PETSC_DEFAULT, &D_rho_M_u_inv);
    MatMatMult(D_rt,  M_u_inv, reuse, PETSC_DEFAULT, &D_rt_M_u_inv );

    MatMatMult(D_rho_M_u_inv, G_pi_N_pi_inv_N_rt, reuse, PETSC_DEFAULT, &L_rho_rt);
    MatMatMult(D_rt_M_u_inv,  G_pi_N_pi_inv_N_rt, reuse, PETSC_DEFAULT, &L_rt_rt );

    // add to the global 3D matrice
    schur->AddFromVertMat(ei, L_rho_rt, schur->L_rho);
    schur->AddFromVertMat(ei, L_rt_rt,  schur->L_rt );
    if(!itt) schur->AddFromVertMat(ei, Q_rt_rho_M_rho_inv, schur->Q);

    // update the residuals
    MatMult(G_pi_N_pi_inv, F_pi, _tmpA1);
    VecAXPY(F_u, -1.0, _tmpA1);

    MatMult(D_rt_M_u_inv, F_u, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);

    MatMult(D_rho_M_u_inv, F_u, _tmpB1);
    VecAXPY(Du, +1.0, _tmpB1);
}

void VertSolve::update_delta_u_3(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                 Vec F_u, Vec F_rho, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi) 
{
    MatReuse reuse = MAT_REUSE_MATRIX;

    // assemble the operators for the coupled system
#ifdef RAYLEIGH
    vo->AssembleLinearWithRayleighInv(ex, ey, 0.5*dt*RAYLEIGH, M_u_inv);
#else
    vo->AssembleLinearInv(ex, ey, M_u_inv);
#endif
    vo->AssembleConst(ex, ey, M_rt);
    vo->AssembleConstInv(ex, ey, M_rho_inv);
#ifdef NEW_EOS
    vo->AssembleN_PiInv(ex, ey, rt, pi, N_pi_inv, true);
    vo->AssembleN_RT(ex, ey, rt, pi, N_rt);
#else
    vo->Assemble_EOS_BlockInv(ex, ey, pi, NULL, N_pi_inv);
    vo->AssembleConst(ex, ey, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &N_rt);
    MatScale(N_rt, -1.0*RD/CV);
#endif
    vo->AssembleConst(ex, ey, vo->VB);
    MatMult(vo->VB, pi, _tmpB1);
    MatMult(vo->V01, _tmpB1, _tmpA1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
    vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
    MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(vo->VAB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &G_rt);
    MatScale(G_rt, 0.5*dt);

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &G_pi);
    MatScale(G_pi, 0.5*dt);

    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &D_rho);
    MatScale(D_rho, 0.5*dt);

    MatMatMult(G_pi, N_pi_inv, reuse, PETSC_DEFAULT, &G_pi_N_pi_inv);
    MatMatMult(G_pi_N_pi_inv, N_rt, reuse, PETSC_DEFAULT, &G_pi_N_pi_inv_N_rt);
    MatAYPX(G_pi_N_pi_inv_N_rt, -1.0, G_rt, DIFFERENT_NONZERO_PATTERN);

    // update the velocity residual
    MatMult(G_pi_N_pi_inv_N_rt, d_rt, _tmpA1);
    VecAXPY(F_u, +1.0, _tmpA1);
    VecScale(F_u, -1.0);
    MatMult(M_u_inv, F_u, d_u);

    // update the pressure residual
    MatMult(N_rt, d_rt, _tmpB1);
    VecAXPY(F_pi, +1.0, _tmpB1);
    VecScale(F_pi, -1.0);
    MatMult(N_pi_inv, F_pi, d_pi);

    // update the density residual
    MatMult(D_rho, d_u, _tmpB1);
    VecAXPY(F_rho, 1.0, _tmpB1);
}
