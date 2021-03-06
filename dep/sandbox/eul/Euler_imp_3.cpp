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
#include "Euler_imp_3.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
#define MAX_IT 100
#define VERT_TOL 1.0e-8
#define HORIZ_TOL 1.0e-12
#define RAYLEIGH 0.2
#define EXNER_EOS

using namespace std;

Euler::Euler(Topo* _topo, Geom* _geom, double _dt) {
    int ii, elOrd2;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    do_visc = true;
    del2 = viscosity();
    step = 0;
    firstStep = true;

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

    // rotational operator
    R = new RotMat(topo, geom, node, edge);

    // mass flux operator
    F = new Uhmat(topo, geom, node, edge);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    // additional vorticity operator
    M1t = new Ut_mat(topo, geom, node, edge);
    Rh = new UtQWmat(topo, geom, node, edge);

    // potential temperature projection operator
    T = new Whmat(topo, geom, edge);

    // equation or state right hand side vector
    eos = new EoSvec(topo, geom, edge);

    // derivative of the equation of state (for the Theta preconditioner operator)
    eos_mat = new EoSmat(topo, geom, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

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

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp1);
    KSPSetOperators(ksp1, M1->M, M1->M);
    KSPSetTolerances(ksp1, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1, KSPGMRES);
    KSPGetPC(ksp1, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp1, "ksp1_");
    KSPSetFromOptions(ksp1);

    // initialize the 2 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPGetPC(ksp2, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, elOrd2, NULL);
    KSPSetOptionsPrefix(ksp2, "ksp2_");
    KSPSetFromOptions(ksp2);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &_Phi_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*elOrd2, &_theta_h);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_tmpA1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_tmpA2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &_tmpB1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &_tmpB2);

    PCz = new Mat[topo->nElsX*topo->nElsX];
    PCx = new Mat[geom->nk];

    _M1invM1 = NULL;
    _PCz = NULL;
    pcz_DTV1 = NULL;
    pct_DTV1 = NULL;
    _DTM2 = NULL;
    _V0_invV0_rt = NULL;

    thetaBar = NULL;
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double Euler::viscosity() {
    double ae = 4.0*M_PI*RAD_EARTH*RAD_EARTH;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void Euler::coriolis() {
    int ii, kk;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fxl, fxg, PtQfxg;

    // initialise the coriolis vector (local and global)
    fg = new Vec[geom->nk];
    fl = new Vec[geom->nk];

    // evaluate the coriolis term at nodes
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < topo->n0; ii++) {
        fArray[ii] = 2.0*OMEGA*sin(geom->s[ii][1]);
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
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg[kk]);
        m0->assemble(kk, 1.0);
        VecPointwiseDivide(fg[kk], PtQfxg, m0->vg);
        VecZeroEntries(fl[kk]);
        VecScatterBegin(topo->gtol_0, fg[kk], fl[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_0, fg[kk], fl[kk], INSERT_VALUES, SCATTER_FORWARD);
    }
    
    delete PtQ;
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

void Euler::initGZ() {
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

Euler::~Euler() {
    int ii;

    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);

    VecDestroy(&_Phi_z);
    VecDestroy(&_theta_h);
    VecDestroy(&_tmpA1);
    VecDestroy(&_tmpA2);
    VecDestroy(&_tmpB1);
    VecDestroy(&_tmpB2);

    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&fg[ii]);
        VecDestroy(&fl[ii]);
        MatDestroy(&PCx[ii]);
    }
    delete[] fg;
    delete[] fl;
    delete[] PCx;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
        VecDestroy(&zv[ii]);
        MatDestroy(&PCz[ii]);
    }
    delete[] gv;
    delete[] zv;
    delete[] PCz;

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;
    delete Rh;
    delete T;
    delete eos;
    delete eos_mat;

    delete edge;
    delete node;
    delete quad;

    delete vo;

    delete thetaBar;

    MatDestroy(&pcz_DTV1);
    MatDestroy(&pcz_GRAD);
    MatDestroy(&pcz_DIV);
    MatDestroy(&pcz_V0_invDTV1);
    MatDestroy(&pcz_V0_invV0_rt);
    MatDestroy(&pcz_DV0_invV0_rt);
    MatDestroy(&pcz_V1_PiDV0_invV0_rt);
    MatDestroy(&pcz_BOUS);

    MatDestroy(&pct_DTV1);
    MatDestroy(&pct_V0_invDTV1);
    MatDestroy(&pct_V0_thetaV0_invDTV1);
    MatDestroy(&pct_V0_invV0_thetaV0_invDTV1);
    MatDestroy(&pct_DV0_invV0_thetaV0_invDTV1);
    MatDestroy(&pct_V10DT);

    MatDestroy(&_V0_invV0_rt);

    MatDestroy(&_PCz);
    MatDestroy(&_Muu);
    MatDestroy(&_Muh);
    MatDestroy(&_Mhu);
    MatDestroy(&_Mhh);

    MatDestroy(&_DTM2);
    MatDestroy(&_M1invDTM2);
    MatDestroy(&_M1thetaM1invDTM2);
    MatDestroy(&_M1invM1thetaM1invDTM2);
    MatDestroy(&_DM1invM1thetaM1invDTM2);
    MatDestroy(&_KDT);
}

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void Euler::grad(bool assemble, Vec phi, Vec* u, int lev) {
    Vec Mphi, dMphi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, u);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mphi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMphi);

    if(assemble) {
        M1->assemble(lev, SCALE, true);
        M2->assemble(lev, SCALE, true);
    }

    MatMult(M2->M, phi, Mphi);
    MatMult(EtoF->E12, Mphi, dMphi);
    KSPSolve(ksp1, dMphi, *u);

    VecDestroy(&Mphi);
    VecDestroy(&dMphi);
}

/*
Take the weak form curl of a 1 form vector field as a 1 form vector field
*/
void Euler::curl(bool assemble, Vec u, Vec* w, int lev, bool add_f) {
    Vec Mu, dMu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &dMu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu);

    if(assemble) {
        m0->assemble(lev, SCALE);
        M1->assemble(lev, SCALE, true);
    }
    MatMult(M1->M, u, Mu);
    MatMult(NtoE->E01, Mu, dMu);
    VecPointwiseDivide(*w, dMu, m0->vg);

    // add the coliolis term
    if(add_f) {
        VecAYPX(*w, 1.0, fg[lev]);
    }
    VecDestroy(&Mu);
    VecDestroy(&dMu);
}

void Euler::laplacian(bool assemble, Vec ui, Vec* ddu, int lev) {
    Vec Du, Cu, RCu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &RCu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Du);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, ui, Du);

    // grad (weak form)
    grad(assemble, Du, ddu, lev);

    /*** rotational component ***/
    // curl (weak form)
    curl(assemble, ui, &Cu, lev, false);

    // rot (strong form)
    MatMult(NtoE->E10, Cu, RCu);

    // add rotational and divergent components
    VecAXPY(*ddu, +1.0, RCu);
    VecScale(*ddu, del2);

    VecDestroy(&Cu);
    VecDestroy(&RCu);
    VecDestroy(&Du);
}

void Euler::dump(Vec* velx, L2Vecs* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* theta, int num) {
    char fieldname[100];
    Vec wi;

    if(!rank) cout << "dumping output for step: " << num << endl;

    theta->UpdateGlobal();
    for(int ii = 0; ii < geom->nk+1; ii++) {
        sprintf(fieldname, "theta");
        geom->write2(theta->vh[ii], fieldname, num, ii, false);
    }

    for(int ii = 0; ii < geom->nk; ii++) {
        if(velx) curl(true, velx[ii], &wi, ii, false);

        if(velx) sprintf(fieldname, "vorticity");
        if(velx) geom->write0(wi, fieldname, num, ii);
        if(velx) sprintf(fieldname, "velocity_h");
        if(velx) geom->write1(velx[ii], fieldname, num, ii);
        sprintf(fieldname, "density");
        geom->write2(rho->vh[ii], fieldname, num, ii, true);
        sprintf(fieldname, "rhoTheta");
        geom->write2(rt->vh[ii], fieldname, num, ii, true);
        sprintf(fieldname, "exner");
        geom->write2(exner->vh[ii], fieldname, num, ii, true);

        if(velx) VecDestroy(&wi);
    }
    sprintf(fieldname, "velocity_z");
    for(int ii = 0; ii < geom->nk-1; ii++) {
        geom->write2(velz->vh[ii], fieldname, num, ii, false);
    }
}

#define SOLVE_X
#define SOLVE_Z

void Euler::solve(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, bool save) {
    int done_l = 0, done;
    int elOrd2 = topo->elOrd*topo->elOrd;
    double norm_max_x, norm_max_z, norm_u, norm_du, norm_max_dz;
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* exner = new L2Vecs(geom->nk, topo, geom);
    Vec* velx_j = new Vec[geom->nk];
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* bous = new L2Vecs(geom->nk, topo, geom);
    Vec du, fu, dw, fw, htmp;
    Vec* _F_x = new Vec[geom->nk];
    Vec* _G_x = new Vec[geom->nk];
    Vec* _F_z = new Vec[topo->nElsX*topo->nElsX];
    Vec* _G_z = new Vec[topo->nElsX*topo->nElsX];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    PC pc;
    KSP ksp_x;
    KSP ksp_z;

    velz_i->UpdateLocal();
    velz_i->HorizToVert();
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rt_i->UpdateLocal();
    rt_i->HorizToVert();

    // diagnose the potential temperature
    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();
    for(int ii = 0; ii < geom->nk; ii++) {
        diagnose_Pi(ii, rt_i->vl[ii], rt_i->vl[ii], exner->vh[ii]);
    }
    exner->UpdateLocal();
    exner->HorizToVert();

    // update the next time level
    velz_j->CopyFromHoriz(velz_i->vh);
    rho_j->CopyFromHoriz(rho_i->vh);
    rt_j->CopyFromHoriz(rt_i->vh);
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();

    // create the preconditioner operators
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &dw);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &fw);
    initBousFac(theta_h, bous->vz);
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_F_z[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_G_z[ii]);

        if(firstStep) PCz[ii] = NULL;
        assemble_precon_z(ii%topo->nElsX, ii/topo->nElsX, theta_h->vz[ii], rho_i->vz[ii], rt_i->vz[ii], rt_j->vz[ii], exner->vz[ii], &PCz[ii], bous->vz[ii]);
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    for(int ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F_x[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G_x[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[ii]);
        VecCopy(velx_i[ii], velx_j[ii]);

        if(firstStep) PCx[ii] = NULL;
        assemble_precon_x(ii, theta_i->vl, rt_i->vl[ii], rt_i->vl[ii], exner->vl[ii], &PCx[ii]);
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);

    diagHorizVort(velx_i, dudz_i);
    for(int ii = 0; ii < geom->nk; ii++) {
        VecCopy(dudz_i[ii], dudz_j[ii]);
    }

    do {
        for(int ii = 0; ii < geom->nk+1; ii++) {
            VecAXPY(theta_h->vl[ii], 1.0, theta_i->vl[ii]);
            VecScale(theta_h->vl[ii], 0.5);
        }
        theta_h->HorizToVert();

        // update the horizontal dynamics for this iteration
        norm_max_x = -1.0;
#ifdef SOLVE_X
        for(int ii = 0; ii < geom->nk; ii++) {
            if(!rank) cout << rank << ":\tassembling (horizontal) residual vector: " << ii;
            assemble_residual_x(ii, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner->vh[ii],
                                velx_i[ii], velx_j[ii], rho_i->vh[ii], rho_j->vh[ii], rt_i->vh[ii], rt_j->vh[ii], 
                                fu, _F_x[ii], _G_x[ii]);
            VecScale(fu, -1.0);

            assemble_precon_x(ii, theta_h->vl, rt_i->vl[ii], rt_j->vl[ii], exner->vl[ii], &PCx[ii]);

            KSPCreate(MPI_COMM_WORLD, &ksp_x);
            KSPSetOperators(ksp_x, PCx[ii], PCx[ii]);
            KSPSetTolerances(ksp_x, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
            KSPSetType(ksp_x, KSPGMRES);
            KSPGetPC(ksp_x, &pc);
            PCSetType(pc, PCBJACOBI);
            PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
            KSPSetOptionsPrefix(ksp_x, "ksp_x_");
            KSPSetFromOptions(ksp_x);
            KSPSolve(ksp_x, fu, du);
            KSPDestroy(&ksp_x);
            VecAXPY(velx_j[ii], 1.0, du);

            VecNorm(velx_j[ii], NORM_2, &norm_u);
            VecNorm(du, NORM_2, &norm_du);
            if(!rank) cout << "\t|dx|: " << norm_du << "\t|x|: " << norm_u << "\t|dx|/|x|: " << norm_du/norm_u << endl;
            if(norm_max_x < norm_du/norm_u) norm_max_x = norm_du/norm_u;
        }
#endif

#ifdef SOLVE_Z
        // update the vertical dynamics for this iteration
        if(!rank) cout << rank << ":\tassembling (vertical) residual vectors" << endl;
        norm_max_z = -1.0;
        initBousFac(theta_h, bous->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            assemble_residual_z(ii%topo->nElsX, ii/topo->nElsX, theta_h->vz[ii], exner->vz[ii],
                                velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], rt_i->vz[ii], rt_j->vz[ii], 
                                fw, _F_z[ii], _G_z[ii]);
            VecScale(fw, -1.0);

            assemble_precon_z(ii%topo->nElsX, ii/topo->nElsX, theta_h->vz[ii], rho_j->vz[ii], rt_i->vz[ii], rt_j->vz[ii], exner->vz[ii], &PCz[ii], bous->vz[ii]);

            KSPCreate(MPI_COMM_SELF, &ksp_z);
            KSPSetOperators(ksp_z, PCz[ii], PCz[ii]);
            KSPGetPC(ksp_z, &pc);
            PCSetType(pc, PCLU);
            KSPSetOptionsPrefix(ksp_z, "ksp_z_");
            KSPSetFromOptions(ksp_z);
            KSPSolve(ksp_z, fw, dw);
            KSPDestroy(&ksp_z);
            VecAXPY(velz_j->vz[ii], 1.0, dw);

            VecNorm(velz_j->vz[ii], NORM_2, &norm_u);
            VecNorm(dw, NORM_2, &norm_du);
            if(norm_max_dz/norm_max_z < norm_du/norm_u) { 
                norm_max_z = norm_u;
                norm_max_dz = norm_du;
            }
        }
        velz_j->VertToHoriz();
        velz_j->UpdateGlobal();
#endif

        for(int ii = 0; ii < geom->nk; ii++) {
            VecCopy(rho_i->vh[ii], rho_j->vh[ii]);
            VecCopy(rt_i->vh[ii], rt_j->vh[ii]);
        }
        rho_j->UpdateLocal();
        rho_j->HorizToVert();
        rt_j->UpdateLocal();
        rt_j->HorizToVert();

#ifdef SOLVE_X
        for(int ii = 0; ii < geom->nk; ii++) {
            MatMult(EtoF->E21, _F_x[ii], htmp);
            VecAXPY(rho_j->vh[ii], -dt, htmp);

            MatMult(EtoF->E21, _G_x[ii], htmp);
            VecAXPY(rt_j->vh[ii], -dt, htmp);
        }
        rho_j->UpdateLocal();
        rho_j->HorizToVert();
        rt_j->UpdateLocal();
        rt_j->HorizToVert();
#endif

#ifdef SOLVE_Z
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            MatMult(vo->V10, _F_z[ii], _tmpB1);
            VecAXPY(rho_j->vz[ii], -dt, _tmpB1);

            MatMult(vo->V10, _G_z[ii], _tmpB1);
            VecAXPY(rt_j->vz[ii], -dt, _tmpB1);
        }
        rho_j->VertToHoriz();
        rho_j->UpdateGlobal();
        rt_j->VertToHoriz();
        rt_j->UpdateGlobal();
#endif

        diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        theta_h->VertToHoriz();
        for(int ii = 0; ii < geom->nk; ii++) {
            diagnose_Pi(ii, rt_i->vl[ii], rt_j->vl[ii], exner->vh[ii]);
        }
        exner->UpdateLocal();
        exner->HorizToVert();
        diagHorizVort(velx_j, dudz_j);

        if(!rank) cout << "|dz|: " << norm_max_dz << "\t|z|: " << norm_max_z << "\t|dz|/|z|: " << norm_max_dz/norm_max_z << endl;

        if(norm_max_x < 1.0e-10 && norm_max_dz/norm_max_z < 1.0e-8) done_l = 1;
        MPI_Allreduce(&done_l, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    } while(!done);

    if(!rank) cout << "done.\n";

    // update the solutions
    for(int ii = 0; ii < geom->nk; ii++) {
        VecCopy(velx_j[ii], velx_i[ii]);
    }
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);

    // write output
    if(save) {
        dump(velx_j, velz_j, rho_j, rt_j, exner, theta_h, step++);
    }
    firstStep = false;

    VecDestroy(&dw);
    VecDestroy(&fw);
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&_F_z[ii]);
        VecDestroy(&_G_z[ii]);
    }
    delete[] _F_z;
    delete[] _G_z;
    VecDestroy(&du);
    VecDestroy(&fu);
    for(int ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&velx_j[ii]);
        VecDestroy(&_F_x[ii]);
        VecDestroy(&_G_x[ii]);
        VecDestroy(&dudz_i[ii]);
        VecDestroy(&dudz_j[ii]);
    }
    VecDestroy(&htmp);
    delete[] velx_j;
    delete[] _F_x;
    delete[] _G_x;
    delete[] dudz_i;
    delete[] dudz_j;
    delete theta_i;
    delete theta_h;
    delete exner;
    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete bous;
}

void Euler::solve_strang(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, bool save) {
    bool done = false;
    int elOrd2 = topo->elOrd*topo->elOrd;
    double norm_max_x, norm_max_z, norm_u, norm_du, norm_max_dz;
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* exner = new L2Vecs(geom->nk, topo, geom);
    Vec* velx_j = new Vec[geom->nk];
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* bous = new L2Vecs(geom->nk, topo, geom);
    Vec du, fu, dw, fw, htmp;
    Vec* _F_x = new Vec[geom->nk];
    Vec* _G_x = new Vec[geom->nk];
    Vec* _F_z = new Vec[topo->nElsX*topo->nElsX];
    Vec* _G_z = new Vec[topo->nElsX*topo->nElsX];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    PC pc;
    KSP ksp_x;
    KSP ksp_z;

    velz_i->UpdateLocal();
    velz_i->HorizToVert();
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rt_i->UpdateLocal();
    rt_i->HorizToVert();

    // diagnose the potential temperature
    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();
    for(int ii = 0; ii < geom->nk; ii++) {
        diagnose_Pi(ii, rt_i->vl[ii], rt_i->vl[ii], exner->vh[ii]);
    }
    exner->UpdateLocal();
    exner->HorizToVert();

    // update the next time level
    velz_j->CopyFromHoriz(velz_i->vh);
    rho_j->CopyFromHoriz(rho_i->vh);
    rt_j->CopyFromHoriz(rt_i->vh);
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();

    velz_j->UpdateLocal();
    rho_j->UpdateLocal();
    rt_j->UpdateLocal();
    velz_j->HorizToVert();
    rho_j->HorizToVert();
    rt_j->HorizToVert();

    // create the preconditioner operators
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &dw);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &fw);
    initBousFac(theta_h, bous->vz);
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_F_z[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_G_z[ii]);

        if(firstStep) PCz[ii] = NULL;
        assemble_precon_z(ii%topo->nElsX, ii/topo->nElsX, theta_h->vz[ii], rho_i->vz[ii], rt_i->vz[ii], rt_j->vz[ii], exner->vz[ii], &PCz[ii], bous->vz[ii]);
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    for(int ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F_x[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G_x[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[ii]);
        VecCopy(velx_i[ii], velx_j[ii]);

        if(firstStep) PCx[ii] = NULL; 
        assemble_precon_x(ii, theta_i->vl, rt_i->vl[ii], rt_i->vl[ii], exner->vl[ii], &PCx[ii]);
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);

    diagHorizVort(velx_i, dudz_i);
    for(int ii = 0; ii < geom->nk; ii++) {
        VecCopy(dudz_i[ii], dudz_j[ii]);
    }

    do {
        for(int ii = 0; ii < geom->nk+1; ii++) {
            VecAXPY(theta_h->vl[ii], 1.0, theta_i->vl[ii]);
            VecScale(theta_h->vl[ii], 0.5);
        }
        theta_h->HorizToVert();

        // update the vertical dynamics for this iteration
        if(!rank) cout << rank << ":\tassembling (vertical) residual vectors" << endl;
        norm_max_dz = 0.0;
        norm_max_z = 1.0;
        initBousFac(theta_h, bous->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            assemble_residual_z(ii%topo->nElsX, ii/topo->nElsX, theta_h->vz[ii], exner->vz[ii],
                                velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], rt_i->vz[ii], rt_j->vz[ii], 
                                fw, _F_z[ii], _G_z[ii]);
            VecScale(fw, -1.0);

            assemble_precon_z(ii%topo->nElsX, ii/topo->nElsX, theta_h->vz[ii], rho_i->vz[ii], rt_i->vz[ii], rt_j->vz[ii], exner->vz[ii], &PCz[ii], bous->vz[ii]);
            KSPCreate(MPI_COMM_SELF, &ksp_z);
            KSPSetOperators(ksp_z, PCz[ii], PCz[ii]);
            KSPGetPC(ksp_z, &pc);
            PCSetType(pc, PCLU);
            KSPSetOptionsPrefix(ksp_z, "ksp_z_");
            KSPSetFromOptions(ksp_z);
            KSPSolve(ksp_z, fw, dw);
            KSPDestroy(&ksp_z);
            VecAXPY(velz_j->vz[ii], 1.0, dw);

            VecNorm(velz_j->vz[ii], NORM_2, &norm_u);
            VecNorm(dw, NORM_2, &norm_du);
            if(norm_max_dz/norm_max_z < norm_du/norm_u) { 
                norm_max_z = norm_u;
                norm_max_dz = norm_du;
            }
        }
        velz_j->VertToHoriz();
        velz_j->UpdateGlobal();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecCopy(rho_i->vz[ii], rho_j->vz[ii]);
            MatMult(vo->V10, _F_z[ii], _tmpB1);
            VecAXPY(rho_j->vz[ii], -dt, _tmpB1);

            VecCopy(rt_i->vz[ii], rt_j->vz[ii]);
            MatMult(vo->V10, _G_z[ii], _tmpB1);
            VecAXPY(rt_j->vz[ii], -dt, _tmpB1);
        }
        rho_j->VertToHoriz();
        rho_j->UpdateGlobal();
        rt_j->VertToHoriz();
        rt_j->UpdateGlobal();

        diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        theta_h->VertToHoriz();
        for(int ii = 0; ii < geom->nk; ii++) {
            diagnose_Pi(ii, rt_i->vl[ii], rt_j->vl[ii], exner->vh[ii]);
        }
        exner->UpdateLocal();
        exner->HorizToVert();
        diagHorizVort(velx_j, dudz_j);

        if(!rank) cout << "|dz|: " << norm_max_dz << "\t|z|: " << norm_max_z << "\t|dz|/|z|: " << norm_max_dz/norm_max_z << endl;
        if(norm_max_dz/norm_max_z < 1.0e-8) done = true;
    } while(!done);

    do {
        // update the horizontal dynamics for this iteration
        norm_max_x = 0.0;
        for(int ii = 0; ii < geom->nk; ii++) {
            if(!rank) cout << rank << ":\tassembling (horizontal) residual vector: " << ii;
            assemble_residual_x(ii, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner->vh[ii],
                                velx_i[ii], velx_j[ii], rho_i->vh[ii], rho_j->vh[ii], rt_i->vh[ii], rt_j->vh[ii], 
                                fu, _F_x[ii], _G_x[ii]);
            VecScale(fu, -1.0);

            KSPCreate(MPI_COMM_WORLD, &ksp_x);
            KSPSetOperators(ksp_x, PCx[ii], PCx[ii]);
            KSPSetTolerances(ksp_x, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
            KSPSetType(ksp_x, KSPGMRES);
            KSPGetPC(ksp_x, &pc);
            PCSetType(pc, PCBJACOBI);
            PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
            KSPSetOptionsPrefix(ksp_x, "ksp_x_");
            KSPSetFromOptions(ksp_x);
            KSPSolve(ksp_x, fu, du);
            KSPDestroy(&ksp_x);
            VecAXPY(velx_j[ii], 1.0, du);

            VecNorm(velx_j[ii], NORM_2, &norm_u);
            VecNorm(du, NORM_2, &norm_du);
            if(!rank) cout << "\t|dx|: " << norm_du << "\t|x|: " << norm_u << "\t|dx|/|x|: " << norm_du/norm_u << endl;
            if(norm_max_x < norm_du/norm_u) norm_max_x = norm_du/norm_u;
        }

        for(int ii = 0; ii < geom->nk; ii++) {
            VecCopy(rho_i->vh[ii], rho_j->vh[ii]);
            MatMult(EtoF->E21, _F_x[ii], htmp);
            VecAXPY(rho_j->vh[ii], -dt, htmp);

            VecCopy(rt_i->vh[ii], rt_j->vh[ii]);
            MatMult(EtoF->E21, _G_x[ii], htmp);
            VecAXPY(rt_j->vh[ii], -dt, htmp);
        }
        rho_j->UpdateLocal();
        rho_j->HorizToVert();
        rt_j->UpdateLocal();
        rt_j->HorizToVert();

        diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        theta_h->VertToHoriz();
        for(int ii = 0; ii < geom->nk; ii++) {
            diagnose_Pi(ii, rt_i->vl[ii], rt_j->vl[ii], exner->vh[ii]);
        }
        exner->UpdateLocal();
        exner->HorizToVert();
        diagHorizVort(velx_j, dudz_j);

        if(norm_max_x < 1.0e-12) done = true;
    } while(!done);

    // update the solutions
    for(int ii = 0; ii < geom->nk; ii++) {
        VecCopy(velx_j[ii], velx_i[ii]);
    }
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);

    // write output
    if(save) {
        dump(velx_j, velz_j, rho_j, rt_j, exner, theta_h, step++);
    }
    firstStep = false;

    VecDestroy(&dw);
    VecDestroy(&fw);
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&_F_z[ii]);
        VecDestroy(&_G_z[ii]);
    }
    delete[] _F_z;
    delete[] _G_z;
    VecDestroy(&du);
    VecDestroy(&fu);
    for(int ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&velx_j[ii]);
        VecDestroy(&_F_x[ii]);
        VecDestroy(&_G_x[ii]);
        VecDestroy(&dudz_i[ii]);
        VecDestroy(&dudz_j[ii]);
    }
    VecDestroy(&htmp);
    delete[] velx_j;
    delete[] _F_x;
    delete[] _G_x;
    delete[] dudz_i;
    delete[] dudz_j;
    delete theta_i;
    delete theta_h;
    delete exner;
    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete bous;
}

void Euler::solve_vert(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, bool save) {
    int done = 0, done_l, itt = 0;
    int elOrd2 = topo->elOrd*topo->elOrd;
    double norm_max_z, norm_u, norm_du, norm_max_dz;
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* exner = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* bous = new L2Vecs(geom->nk, topo, geom);
    Vec dw, fw;
    Vec* _F_z = new Vec[topo->nElsX*topo->nElsX];
    Vec* _G_z = new Vec[topo->nElsX*topo->nElsX];
    PC pc;
    KSP ksp_z;
    int conv[9999];

    velz_i->UpdateLocal();
    velz_i->HorizToVert();
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rt_i->UpdateLocal();
    rt_i->HorizToVert();

    // diagnose the potential temperature
    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();
    for(int ii = 0; ii < geom->nk; ii++) {
        diagnose_Pi(ii, rt_i->vl[ii], rt_i->vl[ii], exner->vh[ii]);
    }
    exner->UpdateLocal();
    exner->HorizToVert();

    // update the next time level
    velz_j->CopyFromHoriz(velz_i->vh);
    rho_j->CopyFromHoriz(rho_i->vh);
    rt_j->CopyFromHoriz(rt_i->vh);
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();

    // create the preconditioner operators
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &dw);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &fw);
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_F_z[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_G_z[ii]);
    }

    for(int ii = 0; ii < 9999; ii++) conv[ii] = 0;

    do {
        for(int ii = 0; ii < geom->nk+1; ii++) {
            VecAXPY(theta_h->vl[ii], 1.0, theta_i->vl[ii]);
            VecScale(theta_h->vl[ii], 0.5);
        }
        theta_h->HorizToVert();

        // update the vertical dynamics for this iteration
        if(!rank) cout << rank << ":\tassembling (vertical) residual vectors" << endl;
        norm_max_z = -1.0;
        initBousFac(theta_h, bous->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            if(conv[ii]) continue;

            assemble_residual_z(ii%topo->nElsX, ii/topo->nElsX, theta_h->vz[ii], exner->vz[ii],
                                velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], rt_i->vz[ii], rt_j->vz[ii], 
                                fw, _F_z[ii], _G_z[ii]);
            VecScale(fw, -1.0);

            if(firstStep) PCz[ii] = NULL;
            assemble_precon_z(ii%topo->nElsX, ii/topo->nElsX, theta_h->vz[ii], rho_i->vz[ii], rt_i->vz[ii], rt_j->vz[ii], exner->vz[ii], &PCz[ii], bous->vz[ii]);

            KSPCreate(MPI_COMM_SELF, &ksp_z);
            KSPSetOperators(ksp_z, PCz[ii], PCz[ii]);
            KSPGetPC(ksp_z, &pc);
            PCSetType(pc, PCLU);
            KSPSetOptionsPrefix(ksp_z, "ksp_z_");
            KSPSetFromOptions(ksp_z);
            KSPSolve(ksp_z, fw, dw);
            KSPDestroy(&ksp_z);
            VecAXPY(velz_j->vz[ii], 1.0, dw);

            VecNorm(velz_j->vz[ii], NORM_2, &norm_u);
            VecNorm(dw, NORM_2, &norm_du);
            if(norm_du/norm_u < 1.0e-6) conv[ii] = 1;
            if(norm_max_dz/norm_max_z < norm_du/norm_u) { 
                norm_max_z = norm_u;
                norm_max_dz = norm_du;
            }
        }
        velz_j->VertToHoriz();
        velz_j->UpdateGlobal();

        for(int ii = 0; ii < geom->nk; ii++) {
            VecCopy(rho_i->vh[ii], rho_j->vh[ii]);
            VecCopy(rt_i->vh[ii], rt_j->vh[ii]);
        }
        rho_j->UpdateLocal();
        rho_j->HorizToVert();
        rt_j->UpdateLocal();
        rt_j->HorizToVert();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            if(conv[ii]) continue;

            MatMult(vo->V10, _F_z[ii], _tmpB1);
            VecAXPY(rho_j->vz[ii], -dt, _tmpB1);

            MatMult(vo->V10, _G_z[ii], _tmpB1);
            VecAXPY(rt_j->vz[ii], -dt, _tmpB1);
        }
        rho_j->VertToHoriz();
        rho_j->UpdateGlobal();
        rt_j->VertToHoriz();
        rt_j->UpdateGlobal();

        diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        theta_h->VertToHoriz();
        for(int ii = 0; ii < geom->nk; ii++) {
            diagnose_Pi(ii, rt_i->vl[ii], rt_j->vl[ii], exner->vh[ii]);
        }
        exner->UpdateLocal();
        exner->HorizToVert();

        if(!rank) cout << itt++ << "\t|dz|: " << norm_max_dz << "\t|z|: " << norm_max_z << "\t|dz|/|z|: " << norm_max_dz/norm_max_z << endl;

        //if(norm_max_dz/norm_max_z < 1.0e-8) done = true;
        done_l = 1;
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) if(!conv[ii]) done_l = 0;
        MPI_Allreduce(&done_l, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    } while(!done);

    if(!rank) cout << "done.\n";

    // update the solutions
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);

    // write output
    if(save) {
        step++;
        dump(NULL, velz_j, rho_j, rt_j, exner, theta_h, step);
    }

    VecDestroy(&dw);
    VecDestroy(&fw);
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&_F_z[ii]);
        VecDestroy(&_G_z[ii]);
    }
    delete[] _F_z;
    delete[] _G_z;
    delete theta_i;
    delete theta_h;
    delete exner;
    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete bous;
}

void Euler::solve_vert_exner(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int ex, ey, elOrd2, itt = 0;
    double norm_x, norm_dx, max_norm_w, max_norm_exner, max_norm_rho, max_norm_rt;
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
    L2Vecs* bous = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_eos = new L2Vecs(geom->nk, topo, geom);
    Vec fw, dw, de, Phi, F_exner;
    PC pc;
    Mat PC_exner = NULL;
    Mat PC_div = NULL;
    Mat PC_grad = NULL;
    KSP kspColA;
    KSP kspColB;
    KSP ksp_exner = NULL;

    elOrd2 = topo->elOrd*topo->elOrd;
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &fw);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &dw);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &de);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &Phi);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_exner);

    KSPCreate(MPI_COMM_SELF, &kspColA);
    KSPSetOperators(kspColA, vo->VA, vo->VA);
    KSPGetPC(kspColA, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColA, "kspColA_");
    KSPSetFromOptions(kspColA);

    KSPCreate(MPI_COMM_SELF, &kspColB);
    KSPSetOperators(kspColB, vo->VB, vo->VB);
    KSPGetPC(kspColB, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColB, "kspColB_");
    KSPSetFromOptions(kspColB);

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

    for(int ii = 0; ii < geom->nk; ii++) {
        diagnose_Pi(ii, rt_i->vl[ii], rt_i->vl[ii], exner_h->vh[ii]);
    }
    exner_h->UpdateLocal();
    exner_h->HorizToVert();

    do {
        max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        // update the exner pressure
        initBousFac(theta_h, bous->vz);

        //for(int ii = 0; ii < geom->nk; ii++) {
        //    diagnose_Pi(ii, rt_j->vl[ii], rt_j->vl[ii], rt_eos->vh[ii]);
        //}
        //rt_eos->UpdateLocal();
        //rt_eos->HorizToVert();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // implicit exner solve
            exner_residual_z(ex, ey, rt_j->vz[ii], dG_z->vz[ii], exner_i->vz[ii], exner_j->vz[ii], F_exner);
            //VecCopy(exner_j->vz[ii], F_exner);
            //VecAXPY(F_exner, -1.0, rt_eos->vz[ii]);

            VecScale(F_exner, -1.0);

            exner_precon_z(ex, ey, dG_z->vz[ii], exner_j->vz[ii], rt_j->vz[ii], bous->vz[ii], theta_h->vz[ii], &PC_exner, &PC_div, &PC_grad);

            assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], fw, F_z->vz[ii], G_z->vz[ii]);
            VecScale(fw, -1.0);

            MatMult(PC_div, fw, _tmpB1);
            VecAYPX(_tmpB1, -dt*RD/CV, F_exner);

            KSPCreate(MPI_COMM_SELF, &ksp_exner);
            KSPSetOperators(ksp_exner, PC_exner, PC_exner);
            KSPGetPC(ksp_exner, &pc);
            PCSetType(pc, PCLU);
            KSPSetOptionsPrefix(ksp_exner, "ksp_exner_");
            KSPSetFromOptions(ksp_exner);
            KSPSolve(ksp_exner, _tmpB1, de);
            KSPDestroy(&ksp_exner);
            VecAXPY(exner_j->vz[ii], 1.0, de);

            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

/*
            vo->AssembleConstWithRho(ex, ey, rt_j->vz[ii], vo->VB);
            MatMult(vo->VB, exner_i->vz[ii], _tmpB1); // rhs

            VecCopy(rt_j->vz[ii], _tmpB2);
            VecAXPY(_tmpB2, dt*RD/CV, dG_z->vz[ii]);
            vo->AssembleConstWithRho(ex, ey, _tmpB2, vo->VB);
            //KSPSolve(kspColB, _tmpB1, exner_j->vz[ii]);
            MatMult(vo->VB, exner_j->vz[ii], _tmpB2);
            VecAXPY(_tmpB1, -1.0, _tmpB2); // residual

            VecCopy(rt_j->vz[ii], _tmpB2);
            VecAXPY(_tmpB2, 0.5*dt*RD/CV, dG_z->vz[ii]);
            vo->AssembleConstWithRho(ex, ey, _tmpB2, vo->VB);

            KSPSolve(kspColB, _tmpB1, de);
            VecAXPY(exner_j->vz[ii], 1.0, de);

            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);
*/
            VecNorm(de, NORM_2, &norm_dx);
            VecNorm(exner_j->vz[ii], NORM_2, &norm_x);
            if(norm_dx/norm_x > max_norm_exner) max_norm_exner = norm_dx/norm_x;

            // update velocity
/*
            diagnose_Phi_z(ex, ey, velz_i->vz[ii], velz_j->vz[ii], Phi);
            MatMult(vo->V01, Phi, fw); // bernoulli function gradient

            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->VB, exner_h->vz[ii], _tmpB1);
            MatMult(vo->V01, _tmpB1, _tmpA1);
            vo->AssembleLinearInv(ex, ey, vo->VA_inv);
            MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
            vo->AssembleLinearWithTheta(ex, ey, theta_h->vz[ii], vo->VA);
            MatMult(vo->VA, _tmpA2, _tmpA1);
            VecAXPY(fw, 1.0, _tmpA1); // pressure gradient term

            vo->AssembleLinear(ex, ey, vo->VA);
            MatMult(vo->VA, velz_i->vz[ii], _tmpA1);
            VecAYPX(fw, -dt, _tmpA1);

            VecCopy(velz_j->vz[ii], dw);
            KSPSolve(kspColA, fw, velz_j->vz[ii]);
            VecAXPY(dw, -1.0, velz_j->vz[ii]);
*/

/*
            assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii],
                                velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], rt_i->vz[ii], rt_j->vz[ii],
                                fw, F_z->vz[ii], G_z->vz[ii]);
            VecScale(fw, -1.0);

            assemble_precon_z(ex, ey, theta_h->vz[ii], rho_i->vz[ii], rt_i->vz[ii], rt_j->vz[ii], exner_h->vz[ii], &PC_z, bous->vz[ii]);

            KSPCreate(MPI_COMM_SELF, &ksp_z);
            KSPSetOperators(ksp_z, PC_z, PC_z);
            KSPGetPC(ksp_z, &pc);
            PCSetType(pc, PCLU);
            KSPSetOptionsPrefix(ksp_z, "ksp_z_");
            KSPSetFromOptions(ksp_z);
            KSPSolve(ksp_z, fw, dw);
            KSPDestroy(&ksp_z);
            VecAXPY(velz_j->vz[ii], 1.0, dw);
*/

            MatMult(PC_grad, F_exner, _tmpA1);
            VecAYPX(_tmpA1, -dt, fw);
            vo->AssembleLinearWithRT(ex, ey, bous->vz[ii], vo->VA, true);
            KSPSolve(kspColA, _tmpA1, dw);
            VecAXPY(velz_j->vz[ii], 1.0, dw);

            VecNorm(dw, NORM_2, &norm_dx);
            VecNorm(velz_j->vz[ii], NORM_2, &norm_x);
            if(norm_dx/norm_x > max_norm_w) max_norm_w = norm_dx/norm_x;

/*
            // mass flux
            diagnose_F_z(ex, ey, velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], F_z->vz[ii]);

            // temperature flux
            vo->AssembleLinearInv(ex, ey, vo->VA_inv);
            vo->AssembleLinearWithTheta(ex, ey, theta_h->vz[ii], vo->VA);
            MatMult(vo->VA, F_z->vz[ii], _tmpA1);
            MatMult(vo->VA_inv, _tmpA1, G_z->vz[ii]);
*/

            MatMult(vo->V10, F_z->vz[ii], dF_z->vz[ii]);
            MatMult(vo->V10, G_z->vz[ii], dG_z->vz[ii]);

            // undate density
            VecCopy(rho_j->vz[ii], de);
            VecCopy(rho_i->vz[ii], rho_j->vz[ii]);
            VecAXPY(rho_j->vz[ii], -dt, dF_z->vz[ii]);
            VecAXPY(de, -1.0, rho_j->vz[ii]);

            VecNorm(de, NORM_2, &norm_dx);
            VecNorm(rho_j->vz[ii], NORM_2, &norm_x);
            if(norm_dx/norm_x > max_norm_rho) max_norm_rho = norm_dx/norm_x;

            // update potential temperature
            VecCopy(rt_j->vz[ii], de);
            VecCopy(rt_i->vz[ii], rt_j->vz[ii]);
            VecAXPY(rt_j->vz[ii], -dt, dG_z->vz[ii]);
            VecAXPY(de, -1.0, rt_j->vz[ii]);

            VecNorm(de, NORM_2, &norm_dx);
            VecNorm(rt_j->vz[ii], NORM_2, &norm_x);
            if(norm_dx/norm_x > max_norm_rt) max_norm_rt = norm_dx/norm_x;
        }

        diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        theta_h->VertToHoriz();
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(theta_h->vz[ii], 1.0, theta_i->vz[ii]);
            VecScale(theta_h->vz[ii], 0.5);
        }
        theta_h->VertToHoriz();

/*
        rt_j->VertToHoriz();
        for(int ii = 0; ii < geom->nk; ii++) {
            VecCopy(exner_h->vh[ii], exner_j->vh[ii]);
            diagnose_Pi(ii, rt_i->vl[ii], rt_j->vl[ii], exner_h->vh[ii]);
            VecAXPY(exner_j->vh[ii], -1.0, exner_h->vh[ii]);
            VecNorm(exner_j->vh[ii], NORM_2, &norm_dx);
            VecNorm(exner_h->vh[ii], NORM_2, &norm_x);
            if(!rank) cout << ii << "\t|d_exner|/|exner|: " << norm_dx/norm_x << endl;
        }
        exner_h->UpdateLocal();
        exner_h->HorizToVert();
*/

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
velz_j->VertToHoriz();
velz_j->UpdateGlobal();
rho_j->VertToHoriz();
rho_j->UpdateGlobal();
rt_j->VertToHoriz();
rt_j->UpdateGlobal();
exner_j->VertToHoriz();
exner_j->UpdateGlobal();
theta_h->VertToHoriz();
theta_h->UpdateGlobal();
dump(NULL, velz_j, rho_j, rt_j, exner_j, theta_h, 9999);
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

    // write output
    if(save) {
        step++;
        dump(NULL, velz_i, rho_i, rt_i, exner_i, theta_h, step);
    }

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
    delete bous;
    delete rt_eos;
    VecDestroy(&fw);
    VecDestroy(&dw);
    VecDestroy(&de);
    VecDestroy(&Phi);
    VecDestroy(&F_exner);
    MatDestroy(&PC_exner);
    MatDestroy(&PC_div);
    MatDestroy(&PC_grad);
    KSPDestroy(&kspColA);
    KSPDestroy(&kspColB);
}

void Euler::solve_vert_coupled(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int ex, ey, elOrd2, itt = 0;
    int nDofsTotal = (4*geom->nk - 1)*vo->n2;
    double norm_x, norm_dx, max_norm_w, max_norm_exner, max_norm_rho, max_norm_rt;
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
    L2Vecs* dG_z_i = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* bous = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_eos = new L2Vecs(geom->nk, topo, geom);
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

    for(int ii = 0; ii < geom->nk; ii++) {
        diagnose_Pi(ii, rt_i->vl[ii], rt_i->vl[ii], exner_h->vh[ii]);
    }
    //exner_h->CopyFromHoriz(exner_i->vh);
    exner_h->UpdateLocal();
    exner_h->HorizToVert();

    initBousFac(theta_h, bous->vz);
    do {
        max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        //initBousFac(theta_h, bous->vz);

#ifdef EXNER_EOS
        // update the exner pressure
//        rt_j->VertToHoriz();
//        for(int ii = 0; ii < geom->nk; ii++) {
//            diagnose_Pi(ii, rt_j->vl[ii], rt_j->vl[ii], rt_eos->vh[ii]);
//        }
//        rt_eos->UpdateLocal();
//        rt_eos->HorizToVert();
#endif

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // implicit coupled solve
            assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
            //assemble_residual_z(ex, ey, theta_h->vz[ii], exner_j->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], F_w, F_z->vz[ii], G_z->vz[ii]);
#ifdef EXNER_EOS
            //VecCopy(exner_j->vz[ii], F_exner);
            //VecAXPY(F_exner, -1.0, rt_eos->vz[ii]);
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner);
#else
            exner_residual_z(ex, ey, rt_j->vz[ii], dG_z->vz[ii], exner_i->vz[ii], exner_j->vz[ii], F_exner);
#endif

            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z->vz[ii], dF_z->vz[ii]);
            MatMult(vo->V10, G_z->vz[ii], dG_z->vz[ii]);
            if(!itt) VecCopy(dG_z->vz[ii], dG_z_i->vz[ii]);

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

            //assemble_operator(ex, ey, theta_h->vz[ii], velz_j->vz[ii], rho_j->vz[ii], rt_j->vz[ii], exner_j->vz[ii], bous->vz[ii], dG_z->vz[ii], &PC_coupled);
            assemble_operator(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_j->vz[ii], bous->vz[ii], dG_z_i->vz[ii], &PC_coupled);

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

            VecNorm(d_exner, NORM_2, &norm_dx);
            VecNorm(exner_j->vz[ii], NORM_2, &norm_x);
            if(norm_dx/norm_x > max_norm_exner) max_norm_exner = norm_dx/norm_x;

            VecNorm(d_w, NORM_2, &norm_dx);
            VecNorm(velz_j->vz[ii], NORM_2, &norm_x);
            if(norm_dx/norm_x > max_norm_w) max_norm_w = norm_dx/norm_x;

            VecNorm(d_rho, NORM_2, &norm_dx);
            VecNorm(rho_j->vz[ii], NORM_2, &norm_x);
            if(norm_dx/norm_x > max_norm_rho) max_norm_rho = norm_dx/norm_x;

            VecNorm(d_rt, NORM_2, &norm_dx);
            VecNorm(rt_j->vz[ii], NORM_2, &norm_x);
            if(norm_dx/norm_x > max_norm_rt) max_norm_rt = norm_dx/norm_x;
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

    // write output
    if(save) {
        step++;
        dump(NULL, velz_i, rho_i, rt_i, exner_i, theta_h, step);
    }

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
    delete dG_z_i;
    delete bous;
    delete rt_eos;
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

void Euler::assemble_precon_z(int ex, int ey, Vec theta, Vec rho, Vec rt_i, Vec rt_j, Vec exner, Mat* _PC, Vec bous) {
    MatReuse reuse = (!pcz_DTV1) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    MatReuse reuse_2 = (!*_PC) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pcz_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pcz_DTV1, reuse, PETSC_DEFAULT, &pcz_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pcz_V0_invDTV1, reuse, PETSC_DEFAULT, &pcz_GRAD);

    vo->AssembleLinearWithRT(ex, ey, rt_j, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pcz_V0_invV0_rt);
    MatMatMult(vo->V10, pcz_V0_invV0_rt, reuse, PETSC_DEFAULT, &pcz_DV0_invV0_rt);

    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(vo->VB, pcz_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pcz_V1_PiDV0_invV0_rt);
    vo->AssembleConstWithRhoInv(ex, ey, rt_i, vo->VB);
    MatMatMult(vo->VB, pcz_V1_PiDV0_invV0_rt, reuse, PETSC_DEFAULT, &pcz_DIV);

    MatMatMult(pcz_GRAD, pcz_DIV, reuse_2, PETSC_DEFAULT, _PC);

    // add the boussinesque approximation
    vo->AssembleLinearWithRT(ex, ey, bous, vo->VA, true);
    MatAYPX(*_PC, -dt*dt*RD/CV, vo->VA, DIFFERENT_NONZERO_PATTERN);

    //vo->AssembleConLin(ex, ey, vo->VBA);
    //MatMatMult(vo->V01, vo->VBA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &vo->VA);//TODO: need separate matrix for this
    //MatAXPY(*_PC, dt, vo->VA, DIFFERENT_NONZERO_PATTERN);
}

void Euler::exner_precon_z(int ex, int ey, Vec dG, Vec exner, Vec rt, Vec bous, Vec theta, Mat *_PC, Mat *_DIV, Mat *_GRAD) {
    MatReuse reuse = (!*_PC) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pce_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pce_DTV1, reuse, PETSC_DEFAULT, &pce_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pce_V0_invDTV1, reuse, PETSC_DEFAULT, _GRAD);

    vo->AssembleLinearWithBousInv(ex, ey, bous, vo->VA_inv); // bous = 1.0 + dt^2/g (d theta/dz)/\bar{theta}
    vo->AssembleLinearWithRT(ex, ey, rt, vo->VA, true);
    MatMatMult(vo->VA, vo->VA_inv, reuse, PETSC_DEFAULT, &pce_V0_V0_inv);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pce_V0_V0_inv, reuse, PETSC_DEFAULT, &pce_V0_invV0_V0_inv);

    MatMatMult(vo->V10, pce_V0_invV0_V0_inv, reuse, PETSC_DEFAULT, &pce_DV0_invV0_rt);
    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(vo->VB, pce_DV0_invV0_rt, reuse, PETSC_DEFAULT, _DIV);

    MatMatMult(*_DIV, *_GRAD, reuse, PETSC_DEFAULT, _PC);

    vo->AssembleConst(ex, ey, vo->VB);
    MatAYPX(*_PC, -dt*dt*RD/CV, vo->VB, DIFFERENT_NONZERO_PATTERN);

    vo->AssembleConstWithRho(ex, ey, dG, vo->VB);
    MatAXPY(*_PC, +dt*RD/CV, vo->VB, DIFFERENT_NONZERO_PATTERN);
}

void DiagMatInv(Mat A, int nk, int nkl, int nDofskG, VecScatter gtol_k, Mat* Ainv) {
    int ii;
    Vec diag_u_l, diag_u_g;
    PetscScalar* uArray;

    VecCreateSeq(MPI_COMM_SELF, nk, &diag_u_l);
    VecCreateMPI(MPI_COMM_WORLD, nkl, nDofskG, &diag_u_g);

    MatGetDiagonal(A, diag_u_g);
    VecScatterBegin(gtol_k, diag_u_g, diag_u_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  gtol_k, diag_u_g, diag_u_l, INSERT_VALUES, SCATTER_FORWARD);
    VecGetArray(diag_u_l, &uArray);
    for(ii = 0; ii < nk; ii++) {
        uArray[ii] = 1.0/uArray[ii];
    }
    VecRestoreArray(diag_u_l, &uArray);
    VecScatterBegin(gtol_k, diag_u_l, diag_u_g, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  gtol_k, diag_u_l, diag_u_g, INSERT_VALUES, SCATTER_REVERSE);
    MatCreate(MPI_COMM_WORLD, Ainv);
    MatSetSizes(*Ainv, nkl, nkl, nDofskG, nDofskG);
    MatSetType(*Ainv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(*Ainv, 1, PETSC_NULL, 1, PETSC_NULL);
    MatDiagonalSet(*Ainv, diag_u_g, INSERT_VALUES);

    VecDestroy(&diag_u_l);
    VecDestroy(&diag_u_g);
}

void Euler::assemble_precon_x(int level, Vec* theta, Vec rt_i, Vec rt_j, Vec exner, Mat* _PC) {
    MatReuse reuse = (!_M1invM1) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    MatReuse reuse_2 = (!*_PC) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    Mat M1inv;
    Mat M2ThetaInv;
    Vec theta_h;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_h);
    VecZeroEntries(theta_h);
    VecAXPY(theta_h, 0.5, theta[level+0]);
    VecAXPY(theta_h, 0.5, theta[level+1]);

    M2->assemble(level, SCALE, true);
    M1->assemble(level, SCALE, true);
    DiagMatInv(M1->M, topo->n1, topo->n1l, topo->nDofs1G, topo->gtol_1, &M1inv);

    T->assemble(rt_i, level, SCALE); 
    DiagMatInv(T->M, topo->n2, topo->n2l, topo->nDofs2G, topo->gtol_2, &M2ThetaInv);

    F->assemble(rt_j, level, true, SCALE);
    MatMatMult(M1inv, F->M, reuse, PETSC_DEFAULT, &_M1invM1);
    MatMatMult(EtoF->E21, _M1invM1, reuse, PETSC_DEFAULT, &_DM1invM1);

    T->assemble(exner, level, SCALE); 
    MatMatMult(T->M, _DM1invM1, reuse, PETSC_DEFAULT, &_PiDM1invM1);
    MatMatMult(M2ThetaInv, _PiDM1invM1, reuse, PETSC_DEFAULT, &_ThetaPiDM1invM1);
    MatMatMult(M2->M, _ThetaPiDM1invM1, reuse, PETSC_DEFAULT, &_M2ThetaPiDM1invM1);
    MatMatMult(EtoF->E12, _M2ThetaPiDM1invM1, reuse, PETSC_DEFAULT, &_DM2ThetaPiDM1invM1);
    MatMatMult(M1inv, _DM2ThetaPiDM1invM1, reuse, PETSC_DEFAULT, &_M1DM2ThetaPiDM1invM1);

    F->assemble(theta_h, level, false, SCALE);
    MatMatMult(F->M, _M1DM2ThetaPiDM1invM1, reuse_2, PETSC_DEFAULT, _PC);
    MatAYPX(*_PC, -dt*dt*RD/CV, M1->M, DIFFERENT_NONZERO_PATTERN);

    R->assemble(fl[level], level, SCALE);
    MatAXPY(*_PC, dt, R->M, DIFFERENT_NONZERO_PATTERN);

    VecDestroy(&theta_h);
    MatDestroy(&M1inv);
    MatDestroy(&M2ThetaInv);
}

void Euler::diagnose_F_x(int level, Vec u1, Vec u2, Vec h1, Vec h2, Vec _F) {
    Vec hu, b, h1l, h2l;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &h1l);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &h2l);

    VecScatterBegin(topo->gtol_2, h1, h1l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h1, h1l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, h2, h2l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, h2, h2l, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);
    VecZeroEntries(_F);
    VecZeroEntries(hu);

    // assemble the nonlinear rhs mass matrix (note that hl is a local vector)
    F->assemble(h1l, level, true, SCALE);

    MatMult(F->M, u1, b);
    VecAXPY(hu, 1.0/3.0, b);

    MatMult(F->M, u2, b);
    VecAXPY(hu, 1.0/6.0, b);

    F->assemble(h2l, level, true, SCALE);

    MatMult(F->M, u1, b);
    VecAXPY(hu, 1.0/6.0, b);

    MatMult(F->M, u2, b);
    VecAXPY(hu, 1.0/3.0, b);

    // solve the linear system
    M1->assemble(level, SCALE, true);
    KSPSolve(ksp1, hu, _F);

    VecDestroy(&hu);
    VecDestroy(&b);
    VecDestroy(&h1l);
    VecDestroy(&h2l);
}

void Euler::diagnose_Phi_x(int level, Vec u1, Vec u2, Vec* Phi) {
    Vec u1l, u2l, b;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &u1l);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &u2l);

    VecScatterBegin(topo->gtol_1, u1, u1l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, u1, u1l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, u2, u2l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, u2, u2l, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Phi);
    VecZeroEntries(*Phi);

    // u^2 terms (0.5 factor incorportated into the matrix assembly)
    K->assemble(u1l, level, SCALE);

    MatMult(K->M, u1, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    MatMult(K->M, u2, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    K->assemble(u2l, level, SCALE);

    MatMult(K->M, u2, b);
    VecAXPY(*Phi, 1.0/3.0, b);

    VecDestroy(&u1l);
    VecDestroy(&u2l);
    VecDestroy(&b);
}

// diagnose the exner pressure using a quadratic approximation to the equation of state, 
// in order to integrate this exactly (in time)
void Euler::diagnose_Pi(int level, Vec rt1, Vec rt2, Vec Pi) {
    Vec rhs;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);

    VecZeroEntries(rhs);
    eos->assemble(rt1, level, SCALE);
    VecAXPY(rhs, 0.5, eos->vg);
    eos->assemble(rt2, level, SCALE); // TODO uninitialised value!
    VecAXPY(rhs, 0.5, eos->vg);
    M2->assemble(level, SCALE, true);
    KSPSolve(ksp2, rhs, Pi);

    VecDestroy(&rhs);
/*
    eos->assemble_quad(rt1, rt2, level, SCALE);
    M2->assemble(level, SCALE, true);
    KSPSolve(ksp2, eos->vg, Pi);
*/
}

// input vectors are all vertical
void Euler::exner_residual_z(int ex, int ey, Vec rt, Vec dG, Vec exner_prev, Vec exner_curr, Vec F_exner) {
#ifndef EXNER_EOS
    vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    MatMult(vo->VB, exner_curr, F_exner);
    MatMult(vo->VB, exner_prev, _tmpB1);
    VecAXPY(F_exner, -1.0, _tmpB1);
    vo->AssembleConstWithRho(ex, ey, dG, vo->VB);
    MatMult(vo->VB, exner_curr, _tmpB1);
    VecAXPY(F_exner, +dt*RD/CV, _tmpB1);
#else
/*
    vo->AssembleConstWithRho(ex, ey, dG, vo->VB);
    MatMult(vo->VB, exner_curr, _tmpB1);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMult(vo->VB_inv, _tmpB1, F_exner);
    VecAYPX(F_exner, dt*RD/CV, exner_curr);
    VecAXPY(F_exner, -1.0, exner_prev);
*/
/*
    vo->Assemble_EOS_RHS(ex, ey, exner_curr, _tmpB1, (P0/RD)*pow(CP, -CV/RD), CV/RD);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMult(vo->VB_inv, _tmpB1, F_exner);
    vo->AssembleConst(ex, ey, vo->VB);
    VecSet(_tmpB1, 1.0);
    MatMult(vo->VB, _tmpB1, _tmpB2);
    VecAXPY(F_exner, -1.0, _tmpB2);
*/
#endif
}

void Euler::diagnose_wxu(int level, Vec u1, Vec u2, Vec* wxu) {
    Vec w1, w2, wl, uh;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, wxu);

    curl(false, u1, &w1, level, true);
    curl(false, u2, &w2, level, true);
    VecAXPY(w1, 1.0, w2);
    VecScale(w1, 0.5);

    VecScatterBegin(topo->gtol_0, w1, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, w1, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(uh);
    VecAXPY(uh, 0.5, u1);
    VecAXPY(uh, 0.5, u2);

    R->assemble(wl, level, SCALE);
    MatMult(R->M, uh, *wxu);

    VecDestroy(&w1);
    VecDestroy(&w2);
    VecDestroy(&wl);
    VecDestroy(&uh);
}

void Euler::diagnose_F_z(int ex, int ey, Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec _F) {
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

void Euler::diagnose_Phi_z(int ex, int ey, Vec velz1, Vec velz2, Vec Phi) {
    int ei = ey*topo->nElsX + ex;

    VecZeroEntries(Phi);

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
void Euler::diagTheta2(Vec* rho, Vec* rt, Vec* theta) {
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

// compute the vorticity components dudz, dvdz
void Euler::diagHorizVort(Vec* velx, Vec* dudz) {
    int ii;
    Vec* Mu = new Vec[geom->nk];
    Vec  du;
    PC pc;
    KSP ksp1_t;

    KSPCreate(MPI_COMM_WORLD, &ksp1_t);
    KSPSetOperators(ksp1_t, M1t->M, M1t->M);
    KSPSetTolerances(ksp1_t, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1_t, KSPGMRES);
    KSPGetPC(ksp1_t, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp1_t, "ksp1_t_");
    KSPSetFromOptions(ksp1_t);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu[ii]);
        M1->assemble(ii, SCALE, true);
        MatMult(M1->M, velx[ii], Mu[ii]);
    }

    for(ii = 0; ii < geom->nk-1; ii++) {
        VecZeroEntries(du);
        VecAXPY(du, +1.0, Mu[ii+1]);
        VecAXPY(du, -1.0, Mu[ii+0]);
        M1t->assemble(ii, SCALE);
        KSPSolve(ksp1_t, du, dudz[ii]);
    }

    VecDestroy(&du);
    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&Mu[ii]);
    }
    delete[] Mu;
    KSPDestroy(&ksp1_t);
}

void Euler::assemble_residual_x(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                                Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fu, Vec _F, Vec _G) 
{
    Vec Phi, dPi, wxu, wxz, utmp, d2u, d4u;
    Vec theta_h, dp, dudz_h, velz_h, dudz_l;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &wxz);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &velz_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dudz_l);

    m0->assemble(level, SCALE);
    M1->assemble(level, SCALE, true);
    M2->assemble(level, SCALE, true);

    // assume theta is 0.5*(theta_i + theta_j)
    VecZeroEntries(theta_h);
    VecAXPY(theta_h, 0.5, theta[level+0]);
    VecAXPY(theta_h, 0.5, theta[level+1]);

    VecZeroEntries(fu);

    // assemble in the skew-symmetric parts of the vector
    diagnose_F_x(level, velx1, velx2, rho1, rho2, _F);
    diagnose_Phi_x(level, velx1, velx2, &Phi);
    grad(false, Pi, &dPi, level);
    diagnose_wxu(level, velx1, velx2, &wxu);

    MatMult(EtoF->E12, Phi, fu);
    VecAXPY(fu, 1.0, wxu);

    // add the pressure gradient force
    F->assemble(theta_h, level, false, SCALE);
    MatMult(F->M, dPi, dp);
    VecAXPY(fu, 1.0, dp);

    // diagnose the temperature flux (assume the H(div) mass matrix has
    // already been assembled at this level
    MatMult(F->M, _F, utmp);
    KSPSolve(ksp1, utmp, _G);

    // second voritcity term
    VecZeroEntries(utmp);
    if(level > 0) {
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, dudz1[level-1]);
        VecAXPY(dudz_h, 0.5, dudz2[level-1]);
        VecScatterBegin(topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);

        VecZeroEntries(velz_h);
        VecAXPY(velz_h, 0.5, velz1[level-1]);
        VecAXPY(velz_h, 0.5, velz2[level-1]);

        Rh->assemble(dudz_l, SCALE);
        MatMult(Rh->M, velz_h, dp);
        VecAXPY(utmp, 0.5, dp);
    }
    if(level < geom->nk-1) {
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, dudz1[level+0]);
        VecAXPY(dudz_h, 0.5, dudz2[level+0]);
        VecScatterBegin(topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);

        VecZeroEntries(velz_h);
        VecAXPY(velz_h, 0.5, velz1[level+0]);
        VecAXPY(velz_h, 0.5, velz2[level+0]);

        Rh->assemble(dudz_l, SCALE);
        MatMult(Rh->M, velz_h, dp);
        VecAXPY(utmp, 0.5, dp);
    }
    VecAXPY(fu, 1.0, utmp);
    VecScale(fu, dt);

    // assemble the mass matrix terms
    MatMult(M1->M, velx2, utmp);
    VecAXPY(fu, +1.0, utmp);
    MatMult(M1->M, velx1, utmp);
    VecAXPY(fu, -1.0, utmp);

    if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, velx1);
        VecAXPY(utmp, 0.5, velx2);
        laplacian(false, utmp, &d2u, level);
        laplacian(false, d2u, &d4u, level);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    // clean up
    VecDestroy(&utmp);
    VecDestroy(&Phi);
    VecDestroy(&dPi);
    VecDestroy(&wxu);
    VecDestroy(&theta_h);
    VecDestroy(&dp);
    VecDestroy(&wxz);
    VecDestroy(&dudz_h);
    VecDestroy(&velz_h);
    VecDestroy(&dudz_l);
}

void Euler::assemble_residual_z(int ex, int ey, Vec theta, Vec Pi, 
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
}

void Euler::init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y) {
    int ex, ey, ii, kk, mp1, mp12;
    int *inds0, *loc02;
    UtQmat* UQ = new UtQmat(topo, geom, node, edge);
    PetscScalar *bArray;
    Vec bl, bg, UQb;
    IS isl, isg;
    VecScatter scat;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    loc02 = new int[2*topo->n0];
    VecCreateSeq(MPI_COMM_SELF, 2*topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, 2*topo->n0l, 2*topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &UQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bg);
        VecGetArray(bl, &bArray);

        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                inds0 = topo->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    bArray[2*inds0[ii]+0] = func_x(geom->x[inds0[ii]], kk);
                    bArray[2*inds0[ii]+1] = func_y(geom->x[inds0[ii]], kk);
                }
            }
        }
        VecRestoreArray(bl, &bArray);

        // create a new vec scatter object to handle vector quantity on nodes
        for(ii = 0; ii < topo->n0; ii++) {
            loc02[2*ii+0] = 2*topo->loc0[ii]+0;
            loc02[2*ii+1] = 2*topo->loc0[ii]+1;
        }
        ISCreateStride(MPI_COMM_WORLD, 2*topo->n0, 0, 1, &isl);
        ISCreateGeneral(MPI_COMM_WORLD, 2*topo->n0, loc02, PETSC_COPY_VALUES, &isg);
        VecScatterCreate(bg, isg, bl, isl, &scat);
        VecScatterBegin(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

        M1->assemble(kk, SCALE, true);
        MatMult(UQ->M, bg, UQb);
        VecScale(UQb, SCALE);
        KSPSolve(ksp1, UQb, u[kk]);

        ISDestroy(&isl);
        ISDestroy(&isg);
        VecScatterDestroy(&scat);
    }

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&UQb);
    delete UQ;
    delete[] loc02;
}

void Euler::init2(Vec* h, ICfunc3D* func) {
    int ex, ey, ii, kk, mp1, mp12, *inds0;
    PetscScalar *bArray;
    Vec bl, bg, WQb;
    WtQmat* WQ = new WtQmat(topo, geom, edge);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &WQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bl);
        VecZeroEntries(bg);
        VecGetArray(bl, &bArray);

        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                inds0 = topo->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    bArray[inds0[ii]] = func(geom->x[inds0[ii]], kk);
                }
            }
        }
        VecRestoreArray(bl, &bArray);
        VecScatterBegin(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

        MatMult(WQ->M, bg, WQb);
        VecScale(WQb, SCALE);          // have to rescale the M2 operator as the metric terms scale
        M2->assemble(kk, SCALE, true); // this down to machine precision, so rescale the rhs as well
        KSPSolve(ksp2, WQb, h[kk]);
    }

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void Euler::initTheta(Vec theta, ICfunc3D* func) {
    int ex, ey, ii, mp1, mp12, *inds0;
    PetscScalar *bArray;
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
                bArray[inds0[ii]] = func(geom->x[inds0[ii]], 0);
            }
        }
    }
    VecRestoreArray(bl, &bArray);
    VecScatterBegin(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    M2->assemble(0, SCALE, false);
    MatMult(WQ->M, bg, WQb);
    VecScale(WQb, SCALE);
    KSPSolve(ksp2, WQb, theta);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void Euler::integrateTheta(Vec* theta, double* tb) {
    int ei, mp1, mp12;
    double th_l, th_q, det, vol_l, vol_g;
    PetscScalar* tArray;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    for(int level = 0; level < geom->nk+1; level++) {
        th_l = 0.0;
        vol_l = 0.0;
        VecGetArray(theta[level], &tArray);
        for(int ey = 0; ey < topo->nElsX; ey++) {
            for(int ex = 0; ex < topo->nElsX; ex++) {
                ei = ey*topo->nElsX + ex;

                for(int ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    geom->interp2_g(ex, ey, ii%mp1, ii/mp1, tArray, &th_q);
                    th_l += det*quad->w[ii%mp1]*quad->w[ii/mp1]*th_q;
                    vol_l += det*quad->w[ii%mp1]*quad->w[ii/mp1];
                }
            }
        }
        VecRestoreArray(theta[level], &tArray);

        MPI_Allreduce(&th_l, &tb[level], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&vol_l, &vol_g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tb[level] /= vol_g;
    }
}

// bous is an array of vertical local vectors (piecewise constant in the vertical)
void Euler::initBousFac(L2Vecs* theta, Vec* bous) {
    int ex, ey, ei, kk, mp1, mp12;
    Vec bg, WQb;
    WtQmat* WQ = new WtQmat(topo, geom, edge);
    double tb[999];
    bool init_avg = (thetaBar) ? false : true;

    if(init_avg) {
        thetaBar = new L2Vecs(geom->nk+1, topo, geom);

        integrateTheta(theta->vl, tb);

        mp1 = quad->n + 1;
        mp12 = mp1*mp1;

        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &bg);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &WQb);

        M2->assemble(0, SCALE, false); // only need to assemble once as independent of vertical levels
        for(kk = 0; kk < geom->nk+1; kk++) {
            VecSet(bg, tb[kk]);
            MatMult(WQ->M, bg, WQb);
            VecScale(WQb, SCALE);
            KSPSolve(ksp2, WQb, thetaBar->vh[kk]);
        }
        thetaBar->UpdateLocal();
        thetaBar->HorizToVert();

        VecDestroy(&bg);
        VecDestroy(&WQb);
    }

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey * topo->nElsX + ex;
            vo->AssembleConst(ex, ey, vo->VB);
            vo->AssembleConstWithThetaInv(ex, ey, thetaBar->vz[ei], vo->VB_inv);
            //MatMult(vo->V10_full, thetaBar->vz[ei], _tmpB1);
            MatMult(vo->V10_full, theta->vz[ei], _tmpB1);
            MatMult(vo->VB, _tmpB1, _tmpB2);

            // assemble as I + dt^2.g.(d theta/dz)/\bar{theta}
            vo->AssembleConLin2(ex, ey, vo->VBA2);
            MatMult(vo->VBA2, thetaBar->vz[ei], _tmpB1);
            VecAYPX(_tmpB2, 0.25*dt*dt*GRAVITY, _tmpB1);

            MatMult(vo->VB_inv, _tmpB2, bous[ei]);
        }
    }

    delete WQ;
}

void Euler::coriolisMatInv(Mat A, Mat* Ainv) {
    int mi, mf, nCols1, nCols2;
    const int *cols1, *cols2;
    const double *vals1;
    const double *vals2;
    double D[2][2], Dinv[2][2], detInv;
    double vals1Inv[9999], vals2Inv[9999];
    int rows[2];

    MatCreate(MPI_COMM_WORLD, Ainv);
    MatSetSizes(*Ainv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(*Ainv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(*Ainv, 2, PETSC_NULL, 2, PETSC_NULL);

    MatGetOwnershipRange(A, &mi, &mf);
    for(int mm = mi; mm < mf; mm += 2) {
        rows[0] = mm+0;
        rows[1] = mm+1;

        MatGetRow(A, mm+0, &nCols1, &cols1, &vals1);
        for(int ci = 0; ci < nCols1; ci++) {
            if(cols1[ci] == mm+0) {
                D[0][0] = vals1[ci+0];
                D[0][1] = vals1[ci+1];
                break;
            }
        }
        MatRestoreRow(A, mm+0, &nCols1, &cols1, &vals1);

        MatGetRow(A, mm+1, &nCols2, &cols2, &vals2);
        for(int ci = 0; ci < nCols2; ci++) {
            if(cols2[ci] == mm+1) {
                D[1][0] = vals2[ci-1];
                D[1][1] = vals2[ci+0];
                break;
            }
        }
        MatRestoreRow(A, mm+1, &nCols2, &cols2, &vals2);

        detInv = 1.0/(D[0][0]*D[1][1] - D[0][1]*D[1][0]);

        Dinv[0][0] = +detInv*D[1][1];
        Dinv[1][1] = +detInv*D[0][0];
        Dinv[0][1] = -detInv*D[1][0];
        Dinv[1][0] = -detInv*D[0][1];

        vals1Inv[0] = Dinv[0][0];
        vals1Inv[1] = Dinv[0][1];
        vals2Inv[0] = Dinv[1][0];
        vals2Inv[1] = Dinv[1][1];

        MatSetValues(*Ainv, 2, rows, 2, rows, vals1Inv, INSERT_VALUES);
    }
    MatAssemblyBegin(*Ainv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  *Ainv, MAT_FINAL_ASSEMBLY);
}

void Euler::assemble_residual_u(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                                Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fu, bool add_curr) 
{
    Vec Phi, dPi, wxu, wxz, utmp, d2u, d4u;
    Vec theta_h, dp, dudz_h, velz_h, dudz_l;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &wxz);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &velz_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_h);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dudz_l);

    m0->assemble(level, SCALE);
    M1->assemble(level, SCALE, true);
    M2->assemble(level, SCALE, true);

    // assume theta = 0.5*(theta_i + theta_j)
    VecZeroEntries(theta_h);
    VecAXPY(theta_h, 0.5, theta[level+0]);
    VecAXPY(theta_h, 0.5, theta[level+1]);

    VecZeroEntries(fu);

    // assemble in the skew-symmetric parts of the vector
    diagnose_Phi_x(level, velx1, velx2, &Phi);
    grad(false, Pi, &dPi, level);
    diagnose_wxu(level, velx1, velx2, &wxu);

    MatMult(EtoF->E12, Phi, fu);
    VecAXPY(fu, 1.0, wxu);

    // add the pressure gradient force
    F->assemble(theta_h, level, false, SCALE);
    MatMult(F->M, dPi, dp);
    VecAXPY(fu, 1.0, dp);

    // second voritcity term
    VecZeroEntries(utmp);
    if(level > 0) {
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, dudz1[level-1]);
        VecAXPY(dudz_h, 0.5, dudz2[level-1]);
        VecScatterBegin(topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);

        VecZeroEntries(velz_h);
        VecAXPY(velz_h, 0.5, velz1[level-1]);
        VecAXPY(velz_h, 0.5, velz2[level-1]);

        Rh->assemble(dudz_l, SCALE);
        MatMult(Rh->M, velz_h, dp);
        VecAXPY(utmp, 0.5, dp);
    }
    if(level < geom->nk-1) {
        VecZeroEntries(dudz_h);
        VecAXPY(dudz_h, 0.5, dudz1[level+0]);
        VecAXPY(dudz_h, 0.5, dudz2[level+0]);
        VecScatterBegin(topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, dudz_h, dudz_l, INSERT_VALUES, SCATTER_FORWARD);

        VecZeroEntries(velz_h);
        VecAXPY(velz_h, 0.5, velz1[level+0]);
        VecAXPY(velz_h, 0.5, velz2[level+0]);

        Rh->assemble(dudz_l, SCALE);
        MatMult(Rh->M, velz_h, dp);
        VecAXPY(utmp, 0.5, dp);
    }
    VecAXPY(fu, 1.0, utmp);
    VecScale(fu, dt);

    // assemble the mass matrix terms
    if(add_curr) {
        MatMult(M1->M, velx2, utmp);
        VecAXPY(fu, +1.0, utmp);
    }
    MatMult(M1->M, velx1, utmp);
    VecAXPY(fu, -1.0, utmp);

    if(do_visc) {
        VecZeroEntries(utmp);
        VecAXPY(utmp, 0.5, velx1);
        VecAXPY(utmp, 0.5, velx2);
        laplacian(false, utmp, &d2u, level);
        laplacian(false, d2u, &d4u, level);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(fu, dt, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    // clean up
    VecDestroy(&utmp);
    VecDestroy(&Phi);
    VecDestroy(&dPi);
    VecDestroy(&wxu);
    VecDestroy(&theta_h);
    VecDestroy(&dp);
    VecDestroy(&wxz);
    VecDestroy(&dudz_h);
    VecDestroy(&velz_h);
    VecDestroy(&dudz_l);
}

void Euler::assemble_residual_w(int ex, int ey, Vec theta, Vec Pi, Vec velz1, Vec velz2, Vec fw, bool add_curr) {
    diagnose_Phi_z(ex, ey, velz1, velz2, _Phi_z);

    // assemble the momentum equation residual
    VecZeroEntries(fw);
    vo->AssembleLinear(ex, ey, vo->VA);
    if(add_curr) {
        MatMult(vo->VA, velz2, fw);
    }

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
}

void Euler::repack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner) {
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

void Euler::unpack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner) {
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

void Euler::assemble_operator(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, Vec bous, Vec dG, Mat* _PC) {
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

    if(!*_PC) {
        MatCreateSeqAIJ(MPI_COMM_SELF, nDofsTotal, nDofsTotal, 12*n2, NULL, _PC);
    }
    MatZeroEntries(*_PC);

    // [u,u] block
    //vo->AssembleLinearWithRT(ex, ey, bous, vo->VA, true);
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

    // [exner,u] block
#ifndef EXNER_EOS
    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_V_PiDV0_invV0_rt);
    MatScale(pc_V_PiDV0_invV0_rt, 0.5*dt*RD/CV);
    MatGetOwnershipRange(pc_V_PiDV0_invV0_rt, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V_PiDV0_invV0_rt, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + 2*nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci];
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_V_PiDV0_invV0_rt, mm, &nCols, &cols, &vals);
    }
#endif

    // [exner,rt] block
#ifndef EXNER_EOS
    MatMatMult(vo->VB, pc_DV0_invV01, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V1DV0_invV01);
    MatAYPX(pc_V1DV0_invV01, 0.5*dt*RD/CV, vo->VB, DIFFERENT_NONZERO_PATTERN);
    MatGetOwnershipRange(pc_V1DV0_invV01, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V1DV0_invV01, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + 2*nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_V1DV0_invV01, mm, &nCols, &cols, &vals);
    }
#else
/*
    vo->AssembleConstWithEOS(ex, ey, rt, vo->VB);
    vo->AssembleConstWithRhoInv(ex, ey, rt, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    vo->AssembleConst(ex, ey, vo->VB);
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
*/
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
#endif

    // [exner,exner] block
#ifndef EXNER_EOS
    vo->AssembleConstWithRho(ex, ey, rt, vo->VB_inv);
    vo->AssembleConstWithRho(ex, ey, dG, vo->VB);
    MatAYPX(vo->VB, 0.5*dt*RD/CV, vo->VB_inv, DIFFERENT_NONZERO_PATTERN);
    MatGetOwnershipRange(vo->VB, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(vo->VB, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + 2*nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + 2*nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(vo->VB, mm, &nCols, &cols, &vals);
    }
#else
/*
    vo->AssembleConst(ex, ey, vo->VB);
    MatGetOwnershipRange(vo->VB, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(vo->VB, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + 2*nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + 2*nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(vo->VB, mm, &nCols, &cols, &vals);
    }
*/
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
#endif

    // [u,theta] block
    MatMult(pc_V0_invDTV1, exner, _tmpA1);
    vo->AssembleLinConWithTheta(ex, ey, vo->VAB, _tmpA1);
    vo->AssembleLinearWithBousInv(ex, ey, rho, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VAB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV01);
    vo->AssembleLinear(ex, ey, vo->VA);
    MatMatMult(vo->VA, pc_V0_invV01, reuse, PETSC_DEFAULT, &pc_V0V0_invV01);
    MatScale(pc_V0V0_invV01, 0.5*dt);
    MatGetOwnershipRange(pc_V0V0_invV01, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V0V0_invV01, mm, &nCols, &cols, &vals);
        ri = mm;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW + nDofsRho;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_V0V0_invV01, mm, &nCols, &cols, &vals);
    }

    // [exner,rho] block
/*
    vo->AssembleConstWithTheta(ex, ey, theta, vo->VB);
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_VBVB_rt_invVB_pi);

    vo->AssembleLinConWithTheta(ex, ey, vo->VAB, theta);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV01);
    vo->AssembleLinearWithTheta(ex, ey, velz, vo->VA);
    MatMatMult(vo->VA, pc_V0_invV01, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0V0_invV01);
    MatMatMult(vo->V10, pc_V0V0_invV01, reuse, PETSC_DEFAULT, &pc_DV0V0_invV01);
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(vo->VB_inv, pc_DV0V0_invV01, reuse, PETSC_DEFAULT, &pc_V1_invDV0V0_invV01);
    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(vo->VB, pc_V1_invDV0V0_invV01, reuse, PETSC_DEFAULT, &pc_V1V1_invDV0V0_invV01);
    MatAYPX(pc_V1V1_invDV0V0_invV01, 0.5*dt*RD/CV, pc_VBVB_rt_invVB_pi, DIFFERENT_NONZERO_PATTERN);

    MatGetOwnershipRange(pc_V1V1_invDV0V0_invV01, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_V1V1_invDV0V0_invV01, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + 2*nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_V1V1_invDV0V0_invV01, mm, &nCols, &cols, &vals);
    }
*/

    MatAssemblyBegin(*_PC, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  *_PC, MAT_FINAL_ASSEMBLY);
}
