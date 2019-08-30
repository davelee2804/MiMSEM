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
#include "Euler_imp_4.h"

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

#define EXPLICIT_RHO_UPDATE
//#define EXPLICIT_THETA_UPDATE

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

    _PCz = NULL;
    _PCx = NULL;
    pc_A_rt = NULL;
    _V0_invV0_rt = NULL;

    schur = new Schur(topo, geom);
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
    }
    delete[] fg;
    delete[] fl;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
        VecDestroy(&zv[ii]);
    }
    delete[] gv;
    delete[] zv;

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

    delete schur;

    MatDestroy(&_PCz);
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
    if(velz) {
        sprintf(fieldname, "velocity_z");
        for(int ii = 0; ii < geom->nk-1; ii++) {
            geom->write2(velz->vh[ii], fieldname, num, ii, false);
        }
    }
}

double MaxNorm(Vec dx, Vec x, double max_norm) {
    double norm_dx, norm_x, new_max_norm;

    VecNorm(dx, NORM_2, &norm_dx);
    VecNorm(x, NORM_2, &norm_x);
    new_max_norm = (norm_dx/norm_x > max_norm) ? norm_dx/norm_x : max_norm;
    return new_max_norm;
}

void Euler::solve_vert_coupled(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
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

void Euler::solve_vert_schur(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
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
            assemble_operator_schur(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_j->vz[ii], 
                                    F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner);

            VecAXPY(velz_j->vz[ii],  1.0, d_w);
#ifndef EXPLICIT_RHO_UPDATE
            VecAXPY(rho_j->vz[ii],   1.0, d_rho);
#endif
#ifndef EXPLICIT_THETA_UPDATE
            VecAXPY(rt_j->vz[ii],    1.0, d_rt);
#endif
            VecAXPY(exner_j->vz[ii], 1.0, d_exner);

            // update the density
#ifdef EXPLICIT_RHO_UPDATE
            diagnose_F_z(ex, ey, velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], F_z);
            MatMult(vo->V10, F_z, dF_z);
            VecCopy(rho_j->vz[ii], d_rho);
            VecCopy(rho_i->vz[ii], rho_j->vz[ii]);
            VecAXPY(rho_j->vz[ii], -dt, dF_z);
            VecAXPY(d_rho, -1.0, rho_j->vz[ii]);
#endif

#ifdef EXPLICIT_THETA_UPDATE
            vo->AssembleLinearInv(ex, ey, vo->VA);
            vo->AssembleLinearWithTheta(ex, ey, theta_h->vz[ii], vo->VA);
            MatMult(vo->VA, F_z, _tmpA1);
            MatMult(vo->VA_inv, _tmpA1, G_z);
            MatMult(vo->V10, G_z, dG_z);
            VecCopy(rt_j->vz[ii], d_rt);
            VecCopy(rt_i->vz[ii], rt_j->vz[ii]);
            VecAXPY(rt_j->vz[ii], -dt, dG_z);
            VecAXPY(d_rt, -1.0, rt_j->vz[ii]);
#endif

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

void Euler::solve_horiz_schur(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int lev, ii, itt = 0;
    double max_norm_u, max_norm_exner, max_norm_rho, max_norm_rt, norm_x;
    Vec* velx_j = new Vec[geom->nk];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* F_exner = new L2Vecs(geom->nk, topo, geom);
    Vec fu, frho, frt, fexner, du, drho, drt, dexner, _F, _G, dF, dG;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &frho);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &frt);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fexner);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &drho);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &drt);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dexner);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dG);
    

    for(lev = 0; lev < geom->nk; lev++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[lev]);
        VecCopy(velx_i[lev], velx_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[lev]);
    }
    rho_j->CopyFromHoriz(rho_i->vh);
    rt_j->CopyFromHoriz(rt_i->vh);
    exner_j->CopyFromHoriz(exner_i->vh);
    exner_h->CopyFromHoriz(exner_i->vh);
    velz_j->CopyFromHoriz(velz_i->vh);

    // diagnose the potential temperature
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rt_i->UpdateLocal();
    rt_i->HorizToVert();
    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->CopyFromVert(theta_i->vz);
    theta_h->VertToHoriz();

    // diagnose the vorticity terms
    diagHorizVort(velx_i, dudz_i);
    for(lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i[lev], dudz_j[lev]);
    }

    rt_j->UpdateLocal();
    rt_j->HorizToVert();

    do {
        max_norm_u = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        // exner pressure residual
        exner_j->UpdateLocal();
        exner_j->HorizToVert();
        for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            vo->Assemble_EOS_Residual(ii%topo->nElsX, ii/topo->nElsX, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
        }
        F_exner->VertToHoriz();
        F_exner->UpdateGlobal();

        for(lev = 0; lev < geom->nk; lev++) {
            // velocity residual
            assemble_residual_x(lev, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
                                velx_i[lev], velx_j[lev], rho_i->vh[lev], rho_j->vh[lev], fu, _F, _G);

            M2->assemble(lev, SCALE, true);

            // density residual
            MatMult(EtoF->E21, _F, dF);
            VecAYPX(dF, dt, rho_j->vh[lev]);
            VecAXPY(dF, -1.0, rho_i->vh[lev]);
            MatMult(M2->M, dF, frho);

            // density weighted potential temperature residual
            MatMult(EtoF->E21, _G, dG);
            VecAYPX(dG, dt, rt_j->vh[lev]);
            VecAXPY(dG, -1.0, rt_i->vh[lev]);
            MatMult(M2->M, dG, frt);

            // delta updates  - velx is a global vector, while theta and exner are local vectors
            assemble_schur_horiz(lev, theta_h->vl, velx_j[lev], rho_i->vl[lev], rt_i->vl[lev], exner_j->vl[lev], 
                                 fu, frho, frt, F_exner->vh[lev], du, drho, drt, dexner);

            VecAXPY(velx_j[lev], 1.0, du);
            VecAXPY(rt_j->vh[lev], 1.0, drt);
            VecAXPY(exner_j->vh[lev], 1.0, dexner);

            // update the density
            VecCopy(rho_j->vh[lev], drho);
            MatMult(EtoF->E21, _F, rho_j->vh[lev]);
            VecAYPX(rho_j->vh[lev], -dt, rho_i->vh[lev]);
            VecAXPY(drho, -1.0, rho_j->vh[lev]);

            max_norm_exner = MaxNorm(dexner, exner_j->vh[lev], max_norm_exner);
            max_norm_u     = MaxNorm(du,     velx_j[lev],      max_norm_u    );
            max_norm_rho   = MaxNorm(drho,   rho_j->vh[lev],   max_norm_rho  );
            max_norm_rt    = MaxNorm(drt,    rt_j->vh[lev],    max_norm_rt   );
        }

        MPI_Allreduce(&max_norm_exner, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_exner = norm_x;
        MPI_Allreduce(&max_norm_u,     &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_u     = norm_x;
        MPI_Allreduce(&max_norm_rho,   &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho   = norm_x;
        MPI_Allreduce(&max_norm_rt,    &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt    = norm_x;

        itt++;

        if(max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_rho < 1.0e-8 && max_norm_rt < 1.0e-8) done = true;
        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                 "\t|d_u|/|u|: "          << max_norm_u     <<
                                 "\t|d_rho|/|rho|: "      << max_norm_rho   <<
                                 "\t|d_rt|/|rt|: "        << max_norm_rt    << endl;

        diagHorizVort(velx_j, dudz_j);

        // diagnose the potential temperature (at the half step)
        rho_j->UpdateLocal();
        rho_j->HorizToVert();
        rt_j->UpdateLocal();
        rt_j->HorizToVert();
        diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        theta_h->VertToHoriz();
        for(lev = 0; lev < geom->nk; lev++) {
            VecScale(theta_h->vl[lev], 0.5);
            VecAXPY(theta_h->vl[lev], 0.5, theta_i->vl[lev]);

            VecZeroEntries(exner_h->vh[lev]);
            VecAXPY(exner_h->vh[lev], 0.5, exner_i->vh[lev]);
            VecAXPY(exner_h->vh[lev], 0.5, exner_j->vh[lev]);
        }
    } while(!done);

    // update the input/output fields
    for(lev = 0; lev < geom->nk; lev++) {
        VecCopy(velx_j[lev], velx_i[lev]);
    }
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_h->vh);

    // write output
    if(save) {
        dump(velx_i, NULL, rho_i, rt_i, exner_i, theta_h, step++);
    }

    for(lev = 0; lev < geom->nk; lev++) {
        VecDestroy(&velx_j[lev]);
        VecDestroy(&dudz_i[lev]);
        VecDestroy(&dudz_j[lev]);
    }
    delete[] velx_j;
    delete[] dudz_i;
    delete[] dudz_j;
    VecDestroy(&fu);
    VecDestroy(&frho);
    VecDestroy(&frt);
    VecDestroy(&fexner);
    VecDestroy(&du);
    VecDestroy(&drho);
    VecDestroy(&drt);
    VecDestroy(&dexner);
    VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&dF);
    VecDestroy(&dG);
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete theta_i;
    delete theta_h;
    delete velz_j;
    delete F_exner;
}

void Euler::solve_schur(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    bool do_vert = true;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_w, max_norm_exner, max_norm_rho, max_norm_rt, norm_x;
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_exner = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    Vec* velx_j = new Vec[geom->nk];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    Vec fu, frho, frt, du, drho, drt, dexner, _F, _G, dF, dG;
    Vec F_w, F_rho, F_rt, d_w, d_rho, d_rt, d_exner, F_z, G_z, dF_z, dG_z;

    for(int lev = 0; lev < geom->nk; lev++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[lev]);
        VecCopy(velx_i[lev], velx_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[lev]);
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &frho);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &frt);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &drho);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &drt);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dexner);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dG);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rt);
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
    velz_j->CopyFromVert(velz_i->vz);
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rho_j->CopyFromVert(rho_i->vz);
    rt_i->UpdateLocal();
    rt_i->HorizToVert();
    rt_j->CopyFromVert(rt_i->vz);
    exner_i->UpdateLocal();
    exner_i->HorizToVert();
    exner_j->CopyFromVert(exner_i->vz);
    exner_h->CopyFromVert(exner_i->vz);

    // diagnose the vorticity terms
    diagHorizVort(velx_i, dudz_i);
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i[lev], dudz_j[lev]);
    }

    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);

    do {
        max_norm_u = max_norm_w = max_norm_exner = max_norm_rho = max_norm_rt = 0.0;

        if(itt) {
            if(do_vert) {
                rho_j->UpdateLocal();
                rho_j->HorizToVert();
                rt_j->UpdateLocal();
                rt_j->HorizToVert();
                exner_j->UpdateLocal();
                exner_j->HorizToVert();
                for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
                    VecZeroEntries(exner_h->vz[ii]);
                    VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
                    VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);
                }
            }
            diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
            for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
                VecScale(theta_h->vz[ii], 0.5);
                VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
            }
            if(!do_vert) {
                theta_h->VertToHoriz();
            }
        }

        // construct the exner pressure residual
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            vo->Assemble_EOS_Residual(ii%topo->nElsX, ii/topo->nElsX, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
        }
        if(!do_vert) {
            F_exner->VertToHoriz();
            F_exner->UpdateGlobal();
        }

        if(do_vert) {
            if(itt>0) {
                rho_j->UpdateLocal();
                rho_j->HorizToVert();
                rt_j->UpdateLocal();
                rt_j->HorizToVert();
                exner_j->UpdateLocal();
                exner_j->HorizToVert();
            }

            for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
                ex = ii%topo->nElsX;
                ey = ii/topo->nElsX;

                // assemble the residual vectors
                assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                    rt_i->vz[ii], rt_j->vz[ii], F_w, F_z, G_z);

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
                //                        F_w, F_rho, F_rt, F_exner->vz[ii], d_w, d_rho, d_rt, d_exner);
                assemble_operator_schur(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_j->vz[ii], 
                                        F_w, F_rho, F_rt, F_exner->vz[ii], d_w, d_rho, d_rt, d_exner);

                VecAXPY(velz_j->vz[ii],  1.0, d_w);
                VecAXPY(rho_j->vz[ii],   1.0, d_rho);
                VecAXPY(rt_j->vz[ii],    1.0, d_rt);
                VecAXPY(exner_j->vz[ii], 1.0, d_exner);

                max_norm_exner = MaxNorm(d_exner, exner_j->vz[ii], max_norm_exner);
                max_norm_w     = MaxNorm(d_w,     velz_j->vz[ii],  max_norm_w    );
                max_norm_rho   = MaxNorm(d_rho,   rho_j->vz[ii],   max_norm_rho  );
                max_norm_rt    = MaxNorm(d_rt,    rt_j->vz[ii],    max_norm_rt   );
            }

            velz_j->VertToHoriz();
            velz_j->UpdateGlobal();
            rho_j->VertToHoriz();
            rho_j->UpdateGlobal();
            rt_j->VertToHoriz();
            rt_j->UpdateGlobal();
            exner_j->VertToHoriz();
            exner_j->UpdateGlobal();
        } else {
            if(itt>1) {
                diagHorizVort(velx_j, dudz_j);
            }

            for(int lev = 0; lev < geom->nk; lev++) {
                // velocity residual
                assemble_residual_x(lev, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
                                    velx_i[lev], velx_j[lev], rho_i->vh[lev], rho_j->vh[lev], fu, _F, _G);

                // density and density weighted potential temperature residuals
                M2->assemble(lev, SCALE, true);
                MatMult(EtoF->E21, _F, dF);
                MatMult(EtoF->E21, _G, dG);
                VecAYPX(dF, dt, rho_j->vh[lev]);
                VecAYPX(dG, dt, rt_j->vh[lev]);
                VecAXPY(dF, -1.0, rho_i->vh[lev]);
                VecAXPY(dG, -1.0, rt_i->vh[lev]);
                MatMult(M2->M, dF, frho);
                MatMult(M2->M, dG, frt);

                // delta updates  - velx is a global vector, while theta and exner are local vectors
                //assemble_schur_horiz(lev, theta_h->vl, velx_j[lev], rho_i->vl[lev], rt_i->vl[lev], exner_j->vl[lev], 
                //                     fu, frho, frt, F_exner->vh[lev], du, drho, drt, dexner);
                assemble_schur_horiz(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_j->vl[lev], 
                                     fu, frho, frt, F_exner->vh[lev], du, drho, drt, dexner);

                VecAXPY(velx_j[lev], 1.0, du);
                VecAXPY(rt_j->vh[lev], 1.0, drt);
                VecAXPY(exner_j->vh[lev], 1.0, dexner);

                // update the density
                VecCopy(rho_j->vh[lev], drho);
                MatMult(EtoF->E21, _F, rho_j->vh[lev]);
                VecAYPX(rho_j->vh[lev], -dt, rho_i->vh[lev]);
                VecAXPY(drho, -1.0, rho_j->vh[lev]);

                max_norm_exner = MaxNorm(dexner, exner_j->vh[lev], max_norm_exner);
                max_norm_u     = MaxNorm(du,     velx_j[lev],      max_norm_u    );
                max_norm_rho   = MaxNorm(drho,   rho_j->vh[lev],   max_norm_rho  );
                max_norm_rt    = MaxNorm(drt,    rt_j->vh[lev],    max_norm_rt   );
            }
        }

        MPI_Allreduce(&max_norm_u,     &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_u     = norm_x;
        MPI_Allreduce(&max_norm_w,     &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w     = norm_x;
        MPI_Allreduce(&max_norm_rho,   &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho   = norm_x;
        MPI_Allreduce(&max_norm_rt,    &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt    = norm_x;
        MPI_Allreduce(&max_norm_exner, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_exner = norm_x;

        if(max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8 && max_norm_rho < 1.0e-8 && max_norm_rt < 1.0e-8) done = true;
        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                 "\t|d_u|/|u|: "          << max_norm_u     <<
                                 "\t|d_w|/|w|: "          << max_norm_w     <<
                                 "\t|d_rho|/|rho|: "      << max_norm_rho   <<
                                 "\t|d_rt|/|rt|: "        << max_norm_rt    << endl;

        do_vert = !do_vert;
        itt++;
    } while(!done);

    // update the input/output fields
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(velx_j[lev], velx_i[lev]);
    }
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_h->vh);

    // write output
    if(save) {
        dump(velx_i, velz_i, rho_i, rt_i, exner_i, theta_h, step++);
    }

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete F_exner;
    delete theta_i;
    delete theta_h;
    for(int lev = 0; lev < geom->nk; lev++) {
        VecDestroy(&velx_j[lev]);
        VecDestroy(&dudz_i[lev]);
        VecDestroy(&dudz_j[lev]);
    }
    delete[] velx_j;
    delete[] dudz_i;
    delete[] dudz_j;
    VecDestroy(&fu);
    VecDestroy(&frho);
    VecDestroy(&frt);
    VecDestroy(&du);
    VecDestroy(&drho);
    VecDestroy(&drt);
    VecDestroy(&dexner);
    VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&dF);
    VecDestroy(&dG);
    VecDestroy(&F_w);
    VecDestroy(&F_rho);
    VecDestroy(&F_rt);
    VecDestroy(&d_w);
    VecDestroy(&d_rho);
    VecDestroy(&d_rt);
    VecDestroy(&d_exner);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
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
                                Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec fu, Vec _F, Vec _G) 
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

void Euler::coriolisMatInv(Mat A, Mat* Ainv) {
    int mi, mf, ci, nCols1, nCols2;
    const int *cols1, *cols2;
    const double *vals1;
    const double *vals2;
    double D[2][2], Dinv[2][2], detInv;
    double valsInv[4];
    int rows[2];

    MatCreate(MPI_COMM_WORLD, Ainv);
    MatSetSizes(*Ainv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(*Ainv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(*Ainv, 2, PETSC_NULL, 2, PETSC_NULL);
    MatZeroEntries(*Ainv);

    MatGetOwnershipRange(A, &mi, &mf);
    for(int mm = mi; mm < mf; mm += 2) {
        rows[0] = mm+0;
        rows[1] = mm+1;

        MatGetRow(A, mm+0, &nCols1, &cols1, &vals1);
        for(ci = 0; ci < nCols1; ci++) {
            if(cols1[ci] == mm+0) {
                D[0][0] = vals1[ci+0];
                D[0][1] = vals1[ci+1];
                break;
            }
        }
        MatRestoreRow(A, mm+0, &nCols1, &cols1, &vals1);

        MatGetRow(A, mm+1, &nCols2, &cols2, &vals2);
        for(ci = 0; ci < nCols2; ci++) {
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
        Dinv[0][1] = -detInv*D[0][1];
        Dinv[1][0] = -detInv*D[1][0];

        valsInv[0] = Dinv[0][0];
        valsInv[1] = Dinv[0][1];
        valsInv[2] = Dinv[1][0];
        valsInv[3] = Dinv[1][1];

        MatSetValues(*Ainv, 2, rows, 2, rows, valsInv, INSERT_VALUES);
    }
    MatAssemblyBegin(*Ainv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  *Ainv, MAT_FINAL_ASSEMBLY);
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

void Euler::assemble_operator(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, Mat* _PC) {
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
    //MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
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

    // [rt,rho] block
/*
    if(firstPass) MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_A_rt);
    vo->AssembleConstWithTheta(ex, ey, theta, pc_A_rt);
    MatGetOwnershipRange(pc_A_rt, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(pc_A_rt, mm, &nCols, &cols, &vals);
        ri = mm + nDofsW + nDofsRho;
        for(ci = 0; ci < nCols; ci++) {
            cols2[ci] = cols[ci] + nDofsW;
        }
        MatSetValues(*_PC, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(pc_A_rt, mm, &nCols, &cols, &vals);
    }
*/

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

void Euler::assemble_operator_schur(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, 
                                    Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner) {
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
    vo->AssembleLinearWithRT(ex, ey, exner, vo->VA, true);
    vo->AssembleLinearWithRhoInv(ex, ey, rho, vo->VA_inv);
    MatMatMult(vo->VA, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(pc_V0_invV0_rt, pc_G, reuse, PETSC_DEFAULT, &pc_A_u);

    //MatMatMult(vo->VA, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    //MatMatMult(pc_V0_invV0_rt, vo->V01, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    //vo->AssembleConst(ex, ey, vo->VB);
    //MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT_VB_pi);
    //vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    //MatMatMult(pc_V0_invV0_rt_DT_VB_pi, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT_VB_pi_VB_inv);
    //vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    //MatMatMult(pc_V0_invV0_rt_DT_VB_pi_VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_A_u);
    
    //vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    //vo->AssembleLinearWithRhoInv(ex, ey, rho, vo->VA_inv);
    //MatMatMult(vo->VA, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    //MatMatMult(pc_V0_invV0_rt, vo->V01, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt_DT);
    //vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    //MatMatMult(pc_V0_invV0_rt_DT, vo->VB, reuse, PETSC_DEFAULT, &pc_A_u);

    MatScale(pc_A_u, RD/CV);

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
    //vo->AssembleConstWithRho(ex, ey, rt, vo->VB);
    //MatMatMult(vo->VB, vo->V10, reuse, PETSC_DEFAULT, &pc_D_rt);
    MatScale(pc_D_rt, 0.5*dt);

    // [rt,rt] block
    vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);                     // -- 
    MatMatMult(vo->VBA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &pc_M_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatAYPX(pc_M_rt, 0.5*dt, vo->VB, DIFFERENT_NONZERO_PATTERN);        // --

    // [rt,rho] block
    if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_A_rt);
    vo->AssembleConstWithTheta(ex, ey, theta, pc_A_rt);
    //vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    //MatMatMult(pc_M_rt, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_M_rt_VB_inv);
    //vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    //MatMatMult(pc_M_rt_VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_A_rt);

    // [exner,rt] block
    if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_N_rt_inv);
    vo->Assemble_EOS_BlockInv(ex, ey, rt, NULL, pc_N_rt_inv);
    MatScale(pc_N_rt_inv, -1.0*CV/RD);

    // [exner,exner] block
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_N_exner);

    // 1. density corrections
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(pc_A_u, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_A_u_VB_inv);
    MatMatMult(pc_A_rt, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_A_rt_VB_inv);

    MatMatMult(pc_A_u_VB_inv, pc_D_rho, reuse, PETSC_DEFAULT, &pc_M_u);
    MatMatMult(pc_A_rt_VB_inv, pc_D_rho, reuse, PETSC_DEFAULT, &pc_A_rt_VB_inv_D_rho);

    vo->AssembleLinear(ex, ey, vo->VA);
    MatAYPX(pc_M_u, -1.0, vo->VA, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(pc_D_rt, -1.0, pc_A_rt_VB_inv_D_rho, DIFFERENT_NONZERO_PATTERN);

    MatMult(pc_A_u_VB_inv, F_rho, _tmpA1);
    VecAXPY(F_w, -1.0, _tmpA1);
    
    MatMult(pc_A_rt_VB_inv, F_rho, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);

    // 2. density weighted potential temperature correction
    MatMatMult(vo->VB, pc_N_rt_inv, reuse, PETSC_DEFAULT, &pc_VB_N_rt_inv);
    //MatMatMult(pc_M_rt, pc_N_rt_inv, reuse, PETSC_DEFAULT, &pc_VB_N_rt_inv);
    MatMatMult(pc_VB_N_rt_inv, pc_N_exner, reuse, PETSC_DEFAULT, &pc_N_exner_2);
    MatMult(pc_VB_N_rt_inv, F_exner, _tmpB1);
    VecAXPY(F_rt, -1.0, _tmpB1);

    // 3. schur complement solve for exner pressure
    if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk-1)*n2, (geom->nk-1)*n2, 1, NULL, &pc_M_u_inv);
    MatGetDiagonal(pc_M_u, _tmpA1);
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
        PC pc;
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
    VecScale(dw, -1.0);

#ifndef EXPLICIT_THETA_UPDATE
    // -- density weighted potential temperature
    MatMult(pc_N_exner, dexner, _tmpB1);
    VecAXPY(_tmpB1, 1.0, F_exner);
    MatMult(pc_N_rt_inv, _tmpB1, drt);
    VecScale(drt, -1.0);
#endif

#ifndef EXPLICIT_RHO_UPDATE
    // -- density
    MatMult(pc_D_rho, dw, _tmpB1);
    VecAXPY(_tmpB1, 1.0, F_rho);
    MatMult(vo->VB_inv, _tmpB1, drho);
    VecScale(drho, -1.0);
#endif
}

void Euler::assemble_schur_horiz(int lev, Vec* theta, Vec velx, Vec rho, Vec rt, Vec exner, 
                                 Vec F_u, Vec F_rho, Vec F_rt, Vec F_exner, Vec du, Vec drho, Vec drt, Vec dexner) {
    bool build_ksp = (!_PCx) ? true : false;
    MatReuse reuse = (!_PCx) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    Vec wg, wl, theta_k, diag_g, ones_g, h_tmp;
    Mat Mu_inv, M1_inv, Mu_prime, M1_OP;
    WmatInv* M2inv = new WmatInv(topo, geom, edge);
    WhmatInv* M2_rho_inv = new WhmatInv(topo, geom, edge);
    N_rt_Inv* M2_pi_inv = new N_rt_Inv(topo, geom, edge);
    N_rt_Inv* M2_rt_inv = new N_rt_Inv(topo, geom, edge);
    PC pc;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &diag_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ones_g);

    MatCreate(MPI_COMM_WORLD, &M1_inv);
    MatSetSizes(M1_inv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M1_inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M1_inv, 1, PETSC_NULL, 1, PETSC_NULL);

    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE, true);
    M2->assemble(lev, SCALE, true);
    M2inv->assemble(lev, SCALE);

    MatGetDiagonal(M1->M, diag_g);
    VecSet(ones_g, 1.0);
    VecPointwiseDivide(diag_g, ones_g, diag_g);
    MatZeroEntries(M1_inv);
    MatDiagonalSet(M1_inv, diag_g, INSERT_VALUES);

    // laplacian
    {
        Vec m0_inv, ones_0;
        Mat M2D, DTM2D, LAP_1, CTM1, M0_invCTM1, M0_inv, VISC, VISC2;

        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &m0_inv);
        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &ones_0);
        VecSet(ones_0, 1.0);
        VecPointwiseDivide(m0_inv, ones_0, m0->vg);

        MatCreate(MPI_COMM_WORLD, &M0_inv);
        MatSetSizes(M0_inv, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
        MatSetType(M0_inv, MATMPIAIJ);
        MatMPIAIJSetPreallocation(M0_inv, 1, PETSC_NULL, 1, PETSC_NULL);
        MatDiagonalSet(M0_inv, m0_inv, INSERT_VALUES);

        MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M2D);
        MatMatMult(EtoF->E12, M2D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DTM2D);
        MatMatMult(M1_inv, DTM2D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &LAP_1);

        MatMatMult(NtoE->E01, M1->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CTM1);
        MatMatMult(M0_inv, CTM1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M0_invCTM1);
        MatMatMult(NtoE->E10, M0_invCTM1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC);

        MatAssemblyBegin(LAP_1, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  LAP_1, MAT_FINAL_ASSEMBLY);
        MatAXPY(VISC, 1.0, LAP_1, DIFFERENT_NONZERO_PATTERN);
        MatScale(VISC, del2);
        MatMatMult(VISC, VISC, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC2);
        MatMatMult(M1->M, VISC2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M1_OP);  
        MatScale(M1_OP, 0.5*dt);

        VecDestroy(&m0_inv);
        VecDestroy(&ones_0);
        MatDestroy(&M2D);
        MatDestroy(&DTM2D);
        MatDestroy(&LAP_1);
        MatDestroy(&CTM1);
        MatDestroy(&M0_invCTM1);
        MatDestroy(&M0_inv);
        MatDestroy(&VISC);
        MatDestroy(&VISC2);
    }

    // [u,u] block
    curl(false, velx, &wg, lev, true);
    VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
    R->assemble(wl, lev, SCALE);
    MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);

    // [u,exner] block
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta[lev+0]);
    VecAXPY(theta_k, 0.5, theta[lev+1]);
    F->assemble(theta_k, lev, false, SCALE);
    MatMatMult(M1_inv, EtoF->E12, reuse, PETSC_DEFAULT, &pcx_M1invD12);
    MatMatMult(pcx_M1invD12, M2->M, reuse, PETSC_DEFAULT, &pcx_M1invD12M2);
    MatMatMult(F->M, pcx_M1invD12M2, reuse, PETSC_DEFAULT, &pcx_G);
    MatScale(pcx_G, 0.5*dt);

    // [rt,u] block
    F->assemble(rt, lev, true, SCALE);
    MatMatMult(M1_inv, F->M, reuse, PETSC_DEFAULT, &pcx_M1invF_rt);
    MatMatMult(EtoF->E21, pcx_M1invF_rt, reuse, PETSC_DEFAULT, &pcx_D21M1invF_rt);
    MatMatMult(M2->M, pcx_D21M1invF_rt, reuse, PETSC_DEFAULT, &pcx_D);
    MatScale(pcx_D, 0.5*dt);

    // [rho,u] block
    F->assemble(rho, lev, true, SCALE);
    MatMatMult(M1_inv, F->M, reuse, PETSC_DEFAULT, &pcx_M1invF_rho);
    MatMatMult(EtoF->E21, pcx_M1invF_rho, reuse, PETSC_DEFAULT, &pcx_D21M1invF_rho);
    MatMatMult(M2->M, pcx_D21M1invF_rho, reuse, PETSC_DEFAULT, &pcx_D_rho);
    MatScale(pcx_D_rho, 0.5*dt);

    // density corrections
    T->assemble(theta_k, lev, SCALE, false);
    MatMatMult(T->M, M2inv->M, reuse, PETSC_DEFAULT, &pcx_A_rtM2_inv);
    MatMatMult(pcx_A_rtM2_inv, pcx_D_rho, reuse, PETSC_DEFAULT, &pcx_D_prime);
    MatAXPY(pcx_D, -1.0, pcx_D_prime, SAME_NONZERO_PATTERN);
    MatMult(pcx_A_rtM2_inv, F_rho, h_tmp);
    VecAXPY(F_rt, -1.0, h_tmp);

    MatGetDiagonal(F->M, diag_g);
    VecPointwiseDivide(diag_g, ones_g, diag_g);
    MatZeroEntries(M1_inv);
    MatDiagonalSet(M1_inv, diag_g, INSERT_VALUES);

    F->assemble(exner, lev, true, SCALE);
    MatMatMult(F->M, M1_inv, reuse, PETSC_DEFAULT, &pcx_M1_exner_M1_inv);
    MatMatMult(pcx_M1_exner_M1_inv, pcx_G, reuse, PETSC_DEFAULT, &pcx_Au);
    MatScale(pcx_Au, RD/CV);
    MatMatMult(pcx_Au, M2inv->M, reuse, PETSC_DEFAULT, &pcx_Au_M2_inv);
    MatMatMult(pcx_Au_M2_inv, pcx_D_rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mu_prime); // invalid read on the second pass
    MatAYPX(Mu_prime, 1.0, R->M, DIFFERENT_NONZERO_PATTERN);
    coriolisMatInv(Mu_prime, &Mu_inv);

    MatAXPY(M1_OP, 1.0, Mu_prime, DIFFERENT_NONZERO_PATTERN);
    MatMult(pcx_Au_M2_inv, F_rho, diag_g);
    VecAXPY(F_u, -1.0, diag_g);

    // setup the corrected velocity mass matrix solver
    KSPCreate(MPI_COMM_WORLD, &ksp_u);
    KSPSetOperators(ksp_u, M1_OP, M1_OP);
    KSPSetTolerances(ksp_u, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_u, KSPGMRES);
    KSPGetPC(ksp_u, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp_u, "ksp1_");
    KSPSetFromOptions(ksp_u);

    // build the preconditioner
    MatMatMult(pcx_D, Mu_inv, reuse, PETSC_DEFAULT, &pcx_D_Mu_inv);
    MatMatMult(pcx_D_Mu_inv, pcx_G, reuse, PETSC_DEFAULT, &pcx_LAP);

    M2_rt_inv->assemble(rt, lev, SCALE, true);
    MatScale(M2_rt_inv->M, -1.0*CV/RD);
    MatMatMult(M2->M, M2_rt_inv->M, reuse, PETSC_DEFAULT, &pcx_M2N_rt_inv);
    M2_pi_inv->assemble(exner, lev, SCALE, false);
    MatMatMult(pcx_M2N_rt_inv, M2_pi_inv->M, reuse, PETSC_DEFAULT, &pcx_M2N_rt_invN_pi);

    MatAssemblyBegin(pcx_M2N_rt_invN_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pcx_M2N_rt_invN_pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(pcx_LAP, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (pcx_LAP, MAT_FINAL_ASSEMBLY);

    if(!_PCx) MatMatMult(pcx_D, pcx_G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &_PCx);
    MatZeroEntries(_PCx);
    MatAXPY(_PCx, -1.0, pcx_LAP, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(_PCx, -1.0, pcx_M2N_rt_invN_pi, DIFFERENT_NONZERO_PATTERN);

    // update the rhs
    MatMult(pcx_M2N_rt_inv, F_exner, h_tmp);
    VecAXPY(F_rt, -1.0, h_tmp);
    MatMult(pcx_D_Mu_inv, F_u, h_tmp);
    //KSPSolve(ksp_u, F_u, ones_g);
    //MatMult(pcx_D, ones_g, h_tmp);
    VecAXPY(F_rt, -1.0, h_tmp);
    VecScale(F_rt, -1.0);

    // exner pressure solve
    if(build_ksp) {
        KSPCreate(MPI_COMM_WORLD, &ksp_exner_x);
        KSPSetOperators(ksp_exner_x, _PCx, _PCx);
        KSPSetTolerances(ksp_exner_x, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
        KSPSetType(ksp_exner_x, KSPGMRES);
        KSPGetPC(ksp_exner_x, &pc);
        PCSetType(pc, PCBJACOBI);
        PCBJacobiSetTotalBlocks(pc, 6*topo->nElsX*topo->nElsX, NULL);
        KSPSetOptionsPrefix(ksp_exner_x, "ksp_exner_x_");
        KSPSetFromOptions(ksp_exner_x);
    }
    KSPSolve(ksp_exner_x, F_rt, dexner);

    // velocity update
    MatMult(pcx_G, dexner, ones_g);
    VecAXPY(F_u, 1.0, ones_g);

    //MatMult(Mu_inv, F_u, du);
    // actual solve for delta u update improves convergence
    KSPSolve(ksp_u, F_u, du);
    KSPDestroy(&ksp_u);

    VecScale(du, -1.0);

    // density weighted potential temperature update
    MatMult(M2_pi_inv->M, dexner, h_tmp);
    VecAXPY(F_exner, 1.0, h_tmp);
    MatMult(M2_rt_inv->M, F_exner, drt);
    VecScale(drt, -1.0);

    // note: do the density update outside!

    VecDestroy(&wl);
    VecDestroy(&wg);
    VecDestroy(&theta_k);
    VecDestroy(&h_tmp);
    VecDestroy(&diag_g);
    VecDestroy(&ones_g);
    MatDestroy(&Mu_inv);
    MatDestroy(&M1_inv);
    MatDestroy(&Mu_prime);
    MatDestroy(&M1_OP);
    delete M2inv;
    delete M2_pi_inv;
    delete M2_rt_inv;
    delete M2_rho_inv;
}

void Euler::assemble_schur_3d(L2Vecs* theta, Vec* velx, Vec* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, 
                              Vec* F_u, Vec* F_w, L2Vecs* F_rho, L2Vecs* F_rt, L2Vecs* F_exner,
                              Vec* du, Vec* dw, L2Vecs* drho, L2Vecs* drt, L2Vecs* dexner) {
    int n2 = topo->elOrd*topo->elOrd, ex, ey;
    bool build_ksp;
    MatReuse reuse;
    Vec wl, wg, theta_k, h_tmp, diag_g, ones_g, m0_inv, ones_0;
    Mat M2D, DTM2D, LAP_1, CTM1, M0_invCTM1, M0_inv, M1_inv, VISC, VISC2, M1_OP, Mu_prime, Mu_inv;
    PC pc;
    KSP ksp_u;
    WmatInv* M2inv = new WmatInv(topo, geom, edge);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &diag_g);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ones_g);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &m0_inv);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &ones_0);

    MatCreate(MPI_COMM_WORLD, &M0_inv);
    MatSetSizes(M0_inv, topo->n0l, topo->n0l, topo->nDofs0G, topo->nDofs0G);
    MatSetType(M0_inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M0_inv, 1, PETSC_NULL, 1, PETSC_NULL);

    MatCreate(MPI_COMM_WORLD, &M1_inv);
    MatSetSizes(M1_inv, topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
    MatSetType(M1_inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M1_inv, 1, PETSC_NULL, 1, PETSC_NULL);

    VecZeroEntries(schur->b);
    MatZeroEntries(schur->M);

    // assemble in the vertical terms
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        build_ksp = (!_PCz) ? true : false;
        reuse = (!_PCz) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

        ex = ei%topo->nElsX;
        ey = ei/topo->nElsX;

        // [u,exner] block
        vo->AssembleConst(ex, ey, vo->VB);
        MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
        vo->AssembleLinearInv(ex, ey, vo->VA_inv);
        MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
        vo->AssembleLinearWithTheta(ex, ey, theta->vz[ei], vo->VA);
        MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &pc_G);
        MatScale(pc_G, 0.5*dt);

        // [u,rho] block
        vo->AssembleLinearWithRT(ex, ey, exner->vz[ei], vo->VA, true);
        vo->AssembleLinearWithRhoInv(ex, ey, rho->vz[ei], vo->VA_inv);
        MatMatMult(vo->VA, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
        MatMatMult(pc_V0_invV0_rt, pc_G, reuse, PETSC_DEFAULT, &pc_A_u);
        MatScale(pc_A_u, RD/CV);

        // [rho,u] block
        vo->AssembleLinearWithRT(ex, ey, rho->vz[ei], vo->VA, true);
        vo->AssembleLinearInv(ex, ey, vo->VA_inv);
        MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
        MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
        vo->AssembleConst(ex, ey, vo->VB);
        MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_D_rho);
        MatScale(pc_D_rho, 0.5*dt);

        // [rt,u] block
        vo->AssembleLinearWithRT(ex, ey, rt->vz[ei], vo->VA, true);
        vo->AssembleLinearInv(ex, ey, vo->VA_inv);
        MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
        MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
        MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_D_rt);
        MatScale(pc_D_rt, 0.5*dt);

        // [rt,rt] block
        vo->AssembleConLinWithW(ex, ey, velz[ei], vo->VBA);
        MatMatMult(vo->VBA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &pc_M_rt);
        vo->AssembleConst(ex, ey, vo->VB);
        MatAYPX(pc_M_rt, 0.5*dt, vo->VB, DIFFERENT_NONZERO_PATTERN);

        // [rt,rho] block
        if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_A_rt);
        vo->AssembleConstWithTheta(ex, ey, theta->vz[ei], pc_A_rt);

        // [exner,rt] block
        if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_N_rt_inv);
        vo->Assemble_EOS_BlockInv(ex, ey, rt->vz[ei], NULL, pc_N_rt_inv);
        MatScale(pc_N_rt_inv, -1.0*CV/RD);

        // [exner,exner] block
        vo->AssembleConstWithRhoInv(ex, ey, exner->vz[ei], vo->VB_inv);
        MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
        MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_N_exner);

        // 1. density corrections
        vo->AssembleConstInv(ex, ey, vo->VB_inv);
        MatMatMult(pc_A_u, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_A_u_VB_inv);
        MatMatMult(pc_A_rt, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_A_rt_VB_inv);

        MatMatMult(pc_A_u_VB_inv, pc_D_rho, reuse, PETSC_DEFAULT, &pc_M_u);
        MatMatMult(pc_A_rt_VB_inv, pc_D_rho, reuse, PETSC_DEFAULT, &pc_A_rt_VB_inv_D_rho);

        vo->AssembleLinear(ex, ey, vo->VA);
        MatAYPX(pc_M_u, -1.0, vo->VA, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(pc_D_rt, -1.0, pc_A_rt_VB_inv_D_rho, DIFFERENT_NONZERO_PATTERN);

        MatMult(pc_A_u_VB_inv, F_rho->vz[ei], _tmpA1);
        VecAXPY(F_w[ei], -1.0, _tmpA1);
    
        MatMult(pc_A_rt_VB_inv, F_rho->vz[ei], _tmpB1);
        VecAXPY(F_rt->vz[ei], -1.0, _tmpB1);

        // 2. density weighted potential temperature correction
        MatMatMult(vo->VB, pc_N_rt_inv, reuse, PETSC_DEFAULT, &pc_VB_N_rt_inv);
        MatMatMult(pc_VB_N_rt_inv, pc_N_exner, reuse, PETSC_DEFAULT, &pc_N_exner_2);
        MatMult(pc_VB_N_rt_inv, F_exner->vz[ei], _tmpB1);
        VecAXPY(F_rt->vz[ei], -1.0, _tmpB1);

        // 3. schur complement solve for exner pressure
        if(build_ksp) MatCreateSeqAIJ(MPI_COMM_SELF, (geom->nk-1)*n2, (geom->nk-1)*n2, 1, NULL, &pc_M_u_inv);
        MatGetDiagonal(pc_M_u, _tmpA1);
        VecSet(_tmpA2, 1.0);
        VecPointwiseDivide(_tmpA2, _tmpA2, _tmpA1);
        MatZeroEntries(pc_M_u_inv);
        MatDiagonalSet(pc_M_u_inv, _tmpA2, INSERT_VALUES);

        MatMatMult(pc_D_rt, pc_M_u_inv, reuse, PETSC_DEFAULT, &pc_D_rt_M_u_inv);
        MatMatMult(pc_D_rt_M_u_inv, pc_G, reuse, PETSC_DEFAULT, &_PCz);
        MatAXPY(_PCz, 1.0, pc_N_exner_2, DIFFERENT_NONZERO_PATTERN);
        MatScale(_PCz, -1.0);

        MatMult(pc_D_rt_M_u_inv, F_w[ei], _tmpB1);
        VecAXPY(F_rt->vz[ei], -1.0, _tmpB1);
        VecScale(F_rt->vz[ei], -1.0);

        // add to the global system
        schur->AddFromVertMat(ei, _PCz);
    }
    F_rt->VertToHoriz();
    F_rt->UpdateGlobal();

    // assemble in the horizontal terms
    for(int kk = 0; kk < geom->nk; kk++) {
        reuse = (!_PCx) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

        M1->assemble(kk, SCALE, true);
        M2->assemble(kk, SCALE, true);
        M2inv->assemble(kk, SCALE);

        MatGetDiagonal(M1->M, diag_g);
        VecSet(ones_g, 1.0);
        VecPointwiseDivide(diag_g, ones_g, diag_g);
        MatZeroEntries(M1_inv);
        MatDiagonalSet(M1_inv, diag_g, INSERT_VALUES);

        // [u,u] block
        curl(false, velx[kk], &wg, kk, true);
        VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
        R->assemble(wl, kk, SCALE);
        MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);
        VecDestroy(&wg);

        // [u,exner] block
        VecZeroEntries(theta_k);
        VecAXPY(theta_k, 0.5, theta->vl[kk+0]);
        VecAXPY(theta_k, 0.5, theta->vl[kk+1]);
        F->assemble(theta_k, kk, false, SCALE);
        MatMatMult(M1_inv, EtoF->E12, reuse, PETSC_DEFAULT, &pcx_M1invD12);
        MatMatMult(pcx_M1invD12, M2->M, reuse, PETSC_DEFAULT, &pcx_M1invD12M2);
        MatMatMult(F->M, pcx_M1invD12M2, reuse, PETSC_DEFAULT, &pcx_G);
        MatScale(pcx_G, 0.5*dt);

        // [rt,u] block
        F->assemble(rt->vl[kk], kk, true, SCALE);
        MatMatMult(M1_inv, F->M, reuse, PETSC_DEFAULT, &pcx_M1invF_rt);
        MatMatMult(EtoF->E21, pcx_M1invF_rt, reuse, PETSC_DEFAULT, &pcx_D21M1invF_rt);
        MatMatMult(M2->M, pcx_D21M1invF_rt, reuse, PETSC_DEFAULT, &pcx_D);
        MatScale(pcx_D, 0.5*dt);

        // [rho,u] block
        F->assemble(rho->vl[kk], kk, true, SCALE);
        MatMatMult(M1_inv, F->M, reuse, PETSC_DEFAULT, &pcx_M1invF_rho);
        MatMatMult(EtoF->E21, pcx_M1invF_rho, reuse, PETSC_DEFAULT, &pcx_D21M1invF_rho);
        MatMatMult(M2->M, pcx_D21M1invF_rho, reuse, PETSC_DEFAULT, &pcx_D_rho);
        MatScale(pcx_D_rho, 0.5*dt);

        // density corrections
        T->assemble(theta_k, kk, SCALE, false);
        MatMatMult(T->M, M2inv->M, reuse, PETSC_DEFAULT, &pcx_A_rtM2_inv);
        MatMatMult(pcx_A_rtM2_inv, pcx_D_rho, reuse, PETSC_DEFAULT, &pcx_D_prime);
        MatAXPY(pcx_D, -1.0, pcx_D_prime, SAME_NONZERO_PATTERN);

        MatGetDiagonal(F->M, diag_g);
        VecPointwiseDivide(diag_g, ones_g, diag_g);
        MatZeroEntries(M1_inv);
        MatDiagonalSet(M1_inv, diag_g, INSERT_VALUES);

        F->assemble(exner->vl[kk], kk, true, SCALE);
        MatMatMult(F->M, M1_inv, reuse, PETSC_DEFAULT, &pcx_M1_exner_M1_inv);
        MatMatMult(pcx_M1_exner_M1_inv, pcx_G, reuse, PETSC_DEFAULT, &pcx_Au);
        MatScale(pcx_Au, RD/CV);
        MatMatMult(pcx_Au, M2inv->M, reuse, PETSC_DEFAULT, &pcx_Au_M2_inv);
        MatMatMult(pcx_Au_M2_inv, pcx_D_rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mu_prime);
        MatAYPX(Mu_prime, 1.0, R->M, DIFFERENT_NONZERO_PATTERN);
        coriolisMatInv(Mu_prime, &Mu_inv);

        MatMult(pcx_Au_M2_inv, F_rho->vh[kk], diag_g);
        VecAXPY(F_u[kk], -1.0, diag_g);

        // build the preconditioner
        MatMatMult(pcx_D, Mu_inv, reuse, PETSC_DEFAULT, &pcx_D_Mu_inv);
        MatMatMult(pcx_D_Mu_inv, pcx_G, reuse, PETSC_DEFAULT, &_PCx);
        MatScale(_PCx, -1.0);

        // update the rhs
        MatMult(pcx_D_Mu_inv, F_u[kk], h_tmp);
        VecAXPY(F_rt->vh[kk], -1.0, h_tmp);
        VecScale(F_rt->vh[kk], -1.0);

        // add to the global system
        schur->AddFromHorizMat(kk, _PCx);

        MatDestroy(&Mu_prime);
        MatDestroy(&Mu_inv);
    }
    F_rt->UpdateLocal();
    schur->RepackFromHoriz(F_rt->vl, schur->b);

    MatAssemblyBegin(schur->M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  schur->M, MAT_FINAL_ASSEMBLY);

    // update the exner pressure
    KSPSolve(schur->ksp, schur->b, schur->x);
    schur->UnpackToHoriz(schur->x, dexner->vl);
    dexner->HorizToVert();
    dexner->UpdateGlobal();

    // update the horizontal velocity
    for(int kk = 0; kk < geom->nk; kk++) {
        m0->assemble(kk, SCALE);
        M1->assemble(kk, SCALE, true);
        M2->assemble(kk, SCALE, true);
        M2inv->assemble(kk, SCALE);

        MatGetDiagonal(M1->M, diag_g);
        VecSet(ones_g, 1.0);
        VecPointwiseDivide(diag_g, ones_g, diag_g);
        MatZeroEntries(M1_inv);
        MatDiagonalSet(M1_inv, diag_g, INSERT_VALUES);

        // [u,u] block
        curl(false, velx[kk], &wg, kk, true);
        VecScatterBegin(topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_0, wg, wl, INSERT_VALUES, SCATTER_FORWARD);
        R->assemble(wl, kk, SCALE);
        MatAYPX(R->M, 0.5*dt, M1->M, DIFFERENT_NONZERO_PATTERN);
        VecDestroy(&wg);

        // [u,exner] block
        VecZeroEntries(theta_k);
        VecAXPY(theta_k, 0.5, theta->vl[kk+0]);
        VecAXPY(theta_k, 0.5, theta->vl[kk+1]);
        F->assemble(theta_k, kk, false, SCALE);
        MatMatMult(M1_inv, EtoF->E12, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M1invD12);
        MatMatMult(pcx_M1invD12, M2->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M1invD12M2);
        MatMatMult(F->M, pcx_M1invD12M2, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_G);
        MatScale(pcx_G, 0.5*dt);

        // [rho,u] block
        F->assemble(rho->vl[kk], kk, true, SCALE);
        MatMatMult(M1_inv, F->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M1invF_rho);
        MatMatMult(EtoF->E21, pcx_M1invF_rho, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_D21M1invF_rho);
        MatMatMult(M2->M, pcx_D21M1invF_rho, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_D_rho);
        MatScale(pcx_D_rho, 0.5*dt);

        // density corrections
        MatGetDiagonal(F->M, diag_g);
        VecPointwiseDivide(diag_g, ones_g, diag_g);
        MatZeroEntries(M1_inv);
        MatDiagonalSet(M1_inv, diag_g, INSERT_VALUES);

        F->assemble(exner->vl[kk], kk, true, SCALE);
        MatMatMult(F->M, M1_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_M1_exner_M1_inv);
        MatMatMult(pcx_M1_exner_M1_inv, pcx_G, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_Au);
        MatScale(pcx_Au, RD/CV);
        MatMatMult(pcx_Au, M2inv->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pcx_Au_M2_inv);
        MatMatMult(pcx_Au_M2_inv, pcx_D_rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mu_prime);
        MatAYPX(Mu_prime, 1.0, R->M, DIFFERENT_NONZERO_PATTERN);

        // viscosity correction
        VecSet(ones_0, 1.0);
        VecPointwiseDivide(m0_inv, ones_0, m0->vg);
        MatDiagonalSet(M0_inv, m0_inv, INSERT_VALUES);

        MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M2D);
        MatMatMult(EtoF->E12, M2D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DTM2D);
        MatMatMult(M1_inv, DTM2D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &LAP_1);

        MatMatMult(NtoE->E01, M1->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CTM1);
        MatMatMult(M0_inv, CTM1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M0_invCTM1);
        MatMatMult(NtoE->E10, M0_invCTM1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC);

        MatAssemblyBegin(LAP_1, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  LAP_1, MAT_FINAL_ASSEMBLY);
        MatAXPY(VISC, 1.0, LAP_1, DIFFERENT_NONZERO_PATTERN);
        MatScale(VISC, del2);
        MatMatMult(VISC, VISC, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC2);
        MatMatMult(M1->M, VISC2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &M1_OP);  
        MatAYPX(M1_OP, 0.5*dt, Mu_prime, DIFFERENT_NONZERO_PATTERN);

        MatMult(pcx_G, dexner->vh[kk], ones_g);
        VecAXPY(F_u[kk], 1.0, ones_g);
        VecScale(F_u[kk], -1.0);

        KSPCreate(MPI_COMM_WORLD, &ksp_u);
        KSPSetOperators(ksp_u, M1_OP, M1_OP);
        KSPSetTolerances(ksp_u, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
        KSPSetType(ksp_u, KSPGMRES);
        KSPGetPC(ksp_u, &pc);
        PCSetType(pc, PCBJACOBI);
        PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
        KSPSetOptionsPrefix(ksp_u, "ksp1_");
        KSPSetFromOptions(ksp_u);
        KSPSolve(ksp_u, F_u[kk], du[kk]);
        KSPDestroy(&ksp_u);

        MatDestroy(&M2D);
        MatDestroy(&DTM2D);
        MatDestroy(&LAP_1);
        MatDestroy(&CTM1);
        MatDestroy(&M0_invCTM1);
        MatDestroy(&VISC);
        MatDestroy(&VISC2);
        MatDestroy(&M1_OP);
        MatDestroy(&Mu_prime);
    }

    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        ex = ei%topo->nElsX;
        ey = ei/topo->nElsX;

        // [u,exner] block
        vo->AssembleConst(ex, ey, vo->VB);
        MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DTV1);
        vo->AssembleLinearInv(ex, ey, vo->VA_inv);
        MatMatMult(vo->VA_inv, pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invDTV1);
        vo->AssembleLinearWithTheta(ex, ey, theta->vz[ei], vo->VA);
        MatMatMult(vo->VA, pc_V0_invDTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_G);
        MatScale(pc_G, 0.5*dt);

        // [u,rho] block
        vo->AssembleLinearWithRT(ex, ey, exner->vz[ei], vo->VA, true);
        vo->AssembleLinearWithRhoInv(ex, ey, rho->vz[ei], vo->VA_inv);
        MatMatMult(vo->VA, vo->VA_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
        MatMatMult(pc_V0_invV0_rt, pc_G, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_A_u);
        MatScale(pc_A_u, RD/CV);

        // [rho,u] block
        vo->AssembleLinearWithRT(ex, ey, rho->vz[ei], vo->VA, true);
        vo->AssembleLinearInv(ex, ey, vo->VA_inv);
        MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
        MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
        vo->AssembleConst(ex, ey, vo->VB);
        MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_D_rho);
        MatScale(pc_D_rho, 0.5*dt);

        // [exner,rt] block
        vo->Assemble_EOS_BlockInv(ex, ey, rt->vz[ei], NULL, pc_N_rt_inv);
        MatScale(pc_N_rt_inv, -1.0*CV/RD);

        // [exner,exner] block
        vo->AssembleConstWithRhoInv(ex, ey, exner->vz[ei], vo->VB_inv);
        MatMatMult(vo->VB_inv, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
        MatMatMult(vo->VB, pc_VB_rt_invVB_pi, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_N_exner);

        vo->AssembleConstInv(ex, ey, vo->VB_inv);
        MatMatMult(pc_A_u, vo->VB_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_A_u_VB_inv);
        MatMatMult(pc_A_u_VB_inv, pc_D_rho, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_M_u);

        vo->AssembleLinear(ex, ey, vo->VA);
        MatAYPX(pc_M_u, -1.0, vo->VA, DIFFERENT_NONZERO_PATTERN);

        MatGetDiagonal(pc_M_u, _tmpA1);
        VecSet(_tmpA2, 1.0);
        VecPointwiseDivide(_tmpA2, _tmpA2, _tmpA1);
        MatZeroEntries(pc_M_u_inv);
        MatDiagonalSet(pc_M_u_inv, _tmpA2, INSERT_VALUES);

        // update the vertical velocity
        MatMult(pc_G, dexner->vz[ei], _tmpA1);
        VecAXPY(_tmpA1, 1.0, F_w[ei]);
        MatMult(pc_M_u_inv, _tmpA1, dw[ei]);
        VecScale(dw[ei], -1.0);

        // update the density
/*
        MatMult(pc_D_rho, dw[ei], _tmpB1);
        VecAXPY(_tmpB1, 1.0, F_rho->vz[ei]);
        MatMult(vo->VB_inv, _tmpB1, drho->vz[ei]);
        VecScale(drho->vz[ei], -1.0); // TODO: horiztonal flux update
*/

        // update the density weighted potential temperature
        MatMult(pc_N_exner, dexner->vz[ei], _tmpB1);
        VecAXPY(_tmpB1, 1.0, F_exner->vz[ei]);
        MatMult(pc_N_rt_inv, _tmpB1, drt->vz[ei]);
        VecScale(drt->vz[ei], -1.0);
    }
    drt->VertToHoriz();
    drt->UpdateGlobal();

    VecDestroy(&wl);
    VecDestroy(&theta_k);
    VecDestroy(&h_tmp);
    VecDestroy(&diag_g);
    VecDestroy(&ones_g);
    VecDestroy(&m0_inv);
    VecDestroy(&ones_0);
    MatDestroy(&M0_inv);
    MatDestroy(&M1_inv);
    delete M2inv;
}

void Euler::solve_schur_3d(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_w, max_norm_exner, norm_x;
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* F_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* F_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_exner = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* d_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_exner = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* l2_tmp  = new L2Vecs(geom->nk, topo, geom);
    Vec* velx_j = new Vec[geom->nk];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    Vec* F_u    = new Vec[geom->nk];
    Vec* d_u    = new Vec[geom->nk];
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp;

    for(int lev = 0; lev < geom->nk; lev++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[lev]);
        VecCopy(velx_i[lev], velx_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &F_u[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &d_u[lev]);
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dG);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);

    velz_i->UpdateLocal();  velz_i->HorizToVert();
    rho_i->UpdateLocal();   rho_i->HorizToVert();
    rt_i->UpdateLocal();    rt_i->HorizToVert();
    exner_i->UpdateLocal(); exner_i->HorizToVert();
    velz_j->CopyFromVert(velz_i->vz);   velz_j->VertToHoriz();  velz_j->UpdateGlobal();
    rho_j->CopyFromVert(rho_i->vz);     rho_j->VertToHoriz();   rho_j->UpdateGlobal();
    rt_j->CopyFromVert(rho_i->vz);      rt_j->VertToHoriz();    rt_j->UpdateGlobal();
    exner_j->CopyFromVert(exner_i->vz); exner_j->VertToHoriz(); exner_j->UpdateGlobal();
    exner_h->CopyFromVert(exner_i->vz); exner_h->VertToHoriz(); exner_h->UpdateGlobal();

    // diagnose the vorticity terms
    diagHorizVort(velx_i, dudz_i);
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i[lev], dudz_j[lev]);
    }

    diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();

    do {
        max_norm_u = max_norm_w = max_norm_exner = 0.0;

        if(itt) {
            rho_j->UpdateLocal();
            rho_j->HorizToVert();
            rt_j->UpdateLocal();
            rt_j->HorizToVert();
            exner_j->HorizToVert(); // local update already done
            for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
                VecZeroEntries(exner_h->vz[ii]);
                VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
                VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);
            }
            diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
            for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
                VecScale(theta_h->vz[ii], 0.5);
                VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
            }
            theta_h->VertToHoriz();

            diagHorizVort(velx_j, dudz_j);
        }

        // assemble the residual vectors
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);

            assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);

            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii]);
            MatMult(vo->VB, dF_z, F_rho->vz[ii]);
            MatMult(vo->VB, dG_z, F_rt->vz[ii]);
        }
        F_exner->VertToHoriz();
        F_exner->UpdateGlobal();
        F_rho->VertToHoriz();
        F_rho->UpdateGlobal();
        F_rt->VertToHoriz();
        F_rt->UpdateGlobal();

        for(int lev = 0; lev < geom->nk; lev++) {
            // horizontal velocity residual
            assemble_residual_x(lev, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
                                velx_i[lev], velx_j[lev], rho_i->vh[lev], rho_j->vh[lev], F_u[lev], _F, _G);

            // density and density weighted potential temperature residuals horiztonal flux components
            M2->assemble(lev, SCALE, true);
            MatMult(EtoF->E21, _F, dF);
            MatMult(EtoF->E21, _G, dG);
            MatMult(M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);
            MatMult(M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);
        }
        F_rho->UpdateLocal();
        F_rho->HorizToVert();
        F_rt->UpdateLocal();
        F_rt->HorizToVert();

        assemble_schur_3d(theta_h, velx_i, velz_i->vz, rho_i, rt_i, exner_j, 
                          F_u, F_w->vz, F_rho, F_rt, F_exner, d_u, d_w->vz, d_rho, d_rt, d_exner);

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(velz_j->vz[ii], 1.0, d_w->vz[ii]);
            max_norm_w = MaxNorm(d_w->vz[ii], velz_j->vz[ii], max_norm_w);
        }
        MPI_Allreduce(&max_norm_w, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w = norm_x;

        for(int lev = 0; lev < geom->nk; lev++) {
            VecAXPY(velx_j[lev],      1.0, d_u[lev]        );
            VecAXPY(rt_j->vh[lev],    1.0, d_rt->vh[lev]   );
            VecAXPY(exner_j->vh[lev], 1.0, d_exner->vh[lev]);
            max_norm_u = MaxNorm(d_u[lev], velx_j[lev], max_norm_u);
        }
        MPI_Allreduce(&max_norm_u, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_u = norm_x;

        exner_j->UpdateLocal();
        VecZeroEntries(schur->b);
        schur->RepackFromHoriz(exner_j->vl, schur->b);
        VecNorm(schur->b, NORM_2, &norm_x);
        VecNorm(schur->x, NORM_2, &max_norm_exner);

        // udpate the density
        for(int lev = 0; lev < geom->nk; lev++) {
            VecCopy(rho_i->vh[lev], l2_tmp->vh[lev]);
            diagnose_F_x(lev, velx_i[lev], velx_j[lev], rho_i->vh[lev], rho_j->vh[lev], _F);
            MatMult(EtoF->E21, _F, dF);
            VecAXPY(l2_tmp->vh[lev], -dt, dF);
        }
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            diagnose_F_z(ex, ey, velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], F_z);
            MatMult(vo->V10, F_z, rho_j->vz[ii]);
        }
        rho_j->VertToHoriz();
        rho_j->UpdateGlobal();
        for(int lev = 0; lev < geom->nk; lev++) {
            VecAYPX(rho_j->vh[lev], -dt, l2_tmp->vh[lev]);
        }
        
        if(!rank) cout << itt << ":\t|d_exner|: "        << max_norm_exner <<
                                 "\t|exner|: "           << norm_x         <<
                                 "\t|d_exner|/|exner|: " << max_norm_exner/norm_x << 
                                 "\t|d_u|/|u|: "         << max_norm_u     <<
                                 "\t|d_w|/|w|: "         << max_norm_w     << endl;
        max_norm_exner /= norm_x;        
        if(max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8) done = true;

        itt++;
    } while(!done);

    velz_j->VertToHoriz();
    velz_j->UpdateGlobal();

    // update the input/output fields
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(velx_j[lev], velx_i[lev]);
    }
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_h->vh);

    // write output
    if(save) {
        dump(velx_i, velz_i, rho_i, rt_i, exner_i, theta_h, step++);
    }

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete theta_i;
    delete theta_h;
    delete F_w;
    delete F_rho;
    delete F_rt;
    delete F_exner;
    delete d_rho;
    delete d_rt;
    delete d_exner;
    delete l2_tmp;
    for(int lev = 0; lev < geom->nk; lev++) {
        VecDestroy(&velx_j[lev]);
        VecDestroy(&dudz_i[lev]);
        VecDestroy(&dudz_j[lev]);
        VecDestroy(&F_u[lev]);
        VecDestroy(&d_u[lev]);
    }
    delete[] velx_j;
    delete[] dudz_i;
    delete[] dudz_j;
    delete[] F_u;
    delete[] d_u;
    VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&dF);
    VecDestroy(&dG);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
    VecDestroy(&h_tmp);
}
