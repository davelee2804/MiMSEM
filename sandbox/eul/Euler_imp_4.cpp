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
#define EXPLICIT_THETA_UPDATE

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
    sprintf(fieldname, "velocity_z");
    for(int ii = 0; ii < geom->nk-1; ii++) {
        geom->write2(velz->vh[ii], fieldname, num, ii, false);
    }
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
    Vec F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner;

    elOrd2 = topo->elOrd*topo->elOrd;
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &F_exner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &d_w);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_rt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &d_exner);

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

        initBousFac(theta_h, bous->vz);

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // assemble the residual vectors
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

            // solve schur complement 
            assemble_operator_schur(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_j->vz[ii], 
                                    F_w, F_rho, F_rt, F_exner, d_w, d_rho, d_rt, d_exner, bous->vz[ii]);

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
            diagnose_F_z(ex, ey, velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], F_z->vz[ii]);
            MatMult(vo->V10, F_z->vz[ii], dF_z->vz[ii]);
            VecCopy(rho_j->vz[ii], d_rho);
            VecCopy(rho_i->vz[ii], rho_j->vz[ii]);
            VecAXPY(rho_j->vz[ii], -dt, dF_z->vz[ii]);
            VecAXPY(d_rho, -1.0, rho_j->vz[ii]);
#endif

#ifdef EXPLICIT_THETA_UPDATE
            vo->AssembleLinearInv(ex, ey, vo->VA);
            vo->AssembleLinearWithTheta(ex, ey, theta_h->vz[ii], vo->VA);
            MatMult(vo->VA, F_z->vz[ii], _tmpA1);
            MatMult(vo->VA_inv, _tmpA1, G_z->vz[ii]);
            MatMult(vo->V10, G_z->vz[ii], dG_z->vz[ii]);
            VecCopy(rt_j->vz[ii], d_rt);
            VecCopy(rt_i->vz[ii], rt_j->vz[ii]);
            VecAXPY(rt_j->vz[ii], -dt, dG_z->vz[ii]);
            VecAXPY(d_rt, -1.0, rt_j->vz[ii]);
#endif

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
    delete bous;
    VecDestroy(&F_w);
    VecDestroy(&F_rho);
    VecDestroy(&F_rt);
    VecDestroy(&F_exner);
    VecDestroy(&d_w);
    VecDestroy(&d_rho);
    VecDestroy(&d_rt);
    VecDestroy(&d_exner);
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

    if(!*_PC) {
        MatCreateSeqAIJ(MPI_COMM_SELF, nDofsTotal, nDofsTotal, 8*n2, NULL, _PC);
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

/*
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
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &pc_GRAD);
    MatScale(pc_GRAD, 0.5*dt);
    
    // [rt,u] block
    vo->AssembleLinearWithRT(ex, ey, rt, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_V1DV0_invV0_rt);
    MatScale(pc_V1DV0_invV0_rt, 0.5*dt);

    MatMatMult(pc_V1DV0_invV0_rt, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_DIV);
    MatMatMult(pc_DIV, pc_GRAD, reuse, PETSC_DEFAULT, &_PCz);

    if(build_ksp) {
        MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_N_rt_inv);
    }
    vo->Assemble_EOS_BlockInv(ex, ey, rt, pc_N_rt_inv);
    MatScale(pc_N_rt_inv, -1.0*CV/RD);
    MatMatMult(vo->VB, pc_N_rt_inv, reuse, PETSC_DEFAULT, &pc_V1V1_rt_inv);

    // [exner,exner] block
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_VBVB_rt_invVB_pi);

    MatMatMult(pc_V1V1_rt_inv, pc_VBVB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_V1V1_rt_invV1);
    MatAXPY(_PCz, -1.0, pc_V1V1_rt_invV1, DIFFERENT_NONZERO_PATTERN);

    if(build_ksp) {
        PC pc;
        KSPCreate(MPI_COMM_SELF, &ksp_exner);
        KSPSetOperators(ksp_exner, _PCz, _PCz);
        KSPGetPC(ksp_exner, &pc);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(ksp_exner, "ksp_exner_");
        KSPSetFromOptions(ksp_exner);
    }

    // assemble the rhs for the exner solve
    MatMult(pc_DIV, F_w, _tmpB1);
    MatMult(pc_V1V1_rt_inv, F_exner, _tmpB2);
    VecAXPY(_tmpB1, 1.0, _tmpB2);
    VecAXPY(_tmpB1, -1.0, F_rt);
    KSPSolve(ksp_exner, _tmpB1, dexner);

    // update the velocity
    MatMult(pc_GRAD, dexner, _tmpA1);
    VecAYPX(_tmpA1, 1.0, F_w);
    MatMult(vo->VA_inv, _tmpA1, dw);
    VecScale(dw, -1.0);

    // update the density weighted potential temperature
    MatMult(pc_VBVB_rt_invVB_pi, dexner, _tmpB1);
    VecAYPX(_tmpB1, 1.0, F_exner);
    MatMult(pc_N_rt_inv, _tmpB1, drt);
    VecScale(drt, -1.0);
    //MatMult(pc_V1DV0_invV0_rt, dw, _tmpB1);
    //VecAYPX(_tmpB1, 1.0, F_rt);
    //VecScale(_tmpB1, -1.0);
    //vo->AssembleConstInv(ex, ey, vo->VB_inv);
    //MatMult(vo->VB_inv, _tmpB1, drt);

    // update the density
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V1DV0_invV0_rt);

    MatMult(pc_V1DV0_invV0_rt, dw, _tmpB1);
    VecAYPX(_tmpB1, 0.5*dt, F_rho);
    VecScale(_tmpB1, -1.0);
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMult(vo->VB_inv, _tmpB1, drho);
}
*/

void Euler::assemble_operator_schur(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, 
                                    Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner, Vec bous) {
    int n2 = topo->elOrd*topo->elOrd;
    bool build_ksp = (!_PCz) ? true : false;
    MatReuse reuse = (!_PCz) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    // [u,exner] block
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &pc_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_DTV1, reuse, PETSC_DEFAULT, &pc_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, pc_V0_invDTV1, reuse, PETSC_DEFAULT, &pc_GRAD);
    MatScale(pc_GRAD, 0.5*dt);
    // -- rho correction
    vo->AssembleLinearWithThetaExp(ex, ey, theta, RD/CV, vo->VA);
    MatMatMult(vo->VA, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_V0_invV0_rt);
    vo->AssembleLinearWithRhoInv(ex, ey, rho, vo->VA_inv);
    MatMatMult(vo->VA_inv, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_VA_invVAVA_inv);
    vo->AssembleLinearWithRhoExp(ex, ey, rho, RD/CV, vo->VA);
    MatMatMult(vo->VA, pc_VA_invVAVA_inv, reuse, PETSC_DEFAULT, &pc_dPidRho);
    MatScale(pc_dPidRho, CP*pow(RD/P0, RD/CV));
    VecSet(_tmpA1, 1.0);
    MatDiagonalSet(pc_dPidRho, _tmpA1, ADD_VALUES);
    MatMatMult(pc_dPidRho, pc_GRAD, reuse, PETSC_DEFAULT, &pc_GRAD_2);
    MatCopy(pc_GRAD_2, pc_GRAD, DIFFERENT_NONZERO_PATTERN);
    
    // [rt,u] block
    vo->AssembleLinearWithRT(ex, ey, rt, vo->VA, true);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt); // already allocated above
    MatMatMult(vo->V10, pc_V0_invV0_rt, reuse, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, reuse, PETSC_DEFAULT, &pc_V1DV0_invV0_rt);
    MatScale(pc_V1DV0_invV0_rt, 0.5*dt);

    vo->AssembleLinearWithRhoInv(ex, ey, bous, vo->VA_inv); // boussinesque correction
    MatMatMult(pc_V1DV0_invV0_rt, vo->VA_inv, reuse, PETSC_DEFAULT, &pc_DIV);
    MatMatMult(pc_DIV, pc_GRAD, reuse, PETSC_DEFAULT, &_PCz);

    if(build_ksp) {
        MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_N_rt_inv);
    }
    vo->Assemble_EOS_BlockInv(ex, ey, rt, theta, pc_N_rt_inv);
    MatScale(pc_N_rt_inv, -1.0*CV/RD);

    // [exner,exner] block
    vo->AssembleConstWithRhoInv(ex, ey, exner, vo->VB_inv);
    MatMatMult(vo->VB_inv, vo->VB, reuse, PETSC_DEFAULT, &pc_VB_rt_invVB_pi);
    MatMatMult(vo->VB, pc_VB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_VBVB_rt_invVB_pi);
    // -- rho correction (dPi/dRho)
    if(build_ksp) {
        MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_VB_rho);
        MatCreateSeqAIJ(MPI_COMM_SELF, geom->nk*n2, geom->nk*n2, n2, NULL, &pc_VB_theta);
    }
    vo->AssembleConstWithRhoExp(ex, ey, rho, RD/CV, pc_VB_rho);
    vo->AssembleConstWithRhoInv(ex, ey, rho, vo->VB_inv);
    MatMatMult(pc_VB_rho, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_VB_rho_exp);
    vo->AssembleConstWithThetaExp(ex, ey, theta, RD/CV, pc_VB_theta);
    MatMatMult(pc_VB_rho_exp, pc_VB_theta, MAT_REUSE_MATRIX, PETSC_DEFAULT, &vo->VB);
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(vo->VB, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_dPidRho_B);
    MatScale(pc_dPidRho_B, CP*pow(RD/P0, RD/CV));
    VecSet(_tmpB1, 1.0);
    MatDiagonalSet(pc_dPidRho_B, _tmpB1, ADD_VALUES);
    MatMatMult(pc_dPidRho_B, pc_VBVB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_VBVB_rt_invVB_pi_2);
    MatCopy(pc_VBVB_rt_invVB_pi_2, pc_VBVB_rt_invVB_pi, DIFFERENT_NONZERO_PATTERN);
/*
*/

    // -- rho correction (dTheta/dRho)
    vo->AssembleConstWithTheta(ex, ey, theta, pc_VB_theta);
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMatMult(pc_VB_theta, vo->VB_inv, reuse, PETSC_DEFAULT, &pc_dRTdRho_B);
    VecSet(_tmpB1, 1.0);
    MatDiagonalSet(pc_dRTdRho_B, _tmpB1, ADD_VALUES);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(pc_dRTdRho_B, vo->VB, reuse, PETSC_DEFAULT, &pc_M_rt);

    //vo->AssembleConst(ex, ey, vo->VB);
    //MatMatMult(vo->VB, pc_N_rt_inv, reuse, PETSC_DEFAULT, &pc_V1V1_rt_inv);
    MatMatMult(pc_M_rt, pc_N_rt_inv, reuse, PETSC_DEFAULT, &pc_V1V1_rt_inv);
    MatMatMult(pc_V1V1_rt_inv, pc_VBVB_rt_invVB_pi, reuse, PETSC_DEFAULT, &pc_V1V1_rt_invV1);
    MatAXPY(_PCz, -1.0, pc_V1V1_rt_invV1, DIFFERENT_NONZERO_PATTERN);

    if(build_ksp) {
        PC pc;
        KSPCreate(MPI_COMM_SELF, &ksp_exner);
        KSPSetOperators(ksp_exner, _PCz, _PCz);
        KSPGetPC(ksp_exner, &pc);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(ksp_exner, "ksp_exner_");
        KSPSetFromOptions(ksp_exner);
    }

    // assemble the rhs for the exner solve
    MatMult(pc_DIV, F_w, _tmpB1);
    MatMult(pc_V1V1_rt_inv, F_exner, _tmpB2);
    VecAXPY(_tmpB1, 1.0, _tmpB2);
    VecAXPY(_tmpB1, -1.0, F_rt);
    KSPSolve(ksp_exner, _tmpB1, dexner);

    // update the velocity
    MatMult(pc_GRAD, dexner, _tmpA1);
    VecAYPX(_tmpA1, 1.0, F_w);
    MatMult(vo->VA_inv, _tmpA1, dw);
    VecScale(dw, -1.0);

    // update the density weighted potential temperature
#ifndef EXPLICIT_THETA_UPDATE
    MatMult(pc_VBVB_rt_invVB_pi, dexner, _tmpB1);
    VecAYPX(_tmpB1, 1.0, F_exner);
    MatMult(pc_N_rt_inv, _tmpB1, drt);
    VecScale(drt, -1.0);
    //MatMult(pc_V1DV0_invV0_rt, dw, _tmpB1);
    //VecAYPX(_tmpB1, 1.0, F_rt);
    //VecScale(_tmpB1, -1.0);
    //vo->AssembleConstInv(ex, ey, vo->VB_inv);
    //MatMult(vo->VB_inv, _tmpB1, drt);
#endif

    // update the density
#ifndef EXPLICIT_RHO_UPDATE
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V0_invV0_rt);
    MatMatMult(vo->V10, pc_V0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_DV0_invV0_rt);
    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->VB, pc_DV0_invV0_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &pc_V1DV0_invV0_rt);

    MatMult(pc_V1DV0_invV0_rt, dw, _tmpB1);
    VecAYPX(_tmpB1, 0.5*dt, F_rho);
    VecScale(_tmpB1, -1.0);
    vo->AssembleConstInv(ex, ey, vo->VB_inv);
    MatMult(vo->VB_inv, _tmpB1, drho);
#endif
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
