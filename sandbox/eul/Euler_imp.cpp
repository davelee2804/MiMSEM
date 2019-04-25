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
#include "Euler_imp.h"

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

using namespace std;

Euler::Euler(Topo* _topo, Geom* _geom, double _dt) {
    int ii;
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

    // coriolis vector (projected onto 0 forms)
    coriolis();

    vo = new VertOps(topo, geom);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_t);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_b_l);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_t_l);

    gv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd*topo->elOrd, &gv[ii]);
    }
    zv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*topo->elOrd*topo->elOrd, &zv[ii]);
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
    PCBJacobiSetTotalBlocks(pc, topo->elOrd*topo->elOrd, NULL);
    KSPSetOptionsPrefix(ksp2, "ksp2_");
    KSPSetFromOptions(ksp2);

    _DTV1 = NULL;
    _M1invM1 = NULL;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*topo->elOrd*topo->elOrd, &_Phi_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*topo->elOrd*topo->elOrd, &_theta_h);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd*topo->elOrd, &_tmpA1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd*topo->elOrd, &_tmpA2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*topo->elOrd*topo->elOrd, &_tmpB1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*topo->elOrd*topo->elOrd, &_tmpB2);

    PCz = new Mat[topo->nElsX*topo->nElsX];
    PCx = new Mat[geom->nk];
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
                    //det = geom->det[ei][ii];
                    //Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
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
    VecDestroy(&theta_b);
    VecDestroy(&theta_t);
    VecDestroy(&theta_b_l);
    VecDestroy(&theta_t_l);

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

    delete edge;
    delete node;
    delete quad;

    delete vo;
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
        M1->assemble(lev, SCALE);
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
        M1->assemble(lev, SCALE);
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

void Euler::solve(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, bool save) {
    char fieldname[100];
    bool done = false;
    int elOrd2 = topo->elOrd*topo->elOrd;
    double norm_max_x, norm_max_z, norm_u, norm_du, norm_max_dz;
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_j = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* exner = new L2Vecs(geom->nk, topo, geom);
    Vec* velx_j = new Vec[geom->nk];
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    Vec du, fu, dw, fw, htmp, wi;
    Vec* _F_x = new Vec[geom->nk];
    Vec* _G_x = new Vec[geom->nk];
    Vec* _F_z = new Vec[topo->nElsX*topo->nElsX];
    Vec* _G_z = new Vec[topo->nElsX*topo->nElsX];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    PC pc;
    KSP ksp_x;
    KSP ksp_z;

    if(firstStep) {
        // assumed these have been initialised from the driver
        VecScatterBegin(topo->gtol_2, theta_b, theta_b_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, theta_b, theta_b_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, theta_t, theta_t_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, theta_t, theta_t_l, INSERT_VALUES, SCATTER_FORWARD);
    }

    velz_i->UpdateLocal();
    velz_i->HorizToVert();
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rt_i->UpdateLocal();
    rt_i->HorizToVert();

    // diagnose the potential temperature
    diagTheta(rho_i->vz, rt_i->vz, theta_i);
    for(int ii = 0; ii < geom->nk; ii++) {
        diagnose_Pi(ii, rt_i->vl[ii], rt_i->vl[ii], exner->vh[ii]);
    }
    exner->UpdateLocal();
    exner->HorizToVert();

    // update the next time level
    velz_j->CopyFromHoriz(velz_i->vh);
    rho_j->CopyFromHoriz(rho_i->vh);
    rt_j->CopyFromHoriz(rt_i->vh);
    theta_j->CopyFromVert(theta_i->vz);
    theta_j->VertToHoriz();

    // create the preconditioner operators
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &dw);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &fw);
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_F_z[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_G_z[ii]);

        assemble_precon_z(ii%topo->nElsX, ii/topo->nElsX, theta_i->vz[ii], rt_i->vz[ii], rt_j->vz[ii], exner->vz[ii], velz_j->vz[ii]);
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

        assemble_precon_x(ii, theta_i->vl, rt_i->vl[ii], exner->vl[ii]);
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);

    diagHorizVort(velx_i, dudz_i);
    for(int ii = 0; ii < geom->nk; ii++) {
        VecCopy(dudz_i[ii], dudz_j[ii]);
    }

    do {
        for(int ii = 0; ii < geom->nk+1; ii++) {
            VecAXPY(theta_j->vl[ii], 1.0, theta_i->vl[ii]);
            VecScale(theta_j->vl[ii], 0.5);
        }

        // update the horizontal dynamics for this iteration
        norm_max_x = -1.0;
        for(int ii = 0; ii < geom->nk; ii++) {
            if(!rank) cout << rank << ":\tassembling (horizontal) residual vector: " << ii;
            assemble_residual_x(ii, theta_i->vl, theta_j->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner->vh[ii],
                                velx_i[ii], velx_j[ii], rho_i->vh[ii], rho_j->vh[ii], rt_i->vh[ii], rt_j->vh[ii], 
                                fu, _F_x[ii], _G_x[ii]);
            VecScale(fu, -1.0);

            assemble_precon_x(ii, theta_j->vl, rt_j->vl[ii], exner->vl[ii]);

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

        // update the vertical dynamics for this iteration
        if(!rank) cout << rank << ":\tassembling (vertical) residual vectors" << endl;
        norm_max_z = -1.0;
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            assemble_residual_z(ii%topo->nElsX, ii/topo->nElsX, theta_i->vz[ii], theta_j->vz[ii], exner->vz[ii],
                                velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], rt_i->vz[ii], rt_j->vz[ii], 
                                fw, _F_z[ii], _G_z[ii]);
            VecScale(fw, -1.0);

            assemble_precon_z(ii%topo->nElsX, ii/topo->nElsX, theta_j->vz[ii], rt_i->vz[ii], rt_j->vz[ii], exner->vz[ii], velz_j->vz[ii]);

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

        for(int ii = 0; ii < geom->nk; ii++) {
            VecCopy(rho_i->vh[ii], rho_j->vh[ii]);
            MatMult(EtoF->E21, _F_x[ii], htmp);
            VecAXPY(rho_j->vh[ii], -dt, htmp);

            VecCopy(rt_i->vh[ii], rt_j->vh[ii]);
            MatMult(EtoF->E21, _G_x[ii], htmp);
            VecAXPY(rt_j->vh[ii], -dt, htmp); // TODO: breaks convergence!
        }
        rho_j->UpdateLocal();
        rho_j->HorizToVert();
        rt_j->UpdateLocal();
        rt_j->HorizToVert();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            MatMult(vo->V10, _F_z[ii], _tmpB1);
            VecAXPY(rho_j->vz[ii], -dt, _tmpB1); // TODO: breaks convergence!

            MatMult(vo->V10, _G_z[ii], _tmpB1);
            VecAXPY(rt_j->vz[ii], -dt, _tmpB1); // TODO: breaks convergence!
        }
        rho_j->VertToHoriz();
        rho_j->UpdateGlobal();
        rt_j->VertToHoriz();
        rt_j->UpdateGlobal();

        diagTheta(rho_j->vz, rt_j->vz, theta_j);
        for(int ii = 0; ii < geom->nk; ii++) {
            diagnose_Pi(ii, rt_i->vl[ii], rt_j->vl[ii], exner->vh[ii]);
        }
        exner->UpdateLocal();
        exner->HorizToVert();
        diagHorizVort(velx_j, dudz_j);

        //if(!rank) cout << "|dx|/|x|: " << norm_max_x << "\t|dz|/|z|: " << norm_max_z << endl;
        if(!rank) cout << "|dz|: " << norm_max_dz << "\t|z|: " << norm_max_z << "\t|dz|/|z|: " << norm_max_dz/norm_max_z << endl;

        if(norm_max_x < 1.0e-12 && norm_max_dz/norm_max_z < 1.0e-8) done = true;
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
        step++;

        theta_j->UpdateGlobal();
        for(int ii = 0; ii < geom->nk+1; ii++) {
            sprintf(fieldname, "theta");
            geom->write2(theta_j->vh[ii], fieldname, step, ii, false);
        }

        for(int ii = 0; ii < geom->nk; ii++) {
            curl(true, velx_j[ii], &wi, ii, false);

            sprintf(fieldname, "vorticity");
            geom->write0(wi, fieldname, step, ii);
            sprintf(fieldname, "velocity_h");
            geom->write1(velx_j[ii], fieldname, step, ii);
            sprintf(fieldname, "density");
            geom->write2(rho_j->vh[ii], fieldname, step, ii, true);
            sprintf(fieldname, "rhoTheta");
            geom->write2(rt_j->vh[ii], fieldname, step, ii, true);
            sprintf(fieldname, "exner");
            geom->write2(exner->vh[ii], fieldname, step, ii, true);

            VecDestroy(&wi);
        }
        sprintf(fieldname, "velocity_z");
        for(int ii = 0; ii < geom->nk-1; ii++) {
            geom->write2(velz_j->vh[ii], fieldname, step, ii, false);
        }
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
    delete theta_j;
    delete exner;
    delete velz_j;
    delete rho_j;
    delete rt_j;
}

void Euler::solve_strang(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, bool save) {
    char fieldname[100];
    bool done = false;
    int elOrd2 = topo->elOrd*topo->elOrd;
    double norm_max_x, norm_max_z, norm_u, norm_du, norm_max_dz;
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_j = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* exner = new L2Vecs(geom->nk, topo, geom);
    Vec* velx_j = new Vec[geom->nk];
    L2Vecs* velz_j = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j = new L2Vecs(geom->nk, topo, geom);
    Vec du, fu, dw, fw, htmp, wi;
    Vec* _F_x = new Vec[geom->nk];
    Vec* _G_x = new Vec[geom->nk];
    Vec* _F_z = new Vec[topo->nElsX*topo->nElsX];
    Vec* _G_z = new Vec[topo->nElsX*topo->nElsX];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    PC pc;
    KSP ksp_x;
    KSP ksp_z;
L2Vecs* exner_i = new L2Vecs(geom->nk, topo, geom);

    if(firstStep) {
        // assumed these have been initialised from the driver
        VecScatterBegin(topo->gtol_2, theta_b, theta_b_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, theta_b, theta_b_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, theta_t, theta_t_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, theta_t, theta_t_l, INSERT_VALUES, SCATTER_FORWARD);
    }

    velz_i->UpdateLocal();
    velz_i->HorizToVert();
    rho_i->UpdateLocal();
    rho_i->HorizToVert();
    rt_i->UpdateLocal();
    rt_i->HorizToVert();

    // diagnose the potential temperature
    diagTheta(rho_i->vz, rt_i->vz, theta_i);
    for(int ii = 0; ii < geom->nk; ii++) {
        diagnose_Pi(ii, rt_i->vl[ii], rt_i->vl[ii], exner->vh[ii]);
    }
    exner->UpdateLocal();
    exner->HorizToVert();
exner_i->CopyFromVert(exner->vz);
for(int ii = 0; ii < geom->nk; ii++) {
sprintf(fieldname, "density");
geom->write2(rho_i->vh[ii], fieldname, 9999, ii, true);
sprintf(fieldname, "rhoTheta");
geom->write2(rt_i->vh[ii], fieldname, 9999, ii, true);
sprintf(fieldname, "exner");
geom->write2(exner->vh[ii], fieldname, 9999, ii, true);
}
sprintf(fieldname, "theta");
for(int ii = 0; ii < geom->nk+1; ii++) {
geom->write2(theta_i->vh[ii], fieldname, 9999, ii, false);
}

    // update the next time level
    velz_j->CopyFromHoriz(velz_i->vh);
    rho_j->CopyFromHoriz(rho_i->vh);
    rt_j->CopyFromHoriz(rt_i->vh);
    theta_j->CopyFromVert(theta_i->vz);
    theta_j->VertToHoriz();

    velz_j->UpdateLocal();
    rho_j->UpdateLocal();
    rt_j->UpdateLocal();
    velz_j->HorizToVert();
    rho_j->HorizToVert();
    rt_j->HorizToVert();

    // create the preconditioner operators
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &dw);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &fw);
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_F_z[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_G_z[ii]);

        assemble_precon_z(ii%topo->nElsX, ii/topo->nElsX, _theta_h, rt_i->vz[ii], rt_j->vz[ii], exner->vz[ii], velz_j->vz[ii]);
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

        assemble_precon_x(ii, theta_i->vl, rt_i->vl[ii], exner->vl[ii]);
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);

    diagHorizVort(velx_i, dudz_i);
    for(int ii = 0; ii < geom->nk; ii++) {
        VecCopy(dudz_i[ii], dudz_j[ii]);
    }

    do {
        // update the vertical dynamics for this iteration
        if(!rank) cout << rank << ":\tassembling (vertical) residual vectors" << endl;
        norm_max_dz = 0.0;
        norm_max_z = 1.0;
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            assemble_residual_z(ii%topo->nElsX, ii/topo->nElsX, theta_i->vz[ii], theta_j->vz[ii], exner->vz[ii],
                                velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], rt_i->vz[ii], rt_j->vz[ii], 
                                fw, _F_z[ii], _G_z[ii]);
            VecScale(fw, -1.0);

            assemble_precon_z(ii%topo->nElsX, ii/topo->nElsX, _theta_h, rt_i->vz[ii], rt_j->vz[ii], exner->vz[ii], velz_j->vz[ii]);
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

        diagTheta(rho_j->vz, rt_j->vz, theta_j);
        for(int ii = 0; ii < geom->nk; ii++) {
            diagnose_Pi(ii, rt_i->vl[ii], rt_j->vl[ii], exner->vh[ii]);
        }
        exner->UpdateLocal();
        exner->HorizToVert();
        diagHorizVort(velx_j, dudz_j);

        if(!rank) cout << "|dz|: " << norm_max_dz << "\t|z|: " << norm_max_z << "\t|dz|/|z|: " << norm_max_dz/norm_max_z << endl;
        if(norm_max_dz/norm_max_z < 1.0e-8) done = true;
    } while(!done);

if(!rank)cout<<"done with veritcal solve...(1)\n";
for(int ii = 0; ii < geom->nk; ii++) {
sprintf(fieldname, "density");
geom->write2(rho_j->vh[ii], fieldname, 9999, ii, true);
sprintf(fieldname, "rhoTheta");
geom->write2(rt_j->vh[ii], fieldname, 9999, ii, true);
sprintf(fieldname, "exner");
geom->write2(exner->vh[ii], fieldname, 9999, ii, true);
}
sprintf(fieldname, "velocity_z");
for(int ii = 0; ii < geom->nk-1; ii++) {
geom->write2(velz_j->vh[ii], fieldname, 9999, ii, false);
}
sprintf(fieldname, "theta");
for(int ii = 0; ii < geom->nk+1; ii++) {
geom->write2(theta_j->vh[ii], fieldname, 9999, ii, false);
}

    do {
        // update the horizontal dynamics for this iteration
        norm_max_x = 0.0;
        for(int ii = 0; ii < geom->nk; ii++) {
            if(!rank) cout << rank << ":\tassembling (horizontal) residual vector: " << ii;
            assemble_residual_x(ii, theta_i->vl, theta_j->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner->vh[ii],
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

        diagTheta(rho_j->vz, rt_j->vz, theta_j);
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
        step++;

        theta_j->UpdateGlobal();
        for(int ii = 0; ii < geom->nk+1; ii++) {
            sprintf(fieldname, "theta");
            geom->write2(theta_j->vh[ii], fieldname, step, ii, false);
        }

        for(int ii = 0; ii < geom->nk; ii++) {
            curl(true, velx_j[ii], &wi, ii, false);

            sprintf(fieldname, "vorticity");
            geom->write0(wi, fieldname, step, ii);
            sprintf(fieldname, "velocity_h");
            geom->write1(velx_j[ii], fieldname, step, ii);
            sprintf(fieldname, "density");
            geom->write2(rho_j->vh[ii], fieldname, step, ii, true);
            sprintf(fieldname, "rhoTheta");
            geom->write2(rt_j->vh[ii], fieldname, step, ii, true);
            sprintf(fieldname, "exner");
            geom->write2(exner->vh[ii], fieldname, step, ii, true);

            VecDestroy(&wi);
        }
        sprintf(fieldname, "velocity_z");
        for(int ii = 0; ii < geom->nk-1; ii++) {
            geom->write2(velz_j->vh[ii], fieldname, step, ii, false);
        }
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
    delete theta_j;
    delete exner;
    delete velz_j;
    delete rho_j;
    delete rt_j;
delete exner_i;
}

void Euler::assemble_precon_z(int ex, int ey, Vec theta, Vec rt_i, Vec rt_j, Vec exner, Vec velz) {
    int ei = ey * topo->nElsX + ex;
    MatReuse reuse = (!_DTV1) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    vo->AssembleConst(ex, ey, vo->VB);
    MatMatMult(vo->V01, vo->VB, reuse, PETSC_DEFAULT, &_DTV1);
    vo->AssembleLinearInv(ex, ey, vo->VA_inv);
    MatMatMult(vo->VA_inv, _DTV1, reuse, PETSC_DEFAULT, &_V0_invDTV1);
    vo->AssembleLinearWithTheta(ex, ey, theta, vo->VA);
    MatMatMult(vo->VA, _V0_invDTV1, reuse, PETSC_DEFAULT, &_GRAD);

    vo->AssembleLinearWithRT(ex, ey, rt_j, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, reuse, PETSC_DEFAULT, &_V0_invV0_rt);
    MatMatMult(vo->V10, _V0_invV0_rt, reuse, PETSC_DEFAULT, &_DV0_invV0_rt);

    // diagnose the exner pressure from hydrostatic balance
    /*{
        PC pc;
        KSP ksp_hb;
        KSPCreate(MPI_COMM_SELF, &ksp_hb);
        KSPSetOperators(ksp_hb, _GRAD, _GRAD);
        KSPGetPC(ksp_hb, &pc);
        PCSetType(pc, PCLU);
        KSPSolve(ksp_hb, gv[ei], _tmpB1);
        KSPDestroy(&ksp_hb);
        VecScale(_tmpB1, -1.0);
        vo->AssembleConstWithRho(ex, ey, _tmpB1, vo->VB);
    }*/

    vo->AssembleConstWithRho(ex, ey, exner, vo->VB);
    MatMatMult(vo->VB, _DV0_invV0_rt, reuse, PETSC_DEFAULT, &_V1_PiDV0_invV0_rt);
    vo->AssembleConstWithRhoInv(ex, ey, rt_i, vo->VB);
    MatMatMult(vo->VB, _V1_PiDV0_invV0_rt, reuse, PETSC_DEFAULT, &_DIV);

    if(firstStep) {
        MatMatMult(_GRAD, _DIV, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &PCz[ei]);
    } else {
        MatMatMult(_GRAD, _DIV, MAT_REUSE_MATRIX, PETSC_DEFAULT, &PCz[ei]);
    }
    vo->AssembleLinear(ex, ey, vo->VA);
    MatAYPX(PCz[ei], -dt*dt*RD/CV, vo->VA, DIFFERENT_NONZERO_PATTERN);

    //vo->AssembleConLinWithW(ex, ey, velz, vo->VBA);
    //MatMatMult(vo->V01, vo->VBA, reuse, PETSC_DEFAULT, &vo->VA);
    //MatAXPY(PCz[ei], +0.5*dt, vo->VA, DIFFERENT_NONZERO_PATTERN);
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

void Euler::assemble_precon_x(int level, Vec* theta, Vec rt, Vec exner) {
    MatReuse reuse = (!_M1invM1) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    Mat M1inv;
    Mat M2ThetaInv;
    Vec theta_h;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_h);
    VecZeroEntries(theta_h);
    VecAXPY(theta_h, 0.5, theta[level+0]);
    VecAXPY(theta_h, 0.5, theta[level+1]);
    F->assemble(theta_h, level, false, SCALE);

    M2->assemble(level, SCALE, true);
    M1->assemble(level, SCALE);
    DiagMatInv(M1->M, topo->n1, topo->n1l, topo->nDofs1G, topo->gtol_1, &M1inv);

    T->assemble(rt, level, SCALE); 
    DiagMatInv(T->M, topo->n2, topo->n2l, topo->nDofs2G, topo->gtol_2, &M2ThetaInv);

    T->assemble(exner, level, SCALE); 
    MatMatMult(M1inv, M1->M, reuse, PETSC_DEFAULT, &_M1invM1);
    MatMatMult(EtoF->E21, _M1invM1, reuse, PETSC_DEFAULT, &_DM1invM1);
    MatMatMult(T->M, _DM1invM1, reuse, PETSC_DEFAULT, &_PiDM1invM1);
    MatMatMult(M2ThetaInv, _PiDM1invM1, reuse, PETSC_DEFAULT, &_ThetaPiDM1invM1);
    MatMatMult(M2->M, _ThetaPiDM1invM1, reuse, PETSC_DEFAULT, &_M2ThetaPiDM1invM1);
    MatMatMult(EtoF->E12, _M2ThetaPiDM1invM1, reuse, PETSC_DEFAULT, &_DM2ThetaPiDM1invM1);
    MatMatMult(M1inv, _DM2ThetaPiDM1invM1, reuse, PETSC_DEFAULT, &_M1DM2ThetaPiDM1invM1);
    if(firstStep) {
        MatMatMult(F->M, _M1DM2ThetaPiDM1invM1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &PCx[level]);
    } else {
        MatMatMult(F->M, _M1DM2ThetaPiDM1invM1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &PCx[level]);
    }
    MatAYPX(PCx[level], -dt*dt*RD/CV, M1->M, DIFFERENT_NONZERO_PATTERN);

    R->assemble(fl[level], level, SCALE);
    MatAXPY(PCx[level], dt, R->M, DIFFERENT_NONZERO_PATTERN);

    VecDestroy(&theta_h);
    MatDestroy(&M1inv);
    MatDestroy(&M2ThetaInv);
}

/*
assemble the boundary condition vector for rho(t) X theta(0)
assume V0^{rho} has already been assembled (omitting internal levels)
*/
void Euler::thetaBCVec(int ex, int ey, Mat A, Vec bTheta) {
    int* inds2 = topo->elInds2_l(ex, ey);
    int ii, n2;
    PetscScalar *vArray, *hArray;

    n2 = topo->elOrd*topo->elOrd;

    // assemble the theta bc vector
    VecZeroEntries(_tmpA2);
    VecGetArray(_tmpA2, &vArray);
    // bottom
    VecGetArray(theta_b_l, &hArray);
    for(ii = 0; ii < n2; ii++) {
        vArray[ii] = hArray[inds2[ii]];
    }
    VecRestoreArray(theta_b_l, &hArray);
    // top
    VecGetArray(theta_t_l, &hArray);
    for(ii = 0; ii < n2; ii++) {
        vArray[(geom->nk-2)*n2+ii] = hArray[inds2[ii]];
    }
    VecRestoreArray(theta_t_l, &hArray);
    VecRestoreArray(_tmpA2, &vArray);

    MatMult(A, _tmpA2, bTheta);
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
    M1->assemble(level, SCALE);
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
    Vec rtl1, rtl2;
    Vec rhs;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &rtl1);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &rtl2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);

    VecScatterBegin(topo->gtol_2, rt1, rtl1, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, rt1, rtl1, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, rt2, rtl2, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, rt2, rtl2, INSERT_VALUES, SCATTER_FORWARD);

    //VecZeroEntries(rhs);
    //eos->assemble(rtl1, level, SCALE);
    //VecAXPY(rhs, 0.5, eos->vg);
    //eos->assemble(rtl2, level, SCALE);
    //VecAXPY(rhs, 0.5, eos->vg);
    //M2->assemble(level, SCALE, true);
    //KSPSolve(ksp2, rhs, Pi);

    eos->assemble_quad(rtl1, rtl2, level, SCALE);
    M2->assemble(level, SCALE, true);
    KSPSolve(ksp2, eos->vg, Pi);

    VecDestroy(&rtl1);
    VecDestroy(&rtl2);
    VecDestroy(&rhs);
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
    VecZeroEntries(_F);

    vo->AssembleLinearInv(ex, ey, vo->VA_inv);

    MatZeroEntries(_V0_invV0_rt);
    vo->AssembleLinearWithRT(ex, ey, rho1, vo->VA, true);
    MatMatMult(vo->VA_inv, vo->VA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &_V0_invV0_rt);

    MatMult(_V0_invV0_rt, velz1, _tmpA1);
    VecAXPY(_F, 1.0/3.0, _tmpA1);

    MatMult(_V0_invV0_rt, velz2, _tmpA1);
    VecAXPY(_F, 1.0/6.0, _tmpA1);

    MatZeroEntries(_V0_invV0_rt);
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

// TODO: use quadrttic approximation to the EoS and integrate this exactly in time
/*void Euler::diagnose_Pi_z(int ex, int ey, Vec rt1, Vec rt2, Vec Pi) {
    VecZeroEntries(Pi);
    vo->AssembleConstInv(ex, ey, vo->VB_inv);

    vo->Assemble_EOS_RHS(ex, ey, rt1, _tmpB1);
    MatMult(vo->VB_inv, _tmpB1, _tmpB2);
    VecAXPY(Pi, 0.5, _tmpB2);

    vo->Assemble_EOS_RHS(ex, ey, rt2, _tmpB1);
    MatMult(vo->VB_inv, _tmpB1, _tmpB2);
    VecAXPY(Pi, 0.5, _tmpB2);
}*/

/*
diagnose theta from rho X theta (with boundary condition)
note: rho, rhoTheta and theta are all LOCAL vectors
*/
void Euler::diagTheta(Vec* rho, Vec* rt, L2Vecs* theta) {
    int ex, ey, ei, ii, kk, elOrd2;
    int* inds2;
    Vec frt, theta_v;
    Mat AB;
    PC pc;
    KSP kspColA;
    PetscScalar *tbArray, *ttArray, *tvArray, *tArray;

    elOrd2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &frt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &theta_v);

    MatCreate(MPI_COMM_SELF, &AB);
    MatSetType(AB, MATSEQAIJ);
    MatSetSizes(AB, (geom->nk-1)*elOrd2, (geom->nk+0)*elOrd2, (geom->nk-1)*elOrd2, (geom->nk+0)*elOrd2);
    MatSeqAIJSetPreallocation(AB, 2*elOrd2, PETSC_NULL);

    KSPCreate(MPI_COMM_SELF, &kspColA);
    KSPSetOperators(kspColA, vo->VA, vo->VA);
    KSPGetPC(kspColA, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColA, "kspColA_");
    KSPSetFromOptions(kspColA);

    VecGetArray(theta_b_l, &tbArray);
    VecGetArray(theta_t_l, &ttArray);

    // do this to initialise the order of the data in this matrix before the bcs are assembled
    vo->AssembleLinear(0, 0, vo->VA);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // add solution to the column vector
            ei = ey*topo->nElsX + ex;
            inds2 = topo->elInds2_l(ex, ey);

            // construct horiztonal rho theta field
            vo->AssembleLinCon(ex, ey, AB);
            MatMult(AB, rt[ei], frt);

            // assemble in the bcs
            vo->AssembleLinearWithRT(ex, ey, rho[ei], vo->VA, false);
            thetaBCVec(ex, ey, vo->VA, _tmpA1);
            VecAXPY(frt, -1.0, _tmpA1);

            vo->AssembleLinearWithRT(ex, ey, rho[ei], vo->VA, true);
            KSPSolve(kspColA, frt, theta_v);

            VecGetArray(theta_v, &tvArray);
            VecGetArray(theta->vz[ei], &tArray);
            for(ii = 0; ii < elOrd2; ii++) {
                tArray[ii]                   = tbArray[inds2[ii]];
                tArray[geom->nk*elOrd2 + ii] = ttArray[inds2[ii]];
            }
            for(kk = 0; kk < geom->nk - 1; kk++) {
                for(ii = 0; ii < elOrd2; ii++) {
                    tArray[(kk+1)*elOrd2 + ii] = tvArray[kk*elOrd2 + ii];
                }
            }
            VecRestoreArray(theta_v, &tvArray);
            VecRestoreArray(theta->vz[ei], &tArray);
        }
    }
    VecRestoreArray(theta_b_l, &tbArray);
    VecRestoreArray(theta_t_l, &ttArray);

    theta->VertToHoriz();
theta->UpdateGlobal();

    VecDestroy(&frt);
    VecDestroy(&theta_v);
    MatDestroy(&AB);
    KSPDestroy(&kspColA);
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
        M1->assemble(ii, SCALE);
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

void Euler::assemble_residual_x(int level, Vec* theta1, Vec* theta2, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
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
    M1->assemble(level, SCALE);
    M2->assemble(level, SCALE, true);

    // assume theta2 is 0.5*(theta_i + theta_j)
    VecZeroEntries(theta_h);
    //VecAXPY(theta_h, 0.25, theta1[level+0]);
    //VecAXPY(theta_h, 0.25, theta1[level+1]);
    //VecAXPY(theta_h, 0.25, theta2[level+0]);
    //VecAXPY(theta_h, 0.25, theta2[level+1]);
    VecAXPY(theta_h, 0.5, theta2[level+0]);
    VecAXPY(theta_h, 0.5, theta2[level+1]);

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
    VecAXPY(fu, 1.0, utmp); // TODO: coupling to vertical solve causing blow up
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

void Euler::assemble_residual_z(int ex, int ey, Vec theta1, Vec theta2, Vec Pi, 
                                Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fw, Vec _F, Vec _G) 
{
    // diagnose the hamiltonian derivatives
    diagnose_F_z(ex, ey, velz1, velz2, rho1, rho2, _F);
    diagnose_Phi_z(ex, ey, velz1, velz2, _Phi_z);

    // diagnose the potential temperature (midpoint)
    VecZeroEntries(_theta_h);
    VecAXPY(_theta_h, 0.5, theta1);
    VecAXPY(_theta_h, 0.5, theta2);

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
    vo->AssembleLinearWithTheta(ex, ey, _theta_h, vo->VA);
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

        M1->assemble(kk, SCALE);
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
