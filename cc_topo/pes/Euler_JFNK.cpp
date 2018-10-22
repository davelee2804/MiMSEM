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
#include "L2Vecs.h"
#include "ElMats.h"
#include "VertOps.h"
#include "Assembly.h"
#include "Euler_JFNK.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
#define RAYLEIGH 0.2

using namespace std;

Euler::Euler(Topo* _topo, Geom* _geom, double _dt) {
    int ii, n2;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    do_visc = true;
    del2 = viscosity();
    vert_visc = viscosity_vert();
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

    // potential temperature projection operator
    T = new Whmat(topo, geom, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

    // assemble the vertical gradient and divergence incidence matrices
    vo = new VertOps(topo, geom);

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

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_t);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_b_l);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_t_l);

    Kv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->elOrd*topo->elOrd, &Kv[ii]);
    }
    Kh = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Kh[ii]);
    }
    gv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd*topo->elOrd, &gv[ii]);
    }
    zv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*topo->elOrd*topo->elOrd, &zv[ii]);
    }

    // initialise the single column mass matrices and solvers
    n2 = topo->elOrd*topo->elOrd;

    KSPCreate(MPI_COMM_SELF, &kspColA);
    KSPSetOperators(kspColA, vo->VA, vo->VA);
    //KSPSetTolerances(kspColA, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    //KSPSetType(kspColA, KSPGMRES);
    KSPGetPC(kspColA, &pc);
    //PCSetType(pc, PCBJACOBI);
    //PCBJacobiSetTotalBlocks(pc, n2, NULL);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspColA, "kspColA_");
    KSPSetFromOptions(kspColA);

    KSPCreate(MPI_COMM_WORLD, &kspE);
    KSPSetOperators(kspE, T->M, T->M);
    KSPSetTolerances(kspE, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspE, KSPGMRES);
    KSPGetPC(kspE, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, n2, NULL);
    KSPSetOptionsPrefix(kspE, "exner_");
    KSPSetFromOptions(kspE);

    initGZ();

    SetupVertOps();

    velz_prev  = new L2Vecs(geom->nk-1, topo, geom);
    rho_prev   = new L2Vecs(geom->nk, topo, geom);
    rt_prev    = new L2Vecs(geom->nk, topo, geom);
    exner_prev = new L2Vecs(geom->nk, topo, geom);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double Euler::viscosity() {
    double ae = 4.0*M_PI*RAD_EARTH*RAD_EARTH;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

double Euler::viscosity_vert() {
    int ii, kk;
    double dzMaxG, dzMax = 1.0e-6;

    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < topo->n0; ii++) {
            if(geom->thick[kk][ii] > dzMax) {
                dzMax = geom->thick[kk][ii];
            }
        }
    }
    MPI_Allreduce(&dzMax, &dzMaxG, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return 4.0*1.0*dzMax/M_PI;
    //return 1.0*1.0*dzMax/M_PI;
    //return 0.0;
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void Euler::coriolis() {
    int ii, kk;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fl, fxl, fxg, PtQfxg;

    // initialise the coriolis vector (local and global)
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    fg = new Vec[geom->nk];

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
    VecScatterEnd(topo->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);

    // project vector onto 0 forms
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PtQfxg);
    VecZeroEntries(PtQfxg);
    MatMult(PtQ->M, fxg, PtQfxg);
    // diagonal mass matrix as vector
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg[kk]);
        m0->assemble(kk, 1.0);
        VecPointwiseDivide(fg[kk], PtQfxg, m0->vg);
    }
    
    delete PtQ;
    VecDestroy(&fl);
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

void Euler::initGZ() {
    int ex, ey, ei, ii, kk, n2, mp12;
    int* inds0;
    int inds2k[99], inds0k[99];
    double det;
    double* WtQflat = new double[vo->W->nDofsJ*vo->Q->nDofsJ];
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

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);
            vo->Q->assemble(ex, ey);

            MatZeroEntries(BQ);
            for(kk = 0; kk < geom->nk; kk++) {
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    vo->Q0[ii][ii]  = vo->Q->A[ii][ii]*(SCALE/det);
                    // for linear field we multiply by the vertical jacobian determinant when
                    // integrating, and do no other trasformations for the basis functions
                    vo->Q0[ii][ii] *= 0.5;
                }
                Mult_FD_IP(vo->W->nDofsJ, vo->Q->nDofsJ, vo->W->nDofsI, vo->Wt, vo->Q0, vo->WtQ);
                Flat2D_IP(vo->W->nDofsJ, vo->Q->nDofsJ, vo->WtQ, WtQflat);

                for(ii = 0; ii < vo->W->nDofsJ; ii++) {
                    inds2k[ii] = ii + kk*vo->W->nDofsJ;
                }

                // assemble the first basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+0)*mp12;
                }
                MatSetValues(BQ, vo->W->nDofsJ, inds2k, vo->Q->nDofsJ, inds0k, WtQflat, ADD_VALUES);
                // assemble the second basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+1)*mp12;
                }
                MatSetValues(BQ, vo->W->nDofsJ, inds2k, vo->Q->nDofsJ, inds0k, WtQflat, ADD_VALUES);
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
}

Euler::~Euler() {
    int ii;

    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);
    KSPDestroy(&kspE);
    KSPDestroy(&kspColA);
    VecDestroy(&theta_b);
    VecDestroy(&theta_t);
    VecDestroy(&theta_b_l);
    VecDestroy(&theta_t_l);

    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&fg[ii]);
        VecDestroy(&Kh[ii]);
    }
    delete[] fg;
    delete[] Kh;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&Kv[ii]);
    }
    delete[] Kv;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
    }
    delete[] gv;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&zv[ii]);
    }
    delete[] zv;

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;
    delete T;

    delete edge;
    delete node;
    delete quad;

    delete vo;

    DestroyVertOps();

    delete velz_prev;
    delete rho_prev;
    delete rt_prev;
    delete exner_prev;
}

/*
*/
void Euler::AssembleKEVecs(Vec* velx, Vec* velz) {
    int ex, ey, ei, kk, n2;
    Mat BA;
    Vec velx_l, *Kh_l, Kv2;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &Kv2);

    // assemble the horiztonal operators
    Kh_l = new Vec[geom->nk];
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &velx_l);
    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(velx_l);
        VecScatterBegin(topo->gtol_1, velx[kk], velx_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, velx[kk], velx_l, INSERT_VALUES, SCATTER_FORWARD);
        K->assemble(velx_l, kk, SCALE);
        VecZeroEntries(Kh[kk]);
        MatMult(K->M, velx[kk], Kh[kk]);

        VecCreateSeq(MPI_COMM_SELF, topo->n2l, &Kh_l[kk]);
        VecScatterBegin(topo->gtol_2, Kh[kk], Kh_l[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, Kh[kk], Kh_l[kk], INSERT_VALUES, SCATTER_FORWARD);
    }
    VecDestroy(&velx_l);

    // update the vertical vector with the horiztonal vector
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            VecZeroEntries(Kv[ei]);
            HorizToVert2(ex, ey, Kh_l, Kv[ei]);
        }
    }

    // assemble the vertical operators
    MatCreate(MPI_COMM_SELF, &BA);
    MatSetType(BA, MATSEQAIJ);
    MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(BA, 2*n2, PETSC_NULL);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            MatZeroEntries(BA);
            ei = ey*topo->nElsX + ex;
            vo->AssembleConLinWithW(ex, ey, velz[ei], BA);

            VecZeroEntries(Kv2);
            MatMult(BA, velz[ei], Kv2);
            VecScale(Kv2, 0.5);

            // add the vertical contribution to the horiztonal vector
            VertToHoriz2(ex, ey, 0, geom->nk, Kv2, Kh_l);
        }
    }
    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, Kh_l[kk], Kh[kk], INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  topo->gtol_2, Kh_l[kk], Kh[kk], INSERT_VALUES, SCATTER_REVERSE);
    }

    VecDestroy(&Kv2);
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Kh_l[kk]);
    }
    delete[] Kh_l;
    MatDestroy(&BA);
}

/*
compute the right hand side for the momentum equation for a given level
note that the vertical velocity, uv, is stored as a different vector for 
each element
*/
void Euler::horizMomRHS(Vec uh, Vec* theta_l, Vec exner, int lev, Vec Fu, Vec Flux) {
    double dot;
    Vec wl, wi, Ru, Ku, Mh, d2u, d4u, theta_k, dExner, dp;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);

    // assemble the mass matrices for use in the weak form grad, curl and laplacian operators
    m0->assemble(lev, SCALE);
    M1->assemble(lev, SCALE);
    M2->assemble(lev, SCALE, true);

    curl(false, uh, &wi, lev, true);
    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(Fu);
    R->assemble(wl, lev, SCALE);
    MatMult(R->M, uh, Ru);
    MatMult(EtoF->E12, Kh[lev], Fu);
    VecAXPY(Fu, 1.0, Ru);

    // add the thermodynamic term (theta is in the same space as the vertical velocity)
    // project theta onto 1 forms
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta_l[lev+0]);
    VecAXPY(theta_k, 0.5, theta_l[lev+1]);

    grad(false, exner, &dExner, lev);
    F->assemble(theta_k, lev, false, SCALE);
    MatMult(F->M, dExner, dp);
    VecAXPY(Fu, 1.0, dp);
    VecDestroy(&dExner);

    // evaluate the kinetic to internal energy exchange diagnostic (for this layer)
    VecScale(dp, 1.0/SCALE);
    VecDot(Flux, dp, &dot);
    k2i += dot;

    // add in the biharmonic viscosity
    if(do_visc) {
        laplacian(false, uh, &d2u, lev);
        laplacian(false, d2u, &d4u, lev);
        VecZeroEntries(d2u);
        MatMult(M1->M, d4u, d2u);
        VecAXPY(Fu, 1.0, d2u);
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }

    VecDestroy(&wl);
    VecDestroy(&wi);
    VecDestroy(&Ru);
    VecDestroy(&Ku);
    VecDestroy(&Mh);
    VecDestroy(&dp);
    VecDestroy(&theta_k);
}

// pi is a local vector
void Euler::massRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* Flux) {
    int kk;
    Vec pu, Fh;

    // compute the horiztonal mass fluxes
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &pu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fh);

    for(kk = 0; kk < geom->nk; kk++) {
        F->assemble(pi[kk], kk, true, SCALE);
        M1->assemble(kk, SCALE);
        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fh);
        MatMult(EtoF->E21, Fh, Fp[kk]);

        VecZeroEntries(Flux[kk]);
        VecCopy(Fh, Flux[kk]);
    }

    VecDestroy(&pu);
    VecDestroy(&Fh);
}

void Euler::tempRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* exner) {
    int kk;
    double dot;
    Vec pu, Fh, dF, theta_l;

    // compute the horiztonal mass fluxes
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_l);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &pu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(theta_l);
        VecAXPY(theta_l, 0.5, pi[kk+0]);
        VecAXPY(theta_l, 0.5, pi[kk+1]);
        F->assemble(theta_l, kk, false, SCALE);

        M1->assemble(kk, SCALE);
        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fh);
        MatMult(EtoF->E21, Fh, Fp[kk]);

        // update the internal to kinetic energy flux diagnostic
        M2->assemble(kk, SCALE, true);
        MatMult(M2->M, Fp[kk], dF);
        VecScale(dF, 1.0/SCALE);
        VecDot(exner[kk], dF, &dot);
        i2k += dot;
    }

    VecDestroy(&pu);
    VecDestroy(&Fh);
    VecDestroy(&dF);
    VecDestroy(&theta_l);
}

/*
assemble the boundary condition vector for rho(t) X theta(0)
assume V0^{rho} has already been assembled (omitting internal levels)
*/
void Euler::thetaBCVec(int ex, int ey, Mat A, Vec* bTheta) {
    int* inds2 = topo->elInds2_l(ex, ey);
    int ii, n2;
    PetscScalar *vArray, *hArray;
    Vec theta_o;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &theta_o);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, bTheta);

    // assemble the theta bc vector
    VecZeroEntries(theta_o);
    VecGetArray(theta_o, &vArray);
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
    VecRestoreArray(theta_o, &vArray);

    MatMult(A, theta_o, *bTheta);
    VecDestroy(&theta_o);
}

/*
diagnose theta from rho X theta (with boundary condition)
note: rho, rhoTheta and theta are all LOCAL vectors
*/
void Euler::diagTheta(Vec* rho, Vec* rt, Vec* theta) {
    int ex, ey, n2, kk;
    Vec rtv, frt, theta_v, bcs;

    n2 = topo->elOrd*topo->elOrd;

    // reset the potential temperature at the internal layer interfaces, not the boundaries
    for(kk = 1; kk < geom->nk; kk++) {
        VecZeroEntries(theta[kk]);
    }

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &rtv);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &frt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &theta_v);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // construct horiztonal rho theta field
            VecZeroEntries(rtv);
            HorizToVert2(ex, ey, rt, rtv);
            vo->AssembleLinCon(ex, ey, vo->VAB);
            MatMult(vo->VAB, rtv, frt);

            // assemble in the bcs
            vo->AssembleLinearWithRho(ex, ey, rho, vo->VA, false);
            thetaBCVec(ex, ey, vo->VA, &bcs);
            VecAXPY(frt, -1.0, bcs);
            VecDestroy(&bcs);

            vo->AssembleLinearWithRho(ex, ey, rho, vo->VA, true);
            KSPSolve(kspColA, frt, theta_v);
            VertToHoriz2(ex, ey, 1, geom->nk, theta_v, theta);
        }
    }

    VecDestroy(&rtv);
    VecDestroy(&frt);
    VecDestroy(&theta_v);
}

// rho, rt and theta are all vertical vectors
void Euler::diagThetaVert(int ex, int ey, Mat AB, Vec rho, Vec rt, Vec theta) {
    int ii, kk;
    int n2 = topo->elOrd*topo->elOrd;
    int* inds2 = topo->elInds2_l(ex, ey);
    Vec frt, theta_v, bcs;
    PetscScalar *ttArray, *tbArray, *tiArray, *tjArray;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &frt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &theta_v);

    // construct horiztonal rho theta field
    vo->AssembleLinCon(ex, ey, AB);
    MatMult(AB, rt, frt);

    // assemble in the bcs
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, false);
    thetaBCVec(ex, ey, vo->VA, &bcs);
    VecAXPY(frt, -1.0, bcs);
    VecDestroy(&bcs);

    // map back to the full column
    vo->AssembleLinearWithRT(ex, ey, rho, vo->VA, true);
    KSPSolve(kspColA, frt, theta_v);

    VecGetArray(theta_v,   &tiArray);
    VecGetArray(theta,     &tjArray);
    for(kk = 0; kk < geom->nk-1; kk++) {
        for(ii = 0; ii < n2; ii++) {
            tjArray[(kk+1)*n2+ii] = tiArray[kk*n2+ii];
        }
    }
    VecRestoreArray(theta_v,   &tiArray);

    VecGetArray(theta_b_l, &tbArray);
    VecGetArray(theta_t_l, &ttArray);
    for(ii = 0; ii < n2; ii++) {
        tjArray[ii]             = tbArray[inds2[ii]];
        tjArray[geom->nk*n2+ii] = ttArray[inds2[ii]];
    }
    VecRestoreArray(theta_b_l, &tbArray);
    VecRestoreArray(theta_t_l, &ttArray);
    VecRestoreArray(theta,     &tjArray);

    VecDestroy(&frt);
    VecDestroy(&theta_v);
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

// rho and rt are local vectors, and exner is a global vector
void Euler::HorizRHS(Vec* velx, Vec* rho, Vec* rt, Vec* exner, Vec* Fu, Vec* Fp, Vec* Ft) {
    int kk;
    Vec* theta;
    Vec* Flux;

    k2i = i2k = 0.0;

    theta = new Vec[geom->nk+1];
    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta[kk]);
    }
    Flux = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Flux[kk]);
    }

    // set the top and bottom potential temperature bcs
    VecCopy(theta_b_l, theta[0]);
    VecCopy(theta_t_l, theta[geom->nk]);
    diagTheta(rho, rt, theta);

    massRHS(velx, rho, Fp, Flux);
    tempRHS(Flux, theta, Ft, exner);

    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx[kk], theta, exner[kk], kk, Fu[kk], Flux[kk]);
    }

    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecDestroy(&theta[kk]);
    }
    delete[] theta;
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Flux[kk]);
    }
    delete[] Flux;
}

Vec* CreateHorizVecs(Topo* topo, Geom* geom) {
    Vec* vecs = new Vec[geom->nk];

    for(int ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &vecs[ii]);
    }
    return vecs;
}

void DestroyHorizVecs(Vec* vecs, Geom* geom) {
    for(int ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&vecs[ii]);
    }
    delete[] vecs;
}

#define FIRST_STEP_EXPLICIT
//#define PROGNOSE_EXNER

void Euler::StrangCarryover(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int     ii;
    char    fieldname[100];
    Vec     wi, bu, xu;
    L2Vecs* velz_1  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_0   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_1   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_2   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_3   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_4   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_5   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_0    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_1    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_2    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_3    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_4    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_5    = new L2Vecs(geom->nk, topo, geom);
#ifdef PROGNOSE_EXNER
    L2Vecs* rt_e    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_1 = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_2 = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_3 = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_4 = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_e = new L2Vecs(geom->nk, topo, geom);
#endif
    L2Vecs* exner_i = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* Fp      = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* Ft      = new L2Vecs(geom->nk, topo, geom);
    Vec*    Fu      = CreateHorizVecs(topo, geom);
    Vec*    velx_2  = CreateHorizVecs(topo, geom);
    Vec*    velx_3  = CreateHorizVecs(topo, geom);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &xu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);

    if(firstStep) {
        VecScatterBegin(topo->gtol_2, theta_b, theta_b_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, theta_b, theta_b_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, theta_t, theta_t_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, theta_t, theta_t_l, INSERT_VALUES, SCATTER_FORWARD);
    }

    // 1.  explicit vertical solve using data from the end of the previous horizontal step
    if(!rank) cout << "vertical half step (1)..............." << endl;

    rho_0->CopyFromHoriz(rho);
    rho_0->UpdateLocal();
    rho_0->HorizToVert();
    rt_0->CopyFromHoriz(rt);
    rt_0->UpdateLocal();
    rt_0->HorizToVert();
    exner_i->CopyFromHoriz(exner);
    exner_i->UpdateLocal();
    exner_i->HorizToVert();

#ifdef FIRST_STEP_EXPLICIT
    if(firstStep) {
#endif
        AssembleKEVecs(velx, velz);

        velz_1->CopyFromVert(velz);
        rho_1->CopyFromVert(rho_0->vz);
        rt_1->CopyFromVert(rt_0->vz);

        VertSolve_JFNK(velz_1->vz, rho_1->vz, rt_1->vz, exner_i->vz, velz, rho_0->vz, rt_0->vz, exner_i->vz);
#ifdef FIRST_STEP_EXPLICIT
    } else {
        // use the same horizontal kinetic energy as the previous horizontal solve, so don't assemble here
        //VertSolve_Explicit(velz_1->vz, rho_1->vz, rt_1->vz, exner_i->vz,
        //                   velz,       rho_0->vz, rt_0->vz, exner_i->vz,
        //                   velz,       rho_0->vz, rt_0->vz, exner_i->vz);
        for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecCopy(velz[ii],      velz_1->vz[ii]);
            VecCopy(rho_0->vz[ii], rho_1->vz[ii] );
            VecCopy(rt_0->vz[ii],  rt_1->vz[ii]  );
            VecScale(velz_1->vz[ii],  2.0);
            VecScale(rho_1->vz[ii],   2.0);
            VecScale(rt_1->vz[ii],    2.0);
            VecScale(exner_i->vz[ii], 2.0);
            VecAXPY(velz_1->vz[ii],  -1.0, velz_prev->vz[ii] );
            VecAXPY(rho_1->vz[ii],   -1.0, rho_prev->vz[ii]  );
            VecAXPY(rt_1->vz[ii],    -1.0, rt_prev->vz[ii]   );
            VecAXPY(exner_i->vz[ii], -1.0, exner_prev->vz[ii]);
        }
    }
#endif
    rho_1->VertToHoriz();
    rho_1->UpdateGlobal();
    rt_1->VertToHoriz();
    rt_1->UpdateGlobal();
#ifdef PROGNOSE_EXNER
    exner_1->CopyFromVert(exner_i->vz);
    exner_1->VertToHoriz();
    exner_1->UpdateGlobal();
#else
    exner_i->VertToHoriz();
    exner_i->UpdateGlobal();
#endif

    // 2.1 first horiztonal substep
    if(!rank) cout << "horiztonal step (1).................." << endl;
    AssembleKEVecs(velx, velz_1->vz);
#ifdef PROGNOSE_EXNER
    HorizRHS(velx, rho_1->vl, rt_1->vl, exner_1->vh, Fu, Fp->vh, Ft->vh);
#else
    HorizRHS(velx, rho_1->vl, rt_1->vl, exner_i->vh, Fu, Fp->vh, Ft->vh);
#endif
    for(ii = 0; ii < geom->nk; ii++) {
        // momentum
        M1->assemble(ii, SCALE);
        MatMult(M1->M, velx[ii], bu);
        VecAXPY(bu, -dt, Fu[ii]);
        KSPSolve(ksp1, bu, velx_2[ii]);

        // continuity
        VecCopy(rho_1->vh[ii], rho_2->vh[ii]);
        VecAXPY(rho_2->vh[ii], -dt, Fp->vh[ii]);

        // internal energy
        VecCopy(rt_1->vh[ii], rt_2->vh[ii]);
        VecAXPY(rt_2->vh[ii], -dt, Ft->vh[ii]);
    }
    rho_2->UpdateLocal();
    rt_2->UpdateLocal();
    rt_2->HorizToVert();

#ifdef PROGNOSE_EXNER
    Ft->UpdateLocal();
    SolveExner(rt_1->vl, Ft->vl, exner_1->vh, exner_2->vh, dt);
#endif

    // 2.2 second horiztonal step
    if(!rank) cout << "horiztonal step (2).................." << endl;
    AssembleKEVecs(velx_2, velz_1->vz);
#ifdef PROGNOSE_EXNER
    HorizRHS(velx_2, rho_2->vl, rt_2->vl, exner_2->vh, Fu, Fp->vh, Ft->vh);
#else
    DiagExner(rt_2->vz, exner_i);
    HorizRHS(velx_2, rho_2->vl, rt_2->vl, exner_i->vh, Fu, Fp->vh, Ft->vh);
#endif
    for(ii = 0; ii < geom->nk; ii++) {
        // momentum
        VecZeroEntries(xu);
        VecAXPY(xu, 0.75, velx[ii]);
        VecAXPY(xu, 0.25, velx_2[ii]);
        M1->assemble(ii, SCALE);
        MatMult(M1->M, xu, bu);
        VecAXPY(bu, -0.25*dt, Fu[ii]);
        KSPSolve(ksp1, bu, velx_3[ii]);

        // continuity
        VecZeroEntries(rho_3->vh[ii]);
        VecAXPY(rho_3->vh[ii], 0.75, rho_1->vh[ii]);
        VecAXPY(rho_3->vh[ii], 0.25, rho_2->vh[ii]);
        VecAXPY(rho_3->vh[ii], -0.25*dt, Fp->vh[ii]);

        // internal energy
        VecZeroEntries(rt_3->vh[ii]);
        VecAXPY(rt_3->vh[ii], 0.75, rt_1->vh[ii]);
        VecAXPY(rt_3->vh[ii], 0.25, rt_2->vh[ii]);
        VecAXPY(rt_3->vh[ii], -0.25*dt, Ft->vh[ii]);
    }
    rho_3->UpdateLocal();
    rt_3->UpdateLocal();
    rt_3->HorizToVert();

#ifdef PROGNOSE_EXNER
    Ft->UpdateLocal();
    for(ii = 0; ii < geom->nk; ii++) {
        VecZeroEntries(exner_e->vh[ii]);
        VecAXPY(exner_e->vh[ii], 0.75, exner_1->vh[ii]);
        VecAXPY(exner_e->vh[ii], 0.25, exner_2->vh[ii]);
        VecZeroEntries(rt_e->vl[ii]);
        VecAXPY(rt_e->vl[ii], 0.75, rt_1->vl[ii]);
        VecAXPY(rt_e->vl[ii], 0.25, rt_2->vl[ii]);
    }
    SolveExner(rt_e->vl, Ft->vl, exner_e->vh, exner_3->vh, dt);
#endif

    // 2.3 third horiztonal step
    if(!rank) cout << "horiztonal step (3).................." << endl;
    AssembleKEVecs(velx_3, velz_1->vz);
#ifdef PROGNOSE_EXNER
    HorizRHS(velx_3, rho_3->vl, rt_3->vl, exner_3->vh, Fu, Fp->vh, Ft->vh);
#else
    DiagExner(rt_3->vz, exner_i);
    HorizRHS(velx_3, rho_3->vl, rt_3->vl, exner_i->vh, Fu, Fp->vh, Ft->vh);
#endif
    for(ii = 0; ii < geom->nk; ii++) {
        // momentum
        VecZeroEntries(xu);
        VecAXPY(xu, 1.0/3.0, velx[ii]);
        VecAXPY(xu, 2.0/3.0, velx_3[ii]);
        M1->assemble(ii, SCALE);
        MatMult(M1->M, xu, bu);
        VecAXPY(bu, (-2.0/3.0)*dt, Fu[ii]);
        KSPSolve(ksp1, bu, velx[ii]);

        // continuity
        VecZeroEntries(rho_4->vh[ii]);
        VecAXPY(rho_4->vh[ii], 1.0/3.0, rho_1->vh[ii]);
        VecAXPY(rho_4->vh[ii], 2.0/3.0, rho_3->vh[ii]);
        VecAXPY(rho_4->vh[ii], (-2.0/3.0)*dt, Fp->vh[ii]);

        // internal energy
        VecZeroEntries(rt_4->vh[ii]);
        VecAXPY(rt_4->vh[ii], 1.0/3.0, rt_1->vh[ii]);
        VecAXPY(rt_4->vh[ii], 2.0/3.0, rt_3->vh[ii]);
        VecAXPY(rt_4->vh[ii], (-2.0/3.0)*dt, Ft->vh[ii]);
    }
    rho_4->UpdateLocal();
    rho_4->HorizToVert();
    rt_4->UpdateLocal();
    rt_4->HorizToVert();

#ifdef PROGNOSE_EXNER
    Ft->UpdateLocal();
    for(ii = 0; ii < geom->nk; ii++) {
        VecZeroEntries(exner_e->vh[ii]);
        VecAXPY(exner_e->vh[ii], 1.0/3.0, exner_1->vh[ii]);
        VecAXPY(exner_e->vh[ii], 2.0/3.0, exner_3->vh[ii]);
        VecZeroEntries(rt_e->vl[ii]);
        VecAXPY(rt_e->vl[ii], 1.0/3.0, rt_1->vl[ii]);
        VecAXPY(rt_e->vl[ii], 2.0/3.0, rt_3->vl[ii]);
    }
    SolveExner(rt_e->vl, Ft->vl, exner_e->vh, exner_4->vh, dt);
    exner_4->UpdateLocal();
    exner_4->HorizToVert();
    exner_4->CopyToVert(exner_i->vz);
#endif

    // carry over the vertical fields to the next time level
    velz_prev->CopyFromVert(velz_1->vz);
    rho_prev->CopyFromVert(rho_4->vz);
    rt_prev->CopyFromVert(rt_4->vz);
#ifdef PROGNOSE_EXNER
    exner_prev->CopyFromVert(exner_4->vz);
#else
    DiagExner(rt_4->vz, exner_i);
    exner_prev->CopyFromVert(exner_i->vz);
#endif

    // 3.  second vertical half step
    if(!rank) cout << "vertical half step (2)..............." << endl;
    AssembleKEVecs(velx, velz_1->vz);
    velz_1->CopyToVert(velz);
    rho_5->CopyFromVert(rho_4->vz);
    rt_5->CopyFromVert(rt_4->vz);
    VertSolve_JFNK(velz, rho_5->vz, rt_5->vz, exner_i->vz, velz_1->vz, rho_4->vz, rt_4->vz, exner_i->vz);

    // update input vectors
    rho_5->VertToHoriz();
    rho_5->UpdateGlobal();
    rho_5->CopyToHoriz(rho);
    rt_5->VertToHoriz();
    rt_5->UpdateGlobal();
    rt_5->CopyToHoriz(rt);
    exner_i->VertToHoriz();
    exner_i->UpdateGlobal();
    exner_i->CopyToHoriz(exner);

    firstStep = false;

    diagnostics(velx, velz, rho, rt, exner);

    // write output
    if(save) {
        step++;

        L2Vecs* l2Theta = new L2Vecs(geom->nk+1, topo, geom);
        VecCopy(theta_b_l, l2Theta->vl[0]       );
        VecCopy(theta_t_l, l2Theta->vl[geom->nk]);
        diagTheta(rho_5->vl, rt_5->vl, l2Theta->vl);

        l2Theta->UpdateGlobal();
        for(ii = 0; ii < geom->nk+1; ii++) {
            sprintf(fieldname, "theta");
            geom->write2(l2Theta->vh[ii], fieldname, step, ii, false);
        }
        delete l2Theta;

        for(ii = 0; ii < geom->nk; ii++) {
            curl(true, velx[ii], &wi, ii, false);

            sprintf(fieldname, "vorticity");
            geom->write0(wi, fieldname, step, ii);
            sprintf(fieldname, "velocity_h");
            geom->write1(velx[ii], fieldname, step, ii);
            sprintf(fieldname, "density");
            geom->write2(rho[ii], fieldname, step, ii, true);
            sprintf(fieldname, "rhoTheta");
            geom->write2(rt[ii], fieldname, step, ii, true);
            sprintf(fieldname, "exner");
            geom->write2(exner[ii], fieldname, step, ii, true);

            VecDestroy(&wi);
        }
        sprintf(fieldname, "velocity_z");
        geom->writeVertToHoriz(velz, fieldname, step, geom->nk-1);
    }

    VecDestroy(&xu);
    VecDestroy(&bu);
    delete velz_1;
    delete rho_0;
    delete rho_1;
    delete rho_2;
    delete rho_3;
    delete rho_4;
    delete rho_5;
    delete rt_0;
    delete rt_1;
    delete rt_2;
    delete rt_3;
    delete rt_4;
    delete rt_5;
#ifdef PROGNOSE_EXNER
    delete rt_e;
    delete exner_1;
    delete exner_2;
    delete exner_3;
    delete exner_4;
    delete exner_e;
#endif
    delete exner_i;
    delete Fp;
    delete Ft;
    DestroyHorizVecs(Fu, geom);
    DestroyHorizVecs(velx_2, geom);
    DestroyHorizVecs(velx_3, geom);
}

void Euler::VertToHoriz2(int ex, int ey, int ki, int kf, Vec pv, Vec* ph) {
    int ii, kk, n2;
    int* inds2 = topo->elInds2_l(ex, ey);
    PetscScalar *hArray, *vArray;

    n2 = topo->elOrd*topo->elOrd;

    VecGetArray(pv, &vArray);
    for(kk = ki; kk < kf; kk++) {
        VecGetArray(ph[kk], &hArray);
        for(ii = 0; ii < n2; ii++) {
            hArray[inds2[ii]] += vArray[(kk-ki)*n2+ii];
        }
        VecRestoreArray(ph[kk], &hArray);
    }
    VecRestoreArray(pv, &vArray);
}

void Euler::HorizToVert2(int ex, int ey, Vec* ph, Vec pv) {
    int ii, kk, n2;
    int* inds2 = topo->elInds2_l(ex, ey);
    PetscScalar *hArray, *vArray;

    n2 = topo->elOrd*topo->elOrd;

    VecGetArray(pv, &vArray);
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(ph[kk], &hArray);
        for(ii = 0; ii < n2; ii++) {
            vArray[kk*n2+ii] += hArray[inds2[ii]];
        }
        VecRestoreArray(ph[kk], &hArray);
    }
    VecRestoreArray(pv, &vArray);
}

void Euler::init0(Vec* q, ICfunc3D* func) {
    int ex, ey, ii, kk, mp1, mp12;
    int* inds0;
    PtQmat* PQ = new PtQmat(topo, geom, node);
    PetscScalar *bArray;
    Vec bl, bg, PQb;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PQb);

    for(kk = 0; kk < geom->nk; kk++) {
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
        VecScatterEnd(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

        m0->assemble(kk, 1.0);
        MatMult(PQ->M, bg, PQb);
        VecPointwiseDivide(q[kk], PQb, m0->vg);
    }

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
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
        VecScatterEnd(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

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
        VecScatterEnd(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

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
    VecScatterEnd(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    M2->assemble(0, SCALE, false);
    MatMult(WQ->M, bg, WQb);
    VecScale(WQb, SCALE);
    KSPSolve(ksp2, WQb, theta);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void Euler::diagnostics(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner) {
    char filename[80];
    ofstream file;
    int kk, ex, ey, ei, n2;
    double keh, kev, pe, ie, k2p, p2k;
    double dot, loc1, loc2, loc3;
    Vec hu, M2Pi, w2, gRho, gi, zi;
    Mat BA;
    L2Vecs* l2_rho = new L2Vecs(geom->nk, topo, geom);

    n2 = topo->elOrd*topo->elOrd;

    keh = kev = pe = ie = k2p = p2k = 0.0;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &w2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &gRho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &gi);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &zi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &hu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &M2Pi);

    MatCreate(MPI_COMM_SELF, &BA);
    MatSetType(BA, MATSEQAIJ);
    MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(BA, 2*n2, PETSC_NULL);

    l2_rho->CopyFromHoriz(rho);
    l2_rho->UpdateLocal();
    l2_rho->HorizToVert();

    // horiztonal kinetic energy
    for(kk = 0; kk < geom->nk; kk++) {
        F->assemble(l2_rho->vl[kk], kk, true, SCALE);
        MatMult(F->M, velx[kk], hu);
        VecScale(hu, 1.0/SCALE);
        VecDot(hu, velx[kk], &dot);
        keh += dot;
    }

    // vertical kinetic energy and kinetic to potential exchange
    loc1 = loc2 = loc3 = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            MatZeroEntries(BA);
            vo->AssembleConLinWithW(ex, ey, velz[ei], BA);
            MatMult(BA, velz[ei], w2);
            VecScale(w2, 1.0/SCALE);
            VecDot(l2_rho->vz[ei], w2, &dot);
            loc1 += dot;

            vo->AssembleLinearWithRT(ex, ey, l2_rho->vz[ei], vo->VA, true);
            MatMult(vo->VA, velz[ei], zi);
            vo->AssembleLinearInv(ex, ey, vo->VA);
            MatMult(vo->VA, zi, gi);
            VecDot(gi, gv[ei], &dot);
            loc2 += dot/SCALE;

            MatMult(vo->V10, gi, w2);
            VecDot(w2, zv[ei], &dot);
            loc3 += dot/SCALE;
        }
    }
    MPI_Allreduce(&loc1, &kev, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&loc2, &k2p, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&loc3, &p2k, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // internal energy
    for(kk = 0; kk < geom->nk; kk++) {
        M2->assemble(kk, SCALE, true);
        MatMult(M2->M, exner[kk], M2Pi);
        VecScale(M2Pi, 1.0/SCALE);
        VecDot(rt[kk], M2Pi, &dot);
        ie += (CV/CP)*dot;
    }

    // potential energy
    loc1 = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            VecDot(zv[ei], l2_rho->vz[ei], &dot);
            loc1 += dot/SCALE;
        }
    }
    MPI_Allreduce(&loc1, &pe,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(!rank) {
        sprintf(filename, "output/energetics.dat");
        file.open(filename, ios::out | ios::app);
        file.precision(16);
        file << keh << "\t";
        file << kev << "\t";
        file << pe  << "\t";
        file << ie  << "\t";
        file << k2p << "\t";
        file << p2k << "\t";
        file << k2i << "\t";
        file << i2k << "\t";
        file << k2i_z << "\t";
        file << i2k_z << "\t";
        file << endl;
        file.close();
    }

    VecDestroy(&M2Pi);
    VecDestroy(&hu);
    VecDestroy(&w2);
    VecDestroy(&gRho);
    VecDestroy(&gi);
    VecDestroy(&zi);
    MatDestroy(&BA);
    delete l2_rho;
}

void Euler::SetupVertOps() {
    int n2 = topo->elOrd*topo->elOrd;
    
    MatCreate(MPI_COMM_SELF, &VA_theta);
    MatSetType(VA_theta, MATSEQAIJ);
    MatSetSizes(VA_theta, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VA_theta, (geom->nk-1)*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VA_rho);
    MatSetType(VA_rho, MATSEQAIJ);
    MatSetSizes(VA_rho, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VA_rho, (geom->nk-1)*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VB_Pi);
    MatSetType(VB_Pi, MATSEQAIJ);
    MatSetSizes(VB_Pi, (geom->nk+0)*n2, (geom->nk+0)*n2, (geom->nk+0)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(VB_Pi, (geom->nk-1)*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VB_rt);
    MatSetType(VB_rt, MATSEQAIJ);
    MatSetSizes(VB_rt, (geom->nk+0)*n2, (geom->nk+0)*n2, (geom->nk+0)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(VB_rt, (geom->nk-1)*n2, PETSC_NULL);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &pTmp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &pTmp2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &wTmp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &pGrad);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &wRho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &wTheta);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &dwTheta);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &theta);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &wNew);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &fw);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &bw);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &rhoNew);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &fRho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &bRho);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &rtNew);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &fRT);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &bRT);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &exnerNew);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &fExner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &bExner);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &eosRhs);

    // preconditioner matrices
    DW2 = NULL;
    DTVB = NULL;
    VA_invDTVB = NULL;
    GRAD = NULL;
    VA_invVA_rt = NULL;
    DVA_invVA_rt = NULL;
    VB_PiDVA_invVA_rt = NULL;
    DIV = NULL;
    LAP = NULL;
}

void Euler::DestroyVertOps() {
    MatDestroy(&VA_theta);
    MatDestroy(&VA_rho);
    MatDestroy(&VB_Pi);
    MatDestroy(&VB_rt);

    VecDestroy(&pTmp);
    VecDestroy(&pTmp2);
    VecDestroy(&wTmp);
    VecDestroy(&pGrad);
    VecDestroy(&wRho);
    VecDestroy(&wTheta);
    VecDestroy(&dwTheta);
    VecDestroy(&theta);
    VecDestroy(&wNew);
    VecDestroy(&fw);
    VecDestroy(&bw);
    VecDestroy(&rhoNew);
    VecDestroy(&fRho);
    VecDestroy(&bRho);
    VecDestroy(&rtNew);
    VecDestroy(&fRT);
    VecDestroy(&bRT);
    VecDestroy(&exnerNew);
    VecDestroy(&fExner);
    VecDestroy(&bExner);
    VecDestroy(&eosRhs);

    // preconditioner matrices
    //MatDestroy(&DW2);
    MatDestroy(&DTVB);
    MatDestroy(&VA_invDTVB);
    MatDestroy(&GRAD);
    MatDestroy(&VA_invVA_rt);
    MatDestroy(&DVA_invVA_rt);
    MatDestroy(&VB_PiDVA_invVA_rt);
    MatDestroy(&DIV);
    MatDestroy(&LAP);
}

void UnpackX(Vec x, Vec w, Vec rho, Vec rt, Vec exner, int n2, int nk) {
    int ii, kk, index;
    const PetscScalar *xArray;
    PetscScalar *wArray, *exnerArray, *rhoArray, *rtArray;

    VecGetArrayRead(x, &xArray);
    VecGetArray(w, &wArray);
    VecGetArray(exner, &exnerArray);
    VecGetArray(rho, &rhoArray);
    VecGetArray(rt, &rtArray);

    index = 0;
    for(kk = 0; kk < nk; kk++) {
        for(ii = 0; ii < n2; ii++) {
            exnerArray[kk*n2+ii] = xArray[index++];
            rhoArray[kk*n2+ii]   = xArray[index++];
            rtArray[kk*n2+ii]    = xArray[index++];
        }
        if(kk < nk-1) {
            for(ii = 0; ii < n2; ii++) {
                wArray[kk*n2+ii] = xArray[index++];
            }
        }
    }

    VecRestoreArray(w, &wArray);
    VecRestoreArray(exner, &exnerArray);
    VecRestoreArray(rho, &rhoArray);
    VecRestoreArray(rt, &rtArray);
    VecRestoreArrayRead(x, &xArray);
}

void RepackX(Vec x, Vec w, Vec rho, Vec rt, Vec exner, int n2, int nk) {
    int ii, kk, index;
    PetscScalar *xArray, *wArray, *exnerArray, *rhoArray, *rtArray;

    VecGetArray(x, &xArray);
    VecGetArray(w, &wArray);
    VecGetArray(exner, &exnerArray);
    VecGetArray(rho, &rhoArray);
    VecGetArray(rt, &rtArray);

    index = 0;
    for(kk = 0; kk < nk; kk++) {
        for(ii = 0; ii < n2; ii++) {
            xArray[index++] = exnerArray[kk*n2+ii];
            xArray[index++] = rhoArray[kk*n2+ii];
            xArray[index++] = rtArray[kk*n2+ii];
        }
        if(kk < nk-1) {
            for(ii = 0; ii < n2; ii++) {
                xArray[index++] = wArray[kk*n2+ii];
            }
        }
    }

    VecRestoreArray(w, &wArray);
    VecRestoreArray(exner, &exnerArray);
    VecRestoreArray(rho, &rhoArray);
    VecRestoreArray(rt, &rtArray);
    VecRestoreArray(x, &xArray);
}

#define ADV_MASS_MAT
#define VERT_MASS_INV

PetscErrorCode _snes_function(SNES snes, Vec x, Vec f, void* ctx) {
    Euler* euler = (Euler*)ctx;
    int n2 = euler->topo->elOrd*euler->topo->elOrd;
    Geom* geom = euler->geom;
    VertOps* vo = euler->vo;

    // unpack the solution
    UnpackX(x, euler->wNew, euler->rhoNew, euler->rtNew, euler->exnerNew, n2, geom->nk);

    VecScale(euler->wNew, SCALE);
    VecScale(euler->rhoNew, SCALE);
    VecScale(euler->rtNew, SCALE);
    VecScale(euler->exnerNew, SCALE);

    // assemble the vertical velocity - first term
    MatMult(vo->VA, euler->wNew, euler->fw);

    // second term
    vo->AssembleConLinWithW(euler->eX, euler->eY, euler->wNew, vo->VBA);
    MatMult(vo->VBA, euler->wNew, euler->pTmp);
    MatMult(vo->V01, euler->pTmp, euler->wTmp);
    VecAXPY(euler->fw, +0.25*euler->dt, euler->wTmp);

    // pressure gradient term
    MatMult(vo->VB, euler->exnerNew, euler->pTmp);
    MatMult(vo->V01, euler->pTmp, euler->wTmp);
#ifdef VERT_MASS_INV
    MatMult(vo->VA_inv, euler->wTmp, euler->pGrad);
#else
    KSPSolve(euler->kspColA, euler->wTmp, euler->pGrad);
#endif
    euler->diagThetaVert(euler->eX, euler->eY, vo->VAB, euler->rhoNew, euler->rtNew, euler->theta);
    vo->AssembleLinearWithTheta(euler->eX, euler->eY, euler->theta, euler->VA_theta);
    MatMult(euler->VA_theta, euler->pGrad, euler->wTmp);
    VecAXPY(euler->fw, 0.5*euler->dt, euler->wTmp);

    // viscous term 
    MatMult(euler->VISC, euler->wNew, euler->wTmp);
    VecAXPY(euler->fw, -0.5*euler->dt*euler->vert_visc, euler->wTmp);

    // rayleigh friction
    MatMult(vo->VR, euler->wNew, euler->wTmp);
    VecAXPY(euler->fw, +0.5*euler->dt*RAYLEIGH, euler->wTmp);

    // assemble the density
    vo->AssembleLinearWithRT(euler->eX, euler->eY, euler->rhoNew, euler->VA_rho, true);
    MatMult(euler->VA_rho, euler->wNew, euler->wTmp);
#ifdef VERT_MASS_INV
    MatMult(vo->VA_inv, euler->wTmp, euler->wRho);
#else
    KSPSolve(euler->kspColA, euler->wTmp, euler->wRho);
#endif
    MatMult(vo->V10, euler->wRho, euler->pTmp);
#ifdef ADV_MASS_MAT
    VecAYPX(euler->pTmp, 0.5*euler->dt, euler->rhoNew);
    MatMult(vo->VB, euler->pTmp, euler->fRho);
#else
    VecCopy(euler->rhoNew, euler->fRho);
    VecAXPY(euler->fRho, 0.5*euler->dt, euler->pTmp);
#endif

    // assemble the density weighted potential temperature
    MatMult(euler->VA_theta, euler->wRho, euler->wTmp);
    //vo->AssembleLinearWithRT(euler->eX, euler->eY, euler->rtNew, euler->VA_rho, true);
    //MatMult(euler->VA_rho, euler->wNew, euler->wTmp);
#ifdef VERT_MASS_INV
    MatMult(vo->VA_inv, euler->wTmp, euler->wTheta);
#else
    KSPSolve(euler->kspColA, euler->wTmp, euler->wTheta);
#endif
    MatMult(vo->V10, euler->wTheta, euler->dwTheta);
#ifdef ADV_MASS_MAT
    VecCopy(euler->rtNew, euler->pTmp);
    VecAXPY(euler->pTmp, 0.5*euler->dt, euler->dwTheta);
    MatMult(vo->VB, euler->pTmp, euler->fRT);
#else
    VecCopy(euler->rtNew, euler->fRT);
    VecAXPY(euler->fRT, 0.5*euler->dt, euler->dwTheta);
#endif

    // assemble the exner pressure
    MatMult(vo->VB, euler->exnerNew, euler->fExner);
    vo->Assemble_EOS_RHS(euler->eX, euler->eY, euler->rtNew, euler->eosRhs);
    VecAXPY(euler->fExner, -1.0, euler->eosRhs);
    //VecScale(euler->fExner, 0.5*euler->dt);

    // assemble f
    RepackX(f, euler->fw, euler->fRho, euler->fRT, euler->fExner, n2, geom->nk);

    VecScale(f, 1.0/SCALE);
    
    return 0;
}

void Euler::AssemblePreconditioner(Mat P) {
    int ii, kk, index;
    int n2 = topo->elOrd*topo->elOrd;
    int inds_w[10000], inds_rho[10000], inds_rt[10000], inds_exner[10000];
    int inds2[100];
    PetscInt nc;
    const PetscInt *colInds;
    const PetscScalar *colVals;
    double vals2[100];

    // generate index sets
    index = 0;
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < n2; ii++) {
            inds_exner[kk*n2+ii] = index++;
            inds_rho[kk*n2+ii] = index++;
            inds_rt[kk*n2+ii] = index++;
        }
        if(kk < geom->nk-1) {
            for(ii = 0; ii < n2; ii++) {
                inds_w[kk*n2+ii] = index++;
            }
        }
    }

    // assemble the velocity PC
    diagThetaVert(eX, eY, vo->VAB, rhoNew, rtNew, theta);
    vo->AssembleLinearWithTheta(eX, eY, theta, VA_theta);
    vo->AssembleLinearWithRT(eX, eY, rtNew, VA_rho, true);
    vo->AssembleConstWithRho(eX, eY, exnerNew, VB_Pi);
    vo->AssembleConstWithRhoInv(eX, eY, rtNew, VB_rt);
    vo->AssembleConLinWithW(eX, eY, wNew, vo->VBA);

    if(!DTVB) {    
        MatMatMult(vo->V01, vo->VB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DTVB);
        MatMatMult(vo->VA_inv, DTVB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VA_invDTVB);
        MatMatMult(VA_theta, VA_invDTVB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &GRAD);

        MatMatMult(vo->VA_inv, VA_rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VA_invVA_rt);
        MatMatMult(vo->V10, VA_invVA_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DVA_invVA_rt);
        MatMatMult(VB_Pi, DVA_invVA_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VB_PiDVA_invVA_rt);
        MatMatMult(VB_rt, VB_PiDVA_invVA_rt, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DIV);

        MatMatMult(GRAD, DIV, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &LAP);

        //MatMatMult(vo->V01, vo->VBA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DW2);
    } else {
        MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DTVB);
        MatMatMult(vo->VA_inv, DTVB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &VA_invDTVB);
        MatMatMult(VA_theta, VA_invDTVB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &GRAD);

        MatMatMult(vo->VA_inv, VA_rho, MAT_REUSE_MATRIX, PETSC_DEFAULT, &VA_invVA_rt);
        MatMatMult(vo->V10, VA_invVA_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DVA_invVA_rt);
        MatMatMult(VB_Pi, DVA_invVA_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &VB_PiDVA_invVA_rt);
        MatMatMult(VB_rt, VB_PiDVA_invVA_rt, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DIV);

        MatMatMult(GRAD, DIV, MAT_REUSE_MATRIX, PETSC_DEFAULT, &LAP);

        //MatMatMult(vo->V01, vo->VBA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DW2);
    }

    MatAYPX(LAP, -0.25*dt*dt*RD/CV, vo->VA, SAME_NONZERO_PATTERN);

    MatZeroEntries(P);

    // copy the vertical velocity preconditioner
    for(kk = 0; kk < n2*(geom->nk - 1); kk++) {
/*
        MatGetRow(DW2, kk, &nc, &colInds, &colVals);
        for(ii = 0; ii < nc; ii++) {
            inds2[ii] = inds_w[colInds[ii]];
            vals2[ii] = 0.5*dt*colVals[ii];
            //vals2[ii] = colVals[ii];
        }
        MatSetValues(P, 1, &inds_w[kk], nc, inds2, vals2, ADD_VALUES);
        MatRestoreRow(DW2, kk, &nc, &colInds, &colVals);
*/
        MatGetRow(LAP, kk, &nc, &colInds, &colVals);
        for(ii = 0; ii < nc; ii++) {
            inds2[ii] = inds_w[colInds[ii]];
            vals2[ii] = colVals[ii];
        }
        MatSetValues(P, 1, &inds_w[kk], nc, inds2, vals2, ADD_VALUES);
        MatRestoreRow(LAP, kk, &nc, &colInds, &colVals);
    }

    for(kk = 0; kk < n2*(geom->nk); kk++) {
        MatGetRow(vo->VB, kk, &nc, &colInds, &colVals);
        for(ii = 0; ii < nc; ii++) {
            inds2[ii] = inds_rho[colInds[ii]];
#ifdef ADV_MASS_MAT
            vals2[ii] = colVals[ii];
#else
            vals2[ii] = 1.0;
#endif
        }
        MatSetValues(P, 1, &inds_rho[kk], nc, inds2, vals2, ADD_VALUES);
        for(ii = 0; ii < nc; ii++) {
            inds2[ii] = inds_rt[colInds[ii]];
#ifdef ADV_MASS_MAT
            vals2[ii] = colVals[ii];
#else
            vals2[ii] = 1.0;
#endif
        }
        MatSetValues(P, 1, &inds_rt[kk], nc, inds2, vals2, ADD_VALUES);
        for(ii = 0; ii < nc; ii++) {
            inds2[ii] = inds_exner[colInds[ii]];
            vals2[ii] = colVals[ii];
        }
        MatSetValues(P, 1, &inds_exner[kk], nc, inds2, vals2, ADD_VALUES);
        MatRestoreRow(vo->VB, kk, &nc, &colInds, &colVals);
    }

    MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
}

PetscErrorCode _snes_jacobian(SNES snes, Vec x, Mat J, Mat P, void* ctx) {
    Euler* euler = (Euler*)ctx;

    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

    euler->AssemblePreconditioner(P);

    euler->iT++;

    return 0;
}

void Euler::VertSolve_JFNK(Vec* velz, Vec* rho, Vec* rt, Vec* exner, Vec* velz_n, Vec* rho_n, Vec* rt_n, Vec* exner_n) {
    int ei, its;
    int n2 = topo->elOrd*topo->elOrd;
    int nf = (4*geom->nk - 1)*n2;
    double loc1, loc2, dot, norm;
    Vec x, b, f;
    Mat J, P;
    Mat DTV1 = NULL;
    SNES snes;
    SNESConvergedReason reason;

    VISC = NULL;

    VecCreateSeq(MPI_COMM_SELF, nf, &x);
    VecCreateSeq(MPI_COMM_SELF, nf, &b);
    VecCreateSeq(MPI_COMM_SELF, nf, &f);

    MatCreate(MPI_COMM_SELF, &P);
    MatSetType(P, MATSEQAIJ);
    MatSetSizes(P, nf, nf, nf, nf);
    MatSeqAIJSetPreallocation(P, 8*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &J);
    MatSetType(J, MATSEQAIJ);
    MatSetSizes(J, nf, nf, nf, nf);
    MatSeqAIJSetPreallocation(J, 8*n2, PETSC_NULL);

    SNESCreate(MPI_COMM_SELF, &snes);
    SNESSetFunction(snes, f, _snes_function, (void*)this);
    SNESSetJacobian(snes, J, P, _snes_jacobian, (void*)this);
    //SNESSetTolerances(snes, 1.0e-12, 1.0e-12, 1.0e-8, 200, 2000);
    SNESSetFromOptions(snes);

    loc1 = loc2 = k2i_z = i2k_z = 0.0;

    for(eY = 0; eY < topo->nElsX; eY++) {
        for(eX = 0; eX < topo->nElsX; eX++) {
            ei = eY*topo->nElsX + eX;
            iT = 0;

            vo->AssembleLinear(eX, eY, vo->VA);
            vo->AssembleLinearInv(eX, eY, vo->VA_inv);
            vo->AssembleConst(eX, eY, vo->VB);
            vo->AssembleLinCon(eX, eY, vo->VAB);
            vo->AssembleRayleigh(eX, eY, vo->VR);

            if(!ei) {
                MatMatMult(vo->V01, vo->VB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DTV1);
                MatMatMult(DTV1, vo->V10, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC);
            } else {
                MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DTV1);
                MatMatMult(DTV1, vo->V10, MAT_REUSE_MATRIX, PETSC_DEFAULT, &VISC);
            }

            MatMult(vo->VA, velz_n[ei], bw);
            VecAXPY(bw, -0.5*dt, gv[ei]);
            MatMult(vo->V01, Kv[ei], wTmp);
            VecAXPY(bw, -0.5*dt, wTmp);
            
#ifdef ADV_MASS_MAT
            MatMult(vo->VB, rho_n[ei], bRho);
            MatMult(vo->VB, rt_n[ei], bRT);
#else
            VecCopy(rho_n[ei], bRho);
            VecCopy(rt_n[ei], bRT);
#endif

            VecZeroEntries(bExner);

            RepackX(b, bw, bRho, bRT, bExner, n2, geom->nk);
            RepackX(x, velz[ei], rho[ei], rt[ei], exner[ei], n2, geom->nk);

            VecScale(x, 1.0/SCALE);
            VecScale(b, 1.0/SCALE);
            SNESSolve(snes, b, x);
            VecScale(x, SCALE);
            UnpackX(x, velz[ei], rho[ei], rt[ei], exner[ei], n2, geom->nk);

            // kinetic to internal energy exchange diagnostics
            MatMult(vo->VA, pGrad, wTmp);
            VecScale(wTmp, 1.0/SCALE);
            VecDot(wTmp, wTheta, &dot);
            loc1 += dot;

            if(!rank) {
                SNESGetNumberFunctionEvals(snes, &its);
                SNESGetConvergedReason(snes, &reason);
                VecNorm(velz[ei], NORM_2, &norm);
                cout << "SNES converged as " << SNESConvergedReasons[reason] << "\titeration: " << its << "\t|w|: " << norm << endl;
            }
        }
    }

    MPI_Allreduce(&loc1, &k2i_z, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&loc2, &i2k_z, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&x);
    VecDestroy(&b);
    VecDestroy(&f);
    MatDestroy(&J);
    MatDestroy(&P);
    MatDestroy(&DTV1);
    MatDestroy(&VISC);
    SNESDestroy(&snes);
}

void Euler::DiagExner(Vec* rtz, L2Vecs* exner) {
    int ex, ey, ei;
    int n2 = topo->elOrd*topo->elOrd;
    Vec eos_rhs;

    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &eos_rhs);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            vo->Assemble_EOS_RHS(ex, ey, rtz[ei], eos_rhs);
            vo->AssembleConstInv(ex, ey, vo->VB_inv);
            MatMult(vo->VB_inv, eos_rhs, exner->vz[ei]);
        }
    }
    exner->VertToHoriz();
    exner->UpdateGlobal();

    VecDestroy(&eos_rhs);
}

void Euler::VertSolve_Explicit(Vec* velz,   Vec* rho,   Vec* rt,   Vec* exner, 
                               Vec* velz_n, Vec* rho_n, Vec* rt_n, Vec* exner_n,
                               Vec* velz_p, Vec* rho_p, Vec* rt_p, Vec* exner_p) {
    int ex, ey, ei;
    int n2    = topo->elOrd*topo->elOrd;
    Vec rhs, tmp;

    VecCreateSeq(MPI_COMM_SELF, n2*(geom->nk-1), &rhs);
    VecCreateSeq(MPI_COMM_SELF, n2*(geom->nk-1), &tmp);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            vo->AssembleLinear(ex, ey, vo->VA);
            vo->AssembleLinearInv(ex, ey, vo->VA_inv);
            vo->AssembleConst(ex, ey, vo->VB);
            vo->AssembleConstInv(ex, ey, vo->VB_inv);
            vo->AssembleRayleigh(ex, ey, vo->VR);

            // solve for the vertical velocity
            MatMult(vo->VA, velz_n[ei], rhs);

            VecAXPY(rhs, -0.5*dt, gv[ei]);

            MatMult(vo->V01, Kv[ei], tmp);
            VecAXPY(rhs, -0.5*dt, tmp);

            vo->AssembleConLinWithW(ex, ey, velz_p[ei], vo->VBA);
            MatMult(vo->VBA, velz_p[ei], pTmp);
            MatMult(vo->V01, pTmp, tmp);
            VecAXPY(rhs, -0.25*dt, tmp);

            MatMult(vo->VB, exner_p[ei], pTmp);
            MatMult(vo->V01, pTmp, wTmp);
#ifdef VERT_MASS_INV
            MatMult(vo->VA_inv, wTmp, pGrad);
#else
            KSPSolve(kspColA, wTmp, pGrad);
#endif
            diagThetaVert(ex, ey, vo->VAB, rho_p[ei], rt_p[ei], theta);
            vo->AssembleLinearWithTheta(ex, ey, theta, VA_theta);
            MatMult(VA_theta, pGrad, tmp);
            VecAXPY(rhs, -0.5*dt, tmp);

            if(!ei) {
                MatMatMult(vo->V01, vo->VB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DTVB);
                MatMatMult(DTVB, vo->V10, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &VISC);
            } else {
                MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DTVB);
                MatMatMult(DTVB, vo->V10, MAT_REUSE_MATRIX, PETSC_DEFAULT, &VISC);
            }
            MatAXPY(vo->VA, -0.5*dt*vert_visc, VISC, DIFFERENT_NONZERO_PATTERN);

            MatMult(vo->VR, velz_p[ei], tmp);
            VecAXPY(rhs, -0.5*dt*RAYLEIGH, tmp);

#ifdef VERT_MASS_INV
            MatMult(vo->VA_inv, rhs, velz[ei]);
#else
            KSPSolve(kspColA, rhs, velz[ei]);
#endif

            // solve for the density
            vo->AssembleLinearWithRT(ex, ey, rho_p[ei], VA_rho, true);
            MatMult(VA_rho, velz_p[ei], wTmp);
#ifdef VERT_MASS_INV
            MatMult(vo->VA_inv, wTmp, wRho);
#else
            KSPSolve(kspColA, wTmp, wRho);
#endif
            MatMult(vo->V10, wRho, rho[ei]);
            VecAYPX(rho[ei], -0.5*dt, rho_n[ei]);

            // solve for the density weighted potential temperature
            MatMult(VA_theta, wRho, wTmp);
            //vo->AssembleLinearWithRT(ex, ey, rt_p[ei], VA_rho, true);
            //MatMult(VA_rho, velz_p[ei], wTmp);
#ifdef VERT_MASS_INV
            MatMult(vo->VA_inv, wTmp, wTheta);
#else
            KSPSolve(kspColA, wTmp, wTheta);
#endif
            MatMult(vo->V10, wTheta, rt[ei]);
            VecAYPX(rt[ei], -0.5*dt, rt_n[ei]);

            // diagnose the exner pressure
            //vo->Assemble_EOS_RHS(ex, ey, rt[ei], eosRhs);
            vo->Assemble_EOS_RHS(ex, ey, rt_n[ei], eosRhs);
            MatMult(vo->VB_inv, eosRhs, exner[ei]);
        }
    }
    MatDestroy(&DTVB);
    MatDestroy(&VISC);
    VecDestroy(&rhs);
    VecDestroy(&tmp);
}

// rt and Ft and local vectors
void Euler::SolveExner(Vec* rt, Vec* Ft, Vec* exner_i, Vec* exner_f, double _dt) {
    int ii;
    Vec rt_sum, rhs;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &rt_sum);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);

    for(ii = 0; ii < geom->nk; ii++) {
        VecCopy(Ft[ii], rt_sum);
        VecScale(rt_sum, -_dt*RD/CV);
        VecAXPY(rt_sum, 1.0, rt[ii]);
        T->assemble(rt_sum, ii, SCALE);
        MatMult(T->M, exner_i[ii], rhs);

        T->assemble(rt[ii], ii, SCALE);
        KSPSolve(kspE, rhs, exner_f[ii]);
    }
    VecDestroy(&rt_sum);
    VecDestroy(&rhs);
}
