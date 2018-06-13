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
#include "ElMats.h"
#include "Assembly.h"
#include "PrimEqns.h"

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
//#define RAD_SPHERE 1.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5

using namespace std;

PrimEqns::PrimEqns(Topo* _topo, Geom* _geom, double _dt) {
    int ii, n2;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    grav = GRAVITY*(RAD_SPHERE/RAD_EARTH);
    omega = OMEGA;
    do_visc = true;
    del2 = viscosity();
    vert_visc = viscosity_vert();
    step = 0;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    E01M1 = NULL;
    E12M2 = NULL;

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
    //T = new UtQWmat(topo, geom, node, edge);
    T = new Whmat(topo, geom, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

    // assemble the vertical gradient and divergence incidence matrices
    vertOps();

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp1);
    KSPSetOperators(ksp1, M1->M, M1->M);
    KSPSetTolerances(ksp1, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1, KSPGMRES);
    KSPGetPC(ksp1, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp1,"ksp1_");
    KSPSetFromOptions(ksp1);

    // initialize the 2 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPGetPC(ksp2, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, topo->elOrd*topo->elOrd, NULL);
    KSPSetOptionsPrefix(ksp2,"ksp2_");
    KSPSetFromOptions(ksp2);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_t);

    Kv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->elOrd*topo->elOrd, &Kv[ii]);
    }

    // initialise the single column mass matrices and solvers
    n2 = topo->elOrd*topo->elOrd;

    MatCreate(MPI_COMM_SELF, &VA);
    MatSetType(VA, MATSEQAIJ);
    MatSetSizes(VA, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VA, topo->elOrd*topo->elOrd, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VB);
    MatSetType(VB, MATSEQAIJ);
    MatSetSizes(VB, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(VB, topo->elOrd*topo->elOrd, PETSC_NULL);

    KSPCreate(MPI_COMM_SELF, &kspColA);
    KSPSetOperators(kspColA, VA, VA);
    KSPSetTolerances(kspColA, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspColA, KSPGMRES);
    KSPGetPC(kspColA, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, n2, NULL);
    KSPSetOptionsPrefix(kspColA, "kspColA_");
    KSPSetFromOptions(kspColA);

    KSPCreate(MPI_COMM_SELF, &kspColB);
    KSPSetOperators(kspColB, VB, VB);
    KSPSetTolerances(kspColB, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspColB, KSPGMRES);
    KSPGetPC(kspColB, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, n2, NULL);
    KSPSetOptionsPrefix(kspColB, "kspColB_");
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double PrimEqns::viscosity() {
    double ae = 4.0*M_PI*RAD_SPHERE*RAD_SPHERE;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

double PrimEqns::viscosity_vert() {
    int ii, kk;
    double dzMinG, dzMin = 1.0e+6;

    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < topo->n0; ii++) {
            if(geom->thick[kk][ii] < dzMin) {
                dzMin = geom->thick[kk][ii];
            }
        }
    }
    MPI_Allreduce(&dzMin, &dzMinG, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    return dzMinG*dzMinG/6.0;//TODO

}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void PrimEqns::coriolis() {
    int ii;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fl, fxl, fxg, PtQfxg;

    // initialise the coriolis vector (local and global)
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg);

    // evaluate the coriolis term at nodes
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < topo->n0; ii++) {
        fArray[ii] = 2.0*omega*sin(geom->s[ii][1]);
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
    VecPointwiseDivide(fg, PtQfxg, m0->vg);
    
    delete PtQ;
    VecDestroy(&fl);
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

PrimEqns::~PrimEqns() {
    int ii;

    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);
    VecDestroy(&theta_b);
    VecDestroy(&theta_t);

    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&Kv[ii]);
    }
    delete[] Kv;

    MatDestroy(&V01);
    MatDestroy(&V10);
    MatDestroy(&VA);
    MatDestroy(&VB);
    KSPDestroy(&kspColA);
    KSPDestroy(&kspColB);

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
}

void PrimEqns::UpdateKEVert(Vec ke, int lev) {
    int ex, ey, n2, ii, jj;
    int *inds2, *inds0;
    Vec kl;
    PetscScalar *khArray, *kvArray;
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double fg[99], zi[99];

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &kl);
    VecScatterBegin(topo->gtol_2, ke, kl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, ke, kl, INSERT_VALUES, SCATTER_FORWARD);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecGetArray(kl, &khArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            inds2 = topo->elInds2_l(ex, ey);

            // add in the vertical gravity term
            Q->assemble(ex, ey);
            Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q->A, WtQ);

            for(ii = 0; ii < Q->nDofsI; ii++) {
                zi[ii] = 0.5*(geom->levs[lev+0][inds0[ii]] + geom->levs[lev+1][inds0[ii]])*GRAVITY;
            }
            for(ii = 0; ii < W->nDofsJ; ii++) {
                fg[ii] = 0.0;
                for(jj = 0; jj < Q->nDofsI; jj++) {
                    fg[ii] += WtQ[ii][jj]*zi[jj];
                }
            }

            VecGetArray(Kv[ey*topo->nElsX + ex], &kvArray);
            for(ii = 0; ii < n2; ii++) {
                kvArray[lev*n2+ii] = khArray[inds2[ii]] + fg[ii];
            }
            VecRestoreArray(Kv[ey*topo->nElsX + ex], &kvArray);
        }
    }
    VecRestoreArray(ke, &khArray);

    VecDestroy(&kl);
    delete Q;
    delete W;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
}

/*
compute the right hand side for the momentum equation for a given level
note that the vertical velocity, uv, is stored as a different vector for 
each element
*/
void PrimEqns::horizMomRHS(Vec uh, Vec* uv, Vec* theta, Vec exner, int lev, Vec *Fu) {
    Vec wl, ul, wi, Ru, Ku, Mh, d2u, d4u, theta_k, theta_k_l, dExner, dp;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k_l);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, Fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_k);

    curl(uh, &wi, lev, true);

    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, uh, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, uh, ul, INSERT_VALUES, SCATTER_FORWARD);

    R->assemble(wl, lev);
    K->assemble(ul, uv, lev);

    MatMult(R->M, uh, Ru);
    MatMult(K->M, uh, Ku);
    MatMult(EtoF->E12, Ku, *Fu);
    VecAXPY(*Fu, 1.0, Ru);

    // must do horiztonal momentum rhs before vertical, so that that kinetic energy 
    // can be added to the vertical vectors
    UpdateKEVert(Ku, lev);

    // add the thermodynamic term (theta is in the same space as the vertical velocity)
    // project theta onto 1 forms
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta[lev]);
    VecAXPY(theta_k, 0.5, theta[lev+1]);
    VecScatterBegin(topo->gtol_2, theta_k, theta_k_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, theta_k, theta_k_l, INSERT_VALUES, SCATTER_FORWARD);
/*
    T->assemble(theta_k_l, lev);
    if(!E12M2) {
        MatMatMult(EtoF->E12, T->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E12M2);
    } else {
        MatMatMult(EtoF->E12, T->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &E12M2);
    }
    MatMult(E12M2, exner, dp);
    VecAXPY(*Fu, 1.0, dp);
*/
    grad(exner, &dExner, lev);
    F->assemble(theta_k_l, NULL, lev, false);
    MatMult(F->M, dExner, dp);
    VecAXPY(*Fu, 1.0, dp);

    // add in the biharmonic voscosity
    if(do_visc) {
        laplacian(uh, &d2u, lev);
        laplacian(d2u, &d4u, lev);
        VecAXPY(*Fu, 1.0, d4u);
    }

    VecDestroy(&wl);
    VecDestroy(&ul);
    VecDestroy(&wi);
    VecDestroy(&Ru);
    VecDestroy(&Ku);
    VecDestroy(&Mh);
    VecDestroy(&dExner);
    VecDestroy(&dp);
    VecDestroy(&theta_k);
    VecDestroy(&theta_k_l);
    if(do_visc) {
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
}

void PrimEqns::vertMomRHS(Vec* ui, Vec* wi, Vec* theta, Vec* exner, Vec* fw) {
    int ex, ey, ei, n2;
    Vec exner_v, de1, de2, de3, dp;

    n2 = topo->elOrd*topo->elOrd;

    // vertical velocity is computer per element, so matrices are local to this processor
    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &de1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &de2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &de3);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &dp);
    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &exner_v);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            // add in the kinetic energy gradient
            MatMult(V01, Kv[ei], fw[ei]);

            // add in the pressure gradient
            VecZeroEntries(exner_v);
            HorizToVert2(ex, ey, exner, exner_v);

            VecZeroEntries(de1);
            VecZeroEntries(de2);
            VecZeroEntries(de3);
            AssembleConst(ex, ey, VB);
            MatMult(VB, exner_v, de1);
            MatMult(V01, de1, de2);
            AssembleLinear(ex, ey, VA);//TODO: skip this and just solve with B(theta)?? on LHS
            KSPSolve(kspColA, de2, de3);

            // interpolate the potential temperature onto the piecewise linear
            // vertical mass matrix and multiply by the weak form vertical gradient of
            // the exner pressure
            AssembleLinearWithTheta(ex, ey, theta, VA);
            MatMult(VA, de3, dp);
            VecAXPY(fw[ei], 1.0, dp);

            // TODO: add in horizontal vorticity terms
        }
    }

    VecDestroy(&exner_v);
    VecDestroy(&de1);
    VecDestroy(&de2);
    VecDestroy(&de3);
    VecDestroy(&dp);
}

/*
compute the continuity equation right hand side for all levels
uh: horiztonal velocity by vertical level
uv: vertical velocity by horiztonal element
*/
void PrimEqns::massRHS(Vec* uh, Vec* uv, Vec* pi, Vec* Fp) {
    int kk, ex, ey, n2;
    Vec Mpu, Fv, Dv;
    Vec pl, pu, Fi, Dh;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Mpu);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Fv);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Dv);

    // compute the vertical mass fluxes (piecewise linear in the vertical)
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            VertFlux(ex, ey, pi, NULL, VA);
            MatMult(VA, uv[ey*topo->nElsX+ex], Mpu);
            AssembleLinear(ex, ey, VA);
            KSPSolve(kspColA, Mpu, Fv);
            // strong form vertical divergence
            MatMult(V10, Fv, Dv);

            // copy the vertical contribution to the divergence into the
            // horiztonal vectors
            VertToHoriz2(ex, ey, 0, geom->nk, Dv, Fp);
        }
    }
    VecDestroy(&Mpu);
    VecDestroy(&Fv);
    VecDestroy(&Dv);

    // compute the horiztonal mass fluxes
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &pl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &pu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Dh);

    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, pi[kk], pl, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(topo->gtol_2, pi[kk], pl, INSERT_VALUES, SCATTER_FORWARD);

        // add the horiztonal fluxes
        F->assemble(pl, NULL, kk, true);
        M1->assemble(kk);
        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fi);
        MatMult(EtoF->E21, Fi, Dh);
        VecAXPY(Fp[kk], 1.0, Dh);
    }

    VecDestroy(&pl);
    VecDestroy(&pu);
    VecDestroy(&Fi);
    VecDestroy(&Dh);
}

/*
Assemble the boundary condition vector for rho(t) X theta(0)
*/
void PrimEqns::thetaBCVec(int ex, int ey, Mat A, Vec* rho, Vec* bTheta, double scale) {
    int* inds2 = topo->elInds2_l(ex, ey);
    int ii, ei, mp1, mp12, n2;
    double det, rk;
    int inds2k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar *rArray, *vArray, *hArray;
    Vec theta_o;

    ei    = ey*topo->nElsX + ex;
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;
    n2    = topo->elOrd*topo->elOrd;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &theta_o);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, bTheta);

    MatZeroEntries(A);

    // bottom boundary
    Q->assemble(ex, ey);
        
    VecGetArray(rho[0], &rArray);
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii][ii] = scale*Q->A[ii][ii]/det/det;

        // multuply by the vertical determinant to integrate, then
        // divide piecewise constant density by the vertical determinant,
        // so these cancel
        geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rArray, &rk);
        Q0[ii][ii] *= rk;
    }
    VecRestoreArray(rho[0], &rArray);

    Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
    Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        inds2k[ii] = ii + 0*W->nDofsJ;
    }
    MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

    // top boundary
    Q->assemble(ex, ey);
        
    VecGetArray(rho[geom->nk-1], &rArray);
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii][ii] = scale*Q->A[ii][ii]/det/det;

        // multuply by the vertical determinant to integrate, then
        // divide piecewise constant density by the vertical determinant,
        // so these cancel
        geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rArray, &rk);
        Q0[ii][ii] *= rk;
    }
    VecRestoreArray(rho[geom->nk-1], &rArray);

    Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
    Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
    Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

    for(ii = 0; ii < W->nDofsJ; ii++) {
        inds2k[ii] = ii + (geom->nk-2)*W->nDofsJ;
    }
    MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // assemble the theta bc vector
    VecGetArray(theta_o, &vArray);
    // bottom
    VecGetArray(theta_b, &hArray);
    for(ii = 0; ii < n2; ii++) {
        vArray[ii] = hArray[inds2[ii]];
    }
    VecRestoreArray(theta_b, &hArray);
    // top
    VecGetArray(theta_t, &hArray);
    for(ii = 0; ii < n2; ii++) {
        vArray[(geom->nk-2)*n2+ii] = hArray[inds2[ii]];
    }
    VecRestoreArray(theta_t, &hArray);
    VecRestoreArray(theta_o, &vArray);

    MatMult(A, theta_o, *bTheta);

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
    VecDestroy(&theta_o);
}

/*
diagnose theta from rho X theta (with boundary condition)
*/
void PrimEqns::diagTheta(Vec* rho, Vec* rt, Vec* theta) {
    int ex, ey, n2, kk;
    double scale = 1.0e8;
    Vec rtv, frt, theta_v, bcs;
    Mat A, AB;

    n2 = topo->elOrd*topo->elOrd;

    // reset the potential temperature at the internal layer interfaces, not the boundaries
    for(kk = 1; kk < geom->nk; kk++) {
        VecZeroEntries(theta[kk]);
    }

    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &rtv);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &frt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &theta_v);

    MatCreate(MPI_COMM_SELF, &A);
    MatSetType(A, MATSEQAIJ);
    MatSetSizes(A, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(A, n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &AB);
    MatSetType(AB, MATSEQAIJ);
    MatSetSizes(AB, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(AB, 2*n2, PETSC_NULL);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            // construct horiztonal rho theta field
            HorizToVert2(ex, ey, rt, rtv);
            AssembleLinCon(ex, ey, AB, scale);
            MatMult(AB, rtv, frt);

            // assemble in the bcs
            thetaBCVec(ex, ey, A, rho, &bcs, scale);
            VecAXPY(frt, -1.0, bcs);

            AssembleLinearWithRho(ex, ey, rho, VA, scale);
            KSPSolve(kspColA, frt, theta_v);
            VertToHoriz2(ex, ey, 1, geom->nk, theta_v, theta);
            VecDestroy(&bcs);
        }
    }

    VecDestroy(&rtv);
    VecDestroy(&frt);
    VecDestroy(&theta_v);
    MatDestroy(&A);
    MatDestroy(&AB);
}

/*
prognose the exner pressure
*/
void PrimEqns::progExner(Vec rt_i, Vec rt_f, Vec exner_i, Vec* exner_f, int lev) {
    Vec rt_l, rhs;
    PC pc;
    KSP kspE;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &rt_l);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, exner_f);

    KSPCreate(MPI_COMM_WORLD, &kspE);
    KSPSetOperators(kspE, T->M, T->M);
    KSPSetTolerances(kspE, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspE, KSPGMRES);
    KSPGetPC(kspE, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, topo->elOrd*topo->elOrd, NULL);
    KSPSetOptionsPrefix(kspE, "exner_");
    KSPSetFromOptions(kspE);

    // assemble the right hand side
    VecScatterBegin(topo->gtol_2, rt_i, rt_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, rt_i, rt_l, INSERT_VALUES, SCATTER_FORWARD);

    T->assemble(rt_l, lev);
    MatMult(T->M, exner_i, rhs);

    // assemble the nonlinear operator
    VecScatterBegin(topo->gtol_2, rt_f, rt_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, rt_f, rt_l, INSERT_VALUES, SCATTER_FORWARD);

    F->assemble(rt_l, NULL, lev, true);
    KSPSolve(kspE, rhs, *exner_f);

    VecDestroy(&rt_l);
    VecDestroy(&rhs);
    KSPDestroy(&kspE);
}

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void PrimEqns::grad(Vec phi, Vec* u, int lev) {
    double scale = 1.0e8;
    Vec dPhi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dPhi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, u);

    M2->assemble(lev, scale);
    if(!E12M2) {
        MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E12M2);
    } else {
        MatMatMult(EtoF->E12, M2->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &E12M2);
    }

    VecZeroEntries(dPhi);
    MatMult(E12M2, phi, dPhi);
    VecScale(dPhi, scale);
    KSPSolve(ksp1, dPhi, *u);

    VecDestroy(&dPhi);
}

/*
Take the weak form curl of a 1 form vector field as a 1 form vector field
*/
void PrimEqns::curl(Vec u, Vec* w, int lev, bool add_f) {
	Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);

    M1->assemble(lev);
    if(!E01M1) {
        MatMatMult(NtoE->E01, M1->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E01M1);
    } else {
        MatMatMult(NtoE->E01, M1->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &E01M1);
    }

    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    VecPointwiseDivide(*w, du, m0->vg);

    // add the coliolis term
    if(add_f) {
        VecAYPX(*w, 1.0, fg);
    }
    VecDestroy(&du);
}

void PrimEqns::laplacian(Vec ui, Vec* ddu, int lev) {
    Vec Du, Cu, RCu, GDu, MDu, dMDu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, ddu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &RCu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &GDu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMDu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &MDu);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, ui, Du);

    // grad (weak form)
    grad(Du, &GDu, lev);

    /*** rotational component ***/
    // curl (weak form)
    curl(ui, &Cu, lev, false);

    // rot (strong form)
    MatMult(NtoE->E10, Cu, RCu);

    // add rotational and divergent components
    VecCopy(GDu, *ddu);
    VecAXPY(*ddu, +1.0, RCu);

    VecScale(*ddu, del2);

    VecDestroy(&Cu);
    VecDestroy(&RCu);
    VecDestroy(&GDu);
    VecDestroy(&dMDu);
    VecDestroy(&Du);
    VecDestroy(&MDu);
}

/*
assemble the vertical gradient and divergence orientation matrices
V10 is the strong form vertical divergence from the linear to the
constant basis functions
*/
void PrimEqns::vertOps() {
    int ii, kk, n2, rows[1], cols[2];
    double vals[2] = {+1.0, -1.0};
    Mat V10t;
    
    n2 = topo->elOrd*topo->elOrd;

    MatCreate(MPI_COMM_SELF, &V10);
    MatSetType(V10, MATSEQAIJ);
    MatSetSizes(V10, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(V10, 2, PETSC_NULL);

    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < n2; ii++) {
            rows[0] = kk*n2 + ii;

            if(kk > 0 && kk < geom->nk - 1) {
                cols[0] = (kk-1)*n2 + ii;
                cols[1] = (kk+0)*n2 + ii;
                MatSetValues(V10, 1, rows, 2, cols, vals, INSERT_VALUES);
            }
            else if(kk == 0) { // bottom level
                cols[0] = ii;
                MatSetValues(V10, 1, rows, 1, cols, &vals[1], INSERT_VALUES);
            }
            else {             // top level
                cols[0] = (kk-1)*n2 + ii;
                MatSetValues(V10, 1, rows, 1, cols, &vals[0], INSERT_VALUES);
            }
        }
    }
    MatAssemblyBegin(V10, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V10, MAT_FINAL_ASSEMBLY);

    MatTranspose(V10, MAT_INITIAL_MATRIX, &V10t);
    MatDuplicate(V10t, MAT_DO_NOT_COPY_VALUES, &V01);
    MatZeroEntries(V01);
    MatAXPY(V01, -1.0, V10t, SAME_NONZERO_PATTERN);
    MatDestroy(&V10t);
}

/*
assemble a 3D mass matrix as a tensor product of 2 forms in the 
horizotnal and constant basis functions in the vertical
*/
void PrimEqns::AssembleConst(int ex, int ey, Mat B) {
    int ii, kk, ei, mp12;
    int *inds0;
    double det;
    int inds2k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];

    inds0 = topo->elInds0_l(ex, ey);
    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(B);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);
        ei = ey*topo->nElsX + ex;
        
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]/det/det;
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii][ii] *= 2.0/geom->thick[kk][inds0[ii]];
        }

        // assemble the piecewise constant mass matrix for level k
        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(B, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

/*
Assemble a 3D mass matrix as a tensor product of 2 forms in the 
horizotnal and linear basis functions in the vertical
*/
void PrimEqns::AssembleLinear(int ex, int ey, Mat A) {
    int ii, kk, ei, mp12;
    int *inds0;
    double det;
    int inds2k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];

    ei = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(A);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);
        
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]/det/det;
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            Q0[ii][ii] *= geom->thick[kk][inds0[ii]]/2.0;
        }

        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

void PrimEqns::AssembleLinCon(int ex, int ey, Mat AB, double scale) {
    int ii, kk, ei, mp12;
    double det;
    int rows[99], cols[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];

    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(AB);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);
        ei = ey*topo->nElsX + ex;
        
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = scale*Q->A[ii][ii]/det/det;

            // multiply by the vertical jacobian, then scale the piecewise constant 
            // basis by the vertical jacobian, so do nothing 
        }

        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            cols[ii] = ii + kk*W->nDofsJ;
        }

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                rows[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(AB, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

void PrimEqns::AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A, double scale) {
    int ii, kk, ei, mp1, mp12;
    double det, rk;
    int inds2k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar *rArray;

    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatZeroEntries(A);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);
        ei = ey*topo->nElsX + ex;
        
        VecGetArray(rho[kk], &rArray);
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = scale*Q->A[ii][ii]/det/det;

            // multuply by the vertical determinant to integrate, then
            // divide piecewise constant density by the vertical determinant,
            // so these cancel
            geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rArray, &rk);
            Q0[ii][ii] *= rk;
        }
        VecRestoreArray(rho[kk], &rArray);

        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

void PrimEqns::AssembleLinearWithTheta(int ex, int ey, Vec* theta, Mat A) {
    int ii, kk, ei, mp1, mp12;
    int *inds0;
    double det, t1, t2;
    int inds2k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** QB = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** QT = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar *t1Array, *t2Array;

    inds0 = topo->elInds0_l(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatZeroEntries(A);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);
        ei = ey*topo->nElsX + ex;
        
        VecGetArray(theta[kk+0], &t1Array);
        VecGetArray(theta[kk+1], &t2Array);
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            QB[ii][ii] = Q->A[ii][ii]/det/det;
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            QB[ii][ii] *= geom->thick[kk][inds0[ii]]/2.0;
            QT[ii][ii] = QB[ii][ii];

            geom->interp2_g(ex, ey, ii%mp1, ii/mp1, t1Array, &t1);
            geom->interp2_g(ex, ey, ii%mp1, ii/mp1, t2Array, &t2);

            QB[ii][ii] *= t1;
            QT[ii][ii] *= t2;
        }
        VecRestoreArray(theta[kk+0], &t1Array);
        VecRestoreArray(theta[kk+1], &t2Array);

        // assemble the first basis function
        if(kk > 0) {
            Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QB, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, QT, WtQ);
            Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
            Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, QB);
    Free2D(Q->nDofsI, QT);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

/*
derive the vertical mass flux
TODO: only need a single piecewise constant field, may be either rho or rho X theta
*/
void PrimEqns::VertFlux(int ex, int ey, Vec* pi, Vec* ti, Mat Mp) {
    int ii, kk, ei, mp1, mp12;
    double det, rho, temp1, temp2;
    int inds2k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar *pArray, *t1Array, *t2Array;

    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(Mp);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);
        ei = ey*topo->nElsX + ex;
        
        VecGetArray(pi[kk], &pArray);
        if(ti) {
            VecGetArray(ti[kk], &t1Array);
            VecGetArray(ti[kk+1], &t2Array);
        }

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]/det/det;

            geom->interp2_g(ex, ey, ii%mp1, ii/mp1, pArray, &rho);
            Q0[ii][ii] *= rho;

            // multiply by the vertical determinant for the vertical integral,
            // then divide by the vertical determinant to rescale the piecewise
            // constant density, so do nothing.

            if(ti) {
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, t1Array, &temp1);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, t2Array, &temp2);
                Q0[ii][ii] *= 0.5*(temp1 + temp2);
            }
        }
        VecRestoreArray(pi[kk], &pArray);
        if(ti) {
            VecRestoreArray(ti[kk], &t1Array);
            VecRestoreArray(ti[kk+1], &t2Array);
        }

        // assemble the piecewise constant mass matrix for level k
        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function (exclude bottom boundary)
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(Mp, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function (exclude top boundary)
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                inds2k[ii] = ii + (kk+0)*W->nDofsJ;
            }
            MatSetValues(Mp, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(Mp, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mp, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

void PrimEqns::AssembleVertOps(int ex, int ey, Mat A) {
    int n2 = topo->elOrd*topo->elOrd;
    Mat B, L, BD;

    MatCreate(MPI_COMM_SELF, &B);
    MatSetType(B, MATSEQAIJ);
    MatSetSizes(B, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(B, n2, PETSC_NULL);

    AssembleLinear(ex, ey, A);
    AssembleConst(ex, ey, B);

    // construct the laplacian mixing operator
    MatMatMult(B, V10, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &BD);
    MatMatMult(V01, BD, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L);

    // assemble the piecewise linear mass matrix (with gravity)
    MatAXPY(A, -vert_visc, L, DIFFERENT_NONZERO_PATTERN);//TODO: check the sign on the viscosity

    MatDestroy(&B);
    MatDestroy(&BD);
    MatDestroy(&L);
}

// note: rho X theta must include the theta boundary conditions
void PrimEqns::SolveRK2(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int ii, kk, ex, ey, n2;
    char fieldname[100];
    Vec *Hu1, *Vu1, *Fp1, *Ft1, *velx_h, *velz_h, *rho_h, *rt_h, *exner_h, bu, bw, wi;
    Vec *Hu2, *Vu2, *Fp2, *Ft2, *rho_i, *rt_i, *exner_i, exner_f, *theta;

    n2 = topo->elOrd*topo->elOrd;

    Hu1 = new Vec[geom->nk];
    Hu2 = new Vec[geom->nk];

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &bw);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    velx_h  = new Vec[geom->nk];
    rho_h   = new Vec[geom->nk];
    rt_h    = new Vec[geom->nk];
    exner_h = new Vec[geom->nk];
    rho_i   = new Vec[geom->nk];
    rt_i    = new Vec[geom->nk];
    exner_i = new Vec[geom->nk];
    velz_h  = new Vec[topo->nElsX*topo->nElsX];
    theta   = new Vec[geom->nk+1];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_h[kk] );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho_h[kk]  );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt_h[kk]   );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho_i[kk]  );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt_i[kk]   );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner_i[kk]);
        // temporary vectors for use in exner pressure prognosis
        VecCopy(rho[kk]  , rho_i[kk]  );
        VecCopy(rt[kk]   , rt_i[kk]   );
        VecCopy(exner[kk], exner_i[kk]);
    }
    // create vectors for the potential temperature at the internal layer interfaces
    for(kk = 1; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta[kk]);
    }
    // set the top and bottom potential temperature bcs
    theta[0]        = theta_b;
    theta[geom->nk] = theta_t;

    Vu1 = new Vec[topo->nElsX*topo->nElsX];
    Vu2 = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &velz_h[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Vu1[ii]   );
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Vu2[ii]   );
        VecZeroEntries(Vu1[ii]);
        VecZeroEntries(Vu2[ii]);
    }

    // continuity and temperature equation rhs vectors
    Fp1 = new Vec[geom->nk];
    Ft1 = new Vec[geom->nk];
    Fp2 = new Vec[geom->nk];
    Ft2 = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Fp1[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ft1[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Fp2[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ft2[kk]);
        VecZeroEntries(Fp1[kk]);
        VecZeroEntries(Ft1[kk]);
        VecZeroEntries(Fp2[kk]);
        VecZeroEntries(Ft2[kk]);
    }

    // construct the right hand side terms for the first substep
    // note: do horiztonal rhs first as this assembles the kinetic energy
    // operator for use in the vertical rhs
    diagTheta(rho, rt, theta);
    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx[kk], velz, theta, exner[kk], kk, &Hu1[kk]);
    }
    vertMomRHS(velx, velz, theta, exner, Vu1);
    massRHS(velx, velz, rho, Fp1);
    massRHS(velx, velz, rt,  Ft1);

    // solve for the half step values
    for(kk = 0; kk < geom->nk; kk++) {
        // horizontal momentum
        VecZeroEntries(bu);
        VecCopy(velx[kk], bu);
        VecAXPY(bu, -dt, Hu1[kk]);
        M1->assemble(kk);
        KSPSolve(ksp1, bu, velx_h[kk]);

        // density
        VecZeroEntries(rho_h[kk]);
        VecCopy(rho_i[kk], rho_h[kk]);
        VecAXPY(rho_h[kk], -dt, Fp1[kk]);

        // potential temperature
        VecZeroEntries(rt_h[kk]);
        VecCopy(rt_i[kk], rt_h[kk]);
        VecAXPY(rt_h[kk], -dt, Ft1[kk]);

        // exner pressure
        progExner(rt_i[kk], rt_h[kk], exner[kk], &exner_h[kk], kk);
    }

    // solve for the vertical velocity
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;

            AssembleVertOps(ex, ey, VA);
            VecZeroEntries(bw);
            VecCopy(velz[ii], bw);
            VecAXPY(bw, -dt, Vu1[ii]);
            KSPSolve(kspColA, bw, velz_h[ii]);
        }
    }

    // construct right hand side terms for the second substep
    diagTheta(rho_h, rt_h, theta);
    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx_h[kk], velz_h, theta, exner_h[kk], kk, &Hu2[kk]);
    }
    vertMomRHS(velx_h, velz_h, theta, exner_h, Vu2);
    massRHS(velx_h, velz_h, rho_h, Fp2);
    massRHS(velx_h, velz_h, rt_h,  Ft2);

    // solve for the full step values
    for(kk = 0; kk < geom->nk; kk++) {
        // horizontal momentum
        VecZeroEntries(bu);
        VecCopy(velx[kk], bu);
        VecAXPY(bu, -0.5*dt, Hu1[kk]);
        VecAXPY(bu, -0.5*dt, Hu2[kk]);
        M1->assemble(kk);
        VecZeroEntries(velx[kk]);
        KSPSolve(ksp1, bu, velx[kk]);

        // density
        VecAXPY(rho[kk], -0.5*dt, Fp1[kk]);
        VecAXPY(rho[kk], -0.5*dt, Fp2[kk]);

        // potential temperature
        VecAXPY(rt[kk], -0.5*dt, Ft1[kk]);
        VecAXPY(rt[kk], -0.5*dt, Ft2[kk]);

        // exner pressure (second order)
        VecScale(rt_i[kk]   , 0.5);
        VecScale(exner_i[kk], 0.5);
        VecAXPY(rt_i[kk]   , 0.5, rt_h[kk]   );
        VecAXPY(exner_i[kk], 0.5, exner_h[kk]);
        progExner(rt_i[kk], rt[kk], exner_i[kk], &exner_f, kk);
        VecCopy(exner_f, exner[kk]);
        VecDestroy(&exner_f);
    }

    // solve for the vertical velocity
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;

            AssembleVertOps(ex, ey, VA);
            VecZeroEntries(bw);
            VecCopy(velz[ii], bw);
            VecAXPY(bw, -0.5*dt, Vu1[ii]);
            VecAXPY(bw, -0.5*dt, Vu2[ii]);
            VecZeroEntries(velz[ii]);
            KSPSolve(kspColA, bw, velz[ii]);
        }
    }

    // write output
    if(save) {
        step++;
        for(kk = 0; kk < geom->nk; kk++) {
            curl(velx[kk], &wi, kk, false);

            sprintf(fieldname, "vorticity");
            geom->write0(wi, fieldname, step, kk);
            sprintf(fieldname, "velocity_h");
            geom->write1(velx[kk], fieldname, step, kk);
            sprintf(fieldname, "density");
            geom->write2(rho[kk], fieldname, step, kk);
            sprintf(fieldname, "rhoTheta");
            geom->write2(rt[kk], fieldname, step, kk);
            sprintf(fieldname, "exner");
            geom->write2(exner[kk], fieldname, step, kk);

            VecDestroy(&wi);
        }

        sprintf(fieldname, "velVert");
        geom->writeSerial(velz, fieldname, topo->nElsX*topo->nElsX, step);
    }

    // deallocate
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Hu1[kk]);
        VecDestroy(&Fp1[kk]);
        VecDestroy(&Ft1[kk]);
        VecDestroy(&Hu2[kk]);
        VecDestroy(&Fp2[kk]);
        VecDestroy(&Ft2[kk]);
        VecDestroy(&velx_h[kk]);
        VecDestroy(&rho_h[kk]);
        VecDestroy(&rt_h[kk]);
        VecDestroy(&exner_h[kk]);
        VecDestroy(&rho_i[kk]);
        VecDestroy(&rt_i[kk]);
        VecDestroy(&exner_i[kk]);
    }
    for(kk = 1; kk < geom->nk; kk++) {
        VecDestroy(&theta[kk]);
    }
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&Vu1[ii]);
        VecDestroy(&Vu2[ii]);
        VecDestroy(&velz_h[ii]);
    }
    delete[] Hu1;
    delete[] Vu1;
    delete[] Fp1;
    delete[] Ft1;
    delete[] Hu2;
    delete[] Vu2;
    delete[] Fp2;
    delete[] Ft2;
    delete[] velx_h;
    delete[] rho_h;
    delete[] rt_h;
    delete[] exner_h;
    delete[] rho_i;
    delete[] rt_i;
    delete[] exner_i;
    delete[] velz_h;
    delete[] theta;
    VecDestroy(&bw);
    VecDestroy(&bu);
}

void PrimEqns::VertToHoriz2(int ex, int ey, int ki, int kf, Vec pv, Vec* ph) {
    int ii, kk, n2;
    int* inds2 = topo->elInds2_l(ex, ey);
    PetscScalar *hArray, *vArray;

    n2 = topo->elOrd*topo->elOrd;

    VecGetArray(pv, &vArray);
    for(kk = ki; kk < kf; kk++) {
        VecGetArray(ph[kk], &hArray);
        for(ii = 0; ii < n2; ii++) {
            hArray[inds2[ii]] += vArray[kk*n2+ii];
        }
        VecRestoreArray(ph[kk], &hArray);
    }
    VecRestoreArray(pv, &vArray);
}

void PrimEqns::HorizToVert2(int ex, int ey, Vec* ph, Vec pv) {
    int ii, kk, n2;
    int* inds2 = topo->elInds2_l(ex, ey);
    PetscScalar *hArray, *vArray;

    n2 = topo->elOrd*topo->elOrd;

    VecZeroEntries(pv);

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

void PrimEqns::init0(Vec* q, ICfunc3D* func) {
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

        MatMult(PQ->M, bg, PQb);
        VecPointwiseDivide(q[kk], PQb, m0->vg);
    }

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void PrimEqns::init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y) {
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

        M1->assemble(kk);
        MatMult(UQ->M, bg, UQb);
        KSPSolve(ksp1, UQb, u[kk]);
    }

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&UQb);
    ISDestroy(&isl);
    ISDestroy(&isg);
    VecScatterDestroy(&scat);
    delete UQ;
    delete[] loc02;
}

void PrimEqns::init2(Vec* h, ICfunc3D* func) {
    int ex, ey, ii, kk, mp1, mp12;
    int *inds0;
    double scale = 1.0e8;
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
        VecScale(WQb, scale);     // have to rescale the M2 operator as the metric terms scale
        M2->assemble(kk,scale);   // this down to machine precision, so rescale the rhs as well
        KSPSolve(ksp2, WQb, h[kk]);
    }

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void PrimEqns::initTheta(Vec theta, ICfunc3D* func) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
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

    M2->assemble(0, 1.0);
    MatMult(WQ->M, bg, WQb);
    KSPSolve(ksp2, WQb, theta);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

#if 0
/*
compute the temperature equation right hand side for all levels
uh: horiztonal velocity by vertical level
uv: vertical velocity by horiztonal element
TODO: this routine may be deprecated if rho X theta is passed in as a single field
*/
void PrimEqns::tempRHS(Vec* uh, Vec* uv, Vec* pi, Vec* theta, Vec **Ft) {
    int kk, ex, ey, n2;
    Vec Mtu, Fv, Dv;
    Vec pl, pu, Fi, Dh;
    Vec theta_k, theta_k_l;
    Mat Mt, B;
    PC pc;
    KSP kspCol;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Mtu);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Fv);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Dv);

    // allocate the RHS vectors at each level
    Ft = new Vec*[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Ft[kk]);
        VecZeroEntries(*Ft[kk]);
    }

    // compute the vertical mass fluxes (piecewise linear in the vertical)
    MatCreate(MPI_COMM_SELF, &B);
    MatSetType(B, MATSEQAIJ);
    MatSetSizes(B, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(B, n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &Mt);
    MatSetType(Mt, MATSEQAIJ);
    MatSetSizes(Mt, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(Mt, topo->elOrd*topo->elOrd, PETSC_NULL);

    KSPCreate(MPI_COMM_SELF, &kspCol);
    KSPSetOperators(kspCol, B, B);
    KSPSetTolerances(kspCol, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspCol, KSPGMRES);
    KSPGetPC(kspCol, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, n2, NULL);
    KSPSetOptionsPrefix(kspCol,"kspCol_");
    KSPSetFromOptions(kspCol);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            VertFlux(ex, ey, pi, theta, Mt);
            MatMult(Mt, uv[ey*topo->nElsX+ex], Mtu);
            AssembleLinear(ex, ey, B);
            KSPSolve(kspCol, Mtu, Fv);
            // strong form vertical divergence
            MatMult(V10, Fv, Dv);

            // copy the vertical contribution to the divergence into the
            // horiztonal vectors
            VertToHoriz2(ex, ey, 0, geom->nk, Dv, *Ft);
        }
    }
    VecDestroy(&Mtu);
    VecDestroy(&Dv);
    MatDestroy(&B);
    MatDestroy(&Mt);
    KSPDestroy(&kspCol);

    // compute the horiztonal mass fluxes
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &pl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k_l);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &pu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Dh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_k);

    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, pi[kk], pl, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(topo->gtol_2, pi[kk], pl, INSERT_VALUES, SCATTER_FORWARD);

        VecZeroEntries(theta_k);
        VecAXPY(theta_k, 0.5, theta[kk]);
        VecAXPY(theta_k, 0.5, theta[kk+1]);
        VecScatterBegin(topo->gtol_2, theta_k, theta_k_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(topo->gtol_2, theta_k, theta_k_l, INSERT_VALUES, SCATTER_FORWARD);

        // add the horiztonal fluxes
        F->assemble(pl, theta_k_l, kk, true);
        M1->assemble(kk);
        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fi);
        MatMult(EtoF->E21, Fi, Dh);
        VecAXPY(*Ft[kk], 1.0, Dh);
    }

    VecDestroy(&pl);
    VecDestroy(&pu);
    VecDestroy(&Fi);
    VecDestroy(&Dh);
    VecDestroy(&theta_k);
    VecDestroy(&theta_k_l);
}
#endif

#if 0
void PrimEqns::VertConstMatInv(int ex, int ey, Mat Binv) {
    int ii, kk, ei, mp1, mp12;
    int* inds0 = topo->elInds0_l(ex, ey);
    int rows[99];
    double det;
    Wii* Q = new Wii(edge->l->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Qaa = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double** WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* Aflat = new double[W->nDofsJ*W->nDofsJ];
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;
    ei = ey*topo->nElsX + ex;

    MatZeroEntries(Binv);

    for(kk = 0; kk < geom->nk; kk++) {
       // incorporate the jacobian transformation for each element
       Q->assemble(ex, ey);

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Qaa[ii][ii] = Q->A[ii][ii]/det/det;
            Qaa[ii][ii] *= 2.0/geom->thick[kk][inds0[ii]];
        }

        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Qaa, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
        Inv(WtQWflat, Aflat, topo->elOrd);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(Binv, W->nDofsJ, rows, W->nDofsJ, rows, Aflat, ADD_VALUES);
    }

    Free2D(Q->nDofsI, Qaa);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    delete W;
    delete Q;
    delete[] Aflat;
    delete[] WtQWflat;
}
#endif

#if 0
void PrimEqns::progExner(Vec rho_i, Vec rho_f, Vec* theta_i, Vec* theta_f, Vec exner_i, Vec* exner_f, int lev) {
    Vec rho_l, theta_k, theta_k_l, rhs;
    PC pc;
    KSP kspE;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &rho_l);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k_l);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_k);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, exner_f);

    KSPCreate(MPI_COMM_WORLD, &kspE);
    KSPSetOperators(kspE, F->M, F->M);
    KSPSetTolerances(kspE, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspE, KSPGMRES);
    KSPGetPC(kspE, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, topo->elOrd*topo->elOrd, NULL);
    KSPSetOptionsPrefix(kspE, "exner_");
    KSPSetFromOptions(kspE);

    // assemble the right hand side
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta_i[lev]);
    VecAXPY(theta_k, 0.5, theta_i[lev+1]);

    VecScatterBegin(topo->gtol_2, rho_i, rho_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, theta_k, theta_k_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, rho_i, rho_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, theta_k, theta_k_l, INSERT_VALUES, SCATTER_FORWARD);

    F->assemble(rho_l, theta_k_l, lev, true);
    MatMult(F->M, exner_i, rhs);

    // assemble the nonlinear operator
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 0.5, theta_f[lev]);
    VecAXPY(theta_k, 0.5, theta_f[lev+1]);

    VecScatterBegin(topo->gtol_2, rho_f, rho_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, theta_k, theta_k_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, rho_f, rho_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, theta_k, theta_k_l, INSERT_VALUES, SCATTER_FORWARD);

    F->assemble(rho_l, theta_k_l, lev, true);
    KSPSolve(kspE, rhs, *exner_f);

    VecDestroy(&rho_l);
    VecDestroy(&theta_k_l);
    VecDestroy(&theta_k);
    VecDestroy(&rhs);
    KSPDestroy(&kspE);
}
#endif

#if 0
void PrimEqns::SolveRK2(Vec* velx, Vec* velz, Vec* rho, Vec* theta, Vec* exner, bool save) {
    int ii, kk, ex, ey, n2;
    char fieldname[100];
    Vec *Hu1, *Vu1, *Fp1, *Ft1, *velx_h, *velz_h, *rho_h, *theta_h, *exner_h, bu, bw, wi;
    Vec *Hu2, *Vu2, *Fp2, *Ft2, *rho_i, *theta_i, *exner_i, exner_f;
    Mat A;
    PC pc;
    KSP kspCol;

    n2 = topo->elOrd*topo->elOrd;

    Hu1 = new Vec[geom->nk];
    Hu2 = new Vec[geom->nk];

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->n2, &bw);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    velx_h  = new Vec[geom->nk];
    rho_h   = new Vec[geom->nk];
    theta_h = new Vec[geom->nk];
    exner_h = new Vec[geom->nk];
    rho_i   = new Vec[geom->nk];
    theta_i = new Vec[geom->nk];
    exner_i = new Vec[geom->nk];
    velz_h  = new Vec[topo->nElsX*topo->nElsX];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_h[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho_h[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_h[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho_i[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_i[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner_i[kk]);
        // temporary vectors for use in exner pressure prognosis
        VecCopy(rho[kk],   rho_i[kk]);
        VecCopy(theta[kk], theta_i[kk]);
        VecCopy(exner[kk], exner_i[kk]);
    }

    MatCreate(MPI_COMM_SELF, &A);
    MatSetType(A, MATSEQAIJ);
    //MatSetSizes(A, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2);
    MatSetSizes(A, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(A, n2, PETSC_NULL);

    KSPCreate(MPI_COMM_SELF, &kspCol);
    KSPSetOperators(kspCol, A, A);
    KSPSetTolerances(kspCol, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspCol, KSPGMRES);
    KSPGetPC(kspCol, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, n2, NULL);
    KSPSetOptionsPrefix(kspCol,"kspVert_");
    KSPSetFromOptions(kspCol);

    // construct the right hand side terms for the first substep
    // note: do horiztonal rhs first as this assembles the kinetic energy
    // operator for use in the vertical rhs
    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx[kk], velz, theta, exner[kk], kk, &Hu1[kk]);
    }
    vertMomRHS(velx, velz, theta, exner, &Vu1);
    massRHS(velx, velz, rho, &Fp1);
    tempRHS(velx, velz, rho, theta, &Ft1);

    // solve for the half step values
    for(kk = 0; kk < geom->nk; kk++) {
        // horizontal momentum
        VecZeroEntries(bu);
        VecCopy(velx[kk], bu);
        VecAXPY(bu, -dt, Hu1[kk]);
        M1->assemble(kk);
        KSPSolve(ksp1, bu, velx_h[kk]);

        // density
        VecZeroEntries(rho_h[kk]);
        VecCopy(rho[kk], rho_h[kk]);
        VecAXPY(rho_h[kk], -dt, Fp1[kk]);

        // potential temperature
        VecZeroEntries(theta_h[kk]);
        VecCopy(theta[kk], theta_h[kk]);
        VecAXPY(theta_h[kk], -dt, Ft1[kk]);

        // exner pressure
        progExner(rho[kk], rho_h[kk], theta, theta_h, exner[kk], &exner_h[kk], kk);
    }

    // solve for the vertical velocity
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;
            VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->n2, &velz_h[ii]);

            AssembleVertOps(ex, ey, A);
            VecZeroEntries(bw);
            VecCopy(velz[ii], bw);
            VecAXPY(bw, -dt, Vu1[ii]);
            KSPSolve(kspCol, bw, velz_h[ii]);
        }
    }

    // construct right hand side terms for the second substep
    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx_h[kk], velz_h, theta_h, exner_h[kk], kk, &Hu2[kk]);
    }
    vertMomRHS(velx_h, velz_h, theta_h, exner_h, &Vu2);
    massRHS(velx_h, velz_h, rho_h, &Fp2);
    tempRHS(velx_h, velz_h, rho_h, theta_h, &Ft2);

    // solve for the full step values
    for(kk = 0; kk < geom->nk; kk++) {
        // horizontal momentum
        VecZeroEntries(bu);
        VecCopy(velx[kk], bu);
        VecAXPY(bu, -0.5*dt, Hu1[kk]);
        VecAXPY(bu, -0.5*dt, Hu2[kk]);
        M1->assemble(kk);
        VecZeroEntries(velx[kk]);
        KSPSolve(ksp1, bu, velx[kk]);

        // density
        VecAXPY(rho[kk], -0.5*dt, Fp1[kk]);
        VecAXPY(rho[kk], -0.5*dt, Fp2[kk]);

        // potential temperature
        VecAXPY(theta[kk], -0.5*dt, Ft1[kk]);
        VecAXPY(theta[kk], -0.5*dt, Ft2[kk]);

        // exner pressure (second order)
        if(kk == 0) {
            for(ii = 0; ii < geom->nk; ii++) {
                VecScale(theta_i[ii], 0.5);
                VecAXPY(theta_i[ii], 0.5, theta_h[ii]);
            }
        }
        VecScale(rho_i[kk],   0.5);
        VecScale(exner_i[kk], 0.5);
        VecAXPY(rho_i[kk],   0.5, rho_h[kk]);
        VecAXPY(exner_i[kk], 0.5, exner_h[kk]);
        progExner(rho_i[kk], rho[kk], theta_i, theta, exner_i[kk], &exner_f, kk);
        VecCopy(exner_f, exner[kk]);
        VecDestroy(&exner_f);
    }

    // solve for the vertical velocity
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;

            AssembleVertOps(ex, ey, A);
            VecZeroEntries(bw);
            VecCopy(velz[ii], bw);
            VecAXPY(bw, -0.5*dt, Vu1[ii]);
            VecAXPY(bw, -0.5*dt, Vu2[ii]);
            VecZeroEntries(velz[ii]);
            KSPSolve(kspCol, bw, velz[ii]);
        }
    }

    // write output
    if(save) {
        step++;
        for(kk = 0; kk < geom->nk; kk++) {
            curl(velx[kk], &wi, kk, false);

            sprintf(fieldname, "vorticity_%.3u", kk);
            geom->write0(wi, fieldname, step);
            sprintf(fieldname, "velocity_%.3u",  kk);
            geom->write1(velx[kk], fieldname, step);
            sprintf(fieldname, "density_%.3u",  kk);
            geom->write2(rho[kk], fieldname, step);
            sprintf(fieldname, "theta_%.3u",  kk);
            geom->write2(theta[kk], fieldname, step);
            sprintf(fieldname, "exner_%.3u",  kk);
            geom->write2(exner[kk], fieldname, step);

            VecDestroy(&wi);
        }
    }

    // deallocate
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Hu1[kk]);
        VecDestroy(&Fp1[kk]);
        VecDestroy(&Ft1[kk]);
        VecDestroy(&Hu2[kk]);
        VecDestroy(&Fp2[kk]);
        VecDestroy(&Ft2[kk]);
        VecDestroy(&velx_h[kk]);
        VecDestroy(&rho_h[kk]);
        VecDestroy(&theta_h[kk]);
        VecDestroy(&exner_h[kk]);
        VecDestroy(&rho_i[kk]);
        VecDestroy(&theta_i[kk]);
        VecDestroy(&exner_i[kk]);
    }
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&Vu1[ii]);
        VecDestroy(&Vu2[ii]);
        VecDestroy(&velz_h[ii]);
    }
    delete[] Hu1;
    delete[] Vu1;
    delete[] Fp1;
    delete[] Ft1;
    delete[] Hu2;
    delete[] Vu2;
    delete[] Fp2;
    delete[] Ft2;
    delete[] velx_h;
    delete[] rho_h;
    delete[] theta_h;
    delete[] exner_h;
    delete[] rho_i;
    delete[] theta_i;
    delete[] exner_i;
    delete[] velz_h;
    VecDestroy(&bw);
    VecDestroy(&bu);
    MatDestroy(&A);
    KSPDestroy(&kspCol);
}
#endif
