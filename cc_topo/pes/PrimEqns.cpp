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

using namespace std;

PrimEqns::PrimEqns(Topo* _topo, Geom* _geom, double _dt) {
    int ii;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    grav = 9.80616*(RAD_SPHERE/RAD_EARTH);
    omega = 7.292e-5;
    del2 = viscosity();
    do_visc = true;
    vert_visc = 1.0; //TODO
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
    //T = new Whmat(topo, geom, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

    // assemble the vertical gradient and divergence incidence matrices
    vertOps();

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp1);
    KSPSetOperators(ksp1, M1->M, M1->M);
    KSPSetTolerances(ksp1, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1, KSPGMRES);
    KSPGetPC(ksp1,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp1,"1_");
    KSPSetFromOptions(ksp1);

    // initialize the 2 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPGetPC(ksp2,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, topo->elOrd*topo->elOrd, NULL);
    KSPSetOptionsPrefix(ksp2,"2_");
    KSPSetFromOptions(ksp2);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_b);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_t);

    Kv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->n2, &Kv[ii]);
    }
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double PrimEqns::viscosity() {
    double ae = 4.0*M_PI*RAD_SPHERE*RAD_SPHERE;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
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

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;
    //delete T;

    delete edge;
    delete node;
    delete quad;
}

void PrimEqns::UpdateKEVert(Vec ke, int lev) {
    int ex, ey, n2, ii;
    int* inds2;
    Vec kl;
    PetscScalar *khArray, *kvArray;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &kl);
    VecScatterBegin(topo->gtol_2, ke, kl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, ke, kl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(kl, &khArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds2 = topo->elInds2_l(ex, ey);

            VecGetArray(Kv[ey*topo->nElsX + ex], &kvArray);
            for(ii = 0; ii < n2; ii++) {
                kvArray[lev*n2+ii] = khArray[inds2[ii]];
            }
            VecRestoreArray(Kv[ey*topo->nElsX + ex], &kvArray);
        }
    }
    VecRestoreArray(ke, &khArray);

    VecDestroy(&kl);
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

void PrimEqns::vertMomRHS(Vec* ui, Vec* wi, Vec* theta, Vec* exner, Vec **fw) {
    int ex, ey, ei, ii, kk, n2;
    int* inds2;
    Vec exner_v, de1, de2, de3, dp;
    Mat A, B;
    PC pc;
    KSP kspCol;
    PetscScalar *evArray, *ehArray;

    n2 = topo->elOrd*topo->elOrd;

    // vertical velocity is computer per element, so matrices are local to this processor
    MatCreate(MPI_COMM_SELF, &A);
    MatSetType(A, MATSEQAIJ);
    MatSetSizes(A, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(A, topo->elOrd*topo->elOrd, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &B);
    MatSetType(B, MATSEQAIJ);
    //MatSetSizes(B, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2);
    MatSetSizes(B, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(B, topo->elOrd*topo->elOrd, PETSC_NULL);

    *fw = new Vec[topo->nElsX*topo->nElsX];
    for(ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        //VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*topo->n2, fw[ei]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->n2, fw[ei]);
        VecZeroEntries(*fw[ei]);
    }

    VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->n2, &de1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->n2, &de2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->n2, &de3);
    VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->n2, &dp);
    VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->n2, &exner_v);

    KSPCreate(MPI_COMM_SELF, &kspCol);
    KSPSetOperators(kspCol, B, B);
    KSPSetTolerances(kspCol, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspCol, KSPGMRES);
    KSPGetPC(kspCol,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, n2, NULL);
    KSPSetOptionsPrefix(kspCol,"kspVert_");
    KSPSetFromOptions(kspCol);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds2 = topo->elInds2_l(ex, ey);

            // add in the kinetic energy gradient
            MatMult(V01, Kv[ei], *fw[ei]);

            // add in the pressure gradient            
            VecGetArray(exner_v, &evArray);
            for(kk = 0; kk < geom->nk; kk++) {
                VecGetArray(exner[kk], &ehArray);
                for(ii = 0; ii < n2; ii++) {
                    evArray[kk*n2+ii] = ehArray[inds2[ii]];
                }
                VecRestoreArray(exner[kk], &ehArray);
            }
            VecRestoreArray(exner_v, &evArray);

            VecZeroEntries(de1);
            VecZeroEntries(de2);
            VecZeroEntries(de3);
            AssembleConst(ex, ey, A);
            MatMult(A, exner_v, de1);
            MatMult(V01, de1, de2);
            KSPSolve(kspCol, de2, de3);

            // interpolate the potential temperature onto the piecewise linear
            // vertical mass matrix and multiply by the weak form vertical gradient of
            // the exner pressure
            AssembleLinearWithTheta(ex, ey, theta, B);
            MatMult(B, de3, dp);
            VecAXPY(*fw[ei], 1.0, dp);

            // TODO: add in horizontal vorticity terms
        }
    }

    VecDestroy(&exner_v);
    VecDestroy(&de1);
    VecDestroy(&de2);
    VecDestroy(&de3);
    VecDestroy(&dp);
    MatDestroy(&A);
    MatDestroy(&B);
    KSPDestroy(&kspCol);
}

/*
compute the continuity equation right hand side for all levels
uh: horiztonal velocity by vertical level
uv: vertical velocity by horiztonal element
*/
void PrimEqns::massRHS(Vec* uh, Vec* uv, Vec* pi, Vec **Fp) {
    int ii, kk, ex, ey, n2, *inds2;
    Vec Mpu, Fv, Dv;
    Vec pl, pu, Fi, Dh;
    Mat Mp, B;
    PC pc;
    KSP kspCol;
    PetscScalar* dArray, *fArray;

    n2 = topo->elOrd*topo->elOrd;

    //VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &Mpu);
    //VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &Fv);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Mpu);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Fv);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Dv);

    // allocate the RHS vectors at each level
    Fp = new Vec*[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Fp[kk]);
        VecZeroEntries(*Fp[kk]);
    }

    // compute the vertical mass fluxes (piecewise linear in the vertical)
    MatCreate(MPI_COMM_SELF, &B);
    MatSetType(B, MATSEQAIJ);
    MatSetSizes(B, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(B, n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &Mp);
    MatSetType(Mp, MATSEQAIJ);
    //MatSetSizes(Mp, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2);
    MatSetSizes(Mp, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(Mp, topo->elOrd*topo->elOrd, PETSC_NULL);

    KSPCreate(MPI_COMM_SELF, &kspCol);
    KSPSetOperators(kspCol, B, B);
    KSPSetTolerances(kspCol, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspCol, KSPGMRES);
    KSPGetPC(kspCol,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, n2, NULL);
    KSPSetOptionsPrefix(kspCol,"kspCol_");
    KSPSetFromOptions(kspCol);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds2 = topo->elInds2_l(ex, ey);

            VertFlux(ex, ey, pi, NULL, Mp);
            MatMult(Mp, uv[ey*topo->nElsX+ex], Mpu);
            AssembleLinear(ex, ey, B, false);
            KSPSolve(kspCol, Mpu, Fv);
            // strong form vertical divergence
            MatMult(V10, Fv, Dv);

            // copy the vertical contribution to the divergence into the
            // horiztonal vectors
            VecGetArray(Dv, &dArray);
            for(kk = 0; kk < geom->nk; kk++) {
                VecGetArray(*Fp[kk], &fArray);
                for(ii = 0; ii < n2; ii++) {
                    fArray[inds2[ii]] += dArray[kk*n2+ii];
                }
                VecRestoreArray(*Fp[kk], &fArray);
            }
            VecRestoreArray(Dv, &dArray);
        }
    }
    VecDestroy(&Mpu);
    VecDestroy(&Fv);
    VecDestroy(&Dv);
    MatDestroy(&B);
    MatDestroy(&Mp);
    KSPDestroy(&kspCol);

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
        VecAXPY(*Fp[kk], 1.0, Dh);
    }

    VecDestroy(&pl);
    VecDestroy(&pu);
    VecDestroy(&Fi);
    VecDestroy(&Dh);
}

/*
compute the temperature equation right hand side for all levels
uh: horiztonal velocity by vertical level
uv: vertical velocity by horiztonal element
*/
void PrimEqns::tempRHS(Vec* uh, Vec* uv, Vec* pi, Vec* theta, Vec **Ft) {
    int ii, kk, ex, ey, n2, *inds2;
    Vec Mtu, Fv, Dv;
    Vec pl, pu, Fi, Dh;
    Vec theta_k, theta_k_l;
    Mat Mt, B;
    PC pc;
    KSP kspCol;
    PetscScalar* dArray, *fArray;

    n2 = topo->elOrd*topo->elOrd;

    //VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &Mtu);
    //VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &Fv);
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
    //MatSetSizes(Mt, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2);
    MatSetSizes(Mt, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(Mt, topo->elOrd*topo->elOrd, PETSC_NULL);

    KSPCreate(MPI_COMM_SELF, &kspCol);
    KSPSetOperators(kspCol, B, B);
    KSPSetTolerances(kspCol, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspCol, KSPGMRES);
    KSPGetPC(kspCol,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, n2, NULL);
    KSPSetOptionsPrefix(kspCol,"kspCol_");
    KSPSetFromOptions(kspCol);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds2 = topo->elInds2_l(ex, ey);

            VertFlux(ex, ey, pi, theta, Mt);
            MatMult(Mt, uv[ey*topo->nElsX+ex], Mtu);
            AssembleLinear(ex, ey, B, false);
            KSPSolve(kspCol, Mtu, Fv);
            // strong form vertical divergence
            MatMult(V10, Fv, Dv);

            // copy the vertical contribution to the divergence into the
            // horiztonal vectors
            VecGetArray(Dv, &dArray);
            for(kk = 0; kk < geom->nk; kk++) {
                VecGetArray(*Ft[kk], &fArray);
                for(ii = 0; ii < n2; ii++) {
                    fArray[inds2[ii]] += dArray[kk*n2+ii];
                }
                VecRestoreArray(*Ft[kk], &fArray);
            }
            VecRestoreArray(Dv, &dArray);
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

/*
diagnose theta from rho X theta (with boundary condition)
*/
void PrimEqns::diagTheta(Vec* rho, Vec* rt, Vec** theta) {
    int ex, ey, n2;
    Mat A;
    PC pc;
    KSP kspCol;

    n2 = topo->elOrd*topo->elOrd;

    *theta = new Vec[topo->nElsX*topo->nElsX];

    MatCreate(MPI_COMM_SELF, &A);
    MatSetType(A, MATSEQAIJ);
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

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            AssembleLinearWithRho(ex, ey, rho, A);
        }
    }

    MatDestroy(&A);
    KSPDestroy(&kspCol);
}


/*
prognose the exner pressure
*/
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
    KSPGetPC(kspE,&pc);
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

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void PrimEqns::grad(Vec phi, Vec* u, int lev) {
    Vec dPhi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dPhi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, u);

    M2->assemble(lev);
    if(!E12M2) {
        MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E12M2);
    } else {
        MatMatMult(EtoF->E12, M2->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &E12M2);
    }

    VecZeroEntries(dPhi);
    MatMult(E12M2, phi, dPhi);
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
    VecAXPY(*ddu, +1.0, RCu); // TODO: check sign here

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

    MatCreate(MPI_COMM_WORLD, &V10);
    MatSetType(V10, MATSEQAIJ);
    //MatSetSizes(V10, (geom->nk+0)*n2, (geom->nk+1)*n2, (geom->nk+0)*n2, (geom->nk+1)*n2);
    MatSetSizes(V10, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(V10, 2, PETSC_NULL);

    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < n2; ii++) {
            rows[0] = kk*n2 + ii;
            //cols[0] = (kk+0)*n2 + ii;
            //cols[1] = (kk+1)*n2 + ii;
            //MatSetValues(V10, 1, rows, 2, cols, vals, INSERT_VALUES);

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
void PrimEqns::AssembleLinear(int ex, int ey, Mat A, bool add_g) {
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

    MatZeroEntries(A);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);
        ei = ey*topo->nElsX + ex;
        
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]/det/det;
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            Q0[ii][ii] *= geom->thick[kk][inds0[ii]]/2.0;

            if(add_g) {
                Q0[ii][ii] *= (1.0 + dt*GRAVITY);
            }
        }

        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                //inds2k[ii] = ii + kk*W->nDofsJ;
                inds2k[ii] = ii + (kk-1)*W->nDofsJ;
            }
            MatSetValues(A, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < W->nDofsJ; ii++) {
                //inds2k[ii] = ii + (kk+1)*W->nDofsJ;
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

void PrimEqns::AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A) {
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
            Q0[ii][ii] = Q->A[ii][ii]/det/det;

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
                //inds2k[ii] = ii + kk*W->nDofsJ;
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
                //inds2k[ii] = ii + (kk+1)*W->nDofsJ;
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
        Inv(WtQW, WtQWinv, topo->elOrd);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, Aflat);

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
}

/*
derive the vertical mass flux
*/
void PrimEqns::VertFlux(int ex, int ey, Vec* pi, Vec* ti, Mat Mp) {
    int ii, kk, ei, mp1, mp12;
    int* inds;
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

    inds  = topo->elInds2_g(ex, ey);
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

        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = inds[ii] + kk*W->nDofsJ;
        }
        MatSetValues(Mp, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = inds[ii] + (kk+1)*W->nDofsJ;
        }
        MatSetValues(Mp, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

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
    Mat B, L, DA, Binv, BA;

    MatCreate(MPI_COMM_SELF, &B);
    MatSetType(B, MATSEQAIJ);
    MatSetSizes(B, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(B, n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &Binv);
    MatSetType(Binv, MATSEQAIJ);
    MatSetSizes(Binv, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(Binv, n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &L);
    MatSetType(L, MATSEQAIJ);
    //MatSetSizes(L, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2);
    MatSetSizes(L, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(L, n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &BA);
    MatSetType(BA, MATSEQAIJ);
    //MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk+1)*n2, (geom->nk+0)*n2, (geom->nk+1)*n2);
    MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(BA, n2, PETSC_NULL);

    AssembleLinear(ex, ey, A, false);
    AssembleConst(ex, ey, B);
    MatMatMult(V01, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DA);

    // get the inverse of the B matrix (locally on this processor)
    VertConstMatInv(ex, ey, Binv);

    MatMatMult(Binv, DA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &BA);
    MatMatMult(V10, BA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &L);

    // assemble the piecewise linear mass matrix (with gravity)
    AssembleLinear(ex, ey, A, true);
    MatAXPY(A, vert_visc, L, SAME_NONZERO_PATTERN);

    MatDestroy(&B);
    MatDestroy(&Binv);
    MatDestroy(&BA);
    MatDestroy(&L);
}

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
    KSPGetPC(kspCol,&pc);
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
            VecScale(theta_i[kk], 0.5);
            VecAXPY(theta_i[kk], 0.5, theta_h[kk]);
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

#if 0
/*
vertical gravity forcing gradient term (to be assembled 
into the left hand side as an implicit term)
*/
void PrimEqns::AssembleGrav(int ex, int ey, Mat Mg) {
    int ii, kk, ei, mp12;
    int* inds, *inds0;
    double det;
    int rows[99], cols[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];

    inds  = topo->elInds2_g(ex, ey);
    inds0 = topo->elInds0_g(ex, ey);
    mp12  = (quad->n + 1)*(quad->n + 1);

    MatZeroEntries(Mg);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);
        ei = ey*topo->nElsX + ex;
       
        // assemble the lower layer 
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]/det/det;

            // row is piecewise constant and column is piecewise linear after scaling 
            // by the jacobian determinant the metric term is 1 (ie: do nothing)
             
            // evaluate gravity at the layer interface
            Q0[ii][ii] *= geom->levs[kk][inds0[ii]]*GRAVITY;
        }

        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = inds[ii] + kk*W->nDofsJ;
            cols[ii] = inds[ii] + kk*W->nDofsJ;
        }
        MatSetValues(Mg, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);

        // assemble the upper layer
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]/det/det;

            // row is piecewise constant and column is piecewise linear after scaling 
            // by the jacobian determinant the metric term is 1 (ie: do nothing)
             
            // evaluate gravity at the layer interface
            Q0[ii][ii] *= geom->levs[kk+1][inds0[ii]]*GRAVITY;
        }

        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = inds[ii] + kk*W->nDofsJ;
            cols[ii] = inds[ii] + (kk+1)*W->nDofsJ;
        }
        MatSetValues(Mg, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
    }
    MatAssemblyBegin(Mg, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mg, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

/*
kinetic energy vector for the 2 form column from the 
horiztonal kinetic energy vectors already assembled
*/
void PrimEqns::VerticalKE(int ex, int ey, Vec* kh, Vec* kv) {
    int kk, jj;
    int n2 = topo->elOrd*topo->elOrd;
    int* inds_2 = topo->elInds2_g(ex, ey);
    PetscScalar *khArray, *kvArray;

    // vertical kinetic energy vector is piecewise constant in each level
    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, kv);
    VecGetArray(*kv, &kvArray);

    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(kh[kk], &khArray);

        for(jj = 0; jj < n2; jj++) {
            kvArray[kk*n2+jj] = khArray[inds_2[jj]];
        }
        VecRestoreArray(kh[kk], &khArray);
    }

    VecRestoreArray(*kv, &kvArray);
}
#endif
