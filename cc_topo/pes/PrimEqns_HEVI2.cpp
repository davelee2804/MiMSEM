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
#include "Assembly.h"
#include "PrimEqns_HEVI2.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define GAMMA (RD/CP)
#define CV (CP/1.4)
#define P0 100000.0

#define SCALE 1.0e+8

using namespace std;

PrimEqns_HEVI2::PrimEqns_HEVI2(Topo* _topo, Geom* _geom, double _dt) {
    int ii, n2;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    grav = GRAVITY;
    omega = OMEGA;
    do_visc = true;
    del2 = viscosity();
    vert_visc = viscosity_vert();
    step = 0;

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
    vertOps();

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

    Kv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->elOrd*topo->elOrd, &Kv[ii]);
    }
    Kh = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Kh[ii]);
    }
    gz = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &gz[ii]);
    }
    gv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd*topo->elOrd, &gv[ii]);
    }

    // initialise the single column mass matrices and solvers
    n2 = topo->elOrd*topo->elOrd;

    MatCreate(MPI_COMM_SELF, &VA);
    MatSetType(VA, MATSEQAIJ);
    MatSetSizes(VA, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VA, 2*topo->elOrd*topo->elOrd, PETSC_NULL);

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
    PCBJacobiSetTotalBlocks(pc, n2, NULL); // TODO: check allocation
    KSPSetOptionsPrefix(kspColB, "kspColB_");
    KSPSetFromOptions(kspColB);

    KSPCreate(MPI_COMM_WORLD, &kspE);
    KSPSetOperators(kspE, T->M, T->M);
    KSPSetTolerances(kspE, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspE, KSPGMRES);
    KSPGetPC(kspE, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, topo->elOrd*topo->elOrd, NULL);
    KSPSetOptionsPrefix(kspE, "exner_");
    KSPSetFromOptions(kspE);

    initGZ();
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double PrimEqns_HEVI2::viscosity() {
    double ae = 4.0*M_PI*RAD_EARTH*RAD_EARTH;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

double PrimEqns_HEVI2::viscosity_vert() {
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
void PrimEqns_HEVI2::coriolis() {
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
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg[kk]);
        m0->assemble(kk, 1.0, true);
        VecPointwiseDivide(fg[kk], PtQfxg, m0->vg);
    }
    
    delete PtQ;
    VecDestroy(&fl);
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

void PrimEqns_HEVI2::initGZ() {
    int ii, kk, ex, ey, ei, n2, mp12;
    int *inds0;
    double det;
    int inds2k[99], inds0k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double* WtQflat = new double[W->nDofsJ*Q->nDofsJ];
    Vec gq;
    Mat AQ;

    n2    = topo->elOrd*topo->elOrd;
    mp12  = (quad->n + 1)*(quad->n + 1);

    VecCreateSeq(MPI_COMM_SELF, geom->nk*mp12, &gq);
    VecSet(gq, grav);

    MatCreate(MPI_COMM_SELF, &AQ);
    MatSetType(AQ, MATSEQAIJ);
    MatSetSizes(AQ, (geom->nk-1)*n2, (geom->nk+0)*mp12, (geom->nk-1)*n2, (geom->nk+0)*mp12);
    MatSeqAIJSetPreallocation(AQ, 2*mp12, PETSC_NULL);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            MatZeroEntries(AQ);

            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);

            // Assemble the matrices
            for(kk = 0; kk < geom->nk; kk++) {
                // build the 2D mass matrix
                Q->assemble(ex, ey);

                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det);
                    // for linear field we multiply by the vertical jacobian determinant when 
                    // integrating, and do no other trasformations for the basis functions
                    Q0[ii][ii] *= geom->thick[kk][inds0[ii]]/2.0;
                }

                Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
                Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
                Flat2D_IP(W->nDofsJ, Q->nDofsJ, WtQ, WtQflat);

                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + kk*mp12;
                }
                // assemble the first basis function
                if(kk > 0) {
                    for(ii = 0; ii < W->nDofsJ; ii++) {
                        inds2k[ii] = ii + (kk-1)*W->nDofsJ;
                    }
                    MatSetValues(AQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQflat, ADD_VALUES);
                }
                // assemble the second basis function
                if(kk < geom->nk - 1) {
                    for(ii = 0; ii < W->nDofsJ; ii++) {
                        inds2k[ii] = ii + (kk+0)*W->nDofsJ;
                    }
                    MatSetValues(AQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQflat, ADD_VALUES);
                }
            }
            MatAssemblyBegin(AQ, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(AQ, MAT_FINAL_ASSEMBLY);

            MatMult(AQ, gq, gv[ei]);
        }
    }

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    delete[] WtQflat;
    delete Q;
    delete W;
    VecDestroy(&gq);
    MatDestroy(&AQ);
}

PrimEqns_HEVI2::~PrimEqns_HEVI2() {
    int ii;

    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);
    KSPDestroy(&kspE);
    VecDestroy(&theta_b);
    VecDestroy(&theta_t);

    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&fg[ii]);
        VecDestroy(&Kh[ii]);
        VecDestroy(&gz[ii]);
    }
    delete[] fg;
    delete[] Kh;
    delete[] gz;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&Kv[ii]);
    }
    delete[] Kv;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
    }
    delete[] gv;

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

/*
*/
void PrimEqns_HEVI2::AssembleKEVecs(Vec* velx, Vec* velz) {
    int ex, ey, ei, ii, jj, kk, mp1, mp12, n2, rows[99], cols[99];
    double det, wb, wt, wi, gamma;
    Mat BA;
    Vec velx_l, *Kh_l;
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar *kvArray;
    Vec* Kv2;

    n2   = topo->elOrd*topo->elOrd;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;

    Kv2 = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &Kv2[ii]);
        VecZeroEntries(Kv2[ii]);
    }

    // assemble the horiztonal operators
    Kh_l = new Vec[geom->nk];
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &velx_l);
    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(velx_l);
        VecScatterBegin(topo->gtol_1, velx[kk], velx_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(topo->gtol_1, velx[kk], velx_l, INSERT_VALUES, SCATTER_FORWARD);
        K->assemble(velx_l, kk, SCALE);
        VecZeroEntries(Kh[kk]);
        MatMult(K->M, velx[kk], Kh[kk]);

        VecCreateSeq(MPI_COMM_SELF, topo->n2l, &Kh_l[kk]);
        VecScatterBegin(topo->gtol_2, Kh[kk], Kh_l[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(topo->gtol_2, Kh[kk], Kh_l[kk], INSERT_VALUES, SCATTER_FORWARD);
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

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            MatZeroEntries(BA);
            ei = ey*topo->nElsX + ex;
            VecGetArray(velz[ei], &kvArray);

            // Assemble the matrices
            for(kk = 0; kk < geom->nk; kk++) {
                // build the 2D mass matrix
                Q->assemble(ex, ey);

                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);

                    // multiply by the vertical jacobian, then scale the piecewise constant 
                    // basis by the vertical jacobian, so do nothing 

                    // interpolate the vertical velocity at the quadrature point
                    wb = wt = 0.0;
                    for(jj = 0; jj < n2; jj++) {
                        gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                        if(kk > 0)            wb += kvArray[(kk-1)*n2+jj]*gamma;
                        if(kk < geom->nk - 1) wt += kvArray[(kk+0)*n2+jj]*gamma;
                    }
                    wi = 0.5*(wb + wt);   // quadrature weights are both 1.0, however ke is 0.5*w^2
                    Q0[ii][ii] *= wi/det; // vertical velocity is a 2 form in the horiztonal
                }

                Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
                Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

                for(ii = 0; ii < W->nDofsJ; ii++) {
                    rows[ii] = ii + kk*W->nDofsJ;
                }

                // assemble the first basis function
                if(kk > 0) {
                    for(ii = 0; ii < W->nDofsJ; ii++) {
                        cols[ii] = ii + (kk-1)*W->nDofsJ;
                    }
                    MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
                }

                // assemble the second basis function
                if(kk < geom->nk - 1) {
                    for(ii = 0; ii < W->nDofsJ; ii++) {
                        cols[ii] = ii + (kk+0)*W->nDofsJ;
                    }
                    MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
                }
            }
            MatAssemblyBegin(BA, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(BA, MAT_FINAL_ASSEMBLY);
            VecRestoreArray(velz[ei], &kvArray);

            VecZeroEntries(Kv2[ei]);
            MatMult(BA, velz[ei], Kv2[ei]);
        }
    }

    // add the vertical contribution to the horiztonal vector
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            VertToHoriz2(ex, ey, 0, geom->nk, Kv2[ei], Kh_l, false);
        }
    }
    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, Kh_l[kk], Kh[kk], INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(topo->gtol_2, Kh_l[kk], Kh[kk], INSERT_VALUES, SCATTER_REVERSE);
    }

    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&Kv2[ii]);
    }
    delete[] Kv2;
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Kh_l[kk]);
    }
    delete[] Kh_l;
    MatDestroy(&BA);
    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

/*
compute the right hand side for the momentum equation for a given level
note that the vertical velocity, uv, is stored as a different vector for 
each element
*/
void PrimEqns_HEVI2::horizMomRHS(Vec uh, Vec* theta_l, Vec exner, int lev, Vec Fu) {
    Vec wl, wi, Ru, Ku, Mh, d2u, d4u, theta_k, dExner, dp;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);

    curl(uh, &wi, lev, true);
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
    VecAXPY(theta_k, 1.0, theta_l[lev+0]); // quadrature weights
    VecAXPY(theta_k, 1.0, theta_l[lev+1]); // are both 1.0

    grad(exner, &dExner, lev);
    F->assemble(theta_k, lev, false, SCALE);
    MatMult(F->M, dExner, dp);
    VecAXPY(Fu, 1.0, dp);
    VecDestroy(&dExner);

    // add in the biharmonic voscosity
    if(do_visc) {
        laplacian(uh, &d2u, lev);
        laplacian(d2u, &d4u, lev);
        VecAXPY(Fu, SCALE, d4u);
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

// theta and exner are local vectors and fw is a vertical vector
void PrimEqns_HEVI2::vertMomRHS(Vec* theta, Vec* exner, Vec* fw) {
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

            // add the vertical gravity vector
            VecAXPY(fw[ei], -1.0, gv[ei]);

            // add in the pressure gradient
            VecZeroEntries(exner_v);
            HorizToVert2(ex, ey, exner, exner_v);

            VecZeroEntries(de1);
            VecZeroEntries(de2);
            VecZeroEntries(de3);
            AssembleConst(ex, ey, VB);
            MatMult(VB, exner_v, de1);
            MatMult(V01, de1, de2);
            AssembleLinear(ex, ey, VA); //TODO: skip this and just solve with B(theta)?? on LHS
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

// pi is a local vector
void PrimEqns_HEVI2::massRHS_h(Vec* uh, Vec* pi, Vec* Fp) {
    int kk;
    Vec pu, Fh;

    // compute the horiztonal mass fluxes
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &pu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fh);

    for(kk = 0; kk < geom->nk; kk++) {
        F->assemble(pi[kk], kk, true, SCALE);
        M1->assemble(kk, SCALE, true);
        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fh);
        MatMult(EtoF->E21, Fh, Fp[kk]);
    }

    VecDestroy(&pu);
    VecDestroy(&Fh);
}

// all three vectors are vertical 
void PrimEqns_HEVI2::massRHS_v(Vec* uz, Vec* pi, Vec* Fp) {
    int ex, ey, ei, n2;
    Vec Mpu, Fv;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Mpu);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Fv);

    // compute the vertical mass fluxes (piecewise linear in the vertical)
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            VecZeroEntries(Fv);
            VertFlux(ex, ey, pi[ei], VA);
            MatMult(VA, uz[ei], Mpu);
            AssembleLinear(ex, ey, VA);
            KSPSolve(kspColA, Mpu, Fv);
            // strong form vertical divergence
            MatMult(V10, Fv, Fp[ei]);
        }
    }
    VecDestroy(&Mpu);
    VecDestroy(&Fv);
}

/*
Assemble the boundary condition vector for rho(t) X theta(0)
*/
void PrimEqns_HEVI2::thetaBCVec(int ex, int ey, Mat A, Vec* rho, Vec* bTheta) {
    int* inds2 = topo->elInds2_l(ex, ey);
    int* inds0 = topo->elInds0_l(ex, ey);
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

    Q->assemble(ex, ey);
    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &theta_o);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, bTheta);

    MatZeroEntries(A);

    // bottom boundary
    VecGetArray(rho[0], &rArray);
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);

        // multuply by the vertical determinant to integrate, then
        // divide piecewise constant density by the vertical determinant,
        // so these cancel
        geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rArray, &rk);
        Q0[ii][ii] *= rk;
        //TODO: scaling seems to work, but don't understand why yet
        Q0[ii][ii] *= 2.0/geom->thick[0][inds0[ii]];
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
    VecGetArray(rho[geom->nk-1], &rArray);
    for(ii = 0; ii < mp12; ii++) {
        det = geom->det[ei][ii];
        Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);

        // multuply by the vertical determinant to integrate, then
        // divide piecewise constant density by the vertical determinant,
        // so these cancel
        geom->interp2_g(ex, ey, ii%mp1, ii/mp1, rArray, &rk);
        Q0[ii][ii] *= rk;
        //TODO: scaling seems to work, but don't understand why yet
        Q0[ii][ii] *= 2.0/geom->thick[geom->nk-1][inds0[ii]];
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
note: rho, rhoTheta and theta are all LOCAL vectors
*/
void PrimEqns_HEVI2::diagTheta(Vec* rho, Vec* rt, Vec* theta) {
    int ex, ey, n2, kk;
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
            VecZeroEntries(rtv);
            HorizToVert2(ex, ey, rt, rtv);
            AssembleLinCon(ex, ey, AB);
            MatMult(AB, rtv, frt);

            // assemble in the bcs
            thetaBCVec(ex, ey, A, rho, &bcs);
            VecAXPY(frt, -1.0, bcs);
            VecDestroy(&bcs);

            AssembleLinearWithRho(ex, ey, rho, VA);
            KSPSolve(kspColA, frt, theta_v);
            VertToHoriz2(ex, ey, 1, geom->nk, theta_v, theta, false);
        }
    }

    VecDestroy(&rtv);
    VecDestroy(&frt);
    VecDestroy(&theta_v);
    MatDestroy(&A);
    MatDestroy(&AB);
}

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void PrimEqns_HEVI2::grad(Vec phi, Vec* u, int lev) {
    Vec Mphi, dMphi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, u);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mphi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMphi);

    M1->assemble(lev, SCALE, false); //TODO: vertical scaling of this operator causes problems??
    M2->assemble(lev, SCALE, false);

    MatMult(M2->M, phi, Mphi);
    MatMult(EtoF->E12, Mphi, dMphi);
    KSPSolve(ksp1, dMphi, *u);

    VecDestroy(&Mphi);
    VecDestroy(&dMphi);
}

/*
Take the weak form curl of a 1 form vector field as a 1 form vector field
*/
void PrimEqns_HEVI2::curl(Vec u, Vec* w, int lev, bool add_f) {
    Vec Mu, dMu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &dMu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu);

    m0->assemble(lev, SCALE, true);
    M1->assemble(lev, SCALE, true);
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

void PrimEqns_HEVI2::laplacian(Vec ui, Vec* ddu, int lev) {
    Vec Du, Cu, RCu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, ddu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &RCu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Du);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, ui, Du);

    // grad (weak form)
    grad(Du, ddu, lev);

    /*** rotational component ***/
    // curl (weak form)
    curl(ui, &Cu, lev, false);

    // rot (strong form)
    MatMult(NtoE->E10, Cu, RCu);

    // add rotational and divergent components
    VecAXPY(*ddu, +1.0, RCu);
    VecScale(*ddu, del2);

    VecDestroy(&Cu);
    VecDestroy(&RCu);
    VecDestroy(&Du);
}

/*
assemble the vertical gradient and divergence orientation matrices
V10 is the strong form vertical divergence from the linear to the
constant basis functions
*/
void PrimEqns_HEVI2::vertOps() {
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
void PrimEqns_HEVI2::AssembleConst(int ex, int ey, Mat B) {
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
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);
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
void PrimEqns_HEVI2::AssembleLinear(int ex, int ey, Mat A) {
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
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det/det);
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

void PrimEqns_HEVI2::AssembleLinCon(int ex, int ey, Mat AB) {
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
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);

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

void PrimEqns_HEVI2::AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A) {
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

    ei   = ey*topo->nElsX + ex;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;

    Q->assemble(ex, ey);
    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatZeroEntries(A);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        VecGetArray(rho[kk], &rArray);
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);

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

void PrimEqns_HEVI2::AssembleLinearWithTheta(int ex, int ey, Vec* theta, Mat A) {
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
            QB[ii][ii]  = Q->A[ii][ii]*(SCALE/det/det);
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            QB[ii][ii] *= geom->thick[kk][inds0[ii]]/2.0;
            QT[ii][ii]  = QB[ii][ii];

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
*/
void PrimEqns_HEVI2::VertFlux(int ex, int ey, Vec pi, Mat Mp) {
    int ii, jj, kk, ei, n2, mp1, mp12;
    int* inds0 = topo->elInds0_l(ex, ey);
    double det, rho, gamma;
    int inds2k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar *pArray;

    ei   = ey*topo->nElsX + ex;
    n2   = topo->elOrd*topo->elOrd;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;

    // build the 2D mass matrix
    Q->assemble(ex, ey);
    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    MatZeroEntries(Mp);

    // assemble the matrices
    VecGetArray(pi, &pArray);
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);

            //geom->interp2_g(ex, ey, ii%mp1, ii/mp1, pArray, &rho);
            rho = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                rho += pArray[(kk+0)*n2+jj]*gamma;
            }
            rho *= 2.0/(geom->thick[kk][inds0[ii]]*det);
            Q0[ii][ii] *= rho;

            // multiply by the vertical determinant for the vertical integral,
            // then divide by the vertical determinant to rescale the piecewise
            // constant density, so do nothing.
        }

        // assemble the piecewise constant mass matrix for level k
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
    VecRestoreArray(pi, &pArray);
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

void PrimEqns_HEVI2::AssembleVertLaplacian(int ex, int ey, Mat A) {
    int n2 = topo->elOrd*topo->elOrd;
    Mat B, L, BD;

    MatCreate(MPI_COMM_SELF, &B);
    MatSetType(B, MATSEQAIJ);
    MatSetSizes(B, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(B, n2, PETSC_NULL);

    AssembleConst(ex, ey, B);

    // construct the laplacian mixing operator
    // TODO: preallocate these
    MatMatMult(B, V10, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &BD);
    MatMatMult(V01, BD, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L);

    // assemble the piecewise linear mass matrix (with gravity)
    MatAXPY(A, -vert_visc, L, DIFFERENT_NONZERO_PATTERN);//TODO: check the sign on the viscosity

    MatDestroy(&B);
    MatDestroy(&BD);
    MatDestroy(&L);
}

void PrimEqns_HEVI2::HorizRHS(Vec* velx, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, Vec* Fu, Vec* Fp, Vec* Ft) {
    int kk;
    Vec* theta;

    theta = new Vec[geom->nk+1];
    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta[kk]);
    }
    // set the top and bottom potential temperature bcs
    VecScatterBegin(topo->gtol_2, theta_b, theta[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_b, theta[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, theta_t, theta[geom->nk], INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_t, theta[geom->nk], INSERT_VALUES, SCATTER_FORWARD);

    diagTheta(rho->vl, rt->vl, theta);

    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx[kk], theta, exner->vh[kk], kk, Fu[kk]);
    }
    massRHS_h(velx, rho->vl, Fp);
    massRHS_h(velx, rt->vl,  Ft);

    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecDestroy(&theta[kk]);
    }
    delete[] theta;
}

// rt and Ft and local vectors
void PrimEqns_HEVI2::SolveExner(Vec* rt, Vec* Ft, Vec* exner_i, Vec* exner_f, double _dt) {
    int ii;
    Vec rt_sum, rhs;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &rt_sum);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);

    for(ii = 0; ii < geom->nk; ii++) {
        VecCopy(Ft[ii], rt_sum);
        VecScale(rt_sum, -_dt*(GAMMA/(1.0-GAMMA))*pow(RD/P0,GAMMA/(1.0-GAMMA)));
        VecAXPY(rt_sum, 1.0, rt[ii]);
        T->assemble(rt_sum, ii, SCALE);
        MatMult(T->M, exner_i[ii], rhs);
        
        T->assemble(rt[ii], ii, SCALE);
        KSPSolve(kspE, rhs, exner_f[ii]);
    }
    VecDestroy(&rt_sum);
    VecDestroy(&rhs);
}

// density, density weighted potential temperature and exner are local vectors and velz is a vertical vector
void PrimEqns_HEVI2::SolveVertMom(Vec* rho, Vec* rt, Vec* exner, Vec* velz, double _dt) {
    int  ii, ex, ey;
    int  n2     = topo->elOrd*topo->elOrd;
    Vec* Fz     = new Vec[n2];
    Vec* theta  = new Vec[geom->nk+1];
    Mat  BA;

    for(ii = 0; ii < geom->nk + 1; ii++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta[ii]);
    }
    // set the top and bottom potential temperature bcs
    VecScatterBegin(topo->gtol_2, theta_b, theta[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_b, theta[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, theta_t, theta[geom->nk], INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_t, theta[geom->nk], INSERT_VALUES, SCATTER_FORWARD);

    diagTheta(rho, rt, theta);

    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Fz[ii]);
    }

    MatCreate(MPI_COMM_SELF, &BA);
    MatSetType(BA, MATSEQAIJ);
    MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(BA, 2*n2, PETSC_NULL);

    vertMomRHS(theta, exner, Fz);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;
            solveMom(_dt, ex, ey, BA, velz[ii], Fz[ii]);
        }
    }

    for(ii = 0; ii < geom->nk + 1; ii++) {
        VecDestroy(&theta[ii]);
    }
    delete[] theta;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&Fz[ii]);
    }
    delete[] Fz;
    MatDestroy(&BA);
}

// velz is a vertical vector and rho is a local vector
void PrimEqns_HEVI2::SolveVertMass(Vec* velz, Vec* rho, double _dt) {
    int ex, ey, ii;
    int n2;
    Vec rho_z, Fv;
    Mat AB;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &rho_z);
    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &Fv);

    MatCreate(MPI_COMM_SELF, &AB);
    MatSetType(AB, MATSEQAIJ);
    MatSetSizes(AB, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(AB, 2*n2, PETSC_NULL);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;
            // update the density
            VecZeroEntries(rho_z);
            HorizToVert2(ex, ey, rho, rho_z);
            VecCopy(rho_z, Fv);
            solveMass(_dt, ex, ey, AB, velz[ii], Fv, rho_z);
            VertToHoriz2(ex, ey, 0, geom->nk, rho_z, rho, true);
        }
    }

    VecDestroy(&rho_z);
    VecDestroy(&Fv);
    MatDestroy(&AB);
}

void PrimEqns_HEVI2::SolveStrang(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int     ii;
    char    fieldname[100];
    Vec     bu, xu, wi;
    Vec*    Fu       = new Vec[geom->nk];
    Vec*    Fp       = new Vec[geom->nk];
    Vec*    velx_i   = new Vec[geom->nk];
    Vec*    rho_i    = new Vec[geom->nk];
    Vec*    rt_i     = new Vec[geom->nk];
    Vec*    exner_i  = new Vec[geom->nk];
    Vec*    exner_j  = new Vec[geom->nk];
    Vec*    velz_i   = new Vec[topo->nElsX*topo->nElsX];
    L2Vecs* l2_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* l2_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* l2_exner = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* l2_Ft    = new L2Vecs(geom->nk, topo, geom);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &xu);
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fu[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Fp[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_i[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho_i[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt_i[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner_i[ii]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner_j[ii]);
    }
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd*topo->elOrd, &velz_i[ii]);
        VecCopy(velz[ii], velz_i[ii]);
    }

    // 1.  half step in the vertical
    l2_rho->CopyFromHoriz(rho);
    l2_rt->CopyFromHoriz(rt);
    l2_exner->CopyFromHoriz(exner);

    l2_rho->UpdateLocal();
    l2_rt->UpdateLocal();
    l2_exner->UpdateLocal();

    // 1.1 vertical momentum
    SolveVertMom(l2_rho->vl, l2_rt->vl, l2_exner->vl, velz_i, 0.5*dt);

    // 1.2 vertical density
    SolveVertMass(velz, l2_rho->vl, 0.5*dt);
    
    // 1.3 vertical density
    SolveVertMass(velz, l2_rt->vl, 0.5*dt);

    // 1.4 vertical exner pressure
    l2_rt->HorizToVert();
    massRHS_v(velz, l2_rt->vz, l2_Ft->vz);
    l2_Ft->VertToHoriz();
    SolveExner(l2_rt->vl, l2_Ft->vl, exner, l2_exner->vh, 0.5*dt);

    // 1.5 update the solution vectors
    l2_rho->UpdateGlobal();
    l2_rt->UpdateGlobal();

    l2_rho->CopyToHoriz(rho);
    l2_rt->CopyToHoriz(rt);
    l2_exner->CopyToHoriz(exner);
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCopy(velz_i[ii], velz[ii]);
    }

    // 2.1 first horiztonal substep
    l2_rho->UpdateLocal();
    l2_rt->UpdateLocal();

    HorizRHS(velx, l2_rho, l2_rt, l2_exner, Fu, Fp, l2_Ft->vh);
    for(ii = 0; ii < geom->nk; ii++) {
        // momentum
        M1->assemble(ii, SCALE, true);
        MatMult(M1->M, velx[ii], bu);
        VecAXPY(bu, -dt, Fu[ii]);
        KSPSolve(ksp1, bu, velx_i[ii]);

        // continuity
        VecCopy(rho[ii], rho_i[ii]);
        VecAXPY(rho_i[ii], -dt, Fp[ii]);
        
        // internal energy
        VecCopy(rt[ii], rt_i[ii]);
        VecAXPY(rt_i[ii], -dt, l2_Ft->vh[ii]);
    }
    l2_Ft->UpdateLocal();
    SolveExner(l2_rt->vl, l2_Ft->vl, exner, exner_i, dt);

    // 2.2 second horiztonal substep
    l2_rho->CopyFromHoriz(rho_i);
    l2_rt->CopyFromHoriz(rt_i);
    l2_exner->CopyFromHoriz(exner_i);

    l2_rho->UpdateLocal();
    l2_rt->UpdateLocal();

    HorizRHS(velx_i, l2_rho, l2_rt, l2_exner, Fu, Fp, l2_Ft->vh);
    for(ii = 0; ii < geom->nk; ii++) {
        // momentum
        VecZeroEntries(xu);
        VecAXPY(xu, 0.75, velx[ii]);
        VecAXPY(xu, 0.25, velx_i[ii]);
        M1->assemble(ii, SCALE, true);
        MatMult(M1->M, xu, bu);
        VecAXPY(bu, -0.25*dt, Fu[ii]);
        KSPSolve(ksp1, bu, velx_i[ii]);

        // continuity
        VecScale(rho_i[ii], 0.25);
        VecAXPY(rho_i[ii], 0.75, rho[ii]);
        VecAXPY(rho_i[ii], -0.25*dt, Fp[ii]);
        
        // internal energy
        VecScale(rt_i[ii], 0.25);
        VecAXPY(rt_i[ii], 0.75, rt[ii]);
        VecAXPY(rt_i[ii], -0.25*dt, l2_Ft->vh[ii]);
    }
    for(ii = 0; ii < geom->nk; ii++) {
        VecZeroEntries(exner_j[ii]);
        VecAXPY(exner_j[ii], 0.75, exner[ii]);
        VecAXPY(exner_j[ii], 0.25, exner_i[ii]);
    }
    l2_Ft->UpdateLocal();
    SolveExner(l2_rt->vl, l2_Ft->vl, exner_j, exner_i, 0.25*dt);

    // 2.3 third horiztonal substep
    l2_rho->CopyFromHoriz(rho_i);
    l2_rt->CopyFromHoriz(rt_i);
    l2_exner->CopyFromHoriz(exner_i);

    l2_rho->UpdateLocal();
    l2_rt->UpdateLocal();

    HorizRHS(velx_i, l2_rho, l2_rt, l2_exner, Fu, Fp, l2_Ft->vh);
    for(ii = 0; ii < geom->nk; ii++) {
        // momentum
        VecZeroEntries(xu);
        VecAXPY(xu, 1.0/3.0, velx[ii]);
        VecAXPY(xu, 2.0/3.0, velx_i[ii]);
        M1->assemble(ii, SCALE, true);
        MatMult(M1->M, xu, bu);
        VecAXPY(bu, (-2.0/3.0)*dt, Fu[ii]);
        KSPSolve(ksp1, bu, velx_i[ii]);

        // continuity
        VecScale(rho_i[ii], 2.0/3.0);
        VecAXPY(rho_i[ii], 1.0/3.0, rho[ii]);
        VecAXPY(rho_i[ii], (-2.0/3.0)*dt, Fp[ii]);
        
        // internal energy
        VecScale(rt_i[ii], 2.0/3.0);
        VecAXPY(rt_i[ii], 1.0/3.0, rt[ii]);
        VecAXPY(rt_i[ii], (-2.0/3.0)*dt, l2_Ft->vh[ii]);
    }
    for(ii = 0; ii < geom->nk; ii++) {
        VecZeroEntries(exner_j[ii]);
        VecAXPY(exner_j[ii], 1.0/3.0, exner[ii]);
        VecAXPY(exner_j[ii], 2.0/3.0, exner_i[ii]);
    }
    l2_Ft->UpdateLocal();
    SolveExner(l2_rt->vl, l2_Ft->vl, exner_j, exner_i, (2.0/3.0)*dt);

    // 2.4 update the solution vectors
    for(ii = 0; ii < geom->nk; ii++) {
        VecCopy(velx_i[ii] , velx[ii] );
        VecCopy(rho_i[ii]  , rho[ii]  );
        VecCopy(rt_i[ii]   , rt[ii]   );
        VecCopy(exner_i[ii], exner[ii]);
    }

    // 3.  half step in the vertical
    l2_rho->CopyFromHoriz(rho);
    l2_rt->CopyFromHoriz(rt);
    l2_exner->CopyFromHoriz(exner);

    l2_rho->UpdateLocal();
    l2_rt->UpdateLocal();
    l2_exner->UpdateLocal();

    // 3.1 vertical momentum
    SolveVertMom(l2_rho->vl, l2_rt->vl, l2_exner->vl, velz_i, 0.5*dt);

    // 3.2 vertical density
    SolveVertMass(velz, l2_rho->vl, 0.5*dt);
    
    // 3.3 vertical density
    SolveVertMass(velz, l2_rt->vl, 0.5*dt);

    // 3.4 vertical exner pressure
    l2_rt->HorizToVert();
    massRHS_v(velz, l2_rt->vz, l2_Ft->vz);
    l2_Ft->VertToHoriz();
    SolveExner(l2_rt->vl, l2_Ft->vl, exner, l2_exner->vh, 0.5*dt);

    // 3.5 update the solution vectors
    l2_rho->UpdateGlobal();
    l2_rt->UpdateGlobal();

    l2_rho->CopyToHoriz(rho);
    l2_rt->CopyToHoriz(rt);
    l2_exner->CopyToHoriz(exner);
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCopy(velz_i[ii], velz[ii]);
    }

    // write output
    if(save) {
        step++;
        for(ii = 0; ii < geom->nk; ii++) {
            curl(velx[ii], &wi, ii, false);

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
        sprintf(fieldname, "velVert");
        geom->writeSerial(velz, fieldname, topo->nElsX*topo->nElsX, step);
    }

    delete l2_rho;
    delete l2_rt;
    delete l2_exner;
    delete l2_Ft;
    VecDestroy(&bu);
    VecDestroy(&xu);
    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&Fu[ii]);
        VecDestroy(&Fp[ii]);
        VecDestroy(&velx_i[ii]);
        VecDestroy(&rho_i[ii]);
        VecDestroy(&rt_i[ii]);
        VecDestroy(&exner_i[ii]);
        VecDestroy(&exner_j[ii]);
    }
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&velz_i[ii]);
    }
    delete[] Fu;
    delete[] Fp;
    delete[] velx_i;
    delete[] rho_i;
    delete[] rt_i;
    delete[] exner_i;
    delete[] exner_j;
    delete[] velz_i;
}

void PrimEqns_HEVI2::VertToHoriz2(int ex, int ey, int ki, int kf, Vec pv, Vec* ph, bool assign) {
    int ii, kk, n2;
    int* inds2 = topo->elInds2_l(ex, ey);
    PetscScalar *hArray, *vArray;

    n2 = topo->elOrd*topo->elOrd;

    VecGetArray(pv, &vArray);
    for(kk = ki; kk < kf; kk++) {
        VecGetArray(ph[kk], &hArray);
        for(ii = 0; ii < n2; ii++) {
            if(assign) {
                hArray[inds2[ii]]  = vArray[(kk-ki)*n2+ii];
            }
            else {
                hArray[inds2[ii]] += vArray[(kk-ki)*n2+ii];
            }
        }
        VecRestoreArray(ph[kk], &hArray);
    }
    VecRestoreArray(pv, &vArray);
}

void PrimEqns_HEVI2::HorizToVert2(int ex, int ey, Vec* ph, Vec pv) {
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

void PrimEqns_HEVI2::init0(Vec* q, ICfunc3D* func) {
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

        m0->assemble(kk, 1.0, true);
        MatMult(PQ->M, bg, PQb);
        VecPointwiseDivide(q[kk], PQb, m0->vg);
    }

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void PrimEqns_HEVI2::init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y) {
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

void PrimEqns_HEVI2::init2(Vec* h, ICfunc3D* func) {
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

void PrimEqns_HEVI2::initTheta(Vec theta, ICfunc3D* func) {
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

    M2->assemble(0, SCALE, true); // note: layer thickness must be set to 2.0 for all layers 
    MatMult(WQ->M, bg, WQb);      //       before M2 matrix is assembled to initialise theta
    VecScale(WQb, SCALE);
    KSPSolve(ksp2, WQb, theta);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void PrimEqns_HEVI2::solveMass(double _dt, int ex, int ey, Mat AB, Vec wz, Vec fv, Vec rho) {
    int ii, jj, kk, ei, n2, mp1, mp12;
    int *inds0;
    double det, wb, wt, wi, gamma;
    int rows[99], cols[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double** WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar* wArray;
    Mat DAinv, Op;
    PC pc;
    KSP kspMass;

    ei    = ey*topo->nElsX + ex;
    inds0 = topo->elInds0_l(ex, ey);
    n2    = topo->elOrd*topo->elOrd;
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    // 1. assemble the piecewise linear/constant matrix
    MatZeroEntries(AB);
    VecGetArray(wz, &wArray);

    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);

            // multiply by the vertical jacobian, then scale the piecewise constant
            // basis by the vertical jacobian, so do nothing

            // interpolate the vertical velocity at the quadrature point
            wb = wt = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                if(kk > 0)            wb += wArray[(kk-1)*n2+jj]*gamma;
                if(kk < geom->nk - 1) wt += wArray[(kk+0)*n2+jj]*gamma;
            }
            wi = 1.0*(wb + wt);   // quadrature weights are both 1.0
            Q0[ii][ii] *= wi/det; // vertical velocity is a 2 form in the horiztonal
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
    VecGetArray(wz, &wArray);
    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);

    // 2. assemble the piecewise linear/linear matrix inverse
    MatZeroEntries(VA);

    for(kk = 0; kk < geom->nk-1; kk++) {
        Q->assemble(ex, ey);

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii]  = Q->A[ii][ii]*(SCALE/det/det);
            // for linear field we multiply by the vertical jacobian determinant when
            // integrating, and do no other trasformations for the basis functions
            Q0[ii][ii] *= (geom->thick[kk+0][inds0[ii]]/2.0 + geom->thick[kk+1][inds0[ii]]/2.0);
        }
        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);

        // take the inverse
        Inv(WtQW, WtQWinv, n2);
        // add to matrix
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQWinv, WtQWflat);
        for(ii = 0; ii < W->nDofsJ; ii++) {
            rows[ii] = ii + kk*W->nDofsJ;
        }
        MatSetValues(VA, W->nDofsJ, rows, W->nDofsJ, rows, WtQWflat, ADD_VALUES);
    }
    MatAssemblyBegin(VA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(VA, MAT_FINAL_ASSEMBLY);

    MatMatMult(V10, VA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DAinv);
    MatMatMult(DAinv, AB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Op);
    MatScale(Op, _dt);
    MatShift(Op, 1.0);

    KSPCreate(MPI_COMM_SELF, &kspMass);
    KSPSetOperators(kspMass, Op, Op);
    KSPSetTolerances(kspMass, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspMass, KSPGMRES);
    KSPGetPC(kspMass, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, W->nDofsJ, NULL); // TODO: check allocation
    KSPSetOptionsPrefix(kspMass, "kspMass_");
    KSPSetFromOptions(kspMass);

    KSPSolve(kspMass, fv, rho);
    KSPDestroy(&kspMass);

    MatDestroy(&DAinv);
    MatDestroy(&Op);
    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

void PrimEqns_HEVI2::solveMom(double _dt, int ex, int ey, Mat BA, Vec wz, Vec fv) {
    int ii, jj, kk, ei, mp1, mp12, n2, it = 0, rank;
    int rows[99], cols[99];
    double det, wb, wt, gamma, eps = 1.0e+9, l2_dif, l2_old;
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double** WtQWinv = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar* zArray;
    Vec wz_f, dw;
    Mat DBA = NULL;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ei    = ey*topo->nElsX + ex;
    n2    = topo->elOrd*topo->elOrd;
    mp1   = quad->n+1;
    mp12  = mp1*mp1;

    VecAYPX(fv, -_dt, wz);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &wz_f);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &dw);

    Q->assemble(ex, ey);
    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    do {
        // assemble the matrix
        MatZeroEntries(BA);
        VecGetArray(wz, &zArray);

        // Assemble the matrices
        for(kk = 0; kk < geom->nk; kk++) {
            // build the 2D mass matrix
            if(kk > 0) {
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);

                    // multiply by the vertical jacobian, then scale the piecewise constant
                    // basis by the vertical jacobian, so do nothing

                    // interpolate the vertical velocity at the quadrature point
                    wb = 0.0;
                    for(jj = 0; jj < n2; jj++) {
                        gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                        wb += zArray[(kk-1)*n2+jj]*gamma;
                    }
                    Q0[ii][ii] *= 0.5*wb/det;
                }

                Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
                Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

                for(ii = 0; ii < W->nDofsJ; ii++) {
                    rows[ii] = ii + (kk+0)*W->nDofsJ;
                    cols[ii] = ii + (kk-1)*W->nDofsJ;
                }
                MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
            }

            if(kk < geom->nk - 1) {
                for(ii = 0; ii < mp12; ii++) {
                    det = geom->det[ei][ii];
                    Q0[ii][ii] = Q->A[ii][ii]*(SCALE/det/det);

                    // multiply by the vertical jacobian, then scale the piecewise constant
                    // basis by the vertical jacobian, so do nothing

                    // interpolate the vertical velocity at the quadrature point
                    wt = 0.0;
                    for(jj = 0; jj < n2; jj++) {
                        gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                        wt += zArray[(kk+0)*n2+jj]*gamma;
                    }
                    Q0[ii][ii] *= 0.5*wt/det;
                }

                Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
                Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
                Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);
                for(ii = 0; ii < W->nDofsJ; ii++) {
                    rows[ii] = ii + (kk+0)*W->nDofsJ;
                    cols[ii] = ii + (kk+0)*W->nDofsJ;
                }
                MatSetValues(BA, W->nDofsJ, rows, W->nDofsJ, cols, WtQWflat, ADD_VALUES);
            }
        }
        MatAssemblyBegin(BA, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(BA, MAT_FINAL_ASSEMBLY);
        VecRestoreArray(wz, &zArray);

        AssembleLinear(ex, ey, VA);
        AssembleVertLaplacian(ex, ey, VA);
        if(!DBA) {
            MatMatMult(V01, BA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DBA);
        } else {
            MatMatMult(V01, BA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DBA);
        }
        MatAXPY(VA, _dt, DBA, DIFFERENT_NONZERO_PATTERN);
        KSPSolve(kspColA, fv, wz_f);

        VecCopy(wz_f, dw);
        VecAXPY(dw, -1.0, wz);
        VecNorm(dw, NORM_2, &l2_dif);
        VecNorm(wz, NORM_2, &l2_old);
        eps = fabs(l2_dif/l2_old);
        VecCopy(wz_f, wz);

        it++;
    } while(it < 100 && eps > 1.0e-12);

    if(!rank) cout << "vert mom, it: " << it << "\teps: " << eps << endl;

    VecDestroy(&wz_f);
    VecDestroy(&dw);
    MatDestroy(&DBA);

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    Free2D(W->nDofsJ, WtQWinv);
    delete[] WtQWflat;
    delete Q;
    delete W;
}
