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
#include "PrimEqns_HEVI.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define GAMMA (RD/CP)
#define CV (CP/1.4)
#define P0 100000.0

using namespace std;

#define ADD_IE
#define ADD_GZ
#define ADD_VERT_EXNER_FLUX

PrimEqns_HEVI::PrimEqns_HEVI(Topo* _topo, Geom* _geom, double _dt) {
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

#ifdef ADD_GZ
    initGZ();
#endif
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double PrimEqns_HEVI::viscosity() {
    double ae = 4.0*M_PI*RAD_EARTH*RAD_EARTH;
    double dx = sqrt(ae/topo->nDofs0G);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

double PrimEqns_HEVI::viscosity_vert() {
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
void PrimEqns_HEVI::coriolis() {
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

void PrimEqns_HEVI::initGZ() {
    int ii, kk, ex, ey, ei, n2, mp12;
    int *inds0;
    double det;
    double scale = 1.0e8;
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
                    Q0[ii][ii]  = Q->A[ii][ii]*(scale/det);
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

PrimEqns_HEVI::~PrimEqns_HEVI() {
    int ii;

    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);
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
void PrimEqns_HEVI::AssembleKEVecs(Vec* velx, Vec* velz, double scale) {
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
        K->assemble(velx_l, kk, scale);
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
                    Q0[ii][ii] = Q->A[ii][ii]*(scale/det/det);

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
void PrimEqns_HEVI::horizMomRHS(Vec uh, Vec* uv, Vec* theta_l, Vec exner, int lev, double scale, Vec *Fu) {
    Vec wl, wi, Ru, Ku, Mh, d2u, d4u, theta_k, dExner, dp;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, Fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);

    curl(uh, &wi, lev, true);
    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(*Fu);
    R->assemble(wl, lev, scale);
    MatMult(R->M, uh, Ru);
    MatMult(EtoF->E12, Kh[lev], *Fu);
    VecAXPY(*Fu, 1.0, Ru);

    // add the thermodynamic term (theta is in the same space as the vertical velocity)
    // project theta onto 1 forms
#ifdef ADD_IE
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 1.0, theta_l[lev+0]); // quadrature weights
    VecAXPY(theta_k, 1.0, theta_l[lev+1]); // are both 1.0

    grad(exner, &dExner, lev);
    F->assemble(theta_k, lev, false, scale);
    MatMult(F->M, dExner, dp);
    VecAXPY(*Fu, 1.0, dp);
    VecDestroy(&dExner);
#endif

    // add in the biharmonic voscosity
    if(do_visc) {
        laplacian(uh, &d2u, lev);
        laplacian(d2u, &d4u, lev);
        VecAXPY(*Fu, scale, d4u);
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

void PrimEqns_HEVI::vertMomRHS(Vec* ui, Vec* wi, Vec* theta, Vec* exner, Vec* fw) {
    int ex, ey, ei, n2;
    double scale = 1.0e8;
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

#ifdef ADD_GZ
            // add the vertical gravity vector
            VecAXPY(fw[ei], -1.0, gv[ei]);
#endif

            // add in the pressure gradient
#ifdef ADD_IE
            VecZeroEntries(exner_v);
            HorizToVert2(ex, ey, exner, exner_v);

            VecZeroEntries(de1);
            VecZeroEntries(de2);
            VecZeroEntries(de3);
            AssembleConst(ex, ey, VB, scale);
            MatMult(VB, exner_v, de1);
            MatMult(V01, de1, de2);
            AssembleLinear(ex, ey, VA, scale);//TODO: skip this and just solve with B(theta)?? on LHS
            KSPSolve(kspColA, de2, de3);

            // interpolate the potential temperature onto the piecewise linear
            // vertical mass matrix and multiply by the weak form vertical gradient of
            // the exner pressure
            AssembleLinearWithTheta(ex, ey, theta, VA, scale);
            MatMult(VA, de3, dp);
            VecAXPY(fw[ei], 1.0, dp);
#endif
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
void PrimEqns_HEVI::massRHS(Vec* uh, Vec* pi, Vec* Fp) {
    int ex, ey, ei, kk;
    double scale = 1.0e8;
    Vec pl, pu, Fh, Dh, *Dh_l;

    Dh_l = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &Dh_l[kk]);
        VecZeroEntries(Dh_l[kk]);
    }

    // compute the horiztonal mass fluxes
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &pl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &pu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Dh);

    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, pi[kk], pl, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(topo->gtol_2, pi[kk], pl, INSERT_VALUES, SCATTER_FORWARD);

        // add the horiztonal fluxes
        F->assemble(pl, kk, true, scale);
        M1->assemble(kk, scale, true);
        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fh);
        MatMult(EtoF->E21, Fh, Dh);

        VecScatterBegin(topo->gtol_2, Dh, Dh_l[kk], INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(topo->gtol_2, Dh, Dh_l[kk], INSERT_VALUES, SCATTER_REVERSE);
    }

    // add horiztonal fluxes to the vertical vectors
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            VecZeroEntries(Fp[ei]);
            HorizToVert2(ex, ey, Dh_l, Fp[ei]);
        }
    }

    VecDestroy(&pl);
    VecDestroy(&pu);
    VecDestroy(&Fh);
    VecDestroy(&Dh);
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Dh_l[kk]);
    }
    delete[] Dh_l;
}

/*
Assemble the boundary condition vector for rho(t) X theta(0)
*/
void PrimEqns_HEVI::thetaBCVec(int ex, int ey, Mat A, Vec* rho, Vec* bTheta, double scale) {
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
        Q0[ii][ii] = Q->A[ii][ii]*(scale/det/det);

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
        Q0[ii][ii] = Q->A[ii][ii]*(scale/det/det);

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
void PrimEqns_HEVI::diagTheta(Vec* rho, Vec* rt, Vec* theta) {
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
            VecZeroEntries(rtv);
            HorizToVert2(ex, ey, rt, rtv);
            AssembleLinCon(ex, ey, AB, scale);
            MatMult(AB, rtv, frt);

            // assemble in the bcs
            thetaBCVec(ex, ey, A, rho, &bcs, scale);
            VecAXPY(frt, -1.0, bcs);
            VecDestroy(&bcs);

            AssembleLinearWithRho(ex, ey, rho, VA, scale);
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
prognose the exner pressure
reference: Gassman QJRMS 2013
*/
void PrimEqns_HEVI::progExner(Vec rt_i, Vec DivH, Vec DivV, Vec exner_i, Vec* exner_f, int lev) {
    double scale = 1.0e8;
    Vec rt_l, rhs, dG_l, rt_sum;
    PC pc;
    KSP kspE;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &rt_l);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &dG_l);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt_sum);
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

    VecScatterBegin(topo->gtol_2, DivH, dG_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, DivH, dG_l, INSERT_VALUES, SCATTER_FORWARD);

    //VecAXPY(rt_l, -dt*RD/CP, dG_l);
    VecAXPY(rt_l, -dt*(GAMMA/(1.0-GAMMA))*pow(RD/P0,GAMMA/(1.0-GAMMA)), dG_l);

    T->assemble(rt_l, lev, scale);
    MatMult(T->M, exner_i, rhs);

    // assemble the nonlinear operator
    // NOTE: density weighted potential temperature from the previous time level is
    //       also used on the left hand side
    VecCopy(rt_i, rt_sum);
    if(DivV) { // add the vertical component of the divergence
        //VecAXPY(rt_sum, dt*RD/CP, DivV);
        VecAXPY(rt_sum, dt*(GAMMA/(1.0-GAMMA))*pow(RD/P0,GAMMA/(1.0-GAMMA)), DivV);
    }
    VecScatterBegin(topo->gtol_2, rt_sum, rt_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, rt_sum, rt_l, INSERT_VALUES, SCATTER_FORWARD);
    T->assemble(rt_l, lev, scale);

    KSPSolve(kspE, rhs, *exner_f);

    VecDestroy(&rt_l);
    VecDestroy(&dG_l);
    VecDestroy(&rhs);
    VecDestroy(&rt_sum);
    KSPDestroy(&kspE);
}

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void PrimEqns_HEVI::grad(Vec phi, Vec* u, int lev) {
    double scale = 1.0e8;
    Vec Mphi, dMphi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, u);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mphi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMphi);

    M1->assemble(lev, scale, false); //TODO: vertical scaling of this operator causes problems??
    M2->assemble(lev, scale, false);

    MatMult(M2->M, phi, Mphi);
    MatMult(EtoF->E12, Mphi, dMphi);
    KSPSolve(ksp1, dMphi, *u);

    VecDestroy(&Mphi);
    VecDestroy(&dMphi);
}

/*
Take the weak form curl of a 1 form vector field as a 1 form vector field
*/
void PrimEqns_HEVI::curl(Vec u, Vec* w, int lev, bool add_f) {
    double scale = 1.0e8;
    Vec Mu, dMu;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &dMu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu);

    m0->assemble(lev, scale, true);
    M1->assemble(lev, scale, true);
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

void PrimEqns_HEVI::laplacian(Vec ui, Vec* ddu, int lev) {
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
void PrimEqns_HEVI::vertOps() {
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
void PrimEqns_HEVI::AssembleConst(int ex, int ey, Mat B, double scale) {
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
            Q0[ii][ii] = Q->A[ii][ii]*(scale/det/det);
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
void PrimEqns_HEVI::AssembleLinear(int ex, int ey, Mat A, double scale) {
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
            Q0[ii][ii]  = Q->A[ii][ii]*(scale/det/det);
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

void PrimEqns_HEVI::AssembleLinCon(int ex, int ey, Mat AB, double scale) {
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
            Q0[ii][ii] = Q->A[ii][ii]*(scale/det/det);

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

void PrimEqns_HEVI::AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A, double scale) {
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
            Q0[ii][ii] = Q->A[ii][ii]*(scale/det/det);

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

void PrimEqns_HEVI::AssembleLinearWithTheta(int ex, int ey, Vec* theta, Mat A, double scale) {
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
            QB[ii][ii]  = Q->A[ii][ii]*(scale/det/det);
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
TODO: only need a single piecewise constant field, may be either rho or rho X theta
*/
void PrimEqns_HEVI::VertFlux(int ex, int ey, Vec pi, Mat Mp, double scale) {
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
            Q0[ii][ii] = Q->A[ii][ii]*(scale/det/det);

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

void PrimEqns_HEVI::AssembleVertLaplacian(int ex, int ey, Mat A, double scale) {
    int n2 = topo->elOrd*topo->elOrd;
    Mat B, L, BD;

    MatCreate(MPI_COMM_SELF, &B);
    MatSetType(B, MATSEQAIJ);
    MatSetSizes(B, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(B, n2, PETSC_NULL);

    AssembleConst(ex, ey, B, scale);

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

// note: rho X theta must include the theta boundary conditions
void PrimEqns_HEVI::SolveRK2(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int ii, kk, ex, ey, n2;
    double scale = 1.0e8;
    char fieldname[100];
    Vec *Hu1, *Vu1, *Fp1, *Ft1, *velx_h, *velz_h, *rho_h, *rt_h, *exner_h, bu, bw, wi;
    Vec *Hu2, *Vu2, *Fp2, *Ft2, *rt_i, *exner_i, exner_f;
    Vec *theta_l, *exner_l, *rho_l, *rt_l;
    Mat AB, BA;

    n2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &bw);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    Hu1     = new Vec[geom->nk];
    Hu2     = new Vec[geom->nk];
    velx_h  = new Vec[geom->nk];
    rho_h   = new Vec[geom->nk];
    rt_h    = new Vec[geom->nk];
    exner_h = new Vec[geom->nk];
    rt_i    = new Vec[geom->nk];
    exner_i = new Vec[geom->nk];
    velz_h  = new Vec[topo->nElsX*topo->nElsX];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_h[kk] );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho_h[kk]  );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt_h[kk]   );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt_i[kk]   );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner_i[kk]);
        // temporary vectors for use in exner pressure prognosis
        VecCopy(rt[kk]   , rt_i[kk]   );
        VecCopy(exner[kk], exner_i[kk]);
    }
    // create local vectors for assembling vertical rhs terms
    rho_l   = new Vec[geom->nk];
    rt_l    = new Vec[geom->nk];
    theta_l = new Vec[geom->nk+1];
    exner_l = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &rho_l[kk]  );
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &rt_l[kk]   );
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &exner_l[kk]);
    }
    // create vectors for the potential temperature at the internal layer interfaces
    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_l[kk]);
    }
    // set the top and bottom potential temperature bcs
    VecScatterBegin(topo->gtol_2, theta_b, theta_l[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_b, theta_l[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, theta_t, theta_l[geom->nk], INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_t, theta_l[geom->nk], INSERT_VALUES, SCATTER_FORWARD);

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

    MatCreate(MPI_COMM_SELF, &AB);
    MatSetType(AB, MATSEQAIJ);
    MatSetSizes(AB, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(AB, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &BA);
    MatSetType(BA, MATSEQAIJ);
    MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(BA, 2*n2, PETSC_NULL);

    // assemble the vertical and horiztonal kinetic energy vectors
    AssembleKEVecs(velx, velz, scale);

    // construct the right hand side terms for the first substep
    // note: do horiztonal rhs first as this assembles the kinetic energy
    // operator for use in the vertical rhs
    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, rho[kk],   rho_l[kk],   INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, rho[kk],   rho_l[kk],   INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, rt[kk],    rt_l[kk],    INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, rt[kk],    rt_l[kk],    INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, exner[kk], exner_l[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, exner[kk], exner_l[kk], INSERT_VALUES, SCATTER_FORWARD);
    }
    diagTheta(rho_l, rt_l, theta_l);
    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx[kk], velz, theta_l, exner[kk], kk, scale, &Hu1[kk]);
    }
    vertMomRHS(velx, velz, theta_l, exner_l, Vu1);
    massRHS(velx, rho, Fp1);
    massRHS(velx, rt,  Ft1);

    // solve for the half step values
    for(kk = 0; kk < geom->nk; kk++) {
        // horizontal momentum
        M1->assemble(kk, scale, true);
        VecZeroEntries(bu);
        MatMult(M1->M, velx[kk], bu);
        VecAXPY(bu, -dt, Hu1[kk]);
        KSPSolve(ksp1, bu, velx_h[kk]);

        // density
        VecZeroEntries(rho_h[kk]);
        VecCopy(rho[kk], rho_h[kk]);
        VecAXPY(rho_h[kk], -dt, Fp1[kk]);

        // potential temperature
        VecZeroEntries(rt_h[kk]);
        VecCopy(rt_i[kk], rt_h[kk]);
        VecAXPY(rt_h[kk], -dt, Ft1[kk]);

        // exner pressure
        progExner(rt_i[kk], Ft1[kk], NULL, exner[kk], &exner_h[kk], kk);
    }

    // solve for the vertical velocity
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;

            VecZeroEntries(bw);
            AssembleLinear(ex, ey, VA, scale);
            MatMult(VA, velz[ii], bw);
            VecAXPY(bw, -dt, Vu1[ii]);
            AssembleVertLaplacian(ex, ey, VA, scale);
            KSPSolve(kspColA, bw, velz_h[ii]);
        }
    }

    // construct right hand side terms for the second substep

    // assemble the vertical and horiztonal kinetic energy vectors
    AssembleKEVecs(velx_h, velz_h, scale);

    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, rho_h[kk],   rho_l[kk],   INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, rho_h[kk],   rho_l[kk],   INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, rt_h[kk],    rt_l[kk],    INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, rt_h[kk],    rt_l[kk],    INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, exner_h[kk], exner_l[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, exner_h[kk], exner_l[kk], INSERT_VALUES, SCATTER_FORWARD);
    }
    diagTheta(rho_l, rt_l, theta_l);
    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx_h[kk], velz_h, theta_l, exner_h[kk], kk, scale, &Hu2[kk]);
    }
    vertMomRHS(velx_h, velz_h, theta_l, exner_l, Vu2);
    massRHS(velx_h, rho_h, Fp2);
    massRHS(velx_h, rt_h,  Ft2);

    // solve for the full step values
    for(kk = 0; kk < geom->nk; kk++) {
        // horizontal momentum
        M1->assemble(kk, scale, true);
        VecZeroEntries(bu);
        MatMult(M1->M, velx[kk], bu);
        VecAXPY(bu, -0.5*dt, Hu1[kk]);
        VecAXPY(bu, -0.5*dt, Hu2[kk]);
        VecZeroEntries(velx[kk]);
        KSPSolve(ksp1, bu, velx[kk]);

        // density
        VecAXPY(rho[kk], -0.5*dt, Fp1[kk]);
        VecAXPY(rho[kk], -0.5*dt, Fp2[kk]);

        // potential temperature
        VecAXPY(rt[kk], -0.5*dt, Ft1[kk]);
        VecAXPY(rt[kk], -0.5*dt, Ft2[kk]);

        // exner pressure (second order)
        VecScale(Ft1[kk], 0.5);
        VecAXPY(Ft1[kk], 0.5, Ft2[kk]);
        progExner(rt_i[kk], Ft1[kk], NULL, exner_i[kk], &exner_f, kk);
        VecCopy(exner_f, exner[kk]);
        VecDestroy(&exner_f);
    }

    // solve for the vertical velocity
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;

            VecZeroEntries(bw);
            AssembleLinear(ex, ey, VA, scale);
            MatMult(VA, velz[ii], bw);
            VecAXPY(bw, -0.5*dt, Vu1[ii]);
            VecAXPY(bw, -0.5*dt, Vu2[ii]);
            AssembleVertLaplacian(ex, ey, VA, scale);
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
            geom->write2(rho[kk], fieldname, step, kk, true);
            sprintf(fieldname, "rhoTheta");
            geom->write2(rt[kk], fieldname, step, kk, true);
            sprintf(fieldname, "exner");
            geom->write2(exner[kk], fieldname, step, kk, true);

            VecDestroy(&wi);
        }
        sprintf(fieldname, "velocity_z");
        geom->writeVertToHoriz(velz, fieldname, step, geom->nk-1);
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
        VecDestroy(&rt_i[kk]);
        VecDestroy(&exner_i[kk]);
        VecDestroy(&rho_l[kk]  );
        VecDestroy(&rt_l[kk]   );
        VecDestroy(&exner_l[kk]);
    }
    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecDestroy(&theta_l[kk]);
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
    delete[] rt_i;
    delete[] exner_i;
    delete[] velz_h;
    delete[] rho_l;
    delete[] rt_l;
    delete[] theta_l;
    delete[] exner_l;
    VecDestroy(&bw);
    VecDestroy(&bu);
    MatDestroy(&AB);
    MatDestroy(&BA);
}

#if 0
void PrimEqns_HEVI::SolveEuler(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int ii, kk, ex, ey, n2, rank;
    double scale = 1.0e8;
    char fieldname[100];
    Vec *Hu1, *Vu1, *Fp1, *Ft1, bu, bw, wi, rho_z;
    Vec *rt_i, *exner_i, exner_f, *rho_l, *rt_l, *theta_l, *exner_l, *Fth, *Ftv;
    Vec Mpu, Dv, Fv, pi;
    Mat AB, BA;

    n2 = topo->elOrd*topo->elOrd;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &rho_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &bw);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    Hu1     = new Vec[geom->nk];
    rt_i    = new Vec[geom->nk];
    exner_i = new Vec[geom->nk];
    rho_l   = new Vec[geom->nk];
    rt_l    = new Vec[geom->nk];
    theta_l = new Vec[geom->nk+1];
    exner_l = new Vec[geom->nk];
    Fth     = new Vec[geom->nk];
    Ftv     = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt_i[kk]   );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner_i[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Fth[kk]    );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ftv[kk]    );
        // temporary vectors for use in exner pressure prognosis
        VecCopy(rt[kk]   , rt_i[kk]   );
        VecCopy(exner[kk], exner_i[kk]);
        VecZeroEntries(Fth[kk]);
        VecZeroEntries(Ftv[kk]);

        VecCreateSeq(MPI_COMM_SELF, topo->n2, &rho_l[kk]  );
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &rt_l[kk]   );
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &exner_l[kk]);
    }
    // create vectors for the potential temperature at the internal layer interfaces
    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_l[kk]);
    }
    // set the top and bottom potential temperature bcs
    VecScatterBegin(topo->gtol_2, theta_b, theta_l[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_b, theta_l[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, theta_t, theta_l[geom->nk], INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_t, theta_l[geom->nk], INSERT_VALUES, SCATTER_FORWARD);

    // rhs vectors
    Vu1 = new Vec[topo->nElsX*topo->nElsX];
    Fp1 = new Vec[topo->nElsX*topo->nElsX];
    Ft1 = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Vu1[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Fp1[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Ft1[ii]);
        VecZeroEntries(Vu1[ii]);
        VecZeroEntries(Fp1[ii]);
        VecZeroEntries(Ft1[ii]);
    }

//#ifdef ADD_VERT_EXNER_FLUX
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Mpu);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Dv);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &pi);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Fv);
//#endif

    MatCreate(MPI_COMM_SELF, &AB);
    MatSetType(AB, MATSEQAIJ);
    MatSetSizes(AB, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(AB, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &BA);
    MatSetType(BA, MATSEQAIJ);
    MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(BA, 2*n2, PETSC_NULL);

    // assemble the vertical and horiztonal kinetic energy vectors
    AssembleKEVecs(velx, velz, scale);

    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, rho[kk],   rho_l[kk],   INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, rho[kk],   rho_l[kk],   INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, rt[kk],    rt_l[kk],    INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, rt[kk],    rt_l[kk],    INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, exner[kk], exner_l[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, exner[kk], exner_l[kk], INSERT_VALUES, SCATTER_FORWARD);
    }
    // construct the right hand side terms for the first substep
    // note: do horiztonal rhs first as this assembles the kinetic energy
    // operator for use in the vertical rhs
    if(!rank)cout<<"\tdiagnosing theta..........."<<endl;
    diagTheta(rho_l, rt_l, theta_l);
    if(!rank)cout<<"\thorizontal momentum rhs...."<<endl;
    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx[kk], velz, theta_l, exner[kk], kk, scale, &Hu1[kk]);
    }
    if(!rank)cout<<"\tvertical momentum rhs......"<<endl;
    vertMomRHS(velx, velz, theta_l, exner_l, Vu1);
    if(!rank)cout<<"\tcontinuity eqn rhs........."<<endl;
    massRHS(velx, rho, Fp1);
    if(!rank)cout<<"\tenergy eqn rhs............."<<endl;
    massRHS(velx, rt,  Ft1);

    // solve for the horiztonal velocity
    if(!rank)cout<<"\thorizontal momentum solve.."<<endl;
    for(kk = 0; kk < geom->nk; kk++) {
        // horizontal momentum
        M1->assemble(kk, scale, true);
        VecZeroEntries(bu);
        MatMult(M1->M, velx[kk], bu);
        VecAXPY(bu, -dt, Hu1[kk]);
        VecZeroEntries(velx[kk]);
        KSPSolve(ksp1, bu, velx[kk]);
    }

    // solve for the vertical velocity
    if(!rank)cout<<"\tvertical momentum solve...."<<endl;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;
            solveMom(dt, ex, ey, scale, BA, velz[ii], Vu1[ii]);
        }
    }

    // solve the continuity and energy equations
    if(!rank)cout<<"\tmass and temperature solve."<<endl;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;

            // update the density
            VecZeroEntries(rho_z);
            HorizToVert2(ex, ey, rho_l, rho_z);
            solveMass(dt, ex, ey, scale, AB, velz[ii], Fp1[ii], rho_z);
            VertToHoriz2(ex, ey, 0, geom->nk, rho_z, rho_l, true);

            // update the density weighted potential temperature
            VecZeroEntries(rho_z);
            HorizToVert2(ex, ey, rt_l,  rho_z);
            solveMass(dt, ex, ey, scale, AB, velz[ii], Ft1[ii], rho_z);
            VertToHoriz2(ex, ey, 0, geom->nk, rho_z, rt_l , true);
        }
    }

    // scatter the density and the density weighted potential temperature back
    // to the global vectors
    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, rho_l[kk], rho[kk], INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  topo->gtol_2, rho_l[kk], rho[kk], INSERT_VALUES, SCATTER_REVERSE);
        VecScatterBegin(topo->gtol_2, rt_l[kk],  rt[kk],  INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  topo->gtol_2, rt_l[kk],  rt[kk],  INSERT_VALUES, SCATTER_REVERSE);
    }

    // exner pressure
    // TODO: Fth should include both vertical and horizontal components 
    //       of the density weighted potential temperature divergence
    if(!rank)cout<<"\texner pressure solve......."<<endl;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;
            VertToHoriz2(ex, ey, 0, geom->nk, Ft1[ii], Fth, false);

            // add the vertical component of the density weighted potential
            // temperature divergence
//#ifdef ADD_VERT_EXNER_FLUX
            // compute the vertical mass fluxes (piecewise linear in the vertical)
            VecZeroEntries(Fv);
            VecZeroEntries(pi);
            HorizToVert2(ex, ey, rt_l, pi);
            VertFlux(ex, ey, pi, VA, scale);
            MatMult(VA, velz[ii], Mpu);
            AssembleLinear(ex, ey, VA, scale);
            KSPSolve(kspColA, Mpu, Fv);
            // strong form vertical divergence
            MatMult(V10, Fv, Dv);
            // copy the vertical contribution to the divergence into the
            // horiztonal vectors
            VertToHoriz2(ex, ey, 0, geom->nk, Dv, Ftv, false);
//#endif
        }
    }
    for(kk = 0; kk < geom->nk; kk++) {
        progExner(rt_i[kk], Fth[kk], Ftv[kk], exner[kk], &exner_f, kk);
        VecCopy(exner_f, exner[kk]);
        VecDestroy(&exner_f);
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
            geom->write2(rho[kk], fieldname, step, kk, true);
            sprintf(fieldname, "rhoTheta");
            geom->write2(rt[kk], fieldname, step, kk, true);
            sprintf(fieldname, "exner");
            geom->write2(exner[kk], fieldname, step, kk, true);

            VecDestroy(&wi);
        }
        sprintf(fieldname, "velocity_z");
        geom->writeVertToHoriz(velz, fieldname, step, geom->nk-1);
        sprintf(fieldname, "velVert");
        geom->writeSerial(velz, fieldname, topo->nElsX*topo->nElsX, step);
    }

    // deallocate
//#ifdef ADD_VERT_EXNER_FLUX
    VecDestroy(&Mpu);
    VecDestroy(&Dv);
    VecDestroy(&pi);
    VecDestroy(&Fv);
//#endif

    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Hu1[kk]    );
        VecDestroy(&rt_i[kk]   );
        VecDestroy(&exner_i[kk]);
        VecDestroy(&rho_l[kk]  );
        VecDestroy(&rt_l[kk]   );
        VecDestroy(&exner_l[kk]);
        VecDestroy(&Fth[kk]    );
        VecDestroy(&Ftv[kk]    );
    }
    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecDestroy(&theta_l[kk]);
    }
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&Vu1[ii]);
        VecDestroy(&Fp1[ii]);
        VecDestroy(&Ft1[ii]);
    }
    delete[] Hu1;
    delete[] Vu1;
    delete[] Fp1;
    delete[] Ft1;
    delete[] rt_i;
    delete[] exner_i;
    delete[] rho_l;
    delete[] rt_l;
    delete[] theta_l;
    delete[] exner_l;
    delete[] Fth;
    delete[] Ftv;
    VecDestroy(&rho_z);
    VecDestroy(&bw);
    VecDestroy(&bu);
    MatDestroy(&AB);
    MatDestroy(&BA);
}
#endif
void PrimEqns_HEVI::SolveEuler(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int ii, kk, ex, ey, n2, rank;
    double scale = 1.0e8;
    char fieldname[100];
    Vec *Hu1, *Vu1, *Fp1, *Ft1, bu, bw, wi, rho_z;
    Vec *rt_i, *exner_i, exner_f, *rho_l, *rt_l, *theta_l, *exner_l, *Fth, *Ftv;
    Vec Mpu, Dv, Fv, pi;
    Mat AB, BA;

    n2 = topo->elOrd*topo->elOrd;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &rho_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &bw);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);
    Hu1     = new Vec[geom->nk];
    rt_i    = new Vec[geom->nk];
    exner_i = new Vec[geom->nk];
    rho_l   = new Vec[geom->nk];
    rt_l    = new Vec[geom->nk];
    theta_l = new Vec[geom->nk+1];
    exner_l = new Vec[geom->nk];
    Fth     = new Vec[geom->nk];
    Ftv     = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt_i[kk]   );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner_i[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Fth[kk]    );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ftv[kk]    );
        // temporary vectors for use in exner pressure prognosis
        VecCopy(rt[kk]   , rt_i[kk]   );
        VecCopy(exner[kk], exner_i[kk]);
        VecZeroEntries(Fth[kk]);
        VecZeroEntries(Ftv[kk]);

        VecCreateSeq(MPI_COMM_SELF, topo->n2, &rho_l[kk]  );
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &rt_l[kk]   );
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &exner_l[kk]);
    }
    // create vectors for the potential temperature at the internal layer interfaces
    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_l[kk]);
    }
    // set the top and bottom potential temperature bcs
    VecScatterBegin(topo->gtol_2, theta_b, theta_l[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_b, theta_l[0],        INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_2, theta_t, theta_l[geom->nk], INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_2, theta_t, theta_l[geom->nk], INSERT_VALUES, SCATTER_FORWARD);

    // rhs vectors
    Vu1 = new Vec[topo->nElsX*topo->nElsX];
    Fp1 = new Vec[topo->nElsX*topo->nElsX];
    Ft1 = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Vu1[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Fp1[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Ft1[ii]);
        VecZeroEntries(Vu1[ii]);
        VecZeroEntries(Fp1[ii]);
        VecZeroEntries(Ft1[ii]);
    }

#ifdef ADD_VERT_EXNER_FLUX
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Mpu);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Dv);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &pi);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Fv);
#endif

    MatCreate(MPI_COMM_SELF, &AB);
    MatSetType(AB, MATSEQAIJ);
    MatSetSizes(AB, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2);
    MatSeqAIJSetPreallocation(AB, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &BA);
    MatSetType(BA, MATSEQAIJ);
    MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(BA, 2*n2, PETSC_NULL);

    // assemble the vertical and horiztonal kinetic energy vectors
    AssembleKEVecs(velx, velz, scale);

    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, rho[kk],   rho_l[kk],   INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, rho[kk],   rho_l[kk],   INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, rt[kk],    rt_l[kk],    INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, rt[kk],    rt_l[kk],    INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(topo->gtol_2, exner[kk], exner_l[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_2, exner[kk], exner_l[kk], INSERT_VALUES, SCATTER_FORWARD);
    }
    // construct the right hand side terms for the first substep
    // note: do horiztonal rhs first as this assembles the kinetic energy
    // operator for use in the vertical rhs
    if(!rank)cout<<"\tdiagnosing theta..........."<<endl;
    diagTheta(rho_l, rt_l, theta_l);
    if(!rank)cout<<"\thorizontal momentum rhs...."<<endl;
    // solve for the horiztonal velocity
    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx[kk], velz, theta_l, exner[kk], kk, scale, &Hu1[kk]);
        M1->assemble(kk, scale, true);
        VecZeroEntries(bu);
        MatMult(M1->M, velx[kk], bu);
        VecAXPY(bu, -dt, Hu1[kk]);
        VecZeroEntries(velx[kk]);
        KSPSolve(ksp1, bu, velx[kk]);
    }

    if(!rank)cout<<"\tcontinuity eqn rhs........."<<endl;
    massRHS(velx, rho, Fp1);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;
            // update the density
            VecZeroEntries(rho_z);
            HorizToVert2(ex, ey, rho_l, rho_z);
            solveMass(dt, ex, ey, scale, AB, velz[ii], Fp1[ii], rho_z);
            VertToHoriz2(ex, ey, 0, geom->nk, rho_z, rho_l, true);
        }
    }

    if(!rank)cout<<"\tenergy eqn rhs............."<<endl;
    massRHS(velx, rt,  Ft1);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;
            // update the density weighted potential temperature
            VecZeroEntries(rho_z);
            HorizToVert2(ex, ey, rt_l,  rho_z);
            solveMass(dt, ex, ey, scale, AB, velz[ii], Ft1[ii], rho_z);
            VertToHoriz2(ex, ey, 0, geom->nk, rho_z, rt_l , true);
        }
    }

    // scatter the density and the density weighted potential temperature back
    // to the global vectors
    for(kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_2, rho_l[kk], rho[kk], INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  topo->gtol_2, rho_l[kk], rho[kk], INSERT_VALUES, SCATTER_REVERSE);
        VecScatterBegin(topo->gtol_2, rt_l[kk],  rt[kk],  INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  topo->gtol_2, rt_l[kk],  rt[kk],  INSERT_VALUES, SCATTER_REVERSE);
    }

    // exner pressure
    if(!rank)cout<<"\texner pressure solve......."<<endl;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;
            VertToHoriz2(ex, ey, 0, geom->nk, Ft1[ii], Fth, false);

            // add the vertical component of the density weighted potential
            // temperature divergence
#ifdef ADD_VERT_EXNER_FLUX
            // compute the vertical mass fluxes (piecewise linear in the vertical)
            VecZeroEntries(Fv);
            VecZeroEntries(pi);
            HorizToVert2(ex, ey, rt_l, pi);
            VertFlux(ex, ey, pi, VA, scale);
            MatMult(VA, velz[ii], Mpu);
            AssembleLinear(ex, ey, VA, scale);
            KSPSolve(kspColA, Mpu, Fv);
            // strong form vertical divergence
            MatMult(V10, Fv, Dv);
            // copy the vertical contribution to the divergence into the
            // horiztonal vectors
            VertToHoriz2(ex, ey, 0, geom->nk, Dv, Ftv, false);
#endif
        }
    }
    for(kk = 0; kk < geom->nk; kk++) {
        progExner(rt_i[kk], Fth[kk], Ftv[kk], exner[kk], &exner_f, kk);
        VecCopy(exner_f, exner[kk]);
        VecDestroy(&exner_f);
    }

    // solve for the vertical velocity
    if(!rank)cout<<"\tvertical momentum rhs......"<<endl;
    vertMomRHS(velx, velz, theta_l, exner_l, Vu1);
    if(!rank)cout<<"\tvertical momentum solve...."<<endl;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ii = ey*topo->nElsX + ex;
            solveMom(dt, ex, ey, scale, BA, velz[ii], Vu1[ii]);
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
            geom->write2(rho[kk], fieldname, step, kk, true);
            sprintf(fieldname, "rhoTheta");
            geom->write2(rt[kk], fieldname, step, kk, true);
            sprintf(fieldname, "exner");
            geom->write2(exner[kk], fieldname, step, kk, true);

            VecDestroy(&wi);
        }
        sprintf(fieldname, "velocity_z");
        geom->writeVertToHoriz(velz, fieldname, step, geom->nk-1);
        sprintf(fieldname, "velVert");
        geom->writeSerial(velz, fieldname, topo->nElsX*topo->nElsX, step);
    }

    // deallocate
#ifdef ADD_VERT_EXNER_FLUX
    VecDestroy(&Mpu);
    VecDestroy(&Dv);
    VecDestroy(&pi);
    VecDestroy(&Fv);
#endif

    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Hu1[kk]    );
        VecDestroy(&rt_i[kk]   );
        VecDestroy(&exner_i[kk]);
        VecDestroy(&rho_l[kk]  );
        VecDestroy(&rt_l[kk]   );
        VecDestroy(&exner_l[kk]);
        VecDestroy(&Fth[kk]    );
        VecDestroy(&Ftv[kk]    );
    }
    for(kk = 0; kk < geom->nk + 1; kk++) {
        VecDestroy(&theta_l[kk]);
    }
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&Vu1[ii]);
        VecDestroy(&Fp1[ii]);
        VecDestroy(&Ft1[ii]);
    }
    delete[] Hu1;
    delete[] Vu1;
    delete[] Fp1;
    delete[] Ft1;
    delete[] rt_i;
    delete[] exner_i;
    delete[] rho_l;
    delete[] rt_l;
    delete[] theta_l;
    delete[] exner_l;
    delete[] Fth;
    delete[] Ftv;
    VecDestroy(&rho_z);
    VecDestroy(&bw);
    VecDestroy(&bu);
    MatDestroy(&AB);
    MatDestroy(&BA);
}

void PrimEqns_HEVI::VertToHoriz2(int ex, int ey, int ki, int kf, Vec pv, Vec* ph, bool assign) {
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

void PrimEqns_HEVI::HorizToVert2(int ex, int ey, Vec* ph, Vec pv) {
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

void PrimEqns_HEVI::init0(Vec* q, ICfunc3D* func) {
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

void PrimEqns_HEVI::init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y) {
    int ex, ey, ii, kk, mp1, mp12;
    int *inds0, *loc02;
    double scale = 1.0e8;
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

        M1->assemble(kk, scale, true);
        MatMult(UQ->M, bg, UQb);
        VecScale(UQb, scale);
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

void PrimEqns_HEVI::init2(Vec* h, ICfunc3D* func) {
    int ex, ey, ii, kk, mp1, mp12, *inds0;
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
        VecScale(WQb, scale);          // have to rescale the M2 operator as the metric terms scale
        M2->assemble(kk, scale, true); // this down to machine precision, so rescale the rhs as well
        KSPSolve(ksp2, WQb, h[kk]);
    }

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void PrimEqns_HEVI::initTheta(Vec theta, ICfunc3D* func) {
    int ex, ey, ii, mp1, mp12, *inds0;
    double scale = 1.0e8;
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

    M2->assemble(0, scale, true); // note: layer thickness must be set to 2.0 for all layers 
    MatMult(WQ->M, bg, WQb);      //       before M2 matrix is assembled to initialise theta
    VecScale(WQb, scale);
    KSPSolve(ksp2, WQb, theta);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void PrimEqns_HEVI::solveMass(double dt, int ex, int ey, double scale, Mat AB, Vec wz, Vec fv, Vec rho) {
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
            Q0[ii][ii] = Q->A[ii][ii]*(scale/det/det);

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
            Q0[ii][ii]  = Q->A[ii][ii]*(scale/det/det);
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
    //MatMatMult(DAinv, AB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &VB);
    //MatScale(VB, dt);
    //MatShift(VB, 1.0);
    MatMatMult(DAinv, AB, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Op);
    MatScale(Op, dt);
    MatShift(Op, 1.0);
    VecAYPX(fv, -dt, rho);

    KSPCreate(MPI_COMM_SELF, &kspMass);
    KSPSetOperators(kspMass, Op, Op);
    KSPSetTolerances(kspMass, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspMass, KSPGMRES);
    KSPGetPC(kspMass, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, W->nDofsJ, NULL); // TODO: check allocation
    KSPSetOptionsPrefix(kspMass, "kspMass_");
    KSPSetFromOptions(kspMass);

    //KSPSolve(kspColB, fv, rho);
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

void PrimEqns_HEVI::solveMom(double dt, int ex, int ey, double scale, Mat BA, Vec wz, Vec fv) {
    int ii, jj, kk, ei, mp1, mp12, n2, it = 0, rank;
    int rows[99], cols[99];
    double det, wb, wt, wi, gamma, eps = 1.0e+9, l2_dif, l2_old;
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

    VecAYPX(fv, -dt, wz);
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

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                Q0[ii][ii] = Q->A[ii][ii]*(scale/det/det);

                // multiply by the vertical jacobian, then scale the piecewise constant
                // basis by the vertical jacobian, so do nothing

                // interpolate the vertical velocity at the quadrature point
                wb = wt = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii%mp1][jj%topo->elOrd]*geom->edge->ejxi[ii/mp1][jj/topo->elOrd];
                    if(kk > 0)            wb += zArray[(kk-1)*n2+jj]*gamma;
                    if(kk < geom->nk - 1) wt += zArray[(kk+0)*n2+jj]*gamma;
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
        VecRestoreArray(wz, &zArray);

        AssembleLinear(ex, ey, VA, scale);
        AssembleVertLaplacian(ex, ey, VA, scale);
        if(!DBA) {
            MatMatMult(V01, BA, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DBA);
        } else {
            MatMatMult(V01, BA, MAT_REUSE_MATRIX, PETSC_DEFAULT, &DBA);
        }
        MatAXPY(VA, dt, DBA, DIFFERENT_NONZERO_PATTERN);
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
