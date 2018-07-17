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
#include "Assembly.h"
#include "PrimEqns.h"

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
#define GRAVITY 9.80616

using namespace std;

#define ADD_IE
#define ADD_GZ
#define ADD_WZ

PrimEqns::PrimEqns(Topo* _topo, Geom* _geom, double _dt) {
    int ii;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    grav = GRAVITY*(RAD_SPHERE/RAD_EARTH);
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
    EtoF = new E21mat(topo);

    // mass flux operator
    F = new Uhmat(topo, geom, node, edge);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    // potential temperature projection operator
    //T = new UtQWmat(topo, geom, node, edge);
    T = new Whmat(topo, geom, edge);

    // assemble the vertical gradient and divergence incidence matrices
    vertOps();

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_SELF, &ksp1);
    KSPSetOperators(ksp1, M1->M, M1->M);
    KSPSetTolerances(ksp1, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1, KSPGMRES);
    KSPGetPC(ksp1, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, (topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp1, "ksp1_");
    KSPSetFromOptions(ksp1);

    // initialize the 2 form linear solver
    KSPCreate(MPI_COMM_SELF, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPGetPC(ksp2, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, topo->elOrd, NULL);
    KSPSetOptionsPrefix(ksp2, "ksp2_");
    KSPSetFromOptions(ksp2);

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_b);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_t);

    Kv = new Vec[topo->nElsX];
    for(ii = 0; ii < topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->elOrd, &Kv[ii]);
    }
    Kh = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &Kh[ii]);
    }

    // initialise the single column mass matrices and solvers
    MatCreate(MPI_COMM_SELF, &VA);
    MatSetType(VA, MATSEQAIJ);
    MatSetSizes(VA, (geom->nk-1)*topo->elOrd, (geom->nk-1)*topo->elOrd, (geom->nk-1)*topo->elOrd, (geom->nk-1)*topo->elOrd);
    MatSeqAIJSetPreallocation(VA, topo->elOrd, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VB);
    MatSetType(VB, MATSEQAIJ);
    MatSetSizes(VB, geom->nk*topo->elOrd, geom->nk*topo->elOrd, geom->nk*topo->elOrd, geom->nk*topo->elOrd);
    MatSeqAIJSetPreallocation(VB, topo->elOrd, PETSC_NULL);

    KSPCreate(MPI_COMM_SELF, &kspColA);
    KSPSetOperators(kspColA, VA, VA);
    KSPSetTolerances(kspColA, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspColA, KSPGMRES);
    KSPGetPC(kspColA, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, topo->elOrd, NULL);
    KSPSetOptionsPrefix(kspColA, "kspColA_");
    KSPSetFromOptions(kspColA);
}

// laplacian viscosity, from Guba et. al. (2014) GMD
double PrimEqns::viscosity() {
    double ae = 4.0*M_PI*RAD_SPHERE*RAD_SPHERE;
    double dx = sqrt(ae/topo->n0);
    double del4 = 0.072*pow(dx,3.2);

    return -sqrt(del4);
}

double PrimEqns::viscosity_vert() {
    int ii, kk;
    double dzMin = 1.0e+6;

    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < topo->n0; ii++) {
            if(geom->thick[kk][ii] < dzMin) {
                dzMin = geom->thick[kk][ii];
            }
        }
    }

    return dzMin*dzMin/6.0;//TODO
}

PrimEqns::~PrimEqns() {
    int ii;

    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);
    VecDestroy(&theta_b);
    VecDestroy(&theta_t);

    //for(ii = 0; ii < geom->nk; ii++) {
    //    VecDestroy(&Kh[ii]);
    //}
    //delete[] Kh;
    for(ii = 0; ii < topo->nElsX; ii++) {
        VecDestroy(&Kv[ii]);
    }
    delete[] Kv;

    MatDestroy(&V01);
    MatDestroy(&V10);
    MatDestroy(&VA);
    MatDestroy(&VB);
    KSPDestroy(&kspColA);

    delete m0;
    delete M1;
    delete M2;

    delete EtoF;

    delete F;
    delete K;
    delete T;

    delete edge;
    delete node;
    delete quad;
}

/*
*/
void PrimEqns::AssembleKEVecs(Vec* velx, Vec* velz) {
    int ex, ii, jj, kk, mp1, n2, rows[99], cols[99], *inds0;
    double det, wb, wt, wi, gamma, zi;
    Mat BA;
    double** Q0 = Alloc2D(quad->n+1, quad->n+1);
    double** Wt = Alloc2D(edge->n, quad->n+1);
    double** WtQ = Alloc2D(edge->n, quad->n+1);
    double** WtQW = Alloc2D(edge->n, edge->n);
    double* WtQWflat = new double[edge->n*edge->n];
    PetscScalar *kvArray, *zqArray;
    WtQmat* WQ = new WtQmat(topo, geom, edge);
    Vec zq, zw;

    n2  = topo->elOrd;
    mp1 = quad->n + 1;

    // assemble the horiztonal operators
    for(kk = 0; kk < geom->nk; kk++) {
        K->assemble(velx[kk], kk);
        VecZeroEntries(Kh[kk]);
        MatMult(K->M, velx[kk], Kh[kk]);
    }

    // assemble the vertical operators
    MatCreate(MPI_COMM_SELF, &BA);
    MatSetType(BA, MATSEQAIJ);
    MatSetSizes(BA, (geom->nk+0)*n2, (geom->nk-1)*n2, (geom->nk+0)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(BA, 2*n2, PETSC_NULL);

    Tran_IP(quad->n+1, edge->n, edge->ejxi, Wt);

    for(ex = 0; ex < topo->nElsX; ex++) {
        MatZeroEntries(BA);
        VecGetArray(velz[ex], &kvArray);

        // Assemble the matrices
        for(kk = 0; kk < geom->nk; kk++) {
            for(ii = 0; ii < mp1; ii++) {
                det = geom->det[ex][ii];
                Q0[ii][ii] = det*quad->w[ii]/det/det;

                // multiply by the vertical jacobian, then scale the piecewise constant 
                // basis by the vertical jacobian, so do nothing 

                // interpolate the vertical velocity at the quadrature point
                wb = wt = 0.0;
                for(jj = 0; jj < n2; jj++) {
                    gamma = geom->edge->ejxi[ii][jj];
                    if(kk > 0)            wb += kvArray[(kk-1)*n2+jj]*gamma;
                    if(kk < geom->nk - 1) wt += kvArray[(kk+0)*n2+jj]*gamma;
                }
                wi = 1.0*(wb + wt); // quadrature weights are both 1.0

                Q0[ii][ii] *= wi;
            }

            Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, Q0, WtQ);
            Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);
            Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

            for(ii = 0; ii < edge->n; ii++) {
                rows[ii] = ii + kk*edge->n;
            }

            // assemble the first basis function
            if(kk > 0) {
                for(ii = 0; ii < edge->n; ii++) {
                    cols[ii] = ii + (kk-1)*edge->n;
                }
                MatSetValues(BA, edge->n, rows, edge->n, cols, WtQWflat, ADD_VALUES);
            }

            // assemble the second basis function
            if(kk < geom->nk - 1) {
                for(ii = 0; ii < edge->n; ii++) {
                    cols[ii] = ii + (kk+0)*edge->n;
                }
                MatSetValues(BA, edge->n, rows, edge->n, cols, WtQWflat, ADD_VALUES);
            }
        }
        MatAssemblyBegin(BA, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(BA, MAT_FINAL_ASSEMBLY);
        VecRestoreArray(velz[ex], &kvArray);

        VecZeroEntries(Kv[ex]);
        MatMult(BA, velz[ex], Kv[ex]);
    }

#ifdef ADD_WZ
    // add the vertical contribution to the horiztonal vector
    for(ex = 0; ex < topo->nElsX; ex++) {
        VertToHoriz2(ex, 0, geom->nk, Kv[ex], Kh);
    }
#endif

    // update the vertical vector with the horiztonal vector
    for(ex = 0; ex < topo->nElsX; ex++) {
        VecZeroEntries(Kv[ex]);
        HorizToVert2(ex, Kh, Kv[ex]);
    }

#ifdef ADD_GZ
    // add in the vertical gravity vector
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &zq);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &zw);
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(zq, &zqArray);
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0(ex);
            for(ii = 0; ii < mp1; ii++) {
                // quadrature weights are both 1.0
                zi = 1.0*(geom->levs[kk+0][inds0[ii]] + geom->levs[kk+1][inds0[ii]])*GRAVITY;
                zqArray[inds0[ii]] = zi;
            }
        }
        VecRestoreArray(zq, &zqArray);
        MatMult(WQ->M, zq, zw);
    }

    for(ex = 0; ex < topo->nElsX; ex++) {
        HorizToVert2(ex, Kh, Kv[ex]);
    }

    VecDestroy(&zq);
    VecDestroy(&zw);
#endif

    MatDestroy(&BA);
    Free2D(quad->n+1, Q0);
    Free2D(edge->n, Wt);
    Free2D(edge->n, WtQ);
    Free2D(edge->n, WtQW);
    delete[] WtQWflat;
    delete WQ;
}

/*
compute the right hand side for the momentum equation for a given level
note that the vertical velocity, uv, is stored as a different vector for 
each element
*/
void PrimEqns::horizMomRHS(Vec uh, Vec* uv, Vec* theta, Vec exner, int lev, Vec *Fu) {
    Vec Ku, Mh, d2u, d4u, theta_k, dExner, dp;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, Fu);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dp);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &Ku);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &Mh);

    VecZeroEntries(*Fu);
    MatMult(EtoF->E12, Kh[lev], *Fu);

    // add the thermodynamic term (theta is in the same space as the vertical velocity)
    // project theta onto 1 forms
#ifdef ADD_IE
    VecZeroEntries(theta_k);
    VecAXPY(theta_k, 1.0, theta[lev+0]); // quadrature weights
    VecAXPY(theta_k, 1.0, theta[lev+1]); // are both 1.0

    grad(exner, &dExner, lev);
    F->assemble(theta_k, lev, false);
    MatMult(F->M, dExner, dp);
    VecAXPY(*Fu, 1.0, dp);
    VecDestroy(&dExner);
#endif

    // add in the biharmonic voscosity
    // TODO: this is causing problems at the moment...
    if(do_visc) {
        laplacian(uh, &d2u, lev);
        laplacian(d2u, &d4u, lev);
        VecAXPY(*Fu, 1.0, d4u);
    }

    VecDestroy(&Ku);
    VecDestroy(&Mh);
    VecDestroy(&dp);
    VecDestroy(&theta_k);
    if(do_visc) {
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
}

void PrimEqns::vertMomRHS(Vec* ui, Vec* wi, Vec* theta, Vec* exner, Vec* fw) {
    int ex, n2;
    Vec exner_v, de1, de2, de3, dp;

    n2 = topo->elOrd;

    // vertical velocity is computer per element, so matrices are local to this processor
    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &de1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &de2);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &de3);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &dp);
    VecCreateSeq(MPI_COMM_SELF, geom->nk*n2, &exner_v);

    for(ex = 0; ex < topo->nElsX; ex++) {
        // add in the kinetic energy gradient
        MatMult(V01, Kv[ex], fw[ex]);

        // add in the pressure gradient
#ifdef ADD_IE
/* removing the vertical pressure gradient term (and the vertical temperature transport)
   in order to remove vertical sound waves in an energetically consistent way
        VecZeroEntries(exner_v);
        HorizToVert2(ex, exner, exner_v);

        VecZeroEntries(de1);
        VecZeroEntries(de2);
        VecZeroEntries(de3);
        AssembleConst(ex, VB);
        MatMult(VB, exner_v, de1);
        MatMult(V01, de1, de2);
        AssembleLinear(ex, VA);//TODO: skip this and just solve with B(theta)?? on LHS
        KSPSolve(kspColA, de2, de3);

        // interpolate the potential temperature onto the piecewise linear
        // vertical mass matrix and multiply by the weak form vertical gradient of
        // the exner pressure
        AssembleLinearWithTheta(ex, theta, VA);
        MatMult(VA, de3, dp);
        VecAXPY(fw[ex], 1.0, dp);
*/
#endif
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
void PrimEqns::massRHS(Vec* uh, Vec* uv, Vec* pi, Vec* Fh, Vec* Fv, Vec* Fp, bool do_vert) {
    int kk, ex, n2;
    Vec Mpu, Dv, pu, Dh;

    n2 = topo->elOrd;

#ifdef ADD_WZ
    if(do_vert) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &Mpu);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &Dv);

        // compute the vertical mass fluxes (piecewise linear in the vertical)
        for(ex = 0; ex < topo->nElsX; ex++) {
            VecZeroEntries(Fv[ex]);

            VertFlux(ex, pi, VA);
            MatMult(VA, uv[ex], Mpu);
            AssembleLinear(ex, VA);
            KSPSolve(kspColA, Mpu, Fv[ex]);
            // strong form vertical divergence
            MatMult(V10, Fv[ex], Dv);

            // copy the vertical contribution to the divergence into the
            // horiztonal vectors
            VertToHoriz2(ex, 0, geom->nk, Dv, Fp);
        }
        VecDestroy(&Mpu);
        VecDestroy(&Dv);
    }
#endif

    // compute the horiztonal mass fluxes
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &pu);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &Dh);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(Fh[kk]);

        // add the horiztonal fluxes
        F->assemble(pi[kk], kk, true);
        M1->assemble(kk, true);
        MatMult(F->M, uh[kk], pu);
        KSPSolve(ksp1, pu, Fh[kk]);
        MatMult(EtoF->E21, Fh[kk], Dh);
        VecAXPY(Fp[kk], 1.0, Dh);
    }

    VecDestroy(&pu);
    VecDestroy(&Dh);
}

/*
Assemble the boundary condition vector for rho(t) X theta(0)
*/
#if 0
void PrimEqns::thetaBCVec(int ex, Mat A, Vec* rho, Vec* bTheta) {
    int* inds2 = topo->elInds2(ex);
    int ii, mp1, n2;
    double det, rk;
    int inds2k[99];
    double** Q0 = Alloc2D(quad->n+1, quad->n+1);
    double** Wt = Alloc2D(edge->n, quad->n+1);
    double** WtQ = Alloc2D(edge->n, quad->n+1);
    double** WtQW = Alloc2D(edge->n, edge->n);
    double* WtQWflat = new double[edge->n*edge->n];
    PetscScalar *rArray, *vArray, *hArray;
    Vec theta_o;

    mp1 = quad->n + 1;
    n2  = topo->elOrd;

    Tran_IP(quad->n+1, edge->n, edge->ejxi, Wt);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &theta_o);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, bTheta);
    MatZeroEntries(A);

    // bottom boundary
    VecGetArray(rho[0], &rArray);
    for(ii = 0; ii < mp1; ii++) {
        det = geom->det[ex][ii];
        Q0[ii][ii] = det*quad->w[ii]/det/det;

        // multuply by the vertical determinant to integrate, then
        // divide piecewise constant density by the vertical determinant,
        // so these cancel
        geom->interp2(ex, ii, rArray, &rk);
        Q0[ii][ii] *= rk;
    }
    VecRestoreArray(rho[0], &rArray);

    Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, Q0, WtQ);
    Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);
    Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

    for(ii = 0; ii < edge->n; ii++) {
        inds2k[ii] = ii + 0*edge->n;
    }
    MatSetValues(A, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);

    // top boundary
    VecGetArray(rho[geom->nk-1], &rArray);
    for(ii = 0; ii < mp1; ii++) {
        det = geom->det[ex][ii];
        Q0[ii][ii] = det*quad->w[ii]/det/det;

        // multuply by the vertical determinant to integrate, then
        // divide piecewise constant density by the vertical determinant,
        // so these cancel
        geom->interp2(ex, ii, rArray, &rk);
        Q0[ii][ii] *= rk;
    }
    VecRestoreArray(rho[geom->nk-1], &rArray);

    Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, Q0, WtQ);
    Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);
    Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

    for(ii = 0; ii < edge->n; ii++) {
        inds2k[ii] = ii + (geom->nk-2)*edge->n;
    }
    MatSetValues(A, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);

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

    Free2D(quad->n+1, Q0);
    Free2D(edge->n, Wt);
    Free2D(edge->n, WtQ);
    Free2D(edge->n, WtQW);
    delete[] WtQWflat;
    VecDestroy(&theta_o);
}
#endif
void PrimEqns::thetaBCVec(int ex, Mat A, Vec* rho, Vec* bTheta) {
    int* inds2 = topo->elInds2(ex);
int* inds0 = topo->elInds0(ex);
    int ii, jj, mp1, n2;
    double det, rk;
    double** Q0 = Alloc2D(quad->n+1, quad->n+1);
    double** Wt = Alloc2D(edge->n, quad->n+1);
    double** WtQ = Alloc2D(edge->n, quad->n+1);
    double** WtQW = Alloc2D(edge->n, edge->n);
    PetscScalar *rArray, *vArray, *hArray;

    mp1 = quad->n + 1;
    n2  = topo->elOrd;

    Tran_IP(quad->n+1, edge->n, edge->ejxi, Wt);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, bTheta);
    VecZeroEntries(*bTheta);
    MatZeroEntries(A);

    // bottom boundary
    VecGetArray(rho[0], &rArray);
    for(ii = 0; ii < mp1; ii++) {
        det = geom->det[ex][ii];
        Q0[ii][ii] = det*quad->w[ii]/det/det;

        // multuply by the vertical determinant to integrate, then
        // divide piecewise constant density by the vertical determinant,
        // so these cancel
        geom->interp2(ex, ii, rArray, &rk);
        Q0[ii][ii] *= rk;
Q0[ii][ii] /= geom->thick[0][inds0[ii]];
    }
    VecRestoreArray(rho[0], &rArray);

    Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, Q0, WtQ);
    Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);

    VecGetArray(*bTheta, &vArray);
    VecGetArray(theta_b, &hArray);
    for(ii = 0; ii < n2; ii++) {
        vArray[ii+0*n2] = 0.0;
        for(jj = 0; jj < n2; jj++) {
            vArray[ii+0*n2] += WtQW[ii][jj]*hArray[inds2[jj]];
        }
    }
    VecRestoreArray(*bTheta, &vArray);
    VecRestoreArray(theta_b, &hArray);

    // top boundary
    VecGetArray(rho[geom->nk-1], &rArray);
    for(ii = 0; ii < mp1; ii++) {
        det = geom->det[ex][ii];
        Q0[ii][ii] = det*quad->w[ii]/det/det;

        // multuply by the vertical determinant to integrate, then
        // divide piecewise constant density by the vertical determinant,
        // so these cancel
        geom->interp2(ex, ii, rArray, &rk);
        Q0[ii][ii] *= rk;
Q0[ii][ii] /= geom->thick[geom->nk-1][inds0[ii]];
    }
    VecRestoreArray(rho[geom->nk-1], &rArray);

    Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, Q0, WtQ);
    Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);

    VecGetArray(*bTheta, &vArray);
    VecGetArray(theta_t, &hArray);
    for(ii = 0; ii < n2; ii++) {
        vArray[ii+(geom->nk-2)*n2] = 0.0;
        for(jj = 0; jj < n2; jj++) {
            vArray[ii+(geom->nk-2)*n2] += WtQW[ii][jj]*hArray[inds2[jj]];
        }
    }
    VecRestoreArray(*bTheta, &vArray);
    VecRestoreArray(theta_t, &hArray);

    Free2D(quad->n+1, Q0);
    Free2D(edge->n, Wt);
    Free2D(edge->n, WtQ);
    Free2D(edge->n, WtQW);
}

/*
diagnose theta from rho X theta (with boundary condition)
note: rho, rhoTheta and theta are all LOCAL vectors
*/
void PrimEqns::diagTheta(Vec* rho, Vec* rt, Vec* theta) {
    int ex, n2, kk;
    Vec rtv, frt, theta_v, bcs;
    Mat A, AB;

    n2 = topo->elOrd;

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

    for(ex = 0; ex < topo->nElsX; ex++) {
        // construct horiztonal rho theta field
        VecZeroEntries(rtv);
        HorizToVert2(ex, rt, rtv);
        AssembleLinCon(ex, AB);
        MatMult(AB, rtv, frt);

        // assemble in the bcs // TODO: BC application error!
        thetaBCVec(ex, A, rho, &bcs);
        VecAXPY(frt, -1.0, bcs);
        VecDestroy(&bcs);

        AssembleLinearWithRho(ex, rho, VA);
        KSPSolve(kspColA, frt, theta_v);
        VertToHoriz2(ex, 1, geom->nk, theta_v, theta);
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
    Vec rhs;
    PC pc;
    KSP kspE;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &rhs);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, exner_f);

    KSPCreate(MPI_COMM_SELF, &kspE);
    KSPSetOperators(kspE, T->M, T->M);
    KSPSetTolerances(kspE, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(kspE, KSPGMRES);
    KSPGetPC(kspE, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, topo->elOrd, NULL);
    KSPSetOptionsPrefix(kspE, "exner_");
    KSPSetFromOptions(kspE);

    T->assemble(rt_i, lev, true);
    MatMult(T->M, exner_i, rhs);

    T->assemble(rt_f, lev, true);
    KSPSolve(kspE, rhs, *exner_f);

    VecDestroy(&rhs);
    KSPDestroy(&kspE);
}

/*
Take the weak form gradient of a 2 form scalar field as a 1 form vector field
*/
void PrimEqns::grad(Vec phi, Vec* u, int lev) {
    Vec Mphi, dMphi;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, u);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &Mphi);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dMphi);

    M1->assemble(lev, false); //TODO: vertical scaling of this operator causes problems??
    M2->assemble(lev, false);

    MatMult(M2->M, phi, Mphi);
    MatMult(EtoF->E12, Mphi, dMphi);
    KSPSolve(ksp1, dMphi, *u);

    VecDestroy(&Mphi);
    VecDestroy(&dMphi);
}

void PrimEqns::laplacian(Vec ui, Vec* ddu, int lev) {
    Vec Du;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, ddu);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &Du);

    /*** divergent component ***/
    // div (strong form)
    MatMult(EtoF->E21, ui, Du);

    // grad (weak form)
    grad(Du, ddu, lev);

    // add rotational and divergent components
    VecScale(*ddu, del2);

    VecDestroy(&Du);
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
    
    n2 = topo->elOrd;

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
void PrimEqns::AssembleConst(int ex, Mat B) {
    int ii, kk, mp1;
    int *inds0;
    double det;
    int inds2k[99];
    double** Q0 = Alloc2D(quad->n+1, quad->n+1);
    double** Wt = Alloc2D(edge->n, quad->n+1);
    double** WtQ = Alloc2D(edge->n, quad->n+1);
    double** WtQW = Alloc2D(edge->n, edge->n);
    double* WtQWflat = new double[edge->n*edge->n];

    inds0 = topo->elInds0(ex);
    mp1   = quad->n + 1;

    MatZeroEntries(B);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            Q0[ii][ii] = det*quad->w[ii]/det/det;
            // for constant field we multiply by the vertical jacobian determinant when integrating, 
            // then divide by the vertical jacobian for both the trial and the test functions
            // vertical determinant is dz/2
            Q0[ii][ii] *= 2.0/geom->thick[kk][inds0[ii]];
        }

        // assemble the piecewise constant mass matrix for level k
        Tran_IP(quad->n+1, edge->n, edge->ejxi, Wt);
        Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, Q0, WtQ);
        Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);
        Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

        for(ii = 0; ii < edge->n; ii++) {
            inds2k[ii] = ii + kk*edge->n;
        }
        MatSetValues(B, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);
    }
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

    Free2D(quad->n+1, Q0);
    Free2D(edge->n, Wt);
    Free2D(edge->n, WtQ);
    Free2D(edge->n, WtQW);
    delete[] WtQWflat;
}

/*
Assemble a 3D mass matrix as a tensor product of 2 forms in the 
horizotnal and linear basis functions in the vertical
*/
void PrimEqns::AssembleLinear(int ex, Mat A) {
    int ii, kk, mp1;
    int *inds0;
    double det;
    int inds2k[99];
    double** Q0 = Alloc2D(quad->n+1, quad->n+1);
    double** Wt = Alloc2D(edge->n, quad->n+1);
    double** WtQ = Alloc2D(edge->n, quad->n+1);
    double** WtQW = Alloc2D(edge->n, edge->n);
    double* WtQWflat = new double[edge->n*edge->n];

    inds0 = topo->elInds0(ex);
    mp1   = quad->n + 1;

    MatZeroEntries(A);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            Q0[ii][ii]  = det*quad->w[ii]/det/det;
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            Q0[ii][ii] *= geom->thick[kk][inds0[ii]]/2.0;
        }

        Tran_IP(quad->n+1, edge->n, edge->ejxi, Wt);
        Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, Q0, WtQ);
        Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);
        Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < edge->n; ii++) {
                inds2k[ii] = ii + (kk-1)*edge->n;
            }
            MatSetValues(A, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < edge->n; ii++) {
                inds2k[ii] = ii + (kk+0)*edge->n;
            }
            MatSetValues(A, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    Free2D(quad->n+1, Q0);
    Free2D(edge->n, Wt);
    Free2D(edge->n, WtQ);
    Free2D(edge->n, WtQW);
    delete[] WtQWflat;
}

void PrimEqns::AssembleLinCon(int ex, Mat AB) {
    int ii, kk, mp1;
    double det;
    int rows[99], cols[99];
    double** Q0 = Alloc2D(quad->n+1, quad->n+1);
    double** Wt = Alloc2D(edge->n, quad->n+1);
    double** WtQ = Alloc2D(edge->n, quad->n+1);
    double** WtQW = Alloc2D(edge->n, edge->n);
    double* WtQWflat = new double[edge->n*edge->n];

    mp1 = quad->n + 1;

    MatZeroEntries(AB);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            Q0[ii][ii] = det*quad->w[ii]/det/det;

            // multiply by the vertical jacobian, then scale the piecewise constant 
            // basis by the vertical jacobian, so do nothing 
        }

        Tran_IP(quad->n+1, edge->n, edge->ejxi, Wt);
        Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, Q0, WtQ);
        Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);
        Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

        for(ii = 0; ii < edge->n; ii++) {
            cols[ii] = ii + kk*edge->n;
        }

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < edge->n; ii++) {
                rows[ii] = ii + (kk-1)*edge->n;
            }
            MatSetValues(AB, edge->n, rows, edge->n, cols, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < edge->n; ii++) {
                rows[ii] = ii + (kk+0)*edge->n;
            }
            MatSetValues(AB, edge->n, rows, edge->n, cols, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(AB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AB, MAT_FINAL_ASSEMBLY);

    Free2D(quad->n+1, Q0);
    Free2D(edge->n, Wt);
    Free2D(edge->n, WtQ);
    Free2D(edge->n, WtQW);
    delete[] WtQWflat;
}

void PrimEqns::AssembleLinearWithRho(int ex, Vec* rho, Mat A) {
    int ii, kk, mp1;
    double det, rk;
    int inds2k[99];
    double** Q0 = Alloc2D(quad->n+1, quad->n+1);
    double** Wt = Alloc2D(edge->n, quad->n+1);
    double** WtQ = Alloc2D(edge->n, quad->n+1);
    double** WtQW = Alloc2D(edge->n, edge->n);
    double* WtQWflat = new double[edge->n*edge->n];
    PetscScalar *rArray;

    mp1  = quad->n + 1;

    Tran_IP(quad->n+1, edge->n, edge->ejxi, Wt);
    MatZeroEntries(A);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        VecGetArray(rho[kk], &rArray);
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            Q0[ii][ii] = det*quad->w[ii]/det/det;

            // multuply by the vertical determinant to integrate, then
            // divide piecewise constant density by the vertical determinant,
            // so these cancel
            geom->interp2(ex, ii, rArray, &rk);
            Q0[ii][ii] *= rk;
        }
        VecRestoreArray(rho[kk], &rArray);

        Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, Q0, WtQ);
        Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);
        Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

        // assemble the first basis function
        if(kk > 0) {
            for(ii = 0; ii < edge->n; ii++) {
                inds2k[ii] = ii + (kk-1)*edge->n;
            }
            MatSetValues(A, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < edge->n; ii++) {
                inds2k[ii] = ii + (kk+0)*edge->n;
            }
            MatSetValues(A, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    Free2D(quad->n+1, Q0);
    Free2D(edge->n, Wt);
    Free2D(edge->n, WtQ);
    Free2D(edge->n, WtQW);
    delete[] WtQWflat;
}

void PrimEqns::AssembleLinearWithTheta(int ex, Vec* theta, Mat A) {
    int ii, kk, mp1;
    int *inds0;
    double det, t1, t2;
    int inds2k[99];
    double** QB = Alloc2D(quad->n+1, quad->n+1);
    double** QT = Alloc2D(quad->n+1, quad->n+1);
    double** Wt = Alloc2D(edge->n, quad->n+1);
    double** WtQ = Alloc2D(edge->n, quad->n+1);
    double** WtQW = Alloc2D(edge->n, edge->n);
    double* WtQWflat = new double[edge->n*edge->n];
    PetscScalar *t1Array, *t2Array;

    inds0 = topo->elInds0(ex);
    mp1   = quad->n + 1;

    Tran_IP(quad->n+1, edge->n, edge->ejxi, Wt);
    MatZeroEntries(A);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(theta[kk+0], &t1Array);
        VecGetArray(theta[kk+1], &t2Array);
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            QB[ii][ii]  = det*quad->w[ii]/det/det;
            // for linear field we multiply by the vertical jacobian determinant when integrating, 
            // and do no other trasformations for the basis functions
            QB[ii][ii] *= geom->thick[kk][inds0[ii]]/2.0;
            QT[ii][ii]  = QB[ii][ii];

            geom->interp2(ex, ii, t1Array, &t1);
            geom->interp2(ex, ii, t2Array, &t2);

            QB[ii][ii] *= t1;
            QT[ii][ii] *= t2;
        }
        VecRestoreArray(theta[kk+0], &t1Array);
        VecRestoreArray(theta[kk+1], &t2Array);

        // assemble the first basis function
        if(kk > 0) {
            Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, QB, WtQ);
            Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);
            Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

            for(ii = 0; ii < edge->n; ii++) {
                inds2k[ii] = ii + (kk-1)*edge->n;
            }
            MatSetValues(A, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function
        if(kk < geom->nk - 1) {
            Mult_IP(edge->n, quad->n+1, quad->n+1, Wt, QT, WtQ);
            Mult_IP(edge->n, edge->n, quad->n+1, WtQ, edge->ejxi, WtQW);
            Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

            for(ii = 0; ii < edge->n; ii++) {
                inds2k[ii] = ii + (kk+0)*edge->n;
            }
            MatSetValues(A, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    Free2D(quad->n+1, QB);
    Free2D(quad->n+1, QT);
    Free2D(edge->n, Wt);
    Free2D(edge->n, WtQ);
    Free2D(edge->n, WtQW);
    delete[] WtQWflat;
}

/*
derive the vertical mass flux
TODO: only need a single piecewise constant field, may be either rho or rho X theta
*/
void PrimEqns::VertFlux(int ex, Vec* pi, Mat Mp) {
    int ii, kk, mp1;
    double det, rho;
    int inds2k[99];
    double** Q0 = Alloc2D(quad->n+1, quad->n+1);
    double** Wt = Alloc2D(edge->n, quad->n+1);
    double** WtQ = Alloc2D(edge->n, quad->n+1);
    double** WtQW = Alloc2D(edge->n, edge->n);
    double* WtQWflat = new double[edge->n*edge->n];
    PetscScalar *pArray;

    mp1  = quad->n + 1;

    // build the 2D mass matrix
    Tran_IP(quad->n+1, edge->n, edge->ejxi, Wt);
    MatZeroEntries(Mp);

    // assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        VecGetArray(pi[kk], &pArray);
        for(ii = 0; ii < mp1; ii++) {
            det = geom->det[ex][ii];
            Q0[ii][ii] = det*node->q->w[ii]/det/det;

            geom->interp2(ex, ii, pArray, &rho);
            Q0[ii][ii] *= rho;

            // multiply by the vertical determinant for the vertical integral,
            // then divide by the vertical determinant to rescale the piecewise
            // constant density, so do nothing.
        }
        VecRestoreArray(pi[kk], &pArray);

        // assemble the piecewise constant mass matrix for level k
        Mult_IP(edge->n, mp1, quad->n+1, Wt, Q0, WtQ);
        Mult_IP(edge->n, edge->n, mp1, WtQ, edge->ejxi, WtQW);
        Flat2D_IP(edge->n, edge->n, WtQW, WtQWflat);

        // assemble the first basis function (exclude bottom boundary)
        if(kk > 0) {
            for(ii = 0; ii < edge->n; ii++) {
                inds2k[ii] = ii + (kk-1)*edge->n;
            }
            MatSetValues(Mp, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);
        }

        // assemble the second basis function (exclude top boundary)
        if(kk < geom->nk - 1) {
            for(ii = 0; ii < edge->n; ii++) {
                inds2k[ii] = ii + (kk+0)*edge->n;
            }
            MatSetValues(Mp, edge->n, inds2k, edge->n, inds2k, WtQWflat, ADD_VALUES);
        }
    }
    MatAssemblyBegin(Mp, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Mp, MAT_FINAL_ASSEMBLY);

    Free2D(mp1, Q0);
    Free2D(edge->n, Wt);
    Free2D(edge->n, WtQ);
    Free2D(edge->n, WtQW);
    delete[] WtQWflat;
}

void PrimEqns::AssembleVertLaplacian(int ex, Mat A) {
    int n2 = topo->elOrd;
    Mat B, L, BD;

    MatCreate(MPI_COMM_SELF, &B);
    MatSetType(B, MATSEQAIJ);
    MatSetSizes(B, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(B, n2, PETSC_NULL);

    AssembleConst(ex, B);

    // construct the laplacian mixing operator
    MatMatMult(B, V10, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &BD);
    MatMatMult(V01, BD, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L);

    // assemble the piecewise linear mass matrix (with gravity)
    MatAXPY(A, -vert_visc, L, DIFFERENT_NONZERO_PATTERN);//TODO: check the sign on the viscosity

    MatDestroy(&B);
    MatDestroy(&BD);
    MatDestroy(&L);
}

void PrimEqns::SolveEuler(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int ii, kk, ex;
    char fieldname[100];
    Vec *Hu1, *Vu1, *Fp1, *Ft1, bu, bw, exner_f;
    Vec *rt_i, *exner_i, *Fh, *Fv, *Gh, *Gv, *theta;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd, &bw);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &bu);
    Hu1     = new Vec[geom->nk];
    rt_i    = new Vec[geom->nk];
    exner_i = new Vec[geom->nk];
    Fh      = new Vec[geom->nk];
    Gh      = new Vec[geom->nk];
    Fv      = new Vec[topo->nElsX];
    Gv      = new Vec[topo->nElsX];
    theta   = new Vec[geom->nk+1];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &rt_i[kk]   );
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &exner_i[kk]);
        // temporary vectors for use in exner pressure prognosis
        VecCopy(rt[kk]   , rt_i[kk]   );
        VecCopy(exner[kk], exner_i[kk]);
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &Fh[kk]);
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &Gh[kk]);
    }
    // create vectors for the potential temperature at the internal layer interfaces
    for(kk = 1; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta[kk]);
    }
    theta[0] = theta_b;
    theta[geom->nk] = theta_t;

    Vu1 = new Vec[topo->nElsX];
    for(ii = 0; ii < topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd, &Vu1[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd, &Fv[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd, &Gv[ii]);
        VecZeroEntries(Vu1[ii]);
    }

    // continuity and temperature equation rhs vectors
    Fp1 = new Vec[geom->nk];
    Ft1 = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &Fp1[kk]);
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &Ft1[kk]);
        VecZeroEntries(Fp1[kk]);
        VecZeroEntries(Ft1[kk]);
    }

    // assemble the vertical and horiztonal kinetic energy vectors
    AssembleKEVecs(velx, velz);

    // construct the right hand side terms for the first substep
    // note: do horiztonal rhs first as this assembles the kinetic energy
    // operator for use in the vertical rhs
#ifdef ADD_IE
    cout<<"\tdiagnosing theta..........."<<endl;
    diagTheta(rho, rt, theta);
#endif
    cout<<"\thorizontal momentum rhs...."<<endl;
    for(kk = 0; kk < geom->nk; kk++) {
        horizMomRHS(velx[kk], velz, theta, exner[kk], kk, &Hu1[kk]);
    }
    cout<<"\tvertical momentum rhs......"<<endl;
    vertMomRHS(velx, velz, theta, exner, Vu1);
    cout<<"\tcontinuity eqn rhs........."<<endl;
    massRHS(velx, velz, rho, Fh, Fv, Fp1, true);
#ifdef ADD_IE
    cout<<"\tenergy eqn rhs............."<<endl;
    massRHS(velx, velz, rt,  Gh, Gv, Ft1, false);
#endif

    // solve for the half step values
    for(kk = 0; kk < geom->nk; kk++) {
        // horizontal momentum
        M1->assemble(kk, true);
        VecZeroEntries(bu);
        MatMult(M1->M, velx[kk], bu);
        VecAXPY(bu, -dt, Hu1[kk]);
        cout<<"\thorizontal momentum solve.."<<endl;
        VecZeroEntries(velx[kk]);
        KSPSolve(ksp1, bu, velx[kk]);

        // density
        VecAXPY(rho[kk], -dt, Fp1[kk]);

#ifdef ADD_IE
        // potential temperature
        VecAXPY(rt[kk], -dt, Ft1[kk]);

        // exner pressure
        cout<<"\texner pressure solve......."<<endl;
        progExner(rt_i[kk], rt[kk], exner[kk], &exner_f, kk);
        VecCopy(exner_f, exner[kk]);
        VecDestroy(&exner_f);
#endif
    }

    // solve for the vertical velocity
    cout<<"\tvertical momentum solve...."<<endl;
    for(ex = 0; ex < topo->nElsX; ex++) {
        VecZeroEntries(bw);
        AssembleLinear(ex, VA);
        MatMult(VA, velz[ex], bw);
        VecAXPY(bw, -dt, Vu1[ex]);
        AssembleVertLaplacian(ex, VA);
        KSPSolve(kspColA, bw, velz[ex]);
    }

    // write output
    if(save) {
        step++;
        sprintf(fieldname, "velocity_h");
        geom->write1(velx, fieldname, step);
        sprintf(fieldname, "density");
        geom->write2(rho, fieldname, step, true);
        sprintf(fieldname, "rhoTheta");
        geom->write2(rt, fieldname, step, true);
        sprintf(fieldname, "exner");
        geom->write2(exner, fieldname, step, true);
        sprintf(fieldname, "velocity_z");
        geom->writeSerial(velz, fieldname, step, geom->nk-1);
diagTheta(rho, rt, theta);
        sprintf(fieldname, "theta");
        geom->write2(theta, fieldname, step, false);
    }

    // deallocate
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&Hu1[kk]);
        VecDestroy(&Fp1[kk]);
        VecDestroy(&Ft1[kk]);
        VecDestroy(&rt_i[kk]);
        VecDestroy(&exner_i[kk]);
        VecDestroy(&Fh[kk]);
        VecDestroy(&Gh[kk]);
    }
    for(kk = 1; kk < geom->nk; kk++) {
        VecDestroy(&theta[kk]);
    }
    for(ii = 0; ii < topo->nElsX; ii++) {
        VecDestroy(&Vu1[ii]);
        VecDestroy(&Fv[ii]);
        VecDestroy(&Gv[ii]);
    }
    delete[] Hu1;
    delete[] Vu1;
    delete[] Fp1;
    delete[] Ft1;
    delete[] rt_i;
    delete[] exner_i;
    delete[] Fh;
    delete[] Gh;
    delete[] Fv;
    delete[] Gv;
    delete[] theta;
    VecDestroy(&bw);
    VecDestroy(&bu);
}

void PrimEqns::VertToHoriz2(int ex, int ki, int kf, Vec pv, Vec* ph) {
    int ii, kk, n2;
    int* inds2 = topo->elInds2(ex);
    PetscScalar *hArray, *vArray;

    n2 = topo->elOrd;

    VecGetArray(pv, &vArray);
    for(kk = ki; kk < kf; kk++) {
        VecGetArray(ph[kk], &hArray);
        for(ii = 0; ii < n2; ii++) {
            //hArray[inds2[ii]] += vArray[kk*n2+ii];
            hArray[inds2[ii]] += vArray[(kk-ki)*n2+ii];
        }
        VecRestoreArray(ph[kk], &hArray);
    }
    VecRestoreArray(pv, &vArray);
}

void PrimEqns::HorizToVert2(int ex, Vec* ph, Vec pv) {
    int ii, kk, n2;
    int* inds2 = topo->elInds2(ex);
    PetscScalar *hArray, *vArray;

    n2 = topo->elOrd;

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
    int ex, ii, kk, mp1;
    int* inds0;
    PtQmat* PQ = new PtQmat(topo, geom, node);
    PetscScalar *bArray;
    Vec bg, PQb;

    mp1 = quad->n + 1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bg);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &PQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bg);
        VecGetArray(bg, &bArray);

        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0(ex);
            for(ii = 0; ii < mp1; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]], kk);
            }
        }
        VecRestoreArray(bg, &bArray);

        m0->assemble(kk, true);
        MatMult(PQ->M, bg, PQb);
        VecPointwiseDivide(q[kk], PQb, m0->vg);
    }

    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void PrimEqns::init1(Vec *u, ICfunc3D* func) {
    int ex, ii, kk, mp1;
    int *inds0;
    UtQmat* UQ = new UtQmat(topo, geom, node, edge);
    PetscScalar *bArray;
    Vec bg, UQb;

    mp1 = quad->n + 1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bg);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &UQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bg);
        VecGetArray(bg, &bArray);
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0(ex);
            for(ii = 0; ii < mp1; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]], kk);
            }
        }
        VecRestoreArray(bg, &bArray);

        M1->assemble(kk, true);
        MatMult(UQ->M, bg, UQb);
        KSPSolve(ksp1, UQb, u[kk]);
    }

    VecDestroy(&bg);
    VecDestroy(&UQb);
    delete UQ;
}

void PrimEqns::init2(Vec* h, ICfunc3D* func) {
    int ex, ii, kk, mp1, *inds0;
    PetscScalar *bArray;
    Vec bg, WQb;
    WtQmat* WQ = new WtQmat(topo, geom, edge);

    mp1 = quad->n + 1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bg);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &WQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bg);
        VecGetArray(bg, &bArray);
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0(ex);
            for(ii = 0; ii < mp1; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]], kk);
            }
        }
        VecRestoreArray(bg, &bArray);

        MatMult(WQ->M, bg, WQb);
        M2->assemble(kk, true);
        KSPSolve(ksp2, WQb, h[kk]);
    }

    delete WQ;
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void PrimEqns::initTheta(Vec theta, ICfunc3D* func) {
    int ex, ii, mp1, *inds0;
    PetscScalar *bArray;
    Vec bg, WQb;
    WtQmat* WQ = new WtQmat(topo, geom, edge);

    mp1 = quad->n + 1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bg);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &WQb);
    VecZeroEntries(bg);

    VecGetArray(bg, &bArray);
    for(ex = 0; ex < topo->nElsX; ex++) {
        inds0 = topo->elInds0(ex);
        for(ii = 0; ii < mp1; ii++) {
            bArray[inds0[ii]] = func(geom->x[inds0[ii]], 0);
        }
    }
    VecRestoreArray(bg, &bArray);

    M2->assemble(0, true); // note: layer thickness must be set to 2.0 for all layers 
    MatMult(WQ->M, bg, WQb);      //       before M2 matrix is assembled to initialise theta
    KSPSolve(ksp2, WQb, theta);

    delete WQ;
    VecDestroy(&bg);
    VecDestroy(&WQb);
}
