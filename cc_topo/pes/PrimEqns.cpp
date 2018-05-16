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

PrimEqns::PrimEqns(Topo* _topo, Geom* _geom) {
    PC pc;

    topo = _topo;
    geom = _geom;

    grav = 9.80616*(RAD_SPHERE/RAD_EARTH);
    omega = 7.292e-5;
    del2 = viscosity();
    do_visc = true;
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

// RK2 time integrator
void PrimEqns::solve_RK2(Vec wi, Vec di, Vec hi, Vec wf, Vec df, Vec hf, double dt, bool save) {
    int rank;
    char fieldname[20];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /*** half step ***/
    if(!rank) cout << "half step..." << endl;

    /*** full step ***/
    if(!rank) cout << "full step..." << endl;

    if(!rank) cout << "...done." << endl;

    // write fields
    if(save) {
        step++;
        sprintf(fieldname, "vorticity");
        geom->write0(wf, fieldname, step);
        sprintf(fieldname, "divergence");
        geom->write1(df, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hf, fieldname, step);
    }
}

/*
void PrimEqns::getRHS(Vec wi, Vec di, Vec hi, Vec* Wi, Vec* Di, Vec* Hi) {
    Vec psi, phi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, Wi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Di);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, Hi);


    // derive the stream function and the potential


    // get solenoidal and divergent mass flux components


}
*/

PrimEqns::~PrimEqns() {
    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);

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

    delete edge;
    delete node;
    delete quad;
}

/*
compute the right hand side for the momentum equation for a given level
note that the vertical velocity, uv, is stored as a different vector for 
each element
*/
void PrimEqns::horizMomRHS(Vec uh, Vec* uv, Vec theta, Vec exner, int lev, Vec *Fu) {
    Vec wl, ul, wi, Ru, Ku, Mh, d2u, d4u, dExner, theta_l;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, Fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ru);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ku);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);

    curl(uh, &wi, lev, true);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(topo->gtol_1, uh, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, uh, ul, INSERT_VALUES, SCATTER_FORWARD);

    R->assemble(wl, lev);
    K->assemble(ul, uv, lev);

    MatMult(R->M, uh, Ru);
    MatMult(K->M, uh, Ku);
    MatMult(EtoF->E12, Ku, *Fu);
    VecAXPY(*Fu, 1.0, Ru);

    // add the thermodynamic term
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &theta_l);//TODO 0 form or 2 form??
    VecScatterBegin(topo->gtol_0, theta, theta_l, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, theta, theta_l, INSERT_VALUES, SCATTER_FORWARD);

    grad(exner, &dExner, lev);


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
    VecDestroy(&theta_l);
    if(do_visc) {
        VecDestroy(&d2u);
        VecDestroy(&d4u);
    }
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

void PrimEqns::VertVelRHS(Vec* ui, Vec* wi, Vec **fw) {
    int ex, ey, n2, iLev;
    Mat A, B, G, DG;

    n2 = topo->elOrd*topo->elOrd;

    // vertical velocity is computer per element, so matrices are local to this processor
    MatCreate(MPI_COMM_SELF, &A);
    MatSetType(A, MATSEQAIJ);
    MatSetSizes(A, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(A, topo->elOrd*topo->elOrd, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &B);
    MatSetType(B, MATSEQAIJ);
    MatSetSizes(B, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2, (geom->nk+1)*n2);
    MatSeqAIJSetPreallocation(B, topo->elOrd*topo->elOrd, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &G);
    MatSetType(G, MATSEQAIJ);
    MatSetSizes(G, geom->nk*n2, geom->nk*n2, (geom->nk+1)*n2, (geom->nk+1)*n2);
    MatSeqAIJSetPreallocation(G, topo->elOrd*topo->elOrd, PETSC_NULL);

    *fw = new Vec[geom->nk+1];
    for(iLev = 0; iLev < geom->nk + 1; iLev++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, fw[iLev]);
        VecZeroEntries(*fw[iLev]);
    }

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            AssembleConst(ex, ey, A);
            AssembleLinear(ex, ey, B);
            AssembleGrav(ex, ey, G);

            MatMatMult(V10, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &DG);
            MatAXPY(B, +1.0, DG, SAME_NONZERO_PATTERN); //TODO check this pattern is ok
        }
    }

    MatDestroy(&A);
    MatDestroy(&B);
    MatDestroy(&G);
    MatDestroy(&DG);
}

/*
Assemble the vertical gradient and divergence orientation matrices
*/
void PrimEqns::vertOps() {
    int ii, kk, n2, rows[2], cols[1];
    double vals[2] = {+1.0, -1.0};
    Mat V10t;
    
    n2 = topo->elOrd*topo->elOrd;

    MatCreate(MPI_COMM_WORLD, &V10);
    MatSetType(V10, MATSEQAIJ);
    MatSetSizes(V10, geom->nk*n2, geom->nk*n2, (geom->nk+1)*n2, (geom->nk+1)*n2);
    MatSeqAIJSetPreallocation(V10, 2, PETSC_NULL);

    for(kk = 0; kk < geom->nk; kk++) {
        for(ii = 0; ii < n2; ii++) {
            rows[0] = kk*n2 + ii;
            rows[1] = (kk+1)*n2 + ii;
            cols[0] = kk*n2 + ii;

            MatSetValues(V10, 2, rows, 1, cols, vals, ADD_VALUES);
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
Assemble a 3D mass matrix as a tensor product of 2 forms in the 
horizotnal and constant basis functions in the vertical
*/
void PrimEqns::AssembleConst(int ex, int ey, Mat M0) {
    int ii, kk, ei, mp12;
    int* inds, *inds0;
    double det;
    int inds2k[99];
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

    MatZeroEntries(M0);

    // Assemble the matrices
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

		// Assemble the piecewise constant mass matrix for level k
        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = inds[ii] + kk*W->nDofsJ;
        }
        MatSetValues(M0, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);
    }
    MatAssemblyBegin(M0, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M0, MAT_FINAL_ASSEMBLY);

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
void PrimEqns::AssembleLinear(int ex, int ey, Mat M1) {
    int ii, kk, ei, mp12;
    int* inds, *inds0;
    double det;
    int inds2k[99];
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

    MatZeroEntries(M1);

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
        }

		// Assemble the piecewise constant mass matrix for level k
        Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);
        Mult_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);
        Mult_IP(W->nDofsJ, W->nDofsJ, Q->nDofsJ, WtQ, W->A, WtQW);
        Flat2D_IP(W->nDofsJ, W->nDofsJ, WtQW, WtQWflat);

        // assemble the first basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = inds[ii] + kk*W->nDofsJ;
        }
        MatSetValues(M1, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

        // assemble the second basis function
        for(ii = 0; ii < W->nDofsJ; ii++) {
            inds2k[ii] = inds[ii] + (kk+1)*W->nDofsJ;
        }
        MatSetValues(M1, W->nDofsJ, inds2k, W->nDofsJ, inds2k, WtQWflat, ADD_VALUES);

    }
    MatAssemblyBegin(M1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M1, MAT_FINAL_ASSEMBLY);

    Free2D(Q->nDofsI, Q0);
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    Free2D(W->nDofsJ, WtQW);
    delete[] WtQWflat;
    delete Q;
    delete W;
}

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
Kinetic energy vector for the 2 form column from the 
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

/*
Derive the vertical mass flux
*/
void PrimEqns::VertFlux(int ex, int ey, Vec* pi, Vec wi, Mat Mp) {
    int ii, kk, ei, mp1, mp12;
    int* inds;
    double det, rho;
    int inds2k[99];
    Wii* Q = new Wii(node->q, geom);
    M2_j_xy_i* W = new M2_j_xy_i(edge);
    double** Q0 = Alloc2D(Q->nDofsI, Q->nDofsJ);
    double** Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    double** WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);
    double** WtQW = Alloc2D(W->nDofsJ, W->nDofsJ);
    double* WtQWflat = new double[W->nDofsJ*W->nDofsJ];
    PetscScalar* pArray;

    inds  = topo->elInds2_g(ex, ey);
    mp1   = quad->n + 1;
    mp12  = mp1*mp1;

    MatZeroEntries(Mp);

    // Assemble the matrices
    for(kk = 0; kk < geom->nk; kk++) {
        // build the 2D mass matrix
        Q->assemble(ex, ey);
        ei = ey*topo->nElsX + ex;
        
        VecGetArray(pi[kk], &pArray);

        for(ii = 0; ii < mp12; ii++) {
            det = geom->det[ei][ii];
            Q0[ii][ii] = Q->A[ii][ii]/det/det;

            geom->interp2_g(ex, ey, ii%mp1, ii/mp1, pArray, &rho);
            Q0[ii][ii] *= rho;

            // multiply by the vertical determinant for the vertical integral,
            // then divide by the vertical determinant to rescale the piecewise
            // constant density, so do nothing.
        }
        VecRestoreArray(pi[kk], &pArray);

		// Assemble the piecewise constant mass matrix for level k
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
