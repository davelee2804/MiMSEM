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
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn_Rosenbrock.h"

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
//#define UP_VORT 1
#define UP_APVM 1
#define H_MEAN 1.0e+4
#define UP_TAU (0.5)

using namespace std;

SWEqn::SWEqn(Topo* _topo, Geom* _geom) {
    PC pc;
    int ii, jj;
    int dof_proc;
    int* loc = new int[_topo->n1+_topo->n2];
    IS is_g, is_l;
    Vec xl, xg;

    topo = _topo;
    geom = _geom;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    grav = 9.80616*(RAD_SPHERE/RAD_EARTH);
    omega = 7.292e-5;
    step = 0;

    quad = new GaussLobatto(geom->quad->n);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // 0 form lumped mass matrix (vector)
    M0 = new Pmat(topo, geom, node);

    // 1 form mass matrix
    M1 = new Umat(topo, geom, node, edge);

    // 2 form mass matrix
    M2 = new Wmat(topo, geom, edge);

    // incidence matrices
    NtoE = new E10mat(topo);
    EtoF = new E21mat(topo);

    // adjoint differential operators (curl and grad)
    MatMatMult(NtoE->E01, M1->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E01M1);
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E12M2);

    // rotational operator
    R = new RotMat(topo, geom, node, edge);
    R_up = new RotMat_up(topo, geom, node, edge);

    // mass flux operator
    M1h = new Uhmat(topo, geom, node, edge);

    M0h = new Phmat(topo, geom, node);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    // initialize the linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp, "sw_");
    KSPSetFromOptions(ksp);

    KSPCreate(MPI_COMM_WORLD, &ksp0);
    KSPSetOperators(ksp0, M0->M, M0->M);
    KSPSetTolerances(ksp0, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp0, KSPGMRES);
    KSPGetPC(ksp0, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp0, "ksp_0_");
    KSPSetFromOptions(ksp0);

    KSPCreate(MPI_COMM_WORLD, &ksp0h);
    KSPSetOperators(ksp0h, M0h->M, M0h->M);
    KSPSetTolerances(ksp0h, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp0h, KSPGMRES);
    KSPGetPC(ksp0h, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp0h, "ksp_0_");
    KSPSetFromOptions(ksp0h);

    // coriolis vector (projected onto 0 forms)
    coriolis();

    A = NULL;
    B = NULL;
    kspA = NULL;
    DM1inv = NULL;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);

    // create the [u,h] -> [x] vec scatter
    for(ii = 0; ii < topo->n1; ii++) {
        dof_proc = topo->loc1[ii] / topo->n1l;
        loc[ii] = dof_proc * (topo->n1l + topo->n2l) + topo->loc1[ii] % topo->n1l;
    }
    for(ii = 0; ii < topo->n2; ii++) {
        jj = ii + topo->n1;
        loc[jj] = rank*(topo->n1l + topo->n2l) + ii + topo->n1l;
    }

    ISCreateStride(MPI_COMM_SELF, topo->n1+topo->n2, 0, 1, &is_l);
    ISCreateGeneral(MPI_COMM_WORLD, topo->n1+topo->n2, loc, PETSC_COPY_VALUES, &is_g);

    VecCreateSeq(MPI_COMM_SELF, topo->n1+topo->n2, &xl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &xg);

    VecScatterCreate(xg, is_g, xl, is_l, &gtol_x);

    delete[] loc;
    ISDestroy(&is_l);
    ISDestroy(&is_g);
    VecDestroy(&xl);
    VecDestroy(&xg);

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ujl);

    KSPCreate(MPI_COMM_WORLD, &ksp1h);
    KSPSetOperators(ksp1h, M1->M, M1->M);
    KSPSetTolerances(ksp1h, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1h, KSPGMRES);
    KSPGetPC(ksp1h, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1h, "Fonh_");
    KSPSetFromOptions(ksp1h);

    // rosenbrock data
    n_rosenbrock = 4;
    ki = new Vec[n_rosenbrock];
    alpha_ij = new double*[n_rosenbrock];
    gamma_ij = new double*[n_rosenbrock];
    for(ii = 0; ii < n_rosenbrock; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &ki[ii]);
	alpha_ij[ii] = new double[n_rosenbrock];
	gamma_ij[ii] = new double[n_rosenbrock];
	for(jj = 0; jj < n_rosenbrock; jj++) {
	    alpha_ij[ii][jj] = gamma_ij[ii][jj] = 0.0;
	}
    }
    // ROS34PW2
    /*
    gamma_0        = 4.3586652150845900e-01;
    alpha_ij[0][0] = 8.7173304301691801e-01;
    alpha_ij[1][0] = 8.4457060015369423e-01;
    alpha_ij[1][1] = -1.1299064236484185e-01;
    alpha_ij[2][2] = 1.0;
    alpha_ij[3][0] = 2.4212380706095346e-01;
    alpha_ij[3][1] = -1.2232505839045147e+00;
    alpha_ij[3][2] = 1.5452602553351020e+00;
    alpha_ij[3][3] = 4.3586652150845900e-01;
    gamma_ij[1][0] = -8.7173304301691801e-01;
    gamma_ij[2][0] = -9.0338057013044082e-01;
    gamma_ij[2][1] = 5.4180672388095326e-02;
    gamma_ij[3][0] = 2.4212380706095346e-01;
    gamma_ij[3][1] = -1.2232505839045147e+00;
    gamma_ij[3][2] = 5.4526025533510214e-01;
    */
    // ROS34PW3
    gamma_0        = 1.0685790213016289e+00;
    alpha_ij[0][0] = 2.5155456020628817e+00;
    alpha_ij[1][0] = 5.0777280103144085e-01;
    alpha_ij[1][1] = 7.5000000000000000e-01;
    alpha_ij[2][0] = 1.3959081404277204e-01;
    alpha_ij[2][1] = -3.3111001065419338e-01;
    alpha_ij[2][2] = 8.2040559712714178e-01;
    alpha_ij[3][0] = 2.2047681286931747e-01;
    alpha_ij[3][1] = 2.7828278331185935e-03;
    alpha_ij[3][2] = 7.1844787635140066e-03;
    alpha_ij[3][3] = 7.6955588053404989e-01;
    gamma_ij[1][0] = -2.5155456020628817e+00;
    gamma_ij[2][0] = -8.7991339217106512e-01;
    gamma_ij[2][1] = -9.6014187766190695e-01;
    gamma_ij[3][0] = -4.1731389379448741e-01;
    gamma_ij[3][1] = 4.1091047035857703e-01;
    gamma_ij[3][2] = -1.3558873204765276e+00;
    // ROSI2Pw (T=0)
    /*
    gamma_0        = 4.3586652150845900e-01;
    alpha_ij[0][0] = 8.7173304301691801e-01;
    alpha_ij[1][0] = 7.8938917169345013e-01;
    alpha_ij[1][1] = -3.9389171693450180e-02;
    alpha_ij[2][0] = 6.2787416864263046e-01;
    alpha_ij[2][1] = 6.9295440480994763e+00;
    alpha_ij[2][2] = -6.5574182167421071e+00;
    alpha_ij[3][0] = 2.4822549716173517e-01;
    alpha_ij[3][1] = -1.4194790767022774e+00;
    alpha_ij[3][2] = 1.7353870580320832e+00;
    alpha_ij[3][3] = 4.3586652150845900e-01;
    gamma_ij[1][0] = -8.7173304301691801e-01;
    gamma_ij[2][0] = -8.4175599602920992e-01;
    gamma_ij[2][1] = -1.2977652642309580e-02;
    gamma_ij[3][0] = -3.7964867148089526e-01;
    gamma_ij[3][1] = -8.3490231248017537e+00;
    gamma_ij[3][2] = 8.2928052747741905e+00;
    */
    // ROSI2PW
    /*
    gamma_0        = 4.3586652150845900e-01;
    alpha_ij[0][0] = 8.7173304301691801e-01;
    alpha_ij[1][0] = -7.9937335839852708e-01;
    alpha_ij[1][1] = -7.9937335839852708e-01;
    alpha_ij[2][0] = 7.0849664917601007e-01;
    alpha_ij[2][1] = 3.1746327955312481e-01;
    alpha_ij[2][2] = -2.5959928729134892e-02;
    alpha_ij[3][0] = 6.0424832458800504e-01;
    alpha_ij[3][1] = 0.0;
    alpha_ij[3][2] = -4.0114846096464034e-02;
    alpha_ij[3][3] = 4.3586652150845900e-01;
    gamma_ij[1][0] = -8.7173304301691801e-01;
    gamma_ij[2][0] = 3.0647867418622479e+00;
    gamma_ij[2][1] = 3.0647867418622479e+00;
    gamma_ij[3][0] = -1.0424832458800504e-01;
    gamma_ij[3][1] = -3.1746327955312481e-01;
    gamma_ij[3][2] = -1.4154917367329144e-02;
    */
    // ROWDAIND2
    /*
    gamma_0        = 3.000000000000000e-01;
    alpha_ij[0][0] = 5.000000000000000e-01;
    alpha_ij[1][0] = 2.800000000000000e-01;
    alpha_ij[1][1] = 7.200000000000000e-01;
    alpha_ij[2][0] = 2.800000000000000e-01;
    alpha_ij[2][1] = 7.200000000000000e-01;
    alpha_ij[3][0] = 6.666666666666666e-01;
    alpha_ij[3][2] = 3.333333333333333e-02;
    alpha_ij[3][3] = 3.000000000000000e-01;
    gamma_ij[1][0] = -1.121794871794876e-01;
    gamma_ij[2][0] = 2.540000000000000e+00;
    gamma_ij[2][1] = -3.840000000000000e+00;
    gamma_ij[3][0] = 3.866666666666667e-01;
    gamma_ij[3][1] = -7.200000000000000e-01;
    gamma_ij[3][2] = 3.333333333333333e-02;
    */
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void SWEqn::coriolis() {
    int ii, size;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fxl, fxg, PtQfxg;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // initialise the coriolis vector (local and global)
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg);

    // evaluate the coriolis term at nodes
    VecCreateSeq(MPI_COMM_SELF, geom->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < geom->n0; ii++) {
#ifdef W2_ALPHA
        fArray[ii] = 2.0*omega*( -cos(geom->s[ii][0])*cos(geom->s[ii][1])*sin(W2_ALPHA) + sin(geom->s[ii][1])*cos(W2_ALPHA) );
#else
        fArray[ii] = 2.0*omega*sin(geom->s[ii][1]);
#endif
    }
    VecRestoreArray(fxl, &fArray);

    // scatter array to global vector
    VecScatterBegin(geom->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  geom->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);

    // project vector onto 0 forms
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PtQfxg);
    VecZeroEntries(PtQfxg);
    MatMult(PtQ->M, fxg, PtQfxg);
    // diagonal mass matrix as vector
    //VecPointwiseDivide(fg, PtQfxg, m0->vg);
    KSPSolve(ksp0, PtQfxg, fg);
    
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    VecScatterBegin(topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, fg, fl, INSERT_VALUES, SCATTER_FORWARD);

    delete PtQ;
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

// derive vorticity (global vector) as \omega = curl u
void SWEqn::curl(Vec u, Vec *w) {
    Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);

    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    KSPSolve(ksp0, du, *w);

    VecDestroy(&du);
}

// dH/du = hu = F
void SWEqn::diagnose_F(Vec _h, Vec _u, Vec F) {
    Vec b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &b);

    M1h->assemble(_h);
    MatMult(M1h->M, _u, b);
    KSPSolve(ksp, b, F);

    VecDestroy(&b);
}

// dH/dh = (1/2)u^2 + gh = \Phi
// note: \Phi is in integral form here
//          \int_{\Omega} \gamma_h,\Phi_h d\Omega
void SWEqn::diagnose_Phi(Vec _h, Vec _u, Vec _ul, Vec Phi) {
    Vec b;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &b);

    // u^2 terms (0.5 factor incorportated into the matrix assembly)
    K->assemble(_ul);
    MatMult(K->M, _u, Phi);

    // gh terms
    MatMult(M2->M, _h, b);
    VecAXPY(Phi, grav, b);

    VecDestroy(&b);
}

void SWEqn::diagnose_q(double _dt, Vec _ug, Vec _ul, Vec _h, Vec qi) {
    Vec rhs, tmp;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &rhs);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &tmp);

    MatMult(M0->M, fg, rhs);
    MatMult(E01M1, _ug, tmp);
    VecAXPY(rhs, 1.0, tmp);
    if(_dt > 1.0e-6) {
        M0h->assemble_up(_ul, _h, UP_TAU, _dt);
    } else {
        M0h->assemble(_h);
    }
    KSPSolve(ksp0h, rhs, qi);

    VecDestroy(&rhs);
    VecDestroy(&tmp);
}

void SWEqn::unpack(Vec x, Vec u, Vec h) {
    Vec xl, ul;
    PetscScalar *xArray, *uArray, *hArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, topo->n1 + topo->n2, &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);

    VecScatterBegin(gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  gtol_x, x, xl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(xl, &xArray);
    VecGetArray(ul, &uArray);
    VecGetArray(h, &hArray);
    for(ii = 0; ii < topo->n1; ii++) {
        uArray[ii] = xArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        hArray[ii] = xArray[ii+topo->n1];
    }
    VecRestoreArray(xl, &xArray);
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(h, &hArray);

    VecScatterBegin(topo->gtol_1, ul, u, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_1, ul, u, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&xl);
    VecDestroy(&ul);
}

void SWEqn::repack(Vec x, Vec u, Vec h) {
    Vec xl, ul;
    PetscScalar *xArray, *uArray, *hArray;
    int ii;

    VecCreateSeq(MPI_COMM_SELF, topo->n1 + topo->n2, &xl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecScatterBegin(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(xl, &xArray);
    VecGetArray(ul, &uArray);
    VecGetArray(h, &hArray);
    for(ii = 0; ii < topo->n1; ii++) {
        xArray[ii] = uArray[ii];
    }
    for(ii = 0; ii < topo->n2; ii++) {
        xArray[ii+topo->n1] = hArray[ii];
    }
    VecRestoreArray(xl, &xArray);
    VecRestoreArray(ul, &uArray);
    VecRestoreArray(h, &hArray);

    VecScatterBegin(gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  gtol_x, xl, x, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&xl);
    VecDestroy(&ul);
}

void SWEqn::assemble_residual(Vec x, Vec f) {
    Vec F, Phi, fu, fh, utmp, htmp, qj, qjl, dql, dqg;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &fu);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &fh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &utmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &qjl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dqg);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dql);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Phi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &qj);

    VecZeroEntries(fu);
    VecZeroEntries(fh);

    unpack(x, uj, hj);

    VecScatterBegin(topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, uj, ujl, INSERT_VALUES, SCATTER_FORWARD);

    // assemble in the skew-symmetric parts of the vector
    diagnose_F(hj, uj, F);
    diagnose_Phi(hj, uj, ujl, Phi);

    // momentum terms
    MatMult(EtoF->E12, Phi, fu);

    diagnose_q(0.0, uj, ujl, hj, qj);
    VecScatterBegin(topo->gtol_0, qj, qjl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_0, qj, qjl, INSERT_VALUES, SCATTER_FORWARD);

//#ifdef UP_VORT
//    R_up->assemble(qjl, ujl, UP_TAU, dt);
//    MatMult(R_up->M, F, utmp);
//#elif UP_APVM
    MatMult(NtoE->E10, qj, dqg);
    VecScatterBegin(topo->gtol_1, dqg, dql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, dqg, dql, INSERT_VALUES, SCATTER_FORWARD);
    R_up->assemble_apvm(qjl, ujl, dql, UP_TAU, dt);
    MatMult(R_up->M, F, utmp);
//#else
//    R->assemble(qjl);
//    MatMult(R->M, F, utmp);
//#endif
    VecAXPY(fu, 1.0, utmp);

    // continuity term
    MatMult(EtoF->E21, F, htmp);
    MatMult(M2->M, htmp, fh);

    repack(f, fu, fh);
    VecScale(f, -dt);

    // clean up
    VecDestroy(&fu);
    VecDestroy(&fh);
    VecDestroy(&utmp);
    VecDestroy(&htmp);
    VecDestroy(&F);
    VecDestroy(&Phi);
    VecDestroy(&qj);
    VecDestroy(&qjl);
    VecDestroy(&dql);
    VecDestroy(&dqg);
}

void SWEqn::assemble_operator(double _dt) {
    int n2 = (topo->elOrd+1)*(topo->elOrd+1);
    int mm, mi, mf, ri, ci, dof_proc;
    int nCols;
    const int *cols;
    const double* vals;
    int cols2[9999];
    Mat Muh, Mhu;

    if(!A) {
        MatCreate(MPI_COMM_WORLD, &A);
        MatSetSizes(A, topo->n1l+topo->n2l, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, topo->nDofs1G+topo->nDofs2G);
        MatSetType(A, MATMPIAIJ);
        MatMPIAIJSetPreallocation(A, 16*n2, PETSC_NULL, 16*n2, PETSC_NULL);

        KSPCreate(MPI_COMM_WORLD, &kspA);
        KSPSetOperators(kspA, A, A);
        KSPSetTolerances(kspA, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
        KSPSetOptionsPrefix(kspA, "A_");
        KSPSetFromOptions(kspA);

        MatCreate(MPI_COMM_WORLD, &B);
        MatSetSizes(B, topo->n1l+topo->n2l, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, topo->nDofs1G+topo->nDofs2G);
        MatSetType(B, MATMPIAIJ);
        MatMPIAIJSetPreallocation(B, 16*n2, PETSC_NULL, 16*n2, PETSC_NULL);
    }

    R->assemble(fl);
    MatAXPY(M1->M, gamma_0*_dt, R->M, DIFFERENT_NONZERO_PATTERN);
    MatAssemblyBegin(M1->M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M1->M, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(M1->M, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(M1->M, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(M1->M, mm, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(M1->M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M1->M, MAT_FINAL_ASSEMBLY);

    // [u,h] block
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Muh);
    MatScale(Muh, gamma_0*_dt*grav);
    MatAssemblyBegin(Muh, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Muh, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Muh, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Muh, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Muh, mm, &nCols, &cols, &vals);
    }

    // [h,u] block
    MatMatMult(M2->M, EtoF->E21, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Mhu);
    MatScale(Mhu, gamma_0*_dt*H_MEAN);
    MatAssemblyBegin(Mhu, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  Mhu, MAT_FINAL_ASSEMBLY);

    MatGetOwnershipRange(Mhu, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Mhu, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Mhu, mm, &nCols, &cols, &vals);
    }

    // [h,h] block
    MatGetOwnershipRange(M2->M, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(M2->M, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
        }
        MatSetValues(A, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(M2->M, mm, &nCols, &cols, &vals);
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);

    //
    R->assemble(fl);
    MatScale(R->M, gamma_0*_dt);
    MatGetOwnershipRange(R->M, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(R->M, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(B, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(R->M, mm, &nCols, &cols, &vals);
    }

    // [u,h] block
    MatGetOwnershipRange(Muh, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Muh, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n1l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n2l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n2l + topo->n1l;
        }
        MatSetValues(B, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Muh, mm, &nCols, &cols, &vals);
    }

    // [h,u] block
    MatGetOwnershipRange(Mhu, &mi, &mf);
    for(mm = mi; mm < mf; mm++) {
        MatGetRow(Mhu, mm, &nCols, &cols, &vals);
        dof_proc = mm / topo->n2l;
        ri = dof_proc * (topo->n1l + topo->n2l) + mm % topo->n2l + topo->n1l;
        for(ci = 0; ci < nCols; ci++) {
            dof_proc = cols[ci] / topo->n1l;
            cols2[ci] = dof_proc * (topo->n1l + topo->n2l) + cols[ci] % topo->n1l;
        }
        MatSetValues(B, 1, &ri, nCols, cols2, vals, INSERT_VALUES);
        MatRestoreRow(Mhu, mm, &nCols, &cols, &vals);
    }

    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  B, MAT_FINAL_ASSEMBLY);

    MatDestroy(&Muh);
    MatDestroy(&Mhu);

    M1->assemble();
}

void SWEqn::solve(Vec un, Vec hn, double _dt, bool save) {
    int ros_i, ros_j;
    double norm = 1.0e+9, norm_dx, norm_x;
    Vec x, f, dx, tmp;

    dt = _dt;

    if(!A) assemble_operator(dt);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &x);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &f);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &dx);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l+topo->n2l, topo->nDofs1G+topo->nDofs2G, &tmp);

    // solution vector
    VecCopy(un, uj);
    VecCopy(hn, hj);
    repack(x, un, hn);

    for(ros_i = 0; ros_i < n_rosenbrock; ros_i++) {
        assemble_residual(x, f);
	if(ros_i > 0) {
            VecZeroEntries(dx);
            for(ros_j = 0; ros_j < ros_i; ros_j++) {
                VecAXPY(dx, gamma_ij[ros_i][ros_j]/gamma_0, ki[ros_j]);
            }
	    MatMult(B, dx, tmp);
	    VecAXPY(f, -1.0, tmp);
        }
        KSPSolve(kspA, f, ki[ros_i]);

        VecZeroEntries(x);
	repack(x, un, hn);
	for(ros_j = 0; ros_j <= ros_i; ros_j++) {
            VecAXPY(x, alpha_ij[ros_i][ros_j], ki[ros_j]);
        }
        unpack(x, uj, hj);

        VecNorm(x, NORM_2, &norm_x);
        VecNorm(ki[ros_i], NORM_2, &norm_dx);
        norm = norm_dx/norm_x;
        if(!rank) {
            cout << scientific;
            cout << "iteration: " << ros_i << "\t|x|: " << norm_x << "\t|dx|: " << norm_dx << "\t|dx|/|x|: " << norm << endl; 
        }
    }

    unpack(x, un, hn);

    if(save) {
        Vec wi;
        char fieldname[20];

        step++;
        curl(un, &wi);

        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(un, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hn, fieldname, step);

        VecDestroy(&wi);
    }

    VecDestroy(&x);
    VecDestroy(&f);
    VecDestroy(&dx);
    VecDestroy(&tmp);
}

SWEqn::~SWEqn() {
    KSPDestroy(&ksp);
    KSPDestroy(&ksp0);
    KSPDestroy(&ksp0h);
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);
    VecDestroy(&fl);
    VecScatterDestroy(&gtol_x);
    VecDestroy(&uj);
    VecDestroy(&hj);
    VecDestroy(&ujl);
    if(A) { 
        MatDestroy(&A);
        MatDestroy(&B);
        KSPDestroy(&kspA);
    }
    if(DM1inv) {
        MatDestroy(&Muf);
        MatDestroy(&G);
        MatDestroy(&D);
        MatDestroy(&DM1inv);
        MatDestroy(&M1inv);
    }

    delete M0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete M1h;
    delete M0h;
    delete K;

    delete edge;
    delete node;
    delete quad;
}

void SWEqn::init0(Vec q, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int* inds0;
    PtQmat* PQ = new PtQmat(topo, geom, node);
    PetscScalar *bArray;
    Vec bl, bg, PQb;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, geom->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);
    VecScatterBegin(geom->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  geom->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(PQ->M, bg, PQb);
    //VecPointwiseDivide(q, PQb, m0->vg);
    KSPSolve(ksp0, PQb, q);

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void SWEqn::init1(Vec u, ICfunc* func_x, ICfunc* func_y) {
    int ex, ey, ii, mp1, mp12;
    int *inds0, *loc02;
    UtQmat* UQ = new UtQmat(topo, geom, node, edge);
    PetscScalar *bArray;
    Vec bl, bg, UQb;
    IS isl, isg;
    VecScatter scat;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, 2*geom->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, 2*geom->n0l, 2*geom->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &UQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[2*inds0[ii]+0] = func_x(geom->x[inds0[ii]]);
                bArray[2*inds0[ii]+1] = func_y(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);

    // create a new vec scatter object to handle vector quantity on nodes
    loc02 = new int[2*geom->n0];
    for(ii = 0; ii < geom->n0; ii++) {
        loc02[2*ii+0] = 2*geom->loc0[ii]+0;
        loc02[2*ii+1] = 2*geom->loc0[ii]+1;
    }
    ISCreateStride(MPI_COMM_WORLD, 2*geom->n0, 0, 1, &isl);
    ISCreateGeneral(MPI_COMM_WORLD, 2*geom->n0, loc02, PETSC_COPY_VALUES, &isg);
    VecScatterCreate(bg, isg, bl, isl, &scat);
    VecScatterBegin(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(UQ->M, bg, UQb);
    KSPSolve(ksp, UQb, u);

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&UQb);
    ISDestroy(&isl);
    ISDestroy(&isg);
    VecScatterDestroy(&scat);
    delete UQ;
    delete[] loc02;
}

void SWEqn::init2(Vec h, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    PetscScalar *bArray;
    KSP ksp2;
    Vec bl, bg, WQb;
    WtQmat* WQ = new WtQmat(topo, geom, edge);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, geom->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &WQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = geom->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);
    VecScatterBegin(geom->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(geom->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(WQ->M, bg, WQb);

    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPSetOptionsPrefix(ksp2, "init2_");
    KSPSetFromOptions(ksp2);
    KSPSolve(ksp2, WQb, h);

    delete WQ;
    KSPDestroy(&ksp2);
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void SWEqn::err0(Vec ug, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds0;
    double det, wd, l_inf;
    double un[1], dun[2], ua[1], dua[2];
    double local_1[2], global_1[2], local_2[2], global_2[2], local_i[2], global_i[2]; // first entry is the error, the second is the norm
    PetscScalar *array_0, *array_1;
    Vec ul, dug, dul;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dug);

    VecScatterBegin(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    MatMult(NtoE->E10, ug, dug);
    VecScatterBegin(topo->gtol_1, dug, dul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, dug, dul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local_1[0] = local_1[1] = 0.0;
    local_2[0] = local_2[1] = 0.0;
    local_i[0] = local_i[1] = 0.0;

    VecGetArray(ul, &array_0);
    VecGetArray(dul, &array_1);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
                geom->interp0(ex, ey, ii%mp1, ii/mp1, array_0, un);
                ua[0] = fw(geom->x[inds0[ii]]);

                det = geom->det[ei][ii];
                wd = det*quad->w[ii%mp1]*quad->w[ii/mp1];

                local_1[0] += wd*fabs(un[0] - ua[0]);
                local_1[1] += wd*fabs(ua[0]);

                local_2[0] += wd*(un[0] - ua[0])*(un[0] - ua[0]);
                local_2[1] += wd*ua[0]*ua[0];

                l_inf = wd*fabs(un[0] - ua[0]);
                if(fabs(l_inf) > local_i[0]) {
                    local_i[0] = fabs(l_inf);
                    local_i[1] = fabs(wd*fabs(ua[0]));
                }

                if(fu != NULL && fv != NULL) {
                    geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, dun);
                    dua[0] = fu(geom->x[inds0[ii]]);
                    dua[1] = fv(geom->x[inds0[ii]]);

                    local_2[0] += wd*((dun[0] - dua[0])*(dun[0] - dua[0]) + (dun[1] - dua[1])*(dun[1] - dua[1]));
                    local_2[1] += wd*(dua[0]*dua[0] + dua[1]*dua[1]);
                }
            }
        }
    }
    VecRestoreArray(ul, &array_0);
    VecRestoreArray(dul, &array_1);

    MPI_Allreduce(local_1, global_1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_2, global_2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_i, global_i, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    VecDestroy(&ul);
    VecDestroy(&dul);
    VecDestroy(&dug);

    norms[0] = global_1[0]/global_1[1];
    norms[1] = sqrt(global_2[0]/global_2[1]);
    norms[2] = global_i[0]/global_i[1];
}

void SWEqn::err1(Vec ug, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds0;
    double det, wd, l_inf;
    double un[2], dun[1], ua[2], dua[1];
    double local_1[2], global_1[2], local_2[2], global_2[2], local_i[2], global_i[2]; // first entry is the error, the second is the norm
    PetscScalar *array_1, *array_2;
    Vec ul, dug, dul;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &dul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dug);

    VecScatterBegin(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    MatMult(EtoF->E21, ug, dug);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local_1[0] = local_1[1] = 0.0;
    local_2[0] = local_2[1] = 0.0;
    local_i[0] = local_i[1] = 0.0;

    VecGetArray(ul, &array_1);
    VecGetArray(dug, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, un);
                ua[0] = fu(geom->x[inds0[ii]]);
                ua[1] = fv(geom->x[inds0[ii]]);

                det = geom->det[ei][ii];
                wd = det*quad->w[ii%mp1]*quad->w[ii/mp1];

                local_1[0] += wd*(fabs(un[0] - ua[0]) + fabs(un[1] - ua[1]));
                local_1[1] += wd*(fabs(ua[0]) + fabs(ua[1]));

                local_2[0] += wd*((un[0] - ua[0])*(un[0] - ua[0]) + (un[1] - ua[1])*(un[1] - ua[1]));
                local_2[1] += wd*(ua[0]*ua[0] + ua[1]*ua[1]);

                l_inf = wd*(fabs(un[0] - ua[0]) + fabs(un[1] - ua[1]));
                if(fabs(l_inf) > local_i[0]) {
                    local_i[0] = fabs(l_inf);
                    local_i[1] = wd*(fabs(ua[0]) + fabs(ua[1]));
                }
 
                if(fp != NULL) {
                    geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, dun);
                    dua[0] = fp(geom->x[inds0[ii]]);

                    local_2[0] += wd*(dun[0] - dua[0])*(dun[0] - dua[0]);
                    local_2[1] += wd*dua[0]*dua[0];
                }
            }
        }
    }
    VecRestoreArray(ul, &array_1);
    VecRestoreArray(dug, &array_2);

    MPI_Allreduce(local_1, global_1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_2, global_2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_i, global_i, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    VecDestroy(&ul);
    VecDestroy(&dul);
    VecDestroy(&dug);

    norms[0] = global_1[0]/global_1[1];
    norms[1] = sqrt(global_2[0]/global_2[1]);
    norms[2] = global_i[0]/global_i[1];
}

void SWEqn::err2(Vec ug, ICfunc* fu, double* norms) {
    int ex, ey, ei, ii, mp1, mp12;
    int *inds0;
    double det, wd, l_inf;
    double un[1], ua[1];
    double local_1[2], global_1[2], local_2[2], global_2[2], local_i[2], global_i[2]; // first entry is the error, the second is the norm
    PetscScalar *array_2;
    Vec ul;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &ul);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local_1[0] = local_1[1] = 0.0;
    local_2[0] = local_2[1] = 0.0;
    local_i[0] = local_i[1] = 0.0;

    VecGetArray(ug, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
if(fabs(geom->s[inds0[ii]][1]) > 0.45*M_PI) continue;
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, un);
                ua[0] = fu(geom->x[inds0[ii]]);

                det = geom->det[ei][ii];
                wd = det*quad->w[ii%mp1]*quad->w[ii/mp1];

                local_1[0] += wd*fabs(un[0] - ua[0]);
                local_1[1] += wd*fabs(ua[0]);

                local_2[0] += wd*(un[0] - ua[0])*(un[0] - ua[0]);
                local_2[1] += wd*ua[0]*ua[0];

                l_inf = wd*fabs(un[0] - ua[0]);
                if(fabs(l_inf) > local_i[0]) {
                    local_i[0] = fabs(l_inf);
                    local_i[1] = fabs(wd*fabs(ua[0]));
                }

            }
        }
    }
    VecRestoreArray(ug, &array_2);

    MPI_Allreduce(local_1, global_1, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_2, global_2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_i, global_i, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    VecDestroy(&ul);

    norms[0] = global_1[0]/global_1[1];
    norms[1] = sqrt(global_2[0]/global_2[1]);
    norms[2] = global_i[0]/global_i[1];
}

double SWEqn::int0(Vec ug) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, uq, local, global;
    PetscScalar *array_0;
    Vec ul;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ul);
    VecScatterBegin(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ul, &array_0);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp0(ex, ey, ii%mp1, ii/mp1, array_0, &uq);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*uq;
            }
        }
    }
    VecRestoreArray(ul, &array_0);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&ul);

    return global;
}

double SWEqn::int2(Vec ug) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, uq, local, global;
    PetscScalar *array_2;
    Vec ul;

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &ul);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ug, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, &uq);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*uq;
            }
        }
    }
    VecRestoreArray(ug, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&ul);

    return global;
}

double SWEqn::intE(Vec ug, Vec hg) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, hq, local, global;
    double uq[2];
    PetscScalar *array_1, *array_2;
    Vec ul, hl;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecScatterBegin(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ul, &array_1);
    VecGetArray(hg, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det = geom->det[ei][ii];
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, uq);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, &hq);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*0.5*(grav*hq*hq + hq*(uq[0]*uq[0] + uq[1]*uq[1]));
            }
        }
    }
    VecRestoreArray(ul, &array_1);
    VecRestoreArray(hg, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    VecDestroy(&ul);
    VecDestroy(&hl);

    return global;
}

void SWEqn::writeConservation(double time, Vec u, Vec h, double mass0, double vort0, double ener0, double enst0) {
    double mass, vort, ener, enst;
    char filename[50];
    ofstream file;
    Vec wj, qj, v0;

    curl(u, &wj);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &v0);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &qj);
    diagnose_q(0.0, u, ujl, h, qj);
    MatMult(M0h->M, qj, v0);
    VecDot(qj, v0, &enst);

    mass = int2(h);
    vort = int0(wj);
    ener = intE(u, h);

    if(!rank) {
        cout << "conservation of mass:      " << (mass - mass0)/mass0 << endl;
        cout << "conservation of vorticity: " << (vort - vort0) << endl;
        cout << "conservation of energy:    " << (ener - ener0)/ener0 << endl;
        cout << "conservation of enstrophy: " << (enst - enst0)/enst0 << endl;

        sprintf(filename, "output/conservation.dat");
        file.open(filename, ios::out | ios::app);
        // write time in days
        file << scientific;
        file << time/60.0/60.0/24.0 << "\t" << (mass-mass0)/mass0 << "\t" << (vort-vort0) << "\t" 
                                            << (ener-ener0)/ener0 << "\t" << (enst-enst0)/enst0 << endl;
        file.close();
    }
    VecDestroy(&wj);
    VecDestroy(&qj);
    VecDestroy(&v0);
} 
