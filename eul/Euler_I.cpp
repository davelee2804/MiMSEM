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
#include "Assembly_Coupled.h"
#include "HorizSolve.h"
#include "VertSolve.h"
#include "Euler_I.h"

#define RAD_EARTH 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8

using namespace std;

Euler_I::Euler_I(Topo* _topo, Geom* _geom, double _dt) {
    int ii, n2, size, n_dofs_locl, n_dofs_glob;
    Vec m2tmp1, m2tmp2;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    hs_forcing = false;
    step = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    Rz = new WtQdUdz_mat(topo, geom, node, edge);

    // potential temperature projection operator
    T = new Whmat(topo, geom, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();

    // initialize the 1 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp1);
    KSPSetOperators(ksp1, M1->M, M1->M);
    KSPSetTolerances(ksp1, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1, KSPGMRES);
    KSPGetPC(ksp1, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1, "ksp1_");
    KSPSetFromOptions(ksp1);

    // initialize the 2 form linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp2);
    KSPSetOperators(ksp2, M2->M, M2->M);
    KSPSetTolerances(ksp2, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp2, KSPGMRES);
    KSPGetPC(ksp2, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp2, "ksp2_");
    KSPSetFromOptions(ksp2);

    gv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd*topo->elOrd, &gv[ii]);
    }
    zv = new Vec[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*topo->elOrd*topo->elOrd, &zv[ii]);
    }
    uz = new Vec[geom->nk-1];
    uzl_i = new Vec[geom->nk-1];
    uzl_j = new Vec[geom->nk-1];
    for(ii = 0; ii < geom->nk - 1; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uz[ii]);
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &uzl_i[ii]);
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &uzl_j[ii]);
    }
    uil = new Vec[geom->nk];
    ujl = new Vec[geom->nk];
    uhl = new Vec[geom->nk];
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &uil[ii]);
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &ujl[ii]);
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &uhl[ii]);
    }
    uuz = new L2Vecs(geom->nk-1, topo, geom);

    // initialise the single column mass matrices and solvers
    n2 = topo->elOrd*topo->elOrd;

    // implicit vertical solver
    vert = new VertSolve(topo, geom, dt);

    MatCreate(MPI_COMM_SELF, &VA);
    MatSetType(VA, MATSEQAIJ);
    MatSetSizes(VA, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2, (geom->nk-1)*n2);
    MatSeqAIJSetPreallocation(VA, 2*n2, PETSC_NULL);

    MatCreate(MPI_COMM_SELF, &VB);
    MatSetType(VB, MATSEQAIJ);
    MatSetSizes(VB, geom->nk*n2, geom->nk*n2, geom->nk*n2, geom->nk*n2);
    MatSeqAIJSetPreallocation(VB, n2, PETSC_NULL);

    Q = new Wii(node->q, geom);
    W = new M2_j_xy_i(edge);
    Q0 = new double[Q->nDofsI];
    Wt = Alloc2D(W->nDofsJ, W->nDofsI);
    WtQ = Alloc2D(W->nDofsJ, Q->nDofsJ);

    Tran_IP(W->nDofsI, W->nDofsJ, W->A, Wt);

    initGZ();

    KT = NULL;
    GRADx = NULL;

    CreateCoupledOperator();
    /*
    M1inv = new Mat[geom->nk];
    VecSet(uz[0], 1.0);
    for(ii = 0; ii < geom->nk; ii++) {
        MatCreate(MPI_COMM_WORLD, &M1inv[ii]);
        MatSetSizes(M1inv[ii], topo->n1l, topo->n1l, topo->nDofs1G, topo->nDofs1G);
        MatSetType(M1inv[ii], MATMPIAIJ);
        MatMPIAIJSetPreallocation(M1inv[ii], 1, PETSC_NULL, 0, PETSC_NULL);
	MatZeroEntries(M1inv[ii]);

	M1->assemble(ii, SCALE, true);
	MatGetDiagonal(M1->M, uz[1]);
	VecPointwiseDivide(uz[1], uz[0], uz[1]);
        MatDiagonalSet(M1inv[ii], uz[1], INSERT_VALUES);
        MatAssemblyBegin(M1inv[ii], MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  M1inv[ii], MAT_FINAL_ASSEMBLY);
    }
    */

    KSPCreate(MPI_COMM_WORLD, &ksp_c);
    KSPSetOperators(ksp_c, M, M);
    KSPSetTolerances(ksp_c, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_c, KSPGMRES);
    KSPGetPC(ksp2, &pc);
    PCSetType(pc, PCJACOBI);
    KSPSetOptionsPrefix(ksp_c, "ksp_c_");
    KSPSetFromOptions(ksp_c);

    CE23M3 = NULL;
    CM2  = new M2mat_coupled(topo, geom, node, edge);
    CM3  = new M3mat_coupled(topo, geom, edge);
    CE32 = new E32_Coupled(topo);
    CK   = new Kmat_coupled(topo, geom, node, edge);

    n_dofs_locl = topo->nk*topo->n1l + (topo->nk-1)*topo->n2l;
    n_dofs_glob = topo->nk*topo->nDofs1G + (topo->nk-1)*topo->nDofs2G;
    VecCreateMPI(MPI_COMM_WORLD, n_dofs_locl, n_dofs_glob, &m2tmp1);
    VecCreateMPI(MPI_COMM_WORLD, n_dofs_locl, n_dofs_glob, &m2tmp2);
    CM2->assemble(SCALE, 1.0, NULL, NULL, true);
    MatGetDiagonal(CM2->M, m2tmp1);
    VecSet(m2tmp2, 1.0);
    VecPointwiseDivide(m2tmp1, m2tmp2, m2tmp1);
    MatCreate(MPI_COMM_WORLD, &CM2inv);
    MatSetSizes(CM2inv, n_dofs_locl, n_dofs_locl, n_dofs_glob, n_dofs_glob);
    MatSetType(CM2inv, MATMPIAIJ);
    MatMPIAIJSetPreallocation(CM2inv, 1, PETSC_NULL, 0, PETSC_NULL);
    MatZeroEntries(CM2inv);
    MatDiagonalSet(CM2inv, m2tmp1, INSERT_VALUES);
    MatAssemblyBegin(CM2inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  CM2inv, MAT_FINAL_ASSEMBLY);
    VecDestroy(&m2tmp1);
    VecDestroy(&m2tmp2);
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void Euler_I::coriolis() {
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
    VecScatterEnd(  topo->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);

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

void Euler_I::initGZ() {
    int ex, ey, ei, ii, kk, n2, mp12;
    int* inds0;
    int inds2k[99], inds0k[99];
    double* WtQflat = new double[W->nDofsJ*Q->nDofsJ];
    Vec gz;
    Mat GRAD, BQ;
    PetscScalar* zArray;
    MatReuse reuse;

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

            reuse = (!ei) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

            MatZeroEntries(BQ);
            for(kk = 0; kk < geom->nk; kk++) {
                for(ii = 0; ii < mp12; ii++) {
                    Q0[ii]  = Q->A[ii]*SCALE;
                    // for linear field we multiply by the vertical jacobian determinant when
                    // integrating, and do no other trasformations for the basis functions
                    Q0[ii] *= 0.5;
                }
                Mult_FD_IP(W->nDofsJ, Q->nDofsJ, W->nDofsI, Wt, Q0, WtQ);

                for(ii = 0; ii < W->nDofsJ; ii++) {
                    inds2k[ii] = ii + kk*W->nDofsJ;
                }

                // assemble the first basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+0)*mp12;
                }
                MatSetValues(BQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQ, ADD_VALUES);
                // assemble the second basis function
                for(ii = 0; ii < mp12; ii++) {
                    inds0k[ii] = ii + (kk+1)*mp12;
                }
                MatSetValues(BQ, W->nDofsJ, inds2k, Q->nDofsJ, inds0k, WtQ, ADD_VALUES);
            }
            MatAssemblyBegin(BQ, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(BQ, MAT_FINAL_ASSEMBLY);

            MatMatMult(vert->vo->V01, BQ, reuse, PETSC_DEFAULT, &GRAD);

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

Euler_I::~Euler_I() {
    int ii;

    delete[] Q0;
    Free2D(W->nDofsJ, Wt);
    Free2D(W->nDofsJ, WtQ);
    delete Q;
    delete W;

    KSPDestroy(&ksp1);
    KSPDestroy(&ksp2);

    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&fg[ii]);
        VecDestroy(&uil[ii]);
        VecDestroy(&ujl[ii]);
        VecDestroy(&uhl[ii]);
    }
    delete[] fg;
    delete[] uil;
    delete[] ujl;
    delete[] uhl;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&gv[ii]);
        VecDestroy(&zv[ii]);
    }
    delete[] gv;
    delete[] zv;
    for(ii = 0; ii < geom->nk-1; ii++) {
        VecDestroy(&uz[ii]);
        VecDestroy(&uzl_i[ii]);
        VecDestroy(&uzl_j[ii]);
    }
    delete[] uz;
    delete[] uzl_i;
    delete[] uzl_j;
    delete uuz;

    MatDestroy(&VA);
    MatDestroy(&VB);

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;
    delete T;
    delete M1t;
    delete Rh;
    delete Rz;

    if(KT) MatDestroy(&KT);

    delete edge;
    delete node;
    delete quad;

    //delete vert;

    VecDestroy(&x);
    VecDestroy(&dx);
    VecDestroy(&b);
    MatDestroy(&M);
    KSPDestroy(&ksp_c);
    delete M1c;
    delete Rc;
    delete M2c;
    delete EoSc;
    /*for(ii = 0; ii < topo->nk; ii++) {
        MatDestroy(&M1inv[ii]);
    }
    delete[] M1inv;*/

    delete CM2;
    delete CM3;
    delete CK;
    delete CE32;
    MatDestroy(&CM2inv);
}

// Take the weak form gradient of a 2 form scalar field as a 1 form vector field
void Euler_I::grad(bool assemble, Vec phi, Vec u, int lev) {
    Vec Mphi, dMphi;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mphi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dMphi);

    if(assemble) {
        M1->assemble(lev, SCALE, true);
        M2->assemble(lev, SCALE, true);
    }

    MatMult(M2->M, phi, Mphi);
    MatMult(EtoF->E12, Mphi, dMphi);
    KSPSolve(ksp1, dMphi, u);

    VecDestroy(&Mphi);
    VecDestroy(&dMphi);
}

// Take the weak form curl of a 1 form vector field as a 1 form vector field
void Euler_I::curl(bool assemble, Vec u, Vec* w, int lev, bool add_f) {
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

void Euler_I::init0(Vec* q, ICfunc3D* func) {
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
        VecScatterEnd(  topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

        m0->assemble(kk, 1.0);
        MatMult(PQ->M, bg, PQb);
        VecPointwiseDivide(q[kk], PQb, m0->vg);
    }

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void Euler_I::init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y) {
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

void Euler_I::init2(Vec* h, ICfunc3D* func) {
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

void Euler_I::initTheta(Vec theta, ICfunc3D* func) {
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

double Euler_I::int2(Vec ug) {
    int ex, ey, ei, ii, mp1, mp12;
    double det, uq, local, global;
    PetscScalar *array_2;

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

    return global;
}

void Euler_I::diagnostics(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner) {
    char filename[80];
    ofstream file;
    int kk, ex, ey, ei, n2;
    double keh, kev, pe, ie, k2p, p2k, mass;
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
    l2_rho->HorizToVert();

    // horiztonal kinetic energy
    for(kk = 0; kk < geom->nk; kk++) {
        F->assemble(l2_rho->vh[kk], kk, true, SCALE);
        MatMult(F->M, velx[kk], hu);
        VecScale(hu, 1.0/SCALE);
        VecDot(hu, velx[kk], &dot);
        keh += 0.5*dot;
    }

    // vertical kinetic energy and kinetic to potential exchange
    loc1 = loc2 = loc3 = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            MatZeroEntries(BA);
            vert->vo->AssembleConLinWithW(ex, ey, velz[ei], BA);
            MatMult(BA, velz[ei], w2);
            VecScale(w2, 1.0/SCALE);
            VecDot(l2_rho->vz[ei], w2, &dot);
            loc1 += 0.5*dot;

            vert->vo->AssembleLinearWithRT(ex, ey, l2_rho->vz[ei], VA, true);
            MatMult(VA, velz[ei], zi);
            vert->vo->AssembleLinearInv(ex, ey, VA);
            MatMult(VA, zi, gi);
            VecDot(gi, gv[ei], &dot);
            loc2 += dot/SCALE;

            MatMult(vert->vo->V10, gi, w2);
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

    // mass
    mass = 0.0;
    for(kk = 0; kk < geom->nk; kk++) {
        mass += int2(rho[kk]);
    }

    // vertical kinetic to internal energy power
    k2i_z = vert->k2i_z;
    k2i   = vert->horiz->k2i;
    i2k = i2k_z = 0.0;

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
        file << mass << "\t";
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

Vec* _CreateHorizVecs(Topo* topo, Geom* geom) {
    Vec* vecs = new Vec[geom->nk];

    for(int ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &vecs[ii]);
    }
    return vecs;
}

void _DestroyHorizVecs(Vec* vecs, Geom* geom) {
    for(int ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&vecs[ii]);
    }
    delete[] vecs;
}

// compute the potential vorticity components dUdz, dVdz
void Euler_I::HorizPotVort(Vec* velx, Vec* rho, Vec* uzl) {
    int ii, size;
    Vec* Mu = new Vec[geom->nk];
    Vec  du, rho_h;
    PC pc;
    KSP ksp1_t;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    KSPCreate(MPI_COMM_WORLD, &ksp1_t);
    // TODO: assemble M1t with density as 0.5(\rho_t + rho_b)
    KSPSetOperators(ksp1_t, M1t->M, M1t->M);
    KSPSetTolerances(ksp1_t, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp1_t, KSPGMRES);
    KSPGetPC(ksp1_t, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetOptionsPrefix(ksp1_t, "ksp1_t_");
    KSPSetFromOptions(ksp1_t);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho_h);
    for(ii = 0; ii < geom->nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu[ii]);
        M1->assemble(ii, SCALE, true);
        MatMult(M1->M, velx[ii], Mu[ii]);
    }

    for(ii = 0; ii < geom->nk-1; ii++) {
        VecZeroEntries(du);
        VecAXPY(du, +1.0, Mu[ii+1]);
        VecAXPY(du, -1.0, Mu[ii+0]);

        VecZeroEntries(rho_h);
        VecAXPY(rho_h, 0.5, rho[ii+0]);
        VecAXPY(rho_h, 0.5, rho[ii+1]);
        M1t->assemble_h(ii, SCALE, rho_h);

        KSPSolve(ksp1_t, du, uz[ii]);
        VecScatterBegin(topo->gtol_1, uz[ii], uzl[ii], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, uz[ii], uzl[ii], INSERT_VALUES, SCATTER_FORWARD);
    }

    VecDestroy(&du);
    VecDestroy(&rho_h);
    for(ii = 0; ii < geom->nk; ii++) {
        VecDestroy(&Mu[ii]);
    }
    delete[] Mu;
    KSPDestroy(&ksp1_t);
}

// compute the contribution of the vorticity vector to the vertical momentum equation
void Euler_I::AssembleVertMomVort(Vec* ul, L2Vecs* velz) {
    int kk;
    Vec _ul, ug, tmp, tmp1, dwdx;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &_ul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dwdx);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ug);

    for(kk = 0; kk < geom->nk-1; kk++) {
        VecZeroEntries(uuz->vh[kk]);
    }

    velz->VertToHoriz();
    for(kk = 0; kk < geom->nk-1; kk++) {
        // take the horizontal gradient of the vertical velocity
        // both matrices are piecewise linear in the vertical. as such they
        // should both be scaled by J_z = dz/2, however since the layer thicknesses
        // are constant for now we just omit these. These will need to be updated
        // for variable thickness layers.
        M1->assemble(kk, SCALE, false);
        M2->assemble(kk, SCALE, false);
        MatMult(M2->M, velz->vh[kk], tmp);
        MatMult(EtoF->E12, tmp, tmp1);
        KSPSolve(ksp1, tmp1, dwdx); // horizontal gradient of vertical velocity

        VecZeroEntries(_ul);
        VecAXPY(_ul, 0.5, ul[kk+0]);
        VecAXPY(_ul, 0.5, ul[kk+1]);
        Rz->assemble(_ul, SCALE);
        MatMult(Rz->M, dwdx, uuz->vh[kk]); // horizontal advection of vertical velocity
    }
    uuz->HorizToVert();

    VecDestroy(&_ul);
    VecDestroy(&ug);
    VecDestroy(&tmp);
    VecDestroy(&tmp1);
    VecDestroy(&dwdx);
}

void Euler_I::VertMassFlux(L2Vecs* velz1, L2Vecs* velz2, L2Vecs* rho1, L2Vecs* rho2, L2Vecs* Fz) {
    int ex, ey;

    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        ex = ii%topo->nElsX;
        ey = ii/topo->nElsX;

        vert->diagnose_F_z(ex, ey, velz1->vz[ii], velz2->vz[ii], rho1->vz[ii], rho2->vz[ii], Fz->vz[ii]);
    }
    Fz->VertToHoriz();
}

void Euler_I::CreateCoupledOperator() {
    int n_locl, n_glob, nnz;
    M1x_j_xy_i* U = new M1x_j_xy_i(node, edge);

    n_locl = topo->nk*topo->n1l + (4*topo->nk-1)*topo->n2l;
    n_glob = topo->nk*topo->nDofs1G + (4*topo->nk-1)*topo->nDofs2G;
    //nnz = 2*U->nDofsJ + 8*topo->nk*W->nDofsJ;
    //nnz = 2*U->nDofsJ + 4*topo->nk*W->nDofsJ;
    nnz = 4*U->nDofsJ + 4*topo->nk*W->nDofsJ;

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, n_locl, n_locl, n_glob, n_glob);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, nnz, PETSC_NULL, nnz/2, PETSC_NULL);
    MatSetOptionsPrefix(M, "mat_c_");
    MatSetFromOptions(M);

    if(!rank) cout << "coupled operator - global size: " << n_glob << endl;

    VecCreateMPI(MPI_COMM_WORLD, n_locl, n_glob, &x);
    VecCreateMPI(MPI_COMM_WORLD, n_locl, n_glob, &dx);
    VecCreateMPI(MPI_COMM_WORLD, n_locl, n_glob, &b);

    delete U;

    M1c  = new Umat_coupled(topo, geom, node, edge);
    Rc   = new RotMat_coupled(topo, geom, node, edge);
    M2c  = new Wmat_coupled(topo, geom, edge);
    EoSc = new EoSmat_coupled(topo, geom, edge);
}

void Euler_I::AssembleCoupledOperator(L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* velz, L2Vecs* theta) {
    int kk, ex, ey, ei, dp_size;
    Vec theta_h, d_pi, d_pi_l;
    Vec *d_pi_h, *d_pi_z;
    MatReuse reuse = (!GRADx) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    MatReuse reuse_kt = (!KT) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;
    MatReuse reuse_c = (!CE23M3) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    dp_size = (geom->nk-1)*topo->elOrd*topo->elOrd;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &d_pi);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &d_pi_l);
    d_pi_h = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &d_pi_h[kk]);
    }
    d_pi_z = new Vec[topo->nElsX*topo->nElsX];
    for(ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        VecCreateSeq(MPI_COMM_SELF, dp_size, &d_pi_z[ei]);
    }

    MatZeroEntries(M);

    CM2->assemble(SCALE, 1.0, NULL, NULL, true);
    CM3->assemble(SCALE, NULL, true, 1.0);
    AddM2_Coupled(topo, CM2->M, M);
    AddM3_Coupled(topo, 0, 0, CM3->M, M);
    AddM3_Coupled(topo, 1, 1, CM3->M, M);
AddM3_Coupled(topo, 2, 2, CM3->M, M);

    MatMatMult(CE32->MT, CM3->M, reuse_c, PETSC_DEFAULT, &CE23M3);
    MatMatMult(CM2inv, CE23M3, reuse_c, PETSC_DEFAULT, &CM2invE23M3);
    CM2->assemble(SCALE, 0.5*dt, theta->vh, theta->vz, false);
    MatMatMult(CM2->M, CM2invE23M3, reuse_c, PETSC_DEFAULT, &CGRAD);
    //AddG_Coupled(topo, 2, CGRAD, M);

    for(kk = 0; kk < geom->nk; kk++) {
        grad(true, exner->vh[kk], d_pi, kk);
        VecScatterBegin(topo->gtol_1, d_pi, d_pi_h[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, d_pi, d_pi_h[kk], INSERT_VALUES, SCATTER_FORWARD);
    }
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            vert->vo->AssembleConst(ex, ey, vert->vo->VB);
            MatMult(vert->vo->VB, exner->vz[ei], vert->_tmpB1);
            MatMult(vert->vo->V01, vert->_tmpB1, vert->_tmpA1);
            vert->vo->AssembleLinearInv(ex, ey, vert->vo->VA_inv);
            MatMult(vert->vo->VA_inv, vert->_tmpA1, d_pi_z[ei]);
	}
    }
    CM3->assemble_inv(SCALE, rho->vh, 0.5*dt);
    MatMatMult(CM3->Minv, CM3->M, reuse_c, PETSC_DEFAULT, &CM3invM3);
    CK->assemble(d_pi_h, d_pi_z, 1.0, SCALE);
    MatMatMult(CK->M, CM3invM3, reuse_c, PETSC_DEFAULT, &CGRAD);
    //AddG_Coupled(topo, 1, CGRAD, M);

    CM2->assemble(SCALE, 0.5*dt, rho->vh, rho->vz, true);
    MatMatMult(CM2inv, CM2->M, reuse_c, PETSC_DEFAULT, &CM2invM2);
    MatMatMult(CE32->M, CM2invM2, reuse_c, PETSC_DEFAULT, &CE32M2invM2);
    MatMatMult(CM3->M, CE32M2invM2, reuse_c, PETSC_DEFAULT, &CDIV);
    //AddD_Coupled(topo, 0, CDIV, M);

    CM3->assemble(SCALE, rt->vh, true, 0.5*dt);
    MatMatMult(CM3->M, CE32->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &CDIV);
    //AddD_Coupled(topo, 1, CDIV, M);

    CM3->assemble(SCALE, theta->vh, false, 0.5*dt);
    MatMatMult(CE32->MT, CM3->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &CE23M3);
    MatMatMult(CM2inv, CE23M3, MAT_REUSE_MATRIX, PETSC_DEFAULT, &CM2invE23M3);
    CK->assemble(ujl, velz->vz, 1.0, SCALE);
    MatTransposeMatMult(CK->M, CM2invE23M3, reuse_c, PETSC_DEFAULT, &CQ);
    //AddM3_Coupled(topo, 1, 0, CQ, M);


/*
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

	    vert->vo->AssembleLinear(ex, ey, vert->vo->VA);
            AddMz_Coupled(topo, ex, ey, 3, vert->vo->VA, M);
	    vert->vo->AssembleConst(ex, ey, vert->vo->VB);
            AddMz_Coupled(topo, ex, ey, 0, vert->vo->VB, M);
            AddMz_Coupled(topo, ex, ey, 1, vert->vo->VB, M);
AddMz_Coupled(topo, ex, ey, 2, vert->vo->VB, M);

            vert->assemble_operators(ex, ey, theta->vz[ei], rho->vz[ei], rt->vz[ei], exner->vz[ei], velz->vz[ei]);
            AddGradz_Coupled(topo, ex, ey, 1, vert->G_rt, M);
            AddGradz_Coupled(topo, ex, ey, 2, vert->G_pi, M);
            AddDivz_Coupled(topo, ex, ey, 0, vert->D_rho, M);
            AddDivz_Coupled(topo, ex, ey, 1, vert->D_rt, M);
            AddQz_Coupled(topo, ex, ey, vert->Q_rt_rho, M);
        }
    }
*/
    //M1c->assemble(SCALE, M);
    Rc->assemble(SCALE, 0.5*dt, vert->horiz->fl, M);
    //M2c->assemble(SCALE, 0, M);
    //M2c->assemble(SCALE, 1, M);
    //EoSc->assemble(SCALE, -1.0*RD/CV, 1, rt->vz, M);
    //EoSc->assemble(SCALE, +1.0, 2, exner->vz, M);
//M2c->assemble(SCALE, 2, M);
/*
    for(kk = 0; kk < topo->nk; kk++) {
	M1->assemble(kk, SCALE, true);
	M2->assemble(kk, SCALE, true);

	// G_pi_x block
        MatMatMult(EtoF->E12, M2->M, reuse, PETSC_DEFAULT, &GRADx);
        MatMatMult(M1inv[kk], GRADx, reuse, PETSC_DEFAULT, &M1invGRADx);
	VecZeroEntries(theta_h);
        VecAXPY(theta_h, 0.25*dt, theta->vh[kk+0]);
        VecAXPY(theta_h, 0.25*dt, theta->vh[kk+1]);
        F->assemble(theta_h, kk, false, SCALE);
        MatMatMult(F->M, M1invGRADx, reuse, PETSC_DEFAULT, &Gpi);
	//AddGradx_Coupled(topo, kk, 2, Gpi, M);

	// G_rt_x_block
        grad(false, exner->vh[kk], d_pi, kk);
        VecScatterBegin(topo->gtol_1, d_pi, d_pi_l, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, d_pi, d_pi_l, INSERT_VALUES, SCATTER_FORWARD);
        EoSc->assemble_rho_inv_mm(SCALE, 0.5*dt, kk, rho->vh[kk], T->M);
        K->assemble(d_pi_l, kk, SCALE);
        MatTranspose(K->M, reuse_kt, &KT);
        MatMatMult(KT, T->M, reuse, PETSC_DEFAULT, &Grt);
	//AddGradx_Coupled(topo, kk, 1, Grt, M);

	// Q_x block
	T->assemble(theta_h, kk, SCALE, false);
        MatMatMult(EtoF->E12, T->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &GRADx);
        MatMatMult(M1inv[kk], GRADx, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M1invGRADx);
        K->assemble(uil[kk], kk, SCALE);
        MatMatMult(K->M, M1invGRADx, reuse, PETSC_DEFAULT, &Qx);
	//AddQx_Coupled(topo, kk, Qx, M);

        // D_rho_x block
	VecCopy(rho->vh[kk], theta_h);
	VecScale(theta_h, 0.5*dt);
	F->assemble(theta_h, kk, true, SCALE);
        MatMatMult(M1inv[kk], F->M, reuse, PETSC_DEFAULT, &M1invM1);
        MatMatMult(EtoF->E21, M1invM1, reuse, PETSC_DEFAULT, &DM1invM1);
        MatMatMult(M2->M, DM1invM1, reuse, PETSC_DEFAULT, &Dx);
        //AddDivx_Coupled(topo, kk, 0, Dx, M);

        // D_rt_x block
	VecCopy(rt->vh[kk], theta_h);
	VecScale(theta_h, 0.5*dt);
	T->assemble(theta_h, kk, SCALE, true);
        MatMatMult(T->M, EtoF->E21, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Dx);
        //AddDivx_Coupled(topo, kk, 1, Dx, M);
    }
*/

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    VecDestroy(&theta_h);
    VecDestroy(&d_pi);
    VecDestroy(&d_pi_l);
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&d_pi_h[kk]);
    }
    delete[] d_pi_h;
    for(ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        VecDestroy(&d_pi_z[ei]);
    }
    delete[] d_pi_z;
}

void Euler_I::AssembleResidual(Vec* velx_i, Vec* velx_j,
                               L2Vecs* rho_i, L2Vecs* rho_j,
                               L2Vecs* rt_i, L2Vecs* rt_j,
                               L2Vecs* exner_i, L2Vecs* exner_j, L2Vecs* exner_h,
                               L2Vecs* velz_i, L2Vecs* velz_j, L2Vecs* theta_i, L2Vecs* theta_h,
                               L2Vecs* Fz, L2Vecs* dFx, L2Vecs* dGx, Vec* dwdx_i, Vec* dwdx_j, 
                               Vec* R_u, Vec* R_rho, Vec* R_rt, Vec* R_pi, Vec* R_w) 
{
    int ii, kk, ex, ey, elOrd2;
    Vec F_z, G_z, dF_z, dG_z, du, Mu;

    elOrd2 = topo->elOrd*topo->elOrd;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Mu);

    vert->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecAXPY(theta_h->vz[ii], 1.0, theta_i->vz[ii]);
        VecScale(theta_h->vz[ii], 0.5);
	VecZeroEntries(exner_h->vz[ii]);
	VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
	VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);
    }
    theta_h->VertToHoriz();
    exner_h->VertToHoriz();

    HorizPotVort(velx_j, rho_j->vh, uzl_j);
    vert->horiz->diagVertVort(velz_j->vh, rho_j->vh, dwdx_j);
    VertMassFlux(velz_i, velz_j, rho_i, rho_j, Fz);
    for(kk = 0; kk < topo->nk; kk++) {
	vert->horiz->momentum_rhs(kk, theta_h->vh, uzl_i, uzl_j, velz_i->vh, velz_j->vh, exner_h->vh[kk],
                                  velx_i[kk], velx_j[kk], uil[kk], ujl[kk], rho_i->vh[kk], rho_j->vh[kk], 
				  R_u[kk], Fz->vh, dwdx_i, dwdx_j);

	VecCopy(velx_j[kk], du);
	VecAXPY(du, -1.0, velx_i[kk]);
	MatMult(vert->horiz->M1->M, du, Mu);
	VecAYPX(R_u[kk], dt, Mu);
/*{
double norm;
VecNorm(R_u[kk],NORM_2,&norm);
if(!rank)cout<<kk<<"\t|R_u|: "<<norm<<endl;
}*/
    }

    AssembleVertMomVort(ujl, velz_j); // uuz TOOD: second order in time
    vert->horiz->advection_rhs(velx_i, velx_j, rho_i->vh, rho_j->vh, theta_h, dFx, dGx, uil, ujl, true);
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        ex = ii%topo->nElsX;
        ey = ii/topo->nElsX;

        // assemble the residual vectors
        vert->assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], 
			        velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii],
                                rt_i->vz[ii], rt_j->vz[ii], R_w[ii], F_z, G_z);

        VecAXPY(R_w[ii], dt, uuz->vz[ii]);
        vert->vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], R_pi[ii]);
        vert->vo->AssembleConst(ex, ey, vert->vo->VB);
        MatMult(vert->vo->V10, F_z, dF_z);
        MatMult(vert->vo->V10, G_z, dG_z);
        VecAYPX(dF_z, dt, rho_j->vz[ii]);
        VecAYPX(dG_z, dt, rt_j->vz[ii]);
        VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
        VecAXPY(dG_z, -1.0, rt_i->vz[ii] );

        // add the horizontal forcing
        VecAXPY(dF_z, dt, dFx->vz[ii]);
        VecAXPY(dG_z, dt, dGx->vz[ii]);

        MatMult(vert->vo->VB, dF_z, R_rho[ii]);
        MatMult(vert->vo->VB, dG_z, R_rt[ii]);
/*{
double norm[4];
VecNorm(R_rho[ii],NORM_2,&norm[0]);
VecNorm(R_rt[ii],NORM_2,&norm[1]);
VecNorm(R_pi[ii],NORM_2,&norm[2]);
VecNorm(R_w[ii],NORM_2,&norm[3]);
if(!rank)cout<<ii<<"\t|R_rho|: "<<norm[0]<<"\t"
	         <<"\t|R_rt|: "<<norm[1]<<"\t"
	         <<"\t|R_pi|: "<<norm[2]<<"\t"
	         <<"\t|R_w|: "<<norm[3]<<"\n";
}*/
    }
    topo->repack(R_u, R_rho, R_rt, R_pi, R_w, b);

    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
    VecDestroy(&du);
    VecDestroy(&Mu);
}

void Euler_I::Solve(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int ii, kk, elOrd2, it = 0;
    double norm_x, norm_dx, norm = 1.0e+9;
    Vec*    velx_j  = _CreateHorizVecs(topo, geom);
    Vec*    R_u     = _CreateHorizVecs(topo, geom);
    L2Vecs* velz_i  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_i   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_j   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_i    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_i = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* Fz      = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* dFx     = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* dGx     = new L2Vecs(geom->nk, topo, geom);
    Vec*    dwdx_i  = new Vec[geom->nk-1];
    Vec*    dwdx_j  = new Vec[geom->nk-1];
    Vec*    R_rho   = new Vec[topo->nElsX*topo->nElsX];
    Vec*    R_rt    = new Vec[topo->nElsX*topo->nElsX];
    Vec*    R_pi    = new Vec[topo->nElsX*topo->nElsX];
    Vec*    R_w     = new Vec[topo->nElsX*topo->nElsX];

    for(kk = 0; kk < geom->nk-1; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &dwdx_i[kk]);
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &dwdx_j[kk]);
    }
    for(kk = 0; kk < geom->nk; kk++) {
	VecCopy(velx[kk], velx_j[kk]);
        VecScatterBegin(topo->gtol_1, velx[kk], uil[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, velx[kk], uil[kk], INSERT_VALUES, SCATTER_FORWARD);
	VecCopy(uil[kk], ujl[kk]);
    }
    velz_i->CopyFromVert(velz);
    velz_j->CopyFromVert(velz);
    velz_i->VertToHoriz();
    velz_j->VertToHoriz();
    rho_i->CopyFromHoriz(rho);
    rho_j->CopyFromHoriz(rho);
    rho_i->HorizToVert();
    rho_j->HorizToVert();
    rt_i->CopyFromHoriz(rt);
    rt_j->CopyFromHoriz(rt);
    rt_i->HorizToVert();
    rt_j->HorizToVert();
    exner_i->CopyFromHoriz(exner);
    exner_j->CopyFromHoriz(exner);
    exner_h->CopyFromHoriz(exner);
    exner_i->HorizToVert();
    exner_j->HorizToVert();
    exner_h->HorizToVert();

    elOrd2 = topo->elOrd*topo->elOrd;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &R_rho[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &R_rt[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &R_pi[ii]);
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &R_w[ii]);
    }

    vert->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_i->VertToHoriz();

    HorizPotVort(velx, rho_i->vh, uzl_i);
    vert->horiz->diagVertVort(velz_i->vh, rho_i->vh, dwdx_i);

    topo->repack(velx, rho_i->vz, rt_i->vz, exner_i->vz, velz_i->vz, x);
    AssembleCoupledOperator(rho_i, rt_i, exner_i, velz_i, theta_i);

    do {
        // precondition....
        vert->solve_schur_vert(velz_i, velz_j, rho_i, rho_j, 
		               rt_i, rt_j, exner_i, exner_j, 
                               NULL, velx, velx_j, uil, ujl, false);

        AssembleResidual(velx, velx_j, rho_i, rho_j, rt_i, rt_j,
                         exner_i, exner_j, exner_h, velz_i, velz_j, 
			 theta_i, theta_h, Fz, dFx, dGx, dwdx_i, dwdx_j, 
                         R_u, R_rho, R_rt, R_pi, R_w);

	KSPSolve(ksp_c, b, dx);
	VecNorm(x, NORM_2, &norm_x);
	VecNorm(dx, NORM_2, &norm_dx);
	VecAXPY(x, -1.0, dx);
        topo->unpack(velx_j, rho_j->vz, rt_j->vz, exner_j->vz, velz_j->vz, x);
	rho_j->VertToHoriz();
	rt_j->VertToHoriz();
	exner_j->VertToHoriz();
	velz_j->VertToHoriz();
        for(kk = 0; kk < geom->nk; kk++) {
            VecScatterBegin(topo->gtol_1, velx_j[kk], ujl[kk], INSERT_VALUES, SCATTER_FORWARD);
            VecScatterEnd(  topo->gtol_1, velx_j[kk], ujl[kk], INSERT_VALUES, SCATTER_FORWARD);
        }

	norm = norm_dx/norm_x;
	if(!rank) {
            cout << scientific;
            cout << "iter: " << it << "\t|x|: " << norm_x << "\t|dx|: " << norm_dx << "\t|dx|/|x|: " << norm << endl;
        }
        it++;
    } while(norm > 1.0-14 && it < 20);

    _DestroyHorizVecs(velx_j, geom);
    _DestroyHorizVecs(R_u, geom);
    delete velz_i;
    delete velz_j;
    delete rho_i;
    delete rho_j;
    delete rt_i;
    delete rt_j;
    delete exner_i;
    delete exner_j;
    delete exner_h;
    delete theta_i;
    delete theta_h;
    delete Fz;
    delete dFx;
    delete dGx;
    for(kk = 0; kk < geom->nk-1; kk++) {
        VecDestroy(&dwdx_i[kk]);
        VecDestroy(&dwdx_j[kk]);
    }
    delete[] dwdx_i;
    delete[] dwdx_j;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        VecDestroy(&R_rho[ii]);
        VecDestroy(&R_rt[ii]);
        VecDestroy(&R_pi[ii]);
        VecDestroy(&R_w[ii]);
    }
    delete[] R_rho;
    delete[] R_rt;
    delete[] R_pi;
    delete[] R_w;
}
