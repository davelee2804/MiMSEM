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
//#define LAMBDA (1.0 + 0.5*sqrt(2.0))
#define LAMBDA 0.5

using namespace std;

Euler_I::Euler_I(Topo* _topo, Geom* _geom, double _dt) {
    int ii, n2, size;
    PC pc;

    dt = _dt;
    topo = _topo;
    geom = _geom;

    hs_forcing = false;
    step = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    quad = new GaussLobatto(geom->quad->n);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

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

    KT = NULL;
    GRADx = NULL;

    CreateCoupledOperator();

    KSPCreate(MPI_COMM_WORLD, &ksp_c);
    KSPSetOperators(ksp_c, M, M);
    KSPSetTolerances(ksp_c, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp_c, KSPGMRES);
    //KSPGetPC(ksp2, &pc);
    //PCSetType(pc, PCJACOBI);
    KSPSetOptionsPrefix(ksp_c, "ksp_c_");
    KSPSetFromOptions(ksp_c);

    CE23M3 = NULL;
    CM2  = new M2mat_coupled(topo, geom, node, edge);
    CM3  = new M3mat_coupled(topo, geom, edge);
    CE32 = new E32_Coupled(topo);
    CK   = new Kmat_coupled(topo, geom, node, edge);
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
        VecDestroy(&uil[ii]);
        VecDestroy(&ujl[ii]);
        VecDestroy(&uhl[ii]);
    }
    delete[] uil;
    delete[] ujl;
    delete[] uhl;
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

    delete CM2;
    delete CM3;
    delete CK;
    delete CE32;
    //MatDestroy(&CM2inv);
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

    loc02 = new int[2*geom->n0];
    VecCreateSeq(MPI_COMM_SELF, 2*geom->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, 2*geom->n0l, 2*geom->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &UQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bg);
        VecGetArray(bl, &bArray);

        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                inds0 = geom->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    bArray[2*inds0[ii]+0] = func_x(geom->x[inds0[ii]], kk);
                    bArray[2*inds0[ii]+1] = func_y(geom->x[inds0[ii]], kk);
                }
            }
        }
        VecRestoreArray(bl, &bArray);

        // create a new vec scatter object to handle vector quantity on nodes
        for(ii = 0; ii < geom->n0; ii++) {
            loc02[2*ii+0] = 2*geom->loc0[ii]+0;
            loc02[2*ii+1] = 2*geom->loc0[ii]+1;
        }
        ISCreateStride(MPI_COMM_WORLD, 2*geom->n0, 0, 1, &isl);
        ISCreateGeneral(MPI_COMM_WORLD, 2*geom->n0, loc02, PETSC_COPY_VALUES, &isg);
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

    VecCreateSeq(MPI_COMM_SELF, geom->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, geom->n0l, geom->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &WQb);

    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(bl);
        VecZeroEntries(bg);
        VecGetArray(bl, &bArray);

        for(ey = 0; ey < topo->nElsX; ey++) {
            for(ex = 0; ex < topo->nElsX; ex++) {
                inds0 = geom->elInds0_l(ex, ey);
                for(ii = 0; ii < mp12; ii++) {
                    bArray[inds0[ii]] = func(geom->x[inds0[ii]], kk);
                }
            }
        }
        VecRestoreArray(bl, &bArray);
        VecScatterBegin(geom->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(  geom->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

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
            VecDot(gi, vert->gv[ei], &dot);
            loc2 += dot/SCALE;

            MatMult(vert->vo->V10, gi, w2);
            VecDot(w2, vert->zv[ei], &dot);
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
            VecDot(vert->zv[ei], l2_rho->vz[ei], &dot);
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
    nnz = 2*U->nDofsJ + 4*topo->nk*W->nDofsJ;

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

void Euler_I::AssembleCoupledOperator(Vec* velx_i, Vec* velx_j, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* velz, L2Vecs* theta) {
    int kk, ex, ey, ei;
    Vec theta_h, d_pi, d_pi_l;
    Vec *d_pi_h, *d_pi_z, *wxl, wxg;
    MatReuse reuse_c = (!CE23M3) ? MAT_INITIAL_MATRIX : MAT_REUSE_MATRIX;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &theta_h);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &d_pi);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &d_pi_l);
    d_pi_h = new Vec[geom->nk];
    wxl    = new Vec[geom->nk];
    for(kk = 0; kk < geom->nk; kk++) {
        VecCreateSeq(MPI_COMM_SELF, topo->n1, &d_pi_h[kk]);
        VecCreateSeq(MPI_COMM_SELF, topo->n0, &wxl[kk]);
    }
    d_pi_z = new Vec[topo->nElsX*topo->nElsX];
    for(ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*topo->elOrd*topo->elOrd, &d_pi_z[ei]);
    }

    MatZeroEntries(M);

    CM2->assemble(SCALE, 1.0, NULL, NULL, true);
    CM2->assemble_inv(SCALE, M1);
    CM3->assemble(SCALE, NULL, true, 1.0);
    AddM2_Coupled(topo, CM2->M, M);
    AddM3_Coupled(topo, 0, 0, CM3->M, M);
    AddM3_Coupled(topo, 1, 1, CM3->M, M);

    MatMatMult(CE32->MT, CM3->M, reuse_c, PETSC_DEFAULT, &CE23M3);
    MatMatMult(CM2->Minv, CE23M3, reuse_c, PETSC_DEFAULT, &CM2invE23M3);
    CM2->assemble(SCALE, LAMBDA*dt, theta->vh, theta->vz, false);
    MatMatMult(CM2->M, CM2invE23M3, reuse_c, PETSC_DEFAULT, &CGRAD1);
    AddG_Coupled(topo, 2, CGRAD1, M);

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
    CM3->assemble_inv(SCALE, rho->vh);
    MatMatMult(CM3->Minv, CM3->M, reuse_c, PETSC_DEFAULT, &CM3invM3);
    CK->assemble(d_pi_h, d_pi_z, LAMBDA*dt, SCALE);
    MatMatMult(CK->M, CM3invM3, reuse_c, PETSC_DEFAULT, &CGRAD2);
    AddG_Coupled(topo, 1, CGRAD2, M);

    CM2->assemble(SCALE, LAMBDA*dt, rho->vh, rho->vz, true);
    MatMatMult(CM2->Minv, CM2->M, reuse_c, PETSC_DEFAULT, &CM2invM2);
    MatMatMult(CE32->M, CM2invM2, reuse_c, PETSC_DEFAULT, &CE32M2invM2);
    MatMatMult(CM3->M, CE32M2invM2, reuse_c, PETSC_DEFAULT, &CDIV1);
    AddD_Coupled(topo, 0, CDIV1, M); // ??

    CM3->assemble(SCALE, rt->vh, true, LAMBDA*dt);
    MatMatMult(CM3->M, CE32->M, reuse_c, PETSC_DEFAULT, &CDIV2);
    //AddD_Coupled(topo, 1, CDIV2, M); // ??

    CM3->assemble(SCALE, theta->vh, false, 1.0);
    MatMatMult(CE32->MT, CM3->M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &CE23M3);
    MatMatMult(CM2->Minv, CE23M3, MAT_REUSE_MATRIX, PETSC_DEFAULT, &CM2invE23M3);
    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(d_pi_h[kk]);
	VecAXPY(d_pi_h[kk], 0.5, uil[kk]);
	VecAXPY(d_pi_h[kk], 0.5, ujl[kk]);
    }
    CK->assemble(d_pi_h, velz->vz, LAMBDA*dt, SCALE);
    MatMatMult(CK->MT, CM2invE23M3, reuse_c, PETSC_DEFAULT, &CQ);
    AddM3_Coupled(topo, 1, 0, CQ, M);

    MatMatMult(CE32->MT, CK->MT, reuse_c, PETSC_DEFAULT, &CE23K);
    AddM2_Coupled(topo, CE23K, M);

    //Rc->assemble(SCALE, LAMBDA*dt, vert->horiz->fl, M);
    for(kk = 0; kk < geom->nk; kk++) {
        VecZeroEntries(d_pi);
	VecAXPY(d_pi, 0.5, velx_i[kk]);
	VecAXPY(d_pi, 0.5, velx_j[kk]);
        vert->horiz->curl(true, d_pi, &wxg, kk, true);
        VecScatterBegin(topo->gtol_0, wxg, wxl[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_0, wxg, wxl[kk], INSERT_VALUES, SCATTER_FORWARD);
	VecDestroy(&wxg);
    }
    Rc->assemble(SCALE, LAMBDA*dt, wxl, M);
    EoSc->assemble(SCALE, -1.0*RD/CV, 1, rt->vz, M);
    EoSc->assemble(SCALE, +1.0, 2, exner->vz, M);
/*
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

            vert->assemble_operators(ex, ey, theta->vz[ei], rho->vz[ei], rt->vz[ei], exner->vz[ei], velz->vz[ei]);
            AddGradz_Coupled(topo, ex, ey, 1, vert->G_rt, M);
            AddQz_Coupled(topo, ex, ey, 1, 0, vert->Q_rt_rho, M);
            AddQz_Coupled(topo, ex, ey, 2, 1, vert->N_rt, M);
            AddQz_Coupled(topo, ex, ey, 2, 2, vert->N_pi, M);
        }
    }
*/
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    VecDestroy(&theta_h);
    VecDestroy(&d_pi);
    VecDestroy(&d_pi_l);
    for(kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&d_pi_h[kk]);
        VecDestroy(&wxl[kk]);
    }
    delete[] d_pi_h;
    delete[] wxl;
    for(ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        VecDestroy(&d_pi_z[ei]);
    }
    delete[] d_pi_z;
}

#if 0
void Euler_I::AssembleCoupledOperator(L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* velz, L2Vecs* theta) {
    int ex, ey, ei;

    MatZeroEntries(M);

    CM2->assemble(SCALE, 1.0, NULL, NULL, true);
    AddM2_Coupled(topo, CM2->M, M);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;

	    vert->vo->AssembleLinear(ex, ey, vert->vo->VA);
            //AddMz_Coupled(topo, ex, ey, 3, vert->vo->VA, M);
	    vert->vo->AssembleConst(ex, ey, vert->vo->VB);
            AddMz_Coupled(topo, ex, ey, 0, vert->vo->VB, M);
            AddMz_Coupled(topo, ex, ey, 1, vert->vo->VB, M);

            vert->assemble_operators(ex, ey, theta->vz[ei], rho->vz[ei], rt->vz[ei], exner->vz[ei], velz->vz[ei]);
            AddGradz_Coupled(topo, ex, ey, 1, vert->G_rt, M);
            AddGradz_Coupled(topo, ex, ey, 2, vert->G_pi, M);
            AddDivz_Coupled(topo, ex, ey, 0, vert->D_rho, M);
            //AddDivz_Coupled(topo, ex, ey, 1, vert->D_rt, M); // ??
            AddQz_Coupled(topo, ex, ey, 1, 0, vert->Q_rt_rho, M);
            AddQz_Coupled(topo, ex, ey, 2, 1, vert->N_rt, M);
            AddQz_Coupled(topo, ex, ey, 2, 2, vert->N_pi, M);
        }
    }
    //EoSc->assemble(SCALE, -1.0*RD/CV, 1, rt->vz, M);
    //EoSc->assemble(SCALE, +1.0, 2, exner->vz, M);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);
}
#endif

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
    vert->horiz->advection_rhs(velx_i, velx_j, rho_i->vh, rho_j->vh, theta_h, dFx, dGx, uil, ujl);
    for(kk = 0; kk < topo->nk; kk++) {
	vert->horiz->momentum_rhs(kk, theta_h->vh, uzl_i, uzl_j, velz_i->vh, velz_j->vh, exner_h->vh[kk],
                                  velx_i[kk], velx_j[kk], uil[kk], ujl[kk], rho_i->vh[kk], rho_j->vh[kk], 
				  R_u[kk], vert->horiz->Fk[kk], Fz->vh, dwdx_i, dwdx_j);

/*{
double norm;
VecNorm(R_u[kk],NORM_2,&norm);
if(!rank)cout<<kk<<"\t|R_u|: "<<dt*norm<<endl;
}*/
	VecCopy(velx_j[kk], du);
	VecAXPY(du, -1.0, velx_i[kk]);
	MatMult(vert->horiz->M1->M, du, Mu);
	VecAYPX(R_u[kk], dt, Mu);
//VecAYPX(R_u[kk], 0.0*dt, Mu);
//VecCopy(du, R_u[kk]);
//VecSet(R_u[kk],0.0);
    }

    AssembleVertMomVort(ujl, velz_j); // uuz TOOD: second order in time
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        ex = ii%topo->nElsX;
        ey = ii/topo->nElsX;

        // assemble the residual vectors
        vert->assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], 
			        velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii],
                                rt_i->vz[ii], rt_j->vz[ii], R_w[ii], F_z, G_z);

        //VecAXPY(R_w[ii], dt, uuz->vz[ii]);
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
if(!rank && !ii)cout<<ii<<"\t|R_rho|: "<<norm[0]<<"\t"
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
    L2Vecs* velz_h  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_h   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h    = new L2Vecs(geom->nk, topo, geom);
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
    velz_h->CopyFromVert(velz);
    velz_i->VertToHoriz();
    velz_j->VertToHoriz();
    velz_h->VertToHoriz();
    rho_i->CopyFromHoriz(rho);
    rho_j->CopyFromHoriz(rho);
    rho_h->CopyFromHoriz(rho);
    rho_i->HorizToVert();
    rho_j->HorizToVert();
    rho_h->HorizToVert();
    rt_i->CopyFromHoriz(rt);
    rt_j->CopyFromHoriz(rt);
    rt_h->CopyFromHoriz(rt);
    rt_i->HorizToVert();
    rt_j->HorizToVert();
    rt_h->HorizToVert();
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
    theta_h->CopyFromHoriz(theta_i->vh);
    theta_h->HorizToVert();

    HorizPotVort(velx, rho_i->vh, uzl_i);
    vert->horiz->diagVertVort(velz_i->vh, rho_i->vh, dwdx_i);

    topo->repack(velx, rho_i->vz, rt_i->vz, exner_i->vz, velz_i->vz, x);

    do {
        // precondition....
        /*if(!it)*/ vert->solve_schur_vert(velz_i, velz_j, velz_h, rho_i, rho_j, rho_h, 
                               rt_i, rt_j, rt_h, exner_i, exner_j, exner_h, 
                               theta_i, theta_h, NULL, velx, velx_j, uil, ujl, false);

        AssembleResidual(velx, velx_j, rho_i, rho_j, rt_i, rt_j,
                         exner_i, exner_j, exner_h, velz_i, velz_j, 
			 theta_i, theta_h, Fz, dFx, dGx, dwdx_i, dwdx_j, 
                         R_u, R_rho, R_rt, R_pi, R_w);

        for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
	    VecZeroEntries(rho_h->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_i->vz[ii]);
            VecAXPY(rho_h->vz[ii], 0.5, rho_j->vz[ii]);
            VecZeroEntries(rt_h->vz[ii]);
            VecAXPY(rt_h->vz[ii], 0.5, rt_i->vz[ii]);
            VecAXPY(rt_h->vz[ii], 0.5, rt_j->vz[ii]);
            VecZeroEntries(velz_h->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_i->vz[ii]);
            VecAXPY(velz_h->vz[ii], 0.5, velz_j->vz[ii]);
        }
        rho_h->VertToHoriz();
        rt_h->VertToHoriz();
        velz_h->VertToHoriz();
        AssembleCoupledOperator(velx, velx_j, rho_h, rt_h, exner_h, velz_h, theta_h);

	VecScale(b, -1.0);
	KSPSolve(ksp_c, b, dx);
	VecNorm(x, NORM_2, &norm_x);
	VecNorm(dx, NORM_2, &norm_dx);
	VecAXPY(x, +1.0, dx);
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
    } while(norm > 1.0e-14 && it < 40);

    for(kk = 0; kk < topo->nk; kk++) {
        VecCopy(velx_j[kk], velx[kk]);
    }
    velz_j->CopyToVert(velz);
    rho_j->CopyToHoriz(rho);
    rt_j->CopyToHoriz(rt);
    exner_j->CopyToHoriz(exner);

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
    delete velz_h;
    delete rho_h;
    delete rt_h;
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

PetscErrorCode _snes_function(SNES snes, Vec x, Vec r, void* _ctx) {
    int ii, kk;
    euler_ctx *ctx = (euler_ctx*)_ctx;

    ctx->eul->topo->unpack(ctx->velx_j, ctx->rho_j->vz, ctx->rt_j->vz, ctx->exner_j->vz, ctx->velz_j->vz, ctx->eul->x);
    ctx->rho_j->VertToHoriz();
    ctx->rt_j->VertToHoriz();
    ctx->exner_j->VertToHoriz();
    ctx->velz_j->VertToHoriz();
    for(kk = 0; kk < ctx->eul->topo->nk; kk++) {
        VecScatterBegin(ctx->eul->topo->gtol_1, ctx->velx_j[kk], ctx->ujl[kk], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  ctx->eul->topo->gtol_1, ctx->velx_j[kk], ctx->ujl[kk], INSERT_VALUES, SCATTER_FORWARD);
    }

    ctx->eul->vert->diagTheta2(ctx->rho_j->vz, ctx->rt_j->vz, ctx->theta_h->vz);
    for(ii = 0; ii < ctx->eul->topo->nElsX*ctx->eul->topo->nElsX; ii++) {
        VecAXPY(ctx->theta_h->vz[ii], 1.0, ctx->theta_i->vz[ii]);
        VecScale(ctx->theta_h->vz[ii], 0.5);
	VecZeroEntries(ctx->exner_h->vz[ii]);
	VecAXPY(ctx->exner_h->vz[ii], 0.5, ctx->exner_i->vz[ii]);
	VecAXPY(ctx->exner_h->vz[ii], 0.5, ctx->exner_j->vz[ii]);
	VecZeroEntries(ctx->rho_h->vz[ii]);
	VecAXPY(ctx->rho_h->vz[ii], 0.5, ctx->rho_i->vz[ii]);
	VecAXPY(ctx->rho_h->vz[ii], 0.5, ctx->rho_j->vz[ii]);
	VecZeroEntries(ctx->rt_h->vz[ii]);
	VecAXPY(ctx->rt_h->vz[ii], 0.5, ctx->rt_i->vz[ii]);
	VecAXPY(ctx->rt_h->vz[ii], 0.5, ctx->rt_j->vz[ii]);
	VecZeroEntries(ctx->velz_h->vz[ii]);
	VecAXPY(ctx->velz_h->vz[ii], 0.5, ctx->velz_i->vz[ii]);
	VecAXPY(ctx->velz_h->vz[ii], 0.5, ctx->velz_j->vz[ii]);
    }
    ctx->theta_h->VertToHoriz();
    ctx->exner_h->VertToHoriz();
    ctx->rho_h->VertToHoriz();
    ctx->rt_h->VertToHoriz();
    ctx->velz_h->VertToHoriz();

    ctx->eul->AssembleResidual(ctx->velx_i, ctx->velx_j, ctx->rho_i, ctx->rho_j, ctx->rt_i, ctx->rt_j,
                               ctx->exner_i, ctx->exner_j, ctx->exner_h, ctx->velz_i, ctx->velz_j, 
			       ctx->theta_i, ctx->theta_h, ctx->Fz, ctx->dFx, ctx->dGx, ctx->dwdx_i, ctx->dwdx_j, 
                               ctx->R_u, ctx->R_rho, ctx->R_rt, ctx->R_exner, ctx->R_w);

    VecScale(r, 1.0e-10);

    return 0;
}

PetscErrorCode _snes_jacobian(SNES snes, Vec x, Mat J, Mat P, void* _ctx) {
    int ii;
    euler_ctx *ctx = (euler_ctx*)_ctx;

    ctx->eul->topo->unpack(ctx->velx_j, ctx->rho_j->vz, ctx->rt_j->vz, ctx->exner_j->vz, ctx->velz_j->vz, ctx->eul->x);
    ctx->rho_j->VertToHoriz();
    ctx->rt_j->VertToHoriz();
    ctx->exner_j->VertToHoriz();
    ctx->velz_j->VertToHoriz();

    ctx->eul->vert->diagTheta2(ctx->rho_j->vz, ctx->rt_j->vz, ctx->theta_h->vz);
    for(ii = 0; ii < ctx->eul->topo->nElsX*ctx->eul->topo->nElsX; ii++) {
        VecAXPY(ctx->theta_h->vz[ii], 1.0, ctx->theta_i->vz[ii]);
        VecScale(ctx->theta_h->vz[ii], 0.5);
	VecZeroEntries(ctx->exner_h->vz[ii]);
	VecAXPY(ctx->exner_h->vz[ii], 0.5, ctx->exner_i->vz[ii]);
	VecAXPY(ctx->exner_h->vz[ii], 0.5, ctx->exner_j->vz[ii]);
	VecZeroEntries(ctx->rho_h->vz[ii]);
	VecAXPY(ctx->rho_h->vz[ii], 0.5, ctx->rho_i->vz[ii]);
	VecAXPY(ctx->rho_h->vz[ii], 0.5, ctx->rho_j->vz[ii]);
	VecZeroEntries(ctx->rt_h->vz[ii]);
	VecAXPY(ctx->rt_h->vz[ii], 0.5, ctx->rt_i->vz[ii]);
	VecAXPY(ctx->rt_h->vz[ii], 0.5, ctx->rt_j->vz[ii]);
	VecZeroEntries(ctx->velz_h->vz[ii]);
	VecAXPY(ctx->velz_h->vz[ii], 0.5, ctx->velz_i->vz[ii]);
	VecAXPY(ctx->velz_h->vz[ii], 0.5, ctx->velz_j->vz[ii]);
    }
    ctx->theta_h->VertToHoriz();
    ctx->exner_h->VertToHoriz();
    ctx->rho_h->VertToHoriz();
    ctx->rt_h->VertToHoriz();
    ctx->velz_h->VertToHoriz();

    ctx->eul->AssembleCoupledOperator(ctx->velx_i, ctx->velx_j, ctx->rho_h, ctx->rt_h, ctx->exner_h, ctx->velz_h, ctx->theta_h);

    MatScale(J, 1.0e-10);

    return 0;
}
void Euler_I::Solve_SNES(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save) {
    int ii, kk, elOrd2;
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
    L2Vecs* velz_h  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_h   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h    = new L2Vecs(geom->nk, topo, geom);
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
    SNES           snes;
    SNESLineSearch linesearch;
    euler_ctx      ctx;

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
    velz_h->CopyFromVert(velz);
    velz_i->VertToHoriz();
    velz_j->VertToHoriz();
    velz_h->VertToHoriz();
    rho_i->CopyFromHoriz(rho);
    rho_j->CopyFromHoriz(rho);
    rho_h->CopyFromHoriz(rho);
    rho_i->HorizToVert();
    rho_j->HorizToVert();
    rho_h->HorizToVert();
    rt_i->CopyFromHoriz(rt);
    rt_j->CopyFromHoriz(rt);
    rt_h->CopyFromHoriz(rt);
    rt_i->HorizToVert();
    rt_j->HorizToVert();
    rt_h->HorizToVert();
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
    theta_h->CopyFromHoriz(theta_i->vh);
    theta_h->HorizToVert();

    HorizPotVort(velx, rho_i->vh, uzl_i);
    vert->horiz->diagVertVort(velz_i->vh, rho_i->vh, dwdx_i);

    ctx.eul     = this;
    ctx.velx_i  = velx;
    ctx.velx_j  = velx_j;
    ctx.uil     = uil;
    ctx.ujl     = ujl;
    ctx.rho_i   = rho_i;
    ctx.rho_j   = rho_j;
    ctx.rho_h   = rho_h;
    ctx.rt_i    = rt_i;
    ctx.rt_j    = rt_j;
    ctx.rt_h    = rt_h;
    ctx.exner_i = exner_i;
    ctx.exner_j = exner_j;
    ctx.exner_h = exner_h;
    ctx.velz_i  = velz_i;
    ctx.velz_j  = velz_j;
    ctx.velz_h  = velz_h;
    ctx.theta_i = theta_i;
    ctx.theta_h = theta_h;
    ctx.Fz      = Fz;
    ctx.dFx     = dFx;
    ctx.dGx     = dGx;
    ctx.dwdx_i  = dwdx_i;
    ctx.dwdx_j  = dwdx_j;
    ctx.R_u     = R_u;
    ctx.R_rho   = R_rho;
    ctx.R_rt    = R_rt;
    ctx.R_exner = R_pi;
    ctx.R_w     = R_w;
    SNESCreate(MPI_COMM_WORLD, &snes);
    SNESSetFunction(snes, b, _snes_function, (void*)&ctx);
    SNESSetJacobian(snes, M, M, _snes_jacobian, (void*)&ctx);
    SNESGetLineSearch(snes, &linesearch);
    SNESLineSearchSetType(linesearch, SNESLINESEARCHBT);
    SNESSetFromOptions(snes);

    // precondition....
    vert->solve_schur_vert(velz_i, velz_j, velz_h, rho_i, rho_j, rho_h, 
                           rt_i, rt_j, rt_h, exner_i, exner_j, exner_h, 
                           theta_i, theta_h, NULL, velx, velx_j, uil, ujl, false);

    topo->repack(velx_j, rho_j->vz, rt_j->vz, exner_j->vz, velz_j->vz, x);
    SNESSolve(snes, NULL, x);
    topo->unpack(velx, rho_j->vz, rt_j->vz, exner_j->vz, velz, x);

    rho_j->VertToHoriz();
    rho_j->CopyToHoriz(rho);
    rt_j->VertToHoriz();
    rt_j->CopyToHoriz(rt);
    exner_j->VertToHoriz();
    exner_j->CopyToHoriz(exner);


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
    delete velz_h;
    delete rho_h;
    delete rt_h;
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
    SNESDestroy(&snes);
}
