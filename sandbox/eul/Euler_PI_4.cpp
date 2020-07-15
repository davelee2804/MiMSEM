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
#include "Schur.h"
#include "VertSolve_4.h"
#include "HorizSolve_4.h"
#include "Euler_PI_4.h"

#define GRAVITY 9.80616
#define SCALE 1.0e+8
#define RD 287.0
#define CP 1004.5
#define CV 717.5
#define P0 100000.0
#define SCALE 1.0e+8
#define RAYLEIGH 0.4
//#define NEW_EOS 1

using namespace std;

Euler::Euler(Topo* _topo, Geom* _geom, double _dt) {
    dt = _dt;
    topo = _topo;
    geom = _geom;

    step = 0;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    vert  = new VertSolve(topo, geom, dt);
    horiz = new HorizSolve(topo, geom, dt);
    schur = new Schur(topo, geom);
    schur_rho = new Schur(topo, geom);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

Euler::~Euler() {
    //delete schur;
    //delete schur_rho;
    delete vert;
    delete horiz;
}

void Euler::dump(Vec* velx, L2Vecs* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* theta, int num) {
    char fieldname[100];
    Vec wi;

    if(!rank) cout << "dumping output for step: " << num << endl;

    theta->UpdateGlobal();
    for(int ii = 0; ii < geom->nk+1; ii++) {
        sprintf(fieldname, "theta");
        geom->write2(theta->vh[ii], fieldname, num, ii, false);
    }
    for(int ii = 0; ii < geom->nk; ii++) {
        if(velx) horiz->curl(true, velx[ii], &wi, ii, false);

        if(velx) sprintf(fieldname, "vorticity");
        if(velx) geom->write0(wi, fieldname, num, ii);
        if(velx) sprintf(fieldname, "velocity_h");
        if(velx) geom->write1(velx[ii], fieldname, num, ii);
        sprintf(fieldname, "density");
        geom->write2(rho->vh[ii], fieldname, num, ii, true);
        sprintf(fieldname, "rhoTheta");
        geom->write2(rt->vh[ii], fieldname, num, ii, true);
        sprintf(fieldname, "exner");
        geom->write2(exner->vh[ii], fieldname, num, ii, true);

        if(velx) VecDestroy(&wi);
    }
    if(velz) {
        sprintf(fieldname, "velocity_z");
        for(int ii = 0; ii < geom->nk-1; ii++) {
            geom->write2(velz->vh[ii], fieldname, num, ii, false);
        }
    }
}

double MaxNorm(Vec dx, Vec x, double max_norm) {
    double norm_dx, norm_x, new_max_norm;

    VecNorm(dx, NORM_2, &norm_dx);
    VecNorm(x, NORM_2, &norm_x);
    new_max_norm = (norm_dx/norm_x > max_norm) ? norm_dx/norm_x : max_norm;
    return new_max_norm;
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
        VecScatterEnd(  scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

        horiz->M1->assemble(kk, SCALE, true);
        MatMult(UQ->M, bg, UQb);
        VecScale(UQb, SCALE);
        KSPSolve(horiz->ksp1, UQb, u[kk]);

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
        VecScatterEnd(  topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

        MatMult(WQ->M, bg, WQb);
        VecScale(WQb, SCALE);          // have to rescale the M2 operator as the metric terms scale
        horiz->M2->assemble(kk, SCALE, true); // this down to machine precision, so rescale the rhs as well
        KSPSolve(horiz->ksp2, WQb, h[kk]);
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
    VecScatterEnd(  topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    horiz->M2->assemble(0, SCALE, false);
    MatMult(WQ->M, bg, WQb);
    VecScale(WQb, SCALE);
    KSPSolve(horiz->ksp2, WQb, theta);

    delete WQ;
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

void Euler::GlobalNorms(int itt, Vec* duh, Vec* uh, L2Vecs* duz, L2Vecs* uz, L2Vecs* drho, L2Vecs* rho, L2Vecs* drt, L2Vecs* rt, L2Vecs* dexner, L2Vecs* exner,
                        double* norm_u, double* norm_w, double* norm_rho, double* norm_rt, double* norm_exner, Vec h_tmp, Vec u_tmp, Vec u_tmp_z, bool prnt) {
    double norm_x_rho, norm_dx_rho, norm_x_rt, norm_dx_rt, norm_x_exner, norm_dx_exner, norm_dx_u, norm_x_u, dot_x, dot_dx;

    norm_x_rho = norm_dx_rho = 0.0;
    norm_x_rt = norm_dx_rt = 0.0;
    norm_x_exner = norm_dx_exner = 0.0;
    for(int kk = 0; kk < geom->nk; kk++) {
        horiz->M2->assemble(kk, SCALE, true);

        // density
        MatMult(horiz->M2->M, drho->vh[kk], h_tmp);
        VecDot(drho->vh[kk], h_tmp, &dot_dx);
        MatMult(horiz->M2->M, rho->vh[kk], h_tmp);
        VecDot(rho->vh[kk], h_tmp, &dot_x);
        norm_x_rho += dot_x;
        norm_dx_rho += dot_dx;

        // density weighted potential temperature
        MatMult(horiz->M2->M, drt->vh[kk], h_tmp);
        VecDot(drt->vh[kk], h_tmp, &dot_dx);
        MatMult(horiz->M2->M, rt->vh[kk], h_tmp);
        VecDot(rt->vh[kk], h_tmp, &dot_x);
        norm_x_rt += dot_x;
        norm_dx_rt += dot_dx;

        // exner
        MatMult(horiz->M2->M, dexner->vh[kk], h_tmp);
        VecDot(dexner->vh[kk], h_tmp, &dot_dx);
        MatMult(horiz->M2->M, exner->vh[kk], h_tmp);
        VecDot(exner->vh[kk], h_tmp, &dot_x);
        norm_x_exner += dot_x;
        norm_dx_exner += dot_dx;
    }
    *norm_rho = sqrt(norm_dx_rho/norm_x_rho);
    *norm_rt = sqrt(norm_dx_rt/norm_x_rt);
    *norm_exner = sqrt(norm_dx_exner/norm_x_exner);

    // vertical velocity
    norm_x_u = norm_dx_u = 0.0;
    for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        vert->vo->AssembleLinear(ii%topo->nElsX, ii/topo->nElsX, vert->vo->VA);
        MatMult(vert->vo->VA, duz->vz[ii], u_tmp_z);
        VecDot(duz->vz[ii], u_tmp_z, &dot_dx);
        MatMult(vert->vo->VA, uz->vz[ii], u_tmp_z);
        VecDot(uz->vz[ii], u_tmp_z, &dot_x);
        norm_x_u += dot_x;
        norm_dx_u += dot_dx;
    }
    MPI_Allreduce(&norm_x_u, &dot_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&norm_dx_u, &dot_dx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    *norm_w = sqrt(dot_dx/dot_x);

    // horizontal velocity
    norm_x_u = norm_dx_u = 0.0;
    for(int kk = 0; kk < geom->nk; kk++) {
        horiz->M1->assemble(kk, SCALE, true);
        MatMult(horiz->M1->M, duh[kk], u_tmp);
        VecDot(duh[kk], u_tmp, &dot_dx);
        MatMult(horiz->M1->M, uh[kk], u_tmp);
        VecDot(uh[kk], u_tmp, &dot_x);
        norm_x_u += dot_x;
        norm_dx_u += dot_dx;
    }
    *norm_u = sqrt(norm_dx_u/norm_x_u);

    if(!rank && prnt) cout << itt << ":\t|d_exner|/|exner|: " << *norm_exner << 
                              "\t|d_rho|/|rho|: "     << *norm_rho   <<
                              "\t|d_rt|/|rt|: "       << *norm_rt    <<
                              "\t|d_u|/|u|: "         << *norm_u     <<
                              "\t|d_w|/|w|: "         << *norm_w     << endl;
}

void Euler::solve_schur(Vec* velx, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_w, max_norm_rho, max_norm_rt, max_norm_exner;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* velz_h  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_h   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* F_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* F_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_exner = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* d_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_exner = new L2Vecs(geom->nk, topo, geom);
    L1Vecs* velx_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* velx_j  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* velx_h  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_j  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* F_u     = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* d_u     = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* gradPi  = new L1Vecs(geom->nk, topo, geom);
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta, u_tmp_l;
    VertOps* vo = vert->vo;

    velx_i->CopyFrom(velx);
    velx_j->CopyFrom(velx);
    velx_h->CopyFrom(velx);
    velx_i->UpdateLocal();
    velx_j->UpdateLocal();
    velx_h->UpdateLocal();

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dG);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_2);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &u_tmp_l);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);

    velz_i->UpdateLocal();  velz_i->HorizToVert();
    rho_i->UpdateLocal();   rho_i->HorizToVert();
    rt_i->UpdateLocal();    rt_i->HorizToVert();
    exner_i->UpdateLocal(); exner_i->HorizToVert();
    velz_j->CopyFromVert(velz_i->vz);   velz_j->VertToHoriz();  velz_j->UpdateGlobal();
    rho_j->CopyFromVert(rho_i->vz);     rho_j->VertToHoriz();   rho_j->UpdateGlobal();
    rt_j->CopyFromVert(rt_i->vz);       rt_j->VertToHoriz();    rt_j->UpdateGlobal();
    exner_j->CopyFromVert(exner_i->vz); exner_j->VertToHoriz(); exner_j->UpdateGlobal();
    exner_h->CopyFromVert(exner_i->vz); exner_h->VertToHoriz(); exner_h->UpdateGlobal();
    rho_h->CopyFromVert(rho_i->vz); rho_h->VertToHoriz(); rho_h->UpdateGlobal();
    rt_h->CopyFromVert(rt_i->vz); rt_h->VertToHoriz(); rt_h->UpdateGlobal();
    velz_h->CopyFromVert(velz_i->vz);

    // diagnose the vorticity terms
    horiz->diagHorizVort(velx_i->vh, dudz_i->vh);
    dudz_i->UpdateLocal();
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i->vh[lev], dudz_j->vh[lev]);
        VecCopy(dudz_i->vl[lev], dudz_j->vl[lev]);
    }
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    do {
        schur->InitialiseMatrix();

        // assemble the vertical residuals
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            vert->assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii],
                              rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);
#ifdef NEW_EOS
            vo->Assemble_EOS_Residual_new(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
#else
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
#endif

            vert->vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii]);
            MatMult(vo->VB, dF_z, F_rho->vz[ii]);
            MatMult(vo->VB, dG_z, F_rt->vz[ii]);
        }
        F_exner->VertToHoriz(); F_exner->UpdateGlobal();
        F_rho->VertToHoriz();   F_rho->UpdateGlobal();
        F_rt->VertToHoriz();    F_rt->UpdateGlobal();

        // assemble the horizontal residuals
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling residuals for level: " << lev << endl;

            // velocity residual
            horiz->assemble_residual(lev, theta_h->vl, dudz_i->vl, dudz_j->vl, velz_i->vh, velz_j->vh, exner_h->vh[lev],
                                     velx_i->vh[lev], velx_j->vh[lev], rho_i->vh[lev], rho_j->vh[lev], F_u->vh[lev], 
                                     _F, _G, velx_i->vl[lev], velx_j->vl[lev], u_tmp_l);
            /*if(!itt)*/ VecCopy(u_tmp_l,gradPi->vl[lev]);

            horiz->M2->assemble(lev, SCALE, true);

            // density residual
            MatMult(horiz->EtoF->E21, _F, dF);
            MatMult(horiz->M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);

            // density weighted potential temperature residual
            MatMult(horiz->EtoF->E21, _G, dG);
            MatMult(horiz->M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);

            // add in the viscous term for the temperature equation
            horiz->M1->assemble(lev, SCALE, true);
            VecZeroEntries(dF);
            VecAXPY(dF, 0.5, theta_h->vh[lev+0]);
            VecAXPY(dF, 0.5, theta_h->vh[lev+1]);

            horiz->grad(false, dF, &dtheta, lev);
            horiz->F->assemble(rho_j->vl[lev], lev, true, SCALE);
            MatMult(horiz->F->M, dtheta, u_tmp_1);
            VecDestroy(&dtheta);

            KSPSolve(horiz->ksp1, u_tmp_1, u_tmp_2);
            MatMult(horiz->EtoF->E21, u_tmp_2, dG);

            horiz->grad(false, dG, &dtheta, lev);
            MatMult(horiz->EtoF->E21, dtheta, dG);
            VecDestroy(&dtheta);
            MatMult(horiz->M2->M, dG, dF);
            VecAXPY(F_rt->vh[lev], dt*horiz->del2*horiz->del2, dF);
        }

        // build the preconditioner matrix (horiztonal part first)
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling operators for level: " << lev << endl;
            horiz->assemble_and_update(lev, theta_i->vl, velx_i->vl[lev], velx_i->vh[lev], rho_i->vl[lev], rt_i->vl[lev], exner_i->vl[lev],
            //horiz->assemble_and_update(lev, theta_h->vl, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
                                       F_u->vh[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], gradPi->vl[lev], &horiz->_PCx, schur);
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        F_rt->UpdateLocal();  F_rt->HorizToVert();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            vert->assemble_and_update(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii],
            //vert->assemble_and_update(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
                                      F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], schur);
        }
        F_rt->VertToHoriz();  F_rt->UpdateGlobal();
        F_rho->VertToHoriz(); F_rho->UpdateGlobal();

        for(int lev = 0; lev < geom->nk; lev++) {
            horiz->assemble_M_rho(lev, velx_i->vl[lev], schur);
        }
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            vert->assemble_M_rho(ex, ey, velz_i->vz[ii], schur);
        }

        MatAssemblyBegin(schur->P, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->P, MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(schur->D, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->D, MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(schur->G, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->G, MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(schur->T, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->T, MAT_FINAL_ASSEMBLY);

        VecSet(schur->x, 1.0);
        MatGetDiagonal(schur->P, schur->b);
        VecPointwiseDivide(schur->x, schur->x, schur->b);
        MatDiagonalSet(schur->Pinv, schur->x, INSERT_VALUES);
        MatAssemblyBegin(schur->Pinv, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->Pinv, MAT_FINAL_ASSEMBLY);

        MatMatMult(schur->D, schur->Pinv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &schur->DPinv);
        MatAssemblyBegin(schur->DPinv, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->DPinv, MAT_FINAL_ASSEMBLY);
        MatMatMult(schur->DPinv, schur->G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &schur->S);
        MatAYPX(schur->S, -1.0, schur->T, DIFFERENT_NONZERO_PATTERN);
        MatAssemblyBegin(schur->S, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->S, MAT_FINAL_ASSEMBLY);

        schur->RepackFromHoriz(F_rho->vl, schur->x);
        MatMult(schur->DPinv, schur->x, schur->b);
        schur->RepackFromHoriz(F_rt->vl, schur->x);
        VecAXPY(schur->b, -1.0, schur->x);

        schur->Solve(schur->ksp_rt, schur->S, d_rt);
        d_rt->HorizToVert(); d_rt->UpdateGlobal();

        MatMult(schur->T, schur->x, schur->b); // L \delta_rt
        schur->RepackFromHoriz(F_rho->vl, schur->x);
        VecAXPY(schur->b, 1.0, schur->x);
        VecScale(schur->b, -1.0);
        
        schur->Solve(schur->ksp_rho, schur->P, d_rho);
        d_rho->HorizToVert(); d_rho->UpdateGlobal();

        MatDestroy(&schur->DPinv);
        MatDestroy(&schur->S);
        
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            vert->update_delta_pi(ex, ey, rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii], F_exner->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii]);
        }
        d_exner->VertToHoriz(); d_exner->UpdateGlobal();
/*
        // assemble the global schur complement operator
        schur->RepackFromHoriz(F_rho->vl, schur->x);
        MatAssemblyBegin(schur->Q, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->Q, MAT_FINAL_ASSEMBLY);
        MatMult(schur->Q, schur->x, schur->b);
        schur->RepackFromHoriz(F_rt->vl, schur->x);
        VecAXPY(schur->b, -1.0, schur->x);

        MatAssemblyBegin(schur->L, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->L, MAT_FINAL_ASSEMBLY);

        MatMatMult(schur->Q, schur->L, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &schur->M);

        MatAssemblyBegin(schur->L_rt, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->L_rt, MAT_FINAL_ASSEMBLY);
        MatAYPX(schur->M, -1.0, schur->L_rt, DIFFERENT_NONZERO_PATTERN);
        MatAssemblyBegin(schur->M, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->M, MAT_FINAL_ASSEMBLY);

        //KSPCreate(MPI_COMM_WORLD, &schur->ksp);
        KSPSetOperators(schur->ksp, schur->M, schur->M);

        // solve for the exner pressure update
        schur->Solve(d_rt);

        // update the delta vectors
        d_rt->HorizToVert(); d_rt->UpdateGlobal();

        // update the density
        if(!rank) cout << "updating density" << endl;
        MatMult(schur->L, schur->x, schur->b); // L \delta_rt
        schur->RepackFromHoriz(F_rho->vl, schur->x);
        VecAXPY(schur->b, 1.0, schur->x);
        VecScale(schur->b, -1.0);
        if(!rank) cout << "assemblying density matrix..." << endl;
        for(int lev = 0; lev < geom->nk; lev++) {
            horiz->assemble_M_rho(lev, velx_i->vl[lev], schur);
        }
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            vert->assemble_M_rho(ex, ey, velz_i->vz[ii], schur);
        }
        MatAssemblyBegin(schur->M_rho, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->M_rho, MAT_FINAL_ASSEMBLY);
        if(!rank) cout << "solving for density update..." << endl;
        KSPDestroy(&schur->ksp);
        KSPCreate(MPI_COMM_WORLD, &schur->ksp);
        KSPSetOperators(schur->ksp, schur->M_rho, schur->M_rho);
        schur->Solve(d_rho);
        d_rho->HorizToVert(); d_rho->UpdateGlobal();
*/

        // back substitution
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            vert->update_delta_u(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii],
            //vert->update_delta_u(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
                                F_w->vz[ii], F_exner->vz[ii], d_w->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii]);
        }
        //d_exner->VertToHoriz(); d_exner->UpdateGlobal();

        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "updating corrections for level: " << lev << endl;

            horiz->update_delta_u(lev, theta_i->vl, velx_i->vl[lev], velx_i->vh[lev], rho_i->vl[lev], rt_i->vl[lev], exner_i->vl[lev],
            //horiz->update_delta_u(lev, theta_h->vl, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
                                 F_u->vh[lev], d_u->vh[lev], d_rho->vh[lev], d_rt->vh[lev], d_exner->vh[lev], gradPi->vl[lev]);
        }

        // update solutions
        for(int lev = 0; lev < geom->nk; lev++) {
            VecAXPY(velx_j->vh[lev], 1.0, d_u->vh[lev]);
            VecZeroEntries(velx_h->vh[lev]);
            VecAXPY(velx_h->vh[lev], 0.5, velx_i->vh[lev]);
            VecAXPY(velx_h->vh[lev], 0.5, velx_j->vh[lev]);
        }
        velx_j->UpdateLocal();
        velx_h->UpdateLocal();
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(velz_j->vz[ii],  1.0, d_w->vz[ii]    );
            VecAXPY(rho_j->vz[ii],   1.0, d_rho->vz[ii]  );
            VecAXPY(rt_j->vz[ii],    1.0, d_rt->vz[ii]   );
            VecAXPY(exner_j->vz[ii], 1.0, d_exner->vz[ii]);
        }
        rho_j->VertToHoriz();   rho_j->UpdateGlobal();
        rt_j->VertToHoriz();    rt_j->UpdateGlobal();
        exner_j->VertToHoriz(); exner_j->UpdateGlobal();
        velz_j->VertToHoriz();  velz_j->UpdateGlobal();

        // update additional fields
        horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            VecScale(theta_h->vz[ii], 0.5);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);

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
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        rho_h->VertToHoriz(); rho_h->UpdateGlobal();
        rt_h->VertToHoriz(); rt_h->UpdateGlobal();

        horiz->diagHorizVort(velx_j->vh, dudz_j->vh);
        dudz_j->UpdateLocal();

        GlobalNorms(itt, d_u->vh, velx_j->vh, d_w, velz_j, d_rho, rho_j, d_rt, rt_j, d_exner, exner_j,
                    &max_norm_u, &max_norm_w, &max_norm_rho, &max_norm_rt, &max_norm_exner, h_tmp, u_tmp_1, F_z, true);

        itt++;
        if((max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8) || itt > 20) done = true;

        schur->DestroyMatrix();
    } while(!done);

    // copy the solutions back to the input vectors
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_j->vh);
    velx_j->CopyTo(velx);

    // write output
    if(save) dump(velx, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
    delete velz_h;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete rho_h;
    delete rt_h;
    delete theta_i;
    delete theta_h;
    delete F_w;
    delete F_rho;
    delete F_rt;
    delete F_exner;
    delete d_w;
    delete d_rho;
    delete d_rt;
    delete d_exner;
    delete velx_i;
    delete velx_j;
    delete velx_h;
    delete dudz_i;
    delete dudz_j;
    delete F_u;
    delete d_u;
    delete gradPi;
    VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&dF);
    VecDestroy(&dG);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
    VecDestroy(&h_tmp);
    VecDestroy(&u_tmp_1);
    VecDestroy(&u_tmp_2);
    VecDestroy(&u_tmp_l);
}

#if 0
void Euler::solve_schur_2(Vec* velx, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_w, max_norm_rho, max_norm_rt, max_norm_exner;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* velz_h  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_h   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* F_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* F_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_exner = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* d_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_exner = new L2Vecs(geom->nk, topo, geom);
    L1Vecs* velx_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* velx_j  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* velx_h  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_j  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* F_u     = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* d_u     = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* gradPi  = new L1Vecs(geom->nk, topo, geom);
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta;
Vec u_tmp_l;
    VertOps* vo = vert->vo;

    velx_i->CopyFrom(velx);
    velx_j->CopyFrom(velx);
    velx_h->CopyFrom(velx);
    velx_i->UpdateLocal();
    velx_j->UpdateLocal();
    velx_h->UpdateLocal();

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dG);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_2);
VecCreateSeq(MPI_COMM_SELF, topo->n1, &u_tmp_l);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);

    velz_i->UpdateLocal();  velz_i->HorizToVert();
    rho_i->UpdateLocal();   rho_i->HorizToVert();
    rt_i->UpdateLocal();    rt_i->HorizToVert();
    exner_i->UpdateLocal(); exner_i->HorizToVert();
    velz_j->CopyFromVert(velz_i->vz);   velz_j->VertToHoriz();  velz_j->UpdateGlobal();
    rho_j->CopyFromVert(rho_i->vz);     rho_j->VertToHoriz();   rho_j->UpdateGlobal();
    rt_j->CopyFromVert(rt_i->vz);       rt_j->VertToHoriz();    rt_j->UpdateGlobal();
    exner_j->CopyFromVert(exner_i->vz); exner_j->VertToHoriz(); exner_j->UpdateGlobal();
    exner_h->CopyFromVert(exner_i->vz); exner_h->VertToHoriz(); exner_h->UpdateGlobal();
    rho_h->CopyFromVert(rho_i->vz); rho_h->VertToHoriz(); rho_h->UpdateGlobal();
    rt_h->CopyFromVert(rt_i->vz); rt_h->VertToHoriz(); rt_h->UpdateGlobal();
    velz_h->CopyFromVert(velz_i->vz);

    // diagnose the vorticity terms
    horiz->diagHorizVort(velx_i->vh, dudz_i->vh);
    dudz_i->UpdateLocal();
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i->vh[lev], dudz_j->vh[lev]);
        VecCopy(dudz_i->vl[lev], dudz_j->vl[lev]);
    }
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    do {
        schur->InitialiseMatrix();

        // assemble the vertical residuals
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            vert->assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii],
                              rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);
#ifdef NEW_EOS
            vo->Assemble_EOS_Residual_new(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
#else
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
#endif
            vert->vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii]);
            MatMult(vo->VB, dF_z, F_rho->vz[ii]);
            MatMult(vo->VB, dG_z, F_rt->vz[ii]);
        }
        F_exner->VertToHoriz(); F_exner->UpdateGlobal();
        F_rho->VertToHoriz();   F_rho->UpdateGlobal();
        F_rt->VertToHoriz();    F_rt->UpdateGlobal();

        // assemble the horizontal residuals
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling residuals for level: " << lev << endl;

            // velocity residual
            horiz->assemble_residual(lev, theta_h->vl, dudz_i->vl, dudz_j->vl, velz_i->vh, velz_j->vh, exner_h->vh[lev],
                                     velx_i->vh[lev], velx_j->vh[lev], rho_i->vh[lev], rho_j->vh[lev], F_u->vh[lev], 
                                     //_F, _G, velx_i->vl[lev], velx_j->vl[lev], gradPi->vl[lev]);
                                     _F, _G, velx_i->vl[lev], velx_j->vl[lev], u_tmp_l);
if(!itt)VecCopy(u_tmp_l,gradPi->vl[lev]);

            horiz->M2->assemble(lev, SCALE, true);

            // density residual
            MatMult(horiz->EtoF->E21, _F, dF);
            MatMult(horiz->M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);

            // density weighted potential temperature residual
            MatMult(horiz->EtoF->E21, _G, dG);
            MatMult(horiz->M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);

            // add in the viscous term for the temperature equation
            horiz->M1->assemble(lev, SCALE, true);
            VecZeroEntries(dF);
            VecAXPY(dF, 0.5, theta_h->vh[lev+0]);
            VecAXPY(dF, 0.5, theta_h->vh[lev+1]);

            horiz->grad(false, dF, &dtheta, lev);
            horiz->F->assemble(rho_j->vl[lev], lev, true, SCALE);
            MatMult(horiz->F->M, dtheta, u_tmp_1);
            VecDestroy(&dtheta);

            KSPSolve(horiz->ksp1, u_tmp_1, u_tmp_2);
            MatMult(horiz->EtoF->E21, u_tmp_2, dG);

            horiz->grad(false, dG, &dtheta, lev);
            MatMult(horiz->EtoF->E21, dtheta, dG);
            VecDestroy(&dtheta);
            MatMult(horiz->M2->M, dG, dF);
            VecAXPY(F_rt->vh[lev], dt*horiz->del2*horiz->del2, dF);
        }

        // build the preconditioner matrix (horiztonal part first)
        MatZeroEntries(schur->M);
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling operators for level: " << lev << endl;

            //horiz->assemble_and_update(lev, theta_h->vl, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
            horiz->assemble_and_update(lev, theta_i->vl, velx_i->vl[lev], velx_i->vh[lev], rho_i->vl[lev], rt_i->vl[lev], exner_i->vl[lev],
                                       F_u->vh[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], gradPi->vl[lev], &horiz->_PCx, NULL);

            schur->AddFromHorizMat(lev, horiz->_PCx, schur->M);
            MatDestroy(&horiz->_PCx);
        }
        F_rho->UpdateLocal();       F_rho->HorizToVert();
        F_rt->UpdateLocal();        F_rt->HorizToVert();

        if(!rank) cout << "assembling vertical operators" << endl;
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            //vert->assemble_and_update(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
            vert->assemble_and_update(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii],
                                      F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], NULL);

            schur->AddFromVertMat(ii, vert->_PCz, schur->M);
        }
        F_rt->VertToHoriz(); F_rt->UpdateGlobal();
MatAssemblyBegin(schur->M, MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(  schur->M, MAT_FINAL_ASSEMBLY);

        // solve for the exner pressure update
        VecZeroEntries(schur->b);
        schur->RepackFromHoriz(F_rt->vl, schur->b);
        VecScale(schur->b, -1.0);
        schur->Solve(d_rt);
        schur->DestroyMatrix();

        // update the delta vectors
        d_rt->HorizToVert(); d_rt->UpdateGlobal();

        // update the pressure field
        schur_rho->InitialiseMatrix();
        MatZeroEntries(schur_rho->M);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            VecAXPY(rt_j->vz[ii], 1.0, d_rt->vz[ii]);

            //vert->update_delta_pi_2(ex, ey, rt_h->vz[ii], exner_h->vz[ii], F_exner->vz[ii], d_rt->vz[ii], d_exner->vz[ii]);
            vert->update_delta_pi_2(ex, ey, rt_i->vz[ii], exner_i->vz[ii], F_exner->vz[ii], d_rt->vz[ii], d_exner->vz[ii]);
            VecAXPY(exner_j->vz[ii], 1.0, d_exner->vz[ii]);
/*
            vo->Assemble_EOS_RHS(ex, ey, rt_j->vz[ii], vert->_tmpB1, CP*pow(RD/P0, RD/CV), RD/CV);
            vo->AssembleConstInv(ex, ey, vo->VB_inv);
            MatMult(vo->VB_inv, vert->_tmpB1, exner_j->vz[ii]);
*/
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);
/*
            VecZeroEntries(d_exner->vz[ii]);
            VecAXPY(d_exner->vz[ii], -1.0, exner_i->vz[ii]);
            VecAXPY(d_exner->vz[ii], +1.0, exner_j->vz[ii]);
*/
        }
        exner_j->VertToHoriz(); exner_j->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        d_exner->VertToHoriz(); d_exner->UpdateGlobal();
        rt_j->VertToHoriz();    rt_j->UpdateGlobal();
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling density operators for level: " << lev << endl;

            horiz->assemble_residual(lev, theta_h->vl, dudz_i->vl, dudz_j->vl, velz_i->vh, velz_j->vh, exner_h->vh[lev],
                                     velx_i->vh[lev], velx_j->vh[lev], rho_i->vh[lev], rho_j->vh[lev], F_u->vh[lev], 
                                     _F, _G, velx_i->vl[lev], velx_j->vl[lev], gradPi->vl[lev]);

            horiz->M2->assemble(lev, SCALE, true);
            MatMult(horiz->EtoF->E21, _F, dF);
            VecAYPX(dF, dt, rho_j->vh[lev]);
            VecAXPY(dF, -1.0, rho_i->vh[lev]);
            MatMult(horiz->M2->M, dF, F_rho->vh[lev]);

            //horiz->assemble_and_update_2(lev, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev],
            horiz->assemble_and_update_2(lev, velx_i->vl[lev], velx_i->vh[lev], rho_i->vl[lev], rt_i->vl[lev],
                                       F_u->vh[lev], F_rho->vh[lev], gradPi->vl[lev]);

            schur_rho->AddFromHorizMat(lev, horiz->L_rho_rho, schur_rho->M);
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        if(!rank) cout << "assembling vertical density operators" << endl;
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            vert->assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii],
                              rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);

            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->VB, dF_z, vert->_tmpB1);
            VecAXPY(F_rho->vz[ii], dt, vert->_tmpB1);

            //vert->assemble_and_update_2(ex, ey, velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
            vert->assemble_and_update_2(ex, ey, velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii],
                                      F_w->vz[ii], F_rho->vz[ii]);
            schur_rho->AddFromVertMat(ii, vert->L_rho_rho, schur_rho->M);
        }
        F_rho->VertToHoriz(); F_rho->UpdateGlobal();
MatAssemblyBegin(schur_rho->M, MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(  schur_rho->M, MAT_FINAL_ASSEMBLY);
        // solve for the density update
        VecZeroEntries(schur_rho->b);
        schur_rho->RepackFromHoriz(F_rho->vl, schur_rho->b);
        VecScale(schur_rho->b, -1.0);
        schur_rho->Solve(d_rho);
        schur_rho->DestroyMatrix();
        d_rho->HorizToVert(); d_rho->UpdateGlobal();
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(rho_j->vz[ii], 1.0, d_rho->vz[ii]);
        }
        rho_j->VertToHoriz(); rho_j->UpdateGlobal();
        // update the velocities
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            //vert->update_delta_u_2(ex, ey, velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
            vert->update_delta_u_2(ex, ey, velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii],
                                F_w->vz[ii], F_rho->vz[ii], d_w->vz[ii], d_rho->vz[ii]);
            VecAXPY(velz_j->vz[ii], 1.0, d_w->vz[ii]);
        }
        velz_j->VertToHoriz(); velz_j->UpdateGlobal();
        for(int lev = 0; lev < geom->nk; lev++) {
            //horiz->update_delta_u_2(lev, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev],
            horiz->update_delta_u_2(lev, velx_i->vl[lev], velx_i->vh[lev], rho_i->vl[lev], rt_i->vl[lev],
                                 F_u->vh[lev], F_rho->vh[lev], gradPi->vl[lev], d_u->vh[lev], d_rho->vh[lev]);
            VecAXPY(velx_j->vh[lev], 1.0, d_u->vh[lev]);
        }
        velx_j->UpdateLocal();
      
        // update additional fields
        for(int lev = 0; lev < geom->nk; lev++) {
            VecZeroEntries(velx_h->vh[lev]);
            VecAXPY(velx_h->vh[lev], 0.5, velx_i->vh[lev]);
            VecAXPY(velx_h->vh[lev], 0.5, velx_j->vh[lev]);
        }
        velx_h->UpdateLocal();

        horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecScale(theta_h->vz[ii], 0.5);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);

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
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        rho_h->VertToHoriz(); rho_h->UpdateGlobal();
        rt_h->VertToHoriz(); rt_h->UpdateGlobal();

        horiz->diagHorizVort(velx_j->vh, dudz_j->vh);
        dudz_j->UpdateLocal();

        GlobalNorms(itt, d_u->vh, velx_j->vh, d_w, velz_j, d_rho, rho_j, d_rt, rt_j, d_exner, exner_j,
                    &max_norm_u, &max_norm_w, &max_norm_rho, &max_norm_rt, &max_norm_exner, h_tmp, u_tmp_1, F_z, true);

        itt++;
        if((max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8) || itt > 20) done = true;
    } while(!done);

    // copy the solutions back to the input vectors
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_j->vh);
    velx_j->CopyTo(velx);

    // write output
    if(save) dump(velx, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
    delete velz_h;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete rho_h;
    delete rt_h;
    delete theta_i;
    delete theta_h;
    delete F_w;
    delete F_rho;
    delete F_rt;
    delete F_exner;
    delete d_w;
    delete d_rho;
    delete d_rt;
    delete d_exner;
    delete velx_i;
    delete velx_j;
    delete velx_h;
    delete dudz_i;
    delete dudz_j;
    delete F_u;
    delete d_u;
    delete gradPi;
    VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&dF);
    VecDestroy(&dG);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
    VecDestroy(&h_tmp);
    VecDestroy(&u_tmp_1);
    VecDestroy(&u_tmp_2);
VecDestroy(&u_tmp_l);
}
#endif

void Euler::solve_schur_3(Vec* velx, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_w, max_norm_rho, max_norm_rt, max_norm_exner, alpha;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* velz_h  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_h   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* F_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* F_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_exner = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* d_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_exner = new L2Vecs(geom->nk, topo, geom);
    L1Vecs* velx_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* velx_j  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* velx_h  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_j  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* F_u     = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* d_u     = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* gradPi  = new L1Vecs(geom->nk, topo, geom);
    L2Vecs* div_u   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_rt_0  = new L2Vecs(geom->nk, topo, geom);
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta, u_tmp_l;
    VertOps* vo = vert->vo;

    velx_i->CopyFrom(velx);
    velx_j->CopyFrom(velx);
    velx_h->CopyFrom(velx);
    velx_i->UpdateLocal();
    velx_j->UpdateLocal();
    velx_h->UpdateLocal();

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dG);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_2);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &u_tmp_l);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);

    velz_i->UpdateLocal();  velz_i->HorizToVert();
    rho_i->UpdateLocal();   rho_i->HorizToVert();
    rt_i->UpdateLocal();    rt_i->HorizToVert();
    exner_i->UpdateLocal(); exner_i->HorizToVert();
    velz_j->CopyFromVert(velz_i->vz);   velz_j->VertToHoriz();  velz_j->UpdateGlobal();
    rho_j->CopyFromVert(rho_i->vz);     rho_j->VertToHoriz();   rho_j->UpdateGlobal();
    rt_j->CopyFromVert(rt_i->vz);       rt_j->VertToHoriz();    rt_j->UpdateGlobal();
    exner_j->CopyFromVert(exner_i->vz); exner_j->VertToHoriz(); exner_j->UpdateGlobal();
    exner_h->CopyFromVert(exner_i->vz); exner_h->VertToHoriz(); exner_h->UpdateGlobal();
    rho_h->CopyFromVert(rho_i->vz); rho_h->VertToHoriz(); rho_h->UpdateGlobal();
    rt_h->CopyFromVert(rt_i->vz); rt_h->VertToHoriz(); rt_h->UpdateGlobal();
    velz_h->CopyFromVert(velz_i->vz);

    // diagnose the vorticity terms
    horiz->diagHorizVort(velx_i->vh, dudz_i->vh);
    dudz_i->UpdateLocal();
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i->vh[lev], dudz_j->vh[lev]);
        VecCopy(dudz_i->vl[lev], dudz_j->vl[lev]);
    }
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    do {
        schur->InitialiseMatrix();

        // assemble the vertical residuals
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            vert->assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii],
                              rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);
#ifdef NEW_EOS
            vo->Assemble_EOS_Residual_new(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
#else
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
#endif
            vert->vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii]);
            MatMult(vo->VB, dF_z, F_rho->vz[ii]);
            MatMult(vo->VB, dG_z, F_rt->vz[ii]);
        }
        F_exner->VertToHoriz(); F_exner->UpdateGlobal();
        F_rho->VertToHoriz();   F_rho->UpdateGlobal();
        F_rt->VertToHoriz();    F_rt->UpdateGlobal();

        // assemble the horizontal residuals
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling residuals for level: " << lev << endl;

            // velocity residual
            horiz->assemble_residual(lev, theta_h->vl, dudz_i->vl, dudz_j->vl, velz_i->vh, velz_j->vh, exner_h->vh[lev],
                                     velx_i->vh[lev], velx_j->vh[lev], rho_i->vh[lev], rho_j->vh[lev], F_u->vh[lev], 
                                     _F, _G, velx_i->vl[lev], velx_j->vl[lev], u_tmp_l);
            if(!itt) VecCopy(u_tmp_l,gradPi->vl[lev]);

            horiz->M2->assemble(lev, SCALE, true);

            // density residual
            MatMult(horiz->EtoF->E21, _F, dF);
            MatMult(horiz->M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);

            // density weighted potential temperature residual
            MatMult(horiz->EtoF->E21, _G, dG);
            MatMult(horiz->M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);

            // add in the viscous term for the temperature equation
            horiz->M1->assemble(lev, SCALE, true);
            VecZeroEntries(dF);
            VecAXPY(dF, 0.5, theta_h->vh[lev+0]);
            VecAXPY(dF, 0.5, theta_h->vh[lev+1]);

            horiz->grad(false, dF, &dtheta, lev);
            horiz->F->assemble(rho_j->vl[lev], lev, true, SCALE);
            MatMult(horiz->F->M, dtheta, u_tmp_1);
            VecDestroy(&dtheta);

            KSPSolve(horiz->ksp1, u_tmp_1, u_tmp_2);
            MatMult(horiz->EtoF->E21, u_tmp_2, dG);

            horiz->grad(false, dG, &dtheta, lev);
            MatMult(horiz->EtoF->E21, dtheta, dG);
            VecDestroy(&dtheta);
            MatMult(horiz->M2->M, dG, dF);
            VecAXPY(F_rt->vh[lev], dt*horiz->del2*horiz->del2, dF);

            VecCopy(F_rt->vh[lev], F_rt_0->vh[lev]);
        }
        F_rt_0->UpdateLocal(); F_rt_0->HorizToVert();

        // build the preconditioner matrix (horiztonal part first)
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling operators for level: " << lev << endl;
            //horiz->assemble_and_update_3(lev, theta_i->vl, velx_i->vl[lev], velx_i->vh[lev], rho_i->vl[lev], rt_i->vl[lev], exner_i->vl[lev],
            horiz->assemble_and_update_3(lev, theta_h->vl, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
                                       F_u->vh[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], gradPi->vl[lev], div_u->vh[lev], schur, itt);
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        F_rt->UpdateLocal();  F_rt->HorizToVert();
        div_u->UpdateLocal(); div_u->HorizToVert();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            //vert->assemble_and_update_3(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii],
            vert->assemble_and_update_3(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
                                      F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], div_u->vz[ii], schur, itt);
        }
        F_rt->VertToHoriz();  F_rt->UpdateGlobal();
        F_rho->VertToHoriz(); F_rho->UpdateGlobal();
        div_u->VertToHoriz();

        MatAssemblyBegin(schur->Q, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->Q, MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(schur->L_rho, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->L_rho, MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(schur->L_rt, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->L_rt, MAT_FINAL_ASSEMBLY);

VecSet(schur->x, 1.0);
MatMult(schur->Q, schur->x, schur->b);
VecDot(schur->x, schur->b, &max_norm_exner);
if(!rank) cout << "||Q||: " << max_norm_exner << endl;

        // build the schur complement opertor
        MatMatMult(schur->Q, schur->L_rho, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &schur->S);
        MatAXPY(schur->S, -1.0, schur->L_rt, DIFFERENT_NONZERO_PATTERN);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            vert->vo->AssembleConst(ex, ey, vo->VB);
            //schur->AddFromVertMat(ii, vo->VB, schur->S);
            schur->AddFromVertMat(ii, vo->VB, schur->A);
        }
        MatAssemblyBegin(schur->A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->A, MAT_FINAL_ASSEMBLY);
        MatAXPY(schur->S, 1.0, schur->A, DIFFERENT_NONZERO_PATTERN);

        //MatAssemblyBegin(schur->VISC, MAT_FINAL_ASSEMBLY);
        //MatAssemblyEnd(  schur->VISC, MAT_FINAL_ASSEMBLY);
        //MatAXPY(schur->S, 1.0, schur->VISC, DIFFERENT_NONZERO_PATTERN);

        MatAssemblyBegin(schur->S, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->S, MAT_FINAL_ASSEMBLY);

        // update the rhs
        schur->RepackFromHoriz(F_rt->vl, schur->b);

        schur->RepackFromHoriz(F_rho->vl, schur->x);
        MatMult(schur->Q, schur->x, schur->t);
        VecAXPY(schur->b, -1.0, schur->t);

        schur->RepackFromHoriz(div_u->vl, schur->x);
        MatMult(schur->Q, schur->x, schur->t);
        VecAXPY(schur->t, +1.0, schur->x);

        VecScale(schur->b, -1.0);

        schur->Solve(schur->ksp_rt, schur->S, d_rt);
        //schur->Solve(schur->ksp_rt, schur->VISC, d_rt);
        d_rt->HorizToVert(); d_rt->UpdateGlobal();

        // back substitution
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            //vert->update_delta_u_3(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii],
            vert->update_delta_u_3(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
                                F_w->vz[ii], F_rho->vz[ii], F_exner->vz[ii], d_w->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii]);
        }
        d_exner->VertToHoriz(); d_exner->UpdateGlobal();
        F_rho->VertToHoriz();   F_rho->UpdateGlobal();

        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "updating corrections for level: " << lev << endl;

            //horiz->update_delta_u_3(lev, theta_i->vl, velx_i->vl[lev], velx_i->vh[lev], rho_i->vl[lev], rt_i->vl[lev], exner_i->vl[lev],
            horiz->update_delta_u_3(lev, theta_h->vl, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
                                 F_u->vh[lev], F_rho->vh[lev], d_u->vh[lev], d_rho->vh[lev], d_rt->vh[lev], d_exner->vh[lev], gradPi->vl[lev]);
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();

        // density update
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            vert->vo->AssembleConstInv(ex, ey, vo->VB_inv);
            VecScale(F_rho->vz[ii], -1.0);
            MatMult(vo->VB_inv, F_rho->vz[ii], d_rho->vz[ii]);
        }
        d_rho->VertToHoriz(); d_rho->UpdateGlobal();

/*
schur->RepackFromHoriz(d_rt->vl, schur->x);
MatMult(schur->A, schur->x, schur->b);
        MatAssemblyBegin(schur->Q2, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->Q2, MAT_FINAL_ASSEMBLY);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
//            vert->vo->AssembleConst(ex, ey, vo->VB);
//            MatMult(vo->VB, d_rt->vz[ii], vert->_tmpB1);
//            VecAXPY(F_rt_0->vz[ii], 1.0, vert->_tmpB1);
            vo->AssembleConstWithRho(ex, ey, rt_h->vz[ii], vo->VB);
            MatMatMult(vo->VB, vo->V10, MAT_REUSE_MATRIX, PETSC_DEFAULT, &vert->D_rt);
            MatMult(vert->D_rt, d_w->vz[ii], vert->_tmpB1);
            VecAXPY(F_rt_0->vz[ii], 0.5*dt, vert->_tmpB1);

            vo->AssembleConstInv(ex, ey, vo->VB_inv);
            schur->AddFromVertMat(ii, vo->VB_inv, schur->Minv);
        }
        F_rt_0->VertToHoriz(); F_rt_0->UpdateGlobal();
        MatAssemblyBegin(schur->Minv, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->Minv, MAT_FINAL_ASSEMBLY);
        for(int lev = 0; lev < geom->nk; lev++) {
            horiz->T->assemble(rt_h->vl[lev], lev, SCALE, true);
            MatMatMult(horiz->T->M, horiz->EtoF->E21, MAT_REUSE_MATRIX, PETSC_DEFAULT, &horiz->D_rt);
            MatMult(horiz->D_rt, d_u->vh[lev], h_tmp);
            VecAXPY(F_rt_0->vh[lev], 0.5*dt, h_tmp);
            VecScale(F_rt_0->vh[lev], -1.0);
        }
        F_rt_0->UpdateLocal();

        MatTranspose(schur->Q2, MAT_INITIAL_MATRIX, &schur->QT);
        MatMatMult(schur->QT, schur->Minv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &schur->QTMinv);
        MatMatMult(schur->QTMinv, schur->Q2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &schur->QTMinvQ);
        schur->RepackFromHoriz(F_rt_0->vl, schur->t);
VecAXPY(schur->t, 1.0, schur->b);
        MatMult(schur->QTMinv, schur->t, schur->b);

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            vert->vo->AssembleConst(ex, ey, vo->VB);
            schur->AddFromVertMat(ii, vo->VB, schur->QTMinvQ);
        }
        MatAssemblyBegin(schur->QTMinvQ, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->QTMinvQ, MAT_FINAL_ASSEMBLY);

        schur->RepackFromHoriz(F_rho->vl, schur->t);
        VecAXPY(schur->b, -1.0, schur->t);

        schur->Solve(schur->ksp_rho, schur->QTMinvQ, d_rho);
        d_rho->HorizToVert(); d_rho->UpdateGlobal();
*/

/*
        schur->RepackFromHoriz(F_rho->vl, schur->b);
        schur->Solve(schur->ksp_rho, schur->A, d_rho);
        d_rho->HorizToVert(); d_rho->UpdateGlobal();
*/

// explicit density update
/*
for(int lev = 0; lev < geom->nk; lev++) {
    horiz->assemble_residual(lev, theta_h->vl, dudz_i->vl, dudz_j->vl, velz_i->vh, velz_j->vh, exner_h->vh[lev],
                             velx_i->vh[lev], velx_j->vh[lev], rho_i->vh[lev], rho_j->vh[lev], F_u->vh[lev], 
                             _F, _G, velx_i->vl[lev], velx_j->vl[lev], gradPi->vl[lev]);
    MatMult(horiz->EtoF->E21, _F, rho_j->vh[lev]);
    VecAYPX(rho_j->vh[lev], -dt, rho_i->vh[lev]);
}
rho_j->UpdateLocal(); rho_j->HorizToVert();
for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
    ex = ii%topo->nElsX;
    ey = ii/topo->nElsX;

    vert->assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii],
                            rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);
    MatMult(vo->V10, F_z, dF_z);
    VecAXPY(rho_j->vz[ii], -dt, dF_z);
}
rho_j->VertToHoriz(); rho_j->UpdateGlobal();
for(int lev = 0; lev < geom->nk; lev++) {
    VecCopy(rho_j->vh[lev], d_rho->vh[lev]);
    VecAXPY(d_rho->vh[lev], -1.0, rho_i->vh[lev]);
    VecZeroEntries(rho_h->vh[lev]);
    VecAXPY(rho_h->vh[lev], 0.5, rho_i->vh[lev]);
    VecAXPY(rho_h->vh[lev], 0.5, rho_j->vh[lev]);
}
d_rho->UpdateLocal(); d_rho->HorizToVert();
rho_h->UpdateLocal(); rho_h->HorizToVert();
*/
        alpha = 1.0;//LineSearch(velx_i, velx_j, d_u, velz_i, velz_j, d_w, rho_i, rho_j, d_rho, rt_i, rt_j, d_rt, theta_i);

        // update solutions
        for(int lev = 0; lev < geom->nk; lev++) {
            VecAXPY(velx_j->vh[lev], alpha, d_u->vh[lev]);
            VecZeroEntries(velx_h->vh[lev]);
            VecAXPY(velx_h->vh[lev], 0.5, velx_i->vh[lev]);
            VecAXPY(velx_h->vh[lev], 0.5, velx_j->vh[lev]);
        }
        velx_j->UpdateLocal();
        velx_h->UpdateLocal();
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(velz_j->vz[ii],  alpha, d_w->vz[ii]    );
            VecAXPY(rho_j->vz[ii],   alpha, d_rho->vz[ii]  );
            VecAXPY(rt_j->vz[ii],    alpha, d_rt->vz[ii]   );
            VecAXPY(exner_j->vz[ii], alpha, d_exner->vz[ii]);
        }
        rho_j->VertToHoriz();   rho_j->UpdateGlobal();
        rt_j->VertToHoriz();    rt_j->UpdateGlobal();
        exner_j->VertToHoriz(); exner_j->UpdateGlobal();
        velz_j->VertToHoriz();  velz_j->UpdateGlobal();

        // update additional fields
        horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            VecScale(theta_h->vz[ii], 0.5);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);

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
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        rho_h->VertToHoriz(); rho_h->UpdateGlobal();
        rt_h->VertToHoriz(); rt_h->UpdateGlobal();

        horiz->diagHorizVort(velx_j->vh, dudz_j->vh);
        dudz_j->UpdateLocal();

        GlobalNorms(itt, d_u->vh, velx_j->vh, d_w, velz_j, d_rho, rho_j, d_rt, rt_j, d_exner, exner_j,
                    &max_norm_u, &max_norm_w, &max_norm_rho, &max_norm_rt, &max_norm_exner, h_tmp, u_tmp_1, F_z, true);

        itt++;
        if((max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8) || itt > 20) done = true;

        schur->DestroyMatrix();
        MatDestroy(&schur->S);
        //MatDestroy(&schur->QT);
        //MatDestroy(&schur->QTMinv);
        //MatDestroy(&schur->QTMinvQ);
    } while(!done);

    // copy the solutions back to the input vectors
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_j->vh);
    velx_j->CopyTo(velx);

    // write output
    if(save) dump(velx, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
    delete velz_h;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete rho_h;
    delete rt_h;
    delete theta_i;
    delete theta_h;
    delete F_w;
    delete F_rho;
    delete F_rt;
    delete F_exner;
    delete d_w;
    delete d_rho;
    delete d_rt;
    delete d_exner;
    delete velx_i;
    delete velx_j;
    delete velx_h;
    delete dudz_i;
    delete dudz_j;
    delete F_u;
    delete d_u;
    delete gradPi;
    delete div_u;
    delete F_rt_0;
    VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&dF);
    VecDestroy(&dG);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
    VecDestroy(&h_tmp);
    VecDestroy(&u_tmp_1);
    VecDestroy(&u_tmp_2);
    VecDestroy(&u_tmp_l);
}

// Gauss-Seidel splitting of horiztonal and vertical pressure updates (horiztonal then vertical)
void Euler::solve_gauss_seidel(Vec* velx, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_w, max_norm_rho, max_norm_rt, max_norm_exner;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* velz_h  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rho_h   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_h    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_i = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    L2Vecs* F_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* F_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_exner = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_w     = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* d_rho   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_rt    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* d_exner = new L2Vecs(geom->nk, topo, geom);
    L1Vecs* velx_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* velx_j  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* velx_h  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_i  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* dudz_j  = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* F_u     = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* d_u     = new L1Vecs(geom->nk, topo, geom);
    L1Vecs* gradPi  = new L1Vecs(geom->nk, topo, geom);
    L2Vecs* lap_d_pi = new L2Vecs(geom->nk, topo, geom);
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta;
    VertOps* vo = vert->vo;
    Mat* PCx = new Mat[geom->nk];
    PC pc;
    KSP ksp_horiz;

    velx_i->CopyFrom(velx);
    velx_j->CopyFrom(velx);
    velx_h->CopyFrom(velx);
    velx_i->UpdateLocal();
    velx_j->UpdateLocal();
    velx_h->UpdateLocal();

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dF);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dG);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &h_tmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_1);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &u_tmp_2);

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dF_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &dG_z);

    velz_i->UpdateLocal();  velz_i->HorizToVert();
    rho_i->UpdateLocal();   rho_i->HorizToVert();
    rt_i->UpdateLocal();    rt_i->HorizToVert();
    exner_i->UpdateLocal(); exner_i->HorizToVert();
    velz_j->CopyFromVert(velz_i->vz);   velz_j->VertToHoriz();  velz_j->UpdateGlobal();
    rho_j->CopyFromVert(rho_i->vz);     rho_j->VertToHoriz();   rho_j->UpdateGlobal();
    rt_j->CopyFromVert(rt_i->vz);       rt_j->VertToHoriz();    rt_j->UpdateGlobal();
    exner_j->CopyFromVert(exner_i->vz); exner_j->VertToHoriz(); exner_j->UpdateGlobal();
    exner_h->CopyFromVert(exner_i->vz); exner_h->VertToHoriz(); exner_h->UpdateGlobal();
    rho_h->CopyFromVert(rho_i->vz); rho_h->VertToHoriz(); rho_h->UpdateGlobal();
    rt_h->CopyFromVert(rt_i->vz); rt_h->VertToHoriz(); rt_h->UpdateGlobal();
    velz_h->CopyFromVert(velz_i->vz);

    // diagnose the vorticity terms
    horiz->diagHorizVort(velx_i->vh, dudz_i->vh);
    dudz_i->UpdateLocal();
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i->vh[lev], dudz_j->vh[lev]);
        VecCopy(dudz_i->vl[lev], dudz_j->vl[lev]);
    }
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    do {
        // assemble the vertical residuals
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            vert->assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii],
                              rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);
#ifdef NEW_EOS
            vo->Assemble_EOS_Residual_new(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
#else
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);
#endif

            vert->vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->V10, F_z, dF_z);
            MatMult(vo->V10, G_z, dG_z);
            VecAYPX(dF_z, dt, rho_j->vz[ii]);
            VecAYPX(dG_z, dt, rt_j->vz[ii]);
            VecAXPY(dF_z, -1.0, rho_i->vz[ii]);
            VecAXPY(dG_z, -1.0, rt_i->vz[ii]);
            MatMult(vo->VB, dF_z, F_rho->vz[ii]);
            MatMult(vo->VB, dG_z, F_rt->vz[ii]);
        }
        F_exner->VertToHoriz(); F_exner->UpdateGlobal();
        F_rho->VertToHoriz();   F_rho->UpdateGlobal();
        F_rt->VertToHoriz();    F_rt->UpdateGlobal();

        // assemble the horizontal residuals
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling residuals for level: " << lev << endl;

            // velocity residual
            horiz->assemble_residual(lev, theta_h->vl, dudz_i->vl, dudz_j->vl, velz_i->vh, velz_j->vh, exner_h->vh[lev],
                                     velx_i->vh[lev], velx_j->vh[lev], rho_i->vh[lev], rho_j->vh[lev], F_u->vh[lev], 
                                     _F, _G, velx_i->vl[lev], velx_j->vl[lev], gradPi->vl[lev]);

            horiz->M2->assemble(lev, SCALE, true);

            // density residual
            MatMult(horiz->EtoF->E21, _F, dF);
            MatMult(horiz->M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);

            // density weighted potential temperature residual
            MatMult(horiz->EtoF->E21, _G, dG);
            MatMult(horiz->M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);

            // add in the viscous term for the temperature equation
            horiz->M1->assemble(lev, SCALE, true);
            VecZeroEntries(dF);
            VecAXPY(dF, 0.5, theta_h->vh[lev+0]);
            VecAXPY(dF, 0.5, theta_h->vh[lev+1]);

            horiz->grad(false, dF, &dtheta, lev);
            horiz->F->assemble(rho_j->vl[lev], lev, true, SCALE);
            MatMult(horiz->F->M, dtheta, u_tmp_1);
            VecDestroy(&dtheta);

            KSPSolve(horiz->ksp1, u_tmp_1, u_tmp_2);
            MatMult(horiz->EtoF->E21, u_tmp_2, dG);

            horiz->grad(false, dG, &dtheta, lev);
            MatMult(horiz->EtoF->E21, dtheta, dG);
            VecDestroy(&dtheta);
            MatMult(horiz->M2->M, dG, dF);
            VecAXPY(F_rt->vh[lev], dt*horiz->del2*horiz->del2, dF);
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        F_rt->UpdateLocal();  F_rt->HorizToVert();

        // build the preconditioner matrix (horiztonal part first)
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling operators for level: " << lev << endl;

            horiz->assemble_and_update(lev, theta_h->vl, velx_i->vl[lev], velx_i->vh[lev], rho_i->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
            //horiz->assemble_and_update(lev, theta_h->vl, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
                                       F_u->vh[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], gradPi->vl[lev], &PCx[lev], NULL);

            MatMult(PCx[lev], d_rt->vh[lev], lap_d_pi->vh[lev]);
            VecScale(lap_d_pi->vh[lev], -1.0);
        }
        F_rho->UpdateLocal();    F_rho->HorizToVert();
        F_rt->UpdateLocal();     F_rt->HorizToVert();
        lap_d_pi->UpdateLocal(); lap_d_pi->HorizToVert();

        // build the vertical preconditioner and do the vertical solve
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            vert->assemble_and_update(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
            //vert->assemble_and_update(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
                                      F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], NULL);

            //VecCopy(lap_d_pi->vz[ii], dF_z);   // TODO: check sign!!
            //VecAXPY(dF_z, -1.0, F_rt->vz[ii]); // TODO: check sign!!
            VecCopy(F_rt->vz[ii], dF_z);
            VecScale(dF_z, -1.0);

            if(!vert->ksp_pi) {
                KSPCreate(MPI_COMM_SELF, &vert->ksp_pi);
                KSPSetOperators(vert->ksp_pi, vert->_PCz, vert->_PCz);
                KSPGetPC(vert->ksp_pi, &pc);
                PCSetType(pc, PCLU);
                KSPSetOptionsPrefix(vert->ksp_pi, "ksp_pi_");
                KSPSetFromOptions(vert->ksp_pi);
            }
            KSPSolve(vert->ksp_pi, dF_z, d_rt->vz[ii]);

            // remove the mass matrix from the preconditioner
            vo->AssembleConst(ex, ey, vo->VB);
            MatAXPY(vert->_PCz, -1.0, vo->VB, DIFFERENT_NONZERO_PATTERN);
            MatMult(vert->_PCz, d_rt->vz[ii], lap_d_pi->vz[ii]);
            VecScale(lap_d_pi->vz[ii], -1.0);
        }
        F_rt->VertToHoriz();     F_rt->UpdateGlobal();
        F_rho->VertToHoriz();    F_rho->UpdateGlobal();
        d_rt->VertToHoriz();     d_rt->UpdateGlobal();
        lap_d_pi->VertToHoriz(); lap_d_pi->UpdateGlobal();

        // horiztonal solve
        for(int lev = 0; lev < geom->nk; lev++) {
            KSPCreate(MPI_COMM_WORLD, &ksp_horiz);
            KSPSetOperators(ksp_horiz, PCx[lev], PCx[lev]);
            KSPSetTolerances(ksp_horiz, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
            KSPSetType(ksp_horiz, KSPGMRES);
            KSPGetPC(ksp_horiz, &pc);
            PCSetType(pc, PCBJACOBI);
            PCBJacobiSetTotalBlocks(pc, 6*topo->nElsX*topo->nElsX, NULL);
            KSPSetOptionsPrefix(ksp_horiz, "ksp_horiz_");
            KSPSetFromOptions(ksp_horiz);

            horiz->M2->assemble(lev, SCALE, true);
            MatAXPY(PCx[lev], 1.0, horiz->M2->M, DIFFERENT_NONZERO_PATTERN);
            VecCopy(lap_d_pi->vh[lev], dF);
            VecAXPY(dF, -1.0, F_rt->vh[lev]);
            KSPSolve(ksp_horiz, dF, d_rt->vh[lev]);

            KSPDestroy(&ksp_horiz);
        }
        d_rt->UpdateLocal(); d_rt->HorizToVert();

        // back substitution
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            vert->update_deltas(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
            //vert->update_deltas(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
                                F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], d_w->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii]);
        }
        d_exner->VertToHoriz(); d_exner->UpdateGlobal();
        d_rho->VertToHoriz();   d_rho->UpdateGlobal();

        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "updating corrections for level: " << lev << endl;

            horiz->update_deltas(lev, theta_h->vl, velx_i->vl[lev], velx_i->vh[lev], rho_i->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
            //horiz->update_deltas(lev, theta_h->vl, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
                                 F_u->vh[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], 
                                 d_u->vh[lev], d_rho->vh[lev], d_rt->vh[lev], d_exner->vh[lev], gradPi->vl[lev]);
        }
        d_rho->UpdateLocal(); d_rho->HorizToVert();

        // update solutions
        for(int lev = 0; lev < geom->nk; lev++) {
            VecAXPY(velx_j->vh[lev], 1.0, d_u->vh[lev]);
            VecZeroEntries(velx_h->vh[lev]);
            VecAXPY(velx_h->vh[lev], 0.5, velx_i->vh[lev]);
            VecAXPY(velx_h->vh[lev], 0.5, velx_j->vh[lev]);
        }
        velx_j->UpdateLocal();
        velx_h->UpdateLocal();
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(velz_j->vz[ii],  1.0, d_w->vz[ii]    );
            VecAXPY(rho_j->vz[ii],   1.0, d_rho->vz[ii]  );
            VecAXPY(rt_j->vz[ii],    1.0, d_rt->vz[ii]   );
            VecAXPY(exner_j->vz[ii], 1.0, d_exner->vz[ii]);
        }
        rho_j->VertToHoriz();   rho_j->UpdateGlobal();
        rt_j->VertToHoriz();    rt_j->UpdateGlobal();
        exner_j->VertToHoriz(); exner_j->UpdateGlobal();
        velz_j->VertToHoriz();  velz_j->UpdateGlobal();

// explicit density update
for(int lev = 0; lev < geom->nk; lev++) {
    horiz->assemble_residual(lev, theta_h->vl, dudz_i->vl, dudz_j->vl, velz_i->vh, velz_j->vh, exner_h->vh[lev],
                             velx_i->vh[lev], velx_j->vh[lev], rho_i->vh[lev], rho_j->vh[lev], F_u->vh[lev], 
                             _F, _G, velx_i->vl[lev], velx_j->vl[lev], gradPi->vl[lev]);
    MatMult(horiz->EtoF->E21, _F, rho_j->vh[lev]);
    VecAYPX(rho_j->vh[lev], -dt, rho_i->vh[lev]);
}
rho_j->UpdateLocal(); rho_j->HorizToVert();
for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
    ex = ii%topo->nElsX;
    ey = ii/topo->nElsX;

    vert->assemble_residual(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii],
                            rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);
    MatMult(vo->V10, F_z, dF_z);
    VecAXPY(rho_j->vz[ii], -dt, dF_z);
}
rho_j->VertToHoriz(); rho_j->UpdateGlobal();
for(int lev = 0; lev < geom->nk; lev++) {
    VecCopy(rho_j->vh[lev], d_rho->vh[lev]);
    VecAXPY(d_rho->vh[lev], -1.0, rho_i->vh[lev]);
}

        // update additional fields
        horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            VecScale(theta_h->vz[ii], 0.5);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);

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
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        rho_h->VertToHoriz(); rho_h->UpdateGlobal();
        rt_h->VertToHoriz(); rt_h->UpdateGlobal();

        horiz->diagHorizVort(velx_j->vh, dudz_j->vh);
        dudz_j->UpdateLocal();

        GlobalNorms(itt, d_u->vh, velx_j->vh, d_w, velz_j, d_rho, rho_j, d_rt, rt_j, d_exner, exner_j,
                    &max_norm_u, &max_norm_w, &max_norm_rho, &max_norm_rt, &max_norm_exner, h_tmp, u_tmp_1, F_z, true);

        itt++;
        if((max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8) || itt > 20) done = true;
    } while(!done);

    // copy the solutions back to the input vectors
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_j->vh);
    velx_j->CopyTo(velx);

    // write output
    if(save) dump(velx, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
    delete velz_h;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
    delete rho_h;
    delete rt_h;
    delete theta_i;
    delete theta_h;
    delete F_w;
    delete F_rho;
    delete F_rt;
    delete F_exner;
    delete d_w;
    delete d_rho;
    delete d_rt;
    delete d_exner;
    delete velx_i;
    delete velx_j;
    delete velx_h;
    delete dudz_i;
    delete dudz_j;
    delete F_u;
    delete d_u;
    delete gradPi;
    delete lap_d_pi;
    VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&dF);
    VecDestroy(&dG);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
    VecDestroy(&h_tmp);
    VecDestroy(&u_tmp_1);
    VecDestroy(&u_tmp_2);
}

double Euler::ComputeAlpha(Vec* velz_i, Vec* velz_j, Vec* d_velz, 
                           Vec* rho_i, Vec* rho_j, Vec* d_rho, 
                           Vec* rt_i, Vec* rt_j, Vec* d_rt, 
                           Vec* pi_i, Vec* pi_j, Vec* d_pi, Vec* pi_h,
                           Vec* theta_i, Vec* theta_h) {
    bool   done  = false;
    int    ex, ey;
    int    n2    = topo->elOrd*topo->elOrd;
    double alpha = 1.0;
    double c1    = 1.0e-4;
    double dot, f_2_sum, f_2_sum_g = 0.0, dfd, g_2_sum, g_2_sum_g = 0.0, fak, fakp1;
    double dfd_i[999], fak_i[999], fakp1_i[999], alpha_min;
    int    _done, done_g;
    Vec    u_tmp, h_tmp, f_tmp, F_z, G_z, velz_k, rho_k, rt_k, pi_k, frt, theta_k, _tmpB1, _tmpA1, _tmpA2;
    PC     pc;
    KSP    kspCol;
    VertOps* vo = vert->vo;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &u_tmp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &f_tmp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &h_tmp);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &F_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &G_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &velz_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &rho_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &rt_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &pi_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &frt);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+1)*n2, &theta_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*n2, &_tmpB1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &_tmpA1);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*n2, &_tmpA2);

    f_2_sum = g_2_sum = 0.0;
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        ex = ei%topo->nElsX;
        ey = ei/topo->nElsX;

        vo->AssembleLinear(ex, ey, vo->VA);
#ifdef RAYLEIGH
        vo->AssembleRayleigh(ex, ey, vo->VA_inv);
        MatAXPY(vo->VA, 0.5*dt*RAYLEIGH, vo->VA_inv, DIFFERENT_NONZERO_PATTERN);
#endif
        MatMult(vo->VA, d_velz[ei], f_tmp);

        vo->AssembleConst(ex, ey, vo->VB);
        MatMult(vo->VB, pi_h[ei], _tmpB1);
        MatMult(vo->V01, _tmpB1, _tmpA1);
        vo->AssembleLinearInv(ex, ey, vo->VA_inv);
        MatMult(vo->VA_inv, _tmpA1, _tmpA2); // pressure gradient
        vo->AssembleConLinWithW(ex, ey, _tmpA2, vo->VBA);
        MatTranspose(vo->VBA, MAT_REUSE_MATRIX, &vo->VAB);
        vo->AssembleConstWithRhoInv(ex, ey, rho_j[ei], vo->VB_inv);
        MatMatMult(vo->VAB, vo->VB_inv, MAT_REUSE_MATRIX, PETSC_DEFAULT, &vert->pc_V0_invV0_rt_DT);
        MatMatMult(vert->pc_V0_invV0_rt_DT, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &vert->G_rt);

        MatMult(vert->G_rt, d_rt[ei], u_tmp);
        VecAXPY(f_tmp, 0.5*dt, u_tmp);

        vo->AssembleConst(ex, ey, vo->VB);
        MatMatMult(vo->V01, vo->VB, MAT_REUSE_MATRIX, PETSC_DEFAULT, &vert->pc_DTV1);
        vo->AssembleLinearInv(ex, ey, vo->VA_inv);
        MatMatMult(vo->VA_inv, vert->pc_DTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &vert->pc_V0_invDTV1);
        vo->AssembleLinearWithTheta(ex, ey, theta_h[ei], vo->VA);
        MatMatMult(vo->VA, vert->pc_V0_invDTV1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &vert->G_pi);

        MatMult(vert->G_pi, d_pi[ei], u_tmp);
        VecAXPY(f_tmp, 0.5*dt, u_tmp);

        VecDot(f_tmp, f_tmp, &dot);
        f_2_sum += dot;
        dfd_i[ei] = sqrt(dot);

        vert->assemble_residual(ex, ey, theta_h[ei], pi_h[ei], velz_i[ei], velz_j[ei], rho_i[ei], rho_j[ei],
                                rt_i[ei], rt_j[ei], f_tmp, F_z, G_z);

        VecDot(f_tmp, f_tmp, &dot);
        g_2_sum += dot;
        fak_i[ei] = sqrt(dot);
    }
    MPI_Allreduce(&f_2_sum, &f_2_sum_g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&g_2_sum, &g_2_sum_g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    dfd = sqrt(f_2_sum_g);
    fak = sqrt(g_2_sum_g);
    MPI_Bcast(&dfd, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fak, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    KSPCreate(MPI_COMM_SELF, &kspCol);
    KSPSetOperators(kspCol, vo->VA2, vo->VA2);
    KSPGetPC(kspCol, &pc);
    PCSetType(pc, PCLU);
    KSPSetOptionsPrefix(kspCol, "kspCol_");
    KSPSetFromOptions(kspCol);

    do {
        f_2_sum = 0.0;
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            ex = ei%topo->nElsX;
            ey = ei/topo->nElsX;

            VecWAXPY(velz_k, alpha, d_velz[ei], velz_j[ei]);
            VecWAXPY(rho_k,  alpha, d_rho[ei],  rho_j[ei] );
            VecWAXPY(rt_k,   alpha, d_rt[ei],   rt_j[ei]  );
            VecWAXPY(pi_k,   alpha, d_pi[ei],   pi_j[ei]  );
            VecAXPY(pi_k, 1.0, pi_i[ei]);
            VecScale(pi_k, 0.5);

            vo->AssembleLinCon2(ex, ey, vo->VAB2);
            MatMult(vo->VAB2, rt_k, frt);
            vo->AssembleLinearWithRho2(ex, ey, rho_k, vo->VA2);
            KSPSolve(kspCol, frt, theta_k);
            VecScale(theta_k, 0.5);
            VecAXPY(theta_k, 0.5, theta_i[ei]);

            vert->assemble_residual(ex, ey, theta_k, pi_k, velz_i[ei], velz_k, rho_i[ei], rho_k,
                              rt_i[ei], rt_k, f_tmp, F_z, G_z);

            VecDot(f_tmp, f_tmp, &dot);
            f_2_sum += dot;
            fakp1_i[ei] = sqrt(dot);
        }
        MPI_Allreduce(&f_2_sum, &f_2_sum_g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        fakp1 = sqrt(f_2_sum_g);
        MPI_Bcast(&fakp1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //if(!rank) cout << fakp1 << "\t" << fak + c1*alpha*dfd << "\t" << alpha << endl;
        //if(fakp1 > fak + c1*alpha*dfd) alpha = 0.9*alpha;
        //else                           done  = true;

        _done = 1;
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            if(fakp1_i[ei] > fak_i[ei] + c1*alpha*dfd_i[ei]) _done = 0;
        }
        MPI_Allreduce(&_done, &done_g, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if(!done_g) alpha = 0.9*alpha;
        else        done = true;
        if(!rank) cout << done_g << "\t" << alpha << endl;

    } while(!done);

    MPI_Allreduce(&alpha, &alpha_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    alpha = alpha_min;

    VecDestroy(&u_tmp);
    VecDestroy(&h_tmp);
    VecDestroy(&f_tmp);
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&velz_k);
    VecDestroy(&rho_k);
    VecDestroy(&rt_k);
    VecDestroy(&pi_k);
    VecDestroy(&frt);
    VecDestroy(&theta_k);
    VecDestroy(&_tmpB1);
    VecDestroy(&_tmpA1);
    VecDestroy(&_tmpA2);
    KSPDestroy(&kspCol);

    return alpha;
}

double Euler::LineSearch(L1Vecs* velx_i, L1Vecs* velx_j, L1Vecs* d_u, 
                       L2Vecs* velz_i, L2Vecs* velz_j, L2Vecs* d_w,
                       L2Vecs* rho_i,  L2Vecs* rho_j,  L2Vecs* d_rho,
                       L2Vecs* rt_i,   L2Vecs* rt_j,   L2Vecs* d_rt, L2Vecs* theta_i)
{
    int elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double alpha = 1.0;
    double _c = 1.0e-4;
    double norm_l, norm_r;
    Vec tmp_u_x, tmp_h_x, tmp_u_z, tmp_h_z, _F, _G, theta_k, _Fz, _Gz;
    L2Vecs* F_rt = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* F_rt_k = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* theta_h = new L2Vecs(geom->nk+1, topo, geom);
    VertOps* vo = vert->vo;
    bool done = false;

    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &tmp_u_z);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk+0)*elOrd2, &tmp_h_z);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &tmp_u_x);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &tmp_h_x);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_F);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &_G);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &theta_k);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_Fz);
    VecCreateSeq(MPI_COMM_SELF, (geom->nk-1)*elOrd2, &_Gz);

    horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        VecScale(theta_h->vz[ei], 0.5);
        VecAXPY(theta_h->vz[ei], 0.5, theta_i->vz[ei]);
    }
    theta_h->VertToHoriz();
    // compute the residual
    for(int lev = 0; lev < geom->nk; lev++) {
        horiz->diagnose_F(lev, velx_i->vh[lev], velx_j->vh[lev], rho_i->vh[lev], rho_j->vh[lev], _F);
        VecZeroEntries(theta_k);
        VecAXPY(theta_k, 0.5, theta_h->vl[lev+0]);
        VecAXPY(theta_k, 0.5, theta_h->vl[lev+1]);
        horiz->F->assemble(theta_k, lev, false, SCALE);
        MatMult(horiz->F->M, _F, tmp_u_x);
        KSPSolve(horiz->ksp1, tmp_u_x, _G);

        horiz->M2->assemble(lev, SCALE, true);
        MatMult(horiz->EtoF->E21, _G, tmp_h_x);
        MatMult(horiz->M2->M, tmp_h_x, F_rt_k->vh[lev]);
        VecScale(F_rt_k->vh[lev], dt);
    }
    F_rt_k->UpdateLocal(); F_rt_k->HorizToVert();
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        ex = ei%topo->nElsX;
        ey = ei/topo->nElsX;

        vert->diagnose_F_z(ex, ey, velz_i->vz[ei], velz_j->vz[ei], rho_i->vz[ei], rho_j->vz[ei], _Fz);
        vo->AssembleLinearInv(ex, ey, vo->VA_inv);
        vo->AssembleLinearWithTheta(ex, ey, theta_h->vz[ei], vo->VA);
        MatMult(vo->VA, _Fz, vert->_tmpA1);
        MatMult(vo->VA_inv, vert->_tmpA1, _Gz);
        MatMult(vo->V10, _Gz, tmp_h_z);

        VecAYPX(tmp_h_z,   dt, rt_i->vz[ei]);
        VecAXPY(tmp_h_z, -1.0, rt_j->vz[ei]);
        vo->AssembleConst(ex, ey, vo->VB);
        MatMult(vo->VB, tmp_h_z, vert->_tmpB1);
        VecAXPY(F_rt_k->vz[ei], 1.0, vert->_tmpB1);
    }
    F_rt_k->VertToHoriz(); F_rt_k->UpdateGlobal();

    do {
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            VecAXPY(rho_j->vz[ei], +alpha, d_rho->vz[ei]);
            VecAXPY(rt_j->vz[ei],  +alpha, d_rt->vz[ei] );
        }
        horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            VecAXPY(rho_j->vz[ei], -alpha, d_rho->vz[ei]);
            VecAXPY(rt_j->vz[ei],  -alpha, d_rt->vz[ei] );

            VecScale(theta_h->vz[ei], 0.5);
            VecAXPY(theta_h->vz[ei], 0.5, theta_i->vz[ei]);
        }
        theta_h->VertToHoriz();
        // compute the residual
        for(int lev = 0; lev < geom->nk; lev++) {
            VecCopy(velx_j->vh[lev], tmp_u_x);
            VecAXPY(tmp_u_x, alpha, d_u->vh[lev]);
            VecCopy(rho_j->vh[lev], tmp_h_x);
            VecAXPY(tmp_h_x, alpha, d_rho->vh[lev]);
            horiz->diagnose_F(lev, velx_i->vh[lev], tmp_u_x, rho_i->vh[lev], tmp_h_x, _F);
            VecZeroEntries(theta_k);
            VecAXPY(theta_k, 0.5, theta_h->vl[lev+0]);
            VecAXPY(theta_k, 0.5, theta_h->vl[lev+1]);
            horiz->F->assemble(theta_k, lev, false, SCALE);
            MatMult(horiz->F->M, _F, tmp_u_x);
            KSPSolve(horiz->ksp1, tmp_u_x, _G);

            horiz->M2->assemble(lev, SCALE, true);
            MatMult(horiz->EtoF->E21, _G, tmp_h_x);
            MatMult(horiz->M2->M, tmp_h_x, F_rt->vh[lev]);
            VecScale(F_rt->vh[lev], dt);
        }
        F_rt->UpdateLocal(); F_rt->HorizToVert();
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            ex = ei%topo->nElsX;
            ey = ei/topo->nElsX;

            VecCopy(velz_j->vz[ei], tmp_u_z);
            VecAXPY(tmp_u_z, alpha, d_w->vz[ei]);
            VecCopy(rho_j->vz[ei], tmp_h_z);
            VecAXPY(tmp_h_z, alpha, d_rho->vz[ei]);
            vert->diagnose_F_z(ex, ey, velz_i->vz[ei], tmp_u_z, rho_i->vz[ei], tmp_h_z, _Fz);
            vo->AssembleLinearInv(ex, ey, vo->VA_inv);
            vo->AssembleLinearWithTheta(ex, ey, theta_h->vz[ei], vo->VA);
            MatMult(vo->VA, _Fz, vert->_tmpA1);
            MatMult(vo->VA_inv, vert->_tmpA1, _Gz);
            MatMult(vo->V10, _Gz, tmp_h_z);
            VecAYPX(tmp_h_z,     dt, rt_i->vz[ei]);
            VecAXPY(tmp_h_z,   -1.0, rt_j->vz[ei]);
            VecAXPY(tmp_h_z, -alpha, d_rt->vz[ei]);
            vo->AssembleConst(ex, ey, vo->VB);
            MatMult(vo->VB, tmp_h_z, vert->_tmpB1);
            VecAXPY(F_rt->vz[ei], 1.0, vert->_tmpB1);
        }
        F_rt->VertToHoriz(); F_rt->UpdateGlobal();

        schur->RepackFromHoriz(d_rt->vl, schur->x);
        MatMult(schur->S, schur->x, schur->b);
        schur->RepackFromHoriz(F_rt_k->vl, schur->x);
        VecAYPX(schur->b, alpha*_c, schur->x);
        VecNorm(schur->b, NORM_2, &norm_r);

        schur->RepackFromHoriz(F_rt->vl, schur->b);
        VecNorm(schur->b, NORM_2, &norm_l);

        if(!rank) cout << "line search, lhs: " << norm_l << "\trhs: " << norm_r << "\talpha: " << alpha << endl;

        //if(norm_l < norm_r || alpha < 0.1) done = true;
        if(norm_l < 1.1*norm_r || alpha < 0.1) done = true;
        else alpha = 0.8*alpha;
    } while(!done);

    VecDestroy(&tmp_u_z);
    VecDestroy(&tmp_h_z);
    VecDestroy(&tmp_u_x);
    VecDestroy(&tmp_h_x);
    VecDestroy(&_F);
    VecDestroy(&_G);
    VecDestroy(&theta_k);
    VecDestroy(&_Fz);
    VecDestroy(&_Gz);
    delete F_rt;
    delete F_rt_k;
    delete theta_h;

    return alpha;
}
