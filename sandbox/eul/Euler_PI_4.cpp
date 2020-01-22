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
#include "VertSolve_4.h"
#include "HorizSolve_4.h"
#include "Schur.h"
#include "Euler_PI_4.h"

#define RD 287.0
#define CV 717.5
#define SCALE 1.0e+8

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

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

Euler::~Euler() {
    delete vert;
    delete horiz;
    delete schur;
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
                        double* norm_u, double* norm_w, double* norm_rho, double* norm_rt, double* norm_exner, Vec h_tmp, Vec u_tmp, Vec u_tmp_z) {
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

    if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << *norm_exner << 
                              "\t|d_rho|/|rho|: "     << *norm_rho   <<
                              "\t|d_rt|/|rt|: "       << *norm_rt    <<
                              "\t|d_u|/|u|: "         << *norm_u     <<
                              "\t|d_w|/|w|: "         << *norm_w     << endl;
}

// Gauss-Seidel splitting of horiztonal and vertical pressure updates (horiztonal then vertical)
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
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta;
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

            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);

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
        //F_rho->UpdateLocal(); F_rho->HorizToVert();
        //F_rt->UpdateLocal();  F_rt->HorizToVert();

        // build the preconditioner matrix (horiztonal part first)
        MatZeroEntries(schur->M);
        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "assembling operators for level: " << lev << endl;

            //horiz->assemble_and_update(lev, theta_h->vl, velx_j->vl[lev], velx_j->vh[lev], rho_j->vl[lev], rt_j->vl[lev], exner_h->vl[lev],
            horiz->assemble_and_update(lev, theta_h->vl, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
                                       F_u->vh[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], gradPi->vl[lev]);

            schur->AddFromHorizMat(lev, horiz->_PCx);
            MatDestroy(&horiz->_PCx);
        }
        F_rho->UpdateLocal();       F_rho->HorizToVert();
        F_rt->UpdateLocal();        F_rt->HorizToVert();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            //vert->assemble_and_update(ex, ey, theta_h->vz[ii], velz_j->vz[ii], rho_j->vz[ii], rt_j->vz[ii], exner_h->vz[ii],
            vert->assemble_and_update(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
                                      F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii]);

            schur->AddFromVertMat(ii, vert->_PCz);
        }
        F_rt->VertToHoriz();

        // solve for the exner pressure update
        VecZeroEntries(schur->b);
        schur->RepackFromHoriz(F_rt->vl, schur->b);
        VecScale(schur->b, -1.0);
        schur->Solve(d_rt);

        // update the delta vectors
        d_rt->HorizToVert(); d_rt->UpdateGlobal();

        // back substitution
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            //vert->update_deltas(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_h->vz[ii],
            vert->update_deltas(ex, ey, theta_h->vz[ii], velz_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii],
                                F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], d_w->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii]);
        }
        d_exner->VertToHoriz(); d_exner->UpdateGlobal();
        d_rho->VertToHoriz();   d_rho->UpdateGlobal();

        for(int lev = 0; lev < geom->nk; lev++) {
            if(!rank) cout << "updating corrections for level: " << lev << endl;

            //horiz->update_deltas(lev, theta_h->vl, velx_j->vl[lev], velx_j->vh[lev], rho_j->vl[lev], rt_j->vl[lev], exner_h->vl[lev],
            horiz->update_deltas(lev, theta_h->vl, velx_h->vl[lev], velx_h->vh[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
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
                    &max_norm_u, &max_norm_w, &max_norm_rho, &max_norm_rt, &max_norm_exner, h_tmp, u_tmp_1, F_z);

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
}
