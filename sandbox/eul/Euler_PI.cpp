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
#include "VertSolve.h"
#include "HorizSolve.h"
#include "Euler_PI.h"

#define SCALE 1.0e+8

using namespace std;

Euler::Euler(Topo* _topo, Geom* _geom, double _dt) {
    dt = _dt;
    topo = _topo;
    geom = _geom;

    step = 0;
    firstStep = true;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    vert  = new VertSolve(topo, geom, dt);
    horiz = new HorizSolve(topo, geom, dt);

    schur = new Schur(topo, geom, false);

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

double GlobalNorm(int n, Vec* dx, Vec* xx) {
    double norm_xx_sq, norm_dx_sq, norm_sq;

    norm_xx_sq = norm_dx_sq = 0.0;

    for(int ii = 0; ii < n; ii++) {
        VecDot(dx[ii], dx[ii], &norm_sq);
        norm_dx_sq += norm_sq;
        VecDot(xx[ii], xx[ii], &norm_sq);
        norm_xx_sq += norm_sq;
    }
    return sqrt(norm_dx_sq / norm_xx_sq);
}

void Euler::solve(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_w, max_norm_rho, max_norm_rt, max_norm_exner, norm_x;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
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
    Vec* velx_j = new Vec[geom->nk];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    Vec* F_u    = new Vec[geom->nk];
    Vec* d_u    = new Vec[geom->nk];
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta;
    VertOps* vo = vert->vo;

    for(int lev = 0; lev < geom->nk; lev++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[lev]);
        VecCopy(velx_i[lev], velx_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &F_u[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &d_u[lev]);
    }
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

    // diagnose the vorticity terms
    horiz->diagHorizVort(velx_i, dudz_i);
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i[lev], dudz_j[lev]);
    }
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    do {
        max_norm_u = max_norm_w = max_norm_rho = max_norm_rt = max_norm_exner = 0.0;

        // residual vectors (vertical components)
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // exner pressure residual vectors
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);

            // vertical velocity residual vectors
            vert->assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);

            // density and density weighted potential temperature residual vectors (vertical components)
            vo->AssembleConst(ex, ey, vo->VB);
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

        // residual vectors (horizontal components)
        for(int lev = 0; lev < geom->nk; lev++) {
            // horizontal velocity residual vectors
            horiz->assemble_residual_x(lev, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
                                velx_i[lev], velx_j[lev], rho_i->vh[lev], rho_j->vh[lev], F_u[lev], _F, _G);

            // density and density weighted potential temperature residual vectors (horizontal components)
            horiz->M2->assemble(lev, SCALE, true);
            MatMult(horiz->EtoF->E21, _F, dF);
            MatMult(horiz->EtoF->E21, _G, dG);
            MatMult(horiz->M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);
            MatMult(horiz->M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);

            if(horiz->do_visc) {
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
        }

        // build the preconditioner matrix
        MatZeroEntries(schur->M);
        if(schur->precon) MatZeroEntries(schur->P);
        for(int lev = 0; lev < geom->nk; lev++) {
            //horiz->assemble_and_update(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_j->vl[lev], 
            horiz->assemble_and_update(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_h->vl[lev], 
                                       F_u[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], true, false, NULL, NULL);

            schur->AddFromHorizMat(lev, horiz->_PCx, schur->M);
            if(schur->precon) {
                horiz->T->assemble(exner_j->vl[lev], lev, SCALE, true);
                schur->AddFromHorizMat(lev, horiz->T->M, schur->P);
            }
        }
        F_rho->UpdateLocal();   F_rho->HorizToVert();
        F_rt->UpdateLocal();    F_rt->HorizToVert();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            //vert->assemble_and_update(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], 
            vert->assemble_and_update(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_h->vz[ii], 
                                      F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], false, false);

            schur->AddFromVertMat(ii, vert->_PCz);
            VecScale(F_rt->vz[ii], -1.0);
        }
        F_rt->VertToHoriz(); 

        // solve for the exner pressure update
        VecZeroEntries(schur->b);
        schur->RepackFromHoriz(F_rt->vl, schur->b);

        schur->Solve(d_exner);

        // update the delta vectors
        d_exner->HorizToVert();
        d_exner->UpdateGlobal();

        for(int lev = 0; lev < geom->nk; lev++) {
            //horiz->set_deltas(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_j->vl[lev], 
            horiz->set_deltas(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_h->vl[lev], 
                       F_u[lev], F_rho->vh[lev], F_exner->vh[lev], d_u[lev], d_rho->vh[lev], d_rt->vh[lev], d_exner->vh[lev], false, false, NULL, NULL);
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            VecZeroEntries(d_rt->vz[ii]);

            //vert->set_deltas(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], 
            vert->set_deltas(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_h->vz[ii], 
                       F_w->vz[ii], F_rho->vz[ii], F_exner->vz[ii], d_w->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii], true, false);

            // inverse assembled in function above
            MatMult(vo->VB_inv, F_rho->vz[ii], d_rho->vz[ii]);
            VecScale(d_rho->vz[ii], -1.0);
            VecScale(d_rt->vz[ii], -1.0);
        }

        // update solutions
        for(int lev = 0; lev < geom->nk; lev++) {
            VecAXPY(velx_j[lev], 1.0, d_u[lev]);
            max_norm_u = MaxNorm(d_u[lev], velx_j[lev], max_norm_u);
        }
        MPI_Allreduce(&max_norm_u, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_u = norm_x;
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(velz_j->vz[ii],  1.0, d_w->vz[ii]    );
            VecAXPY(rho_j->vz[ii],   1.0, d_rho->vz[ii]  );
            VecAXPY(rt_j->vz[ii],    1.0, d_rt->vz[ii]   );
            VecAXPY(exner_j->vz[ii], 1.0, d_exner->vz[ii]);
            max_norm_w = MaxNorm(d_w->vz[ii], velz_j->vz[ii], max_norm_w);
            max_norm_rho = MaxNorm(d_rho->vz[ii], rho_j->vz[ii], max_norm_rho);
            max_norm_rt  = MaxNorm(d_rt->vz[ii],  rt_j->vz[ii],  max_norm_rt );
        }
        rho_j->VertToHoriz();   rho_j->UpdateGlobal();
        rt_j->VertToHoriz();    rt_j->UpdateGlobal();
        exner_j->VertToHoriz(); exner_j->UpdateGlobal();
        velz_j->VertToHoriz();  velz_j->UpdateGlobal();
        MPI_Allreduce(&max_norm_w, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w = norm_x;
        MPI_Allreduce(&max_norm_rho, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho = norm_x;
        MPI_Allreduce(&max_norm_rt,  &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt  = norm_x;

        VecZeroEntries(schur->b);
        schur->RepackFromHoriz(exner_j->vl, schur->b);
        VecNorm(schur->b, NORM_2, &norm_x);
        VecNorm(schur->x, NORM_2, &max_norm_exner);
        max_norm_exner /= norm_x;

        // update additional fields
        horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            VecScale(theta_h->vz[ii], 0.5);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
        }
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        horiz->diagHorizVort(velx_j, dudz_j);

        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                  "\t|d_rho|/|rho|: "     << max_norm_rho   <<
                                  "\t|d_rt|/|rt|: "       << max_norm_rt    <<
                                  "\t|d_u|/|u|: "         << max_norm_u     <<
                                  "\t|d_w|/|w|: "         << max_norm_w     << endl;

        firstStep = false;
        if(max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8) done = true;
        itt++;
        if(itt > 20) done = true;
    } while(!done);

    // copy the solutions back to the input vectors
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_j->vh);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecCopy(velx_j[kk], velx_i[kk]);
    }

    // write output
    if(save) dump(velx_i, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
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
    for(int lev = 0; lev < geom->nk; lev++) {
        VecDestroy(&velx_j[lev]);
        VecDestroy(&dudz_i[lev]);
        VecDestroy(&dudz_j[lev]);
        VecDestroy(&F_u[lev]);
        VecDestroy(&d_u[lev]);
    }
    delete[] velx_j;
    delete[] dudz_i;
    delete[] dudz_j;
    delete[] F_u;
    delete[] d_u;
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

//#define COLUMN_SOLVE
void Euler::solve_vert(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_w, max_norm_rho, max_norm_rt, max_norm_exner, norm_x;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
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
    Vec F_z, G_z, dF_z, dG_z;
    VertOps* vo = vert->vo;
#ifdef COLUMN_SOLVE
    PC pc;
    KSP ksp_exner;
#endif

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

    // diagnose the vorticity terms
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    do {
        max_norm_w = max_norm_rho = max_norm_rt = max_norm_exner = 0.0;

        // residual vectors (vertical components)
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // exner pressure residual vectors
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);

            // vertical velocity residual vectors
            vert->assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);

            // density and density weighted potential temperature residual vectors (vertical components)
            vo->AssembleConst(ex, ey, vo->VB);
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

        // build the preconditioner matrix
        MatZeroEntries(schur->M);
        if(schur->precon) MatZeroEntries(schur->P);
        F_rho->UpdateLocal();   F_rho->HorizToVert();
        F_rt->UpdateLocal();    F_rt->HorizToVert();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            vert->assemble_and_update(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], 
                                      F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], true, true);

            VecScale(F_rt->vz[ii], -1.0);
#ifdef COLUMN_SOLVE
            KSPCreate(MPI_COMM_SELF, &ksp_exner);
            KSPSetOperators(ksp_exner, vert->_PCz, vert->_PCz);
            KSPGetPC(ksp_exner, &pc);
            PCSetType(pc, PCLU);
            KSPSetOptionsPrefix(ksp_exner, "ksp_exner_");
            KSPSetFromOptions(ksp_exner);
            KSPSolve(ksp_exner, F_rt->vz[ii], d_exner->vz[ii]);
            KSPDestroy(&ksp_exner);
#else
            schur->AddFromVertMat(ii, vert->_PCz);
#endif
        }
        F_rt->VertToHoriz(); 

#ifdef COLUMN_SOLVE
        d_exner->VertToHoriz();
#else
        // solve for the exner pressure update
        VecZeroEntries(schur->b);
        schur->RepackFromHoriz(F_rt->vl, schur->b);
        MatAssemblyBegin(schur->M, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  schur->M, MAT_FINAL_ASSEMBLY);
        KSPSolve(schur->ksp, schur->b, schur->x);
        schur->UnpackToHoriz(schur->x, d_exner->vl);
#endif

        // update the delta vectors
        d_exner->HorizToVert();
        d_exner->UpdateGlobal();
        d_rho->UpdateLocal(); d_rho->HorizToVert();
        
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            VecZeroEntries(d_rt->vz[ii]);

            vert->set_deltas(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], 
                       F_w->vz[ii], F_rho->vz[ii], F_exner->vz[ii], d_w->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii], true, false);

            // inverse assembled in function above
            MatMult(vo->VB_inv, F_rho->vz[ii], d_rho->vz[ii]);
            VecScale(d_rho->vz[ii], -1.0);
            VecScale(d_rt->vz[ii], -1.0);
        }

        // update solutions
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(velz_j->vz[ii],  1.0, d_w->vz[ii]    );
            VecAXPY(rho_j->vz[ii],   1.0, d_rho->vz[ii]  );
            VecAXPY(rt_j->vz[ii],    1.0, d_rt->vz[ii]   );
            VecAXPY(exner_j->vz[ii], 1.0, d_exner->vz[ii]);
            max_norm_w = MaxNorm(d_w->vz[ii], velz_j->vz[ii], max_norm_w);
            max_norm_rho = MaxNorm(d_rho->vz[ii], rho_j->vz[ii], max_norm_rho);
            max_norm_rt  = MaxNorm(d_rt->vz[ii],  rt_j->vz[ii],  max_norm_rt );
#ifdef COLUMN_SOLVE
            max_norm_exner = MaxNorm(d_exner->vz[ii], exner_j->vz[ii], max_norm_exner);
#endif
        }
        rho_j->VertToHoriz();   rho_j->UpdateGlobal();
        rt_j->VertToHoriz();    rt_j->UpdateGlobal();
        exner_j->VertToHoriz(); exner_j->UpdateGlobal();
        velz_j->VertToHoriz();  velz_j->UpdateGlobal();
        MPI_Allreduce(&max_norm_w, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w = norm_x;
        MPI_Allreduce(&max_norm_rho, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho = norm_x;
        MPI_Allreduce(&max_norm_rt,  &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt  = norm_x;

#ifndef COLUMN_SOLVE
        VecZeroEntries(schur->b);
        schur->RepackFromHoriz(exner_j->vl, schur->b);
        VecNorm(schur->b, NORM_2, &norm_x);
        VecNorm(schur->x, NORM_2, &max_norm_exner);
        max_norm_exner /= norm_x;
#endif

        // update additional fields
        horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            VecScale(theta_h->vz[ii], 0.5);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
        }
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();

        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                  "\t|d_rho|/|rho|: "     << max_norm_rho   <<
                                  "\t|d_rt|/|rt|: "       << max_norm_rt    <<
                                  "\t|d_w|/|w|: "         << max_norm_w     << endl;

        firstStep = false;
        if(max_norm_exner < 1.0e-8 && max_norm_w < 1.0e-8) done = true;
        itt++;
    } while(!done);

    // write output
    if(save) dump(velx_i, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
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
    VecDestroy(&F_z);
    VecDestroy(&G_z);
    VecDestroy(&dF_z);
    VecDestroy(&dG_z);
}

#define LAYER_SOLVE
void Euler::solve_horiz(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_rho, max_norm_rt, max_norm_exner, norm_x;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
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
    Vec* velx_j = new Vec[geom->nk];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    Vec* F_u    = new Vec[geom->nk];
    Vec* d_u    = new Vec[geom->nk];
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta;
    VertOps* vo = vert->vo;
#ifdef LAYER_SOLVE
    int size;
    PC pc;
    KSP ksp_exner;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    for(int lev = 0; lev < geom->nk; lev++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[lev]);
        VecCopy(velx_i[lev], velx_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &F_u[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &d_u[lev]);
    }
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

    // diagnose the vorticity terms
    horiz->diagHorizVort(velx_i, dudz_i);
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i[lev], dudz_j[lev]);
    }
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    do {
        max_norm_u = max_norm_rho = max_norm_rt = max_norm_exner = 0.0;

        // residual vectors (vertical components)
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // exner pressure residual vectors
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);

            // vertical velocity residual vectors
            VecZeroEntries(F_z);
            VecZeroEntries(G_z);

            // density and density weighted potential temperature residual vectors (vertical components)
            vo->AssembleConst(ex, ey, vo->VB);
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

        // residual vectors (horizontal components)
        for(int lev = 0; lev < geom->nk; lev++) {
            // horizontal velocity residual vectors
            horiz->assemble_residual_x(lev, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
                                velx_i[lev], velx_j[lev], rho_i->vh[lev], rho_j->vh[lev], F_u[lev], _F, _G);

            // density and density weighted potential temperature residual vectors (horizontal components)
            horiz->M2->assemble(lev, SCALE, true);
            MatMult(horiz->EtoF->E21, _F, dF);
            MatMult(horiz->EtoF->E21, _G, dG);
            MatMult(horiz->M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);
            MatMult(horiz->M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);

            if(horiz->do_visc) {
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
        }

        // build the preconditioner matrix
        MatZeroEntries(schur->M);
        if(schur->precon) MatZeroEntries(schur->P);
        for(int lev = 0; lev < geom->nk; lev++) {
            //horiz->assemble_and_update(lev, theta_i->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_i->vl[lev], 
            //horiz->assemble_and_update(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_j->vl[lev], 
            horiz->assemble_and_update(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_h->vl[lev], 
                                       F_u[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], true, false, NULL, NULL);

            VecScale(F_rt->vh[lev], -1.0);
#ifdef LAYER_SOLVE
            //VecScale(F_rt->vh[lev], -1.0);
            KSPCreate(MPI_COMM_WORLD, &ksp_exner);
            MatAssemblyBegin(horiz->_PCx, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(  horiz->_PCx, MAT_FINAL_ASSEMBLY);
            KSPSetOperators(ksp_exner, horiz->_PCx, horiz->_PCx);
            KSPSetTolerances(ksp_exner, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
            KSPSetType(ksp_exner, KSPGMRES);
            KSPGetPC(ksp_exner, &pc);
            PCSetType(pc, PCBJACOBI);
            PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
            KSPSetOptionsPrefix(ksp_exner, "ksp_exner_x_");
            KSPSetFromOptions(ksp_exner);
            KSPSolve(ksp_exner, F_rt->vh[lev], d_exner->vh[lev]);
            KSPDestroy(&ksp_exner);
#else
            schur->AddFromHorizMat(lev, horiz->_PCx, schur->M);
            if(schur->precon) {
                horiz->T->assemble(exner_j->vl[lev], lev, SCALE, true);
                schur->AddFromHorizMat(lev, horiz->T->M, schur->P);
                //schur->AddFromHorizMat(lev, horiz->M2_pi_inv->M, schur->P);
            }
#endif
        }
        F_rho->UpdateLocal();   F_rho->HorizToVert();
        F_rt->UpdateLocal();    F_rt->HorizToVert();

#ifdef LAYER_SOLVE
        d_exner->UpdateLocal();
        d_exner->HorizToVert();
#else
        //for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        //    VecScale(F_rt->vz[ii], -1.0);
        //}
        //F_rt->VertToHoriz(); 

        // solve for the exner pressure update
        VecZeroEntries(schur->b);
        schur->RepackFromHoriz(F_rt->vl, schur->b);

        schur->Solve(d_exner);

        // update the delta vectors
        d_exner->HorizToVert();
        d_exner->UpdateGlobal();
#endif
        for(int lev = 0; lev < geom->nk; lev++) {
            VecAXPY(exner_j->vh[lev], 1.0, d_exner->vh[lev]);
        }
        exner_j->UpdateLocal();
        exner_j->HorizToVert();

        for(int lev = 0; lev < geom->nk; lev++) {
            //horiz->set_deltas(lev, theta_i->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_i->vl[lev], 
            //horiz->set_deltas(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_j->vl[lev], 
            horiz->set_deltas(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_h->vl[lev], 
                       F_u[lev], F_rho->vh[lev], F_exner->vh[lev], d_u[lev], d_rho->vh[lev], d_rt->vh[lev], d_exner->vh[lev], true, false, NULL, NULL);
                       //F_u[lev], F_rho->vh[lev], F_exner->vh[lev], d_u[lev], d_rho->vh[lev], d_rt->vh[lev], d_exner->vh[lev], false, false, NULL, NULL);
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        d_rt->UpdateLocal(); d_rt->HorizToVert();
        
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            // inverse assembled in function above
            vo->AssembleConstInv(ii%topo->nElsX, ii/topo->nElsX, vo->VB_inv);
            MatMult(vo->VB_inv, F_rho->vz[ii], d_rho->vz[ii]);
            VecScale(d_rho->vz[ii], -1.0);
            VecScale(d_rt->vz[ii], -1.0);
        }

        // update solutions
        for(int lev = 0; lev < geom->nk; lev++) {
            VecAXPY(velx_j[lev], 1.0, d_u[lev]);
            max_norm_u = MaxNorm(d_u[lev], velx_j[lev], max_norm_u);
        }
        MPI_Allreduce(&max_norm_u, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_u = norm_x;
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecAXPY(rho_j->vz[ii],   1.0, d_rho->vz[ii]  );
            VecAXPY(rt_j->vz[ii],    1.0, d_rt->vz[ii]   );
            max_norm_rho = MaxNorm(d_rho->vz[ii], rho_j->vz[ii], max_norm_rho);
            max_norm_rt  = MaxNorm(d_rt->vz[ii],  rt_j->vz[ii],  max_norm_rt );
            max_norm_exner = MaxNorm(d_exner->vz[ii], exner_j->vz[ii], max_norm_exner);
        }
        rho_j->VertToHoriz();   rho_j->UpdateGlobal();
        rt_j->VertToHoriz();    rt_j->UpdateGlobal();
        MPI_Allreduce(&max_norm_rho, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho = norm_x;
        MPI_Allreduce(&max_norm_rt,  &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt  = norm_x;
        MPI_Allreduce(&max_norm_exner, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_exner = norm_x;

/*
        VecZeroEntries(schur->b);
        schur->RepackFromHoriz(exner_j->vl, schur->b);
        VecNorm(schur->b, NORM_2, &norm_x);
        VecNorm(schur->x, NORM_2, &max_norm_exner);
        max_norm_exner /= norm_x;
*/

        // update additional fields
        horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            VecScale(theta_h->vz[ii], 0.5);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
        }
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        horiz->diagHorizVort(velx_j, dudz_j);

        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                  "\t|d_rho|/|rho|: "     << max_norm_rho   <<
                                  "\t|d_rt|/|rt|: "       << max_norm_rt    <<
                                  "\t|d_u|/|u|: "         << max_norm_u     << endl;

        firstStep = false;
        if(max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8) done = true;
        itt++;
    } while(!done);

    // write output
    if(save) dump(velx_i, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
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
    for(int lev = 0; lev < geom->nk; lev++) {
        VecDestroy(&velx_j[lev]);
        VecDestroy(&dudz_i[lev]);
        VecDestroy(&dudz_j[lev]);
        VecDestroy(&F_u[lev]);
        VecDestroy(&d_u[lev]);
    }
    delete[] velx_j;
    delete[] dudz_i;
    delete[] dudz_j;
    delete[] F_u;
    delete[] d_u;
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

void Euler::solve_split(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false, do_horiz = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey, size;
    double max_norm_u, max_norm_w, max_norm_rho, max_norm_rt, max_norm_exner, norm_x;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
    L2Vecs* rho_j   = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* rt_j    = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_j = new L2Vecs(geom->nk, topo, geom);
    L2Vecs* exner_h = new L2Vecs(geom->nk, topo, geom);
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
    Vec* velx_j = new Vec[geom->nk];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    Vec* F_u    = new Vec[geom->nk];
    Vec* d_u    = new Vec[geom->nk];
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta;
    VertOps* vo = vert->vo;
    KSP ksp_exner;
    PC pc;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for(int lev = 0; lev < geom->nk; lev++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[lev]);
        VecCopy(velx_i[lev], velx_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &F_u[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &d_u[lev]);
    }
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

    // diagnose the vorticity terms
    horiz->diagHorizVort(velx_i, dudz_i);
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i[lev], dudz_j[lev]);
    }
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    do {
        max_norm_u = max_norm_w = max_norm_rho = max_norm_rt = max_norm_exner = 0.0;

        // residual vectors (vertical components)
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // exner pressure residual vectors
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);

            // vertical velocity residual vectors
            if(firstStep || !do_horiz) {
                vert->assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);
            } else {
                VecZeroEntries(F_z);
                VecZeroEntries(G_z);
            }

            // density and density weighted potential temperature residual vectors (vertical components)
            vo->AssembleConst(ex, ey, vo->VB);
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

        // residual vectors (horizontal components)
        for(int lev = 0; lev < geom->nk; lev++) {
            // horizontal velocity residual vectors
            if(firstStep || do_horiz) {
                horiz->assemble_residual_x(lev, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
                                velx_i[lev], velx_j[lev], rho_i->vh[lev], rho_j->vh[lev], F_u[lev], _F, _G);
            } else {
                VecZeroEntries(_F);
                VecZeroEntries(_G);
            }

            // density and density weighted potential temperature residual vectors (horizontal components)
            horiz->M2->assemble(lev, SCALE, true);
            MatMult(horiz->EtoF->E21, _F, dF);
            MatMult(horiz->EtoF->E21, _G, dG);
            MatMult(horiz->M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);
            MatMult(horiz->M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);

            if(horiz->do_visc && (firstStep || do_horiz)) {
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
        }

        // build the preconditioner matrix
        MatZeroEntries(schur->M);
        if(do_horiz || firstStep) {
            for(int lev = 0; lev < geom->nk; lev++) {
                horiz->assemble_and_update(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_j->vl[lev], 
                                           F_u[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], true, false, NULL, NULL);

                if(firstStep) {
                    schur->AddFromHorizMat(lev, horiz->_PCx, schur->M);
                } else {
                    VecScale(F_rt->vh[lev], -1.0);
                    KSPCreate(MPI_COMM_WORLD, &ksp_exner);
                    MatAssemblyBegin(horiz->_PCx, MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(  horiz->_PCx, MAT_FINAL_ASSEMBLY);
                    KSPSetOperators(ksp_exner, horiz->_PCx, horiz->_PCx);
                    KSPSetTolerances(ksp_exner, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
                    KSPSetType(ksp_exner, KSPGMRES);
                    KSPGetPC(ksp_exner, &pc);
                    PCSetType(pc, PCBJACOBI);
                    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
                    KSPSetOptionsPrefix(ksp_exner, "ksp_exner_x_");
                    KSPSetFromOptions(ksp_exner);
                    KSPSolve(ksp_exner, F_rt->vh[lev], d_exner->vh[lev]);
                    KSPDestroy(&ksp_exner);
                }
            }
        }
        F_rho->UpdateLocal();   F_rho->HorizToVert();
        F_rt->UpdateLocal();    F_rt->HorizToVert();
        if(!firstStep && do_horiz) {
            d_exner->UpdateLocal(); d_exner->HorizToVert();
        }

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            if(!do_horiz || firstStep) {
                ex = ii%topo->nElsX;
                ey = ii/topo->nElsX;
                vert->assemble_and_update(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], 
                                          F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], !firstStep, !firstStep);//??

                if(firstStep) {
                    schur->AddFromVertMat(ii, vert->_PCz);
                }
                VecScale(F_rt->vz[ii], -1.0);
            }
            if(!do_horiz && !firstStep) {
                KSPCreate(MPI_COMM_SELF, &ksp_exner);
                KSPSetOperators(ksp_exner, vert->_PCz, vert->_PCz);
                KSPGetPC(ksp_exner, &pc);
                PCSetType(pc, PCLU);
                KSPSetOptionsPrefix(ksp_exner, "ksp_exner_z_");
                KSPSetFromOptions(ksp_exner);
                KSPSolve(ksp_exner, F_rt->vz[ii], d_exner->vz[ii]);
                KSPDestroy(&ksp_exner);
            }
        }
        if(!do_horiz && !firstStep) {
            d_exner->VertToHoriz();
            d_exner->UpdateGlobal();
        }
        F_rt->VertToHoriz(); 

        // solve for the exner pressure update
        if(firstStep) {
            VecZeroEntries(schur->b);
            schur->RepackFromHoriz(F_rt->vl, schur->b);
            schur->Solve(d_exner);
        }

        // update the delta vectors
        d_exner->HorizToVert();
        d_exner->UpdateGlobal();

        if(do_horiz || firstStep) {
            for(int lev = 0; lev < geom->nk; lev++) {
                horiz->set_deltas(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_j->vl[lev], 
                       F_u[lev], F_rho->vh[lev], F_exner->vh[lev], d_u[lev], d_rho->vh[lev], d_rt->vh[lev], d_exner->vh[lev], !firstStep, false, NULL, NULL);
            }
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            if(!do_horiz || firstStep) {
                VecZeroEntries(d_rt->vz[ii]);

                vert->set_deltas(ex, ey, theta_i->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], 
                       F_w->vz[ii], F_rho->vz[ii], F_exner->vz[ii], d_w->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii], true, false);
            } else {
                vo->AssembleConstInv(ex, ey, vo->VB_inv);
            }

            // inverse assembled in function above
            MatMult(vo->VB_inv, F_rho->vz[ii], d_rho->vz[ii]);
            VecScale(d_rho->vz[ii], -1.0);
            VecScale(d_rt->vz[ii], -1.0);
        }

        // update solutions
        if(do_horiz || firstStep) {
            for(int lev = 0; lev < geom->nk; lev++) {
                VecAXPY(velx_j[lev], 1.0, d_u[lev]);
                max_norm_u = MaxNorm(d_u[lev], velx_j[lev], max_norm_u);
            }
            MPI_Allreduce(&max_norm_u, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_u = norm_x;
        }
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            if(!do_horiz || firstStep) {
                VecAXPY(velz_j->vz[ii],  1.0, d_w->vz[ii]    );
                max_norm_w = MaxNorm(d_w->vz[ii], velz_j->vz[ii], max_norm_w);
            }
            VecAXPY(rho_j->vz[ii],   1.0, d_rho->vz[ii]  );
            VecAXPY(rt_j->vz[ii],    1.0, d_rt->vz[ii]   );
            VecAXPY(exner_j->vz[ii], 1.0, d_exner->vz[ii]);
            max_norm_rho = MaxNorm(d_rho->vz[ii], rho_j->vz[ii], max_norm_rho);
            max_norm_rt  = MaxNorm(d_rt->vz[ii],  rt_j->vz[ii],  max_norm_rt );
        }
        rho_j->VertToHoriz();   rho_j->UpdateGlobal();
        rt_j->VertToHoriz();    rt_j->UpdateGlobal();
        exner_j->VertToHoriz(); exner_j->UpdateGlobal();
        MPI_Allreduce(&max_norm_rho, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rho = norm_x;
        MPI_Allreduce(&max_norm_rt,  &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_rt  = norm_x;
        if(!do_horiz || firstStep) {
            velz_j->VertToHoriz();  velz_j->UpdateGlobal();
            MPI_Allreduce(&max_norm_w, &norm_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); max_norm_w = norm_x;
        }

        // update additional fields
        horiz->diagTheta2(rho_j->vz, rt_j->vz, theta_h->vz);
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            VecZeroEntries(exner_h->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_i->vz[ii]);
            VecAXPY(exner_h->vz[ii], 0.5, exner_j->vz[ii]);

            VecScale(theta_h->vz[ii], 0.5);
            VecAXPY(theta_h->vz[ii], 0.5, theta_i->vz[ii]);
        }
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        horiz->diagHorizVort(velx_j, dudz_j);

        if(!rank) cout << itt << ":\t|d_exner|/|exner|: " << max_norm_exner << 
                                  "\t|d_rho|/|rho|: "     << max_norm_rho   <<
                                  "\t|d_rt|/|rt|: "       << max_norm_rt    <<
                                  "\t|d_u|/|u|: "         << max_norm_u     <<
                                  "\t|d_w|/|w|: "         << max_norm_w     << endl;

        firstStep = false;
        do_horiz = !do_horiz;
        if(max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8) done = true;
        itt++;
    } while(!done);

    // write output
    if(save) dump(velx_i, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
    delete rho_j;
    delete rt_j;
    delete exner_j;
    delete exner_h;
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
    for(int lev = 0; lev < geom->nk; lev++) {
        VecDestroy(&velx_j[lev]);
        VecDestroy(&dudz_i[lev]);
        VecDestroy(&dudz_j[lev]);
        VecDestroy(&F_u[lev]);
        VecDestroy(&d_u[lev]);
    }
    delete[] velx_j;
    delete[] dudz_i;
    delete[] dudz_j;
    delete[] F_u;
    delete[] d_u;
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

// TODO (fixing drift over subsequent time steps):
// *) solve as vertical then horizontal gauss-seidel splitting at each newton iteration
// *) alternate vertical/horizontal and horizontal/vertical gauss-seidel splitting between time steps
// *) is the exner pressure residual assembly ok??
// *) do the viscous terms at the new time level only

// Gauss-Seidel splitting of horiztonal and vertical pressure updates (horiztonal then vertical)
void Euler::solve_gs(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_w, max_norm_rho, max_norm_rt, max_norm_exner;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
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
    L2Vecs* lap_d_exner = new L2Vecs(geom->nk, topo, geom);
    Vec* velx_j = new Vec[geom->nk];
    Vec* velx_h = new Vec[geom->nk];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    Vec* F_u    = new Vec[geom->nk];
    Vec* d_u    = new Vec[geom->nk];
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta;
    VertOps* vo = vert->vo;

    for(int lev = 0; lev < geom->nk; lev++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[lev]);
        VecCopy(velx_i[lev], velx_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_h[lev]);
        VecCopy(velx_i[lev], velx_h[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &F_u[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &d_u[lev]);
    }
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

    // diagnose the vorticity terms
    horiz->diagHorizVort(velx_i, dudz_i);
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i[lev], dudz_j[lev]);
    }
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    // just to initialise everything...
    vert->assemble_and_update(0, 0, theta_h->vz[0], velz_i->vz[0], rho_i->vz[0], rt_i->vz[0], exner_h->vz[0], F_z, dF_z, dF_z, dF_z, true, true);

    do {
        // residual vectors (vertical components)
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // exner pressure residual vectors
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);

            // vertical velocity residual vectors
            vert->assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);

            // density and density weighted potential temperature residual vectors (vertical components)
            vo->AssembleConst(ex, ey, vo->VB);
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

        // residual vectors (horizontal components)
        for(int lev = 0; lev < geom->nk; lev++) {
            // horizontal velocity residual vectors
            horiz->assemble_residual_x(lev, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
                                velx_i[lev], velx_j[lev], rho_i->vh[lev], rho_j->vh[lev], F_u[lev], _F, _G);

            // density and density weighted potential temperature residual vectors (horizontal components)
            horiz->M2->assemble(lev, SCALE, true);
            MatMult(horiz->EtoF->E21, _F, dF);
            MatMult(horiz->EtoF->E21, _G, dG);
            MatMult(horiz->M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);
            MatMult(horiz->M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);

            if(horiz->do_visc) {
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
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        F_rt->UpdateLocal();  F_rt->HorizToVert();

        // updata the density weighted potential temperature residuals with the vertical velocity residual gradient
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            //vert->update_residuals(ex, ey, theta_h->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_i->vz[ii], 
            vert->update_residuals(ex, ey, theta_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii], 
                                   F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii]);

            if(itt) {
                MatMult(vert->_PCz, d_exner->vz[ii], lap_d_exner->vz[ii]);
            }
        }
        F_rho->VertToHoriz(); F_rho->UpdateGlobal();
        F_rt->VertToHoriz();  F_rt->UpdateGlobal();
        if(itt) {
            lap_d_exner->VertToHoriz(); lap_d_exner->UpdateGlobal();
        }

        // build the preconditioner matrix
        for(int lev = 0; lev < geom->nk; lev++) {
            //horiz->assemble_and_update(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_h->vl[lev], 
            //horiz->assemble_and_update(lev, theta_h->vl, velx_i[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
            horiz->assemble_and_update(lev, theta_h->vl, velx_h[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
                                       F_u[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], false, false, velz_i, velz_j); // eos update done with vert residual update

            VecScale(F_rt->vh[lev], -1.0);
            VecCopy(F_rt->vh[lev], h_tmp);
            if(itt) {
                VecAXPY(h_tmp, 1.0, lap_d_exner->vh[lev]);
            }
            KSPSolve(horiz->ksp_exner, h_tmp, d_exner->vh[lev]);
            // replace DM^{-1}G\delta\Pi with horiztonal update
            if(itt) {
                MatMult(horiz->pcx_LAP, d_exner->vh[lev], lap_d_exner->vh[lev]);
            }
        }
        F_rho->UpdateLocal();       F_rho->HorizToVert();
        F_rt->UpdateLocal();        F_rt->HorizToVert();
        lap_d_exner->UpdateLocal(); lap_d_exner->HorizToVert();

        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;
            //vert->assemble_pc(ex, ey, theta_h->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_h->vz[ii], true);
            vert->assemble_pc(ex, ey, theta_h->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii], true);

            VecCopy(F_rt->vz[ii], dF_z);
            VecAXPY(dF_z, 1.0, lap_d_exner->vz[ii]);
            KSPSolve(vert->ksp_exner, dF_z, d_exner->vz[ii]);
        }
        d_exner->VertToHoriz(); d_exner->UpdateGlobal();

        // back substitution
        for(int lev = 0; lev < geom->nk; lev++) {
            //horiz->set_deltas(lev, theta_h->vl, velx_i[lev], rho_i->vl[lev], rt_i->vl[lev], exner_h->vl[lev], 
            //horiz->set_deltas(lev, theta_h->vl, velx_i[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev], 
            horiz->set_deltas(lev, theta_h->vl, velx_h[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev], 
                       F_u[lev], F_rho->vh[lev], F_exner->vh[lev], d_u[lev], d_rho->vh[lev], d_rt->vh[lev], d_exner->vh[lev], false, false, velz_i, velz_j);
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            VecZeroEntries(d_rt->vz[ii]);

            //vert->set_deltas(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_i->vz[ii], rt_i->vz[ii], exner_h->vz[ii], 
            vert->set_deltas(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii], 
                       F_w->vz[ii], F_rho->vz[ii], F_exner->vz[ii], d_w->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii], true, false);

            // inverse assembled in function above
            MatMult(vo->VB_inv, F_rho->vz[ii], d_rho->vz[ii]);
            VecScale(d_rho->vz[ii], -1.0);
            VecScale(d_rt->vz[ii], -1.0);
        }

        // update solutions
        for(int lev = 0; lev < geom->nk; lev++) {
            VecAXPY(velx_j[lev], 1.0, d_u[lev]);
            VecZeroEntries(velx_h[lev]);
            VecAXPY(velx_h[lev], 0.5, velx_i[lev]);
            VecAXPY(velx_h[lev], 0.5, velx_j[lev]);
        }
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
        }
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        rho_h->VertToHoriz(); rho_h->UpdateGlobal();
        rt_h->VertToHoriz(); rt_h->UpdateGlobal();
        horiz->diagHorizVort(velx_j, dudz_j);

        d_rho->VertToHoriz(); d_rho->UpdateGlobal();
        d_rt->VertToHoriz(); d_rt->UpdateGlobal();
        GlobalNorms(itt, d_u, velx_j, d_w, velz_j, d_rho, rho_j, d_rt, rt_j, d_exner, exner_j,
                    &max_norm_u, &max_norm_w, &max_norm_rho, &max_norm_rt, &max_norm_exner, h_tmp, u_tmp_1, F_z);

        firstStep = false;
        if(max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8) done = true;
        itt++;
        if(itt > 20) done = true;
    } while(!done);

    // copy the solutions back to the input vectors
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_j->vh);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecCopy(velx_j[kk], velx_i[kk]);
    }

    // write output
    if(save) dump(velx_i, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
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
    delete lap_d_exner;
    for(int lev = 0; lev < geom->nk; lev++) {
        VecDestroy(&velx_j[lev]);
        VecDestroy(&velx_h[lev]);
        VecDestroy(&dudz_i[lev]);
        VecDestroy(&dudz_j[lev]);
        VecDestroy(&F_u[lev]);
        VecDestroy(&d_u[lev]);
    }
    delete[] velx_j;
    delete[] velx_h;
    delete[] dudz_i;
    delete[] dudz_j;
    delete[] F_u;
    delete[] d_u;
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

// Gauss-Seidel splitting of horiztonal and vertical pressure updates (vertical then horizontal)
void Euler::solve_gs_vh(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save) {
    bool done = false;
    int itt = 0, elOrd2 = topo->elOrd*topo->elOrd, ex, ey;
    double max_norm_u, max_norm_w, max_norm_rho, max_norm_rt, max_norm_exner;
    L2Vecs* velz_j  = new L2Vecs(geom->nk-1, topo, geom);
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
    L2Vecs* lap_d_exner = new L2Vecs(geom->nk, topo, geom);
    Vec* velx_j = new Vec[geom->nk];
    Vec* dudz_i = new Vec[geom->nk];
    Vec* dudz_j = new Vec[geom->nk];
    Vec* F_u    = new Vec[geom->nk];
    Vec* d_u    = new Vec[geom->nk];
    Vec _F, _G, dF, dG, F_z, G_z, dF_z, dG_z, h_tmp, u_tmp_1, u_tmp_2, dtheta;
    VertOps* vo = vert->vo;

    for(int lev = 0; lev < geom->nk; lev++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx_j[lev]);
        VecCopy(velx_i[lev], velx_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_i[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dudz_j[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &F_u[lev]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &d_u[lev]);
    }
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

    // diagnose the vorticity terms
    horiz->diagHorizVort(velx_i, dudz_i);
    for(int lev = 0; lev < geom->nk; lev++) {
        VecCopy(dudz_i[lev], dudz_j[lev]);
    }
    horiz->diagTheta2(rho_i->vz, rt_i->vz, theta_i->vz);
    theta_h->CopyFromVert(theta_i->vz);
    theta_i->VertToHoriz();
    theta_h->VertToHoriz();
    theta_h->UpdateGlobal();

    // just to initialise everything...
    horiz->assemble_and_update(0, theta_h->vl, velx_i[0], rho_h->vl[0], rt_h->vl[0], exner_h->vl[0], _F, dF, dF, dF, true, false, NULL, NULL);

    do {
        // residual vectors (vertical components)
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            // exner pressure residual vectors
            vo->Assemble_EOS_Residual(ex, ey, rt_j->vz[ii], exner_j->vz[ii], F_exner->vz[ii]);

            // vertical velocity residual vectors
            vert->assemble_residual_z(ex, ey, theta_h->vz[ii], exner_h->vz[ii], velz_i->vz[ii], velz_j->vz[ii], rho_i->vz[ii], rho_j->vz[ii], 
                                rt_i->vz[ii], rt_j->vz[ii], F_w->vz[ii], F_z, G_z);

            // density and density weighted potential temperature residual vectors (vertical components)
            vo->AssembleConst(ex, ey, vo->VB);
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

        // residual vectors (horizontal components)
        for(int lev = 0; lev < geom->nk; lev++) {
            // horizontal velocity residual vectors
            horiz->assemble_residual_x(lev, theta_h->vl, dudz_i, dudz_j, velz_i->vh, velz_j->vh, exner_h->vh[lev], 
                                velx_i[lev], velx_j[lev], rho_i->vh[lev], rho_j->vh[lev], F_u[lev], _F, _G);

            // density and density weighted potential temperature residual vectors (horizontal components)
            horiz->M2->assemble(lev, SCALE, true);
            MatMult(horiz->EtoF->E21, _F, dF);
            MatMult(horiz->EtoF->E21, _G, dG);
            MatMult(horiz->M2->M, dF, h_tmp);
            VecAXPY(F_rho->vh[lev], dt, h_tmp);
            MatMult(horiz->M2->M, dG, h_tmp);
            VecAXPY(F_rt->vh[lev], dt, h_tmp);

            if(horiz->do_visc) {
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
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        F_rt->UpdateLocal();  F_rt->HorizToVert();

        // update the density weighted potential temperature residuals with the vertical velocity residual gradient
        for(int lev = 0; lev < geom->nk; lev++) {
            horiz->update_residuals(lev, theta_h->vl, velx_i[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev],
                                    F_u[lev], F_rho->vh[lev], F_rt->vh[lev], F_exner->vh[lev], NULL, NULL);

            if(itt) {
                MatMult(horiz->_PCx, d_exner->vh[lev], lap_d_exner->vh[lev]);
            }
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        F_rt->UpdateLocal();  F_rt->HorizToVert();
        if(itt) {
            lap_d_exner->UpdateLocal(); lap_d_exner->HorizToVert();
        }

        // build the preconditioner matrix
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            vert->assemble_and_update(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii], 
                                      F_w->vz[ii], F_rho->vz[ii], F_rt->vz[ii], F_exner->vz[ii], false, true); // eos update for F_rt done with horiz residual update

            VecScale(F_rt->vz[ii], -1.0);
            VecCopy(F_rt->vz[ii], dF_z);
            if(itt) {
                VecAXPY(dF_z, 1.0, lap_d_exner->vz[ii]);
            }
            KSPSolve(vert->ksp_exner, dF_z, d_exner->vz[ii]);
            // replace DM^{-1}G\delta\Pi with horiztonal update
            if(itt) {
                MatMult(vert->pc_LAP, d_exner->vz[ii], lap_d_exner->vz[ii]);
            }
        }
        F_rho->VertToHoriz();       F_rho->UpdateGlobal();
        F_rt->VertToHoriz();        F_rt->UpdateGlobal();
        lap_d_exner->VertToHoriz(); lap_d_exner->UpdateGlobal();

        for(int lev = 0; lev < geom->nk; lev++) {
            horiz->assemble_pc(lev, theta_h->vl, F_u[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev], true, NULL, NULL);

            VecCopy(F_rt->vh[lev], dF);
            VecAXPY(dF, 1.0, lap_d_exner->vh[lev]);
            KSPSolve(horiz->ksp_exner, dF, d_exner->vh[lev]);
        }
        d_exner->UpdateLocal(); d_exner->HorizToVert();

        // back substitution
        for(int lev = 0; lev < geom->nk; lev++) {
            horiz->set_deltas(lev, theta_h->vl, velx_i[lev], rho_h->vl[lev], rt_h->vl[lev], exner_h->vl[lev], 
                       F_u[lev], F_rho->vh[lev], F_exner->vh[lev], d_u[lev], d_rho->vh[lev], d_rt->vh[lev], d_exner->vh[lev], false, false, NULL, NULL);
        }
        F_rho->UpdateLocal(); F_rho->HorizToVert();
        
        for(int ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
            ex = ii%topo->nElsX;
            ey = ii/topo->nElsX;

            VecZeroEntries(d_rt->vz[ii]);

            vert->set_deltas(ex, ey, theta_h->vz[ii], velz_i->vz[ii], rho_h->vz[ii], rt_h->vz[ii], exner_h->vz[ii], 
                       F_w->vz[ii], F_rho->vz[ii], F_exner->vz[ii], d_w->vz[ii], d_rho->vz[ii], d_rt->vz[ii], d_exner->vz[ii], true, false);

            // inverse assembled in function above
            MatMult(vo->VB_inv, F_rho->vz[ii], d_rho->vz[ii]);
            VecScale(d_rho->vz[ii], -1.0);
            VecScale(d_rt->vz[ii], -1.0);
        }

        // update solutions
        for(int lev = 0; lev < geom->nk; lev++) {
            VecAXPY(velx_j[lev], 1.0, d_u[lev]);
        }
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
        }
        theta_h->VertToHoriz(); theta_h->UpdateGlobal();
        exner_h->VertToHoriz(); exner_h->UpdateGlobal();
        rho_h->VertToHoriz(); rho_h->UpdateGlobal();
        rt_h->VertToHoriz(); rt_h->UpdateGlobal();
        horiz->diagHorizVort(velx_j, dudz_j);

        d_rho->VertToHoriz(); d_rho->UpdateGlobal();
        d_rt->VertToHoriz(); d_rt->UpdateGlobal();
        GlobalNorms(itt, d_u, velx_j, d_w, velz_j, d_rho, rho_j, d_rt, rt_j, d_exner, exner_j,
                    &max_norm_u, &max_norm_w, &max_norm_rho, &max_norm_rt, &max_norm_exner, h_tmp, u_tmp_1, F_z);

        firstStep = false;
        if(max_norm_exner < 1.0e-8 && max_norm_u < 1.0e-8 && max_norm_w < 1.0e-8) done = true;
        itt++;
        if(itt > 10) done = true;
    } while(!done);

    // copy the solutions back to the input vectors
    velz_i->CopyFromHoriz(velz_j->vh);
    rho_i->CopyFromHoriz(rho_j->vh);
    rt_i->CopyFromHoriz(rt_j->vh);
    exner_i->CopyFromHoriz(exner_j->vh);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecCopy(velx_j[kk], velx_i[kk]);
    }

    // write output
    if(save) dump(velx_i, velz_i, rho_i, rt_i, exner_i, theta_h, step++);

    delete velz_j;
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
    delete lap_d_exner;
    for(int lev = 0; lev < geom->nk; lev++) {
        VecDestroy(&velx_j[lev]);
        VecDestroy(&dudz_i[lev]);
        VecDestroy(&dudz_j[lev]);
        VecDestroy(&F_u[lev]);
        VecDestroy(&d_u[lev]);
    }
    delete[] velx_j;
    delete[] dudz_i;
    delete[] dudz_j;
    delete[] F_u;
    delete[] d_u;
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
