#include <iostream>
#include <fstream>

#include <cmath>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>

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

using namespace std;

#define RAD_EARTH 6371220.0
#define NK 8
#define P0 100000.0
#define RD 287.0
#define GAMMA 0.005
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define CP 1004.5
#define CV 717.5
#define TE 310.0
#define TP 240.0
#define T0 (0.5*(TE + TP))
#define KP 3.0
#define GAMMA 0.005
#define ZT 15000.0
#define ZTOP 30000.0
#define LAMBDA_C (M_PI/9.0)
#define PHI_C (2.0*M_PI/9.0)
#define VP 1.0
#define D0 (RAD_EARTH/6.0)

double torr_1(double r) {
    double A    = 1.0/GAMMA;
    double B    = (TE - TP)/((TE + TP)*TP);
    double H    = RD*T0/GRAVITY;
    double b    = 2.0;
    double fac  = (r - RAD_EARTH)/(b*H);
    double fac2 = fac*fac;
    
    return (A*GAMMA/T0)*exp(GAMMA*(r - RAD_EARTH)/T0) + B*(1.0 - 2.0*fac2)*exp(-fac2);
}

double torr_2(double r) {
    double C    = 0.5*(KP + 2.0)*(TE - TP)/(TE*TP);
    double H    = RD*T0/GRAVITY;
    double b    = 2.0;
    double fac  = (r - RAD_EARTH)/(b*H);
    double fac2 = fac*fac;
    
    return C*(1.0 - 2.0*fac2)*exp(-fac2);
}

double int_torr_1(double r) {
    double A    = 1.0/GAMMA;
    double B    = (TE - TP)/((TE + TP)*TP);
    double H    = RD*T0/GRAVITY;
    double b    = 2.0;
    double fac  = (r - RAD_EARTH)/(b*H);
    double fac2 = fac*fac;

    return A*(exp(GAMMA*(r - RAD_EARTH)/T0) - 1.0) + B*(r - RAD_EARTH)*exp(-fac2);
}

double int_torr_2(double r) {
    double C    = 0.5*(KP + 2.0)*(TE - TP)/(TE*TP);
    double H    = RD*T0/GRAVITY;
    double b    = 2.0;
    double fac  = (r - RAD_EARTH)/(b*H);
    double fac2 = fac*fac;

    return C*(r - RAD_EARTH)*exp(-fac2);
}

double temp(double* x, double r) {
    double torr1 = torr_1(r);
    double torr2 = torr_2(r);
    double phi   = asin(x[2]/RAD_EARTH);
    double cp    = cos(phi);
    double cpk   = pow(cp, KP);
    double cpkp2 = pow(cp, KP+2.0);
    double fac   = cpk - (KP/(KP+2.0))*cpkp2;
    double Tinv  = torr1 - torr2*fac;

    return 1.0/Tinv;
}

double pres(double* x, double r) {
    double it1   = int_torr_1(r);
    double it2   = int_torr_2(r);
    double phi   = asin(x[2]/RAD_EARTH);
    double cp    = cos(phi);
    double cpk   = pow(cp, KP);
    double cpkp2 = pow(cp, KP+2.0);
    double fac   = cpk - (KP/(KP+2.0))*cpkp2;

    return P0*exp(-GRAVITY*it1/RD + GRAVITY*it2*fac/RD);
}

double u_mean(double* x, double r) {
    double phi   = asin(x[2]/RAD_EARTH);
    double cp    = cos(phi);
    double cpm1  = pow(cp, KP-1.0);
    double cpp1  = pow(cp, KP+1.0);
    double it2   = int_torr_2(r);
    double T     = temp(x, r);
    double U     = (GRAVITY*KP/RAD_EARTH)*it2*(cpm1 - cpp1)*T;

    return -OMEGA*RAD_EARTH*cp + sqrt(OMEGA*OMEGA*RAD_EARTH*RAD_EARTH*cp*cp + RAD_EARTH*cp*U);
}

double z_at_level(double* x, int ki) {
    double mu   = 15.0;
    double frac = (1.0*ki)/NK;

    return ZTOP*(sqrt(mu*frac*frac + 1.0) - 1.0)/(sqrt(mu + 1.0) - 1.0);
}

double z_taper(double* x, int ki) {
    double z    = z_at_level(x, ki);
    double frac = z/ZT;

    if(z > ZT) return 0.0;

    return 1.0 - 3.0*frac*frac + 2.0*frac*frac*frac;
}

double gc_dist(double* x) {
    double phi    = asin(x[2]/RAD_EARTH);
    double lambda = atan2(x[1], x[0]);

    return RAD_EARTH*acos(sin(PHI_C)*sin(phi) + cos(PHI_C)*cos(phi)*cos(lambda - LAMBDA_C));
}

double u_pert(double* x, int ki) {
    double phi    = asin(x[2]/RAD_EARTH);
    double lambda = atan2(x[1], x[0]);
    double gc     = gc_dist(x);
    double zt     = z_taper(x, ki);
    double theta  = 0.5*M_PI*gc/D0;
    double ct     = cos(theta);
    double st     = sin(theta);
    double fac    = -sin(PHI_C)*cos(phi) + cos(PHI_C)*sin(phi)*cos(lambda-LAMBDA_C);

    if(fabs(gc - 0.0           ) < 1.0e-4) return 0.0;
    if(fabs(gc - RAD_EARTH*M_PI) < 1.0e-4) return 0.0;

    if(gc > D0) return 0.0;

    return -16.0*VP*zt/(3.0*sqrt(3.0))*ct*ct*ct*st*fac/sin(gc/RAD_EARTH);
}

double v_pert(double* x, int ki) {
    double lambda = atan2(x[1], x[0]);
    double gc     = gc_dist(x);
    double zt     = z_taper(x, ki);
    double theta  = 0.5*M_PI*gc/D0;
    double ct     = cos(theta);
    double st     = sin(theta);
    double fac    = cos(PHI_C)*sin(lambda-LAMBDA_C);

    if(fabs(gc - 0.0           ) < 1.0e-4) return 0.0;
    if(fabs(gc - RAD_EARTH*M_PI) < 1.0e-4) return 0.0;

    if(gc > D0) return 0.0;

    return +16.0*VP*zt/(3.0*sqrt(3.0))*ct*ct*ct*st*fac/sin(gc/RAD_EARTH);
}

double u_init(double* x, int ki) {
    double zi = 0.5*(z_at_level(x, ki+0) + z_at_level(x, ki+1));
    double um = u_mean(x, zi+RAD_EARTH);
    double up = 0.5*(u_pert(x, ki+0) + u_pert(x, ki+1));

    return  um + up;
}

double v_init(double* x, int ki) {
    double vp = 0.5*(v_pert(x, ki+0) + v_pert(x, ki+1));

    return vp;
}

double theta_init(double* x, int ki) {
    double zi = z_at_level(x, ki);
    double ti = temp(x, zi+RAD_EARTH);
    double pi = pres(x, zi+RAD_EARTH);

    return ti*pow(P0/pi, RD/CP);
}

double rho_init(double* x, int ki) {
    double zi = 0.5*(z_at_level(x, ki+0) + z_at_level(x, ki+1));
    double ti = temp(x, zi+RAD_EARTH);
    double pi = pres(x, zi+RAD_EARTH);

    return pi/(RD*ti);
}

double rt_init(double* x, int ki) {
    double rho   = rho_init(x, ki);
    double theta = 0.5*(theta_init(x, ki+0) + theta_init(x, ki+1));

    return rho*theta;
}

double exner_init(double* x, int ki) {
    double zi = 0.5*(z_at_level(x, ki+0) + z_at_level(x, ki+1));
    double pi = pres(x, zi+RAD_EARTH);

    return CP*pow(pi/P0, RD/CP);
}

double theta_t_init(double* x, int ki) {
    return theta_init(x, NK);
}

double theta_b_init(double* x, int ki) {
    return theta_init(x, 0);
}

double f_topog(double* x) {
    return 0.0;
}

void LoadVecs(Vec* vecs, int nk, char* fieldname, int step) {
    int ki;
    char filename[100];
    PetscViewer viewer;

    for(ki = 0; ki < nk; ki++) {
        sprintf(filename, "output/%s_%.3u_%.4u.vec", fieldname, ki, step);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
        VecLoad(vecs[ki], viewer);
        PetscViewerDestroy(&viewer);
    }
}

void LoadVecsVert(Vec* vecs, int nk, char* fieldname, int step, Topo* topo, Geom* geom) {
    int ki;
    char filename[100];
    PetscViewer viewer;
    L2Vecs* l2Vecs = new L2Vecs(nk, topo, geom);

    for(ki = 0; ki < nk; ki++) {
        sprintf(filename, "output/%s_%.3u_%.4u.vec", fieldname, ki, step);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
        VecLoad(l2Vecs->vh[ki], viewer);
        PetscViewerDestroy(&viewer);
    }
    l2Vecs->UpdateLocal();
    l2Vecs->HorizToVert();
    l2Vecs->CopyToVert(vecs);

    delete l2Vecs;
}

void write(HorizSolve* hs, Vec* velx, L2Vecs* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* theta, int num) {
    char fieldname[100];
    Vec wi;
    Geom* geom = hs->geom;

    if(theta) {
        theta->UpdateGlobal();
        for(int ii = 0; ii < geom->nk+1; ii++) {
            sprintf(fieldname, "theta");
            geom->write2(theta->vh[ii], fieldname, num, ii, false);
        }
    }
    for(int ii = 0; ii < geom->nk; ii++) {
        if(velx) hs->curl(true, velx[ii], &wi, ii, false);

        if(velx) sprintf(fieldname, "vorticity");
        if(velx) hs->geom->write0(wi, fieldname, num, ii);
        if(velx) sprintf(fieldname, "velocity_h");
        if(velx) hs->geom->write1(velx[ii], fieldname, num, ii);
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

int main(int argc, char** argv) {
    int size, rank, step, ki;
    static char help[] = "petsc";
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    double dt = 600.0;//360.0;
    int nSteps = 10*24*10;
    int dumpEvery = 20; //dump every two hours (for now)
    ofstream file;
    Topo* topo;
    Geom* geom;
    Euler* pe;
    Vec *velx;
    L2Vecs *velz, *rho, *rt, *exner;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo, NK);
    // initialise the z coordinate layer heights
    geom->initTopog(f_topog, z_at_level);
    pe   = new Euler(topo, geom, dt);
    pe->step = startStep;

    velx  = new Vec[NK];
    for(ki = 0; ki < NK; ki++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx[ki] );
    }
    velz = new L2Vecs(geom->nk-1, topo, geom);
    rho = new L2Vecs(geom->nk, topo, geom);
    rt = new L2Vecs(geom->nk, topo, geom);
    exner = new L2Vecs(geom->nk, topo, geom);

    if(startStep == 0) {
        pe->init1(velx,      u_init, v_init);
        pe->init2(rho->vh,   rho_init      );
        pe->init2(rt->vh,    rt_init       );
        pe->init2(exner->vh, exner_init    );

        for(ki = 0; ki < NK; ki++) {
            sprintf(fieldname,"velocity_h");
            geom->write1(velx[ki],fieldname,0,ki);
            sprintf(fieldname,"density");
            geom->write2(rho->vh[ki],fieldname,0,ki, true);
            sprintf(fieldname,"rhoTheta");
            geom->write2(rt->vh[ki],fieldname,0,ki, true);
            sprintf(fieldname,"exner");
            geom->write2(exner->vh[ki],fieldname,0,ki, true);
        }
        {
            L2Vecs* theta = new L2Vecs(NK+1, topo, geom);
            rho->UpdateLocal();
            rho->HorizToVert();
            rt->UpdateLocal();
            rt->HorizToVert();
            pe->horiz->diagTheta2(rho->vz, rt->vz, theta->vz);
            theta->VertToHoriz();
            theta->UpdateGlobal();
            for(ki = 0; ki < NK+1; ki++) {
                sprintf(fieldname,"theta");
                geom->write2(theta->vh[ki],fieldname,0,ki,false);
            }
            delete theta;
        }
    } else {
        sprintf(fieldname,"density");
        LoadVecs(rho->vh  , NK, fieldname, startStep);
        sprintf(fieldname,"velocity_h");
        LoadVecs(velx , NK, fieldname, startStep);
        sprintf(fieldname,"rhoTheta");
        LoadVecs(rt->vh   , NK, fieldname, startStep);
        sprintf(fieldname,"exner");
        LoadVecs(exner->vh, NK, fieldname, startStep);
        sprintf(fieldname,"velocity_z");
        LoadVecs(velz->vh, NK-1, fieldname, startStep);
    }
    velz->UpdateLocal();
    velz->HorizToVert();
    rho->UpdateLocal();
    rho->HorizToVert();
    rt->UpdateLocal();
    rt->HorizToVert();
    exner->UpdateLocal();
    exner->HorizToVert();

/*
    pe->solve_vert(velz, rho, rt, true);
    rt->UpdateLocal();
    for(int ii = 0; ii < geom->nk; ii++) {
        pe->diagnose_Pi(ii, rt->vl[ii], rt->vl[ii], exner->vh[ii]);
    }
    exner->UpdateLocal();
    exner->HorizToVert();

    pe->solve_vert_exner(velz, rho, rt, exner, false);
*/
    if(startStep==0) {
        L2Vecs* rho_tmp = new L2Vecs(geom->nk, topo, geom);
        L2Vecs* rt_tmp = new L2Vecs(geom->nk, topo, geom);
        L2Vecs* exner_tmp = new L2Vecs(geom->nk, topo, geom);
        rho_tmp->CopyFromHoriz(rho->vh);
        rt_tmp->CopyFromHoriz(rt->vh);
        exner_tmp->CopyFromHoriz(exner->vh);

        pe->vert->solve_coupled(velz, rho, rt, exner);

        rho->CopyFromHoriz(rho_tmp->vh);
        rt->CopyFromHoriz(rt_tmp->vh);
        exner->CopyFromHoriz(exner_tmp->vh);
        delete rho_tmp;
        delete rt_tmp;
        delete exner_tmp;

        write(pe->horiz, velx, velz, rho, rt, exner, NULL, 1);
    }

    //pe->step = 5;
    //pe->solve_gs(velx, velz, rho, rt, exner, true);
    //write(hs, velx, velz, rho, rt, exner, NULL, 2);
    
    pe->step = 2;
    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        pe->solve_gs(velx, velz, rho, rt, exner, dump);
    }

    for(ki = 0; ki < NK; ki++) {
        VecDestroy(&velx[ki]);
    }
    delete[] velx;
    delete velz;
    delete rho;
    delete rt;
    delete exner;

    delete pe;
    delete geom;
    delete topo;

    PetscFinalize();

    return 0;
}
