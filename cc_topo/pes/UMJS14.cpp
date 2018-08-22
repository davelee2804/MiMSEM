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
#include "Assembly.h"
#include "PrimEqns_HEVI3.h"

using namespace std;

#define RAD_EARTH 6371220.0
#define NK 30
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

    return RAD_EARTH/cos(sin(PHI_C)*sin(phi) + cos(PHI_C)*cos(phi)*cos(lambda - LAMBDA_C));
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

void LoadVecs(Vec* vecs, int nk, char* fieldname, int step, bool para) {
    int ki;
    char filename[100];
    PetscViewer viewer;

    for(ki = 0; ki < NK; ki++) {
        sprintf(filename, "output/%s_%.4u_%.4u.vec", fieldname, ki, step);
        if(para) {
            PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        }
        else {
            PetscViewerBinaryOpen(PETSC_COMM_SELF, fieldname, FILE_MODE_READ, &viewer);
        }
        VecLoad(vecs[ki], viewer);
        PetscViewerDestroy(&viewer);
    }
}

int main(int argc, char** argv) {
    int size, rank, step, ii, ki, n2;
    static char help[] = "petsc";
    //double vort_0, mass_0, ener_0;
    //double vort_n, mass_n, ener_n;
    char fieldname[50];//, filename[50];
    bool dump;
    int startStep = atoi(argv[1]);
    double dt = 60.0;
    int nSteps = 48*60;
    int dumpEvery = 30;
    ofstream file;
    Topo* topo;
    Geom* geom;
    PrimEqns_HEVI3* pe;
    Vec *velx, *velz, *rho, *rt, *exner;
    PetscViewer viewer;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo, NK);
    // initialise the z coordinate layer heights
    geom->initTopog(f_topog, z_at_level);
    pe   = new PrimEqns_HEVI3(topo, geom, dt);
    pe->step = startStep;

    n2 = topo->nElsX*topo->nElsX;

    velx  = new Vec[NK];
    rho   = new Vec[NK];
    rt    = new Vec[NK];
    exner = new Vec[NK];
    velz  = new Vec[n2];
    for(ki = 0; ki < NK; ki++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx[ki] );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho[ki]  );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner[ki]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt[ki]   );
    }
    for(ii = 0; ii < n2; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (NK-1)*topo->elOrd*topo->elOrd, &velz[ii]);
        VecZeroEntries(velz[ii]);
    }

    // initialise the potential temperature top and bottom boundary conditions
    // these are 2d fields at the top and bottom, so re-set the thicknesses to
    // unity before initialising, then reset to correct values afterwards
    geom->initTopog(f_topog, NULL);
    pe->initTheta(pe->theta_b, theta_b_init);
    pe->initTheta(pe->theta_t, theta_t_init);
    geom->initTopog(f_topog, z_at_level);

    if(startStep == 0) {
        pe->init1(velx, u_init, v_init);
        pe->init2(rho,   rho_init  );
        pe->init2(exner, exner_init);
        pe->init2(rt,    rt_init   );

        for(ki = 0; ki < NK; ki++) {
            sprintf(fieldname,"velocity_h");
            geom->write1(velx[ki],fieldname,0,ki);
            sprintf(fieldname,"density");
            geom->write2(rho[ki],fieldname,0,ki, true);
            sprintf(fieldname,"exner");
            geom->write2(exner[ki],fieldname,0,ki, true);
            sprintf(fieldname,"rhoTheta");
            geom->write2(rt[ki],fieldname,0,ki, true);
        }
        for(ii = 0; ii < n2; ii++) {
            sprintf(fieldname, "output/velocity_z_%.4u_%.4u.vec", ii, 0);
            PetscViewerBinaryOpen(MPI_COMM_SELF, fieldname, FILE_MODE_WRITE, &viewer);
            VecView(velz[ii], viewer);
            PetscViewerDestroy(&viewer);
        }
    } else {
        sprintf(fieldname,"density");
        LoadVecs(rho  , NK, fieldname, startStep, true );
        sprintf(fieldname,"velocity_h");
        LoadVecs(velx , NK, fieldname, startStep, true );
        sprintf(fieldname,"exner");
        LoadVecs(exner, NK, fieldname, startStep, true );
        sprintf(fieldname,"rhoTheta");
        LoadVecs(rt   , NK, fieldname, startStep, true );
        sprintf(fieldname,"velociyt_z");
        LoadVecs(velz , n2, fieldname, startStep, false);
    }

    //vort_0 = mass_0 = ener_0 = 0.0;
    //for(ki = 0; ki < NK; ki++) {
    //    pe->curl(velx[ki], &wi, 0, false);
    //    vort_0 += sw->int0(wi);
    //    mass_0 += sw->int2(rho[ki]);
    //    ener_0 += sw->intE(velx[ki], rho[ki]);
    //    VecDestroy(&wi);
    //}

    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        pe->SolveStrang(velx, velz, rho, rt, exner, dump);

        //if(dump) {
            //vort_n = mass_n = ener_n = 0.0;
            //for(ki = 0; ki < NK; ki++) {
                //pe->curl(velx[ki], &wi, 0, false);
                //vort_n += sw->int0(wi);
                //mass_n += sw->int2(rho[ki]);
                //ener_n += sw->intE(velx[ki], rho[ki]);
                //VecDestroy(&wi);

                //sprintf(filename, "output/conservation.dat");
                //file.open(filename, ios::out | ios::app);
                //file << (step*dt)/60.0/60.0/24.0 << "\t" << (mass_n-mass_0)/mass_0 
                //                                 << "\t" << (vort_n-vort_0) 
                //                                 << "\t" << (ener_n-ener_0)/ener_0 << endl;
                //file.close();
            //}
        //}
    }

    delete pe;
    delete geom;
    delete topo;

    for(ki = 0; ki < NK; ki++) {
        VecDestroy(&velx[ki] );
        VecDestroy(&rho[ki]  );
        VecDestroy(&rt[ki]   );
        VecDestroy(&exner[ki]);
    }
    for(ii = 0; ii < n2; ii++) {
        VecDestroy(&velz[ii]);
    }
    delete[] velx;
    delete[] rho;
    delete[] exner;
    delete[] rt;
    delete[] velz;

    PetscFinalize();

    return 0;
}
