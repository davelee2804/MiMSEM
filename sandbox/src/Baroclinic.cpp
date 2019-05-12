#include <iostream>

#include <cmath>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn_2L.h"

/*
  Bottom level:
    H_b   = 10000.0
    rho_b =     0.749152837846
    vel_b =    18.1584170112

  Top level:
    H_t   = 20000.0
    rho_t =     0.131703004382
    vel_t =    14.801012961
 */

using namespace std;

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define DELTA 100.0
//#define HTOP 8000.0
//#define HBOT 2000.0
//#define UTOP 21.4589470412
//#define UBOT 4.95652139969
//#define RHO_TOP 0.661798009439
//#define RHO_BOT 1.09857215148
#define HTOP 10000.0
#define HBOT 2000.0
#define UTOP 40.0
#define UBOT 0.0
#define RHO_TOP 0.600192080827
#define RHO_BOT 1.09857215148

double vel_func(double* x, double U0) {
    double eps = 1.0e-8;
    double phi0 = M_PI/7.0;
    double phi1 = M_PI/2.0 - phi0;
    double en = exp(-4.0/((phi1 - phi0)*(phi1 - phi0)));
    double phi = asin(x[2]/RAD_SPHERE);

    if(phi > phi0 + eps && phi < phi1 - eps) {
        return (U0/en)*exp(1.0/((phi - phi0)*(phi - phi1)));
    }
    else {
        return 0.0;
    }
}

double v_init(double* x) {
    double eps = 1.0e-3;

    return eps*2.0*(rand()/RAND_MAX - 0.5);
}

double hgt_func(double* x, int level) {
    int ii, ni = 10000;
    double phiPrime = 0.0;
    double phi = asin(x[2]/RAD_SPHERE);
    double dphi = fabs(phi/ni);
    double h = (level == 1) ? HTOP : HBOT;
    double scale = (level == 1) ? 1.0/GRAVITY/(1.0 - RHO_TOP/RHO_BOT) : 1.0/GRAVITY/(1.0 - RHO_BOT/RHO_TOP);
    double bot_fac = (level == 1) ? 1.0 : RHO_BOT/RHO_TOP; 
    double ut, ub, f;
    double x2[3];
    int sgn = (phi > 0) ? +1 : -1;

    x2[0] = x[0];
    x2[1] = x[1];
    for(ii = 0; ii < ni; ii++) {
        phiPrime += sgn*dphi;
        x2[2] = RAD_SPHERE*sin(phiPrime);
        f = 2.0*OMEGA*sin(phiPrime);
        ut = vel_func(x2, UTOP);
        ub = vel_func(x2, UBOT);
        h -= scale*RAD_SPHERE*ut*(f + tan(phiPrime)*ut/RAD_SPHERE)*dphi;
        h += scale*RAD_SPHERE*ub*(f + tan(phiPrime)*ub/RAD_SPHERE)*dphi*bot_fac;
    }

    return h;
}

double h_top_init(double* x) {
    return hgt_func(x, 1);
/*
    return HTOP;
    int ii, ni = 1000;
    double phiPrime = 0.0;
    double phi = asin(x[2]/RAD_SPHERE);
    double dphi = fabs(phi/ni);
    double h = HTOP;
    double u, f;
    double x2[3];
    int sgn = (phi > 0) ? +1 : -1;

    x2[0] = x[0];
    x2[1] = x[1];
    for(ii = 0; ii < ni; ii++) {
        phiPrime += sgn*dphi;
        x2[2] = RAD_SPHERE*sin(phiPrime);
        u = u_init(x2);
        f = 2.0*OMEGA*sin(phiPrime);
        h -= RAD_SPHERE*u*(f + tan(phiPrime)*u/RAD_SPHERE)*dphi/GRAVITY;
    }
    return h;
*/
}

double h_bot_init(double* x) {
    return hgt_func(x, 2);
/*
    double phi   = asin(x[2]/RAD_SPHERE);
    double phi_0 = 0.25*M_PI;
    double y     = RAD_SPHERE*(phi - phi_0);
    double L     = RAD_SPHERE*M_PI/40.0;

    if(x[2]/RAD_SPHERE > 0.9) return HBOT + DELTA;

    return HBOT - DELTA*tanh(y/L);
    int ii, ni = 1000;
    double phiPrime = 0.0;
    double phi = asin(x[2]/RAD_SPHERE);
    double dphi = fabs(phi/ni);
    double h = HBOT;
    double u, f;
    double x2[3];
    int sgn = (phi > 0) ? +1 : -1;

    x2[0] = x[0];
    x2[1] = x[1];
    for(ii = 0; ii < ni; ii++) {
        phiPrime += sgn*dphi;
        x2[2] = RAD_SPHERE*sin(phiPrime);
        u = u_init(x2);
        f = 2.0*OMEGA*sin(phiPrime);
        h -= RAD_SPHERE*u*(f + tan(phiPrime)*u/RAD_SPHERE)*dphi/GRAVITY;
    }

    return h;
*/
}

double h_pert_init(double* x) {
    double h = 0.0;
    double hHat = 120.0*(RAD_SPHERE/RAD_EARTH);
    double phi = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double alpha = 1.0/3.0;
    double beta = 1.0/15.0;
    double phi2 = M_PI/4.0;

    h += hHat*cos(phi)*exp(-1.0*(lambda/alpha)*(lambda/alpha))*exp(-1.0*((phi2 - phi)/beta)*((phi2 - phi)/beta));

    return h;    
}

double u_top_init(double* x) {
    double eps = 1.0e-4;

    return vel_func(x, UTOP) + eps*2.0*(rand()/RAND_MAX - 0.5);
/*
    double phi   = asin(x[2]/RAD_SPHERE);
    double f     = 2.0*OMEGA*sin(phi);
    double phi_0 = 0.25*M_PI;
    double y     = RAD_SPHERE*(phi - phi_0);
    double L     = RAD_SPHERE*M_PI/40.0;

    if(fabs(phi - phi_0) > 0.25*M_PI*(7.0/8.0)) return 0.0;

    return (GRAVITY*DELTA/f/L)*(1.0 - tanh(y/L)*tanh(y/L));
*/
}

double u_bot_init(double* x) {
    double eps = 1.0e-4;

    return vel_func(x, UBOT) + eps*2.0*(rand()/RAND_MAX - 0.5);
//    return u_top_init(x) + eps*2.0*(rand() - 0.5)/RAND_MAX;
}

// initial condition given by:
//     Galewsky, Scott and Polvani (2004) Tellus, 56A 429-440
double u_init(double* x) {
    double eps = 1.0e-8;
    double umax = 80.0*(RAD_SPHERE/RAD_EARTH);
    double phi0 = M_PI/7.0;
    double phi1 = M_PI/2.0 - phi0;
    //double phi0 = -M_PI/4.0;
    //double phi1 = +M_PI/4.0;
    double en = exp(-4.0/((phi1 - phi0)*(phi1 - phi0)));
    double phi = asin(x[2]/RAD_SPHERE);

    if(phi > phi0 + eps && phi < phi1 - eps) {
        return (umax/en)*exp(1.0/((phi - phi0)*(phi - phi1)));
    }
    else {
        return 0.0;
    }
}

double h_init(double* x) {
    int ii, ni = 1000;
    double phiPrime = 0.0;
    //double phiPrime = -0.5*M_PI;
    double phi = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    //double dphi = phi/ni;
    double dphi = fabs(phi/ni);
    double hHat = 120.0*(RAD_SPHERE/RAD_EARTH);
    double h = 10000.0*(RAD_SPHERE/RAD_EARTH);
    double grav = 9.80616*(RAD_SPHERE/RAD_EARTH);
    double omega = 7.292e-5;
    double u, f;
    double alpha = 1.0/3.0;
    double beta = 1.0/15.0;
    double phi2 = M_PI/4.0;
    double x2[3];
    int sgn = (phi > 0) ? +1 : -1;
    //int sgn = +1;

    x2[0] = x[0];
    x2[1] = x[1];
    for(ii = 0; ii < ni; ii++) {
        phiPrime += sgn*dphi;
        x2[2] = RAD_SPHERE*sin(phiPrime);
        u = u_init(x2);
        f = 2.0*omega*sin(phiPrime);
        h -= RAD_SPHERE*u*(f + tan(phiPrime)*u/RAD_SPHERE)*dphi/grav;
    }

    h += hHat*cos(phi)*exp(-1.0*(lambda/alpha)*(lambda/alpha))*exp(-1.0*((phi2 - phi)/beta)*((phi2 - phi)/beta));

    return h;
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    double dt = 300.0;//15.0;
    double vort_0, mass_0, ener_0;
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    int nSteps = 60*24*12;//240;
    int dumpEvery = 6*12;//240;
    Topo* topo;
    Geom* geom;
    SWEqn_2L* sw;
    Vec ut, ht, wt;
    Vec ub, hb, wb;
    PetscViewer viewer;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo);
    sw = new SWEqn_2L(topo, geom);
    sw->step = startStep;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ut);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &ht);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ub);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hb);

    if(startStep == 0) {
        srand(101);

        sw->init1(ut, u_top_init, v_init);
        sw->init2(ht, h_top_init);
        sw->init1(ub, u_bot_init, v_init);
        sw->init2(hb, h_bot_init);
/*
        sw->init1(ut, u_init, v_init);
        sw->init2(ht, h_init);
        sw->init1(ub, u_init, v_init);
        sw->init2(hb, h_init);
*/

        sprintf(fieldname,"velocity_t");
        geom->write1(ut,fieldname,0);
        sprintf(fieldname,"pressure_t");
        geom->write2(ht,fieldname,0);
        sprintf(fieldname,"velocity_b");
        geom->write1(ub,fieldname,0);
        sprintf(fieldname,"pressure_b");
        geom->write2(hb,fieldname,0);
    } else {
        sprintf(fieldname, "output/pressure_t_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(ht, viewer);
        PetscViewerDestroy(&viewer);

        sprintf(fieldname, "output/velocity_t_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(ut, viewer);
        PetscViewerDestroy(&viewer);

        sprintf(fieldname, "output/pressure_b_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(hb, viewer);
        PetscViewerDestroy(&viewer);

        sprintf(fieldname, "output/velocity_b_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(ub, viewer);
        PetscViewerDestroy(&viewer);

        /*{
            Vec hPert;
            VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hPert);
            sw->init2(hPert, h_pert_init);
            VecAXPY(ht, 1.0, hPert);
            VecDestroy(&hPert);
        }*/
    }

    sw->curl(ut, &wt);
    sw->curl(ub, &wb);
    vort_0 = sw->int0(wt) + sw->int0(wb);
    mass_0 = sw->int2(ht) + sw->int2(hb);
    ener_0 = sw->intE(GRAVITY, ut, ht) + sw->intE(GRAVITY, ub, hb);
    VecDestroy(&wt);
    VecDestroy(&wb);

    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        sw->solve(ut, ub, ht, hb, dt, dump);
        if(dump) {
            sw->writeConservation(step*dt, ut, ub, ht, hb, mass_0, vort_0, ener_0);
        }
    }

    delete sw;
    delete geom;
    delete topo;

    VecDestroy(&ut);
    VecDestroy(&ht);
    VecDestroy(&ub);
    VecDestroy(&hb);

    PetscFinalize();

    return 0;
}
