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

  Top level:
    H_t   = 20000.0
    rho_t =     0.131703004382
 */

using namespace std;

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define DELTA 100.0
#define HTOP 20000.0
#define HBOT 10000.0

double u_init(double* x) {
    double eps = 1.0e-8;
    double umax = 80.0*(RAD_SPHERE/RAD_EARTH);
    double phi0 = M_PI/7.0;
    double phi1 = M_PI/2.0 - phi0;
    double en = exp(-4.0/((phi1 - phi0)*(phi1 - phi0)));
    double phi = asin(x[2]/RAD_SPHERE);

    if(phi > phi0 + eps && phi < phi1 - eps) {
        return (umax/en)*exp(1.0/((phi - phi0)*(phi - phi1)));
    }
    else {
        return 0.0;
    }
}

double v_init(double* x) {
    return 0.0;
}

double h_top_init(double* x) {
    return HTOP;
/*
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
    double phi   = asin(x[2]/RAD_SPHERE);
    double phi_0 = 0.25*M_PI;
    double y     = RAD_SPHERE*(phi - phi_0);
    double L     = RAD_SPHERE*M_PI/40.0;

    if(x[2]/RAD_SPHERE > 0.9) return HBOT + DELTA;

    return HBOT + DELTA*tanh(y/L);
/*
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

double u_top_init(double* x) {
    double phi   = asin(x[2]/RAD_SPHERE);
    double f     = 2.0*OMEGA*sin(phi);
    double phi_0 = 0.25*M_PI;
    double y     = RAD_SPHERE*(phi - phi_0);
    double L     = RAD_SPHERE*M_PI/40.0;

    if(fabs(phi - phi_0) > 0.25*M_PI*(7.0/8.0)) return 0.0;

    return (-GRAVITY*DELTA/f/L)*(1.0 - tanh(y/L)*tanh(y/L));
}

double u_bot_init(double* x) {
    double eps = 1.0e-8;

    return u_top_init(x) + eps*2.0*(rand() - 0.5)/RAND_MAX;
}

/*
double h_init(double* x) {
    int ii, ni = 1000;
    double phiPrime = 0.0;
    double phi = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
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
*/

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    double dt = 300.0;
    double vort_0, mass_0, ener_0;
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    int nSteps = 28*24*12;
    int dumpEvery = 6*12;
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
        //sw->init1(ut, u_init, v_init);
        //sw->init2(ht, h_init);
        //sw->init1(ub, u_init, v_init);
        //sw->init2(hb, h_init);

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

    delete topo;
    delete geom;
    delete sw;

    VecDestroy(&ut);
    VecDestroy(&ht);
    VecDestroy(&ub);
    VecDestroy(&hb);

    PetscFinalize();

    return 0;
}