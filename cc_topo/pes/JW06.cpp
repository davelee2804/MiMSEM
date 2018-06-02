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
#include "SWEqn.h"
#include "PrimEqns.h"

using namespace std;

#define RAD_EARTH 6371229.0
#define RAD_SPHERE 6371229.0
#define NK 26
#define P0 100000.0
#define U0 35.0
#define T0 288.0
#define RD 287.04
#define DELTA_T 480000.0
#define GAMMA 0.005
#define ETA_T 0.2
#define ETA_0 0.252
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5

double A[27] = {0.002194067,0.004895209,0.009882418,0.01805201,0.02983724,0.04462334,0.06160587,0.07851243,0.07731271,
                0.07590131, 0.07424086, 0.07228744, 0.06998933,0.06728574,0.06410509,0.06036322,0.05596111,0.05078225,
                0.04468960, 0.03752191, 0.02908949, 0.02084739,0.01334443,0.00708499,0.00252136,0.0,       0.0       };
double B[27] = {0.0,        0.0,        0.0,        0.0,       0.0,       0.0,       0.0,       0.0,       0.01505309,
                0.03276228, 0.05359622, 0.07810627, 0.1069411, 0.1408637, 0.1807720, 0.2277220, 0.2829562, 0.3479364,
                0.4243822,  0.5143168,  0.6201202,  0.7235355, 0.8176768, 0.8962153, 0.9534761, 0.9851122, 1.0       };

double t_bar(double eta) {
    if(eta < ETA_T) {
        return T0*pow(eta, RD*GAMMA/GRAVITY) + DELTA_T*pow(ETA_T - eta, 5.0);
    }
    else {
        return T0*pow(eta, RD*GAMMA/GRAVITY);
    }
}

// hybrid coordinate coefficients, A and B derived from:
//     Lauritzen, Jablonowski, Taylor and Nair, JAMES, 2010
double z_from_eta(int ki) {
    int ii;
    double pi, pj, ph, dp, temp, rho, eta_h, dz, z;

    z = 0.0;

    for(ii = 25; ii >= ki; ii--) {
        pi    = A[ii+1]*P0 + B[ii+1]*P0;
        pj    = A[ii+0]*P0 + B[ii+0]*P0;
        ph    = 0.5*(pi + pj);
        dp    = fabs(pi - pj);
        eta_h = ph/P0;
        temp  = t_bar(eta_h);
        rho   = ph/RD/temp;
        dz    = dp/rho/GRAVITY;
        z    += dz;
    }

    return z;
}

// initial condition given by:
//     Jablonowski and Williamson, QJRMS, 2006
double u_init(double* x, int ki) {
    double phi     = asin(x[2]/RAD_EARTH);
    double theta   = atan2(x[1], x[0]);
    double eta     = 0.5*(A[ki+0] + B[ki+0] + A[ki+1] + B[ki+1]);
    double eta_v   = 0.5*(eta - ETA_0)*M_PI;
    double us      = U0*pow(cos(eta_v), 1.5)*sin(2.0*phi)*sin(2.0*phi);
    double theta_c = 1.0*M_PI/9.0;
    double phi_c   = 2.0*M_PI/9.0;
    double rad     = acos(sin(phi_c)*sin(phi) + cos(phi_c)*cos(phi)*cos(theta - theta_c));
    double up      = 1.0*exp(-rad*rad/10.0/10.0);
    
    return us + up;
}

double v_init(double* x, int ki) {
    return 0.0;
}

double t_init(double* x, int ki) {
    double phi     = asin(x[2]/RAD_EARTH);
    double eta     = 0.5*(A[ki+0] + B[ki+0] + A[ki+1] + B[ki+1]);
    double eta_v   = 0.5*(eta - ETA_0)*M_PI;
    double t_avg   = t_bar(eta);
    double a       = 10.0/63.0 - 2.0*pow(sin(phi), 6.0)*(cos(phi)*cos(phi) + 1.0/3.0);
    double b       = 1.6*pow(cos(phi), 3.0)*(sin(phi)*sin(phi) + 2.0/3.0) - 0.25*M_PI;
    double temp    = t_avg + 0.75*eta*M_PI*U0/RD*sin(eta_v)*sqrt(cos(eta_v))*(a*2.0*U0*pow(cos(eta_v), 1.5) + b*RAD_EARTH*OMEGA);
    
    return temp;
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    double dt = 120.0;
    double vort_0, mass_0, ener_0;
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    int nSteps = 5040;
    int dumpEvery = 30;
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    PrimEqns* pe;
    Vec wi, ui, hi, uf, hf;
    PetscViewer viewer;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo, NK);
    sw   = new SWEqn(topo, geom);
    pe   = new PrimEqns(topo, geom, dt);
    pe->step = startStep;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uf);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hf);

    if(startStep == 0) {
        //sw->init1(ui, u_init, v_init);
        //sw->init2(hi, h_init);

        sprintf(fieldname,"velocity");
        geom->write1(ui,fieldname,0);
        sprintf(fieldname,"pressure");
        geom->write2(hi,fieldname,0);
    } else {
        sprintf(fieldname, "output/pressure_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(hi, viewer);
        PetscViewerDestroy(&viewer);

        sprintf(fieldname, "output/velocity_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(ui, viewer);
        PetscViewerDestroy(&viewer);
    }

    pe->curl(ui, &wi, 0, false);
    vort_0 = sw->int0(wi);
    mass_0 = sw->int2(hi);
    ener_0 = sw->intE(ui, hi);
    VecDestroy(&wi);

    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        sw->solve_RK2_SS(ui, hi, uf, hf, dt, dump);
        VecCopy(uf,ui);
        VecCopy(hf,hi);
        if(dump) {
            sw->writeConservation(step*dt, ui, hi, mass_0, vort_0, ener_0);
        }
    }

    delete topo;
    delete geom;
    delete sw;
    delete pe;

    VecDestroy(&ui);
    VecDestroy(&uf);
    VecDestroy(&hi);
    VecDestroy(&hf);

    PetscFinalize();

    return 0;
}
