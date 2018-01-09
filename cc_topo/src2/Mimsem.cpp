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
#include "Test.h"

using namespace std;

#define EL_ORD 3
#define N_ELS_X_LOC 8
#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0

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

double v_init(double* x) {
    return 0.0;
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
    double dt = 120.0;//0.1*(2.0*M_PI/(4.0*12))/80.0;
    char fieldname[20];
    bool dump;
    int nSteps = 2;
    int dumpEvery = 1;
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Test* test;
    Vec ui, hi, uf, hf;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank, EL_ORD, N_ELS_X_LOC);
    geom = new Geom(rank, topo);
    sw = new SWEqn(topo, geom);
    test = new Test(sw);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uf);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hf);

    test->vorticity(u_init, v_init);
    test->gradient(h_init);
    test->divergence(u_init, v_init);
    test->convection(u_init, v_init);
    test->massFlux(u_init,v_init, h_init);
    test->kineticEnergy(u_init,v_init);

    sw->init1(ui, u_init, v_init);
    sw->init2(hi, h_init);

    sprintf(fieldname,"velocity");
    geom->write1(ui,fieldname,0);
    sprintf(fieldname,"pressure");
    geom->write2(hi,fieldname,0);

    for(step = 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step: " << step << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        sw->solve(ui, hi, uf, hf, dt, dump);
        VecCopy(uf,ui);
        VecCopy(hf,hi);
    }

    delete topo;
    delete geom;
    delete sw;
    delete test;

    VecDestroy(&ui);
    VecDestroy(&uf);
    VecDestroy(&hi);
    VecDestroy(&hf);

    PetscFinalize();

    return 0;
}
