#include <iostream>

#include <cmath>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>

#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"

using namespace std;

#define EL_ORD 3
#define N_ELS_X_LOC 12

// initial condition given by:
//     Galewsky, Scott and Polvani (2004) Tellus, 56A 429-440
double u_init(double* x) {
    double umax = 80.0/6371220.0;
    double phi0 = M_PI/7.0;
    double phi1 = M_PI/2.0 - phi0;
    double en = exp(-4.0/((phi1 - phi0)*(phi1 - phi0)));
    double phi = sin(x[2]);

    if(phi < phi0 || phi > phi1) {
        return 0.0;
    }
    else {
        return (umax/en)*exp(1.0/((phi - phi0)*(phi - phi1)));
    }
    //return 1.0/(cosh(x[2])*cosh(x[2]));
}

double v_init(double* x) {
    return 0.0;
}

double h_init(double* x) {
    int ii, ni = 1000;
    double phiPrime = 0.0;
    double phi = sin(x[2]);
    double lambda = atan2(x[1],x[0]);
    double dphi = phi/ni;
    double h0 = 1000.0;
    double hHat = 120.0;
    double h = h0;
    double omega = 7.292e-5;
    double u, f;
    double alpha = 1.0/3.0;
    double beta = 1.0/15.0;
    double phi2 = M_PI/4.0;
    double x2[3];
    int sgn = (phi > 0) ? +1 : -1;

    x2[0] = x[0]; x2[1] = x[1];
    for(ii = 0; ii < ni; ii++) {
        phiPrime += sgn*dphi;
        x2[2] = asin(phiPrime);
        u = u_init(x2);
        f = 2.0*omega*sin(phiPrime);
        h -= 1.0*u*(f + tan(phiPrime)*u)*dphi;
    }

    h += hHat*cos(phi)*exp(-1.0*(lambda/alpha)*(lambda/alpha))*exp(-1.0*((phi2 - phi)/beta)*((phi2 - phi)/beta));

    return h;
    //return 1.0 + 0.1*tanh(x[2]);
}

int main(int argc, char** argv) {
    int rank, size, step;
    static char help[] = "petsc";
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Vec ui, hi, uf, hf;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank, EL_ORD, N_ELS_X_LOC);
    geom = new Geom(rank, topo);

    sw = new SWEqn(topo, geom);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uf);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hf);

    sw->init1(ui, u_init, v_init);
    sw->init2(hi, h_init);

    for(step = 1; step <= 10; step++) {
        if(!rank) {
            cout << "doing step: " << step << endl;
        }
        sw->solve(ui, hi, uf, hf, 0.001, true);
        VecCopy(ui,uf);
        VecCopy(hi,hf);
    }

    delete topo;
    delete geom;
    delete sw;

    VecDestroy(&ui);
    VecDestroy(&uf);
    VecDestroy(&hi);
    VecDestroy(&hf);

    PetscFinalize();

    return 0;
}
