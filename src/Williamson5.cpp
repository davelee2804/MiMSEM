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
#include "SWEqn_Picard.h"

using namespace std;

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0

#define W2_GRAV 9.80616
#define W2_OMEGA 7.292e-5
#define W2_U0 20.0
#define W2_H0 5960.0
#define W2_ALPHA 0.0

/* Williamson test case 5
   Reference:
       Williamson, Drake, Hack, Jakob and Swartzrauber (1992) 
       J. Comp. Phys. 102 211-224
*/

double w_init(double* x) {
    double theta = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);

    return (2.0*W2_U0/RAD_SPHERE + 2.0*W2_OMEGA)*(-cos(lambda)*cos(theta)*sin(W2_ALPHA) + sin(theta)*cos(W2_ALPHA));
}

double u_init(double* x) {
    double theta = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);

    return W2_U0*(cos(theta)*cos(W2_ALPHA) + cos(lambda)*sin(theta)*sin(W2_ALPHA));
}

double v_init(double* x) {
    double lambda = atan2(x[1],x[0]);

    return -W2_U0*sin(lambda)*sin(W2_ALPHA);
}

double b_init(double* x) {
    double theta    = asin(x[2]/RAD_SPHERE);
    double lambda   = atan2(x[1],x[0]);
    double theta_c  =  M_PI/6.0;
    double lambda_c = -M_PI/2.0;
    double bo       = 2000.0;
    double rad      = M_PI/9.0;
    double rsq      = (lambda - lambda_c)*(lambda - lambda_c) + (theta - theta_c)*(theta - theta_c);
    double r        = sqrt(rsq);
    double b        = 0.0;

    if(r < rad) b = bo*(1.0 - r/rad);

    return b;
}

double h_init(double* x) {
    double theta = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double b = -cos(lambda)*cos(theta)*sin(W2_ALPHA) + sin(theta)*cos(W2_ALPHA);
    
    return W2_H0 - (RAD_SPHERE*W2_OMEGA*W2_U0 + 0.5*W2_U0*W2_U0)*b*b/W2_GRAV - b_init(x);
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    double vort_0, mass_0, ener_0, enst_0;
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    double dt = 600.0;
    int nSteps = 20*24*6;
    int dumpEvery = 24*6;
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Vec ui, hi, wi, qi, vi, bot;
    PetscViewer viewer;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo();
    geom = new Geom(topo);
    sw = new SWEqn(topo, geom);
    sw->step = startStep;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &bot);

    if(startStep == 0) {
        sw->init1(ui, u_init, v_init);
        sw->init2(hi, h_init);

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
    sw->init2(bot, b_init);

    sw->curl(ui, &wi);
    vort_0 = sw->int0(wi);
    mass_0 = sw->int2(hi);
    ener_0 = sw->intE(ui, hi, bot);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vi);
    sw->diagnose_q(0.0, ui, sw->uil, hi, &qi);
    MatMult(sw->M0h->M, qi, vi);
    VecDot(qi, vi, &enst_0);
    if(!rank) {
        cout << "w0: " << vort_0 << "\tm0: " << mass_0 << "\tE0: " << ener_0 << "\tZ0" << enst_0 << endl;
    }
    VecDestroy(&wi);
    VecDestroy(&qi);
    VecDestroy(&vi);

    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        sw->solve(ui, hi, dt, dump, 2, true, bot);
        sw->writeConservation(step*dt, ui, hi, mass_0, vort_0, ener_0, enst_0, bot);
    }

    delete sw;
    delete geom;
    delete topo;

    VecDestroy(&ui);
    VecDestroy(&hi);
    VecDestroy(&bot);

    PetscFinalize();

    return 0;
}
