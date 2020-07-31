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
#include "AdvEqn.h"

using namespace std;

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0

#define T_PERIOD (12.0*24.0*60.0*60.0)

double deform_flow_time = 0.0;

double u_init(double* x) {
    double phi = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double lambda_prime = lambda - 2.0*M_PI*deform_flow_time/T_PERIOD;

    return (10.0*RAD_EARTH/T_PERIOD)*sin(lambda_prime)*sin(lambda_prime)*sin(2.0*phi)*cos(M_PI*deform_flow_time/T_PERIOD) +
           (2.0*M_PI*RAD_EARTH/T_PERIOD)*cos(phi);
}

double v_init(double* x) {
    double phi = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double lambda_prime = lambda - 2.0*M_PI*deform_flow_time/T_PERIOD;

    return (10.0*RAD_EARTH/T_PERIOD)*sin(2.0*lambda_prime)*cos(phi)*cos(M_PI*deform_flow_time/T_PERIOD);
}

double rad(double lambda, double phi, double lambda_0, double phi_0) {
    return RAD_SPHERE*acos(sin(phi_0)*sin(phi) + cos(phi_0)*cos(phi)*cos(lambda-lambda_0));
}

double h_init(double* x) {
    double phi = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double lambda_1 = 5.0*M_PI/6.0 + M_PI;
    double lambda_2 = 7.0*M_PI/6.0 + M_PI;
    double rad_0 = 0.5*RAD_EARTH;
    double rad_1 = rad(lambda, phi, lambda_1, 0.0);
    double rad_2 = rad(lambda, phi, lambda_2, 0.0);

    if(rad_1 < rad_0) return 0.5*(1.0 + cos(M_PI*rad_1/rad_0));
    if(rad_2 < rad_0) return 0.5*(1.0 + cos(M_PI*rad_2/rad_0));
    return 0.0;
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    //int nSteps = 384;
    int nSteps = 800;
    int dumpEvery = 100;
    double dt = T_PERIOD/nSteps;
    Topo* topo;
    Geom* geom;
    AdvEqn* ad;
    Vec ui, hi, uj, hj;
    PetscViewer viewer;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo();
    geom = new Geom(topo);
    ad = new AdvEqn(topo, geom);
    ad->step = startStep;

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);

    if(startStep == 0) {
        ad->init2(hi, h_init);

        sprintf(fieldname,"velocity");
        geom->write1(ui,fieldname,0);
        sprintf(fieldname,"tracer");
        geom->write2(hi,fieldname,0);
    } else {
        sprintf(fieldname, "output/tracer_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(hi, viewer);
        PetscViewerDestroy(&viewer);

        sprintf(fieldname, "output/velocity_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(ui, viewer);
        PetscViewerDestroy(&viewer);

        VecCopy(ui, uj);
    }

    deform_flow_time = startStep * dumpEvery * dt;
    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        ad->init1(ui, u_init, v_init);
        deform_flow_time += dt;
        ad->init1(uj, u_init, v_init);

        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        ad->solve(ui, hi, uj, hj, dt, dump);
        VecCopy(hj, hi);
    }

//    delete ad;
//    delete geom;
//    delete topo;

    VecDestroy(&ui);
    VecDestroy(&hi);
    VecDestroy(&uj);
    VecDestroy(&hj);

    PetscFinalize();

    return 0;
}
