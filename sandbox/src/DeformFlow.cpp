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
/*
    double x_1 = RAD_SPHERE*cos(0.0)*cos(lambda_1);
    double y_1 = RAD_SPHERE*cos(0.0)*sin(lambda_1);
    double z_1 = RAD_SPHERE*sin(0.0);
    double r_1 = (x[0] - x_1)*(x[0] - x_1) + (x[1] - y_1)*(x[1] - y_1) + (x[2] - z_1)*(x[2] - z_1);

    double x_2 = RAD_SPHERE*cos(0.0)*cos(lambda_2);
    double y_2 = RAD_SPHERE*cos(0.0)*sin(lambda_2);
    double z_2 = RAD_SPHERE*sin(0.0);
    double r_2 = (x[0] - x_2)*(x[0] - x_2) + (x[1] - y_2)*(x[1] - y_2) + (x[2] - z_2)*(x[2] - z_2);

    return 0.95*(exp(-5.0*r_1/RAD_EARTH/RAD_EARTH) + exp(-5.0*r_2/RAD_EARTH/RAD_EARTH));
*/
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    ofstream file;
    char fieldname[50];
    char filename[50];
    bool dump;
    int startStep = atoi(argv[1]);
    int nSteps = 8000;
    int dumpEvery = 1000;
    double dt = T_PERIOD/nSteps;
    double mass, mass0, l2err2, l2norm2;
    Topo* topo;
    Geom* geom;
    AdvEqn* ad;
    Vec ui, hi, uj, hj, ho, ones, Mh;
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
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &ho);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &ones);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Mh);

    VecSet(ones, 1.0);

    if(startStep == 0) {
        ad->init2(hi, h_init);
        VecCopy(hi, ho);
        mass0 = ad->int2(ho);

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
    sprintf(filename, "output/conservation.dat");
    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        ad->init1(ui, u_init, v_init);
        deform_flow_time += dt;
        //deform_flow_time += 0.5*dt;
        ad->init1(uj, u_init, v_init);
        //deform_flow_time += 0.5*dt;

        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        ad->solve(ui, hi, uj, hj, dt, dump);
        VecCopy(hj, hi);

        mass = ad->int2(hi);
        if(!rank) {
            file.open(filename, ios::out | ios::app);
            file << deform_flow_time/(60.0*60.0*24) << "\t" << mass0 << "\t" << mass << "\t" << (mass-mass0)/mass0 << endl;
            file.close();
        }
    }

    // final l2 error
    MatMult(ad->M2->M, hj, Mh);
    VecDot(hj, Mh, &l2norm2);
    VecAXPY(hj, -1.0, ho);
    MatMult(ad->M2->M, hj, Mh);
    VecDot(hj, Mh, &l2err2);
    if(!rank) {
        cout << "error: " << sqrt(l2err2) << "\tnorm: " << sqrt(l2norm2) << "\terror/norm: " << sqrt(l2err2/l2norm2) << endl;
    }

//    delete ad;
//    delete geom;
//    delete topo;

    VecDestroy(&ui);
    VecDestroy(&hi);
    VecDestroy(&uj);
    VecDestroy(&hj);
    VecDestroy(&ho);
    VecDestroy(&ones);
    VecDestroy(&Mh);

    PetscFinalize();

    return 0;
}
