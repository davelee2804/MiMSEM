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

double u_init(double* x) {
    double phi = asin(x[2]/RAD_SPHERE);

    return 20.0*cos(phi);
}

double v_init(double* x) {
    return 0.0;
}

double h_init(double* x) {
    double phi = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double hHat = 1.0;//120.0;
    double h;
    double alpha = 1.0/3.0;
    double beta = 1.0/15.0;
    double phi2 = 0.0;//M_PI/4.0;

    alpha *= 2.0;
    beta = alpha;

    h = hHat*cos(phi)*exp(-1.0*(lambda/alpha)*(lambda/alpha))*exp(-1.0*((phi2 - phi)/beta)*((phi2 - phi)/beta));

    return h;
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    double dt = 10424.884347085605;
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    int nSteps = 192;
    int dumpEvery = 24;
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
        ad->init1(ui, u_init, v_init);
        ad->init1(uj, u_init, v_init);
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

    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        ad->solve(ui, hi, uj, hj, dt, dump);
        VecCopy(hj, hi);
    }

    delete ad;
    delete geom;
    delete topo;

    VecDestroy(&ui);
    VecDestroy(&hi);
    VecDestroy(&uj);
    VecDestroy(&hj);

    PetscFinalize();

    return 0;
}
