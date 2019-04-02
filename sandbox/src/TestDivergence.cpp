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

//#define RAD_SPHERE 6371220.0
#define RAD_SPHERE 1.0

double u_init(double* x) {
    return x[0]; // R.cos(phi).cos(theta)
}

double v_init(double* x) {
    return x[1]; // R.cos(phi).sin(theta)
}

// 1/(R.cos(phi))(d(v.cos(phi))/d.phi + d.u/d.theta)
double p_init(double* x) {
    double theta = atan2(x[1],x[0]);
    double phi = asin(x[2]/RAD_SPHERE);
    //double a = +1.0/RAD_SPHERE/cos(phi);
    //double b = -2.0*RAD_SPHERE*sin(phi)*cos(phi)*sin(theta);
    //double c = -RAD_SPHERE*cos(phi)*sin(theta);
    //return a*(b + c);
    double b = -2.0*sin(phi)*sin(theta);
    double c = -sin(theta);
    return b + c;
}

int main(int argc, char** argv) {
    int size, rank;
    double err[3];
    static char help[] = "petsc";
    char fieldname[20];
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Test* test;
    Vec pn, pa, ui;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo);
    sw = new SWEqn(topo, geom);
    test = new Test(sw);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &pn);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &pa);

    sw->init1(ui, u_init, v_init);
    sw->init2(pa, p_init);

    MatMult(sw->EtoF->E21, ui, pn);

    sprintf(fieldname,"velocity");
    geom->write1(ui,fieldname,0);
    sprintf(fieldname,"pres_n");
    geom->write2(pn,fieldname,0);
    sprintf(fieldname,"pres_a");
    geom->write2(pa,fieldname,0);

    sw->err2(pn, p_init, err);
    if(!rank) cout << "L2 divergence error: " << err[1] << endl;

    delete topo;
    delete geom;
    delete sw;
    delete test;

    VecDestroy(&pn);
    VecDestroy(&pa);
    VecDestroy(&ui);

    PetscFinalize();

    return 0;
}
