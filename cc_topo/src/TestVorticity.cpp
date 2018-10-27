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

// -1/(R.cos(phi))(d(u.cos(phi))/d.phi - d.v/d.theta)
double w_init(double* x) {
    double theta = atan2(x[1],x[0]);
    double phi = asin(x[2]/RAD_SPHERE);
    //double a = +1.0/RAD_SPHERE/cos(phi);
    //double b = -2.0*RAD_SPHERE*sin(phi)*cos(phi)*cos(theta);
    //double c = +RAD_SPHERE*cos(phi)*cos(theta);
    //return a*(c - b);
    double b = 2.0*sin(phi)*cos(theta);
    double c = cos(theta);
    return c + b;
}

double dwx_init(double* x) {
    double theta = atan2(x[1],x[0]);
    double phi = asin(x[2]/RAD_SPHERE);

    //return (-1.0/RAD_SPHERE*cos(phi))*(2.0*sin(phi)*sin(theta) - sin(theta));
    return (-1.0/RAD_SPHERE)*(2.0*cos(phi)*cos(theta));
}

double dwy_init(double* x) {
    double theta = atan2(x[1],x[0]);
    double phi = asin(x[2]/RAD_SPHERE);

    //return (-1.0/RAD_SPHERE)*(-2.0*cos(phi)*cos(theta));
    return (1.0/RAD_SPHERE/cos(phi))*(-2.0*sin(phi)*sin(theta) - sin(theta));
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
    Vec wn, wa, du, ui, dwa, dwn;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo);
    sw = new SWEqn(topo, geom);
    test = new Test(sw);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &wn);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &wa);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dwa);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dwn);

    sw->init0(wa, w_init);
    sw->init1(ui, u_init, v_init);

    VecZeroEntries(du);
    MatMult(sw->E01M1, ui, du);
    VecPointwiseDivide(wn, du, sw->m0->vg);

    sprintf(fieldname,"vort_n");
    geom->write0(wn,fieldname,0);
    sprintf(fieldname,"vort_a");
    geom->write0(wa,fieldname,0);
    sprintf(fieldname,"velocity");
    geom->write1(ui,fieldname,0);

    sw->init1(dwa, dwx_init, dwy_init);
    MatMult(sw->NtoE->E10, wn, dwn);

    sprintf(fieldname,"dwa");
    geom->write1(dwa,fieldname,0);
    sprintf(fieldname,"dwn");
    geom->write1(dwn,fieldname,0);

    cout << "global vorticity " << sw->int0(wn) << endl;

    // TODO: check that the velocity components are correct for H(rot) error
    sw->err0(wn, w_init, dwx_init, dwy_init, err);
    if(!rank) cout << "H(rot) vorticity error: " << err[1] << endl;

    delete topo;
    delete geom;
    delete sw;
    delete test;

    VecDestroy(&wn);
    VecDestroy(&wa);
    VecDestroy(&du);
    VecDestroy(&ui);
    VecDestroy(&dwa);
    VecDestroy(&dwn);

    PetscFinalize();

    return 0;
}
