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

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0

#define RH_ANGFREQ 7.292e-5
#define RH_GRAV 9.80616
#define RH_OMEGA 7.848e-6
#define RH_R 4.0
#define RH_K 7.848e-6
#define RH_A 6371220.0
#define RH_H0 8.0e+3

/* Rossby Haurwitz test case
   Reference:
       Williamson, Drake, Hack, Jakob and Swartzrauber (1992) 
       J. Comp. Phys. 102 211-224

   plot extrema:
       depth       +7980.0   +10565.0
       velocity-x   0.0      +100.0
       velocity-y  -64.0     +64.0
       vorticity   -7.5e-5   +7.5e-5
*/

double w_init(double* x) {
    double theta = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double ct = cos(theta);
    double st = sin(theta);
    double omega = 0.0;

    omega += 2.0*RH_OMEGA*st;
    omega -= RH_K*st*pow(ct,RH_R)*(RH_R*RH_R + 3.0*RH_R + 2.0)*cos(RH_R*lambda);

    return omega;
}

double u_init(double* x) {
    double theta = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double ct = cos(theta);
    double st = sin(theta);
    double u = 0.0;

    u += RH_A*RH_OMEGA*ct;
    u += RH_A*RH_K*pow(ct,RH_R-1.0)*(RH_R*st*st - ct*ct)*cos(RH_R*lambda);

    return u;
}

double v_init(double* x) {
    double theta = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double ct = cos(theta);
    double st = sin(theta);
    double v = -RH_A*RH_K*RH_R*pow(ct,RH_R-1.0)*st*sin(RH_R*lambda);

    return v;
}

double h_init(double* x) {
    double theta = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double ct = cos(theta);
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double h = 0.0;

    a += 0.5*RH_OMEGA*(2.0*RH_ANGFREQ + RH_OMEGA)*ct*ct;
    a += 0.25*RH_K*RH_K*pow(ct,2.0*RH_R)*((RH_R+1.0)*ct*ct + (2.0*RH_R*RH_R - RH_R - 2.0) - 2.0*RH_R*RH_R*pow(ct,-2.0));

    b += 2.0*(RH_ANGFREQ + RH_OMEGA)*RH_K/(RH_R+1.0)/(RH_R+2.0)*pow(ct,RH_R)*((RH_R*RH_R + 2.0*RH_R + 2.0) - (RH_R+1.0)*(RH_R+1.0)*ct*ct);

    c += 0.25*RH_K*RH_K*pow(ct,2.0*RH_R)*((RH_R+1.0)*ct*ct - (RH_R+2.0));

    h += RH_H0;
    h += RAD_SPHERE*RAD_SPHERE*a/RH_GRAV;
    h += RAD_SPHERE*RAD_SPHERE*b*cos(RH_R*lambda)/RH_GRAV;
    h += RAD_SPHERE*RAD_SPHERE*c*cos(2.0*RH_R*lambda)/RH_GRAV;

    return h;
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    //double dt = 10.0*60.0; time step for 4 3rd order elements per dimensnion per face
    double dt = 6.0*60.0;
    double vort_0, mass_0, ener_0, vort, mass, ener;
    char fieldname[20];
    bool dump;
    int nSteps = 4250;
    int dumpEvery = 25;
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Test* test;
    Vec wi, ui, hi, uf, hf;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo);
    sw = new SWEqn(topo, geom);
    test = new Test(sw);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &wi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uf);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hf);

    sw->init0(wi, w_init);
    sw->init1(ui, u_init, v_init);
    sw->init2(hi, h_init);

    sprintf(fieldname,"vorticity");
    geom->write0(wi,fieldname,0);
    sprintf(fieldname,"velocity");
    geom->write1(ui,fieldname,0);
    sprintf(fieldname,"pressure");
    geom->write2(hi,fieldname,0);

    VecDestroy(&wi);

    sw->diagnose_w(ui, &wi, false);
    vort_0 = sw->int0(wi);
    mass_0 = sw->int2(hi);
    ener_0 = sw->intE(ui, hi);
    VecDestroy(&wi);

    for(step = 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step: " << step << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        sw->solve_RK2_SS(ui, hi, uf, hf, dt, dump);
        VecCopy(uf,ui);
        VecCopy(hf,hi);
        if(dump) {
            sw->writeConservation(ui, hi, mass_0, vort_0, ener_0);
        }
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
