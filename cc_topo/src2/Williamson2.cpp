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
#include "SWEqn.h"
#include "Test.h"

using namespace std;

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0

#define W2_GRAV 9.80616
#define W2_OMEGA 7.292e-5
#define W2_U0 38.61068276698372
#define W2_H0 2998.1154702758267
#define W2_ALPHA 0.0

/* Williamson test case 2
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

double h_init(double* x) {
    double theta = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double b = -cos(lambda)*cos(theta)*sin(W2_ALPHA) + sin(theta)*cos(W2_ALPHA);
    
    return W2_H0 - (RAD_SPHERE*W2_OMEGA*W2_U0 + 0.5*W2_U0*W2_U0)*b*b/W2_GRAV;
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    double dt = 1.0*6.0*60.0; // time step for 4 3rd order elements per dimension per face
    double vort_0, mass_0, ener_0, err_w[3], err_u[3], err_h[3];
    char fieldname[20], filename[50];
    bool dump;
    int nSteps = 1*1200;
    int dumpEvery = 1*10;
    ofstream file;
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Vec wi, ui, hi, uf, hf;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo);
    sw = new SWEqn(topo, geom);
    sw->do_visc = false;

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
            sw->writeConservation(step*dt, ui, hi, mass_0, vort_0, ener_0);

            sw->diagnose_w(ui, &wi, false);
            sw->err0(wi, w_init, NULL, NULL, err_w);
            sw->err1(ui, u_init, v_init, NULL, err_u);
            sw->err2(hi, h_init, err_h);
            VecDestroy(&wi);

            if(!rank) {
                sprintf(filename, "output/l2Errs.dat");
                file.open(filename, ios::out | ios::app);
                file << step*dt/60.0/60.0/24.0 << "\t" << err_w[0] << "\t" << err_u[0] << "\t" << err_h[0] <<
                                                  "\t" << err_w[1] << "\t" << err_u[1] << "\t" << err_h[1] <<
                                                  "\t" << err_w[2] << "\t" << err_u[2] << "\t" << err_h[2] << endl;
                file.close();
            }
        }
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
