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
#include "SWEqn_JFNK.h"

using namespace std;

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0

#define W2_GRAV 9.80616
#define W2_OMEGA 7.292e-5
#define W2_U0 38.61068276698372
#define W2_H0 2998.1154702758267
#define W2_ALPHA (0.25*M_PI)

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

double q_init(double* x) {
    return w_init(x) / h_init(x);
}

double q2_init(double* x) {
    return q_init(x) * q_init(x);
}

int main(int argc, char** argv) {
    int size, rank;
    static char help[] = "petsc";
    char fieldname[20];
    ofstream file;
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Vec wi, ui, hi;

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
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);

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

    {
        Vec qa, qn;
        Vec* Zi;

        VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &qa);
        sw->init0(qa, q_init);
        sprintf(fieldname,"pv_analytic");
        geom->write0(qa,fieldname,0);

        sw->diagnose_q(ui, hi, &qn);
        sprintf(fieldname,"pv_numeric");
        geom->write0(qn,fieldname,0);

        Zi = sw->diagnose_null_space_vecs(ui, hi, 2);
        sw->unpack(Zi[0], ui, hi);
        sprintf(fieldname, "Z0_u");
        geom->write1(ui,fieldname,0);
        sprintf(fieldname, "Z0_h_numeric");
        geom->write2(hi,fieldname,0);

        sw->init2(hi, q2_init);
        VecScale(hi, 1.0/(0.0 + 2.0) - 1.0);
        sprintf(fieldname, "Z0_h_analytic");
        geom->write2(hi,fieldname,0);

        VecDestroy(&qa);
        VecDestroy(&qn);
    }

    delete topo;
    delete geom;
    delete sw;

    VecDestroy(&ui);
    VecDestroy(&hi);

    PetscFinalize();

    return 0;
}
