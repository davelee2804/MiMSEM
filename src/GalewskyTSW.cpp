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
#include "ThermalSW.h"

using namespace std;

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0

// initial condition given by:
//     Galewsky, Scott and Polvani (2004) Tellus, 56A 429-440
double u_init(double* x) {
    double eps = 1.0e-8;
    double umax = 80.0*(RAD_SPHERE/RAD_EARTH);
    double phi0 = M_PI/7.0;
    double phi1 = M_PI/2.0 - phi0;
    double en = exp(-4.0/((phi1 - phi0)*(phi1 - phi0)));
    double phi = asin(x[2]/RAD_SPHERE);

    if(phi > phi0 + eps && phi < phi1 - eps) {
        return (umax/en)*exp(1.0/((phi - phi0)*(phi - phi1)));
    }
    else {
        return 0.0;
    }
}

double v_init(double* x) {
    return 0.0;
}

double h_init(double* x) {
    int ii, ni = 1000;
    double phiPrime = 0.0;
    double phi = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double dphi = fabs(phi/ni);
    double hHat = 120.0*(RAD_SPHERE/RAD_EARTH);
    double h = 10000.0*(RAD_SPHERE/RAD_EARTH);
    double grav = 9.80616*(RAD_SPHERE/RAD_EARTH);
    double omega = 7.292e-5;
    double u, f;
    double alpha = 1.0/3.0;
    double beta = 1.0/15.0;
    double phi2 = M_PI/4.0;
    double x2[3];
    int sgn = (phi > 0) ? +1 : -1;

    x2[0] = x[0];
    x2[1] = x[1];
    for(ii = 0; ii < ni; ii++) {
        phiPrime += sgn*dphi;
        x2[2] = RAD_SPHERE*sin(phiPrime);
        u = u_init(x2);
        f = 2.0*omega*sin(phiPrime);
        h -= RAD_SPHERE*u*(f + tan(phiPrime)*u/RAD_SPHERE)*dphi/grav;
    }

    h += hHat*cos(phi)*exp(-1.0*(lambda/alpha)*(lambda/alpha))*exp(-1.0*((phi2 - phi)/beta)*((phi2 - phi)/beta));

    return h;
}

double s_init(double* x) {
    double s = 9.80616;
    double phi = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double alpha = 1.0/3.0;
    double beta = 1.0/15.0;
    double phi2 = M_PI/4.0;

    s -= 0.1*9.80616*cos(phi)*exp(-1.0*(lambda/alpha)*(lambda/alpha))*exp(-1.0*((phi2 - phi)/beta)*((phi2 - phi)/beta));

    return s;
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    double dt = 30.0;
    double vort_0, mass_0, ener_0, enst_0, buoy_0;
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    int nSteps = 12*24*120;
    int dumpEvery = 24*120;
    Topo* topo;
    Geom* geom;
    ThermalSW* tsw;
    Vec qi, vi, htmp;
    PetscViewer viewer;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo();
    geom = new Geom(topo);
    tsw = new ThermalSW(topo, geom);
    tsw->step = startStep;

    if(startStep == 0) {
        tsw->init1(tsw->ui, u_init, v_init);
        tsw->init2(tsw->hi, h_init);
        tsw->init2(tsw->si, s_init);

        sprintf(fieldname,"velocity");
        geom->write1(tsw->ui,fieldname,0);
        sprintf(fieldname,"pressure");
        geom->write2(tsw->hi,fieldname,0);
        sprintf(fieldname,"buoyancy");
        geom->write2(tsw->si,fieldname,0);
    } else {
        sprintf(fieldname, "output/pressure_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(tsw->hi, viewer);
        PetscViewerDestroy(&viewer);

        sprintf(fieldname, "output/buoyancy_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(tsw->si, viewer);
        PetscViewerDestroy(&viewer);

        sprintf(fieldname, "output/velocity_%.4u.vec", startStep);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        VecLoad(tsw->ui, viewer);
        PetscViewerDestroy(&viewer);
    }
    VecScatterBegin(topo->gtol_1, tsw->ui, tsw->uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, tsw->ui, tsw->uil, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vi);
    tsw->curl(tsw->ui);
    tsw->diagnose_q(tsw->ui, tsw->hi, &qi);
    MatMult(tsw->M0h->M, qi, vi);
    VecDot(qi, vi, &enst_0);
    MatMult(tsw->M2->M, tsw->hi, htmp);
    //VecSum(htmp, &mass_0);
    mass_0 = tsw->int2(tsw->hi);
    VecDot(htmp, tsw->si, &buoy_0);
    MatMult(tsw->M0->M, tsw->wi, vi);
    VecSum(vi, &vort_0);
    tsw->K->assemble(tsw->uil);
    MatMult(tsw->K->M, tsw->ui, htmp);
    VecDot(tsw->hi, htmp, &ener_0);
    ener_0 += 0.5*buoy_0;
    ener_0 = tsw->intE(tsw->uil, tsw->hi, tsw->si);
    if(!rank) {
        cout << "w0: " << vort_0 << "\tm0: " << mass_0 << "\tE0: " << ener_0 << "\tZ0: " << enst_0 << "\tS0: " << buoy_0 << endl;
    }
    VecDestroy(&qi);
    VecDestroy(&vi);
    VecDestroy(&htmp);

    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        tsw->solve_ssp_rk2(dt, dump);
        tsw->writeConservation(step*dt, mass_0, vort_0, ener_0, enst_0, buoy_0);
    }

    delete tsw;
    delete geom;
    delete topo;

    PetscFinalize();

    return 0;
}
