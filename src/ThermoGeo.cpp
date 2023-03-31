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
#include "ThermalSW_EEC_2.h"

using namespace std;

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
#define GRAVITY 9.80616
#define OMEGA 7.292e-5

// initial condition given by:
//     Galewsky, Scott and Polvani (2004) Tellus, 56A 429-440
double u_init(double* x) {
    double phi    = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double u0     = 2.0*M_PI*RAD_EARTH/(12.0*24.0*3600.0);

    return u0*cos(phi);
}

double v_init(double* x) {
    return 0.0;
}

double h_init(double* x) {
    double phi    = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double h0     = 2.94e+4/GRAVITY;
    double u0     = 2.0*M_PI*RAD_EARTH/(12.0*24.0*3600.0);

    return h0 - (1.0/GRAVITY)*(RAD_EARTH*OMEGA*u0 + 0.5*u0*u0)*sin(phi)*sin(phi);
}

double s_init(double* x) {
    double phi    = asin(x[2]/RAD_SPHERE);
    double lambda = atan2(x[1],x[0]);
    double h0     = 2.94e+4/GRAVITY;
    double h      = h_init(x);

    return GRAVITY*(1.0 + 0.05*(h0/h)*(h0/h));
}

double S_init(double* x) {
    double h = h_init(x);
    double s = s_init(x);

    return h*s;
}

int main(int argc, char** argv) {
    int size, rank, step;
    static char help[] = "petsc";
    double dt = 30.0;
    double vort_0, mass_0, ener_0, enst_0, buoy_0, entr_0;
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    int nSteps = 5*24*120;
    int dumpEvery = 24*120;
    Topo* topo;
    Geom* geom;
    ThermalSW_EEC_2* tsw;
    Vec qi, vi, htmp, htmp2;
    double err_u[3], err_h[3], err_S[3];
    char filename[50];
    ofstream file;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo();
    geom = new Geom(topo);
    tsw = new ThermalSW_EEC_2(topo, geom);
    tsw->step = startStep;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &htmp2);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &vi);

    tsw->init1(tsw->ui, u_init, v_init);
    tsw->init2(tsw->hi, h_init);
    tsw->init2(tsw->S_on_h, s_init);
    tsw->M2h->assemble(tsw->hi);
    MatMult(tsw->M2h->M, tsw->S_on_h, htmp);
    KSPSolve(tsw->ksp2, htmp, tsw->Si);

    sprintf(fieldname,"velocity");
    geom->write1(tsw->ui,fieldname,0);
    sprintf(fieldname,"pressure");
    geom->write2(tsw->hi,fieldname,0);
    sprintf(fieldname,"buoyancy");
    geom->write2(tsw->S_on_h,fieldname,0);
    sprintf(fieldname,"depth_buoyancy");
    geom->write2(tsw->Si,fieldname,0);

    VecScatterBegin(topo->gtol_1, tsw->ui, tsw->uil, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  topo->gtol_1, tsw->ui, tsw->uil, INSERT_VALUES, SCATTER_FORWARD);

    tsw->curl(tsw->ui);
    tsw->diagnose_q(tsw->ui, tsw->hi, &qi);
    MatMult(tsw->M0h->M, qi, vi);
    VecDot(qi, vi, &enst_0);
    MatMult(tsw->M2->M, tsw->hi, htmp);
    mass_0 = tsw->int2(tsw->hi);
    buoy_0 = tsw->int2(tsw->Si);
    MatMult(tsw->M0->M, tsw->wi, vi);
    VecSum(vi, &vort_0);
    tsw->K->assemble(tsw->uil);
    MatMult(tsw->K->M, tsw->ui, htmp);
    VecDot(tsw->hi, htmp, &ener_0);
    ener_0 += 0.5*buoy_0;
    ener_0 = tsw->intE(tsw->uil, tsw->hi, tsw->Si);
    tsw->M2h->assemble(tsw->hi);
    MatMult(tsw->M2->M, tsw->Si, htmp);
    KSPSolve(tsw->ksp2h, htmp, htmp2);
    MatMult(tsw->M2->M, htmp2, htmp);
    VecDot(htmp, tsw->Si, &entr_0);
    entr_0 *= 0.5;

    if(!rank) {
        cout << "w0: " << vort_0 << "\tm0: " << mass_0 << "\tE0: " << ener_0 << 
	      "\tZ0: " << enst_0 << "\tS0: " << buoy_0 << "\tEntropy0: " << entr_0 << endl;
    }
    VecDestroy(&qi);
    VecDestroy(&vi);
    VecDestroy(&htmp);
    VecDestroy(&htmp2);

    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        tsw->solve_rk(dt, dump);
        tsw->writeConservation(step*dt, mass_0, vort_0, ener_0, enst_0, buoy_0, entr_0);

        tsw->err1(tsw->ui, u_init, v_init, NULL, err_u);
        tsw->err2(tsw->hi, h_init, err_h);
        tsw->err2(tsw->Si, S_init, err_S);

        if(!rank) {
            sprintf(filename, "output/errors.dat");
            file.open(filename, ios::out | ios::app);
            file << std::scientific << step*dt/60.0/60.0/24.0 << 
                    "\t" << err_h[0] << "\t" << err_u[0] << "\t" << err_S[0] <<
                    "\t" << err_h[1] << "\t" << err_u[1] << "\t" << err_S[1] <<
                    "\t" << err_h[2] << "\t" << err_u[2] << "\t" << err_S[2] << endl;
            file.close();
        }
    }

    delete tsw;
    delete geom;
    delete topo;

    PetscFinalize();

    return 0;
}
