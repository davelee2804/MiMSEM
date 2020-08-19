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
#include "L2Vecs.h"
#include "ElMats.h"
#include "VertOps.h"
#include "Assembly.h"
#include "VertSolve.h"
#include "Euler_2.h"

using namespace std;

#define NK 150
#define ZTOP 1500.0
#define GRAVITY 9.80616
#define CP 1004.5
#define CV 717.5
#define RD 287.0
#define P0 100000.0
#define _LX (1000.0)
#define THETA_0 300.0

/*
gravity wave on a reduced planet (DCMIP 2012 test case 3.1)
reference:
  https://www.earthsystemcog.org/projects/dcmip-2012/test_cases
*/

double z_at_level(double* x, int ki) {
    return ki*(ZTOP/NK);
}

double pres(double* x, int ki) {
    double z     = z_at_level(x, ki);

    return P0*pow(1.0 - GRAVITY*z/(CP*THETA_0), CP/RD);
}

double theta_init(double* x, int ki) {
    double z     = z_at_level(x, ki);
    double xi    = x[0] - 0.5*_LX;
    double yi    = x[1] - 0.5*_LX;
    double zi    = z    - 350.0;
    double rsq   = xi*xi + yi*yi + zi*zi;
    double r     = sqrt(rsq);
    double theta = THETA_0;

    if(r < 250.0) theta += 0.25*(1.0 + cos(M_PI*r/250.0));

    return theta;
}

double u_init(double* x, int ki) {
    return 0.0;
}

double v_init(double* x, int ki) {
    return 0.0;
}

double exner_init(double* x, int ki) {
    double pi = pres(x, ki);

    return CP*pow(pi/P0, RD/CP);
}

double rho_init(double* x, int ki) {
    double exner = exner_init(x, ki);

    return (P0/(RD*THETA_0))*pow(exner/CP, CV/RD);
}

double rt_init(double* x, int ki) {
    return rho_init(x, ki)*theta_init(x,ki);
}

double f_topog(double* x) {
    return 0.0;
}

void LoadVecs(Vec* vecs, int nk, char* fieldname, int step) {
    int ki;
    char filename[100];
    PetscViewer viewer;

    for(ki = 0; ki < nk; ki++) {
        sprintf(filename, "output/%s_%.3u_%.4u.vec", fieldname, ki, step);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
        VecLoad(vecs[ki], viewer);
        PetscViewerDestroy(&viewer);
    }
}

void LoadVecsVert(Vec* vecs, int nk, char* fieldname, int step, Topo* topo, Geom* geom) {
    int ki;
    char filename[100];
    PetscViewer viewer;
    L2Vecs* l2Vecs = new L2Vecs(nk, topo, geom);

    for(ki = 0; ki < nk; ki++) {
        sprintf(filename, "output/%s_%.3u_%.4u.vec", fieldname, ki, step);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
        VecLoad(l2Vecs->vh[ki], viewer);
        PetscViewerDestroy(&viewer);
    }
    l2Vecs->HorizToVert();
    l2Vecs->CopyToVert(vecs);

    delete l2Vecs;
}

int main(int argc, char** argv) {
    int size, rank, step, ii, ki, n2;
    static char help[] = "petsc";
    char fieldname[50];
    bool dump;
    int startStep = atoi(argv[1]);
    //double dt = 0.005;
    //int nSteps = 400*200; // 1 hour
    //int dumpEvery = 200;//100;
    //double dt = 0.01;
    //int nSteps = 400*100; // 1 hour
    //int dumpEvery = 200;//100;
    double dt = 0.02;
    int nSteps = 400*50; // 1 hour
    int dumpEvery = 400;//100;
    ofstream file;
    Topo* topo;
    Geom* geom;
    Euler* pe;
    Vec *velx, *velz, *rho, *rt, *exner;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo();
    geom = new Geom(topo, NK);
    // initialise the z coordinate layer heights
    geom->initTopog(f_topog, z_at_level);
    pe   = new Euler(topo, geom, dt);
    pe->step = startStep;

    n2 = topo->nElsX*topo->nElsX;

    velx  = new Vec[NK];
    rho   = new Vec[NK];
    rt    = new Vec[NK];
    exner = new Vec[NK];
    velz  = new Vec[n2];
    for(ki = 0; ki < NK; ki++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx[ki] );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho[ki]  );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner[ki]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt[ki]   );
    }
    for(ii = 0; ii < n2; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (NK-1)*topo->elOrd*topo->elOrd, &velz[ii]);
        VecZeroEntries(velz[ii]);
    }

    if(startStep == 0) {
        pe->init1(velx, u_init, v_init);
        pe->init2(rho,   rho_init  );
        pe->init2(exner, exner_init);
        pe->init2(rt,    rt_init   );

        for(ki = 0; ki < NK; ki++) {
            sprintf(fieldname,"velocity_h");
            geom->write1(velx[ki],fieldname,0,ki);
            sprintf(fieldname,"density");
            geom->write2(rho[ki],fieldname,0,ki, true);
            sprintf(fieldname,"exner");
            geom->write2(exner[ki],fieldname,0,ki, true);
            sprintf(fieldname,"rhoTheta");
            geom->write2(rt[ki],fieldname,0,ki, true);
        }
    } else {
        sprintf(fieldname,"density");
        LoadVecs(rho  , NK, fieldname, startStep);
        sprintf(fieldname,"velocity_h");
        LoadVecs(velx , NK, fieldname, startStep);
        sprintf(fieldname,"exner");
        LoadVecs(exner, NK, fieldname, startStep);
        sprintf(fieldname,"rhoTheta");
        LoadVecs(rt   , NK, fieldname, startStep);
        sprintf(fieldname,"velocity_z");
        LoadVecsVert(velz , NK-1, fieldname, startStep, topo, geom);
    }

    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (hours): \t" << step*dt/60.0/60.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        pe->Trapazoidal(velx, velz, rho, rt, exner, dump);
    }

    delete pe;
    delete geom;
    delete topo;

    for(ki = 0; ki < NK; ki++) {
        VecDestroy(&velx[ki] );
        VecDestroy(&rho[ki]  );
        VecDestroy(&rt[ki]   );
        VecDestroy(&exner[ki]);
    }
    for(ii = 0; ii < n2; ii++) {
        VecDestroy(&velz[ii]);
    }
    delete[] velx;
    delete[] rho;
    delete[] exner;
    delete[] rt;
    delete[] velz;

    PetscFinalize();

    return 0;
}
