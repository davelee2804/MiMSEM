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
#include "Solve3D.h"
#include "Euler_2.h"

using namespace std;

#define RAD_EARTH (6371220.0/125.0)
#define NK 16
#define OMEGA 0.0
#define ZTOP 10000.0
#define U0 20.0
#define N2 0.0001
#define TEQ 300.0
#define GRAVITY 9.80616
#define CP 1004.5
#define CV 717.5
#define RD 287.0
#define P0 100000.0
//#define LAMBDA_C (2.0*M_PI/3.0)
#define LAMBDA_C (2.0*M_PI/3.0 - M_PI)
#define PHI_C 0.0
#define THETA_PRIME_D 5000.0
#define THETA_PRIME_DELTA 1.0
#define THETA_PRIME_LZ 20000.0

/*
gravity wave on a reduced planet (DCMIP 2012 test case 3.1)
reference:
  https://www.earthsystemcog.org/projects/dcmip-2012/test_cases
*/

double z_at_level(double* x, int ki) {
    return ki*(ZTOP/NK);
}

double temp_surf(double* x) {
    double phi    = asin(x[2]/RAD_EARTH);
    double G      = GRAVITY*GRAVITY/N2/CP;
    double fac    = 0.25*U0*N2/GRAVITY/GRAVITY;

    return G + (TEQ - G)*exp(-fac*(U0 + 2.0*OMEGA*RAD_EARTH)*(cos(2.0*phi) - 1.0));
}

double temp_back(double* x, int ki) {
    double zi     = z_at_level(x, ki);
    double G      = GRAVITY*GRAVITY/N2/CP;
    double Ts     = temp_surf(x);
    double fac    = N2*zi/GRAVITY;

    return G*(1.0 - exp(fac)) + Ts*exp(fac);
}

double pres_surf(double* x) {
    double phi    = asin(x[2]/RAD_EARTH);
    double kappa  = RD/CP;
    double G      = GRAVITY*GRAVITY/N2/CP;
    double Ts     = temp_surf(x);
    double fac1   = (0.25*U0/G/RD)*(U0 + 2.0*OMEGA*RAD_EARTH)*(cos(2.0*phi) - 1.0);
    double fac2   = pow(Ts/TEQ, 1.0/kappa);

    return P0*exp(fac1)*fac2;
}

double pres(double* x, int ki) {
    double zi     = z_at_level(x, ki);
    double kappa  = RD/CP;
    double G      = GRAVITY*GRAVITY/N2/CP;
    double Ts     = temp_surf(x);
    double Ps     = pres_surf(x);

    return Ps*pow((G/Ts)*exp(-N2*zi/GRAVITY) + 1.0 - (G/Ts), 1.0/kappa);
}

double theta_back(double* x, int ki) {
    double zi     = z_at_level(x, ki);
    double Ts     = temp_surf(x);
    double Ps     = pres_surf(x);
    double kappa  = RD/CP;

    return Ts*pow(P0/Ps, kappa)*exp(N2*zi/GRAVITY);
}

double rho_init(double* x, int ki) {
    double P  = pres(x, ki);
    double Tb = temp_back(x, ki);

    return P/RD/Tb;
}

double gc_dist(double* x) {
    double phi    = asin(x[2]/RAD_EARTH);
    double lambda = atan2(x[1], x[0]);

    return RAD_EARTH*acos(sin(PHI_C)*sin(phi) + cos(PHI_C)*cos(phi)*cos(lambda - LAMBDA_C));
}

double theta_prime(double* x, int ki) {
    double zi     = z_at_level(x, ki);
    double r      = gc_dist(x);
    double s      = THETA_PRIME_D*THETA_PRIME_D/(THETA_PRIME_D*THETA_PRIME_D + r*r);

    return THETA_PRIME_DELTA*s*sin(2.0*M_PI*zi/THETA_PRIME_LZ);
}

double theta_init(double* x, int ki) {
    return theta_back(x, ki) + theta_prime(x, ki);
}

double u_init(double* x, int ki) {
    double phi = asin(x[2]/RAD_EARTH);

    return U0*cos(phi);
}

double v_init(double* x, int ki) {
    return 0.0;
}

double rt_init(double* x, int ki) {
    return rho_init(x, ki)*theta_init(x,ki);
}

double exner_init(double* x, int ki) {
    double pi = pres(x, ki);

    return CP*pow(pi/P0, RD/CP);
}

double theta_t_init(double* x, int ki) {
    return theta_init(x, NK);
}

double theta_b_init(double* x, int ki) {
    return theta_init(x, 0);
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
    l2Vecs->UpdateLocal();
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
    double dt = 0.5;
    int nSteps = 7200; // 1 hour
    int dumpEvery = 900;//100;
    ofstream file;
    Topo* topo;
    Geom* geom;
    Euler* pe;
    Vec *velx, *velz, *rho, *rt, *exner;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo, NK);
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

    // initialise the potential temperature top and bottom boundary conditions
    //pe->initTheta(pe->theta_b, theta_b_init);
    //pe->initTheta(pe->theta_t, theta_t_init);

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
