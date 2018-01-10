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

#define EL_ORD 3
#define N_ELS_X_LOC 8
#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0

// 1/(R.cos(phi))d.p/d.theta
double u_init(double* x) {
    double theta = atan2(x[1],x[0]);
    double phi = asin(x[2]/RAD_SPHERE);

    //return 1.0/RAD_SPHERE/cos(phi)*(-RAD_SPHERE*cos(phi)*sin(theta));
    return -sin(theta);
}

// (1/R)d.p/d.phi
double v_init(double* x) {
    double theta = atan2(x[1],x[0]);
    double phi = asin(x[2]/RAD_SPHERE);

    //return 1.0/RAD_SPHERE*(-RAD_SPHERE*sin(phi)*cos(theta));
    return -sin(phi)*cos(theta);
}

double p_init(double* x) {
    return x[0]; // R.cos(phi).cos(theta)
}

int main(int argc, char** argv) {
    int size, rank;
    double err;
    static char help[] = "petsc";
    char fieldname[20];
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Test* test;
    Vec un, ua, pi, M2h, dM2h;
    PC pc;
    KSP ksp;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank, EL_ORD, N_ELS_X_LOC);
    geom = new Geom(rank, topo);
    sw = new SWEqn(topo, geom);
    test = new Test(sw);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &un);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ua);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &pi);

    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n1l, sw->topo->nDofs1G, &dM2h);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &M2h);

    sw->init1(ua, u_init, v_init);
    sw->init2(pi, p_init);

    MatMult(sw->M2->M, pi, M2h);
    MatMult(sw->EtoF->E12, M2h, dM2h);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, sw->M1->M, sw->M1->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*sw->topo->elOrd*(sw->topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp,"test_grad_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, dM2h, un);

    sprintf(fieldname,"velocity_n");
    geom->write1(un,fieldname,0);
    sprintf(fieldname,"velocity_a");
    geom->write1(ua,fieldname,0);
    sprintf(fieldname,"pressure");
    geom->write2(pi,fieldname,0);

    // TODO: check that the pressure component is correct for H(div) error
    err = sw->err1(un, u_init, v_init, p_init);
    if(!rank) cout << "H(div) velocity error: " << err << endl;

    delete topo;
    delete geom;
    delete sw;
    delete test;

    VecDestroy(&un);
    VecDestroy(&ua);
    VecDestroy(&pi);
    VecDestroy(&M2h);
    VecDestroy(&dM2h);
    KSPDestroy(&ksp);

    PetscFinalize();

    return 0;
}
