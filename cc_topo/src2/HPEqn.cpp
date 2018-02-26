#include <iostream>
#include <fstream>

#include <mpi.h>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscksp.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"
#include "HPEqn.h"

using namespace std;

HPEqn::HPEqn(Topo* _topo, Geom* _geom, int _nLevs, double* _levs) {
    int k;

    topo = _topo;
    geom = _geom;
    nLevs = _nLevs;
    levs = _levs;

    R = 8.3144598;
    cp = 1.0035;

    sw = new SWEqn(topo, geom);

    ui = new Vec[nLevs];
    uh = new Vec[nLevs];
    uf = new Vec[nLevs];

    wi = new Vec[nLevs];
    wh = new Vec[nLevs];
    wf = new Vec[nLevs];

    Phii = new Vec[nLevs];
    Phih = new Vec[nLevs];
    Phif = new Vec[nLevs];

    Ti = new Vec[nLevs];
    Th = new Vec[nLevs];
    Tf = new Vec[nLevs];

    for(k = 0; k < nLevs; k++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui[k]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uh[k]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uf[k]);

        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &wi[k]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &wh[k]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &wf[k]);

        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Phii[k]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Phih[k]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Phif[k]);

        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Ti[k]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Th[k]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Tf[k]);
    }
}

HPEqn::~HPEqn() {
    int k;

    delete sw;

    for(k = 0; k < nLevs; k++) {
        VecDestroy(&ui[k]);
        VecDestroy(&uh[k]);
        VecDestroy(&uf[k]);

        VecDestroy(&wi[k]);
        VecDestroy(&wh[k]);
        VecDestroy(&wf[k]);

        VecDestroy(&Phii[k]);
        VecDestroy(&Phih[k]);
        VecDestroy(&Phif[k]);

        VecDestroy(&Ti[k]);
        VecDestroy(&Th[k]);
        VecDestroy(&Tf[k]);
    }

    delete[] ui;
    delete[] uh;
    delete[] uf;

    delete[] wi;
    delete[] wh;
    delete[] wf;

    delete[] Phii;
    delete[] Phih;
    delete[] Phif;

    delete[] Ti;
    delete[] Th;
    delete[] Tf;
}

// diagnose vertical velocities via point wise representation of the divergence theorem
void HPEqn::diagnose_vertVel(Vec* u, Vec* w) {
    int k;
    Vec du, wj;

    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &du);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &wj);
    VecZeroEntries(wj);

    for(k = nLevs - 1; k > -1; k--) {
        MatMult(sw->NtoE->E10, u[k], du);
        VecCopy(wj, w[k]);
        VecAXPY(w[k], -1.0, du);

        VecCopy(w[k], wj);
    }

    VecDestroy(&du);
    VecDestroy(&wj);
}

void HPEqn::diagnose_geoPot(Vec* T, Vec* Phi) {
}

void HPEqn::diagnose_wT(Vec* w, Vec* T, Vec* wT) {
}

void HPEqn::diagnose_wdudz(Vec* u) {
}

void HPEqn::solve(double dt) {
}
