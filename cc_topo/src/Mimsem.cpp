#include <iostream>

#include <cmath>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>

#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"

using namespace std;

#define EL_ORD 2
#define N_ELS_X_LOC 2

void init(Geom* geom, Vec h) {
    int ii;
    Vec hl;
    PetscScalar* hArray;

    // TODO: galerkin projection
    VecCreateMPI(MPI_COMM_WORLD, geom->topo->n2, PETSC_DETERMINE, &hl);
    VecGetArray(hl, &hArray);
    for(ii = 0; ii < geom->topo->n2; ii++) {
        hArray[ii] = 0.01*pow(geom->x[ii][2],2.0);
    }
    VecRestoreArray(hl, &hArray);
    VecScatterBegin(geom->topo->gtol_2, hl, h, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(geom->topo->gtol_2, hl, h, INSERT_VALUES, SCATTER_REVERSE);

    VecDestroy(&hl);
}

int main(int argc, char** argv) {
	int rank, size;
    static char help[] = "petsc";
    Topo* topo;
    Geom* geom;
    SWEqn* sw;
    Vec ui, hi, uf, hf;

    PetscInitialize(&argc, &argv, (char*)0, help);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank, EL_ORD, N_ELS_X_LOC);
    geom = new Geom(rank, topo);

    sw = new SWEqn(topo, geom);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uf);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hf);

    init(geom, hi);

    sw->solve(ui, hi, uf, hf, 0.1);

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
