#include <iostream>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>

#include "Basis.h"
#include "ElMats.h"
#include "Topo.h"
#include "Geom.h"
#include "Assembly.h"
#include "SWEqn.h"

using namespace std;

#define EL_ORD 2
#define N_ELS_X_LOC 4

int main(int argc, char** argv) {
	int rank, size;
    static char help[] = "petsc";
    Topo* topo;
    Geom* geom;
    SWEqn* sw;

    PetscInitialize(&argc, &argv, (char*)0, help);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank, EL_ORD, N_ELS_X_LOC);
    geom = new Geom(rank);

    sw = new SWEqn(topo, geom);

    delete topo;
    delete geom;
    delete sw;

    PetscFinalize();

    return 0;
}
