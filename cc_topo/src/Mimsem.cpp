#include <iostream>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>

#include "Topo.h"

using namespace std;

int main(int argc, char** argv) {
	int rank, size;
    static char help[] = "petsc";
    Topo* topo;

    PetscInitialize(&argc, &argv, (char*)0, help);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);

    delete topo;

    PetscFinalize();

    return 0;
}
