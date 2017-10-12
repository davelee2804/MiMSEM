#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <petscvec.h>

#include "Topo.h"

using namespace std;
using std::string;

Topo::Topo(int _pi) {
    ifstream file;
	char filename[100];
	string line;

    pi = _pi;

    // loading the nodes for this processor
    sprintf(filename, "nodes_%.4u.txt", pi);
    file.open(filename);
    n0 = 0;
    while (std::getline(file, line))
        ++n0;
	file.close();

    cout << "number of nodes on processor " << pi << ":\t" << n0 << endl;

    inds0 = new int[n0];
    LoadObjs(filename, inds0);

    // loading the edges (x-normal) for this processor
    sprintf(filename, "edges_x_%.4u.txt", pi);
    file.open(filename);
    n1x = 0;
    while (std::getline(file, line))
        ++n1x;
	file.close();

    cout << "number of edges (x-normal) on processor " << pi << ":\t" << n1x << endl;

    inds1x = new int[n1x];
    LoadObjs(filename, inds1x);

    // loading the edges (y-normal) for this processor
    sprintf(filename, "edges_y_%.4u.txt", pi);
    file.open(filename);
    n1y = 0;
    while (std::getline(file, line))
        ++n1y;
	file.close();

    cout << "number of edges (y-normal) on processor " << pi << ":\t" << n1y << endl;

    inds1y = new int[n1y];
    LoadObjs(filename, inds1y);

    // loading the faces for this processor
    sprintf(filename, "faces_%.4u.txt", pi);
    file.open(filename);
    n2 = 0;
    while (std::getline(file, line))
        ++n2;
	file.close();

    cout << "number of faces on processor " << pi << ":\t" << n2 << endl;

    inds2 = new int[n2];
    LoadObjs(filename, inds2);

    // create the local to global index mappings
    ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, n0, inds0, PETSC_COPY_VALUES, &map0);
    ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, n1x, inds1x, PETSC_COPY_VALUES, &map1x);
    ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, n1y, inds1y, PETSC_COPY_VALUES, &map1y);
    ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, n2, inds2, PETSC_COPY_VALUES, &map2);
}

Topo::~Topo() {
    delete[] inds0;
    delete[] inds1x;
    delete[] inds1y;
    delete[] inds2;

    ISLocalToGlobalMappingDestroy(&map0);
    ISLocalToGlobalMappingDestroy(&map1x);
    ISLocalToGlobalMappingDestroy(&map1y);
    ISLocalToGlobalMappingDestroy(&map2);
}

void Topo::LoadObjs(char* filename, int* inds) {
    int ii = 0;
    ifstream file;
	string line;

    file.open(filename);
    while (std::getline(file, line)) {
        inds0[ii] = atoi(line.c_str());
        //cout << pi << ":\t" << ii << "\t" << inds0[ii] << endl;
        ii++;
    }
	file.close();
}
