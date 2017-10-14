#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <petscvec.h>

#include "Topo.h"

using namespace std;
using std::string;

Topo::Topo(int _pi, int _elOrd, int _nElsX) {
    int ii;
    ifstream file;
	char filename[100];
	string line;

    pi = _pi;
    elOrd = _elOrd;
    nElsX = _nElsX;
    nDofsX = elOrd*nElsX;

    // loading the nodes for this processor
    sprintf(filename, "nodes_%.4u.txt", pi);
    file.open(filename);
    n0 = 0;
    while (std::getline(file, line))
        ++n0;
	file.close();

    cout << "number of nodes on processor " << pi << ":\t" << n0 << endl;

    loc0 = new int[n0];
    loadObjs(filename, loc0);

    // loading the edges (x-normal) for this processor
    sprintf(filename, "edges_x_%.4u.txt", pi);
    file.open(filename);
    n1x = 0;
    while (std::getline(file, line))
        ++n1x;
	file.close();

    cout << "number of edges (x-normal) on processor " << pi << ":\t" << n1x << endl;

    loc1x = new int[n1x];
    loadObjs(filename, loc1x);

    // loading the edges (y-normal) for this processor
    sprintf(filename, "edges_y_%.4u.txt", pi);
    file.open(filename);
    n1y = 0;
    while (std::getline(file, line))
        ++n1y;
	file.close();

    cout << "number of edges (y-normal) on processor " << pi << ":\t" << n1y << endl;

    loc1y = new int[n1y];
    loadObjs(filename, loc1y);

    // add x-normal and y-normal edges into single set of indices
    loc1 = new int[n1x+n1y];
    for(ii = 0; ii < n1x; ii++) {
       loc1[ii] = loc1x[ii];
    }
    for(ii = 0; ii < n1y; ii++) {
       loc1[ii+n1x] = loc1y[ii];
    }

    // loading the faces for this processor
    sprintf(filename, "faces_%.4u.txt", pi);
    file.open(filename);
    n2 = 0;
    while (std::getline(file, line))
        ++n2;
	file.close();

    cout << "number of faces on processor " << pi << ":\t" << n2 << endl;

    loc2 = new int[n2];
    loadObjs(filename, loc2);

    // create the local to global index mappings
    ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, n0, loc0, PETSC_COPY_VALUES, &map0);
    ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, n1x+n1y, loc1, PETSC_COPY_VALUES, &map1);
    ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, n2, loc2, PETSC_COPY_VALUES, &map2);

    // allocate the element indices arrays
    inds0 = new int[(elOrd+1)*(elOrd+1)];
    inds1x = new int[(elOrd)*(elOrd+1)];
    inds1y = new int[(elOrd+1)*(elOrd)];
    inds2 = new int[(elOrd)*(elOrd)];
}

Topo::~Topo() {
    delete[] loc0;
    delete[] loc1;
    delete[] loc1x;
    delete[] loc1y;
    delete[] loc2;

    delete[] inds0;
    delete[] inds1x;
    delete[] inds1y;
    delete[] inds2;

    ISLocalToGlobalMappingDestroy(&map0);
    ISLocalToGlobalMappingDestroy(&map1);
    ISLocalToGlobalMappingDestroy(&map2);
}

void Topo::loadObjs(char* filename, int* loc) {
    int ii = 0;
    ifstream file;
	string line;

    file.open(filename);
    while (std::getline(file, line)) {
        loc[ii] = atoi(line.c_str());
        //cout << pi << ":\t" << ii << "\t" << loc[ii] << endl;
        ii++;
    }
	file.close();
}

int* Topo::elInds0(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd + 1;  iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds0[kk] = (ey*elOrd + iy)*(nDofsX + 1) + ex*nDofsX + ix;
            kk++;
        }
    }

    return inds0;
}

int* Topo::elInds1x(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds1x[kk] = (ey*elOrd + iy)*(nDofsX + 1) + ex*elOrd + ix;
            kk++;
        }
    }

    return inds1x;
}

int* Topo::elInds1y(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd + 1; iy++) {
        for(ix = 0; ix < elOrd; ix++) {
            inds1y[kk] = (ey*elOrd + iy)*(nDofsX) + ex*elOrd + ix;
            kk++;
        }
    }

    return inds1y;
}

int* Topo::elInds2(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd ; iy++) {
        for(ix = 0; ix < elOrd; ix++) {
            inds2[kk] = (ey*elOrd + iy)*(nDofsX) + ex*elOrd + ix;
            kk++;
        }
    }

    return inds2;
}
