#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>

#include "Topo.h"

using namespace std;
using std::string;

Topo::Topo(int _elOrd, int _nElsX) {
    elOrd = _elOrd;
    nElsX = _nElsX;
    nDofsX = elOrd*nElsX;

    n0 = nDofsX;
    n1 = nDofsX;
    n2 = nDofsX;

    inds0 = new int[n0];
    inds1 = new int[n1];
    inds2 = new int[n2];
}

Topo::~Topo() {
    delete[] inds0;
    delete[] inds1;
    delete[] inds2;
}

int* Topo::elInds0(int ex) {
    int ix;

    for(ix = 0; ix < elOrd + 1; ix++) {
        inds0[ix] = elOrd*ex + ix;
    }
    if(ex == nElsX-1) inds0[elOrd] = 0;

    return inds0;
}

int* Topo::elInds1(int ex) {
    int ix;

    for(ix = 0; ix < elOrd + 1; ix++) {
        inds1[ix] = elOrd*ex + ix;
    }
    if(ex == nElsX-1) inds1[elOrd] = 0;

    return inds1;
}

int* Topo::elInds2(int ex) {
    int ix;

    for(ix = 0; ix < elOrd + 1; ix++) {
        inds2[ix] = elOrd*ex + ix;
    }

    return inds2;
}
