#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "Topo.h"
#include "Geom.h"

Geom::Geom(int _pi) {
    int ii, jj;
    double coord;

    pi = _pi;

    x = new double*[n0];
    for(ii = 0; ii < n0; ii++) {
        x[ii] = new double[3];
    }

    // determine the number of global nodes
    sprintf(filename, "geom_%.4u.txt", pi);
    file.open(filename);
    nl = 0;
    while (std::getline(file, line))
        ++nl;
    file.close();

    sprintf(filename, "geom_%.4u.txt", pi);
    file.open(filename);
    ii = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        jj = 0;
        while (ss >> value) {
           x[ii,jj] = value;
           jj++;
        }
    }
    file.close();
}

Geom::~Geom() {
    for(ii = 0; ii < n0; ii++) {
        delete[] x[ii];
    }
    delete[] x;
}
