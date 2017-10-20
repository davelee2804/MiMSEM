#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <petscvec.h>

#include "Topo.h"
#include "Geom.h"

using namespace std;
using std::string;

Geom::Geom(int _pi) {
    int ii, jj;
    ifstream file;
    char filename[100];
    string line;
    double value;

    pi = _pi;

    // determine the number of global nodes
    sprintf(filename, "geom_%.4u.txt", pi);
    file.open(filename);
    nl = 0;
    while (std::getline(file, line))
        ++nl;
    file.close();

    if (!nl) {
        cout << "ERROR! geometry file reading: " << nl << endl;
    }

    x = new double*[nl];
    for(ii = 0; ii < nl; ii++) {
        x[ii] = new double[3];
    }

    sprintf(filename, "geom_%.4u.txt", pi);
    file.open(filename);
    ii = 0;
    while (std::getline(file, line)) {
        stringstream ss(line);
        jj = 0;
        while (ss >> value) {
           x[ii][jj] = value;
           jj++;
        }
        //cout << ii << "\t" << x[ii][0] << "\t" << x[ii][1] << "\t" << x[ii][2] << endl;
        ii++;
    }
    file.close();
}

Geom::~Geom() {
    int ii;

    for(ii = 0; ii < nl; ii++) {
        delete[] x[ii];
    }
    delete[] x;
}
