#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <petscvec.h>
#include <petscviewerhdf5.h>

#include "Basis.h"
#include "Topo.h"
#include "Geom.h"

using namespace std;
using std::string;

//#define WITH_HDF5

Geom::Geom(int _pi, Topo* _topo) {
    int ii, jj;
    ifstream file;
    char filename[100];
    string line;
    double value;

    pi = _pi;
    topo = _topo;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

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
    s = new double*[nl];
    for(ii = 0; ii < nl; ii++) {
        x[ii] = new double[3];
        s[ii] = new double[2];
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
        s[ii][0] = atan2(x[ii][1],x[ii][0]);
        s[ii][1] = asin(x[ii][2]);
        //cout << ii << "\t" << x[ii][0] << "\t" << x[ii][1] << "\t" << x[ii][2] << endl;
        ii++;
    }
    file.close();
}

Geom::~Geom() {
    int ii;

    for(ii = 0; ii < nl; ii++) {
        delete[] x[ii];
        delete[] s[ii];
    }
    delete[] x;
    delete[] s;

    delete edge;
    delete node;
    delete quad;
}

// isoparametric jacobian, with the global coordinate approximated as an expansion over the test 
// functions. derivatives are evaluated from the lagrange polynomial derivatives within each element
void Geom::jacobian(int ex, int ey, int px, int py, double** J) {
/*
    int ii, jj, mp1, mp12;
    int* inds_0 = topo->elInds0_l(ex, ey);
    double a, b, la, lb, dla, dlb;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;
    a = quad->x[px];
    b = quad->x[py];

    J[0][0] = J[0][1] = J[1][0] = J[1][1] = 0.0;

    for(ii = 0; ii < mp12; ii++) {
        jj = inds_0[ii];

        la = node->eval(a, ii%mp1);
        lb = node->eval(b, ii/mp1);
        dla = node->evalDeriv(a, ii%mp1);
        dlb = node->evalDeriv(b, ii/mp1);

        J[0][0] += dla*lb*s[jj][0]*cos(s[jj][1]);
        J[0][1] += la*dlb*s[jj][0]*cos(s[jj][1]);
        J[1][0] += dla*lb*s[jj][1];
        J[1][1] += la*dlb*s[jj][1];
    }
*/
    int ii, jj, kk, mp1 = quad->n + 1;
    int* inds_0 = topo->elInds0_l(ex, ey);
    double* c1 = x[inds_0[0]];
    double* c2 = x[inds_0[mp1-1]];
    double* c3 = x[inds_0[mp1*mp1-1]];
    double* c4 = x[inds_0[(mp1-1)*(mp1)]];
    double* ss = s[inds_0[py*mp1+px]];
    double x1, x2, rTildeMagInv;
    double rTilde[3];
    double A[2][3], B[3][3], C[3][4], D[4][2], AB[2][3], ABC[2][4];

    x1 = quad->x[px];
    x2 = quad->x[py];
    rTilde[0] = 0.25*((1.0-x1)*(1.0-x2)*c1[0] + (1.0+x1)*(1.0-x2)*c2[0] + (1.0+x1)*(1.0+x2)*c3[0] + (1.0-x1)*(1.0+x2)*c4[0]);
    rTilde[1] = 0.25*((1.0-x1)*(1.0-x2)*c1[1] + (1.0+x1)*(1.0-x2)*c2[1] + (1.0+x1)*(1.0+x2)*c3[1] + (1.0-x1)*(1.0+x2)*c4[1]);
    rTilde[2] = 0.25*((1.0-x1)*(1.0-x2)*c1[2] + (1.0+x1)*(1.0-x2)*c2[2] + (1.0+x1)*(1.0+x2)*c3[2] + (1.0-x1)*(1.0+x2)*c4[2]);
    rTildeMagInv = 1.0/sqrt(rTilde[0]*rTilde[0] + rTilde[1]*rTilde[1] + rTilde[2]*rTilde[2]);

    A[0][0] = -sin(ss[0]); A[0][1] = +cos(ss[0]); A[0][2] = 0.0;
    A[1][0] =         0.0; A[1][1] =         0.0; A[1][2] = 1.0;

    B[0][0] = +sin(ss[0])*sin(ss[0])*cos(ss[1])*cos(ss[1]) + sin(ss[1])*sin(ss[1]);
    B[0][1] = -0.5*sin(2.0*ss[0])*cos(ss[1])*cos(ss[1]);
    B[0][2] = -0.5*cos(ss[0])*sin(2.0*ss[1]);

    B[1][0] = -0.5*sin(2.0*ss[0])*cos(ss[1])*cos(ss[1]);
    B[1][1] = +cos(ss[0])*cos(ss[0])*cos(ss[1])*cos(ss[1]) + sin(ss[1])*sin(ss[1]);
    B[1][2] = -0.5*sin(ss[0])*sin(2.0*ss[1]);

    B[2][0] = -cos(ss[0])*sin(ss[1]);
    B[2][1] = -sin(ss[0])*sin(ss[1]);
    B[2][2] = +cos(ss[1]);

    C[0][0] = c1[0]; C[0][1] = c2[0]; C[0][2] = c3[0]; C[0][3] = c4[0];
    C[1][0] = c1[1]; C[1][1] = c2[1]; C[1][2] = c3[1]; C[1][3] = c4[1];
    C[2][0] = c1[2]; C[2][1] = c2[2]; C[2][2] = c3[2]; C[2][3] = c4[2];

    D[0][0] = -1.0 + x2; D[0][1] = -1.0 + x1;
    D[1][0] = +1.0 - x2; D[1][1] = -1.0 - x1;
    D[2][0] = +1.0 + x2; D[2][1] = +1.0 + x1;
    D[3][0] = -1.0 - x2; D[3][1] = +1.0 - x1;

    for(ii = 0; ii < 2; ii++) {
        for(jj = 0; jj < 3; jj++) {
            AB[ii][jj] = 0.0;
            for(kk = 0; kk < 3; kk++) {
                AB[ii][jj] += A[ii][kk]*B[kk][jj];
            }
        }
    }

    for(ii = 0; ii < 2; ii++) {
        for(jj = 0; jj < 4; jj++) {
            ABC[ii][jj] = 0.0;
            for(kk = 0; kk < 3; kk++) {
                ABC[ii][jj] += AB[ii][kk]*C[kk][jj];
            }
        }
    }

    for(ii = 0; ii < 2; ii++) {
        for(jj = 0; jj < 2; jj++) {
            J[ii][jj] = 0.0;
            for(kk = 0; kk < 4; kk++) {
                J[ii][jj] += ABC[ii][kk]*D[kk][jj];
            }
        }
    }

    J[0][0] *= 0.25*rTildeMagInv;
    J[0][1] *= 0.25*rTildeMagInv;
    J[1][0] *= 0.25*rTildeMagInv;
    J[1][1] *= 0.25*rTildeMagInv;
}

double Geom::jacDet(int ex, int ey, int px, int py, double** J) {
    jacobian(ex, ey, px, py, J);

    return (J[0][0]*J[1][1] - J[0][1]*J[1][0]);
}

void Geom::write0(Vec q, char* fieldname, int tstep) {
    int ex, ey, ii, jj, mp1, mp12;
    int* inds0;
    char filename[100];
    PetscScalar *qArray, *qxArray;
    Vec ql, qxl, qxg;
    PetscViewer viewer;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
    VecZeroEntries(ql);
    VecScatterBegin(topo->gtol_0, q, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, q, ql, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &qxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &qxg);

    VecGetArray(ql, &qArray);
    VecGetArray(qxl, &qxArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                jj = inds0[ii];
                qxArray[jj] = qArray[jj];
            }
        }
    }
    VecRestoreArray(ql, &qArray);
    VecRestoreArray(qxl, &qxArray);

    VecScatterBegin(topo->gtol_0, qxl, qxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, qxl, qxg, INSERT_VALUES, SCATTER_REVERSE);

#ifdef WITH_HDF5
    sprintf(filename, "%s_%.4u.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "%s_%.4u.dat", fieldname, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecView(qxg, viewer);
    PetscViewerDestroy(&viewer);

    VecDestroy(&ql);
    VecDestroy(&qxl);
    VecDestroy(&qxg);
}

void Geom::write1(Vec u, char* fieldname, int tstep) {
    int ex, ey, ii, jj, mp1, mp12, nn, np1, n2;
    int *inds0, *inds1x, *inds1y;
    char filename[100];
    double valx, valy, jac;
    double** Jm;
    Vec ul, uxg, uxl, vxl;
    PetscViewer viewer;
    PetscScalar *uArray, *uxArray, *vxArray;

    nn = topo->elOrd;
    np1 = topo->elOrd + 1;
    n2 = nn*np1;
    mp1 = quad->n + 1;
    mp12 = mp1*mp1;
    Jm = new double*[2];
    Jm[0] = new double[2];
    Jm[1] = new double[2];

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecZeroEntries(ul);
    VecScatterBegin(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &uxl);
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &vxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &uxg);

    VecGetArray(ul, &uArray);
    VecGetArray(uxl, &uxArray);
    VecGetArray(vxl, &vxArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            inds1x = topo->elInds1x_l(ex, ey);
            inds1y = topo->elInds1y_l(ex, ey);

            // loop over quadrature points
            for(ii = 0; ii < mp12; ii++) {
                valx = 0.0;
                valy = 0.0;
                // loop over trial functions
                for(jj = 0; jj < n2; jj++) {
                    valx += uArray[inds1x[jj]]*node->ljxi[ii%mp1][jj%np1]*edge->ejxi[ii/mp1][jj/np1];
                    valy += uArray[inds1y[jj]]*edge->ejxi[ii%mp1][jj%nn]*node->ljxi[ii/mp1][jj/nn];
                }
                jac = jacDet(ex, ey, ii%mp1, ii/mp1, Jm);

                uxArray[inds0[ii]] = (Jm[0][0]*valx + Jm[0][1]*valy)/jac;
                vxArray[inds0[ii]] = (Jm[1][0]*valx + Jm[1][1]*valy)/jac;
            }
        }
    }
    VecRestoreArray(uxl, &uxArray);
    VecRestoreArray(vxl, &vxArray);
    VecRestoreArray(ul, &uArray);

    // scatter and write the zonal components
#ifdef WITH_HDF5
    sprintf(filename, "%s_%.4u_x.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "%s_%.4u_x.dat", fieldname, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecZeroEntries(uxg);
    VecScatterBegin(topo->gtol_0, uxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, uxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecView(uxg, viewer);
    PetscViewerDestroy(&viewer);

    // scatter and write the meridional components
#ifdef WITH_HDF5
    sprintf(filename, "%s_%.4u_y.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "%s_%.4u_y.dat", fieldname, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecZeroEntries(uxg);
    VecScatterBegin(topo->gtol_0, vxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, vxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecView(uxg, viewer);
    PetscViewerDestroy(&viewer);

    VecDestroy(&ul);
    VecDestroy(&uxl);
    VecDestroy(&vxl);
    VecDestroy(&uxg);

    delete[] Jm[0];
    delete[] Jm[1];
    delete[] Jm;
}

// interpolate 2 form field to quadrature points
void Geom::write2(Vec h, char* fieldname, int tstep) {
    int ex, ey, ii, jj, mp1, mp12, nn, n2;
    int *inds0, *inds2;
    char filename[100];
    double jac, val;
    double** Jm;
    Vec hl, hxl, hxg;
    PetscScalar *hxArray, *hArray;
    PetscViewer viewer;

    nn = topo->elOrd;
    n2 = nn*nn;
    mp1 = quad->n + 1;
    mp12 = mp1*mp1;
    Jm = new double*[2];
    Jm[0] = new double[2];
    Jm[1] = new double[2];

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecScatterBegin(topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &hxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &hxg);
    VecZeroEntries(hxg);

    VecGetArray(hl, &hArray);
    VecGetArray(hxl, &hxArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            inds2 = topo->elInds2_l(ex, ey);

            // loop over quadrature points
            for(ii = 0; ii < mp12; ii++) {
                val = 0.0;
                // loop over trial functions
                for(jj = 0; jj < n2; jj++) {
                    val += hArray[inds2[jj]]*edge->ejxi[ii%mp1][jj%nn]*edge->ejxi[ii/mp1][jj/nn];
                }
                jac = jacDet(ex, ey, ii%mp1, ii/mp1, Jm);

                hxArray[inds0[ii]] = val/jac;
            }
        }
    }
    VecRestoreArray(hl, &hArray);
    VecRestoreArray(hxl, &hxArray);

    VecScatterBegin(topo->gtol_0, hxl, hxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, hxl, hxg, INSERT_VALUES, SCATTER_REVERSE);

#ifdef WITH_HDF5
    sprintf(filename, "%s_%.4u.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "%s_%.4u.dat", fieldname, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecView(hxg, viewer);

    PetscViewerDestroy(&viewer);
    VecDestroy(&hxg);
    VecDestroy(&hxl);
    VecDestroy(&hl);

    delete[] Jm[0];
    delete[] Jm[1];
    delete[] Jm;
}
