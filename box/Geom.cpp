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
#define _LX (1000.0)

Geom::Geom(Topo* _topo, int _nk) {
    int ii, jj;
    ifstream file;
    char filename[100];
    string line;
    double value;

    pi   = _topo->pi;
    topo = _topo;
    nk   = _nk;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // determine the number of global nodes
    sprintf(filename, "input/geom_%.4u.txt", pi);
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

    sprintf(filename, "input/geom_%.4u.txt", pi);
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

    det = new double*[topo->nElsX*topo->nElsX];
    J = new double***[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        det[ii] = new double[(quad->n+1)*(quad->n+1)];
        J[ii] = new double**[(quad->n+1)*(quad->n+1)];
        for(jj = 0; jj < (quad->n+1)*(quad->n+1); jj++) {
            J[ii][jj] = new double*[2];
            J[ii][jj][0] = new double[2];
            J[ii][jj][1] = new double[2];
        }
    }

    initJacobians();

    // initialise vertical level arrays
    topog = new double[topo->n0];
    levs = new double*[nk+1];
    for(ii = 0; ii < nk + 1; ii++) {
        levs[ii] = new double[topo->n0];
    }
    thick = new double*[nk];
    for(ii = 0; ii < nk; ii++) {
        thick[ii] = new double[topo->n0];
    }
}

Geom::~Geom() {
    int ii, jj;

    for(ii = 0; ii < nl; ii++) {
        delete[] x[ii];
    }
    delete[] x;

    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        for(jj = 0; jj < (quad->n+1)*(quad->n+1); jj++) {
            delete[] J[ii][jj][0];
            delete[] J[ii][jj][1];
            delete[] J[ii][jj];
        }
        delete[] J[ii];
        delete[] det[ii];
    }
    delete[] J;
    delete[] det;

    delete edge;
    delete node;
    delete quad;

    // free the topography
    delete[] topog;
    for(ii = 0; ii < nk + 1; ii++) {
        delete[] levs[ii];
    }
    delete[] levs;
    for(ii = 0; ii < nk; ii++) {
        delete[] thick[ii];
    }
    delete[] thick;
}

// Local to global Jacobian mapping
void Geom::jacobian(int ex, int ey, int px, int py, double** jac, int n_procs_x) {
    jac[0][0] = 0.5*_LX/(topo->nElsX*n_procs_x);
    jac[0][1] = 0.0;
    jac[1][0] = 0.0;
    jac[1][1] = 0.5*_LX/(topo->nElsX*n_procs_x);
}

double Geom::jacDet(int ex, int ey, int px, int py, double** jac, int n_procs_x) {
    jacobian(ex, ey, px, py, jac, n_procs_x);

    return fabs(jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]);
}

void Geom::interp0(int ex, int ey, int px, int py, double* vec, double* val) {
    int jj, mp1;
    int* inds0 = topo->elInds0_l(ex, ey);

    mp1 = quad->n + 1;
    jj = py*mp1 + px;

    // assumes diagonal mass matrix for 0 forms
    val[0] = vec[inds0[jj]];
}

void Geom::interp1_l(int ex, int ey, int px, int py, double* vec, double* val) {
    int jj, nn, np1, n2;
    int *inds1x, *inds1y;

    inds1x = topo->elInds1x_l(ex, ey);
    inds1y = topo->elInds1y_l(ex, ey);

    nn = topo->elOrd;
    np1 = topo->elOrd + 1;
    n2 = nn*np1;

    val[0] = 0.0;
    val[1] = 0.0;
    for(jj = 0; jj < n2; jj++) {
        val[0] += vec[inds1x[jj]]*node->ljxi[px][jj%np1]*edge->ejxi[py][jj/np1];
        val[1] += vec[inds1y[jj]]*edge->ejxi[px][jj%nn]*node->ljxi[py][jj/nn];
    }
}

void Geom::interp2_l(int ex, int ey, int px, int py, double* vec, double* val) {
    int jj, nn, n2;
    int* inds2 = topo->elInds2_l(ex, ey);

    nn = topo->elOrd;
    n2 = nn*nn;

    val[0] = 0.0;
    for(jj = 0; jj < n2; jj++) {
        val[0] += vec[inds2[jj]]*edge->ejxi[px][jj%nn]*edge->ejxi[py][jj/nn];
    }
}

void Geom::interp1_g(int ex, int ey, int px, int py, double* vec, double* val) {
    int el = ey*topo->nElsX + ex;
    int pi = py*(quad->n+1) + px;
    double val_l[2];
    double dj = det[el][pi];
    double** jac = J[el][pi];

    interp1_l(ex, ey, px, py, vec, val_l);

    val[0] = (jac[0][0]*val_l[0] + jac[0][1]*val_l[1])/dj;
    val[1] = (jac[1][0]*val_l[0] + jac[1][1]*val_l[1])/dj;
}

// global interpolation with H(curl) form of the Piola transformation
// output is [du/dz, dvdz]
void Geom::interp1_g_t(int ex, int ey, int px, int py, double* vec, double* val) {
    int el = ey*topo->nElsX + ex;
    int pi = py*(quad->n+1) + px;
    double val_l[2];
    double dj = det[el][pi];
    double** jac = J[el][pi];

    interp1_l(ex, ey, px, py, vec, val_l);

    // once we have mapped degrees of freedom from inner orientations
    // to outer orientations, this transformation is the same as for
    // the H(div) space, and so may be depricated
    val[0] = (jac[0][0]*val_l[0] + jac[0][1]*val_l[1])/dj;
    val[1] = (jac[1][0]*val_l[0] + jac[1][1]*val_l[1])/dj;
}

void Geom::interp2_g(int ex, int ey, int px, int py, double* vec, double* val) {
    int el = ey*topo->nElsX + ex;
    int pi = py*(quad->n+1) + px;
    double val_l[1];
    double dj = det[el][pi];

    interp2_l(ex, ey, px, py, vec, val_l);

    val[0] = val_l[0]/dj;
}

void Geom::write0(Vec q, char* fieldname, int tstep, int lev) {
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
                // assume piecewise constant in the vertical, so rescale by
                // the vertical determinant inverse
                qxArray[jj] *= 1.0/thick[lev][jj];
            }
        }
    }
    VecRestoreArray(ql, &qArray);
    VecRestoreArray(qxl, &qxArray);

    VecScatterBegin(topo->gtol_0, qxl, qxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, qxl, qxg, INSERT_VALUES, SCATTER_REVERSE);

#ifdef WITH_HDF5
    sprintf(filename, "output/%s_%.3u_%.4u.h5", fieldname, lev, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "output/%s_%.3u_%.4u.dat", fieldname, lev, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecView(qxg, viewer);
    PetscViewerDestroy(&viewer);

    VecDestroy(&ql);
    VecDestroy(&qxl);
    VecDestroy(&qxg);
}

void Geom::write1(Vec u, char* fieldname, int tstep, int lev) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    char filename[100];
    double val[2];
    Vec ul, uxg, uxl, vxl;
    PetscViewer viewer;
    PetscScalar *uArray, *uxArray, *vxArray;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

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

            // loop over quadrature points
            for(ii = 0; ii < mp12; ii++) {
                interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, val);

                uxArray[inds0[ii]] = val[0];
                vxArray[inds0[ii]] = val[1];
                // assume piecewise constant in the vertical, so rescale by
                // the vertical determinant inverse
                uxArray[inds0[ii]] *= 1.0/thick[lev][inds0[ii]];
                vxArray[inds0[ii]] *= 1.0/thick[lev][inds0[ii]];
            }
        }
    }
    VecRestoreArray(uxl, &uxArray);
    VecRestoreArray(vxl, &vxArray);
    VecRestoreArray(ul, &uArray);

    // scatter and write the zonal components
#ifdef WITH_HDF5
    sprintf(filename, "output/%s_x_%.3u_%.4u.h5", fieldname, lev, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "output/%s_x_%.3u_%.4u.dat", fieldname, lev, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecZeroEntries(uxg);
    VecScatterBegin(topo->gtol_0, uxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, uxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecView(uxg, viewer);
    PetscViewerDestroy(&viewer);

    // scatter and write the meridional components
#ifdef WITH_HDF5
    sprintf(filename, "output/%s_y_%.3u_%.4u.h5", fieldname, lev, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "output/%s_y_%.3u_%.4u.dat", fieldname, lev, tstep);
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

    // also write the vector itself
    sprintf(filename, "output/%s_%.3u_%.4u.vec", fieldname, lev, tstep);
    PetscViewerBinaryOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    VecView(u, viewer);
    PetscViewerDestroy(&viewer);
}

// interpolate 2 form field to quadrature points
void Geom::write2(Vec h, char* fieldname, int tstep, int lev, bool vert_scale) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    char filename[100];
    double val, fac;
    Vec hxl, hxg;
    PetscScalar *hxArray, *hArray;
    PetscViewer viewer;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

//    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
//    VecScatterBegin(topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);
//    VecScatterEnd(topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &hxl);
    VecZeroEntries(hxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &hxg);
    VecZeroEntries(hxg);

//    VecGetArray(hl, &hArray);
    VecGetArray(h, &hArray);
    VecGetArray(hxl, &hxArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);

            // loop over quadrature points
            for(ii = 0; ii < mp12; ii++) {
                interp2_g(ex, ey, ii%mp1, ii/mp1, hArray, &val);

                // assume piecewise constant in the vertical, so rescale by
                // the vertical determinant inverse
                if(vert_scale) {
                    val *= 1.0/thick[lev][inds0[ii]];
                }
                if(ii == 0 || ii == mp1-1 || ii == (mp1-1)*mp1 || ii == mp1*mp1-1) {
                    fac = 0.25;
                } else if(ii/mp1 == 0 || ii/mp1 == mp1-1 || ii%mp1 == 0 || ii%mp1 == mp1-1) {
                    fac = 0.5;
                } else {
                    fac = 1.0;
                }
                hxArray[inds0[ii]] += (fac*val);
/*
                hxArray[inds0[ii]] = val;
*/
            }
        }
    }
//    VecRestoreArray(hl, &hArray);
    VecRestoreArray(h, &hArray);
    VecRestoreArray(hxl, &hxArray);

//    VecScatterBegin(topo->gtol_0, hxl, hxg, INSERT_VALUES, SCATTER_REVERSE);
//    VecScatterEnd(  topo->gtol_0, hxl, hxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterBegin(topo->gtol_0, hxl, hxg, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  topo->gtol_0, hxl, hxg, ADD_VALUES, SCATTER_REVERSE);

#ifdef WITH_HDF5
    sprintf(filename, "output/%s_%.3u_%.4u.h5", fieldname, lev, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "output/%s_%.3u_%.4u.dat", fieldname, lev, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecView(hxg, viewer);

    PetscViewerDestroy(&viewer);
    VecDestroy(&hxg);
    VecDestroy(&hxl);
//    VecDestroy(&hl);

    // also write the vector itself
    sprintf(filename, "output/%s_%.3u_%.4u.vec", fieldname, lev, tstep);
    PetscViewerBinaryOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    VecView(h, viewer);
    PetscViewerDestroy(&viewer);
}

void Geom::writeVertToHoriz(Vec* vecs, char* fieldname, int tstep, int nv) {
    int ex, ey, ei, ii, kk, n2, *inds2;
    PetscScalar *hArray, *vArray;
    Vec *hvecs, gvec;

    n2 = topo->elOrd*topo->elOrd;

    hvecs = new Vec[nv];
    for(kk = 0; kk < nv; kk++) {
//        VecCreateSeq(MPI_COMM_SELF, topo->n2, &hvecs[kk]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hvecs[kk]);
        VecZeroEntries(hvecs[kk]);
    }
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &gvec);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds2 = topo->elInds2_l(ex, ey);

            VecGetArray(vecs[ei], &vArray);
            for(kk = 0; kk < nv; kk++) {
                VecGetArray(hvecs[kk], &hArray);
                for(ii = 0; ii < n2; ii++) {
                    hArray[inds2[ii]] += vArray[kk*n2+ii];
                }
                VecRestoreArray(hvecs[kk], &hArray);
            }
            VecRestoreArray(vecs[ei], &vArray);
        }
    }

    for(kk = 0; kk < nv; kk++) {
//        VecZeroEntries(gvec);
//        VecScatterBegin(topo->gtol_2, hvecs[kk], gvec, INSERT_VALUES, SCATTER_REVERSE);
//        VecScatterEnd(  topo->gtol_2, hvecs[kk], gvec, INSERT_VALUES, SCATTER_REVERSE);
        VecCopy(hvecs[kk], gvec);
        write2(gvec, fieldname, tstep, kk, false);
    }

    for(kk = 0; kk < nv; kk++) {
        VecDestroy(&hvecs[kk]);
    }
    delete[] hvecs;
    VecDestroy(&gvec);
}

void Geom::initJacobians() {
    int ex, ey, el, mp1, mp12, ii, n_procs, n_procs_x;

    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    n_procs_x = int(sqrt(n_procs));
    cout << "num procs per dim: " << n_procs_x << endl;
    if(fabs(n_procs - n_procs_x*n_procs_x) > 0.0001) {
        cout << "ERROR! number of processors per dimension is not consistent with total: " << n_procs << endl;
        abort();
    }

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            el = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det[el][ii] = jacDet(ex, ey, ii%mp1, ii/mp1, J[el][ii], n_procs_x);
            }
        }
    }
}

void Geom::initTopog(TopogFunc* ft, LevelFunc* fl) {
    int ii, jj;
    double zo;
    double max_height = (fl) ? fl(x[0], nk) : 1.0;//TODO: assumes x[0] is at the sea surface, fix later

    for(ii = 0; ii < topo->n0; ii++) {
        topog[ii] = ft(x[ii]);
    }

    for(ii = 0; ii < nk + 1; ii++) {
        for(jj = 0; jj < topo->n0; jj++) {
            zo = fl(x[jj], ii);
            levs[ii][jj] = (max_height - topog[jj])*zo/max_height + topog[jj];
        }
    }
    for(ii = 0; ii < nk; ii++) {
        for(jj = 0; jj < topo->n0; jj++) {
            thick[ii][jj] = levs[ii+1][jj] - levs[ii][jj];
        }
    }
}

void Geom::writeColumn(char* filename, int ei, int nv, Vec vec, bool vert_scale) {
    int ii, jj, kk, n2, mp1, mp12;
    double gamma, vq;
    PetscScalar* vArray;
    ofstream file;

    n2   = topo->elOrd*topo->elOrd;
    mp1  = quad->n + 1;
    mp12 = mp1*mp1;

    file.open(filename);

    VecGetArray(vec, &vArray);
    for(kk = 0; kk < nv; kk++) {
        for(ii = 0; ii < mp12; ii++) {
            vq = 0.0;
            for(jj = 0; jj < n2; jj++) {
                gamma = edge->ejxi[ii%mp1][jj%topo->elOrd]*edge->ejxi[ii/mp1][jj/topo->elOrd];
                vq += vArray[kk*n2+jj]*gamma;
            }
            vq /= det[ei][ii];

            if(vert_scale) vq /= thick[kk][ii];

            file << vq << "\t";
        }
        file << endl;
    }
    VecRestoreArray(vec, &vArray);

    file.close();
}
