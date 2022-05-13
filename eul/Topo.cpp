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

Topo::Topo(int _nk) {
    Vec vl, vg;
    int ii, n_procs;
    int nLoc[4];
    ifstream file;
    char filename[100];
    string line;

    nk = _nk;

    MPI_Comm_rank(MPI_COMM_WORLD, &pi);

    // read the element order and number of elements in each dimension per processor
    sprintf(filename, "input/grid_res.txt");
    file.open(filename);
    std::getline(file, line);
    elOrd = atoi(line.c_str());
    std::getline(file, line);
    nElsX = atoi(line.c_str());
    file.close();

    if(!pi) cout << "element order: " << elOrd << "\tno. els in each dimension per proc. " << nElsX << endl;

    nDofsX = elOrd*nElsX;

    // loading the nodes for this processor
    sprintf(filename, "input/nodes_%.4u.txt", pi);
    file.open(filename);
    n0 = 0;
    while (std::getline(file, line))
        ++n0;
    file.close();

    //cout << "number of nodes on processor " << pi << ":\t" << n0 << endl;

    loc0 = new int[n0];
    loadObjs(filename, loc0);

    // loading the edges (x-normal) for this processor
    sprintf(filename, "input/edges_x_%.4u.txt", pi);
    file.open(filename);
    n1x = 0;
    while (std::getline(file, line))
        ++n1x;
    file.close();

    //cout << "number of edges (x-normal) on processor " << pi << ":\t" << n1x << endl;

    loc1x = new int[n1x];
    loadObjs(filename, loc1x);

    // loading the edges (y-normal) for this processor
    sprintf(filename, "input/edges_y_%.4u.txt", pi);
    file.open(filename);
    n1y = 0;
    while (std::getline(file, line))
        ++n1y;
    file.close();

    n1 = n1x + n1y;

    //cout << "number of edges (y-normal) on processor " << pi << ":\t" << n1y << endl;

    loc1y = new int[n1y];
    loadObjs(filename, loc1y);

    // add x-normal and y-normal edges into single set of indices
    loc1 = new int[n1x+n1y];
    for(ii = 0; ii < n1x; ii++) {
        loc1[2*ii+0] = loc1x[ii];
        loc1[2*ii+1] = loc1y[ii];
    }

    // loading the faces for this processor
    sprintf(filename, "input/faces_%.4u.txt", pi);
    file.open(filename);
    n2 = 0;
    while (std::getline(file, line))
        ++n2;
    file.close();

    //cout << "number of faces on processor " << pi << ":\t" << n2 << endl;

    loc2 = new int[n2];
    loadObjs(filename, loc2);

    // allocate the element indices arrays
    inds0_l  = new int[(elOrd+1)*(elOrd+1)];
    inds1x_l = new int[(elOrd)*(elOrd+1)];
    inds1y_l = new int[(elOrd+1)*(elOrd)];
    inds2_l  = new int[(elOrd)*(elOrd)];
    inds0_g  = new int[(elOrd+1)*(elOrd+1)];
    inds1x_g = new int[(elOrd)*(elOrd+1)];
    inds1y_g = new int[(elOrd+1)*(elOrd)];
    inds2_g  = new int[(elOrd)*(elOrd)];

    // global number of degrees of freedom
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    nDofs0G = n_procs*nDofsX*nDofsX + 2;
    nDofs1G = 2*n_procs*nDofsX*nDofsX;
    nDofs2G = n_procs*nDofsX*nDofsX;
    //cout << "n dofs global: " << nDofs0G << "\t" << nDofs1G << "\t" << nDofs2G << endl;

    // create the global index sets
    ISCreateGeneral(MPI_COMM_WORLD, n0, loc0, PETSC_COPY_VALUES, &is_g_0);
    ISCreateGeneral(MPI_COMM_WORLD, n1, loc1, PETSC_COPY_VALUES, &is_g_1);

    // create the local index sets
    ISCreateStride(MPI_COMM_SELF, n0, 0, 1, &is_l_0);
    ISCreateStride(MPI_COMM_SELF, n1, 0, 1, &is_l_1);

    // load the local sizes
    sprintf(filename, "input/local_sizes_%.4u.txt", pi);
    file.open(filename);
    ii = 0;
    while (std::getline(file, line)) {
        nLoc[ii] = atoi(line.c_str());
        ii++;
    }
    file.close();

    n0l = nLoc[0];
    n1xl = nLoc[1];
    n1yl = nLoc[2];
    n2l = nLoc[3];
    n1l = n1xl + n1yl;

    cout << "local sizes on " << pi << ":\t" << n0l << "\t" << n1l << "\t" << n2l << endl;

    // initialise the vec scatter objects for nodes/edges/faces
    VecCreateSeq(MPI_COMM_SELF, n0, &vl);
    VecCreateMPI(MPI_COMM_WORLD, n0l, nDofs0G, &vg);
    VecScatterCreate(vg, is_g_0, vl, is_l_0, &gtol_0);
    VecDestroy(&vl);
    VecDestroy(&vg);

    VecCreateSeq(MPI_COMM_SELF, n1, &vl);
    VecCreateMPI(MPI_COMM_WORLD, n1l, nDofs1G, &vg);
    VecScatterCreate(vg, is_g_1, vl, is_l_1, &gtol_1);
    VecDestroy(&vl);
    VecDestroy(&vg);

    dofs_per_proc = nk*(n1l + 4*n2l) - n2l;
    inds_velx_g  = new int[(elOrd+1)*(elOrd)];
    inds_vely_g  = new int[(elOrd+1)*(elOrd)];
    inds_rho_g   = new int[(elOrd)*(elOrd)];
    inds_theta_g = new int[(elOrd)*(elOrd)];
    inds_exner_g = new int[(elOrd)*(elOrd)];
    inds_velz_g  = new int[(elOrd)*(elOrd)];
    inds_velx_l  = new int[(elOrd+1)*(elOrd)];
    inds_vely_l  = new int[(elOrd+1)*(elOrd)];
    inds_rho_l   = new int[(elOrd)*(elOrd)];
    inds_theta_l = new int[(elOrd)*(elOrd)];
    inds_exner_l = new int[(elOrd)*(elOrd)];
    inds_velz_l  = new int[(elOrd)*(elOrd)];
}

Topo::~Topo() {
    delete[] loc0;
    delete[] loc1;
    delete[] loc1x;
    delete[] loc1y;
    delete[] loc2;

    delete[] inds0_l;
    delete[] inds1x_l;
    delete[] inds1y_l;
    delete[] inds2_l;
    delete[] inds0_g;
    delete[] inds1x_g;
    delete[] inds1y_g;
    delete[] inds2_g;

    ISDestroy(&is_g_0);
    ISDestroy(&is_g_1);

    ISDestroy(&is_l_0);
    ISDestroy(&is_l_1);

    VecScatterDestroy(&gtol_0);
    VecScatterDestroy(&gtol_1);

    delete[] inds_velx_g;
    delete[] inds_vely_g;
    delete[] inds_rho_g;
    delete[] inds_theta_g;
    delete[] inds_exner_g;
    delete[] inds_velz_g;
    delete[] inds_velx_l;
    delete[] inds_vely_l;
    delete[] inds_rho_l;
    delete[] inds_theta_l;
    delete[] inds_exner_l;
    delete[] inds_velz_l;
}

void Topo::loadObjs(char* filename, int* loc) {
    int ii = 0;
    ifstream file;
    string line;

    cout << pi << ":\tloading file: " << filename << endl;

    file.open(filename);
    while (std::getline(file, line)) {
        loc[ii] = atoi(line.c_str());
        //cout << pi << ":\t" << ii << "\t" << loc[ii] << endl;
        ii++;
    }
	file.close();
}

int* Topo::elInds0_l(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd + 1;  iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds0_l[kk] = (ey*elOrd + iy)*(nDofsX + 1) + ex*elOrd + ix;
            kk++;
        }
    }

    return inds0_l;
}

int* Topo::elInds1x_l(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds1x_l[kk] = 2*((ey*elOrd + iy)*(nDofsX + 1) + ex*elOrd + ix) + 0;
            kk++;
        }
    }

    return inds1x_l;
}

int* Topo::elInds1y_l(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd + 1; iy++) {
        for(ix = 0; ix < elOrd; ix++) {
            inds1y_l[kk] = 2*((ey*elOrd + iy)*(nDofsX) + ex*elOrd + ix) + 1;
            kk++;
        }
    }

    return inds1y_l;
}

int* Topo::elInds2_l(int ex, int ey) {
    int kk;
    int offset = (ey*nElsX+ex)*elOrd*elOrd;

    for(kk = 0; kk < elOrd*elOrd; kk++) {
        inds2_l[kk] = offset + kk;
    }

    return inds2_l;
}

int* Topo::elInds0_g(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd + 1; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds0_g[kk] = loc0[(ey*elOrd + iy)*(nDofsX + 1) + ex*elOrd + ix];
            kk++;
        }
    }

    return inds0_g;
}

int* Topo::elInds1x_g(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds1x_g[kk] = loc1x[(ey*elOrd + iy)*(nDofsX + 1) + ex*elOrd + ix];
            kk++;
        }
    }

    return inds1x_g;
}

int* Topo::elInds1y_g(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd + 1; iy++) {
        for(ix = 0; ix < elOrd; ix++) {
            inds1y_g[kk] = loc1y[(ey*elOrd + iy)*(nDofsX) + ex*elOrd + ix];
            kk++;
        }
    }

    return inds1y_g;
}

int* Topo::elInds2_g(int ex, int ey) {
    int kk;
    int offset = pi*n2;

    inds2_l = elInds2_l(ex, ey);
    for(kk = 0; kk < elOrd*elOrd; kk++) {
        inds2_g[kk] = inds2_l[kk] + offset;
    }

    return inds2_g;
}

int* Topo::elInds_velx_g(int ex, int ey, int lev) {
    int jj, nj, pj, qj;

    nj = (elOrd+1)*elOrd;

    elInds1x_g(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	pj = inds1x_g[jj] / n1l;
	qj = inds1x_g[jj] % n1l;
        inds_velx_g[jj] = pj*dofs_per_proc + lev*n1l + qj;
    }
    return inds_velx_g;
}

int* Topo::elInds_velx_l(int ex, int ey, int lev) {
    int jj, nj;

    nj = (elOrd+1)*elOrd;

    elInds1x_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
        inds_velx_l[jj] = lev*n1l + inds1x_l[jj];
    }
    return inds_velx_l;
}

int* Topo::elInds_vely_g(int ex, int ey, int lev) {
    int jj, nj, pj, qj;

    nj = (elOrd+1)*elOrd;

    elInds1y_g(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	pj = inds1y_g[jj] / n1l;
	qj = inds1y_g[jj] % n1l;
        inds_vely_g[jj] = pj*dofs_per_proc + lev*n1l + qj;
    }
    return inds_vely_g;
}

int* Topo::elInds_vely_l(int ex, int ey, int lev) {
    int jj, nj;

    nj = (elOrd+1)*elOrd;

    elInds1y_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
        inds_vely_l[jj] = lev*n1l + inds1y_l[jj];
    }
    return inds_vely_l;
}

int* Topo::elInds_rho_g(int ex, int ey, int lev) {
    int jj, nj, qj;
    int dofs_per_lev = 4;

    nj = elOrd*elOrd;

    elInds2_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	qj = inds2_l[jj];
        inds_rho_g[jj] = pi*dofs_per_proc + nk*n1l + (4*nk-1)*qj + dofs_per_lev*lev+0;
    }
    return inds_rho_g;
}

int* Topo::elInds_rho_l(int ex, int ey, int lev) {
    int jj, nj, qj;
    int dofs_per_lev = 4;

    nj = elOrd*elOrd;

    elInds2_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	qj = inds2_l[jj];
        inds_rho_l[jj] = nk*n1l + (4*nk-1)*qj + dofs_per_lev*lev+0;
    }
    return inds_rho_l;
}

int* Topo::elInds_theta_g(int ex, int ey, int lev) {
    int jj, nj, qj;
    int dofs_per_lev = 4;

    nj = elOrd*elOrd;

    elInds2_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	qj = inds2_l[jj];
        inds_theta_g[jj] = pi*dofs_per_proc + nk*n1l + (4*nk-1)*qj + dofs_per_lev*lev+1;
    }
    return inds_theta_g;
}

int* Topo::elInds_theta_l(int ex, int ey, int lev) {
    int jj, nj, qj;
    int dofs_per_lev = 4;

    nj = elOrd*elOrd;

    elInds2_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	qj = inds2_l[jj];
        inds_theta_l[jj] = nk*n1l + (4*nk-1)*qj + dofs_per_lev*lev+1;
    }
    return inds_theta_l;
}

int* Topo::elInds_exner_g(int ex, int ey, int lev) {
    int jj, nj, qj;
    int dofs_per_lev = 4;

    nj = elOrd*elOrd;

    elInds2_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	qj = inds2_l[jj];
        inds_exner_g[jj] = pi*dofs_per_proc + nk*n1l + (4*nk-1)*qj + dofs_per_lev*lev+2;
    }
    return inds_exner_g;
}

int* Topo::elInds_exner_l(int ex, int ey, int lev) {
    int jj, nj, qj;
    int dofs_per_lev = 4;

    nj = elOrd*elOrd;

    elInds2_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	qj = inds2_l[jj];
        inds_exner_l[jj] = nk*n1l + (4*nk-1)*qj + dofs_per_lev*lev+2;
    }
    return inds_exner_l;
}

int* Topo::elInds_velz_g(int ex, int ey, int lev) {
    int jj, nj, qj;

    nj = elOrd*elOrd;

    elInds2_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	qj = inds2_l[jj];
        inds_velz_g[jj] = pi*dofs_per_proc + nk*n1l + (4*nk-1)*qj + 4*lev+3;
    }
    return inds_velz_g;
}

int* Topo::elInds_velz_l(int ex, int ey, int lev) {
    int jj, nj, qj;

    nj = elOrd*elOrd;

    elInds2_l(ex, ey);
    for(jj = 0; jj < nj; jj++) {
	qj = inds2_l[jj];
        inds_velz_l[jj] = nk*n1l + (4*nk-1)*qj + 4*lev+3;
    }
    return inds_velz_l;
}

void Topo::repack(Vec* velx, Vec* rho, Vec* theta, Vec* exner, Vec* velz, Vec _x) {
    int ii, jj, kk, ex, ey, elOrd2;
    PetscScalar *xArray, *velxArray, *rhoArray, *thetaArray, *exnerArray, *velzArray;

    elOrd2 = elOrd*elOrd;

    VecGetArray(_x, &xArray);
    for(kk = 0; kk < nk; kk++) {
        VecGetArray(velx[kk], &velxArray);
        for(ii = 0; ii < n1l; ii++) {
            xArray[kk*n1l+ii] = velxArray[ii];
        }
        VecRestoreArray(velx[kk], &velxArray);
    }
    for(ii = 0; ii < nElsX*nElsX; ii++) {
        ex = ii%nElsX;
        ey = ii/nElsX;

        VecGetArray(rho[ii],   &rhoArray);
        VecGetArray(theta[ii], &thetaArray);
        VecGetArray(exner[ii], &exnerArray);
        VecGetArray(velz[ii],  &velzArray);
	for(kk = 0; kk < nk; kk++) {
            elInds_rho_l(ex, ey, kk);
            elInds_theta_l(ex, ey, kk);
            elInds_exner_l(ex, ey, kk);
	    for(jj = 0; jj < elOrd2; jj++) {
                xArray[inds_rho_l[jj]]   = rhoArray[kk*elOrd2+jj];
                xArray[inds_theta_l[jj]] = thetaArray[kk*elOrd2+jj];
                xArray[inds_exner_l[jj]] = exnerArray[kk*elOrd2+jj];
            }
            if(kk < nk-1) {
                elInds_velz_l(ex, ey, kk);
	        for(jj = 0; jj < elOrd2; jj++) {
                    xArray[inds_velz_l[jj]] = velzArray[kk*elOrd2+jj];
		}
	    }
        }
        VecRestoreArray(rho[ii],   &rhoArray);
        VecRestoreArray(exner[ii], &exnerArray);
        VecRestoreArray(theta[ii], &thetaArray);
        VecRestoreArray(velz[ii],  &velzArray);
    }
    VecRestoreArray(_x, &xArray);
}

void Topo::unpack(Vec* velx, Vec* rho, Vec* theta, Vec* exner, Vec* velz, Vec _x) {
    int ii, jj, kk, ex, ey, elOrd2;
    PetscScalar *xArray, *velxArray, *rhoArray, *thetaArray, *exnerArray, *velzArray;

    elOrd2 = elOrd*elOrd;

    VecGetArray(_x, &xArray);
    for(kk = 0; kk < nk; kk++) {
        VecGetArray(velx[kk], &velxArray);
        for(ii = 0; ii < n1l; ii++) {
            velxArray[ii] = xArray[kk*n1l+ii];
        }
        VecRestoreArray(velx[kk], &velxArray);
    }
    for(ii = 0; ii < nElsX*nElsX; ii++) {
        ex = ii%nElsX;
        ey = ii/nElsX;

        VecGetArray(rho[ii],   &rhoArray);
        VecGetArray(theta[ii], &thetaArray);
        VecGetArray(exner[ii], &exnerArray);
        VecGetArray(velz[ii],  &velzArray);
	for(kk = 0; kk < nk; kk++) {
            elInds_rho_l(ex, ey, kk);
            elInds_theta_l(ex, ey, kk);
            elInds_exner_l(ex, ey, kk);
	    for(jj = 0; jj < elOrd2; jj++) {
		rhoArray[kk*elOrd2+jj]   = xArray[inds_rho_l[jj]];
		thetaArray[kk*elOrd2+jj] = xArray[inds_theta_l[jj]];
		exnerArray[kk*elOrd2+jj] = xArray[inds_exner_l[jj]];
            }
            if(kk < nk-1) {
                elInds_velz_l(ex, ey, kk);
	        for(jj = 0; jj < elOrd2; jj++) {
                    velzArray[kk*elOrd2+jj] = xArray[inds_velz_l[jj]];
                }
	    }
        }
        VecRestoreArray(rho[ii],   &rhoArray);
        VecRestoreArray(exner[ii], &exnerArray);
        VecRestoreArray(theta[ii], &thetaArray);
        VecRestoreArray(velz[ii],  &velzArray);
    }
    VecRestoreArray(_x, &xArray);
}
