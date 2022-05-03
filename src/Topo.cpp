#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>

#include "Topo.h"

#define COARSE 1

using namespace std;
using std::string;

Topo::Topo() {
    Vec vl, vg;

    int ii, n_procs;
    int nLoc[4];
    ifstream file;
    char filename[100];
    string line;

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
#ifdef COARSE
    coarseInds();
    skelIntlInds();
#endif
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
#ifdef COARSE
    delete[] dd_skel_locl_x;
    delete[] dd_skel_locl_y;
    delete[] dd_intl_locl_x;
    delete[] dd_intl_locl_y;
    delete[] dd_skel_global;
    ISDestroy(&is_skel_l);
    ISDestroy(&is_skel_g);
    VecScatterDestroy(&gtol_skel);
    delete[] dd_skel_locl_glob_map;
#endif
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

#define COARSE_ORD 1
#define TWOX_COARSE_ORD (2*COARSE_ORD)
#define COARSE_ORD_P1 (COARSE_ORD+1)
#define TWOX_COARSE_ORD_P1 (2*COARSE_ORD_P1)

void Topo::coarseInds() {
    int ii, n_procs;
    ifstream file;
    char filename[100];
    string line;
    Vec vl, vg;

    coarse_inds_x = new int[COARSE_ORD_P1];
    coarse_inds_y = new int[COARSE_ORD_P1];
    coarse_inds = new int[TWOX_COARSE_ORD_P1];

    sprintf(filename, "input/edges_x_coarse_%.4u.txt", pi);
    file.open(filename);
    ii = 0;
    while (std::getline(file, line)) {
        coarse_inds_x[ii] = atoi(line.c_str());
        ii++;
    }
    file.close();

    sprintf(filename, "input/edges_y_coarse_%.4u.txt", pi);
    file.open(filename);
    ii = 0;
    while (std::getline(file, line)) {
        coarse_inds_y[ii] = atoi(line.c_str());
        ii++;
    }
    file.close();

    for(ii = 0; ii < COARSE_ORD_P1; ii++) {
        coarse_inds[2*ii+0] = coarse_inds_x[ii];
        coarse_inds[2*ii+1] = coarse_inds_y[ii];
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &n_procs);

    ISCreateStride(MPI_COMM_SELF, TWOX_COARSE_ORD_P1, 0, 1, &is_l_coarse);
    ISCreateGeneral(MPI_COMM_WORLD, TWOX_COARSE_ORD_P1, coarse_inds, PETSC_COPY_VALUES, &is_g_coarse);

    VecCreateSeq(MPI_COMM_SELF, TWOX_COARSE_ORD_P1, &vl);
    VecCreateMPI(MPI_COMM_WORLD, TWOX_COARSE_ORD_P1, TWOX_COARSE_ORD*n_procs, &vg);
    VecScatterCreate(vg, is_g_coarse, vl, is_l_coarse, &gtol_coarse);
    VecDestroy(&vl);
    VecDestroy(&vg);
}

void Topo::skelIntlInds() {
    int ii;
    int *inds_g, skel_l, indx_g;
    char filename[100];
    Vec vl, vg;

    dd_skel_locl_x = new int[n1];
    dd_skel_locl_y = new int[n1];
    dd_intl_locl_x = new int[n1];
    dd_intl_locl_y = new int[n1];
    dd_skel_global = new int[n1];

    sprintf(filename, "input/skeleton_inds_x_%.4u.txt", pi);
    loadObjs(filename, dd_skel_locl_x);
    sprintf(filename, "input/skeleton_inds_y_%.4u.txt", pi);
    loadObjs(filename, dd_skel_locl_y);
    sprintf(filename, "input/internal_inds_x_%.4u.txt", pi);
    loadObjs(filename, dd_intl_locl_x);
    sprintf(filename, "input/internal_inds_y_%.4u.txt", pi);
    loadObjs(filename, dd_intl_locl_y);
    sprintf(filename, "input/global_to_skeleton.txt");
    loadObjs(filename, dd_skel_global);

    dd_n_skel_locl = 0;
    dd_n_intl_locl = 0;
    dd_n_skel_glob = 0;
    for(ii = 0; ii < n1; ii++) {
        if(dd_skel_locl_x[ii] > dd_n_skel_locl) dd_n_skel_locl = dd_skel_locl_x[ii];
        if(dd_skel_locl_y[ii] > dd_n_skel_locl) dd_n_skel_locl = dd_skel_locl_y[ii];
        if(dd_intl_locl_x[ii] > dd_n_intl_locl) dd_n_intl_locl = dd_intl_locl_x[ii];
        if(dd_intl_locl_y[ii] > dd_n_intl_locl) dd_n_intl_locl = dd_intl_locl_y[ii];
    }
    for(ii = 0; ii < nDofs1G; ii++) if(dd_skel_global[ii] > dd_n_skel_glob) dd_n_skel_glob = dd_skel_global[ii];
    // sanity check
    if(dd_n_skel_locl%4 != 0) {
        cerr << pi << ":\tERROR! no. local skeleton dofs: " << dd_n_skel_locl << " not evenly divisible by 4!\n";
	abort();
    }

    inds_intl_x_l = new int[(elOrd)*(elOrd+1)];
    inds_intl_y_l = new int[(elOrd+1)*(elOrd)];
    inds_skel_x_l = new int[(elOrd)*(elOrd+1)];
    inds_skel_y_l = new int[(elOrd+1)*(elOrd)];
    inds_skel_x_g = new int[(elOrd)*(elOrd+1)];
    inds_skel_y_g = new int[(elOrd+1)*(elOrd)];

    // create the global to local vec scatter for the skeleton dofs
    inds_g = new int[dd_n_skel_locl];
    for(ii = 0; ii < n1x; ii++) {
        skel_l = dd_skel_locl_x[ii];
        if(skel_l > -1) {
            indx_g = loc1x[ii];
	    inds_g[skel_l] = dd_skel_global[indx_g];
	}
    }
    for(ii = 0; ii < n1y; ii++) {
        skel_l = dd_skel_locl_y[ii];
        if(skel_l > -1) {
            indx_g = loc1y[ii];
	    inds_g[skel_l] = dd_skel_global[indx_g];
	}
    }

    ISCreateStride(MPI_COMM_SELF, dd_n_skel_locl, 0, 1, &is_skel_l);
    ISCreateGeneral(MPI_COMM_WORLD, dd_n_skel_glob, inds_g, PETSC_COPY_VALUES, &is_skel_g);

    VecCreateSeq(MPI_COMM_SELF, dd_n_skel_locl, &vl);
    VecCreateMPI(MPI_COMM_WORLD, dd_n_skel_locl/2, dd_n_skel_glob, &vg);
    VecScatterCreate(vg, is_skel_g, vl, is_skel_l, &gtol_skel);
    VecDestroy(&vl);
    VecDestroy(&vg);

    // TODO: test to see if this is the same as 'inds_g'
    dd_skel_locl_glob_map = new int[n1];
    sprintf(filename, "input/skeleton_local_to_global_map_%.4u.txt", pi);
    loadObjs(filename, dd_skel_locl_glob_map);

    delete[] inds_g;
}

int* Topo::elInds_intl_x_l(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds_intl_x_l[kk] = dd_intl_locl_x[2*((ey*elOrd + iy)*(nDofsX + 1) + ex*elOrd + ix) + 0];
            kk++;
        }
    }

    return inds_intl_x_l;
}

int* Topo::elInds_intl_y_l(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds_intl_y_l[kk] = dd_intl_locl_y[2*((ey*elOrd + iy)*(nDofsX) + ex*elOrd + ix) + 1];
            kk++;
        }
    }

    return inds_intl_y_l;
}

int* Topo::elInds_skel_x_l(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds_skel_x_l[kk] = dd_skel_locl_x[2*((ey*elOrd + iy)*(nDofsX + 1) + ex*elOrd + ix) + 0];
            kk++;
        }
    }

    return inds_skel_x_l;
}

int* Topo::elInds_skel_y_l(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            inds_skel_y_l[kk] = dd_skel_locl_y[2*((ey*elOrd + iy)*(nDofsX) + ex*elOrd + ix) + 1];
            kk++;
        }
    }

    return inds_skel_y_l;
}

int* Topo::elInds_skel_x_g(int ex, int ey) {
    int ix, iy, kk, ind_g;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            ind_g = loc1x[(ey*elOrd + iy)*(nDofsX + 1) + ex*elOrd + ix];
            inds_skel_x_g[kk] = dd_skel_global[ind_g];
            kk++;
        }
    }

    return inds_skel_x_g;
}

int* Topo::elInds_skel_y_g(int ex, int ey) {
    int ix, iy, kk, ind_g;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd + 1; ix++) {
            ind_g = loc1y[(ey*elOrd + iy)*(nDofsX) + ex*elOrd + ix];
            inds_skel_y_g[kk] = dd_skel_global[ind_g];
            kk++;
        }
    }

    return inds_skel_y_g;
}
