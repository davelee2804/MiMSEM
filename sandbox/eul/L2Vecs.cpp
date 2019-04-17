#include <iostream>
#include <cmath>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "L2Vecs.h"

using namespace std;

L2Vecs::L2Vecs(int _nk, Topo* _topo, Geom* _geom) {
    int ii, n2;

    nk = _nk;
    topo = _topo;
    geom = _geom;

    n2 = topo->nElsX*topo->nElsX;

    vh = new Vec[nk];
    vz = new Vec[n2];
    vl = new Vec[nk];

    for(ii = 0; ii < nk; ii++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &vh[ii]);
        VecCreateSeq(MPI_COMM_SELF, topo->n2, &vl[ii]);
        VecZeroEntries(vh[ii]);
        VecZeroEntries(vl[ii]);
    }

    for(ii = 0; ii < n2; ii++) {
        VecCreateSeq(MPI_COMM_SELF, nk*topo->elOrd*topo->elOrd, &vz[ii]);
        VecZeroEntries(vz[ii]);
    }
}

L2Vecs::~L2Vecs() {
    int ii, n2;

    n2 = topo->nElsX*topo->nElsX;

    for(ii = 0; ii < nk; ii++) {
        VecDestroy(&vh[ii]);
        VecDestroy(&vl[ii]);
    }
    delete[] vh;
    delete[] vl;

    for(ii = 0; ii < n2; ii++) {
        VecDestroy(&vz[ii]);
    }
    delete[] vz;
}

void L2Vecs::VertToHoriz() {
    int ii, kk, ex, ey, ei, n2, *inds2;
    PetscScalar *zArray, *hArray;

    n2 = topo->elOrd*topo->elOrd;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds2 = topo->elInds2_l(ex, ey);

            VecGetArray(vz[ei], &zArray);
            for(kk = 0; kk < nk; kk++) {
                VecGetArray(vl[kk], &hArray);
                for(ii = 0; ii < n2; ii++) {
                    hArray[inds2[ii]] = zArray[kk*n2+ii];
                }
                VecRestoreArray(vl[kk], &hArray);
            }
            VecRestoreArray(vz[ei], &zArray);
        }
    }
}

void L2Vecs::HorizToVert() {
    int ii, kk, ex, ey, ei, n2, *inds2;
    PetscScalar *zArray, *hArray;

    n2 = topo->elOrd*topo->elOrd;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            ei = ey*topo->nElsX + ex;
            inds2 = topo->elInds2_l(ex, ey);

            VecGetArray(vz[ei], &zArray);
            for(kk = 0; kk < nk; kk++) {
                VecGetArray(vl[kk], &hArray);
                for(ii = 0; ii < n2; ii++) {
                    zArray[kk*n2+ii] = hArray[inds2[ii]];
                }
                VecRestoreArray(vl[kk], &hArray);
            }
            VecRestoreArray(vz[ei], &zArray);
        }
    }
}

void L2Vecs::UpdateLocal() {
    int ii;

    for(ii = 0; ii < nk; ii++) {
        VecScatterBegin(topo->gtol_2, vh[ii], vl[ii], INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(topo->gtol_2,   vh[ii], vl[ii], INSERT_VALUES, SCATTER_FORWARD);
    }
}

void L2Vecs::UpdateGlobal() {
    int ii;

    for(ii = 0; ii < nk; ii++) {
        VecScatterBegin(topo->gtol_2, vl[ii], vh[ii], INSERT_VALUES, SCATTER_REVERSE);
        VecScatterEnd(topo->gtol_2,   vl[ii], vh[ii], INSERT_VALUES, SCATTER_REVERSE);
    }
}

void L2Vecs::CopyFromVert(Vec* vf) {
    int ii, n2;

    n2 = topo->nElsX*topo->nElsX;
    for(ii = 0; ii < n2; ii++) {
        VecCopy(vf[ii], vz[ii]);
    }
}

void L2Vecs::CopyFromHoriz(Vec* vf) {
    int ii;

    for(ii = 0; ii < nk; ii++) {
        VecCopy(vf[ii], vh[ii]);
    }
}

void L2Vecs::CopyToVert(Vec* vf) {
    int ii, n2;

    n2 = topo->nElsX*topo->nElsX;
    for(ii = 0; ii < n2; ii++) {
        VecCopy(vz[ii], vf[ii]);
    }
}

void L2Vecs::CopyToHoriz(Vec* vf) {
    int ii;

    for(ii = 0; ii < nk; ii++) {
        VecCopy(vh[ii], vf[ii]);
    }
}
