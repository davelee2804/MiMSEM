#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "L2Vecs.h"
#include "Schur.h"

using namespace std;

//#define CONTIGUOUS_ELEMENT_DOFS

Schur::Schur(Topo* _topo, Geom* _geom, bool _precon) {
    int elOrd2 = _topo->elOrd * _topo->elOrd;
    int lSize = _geom->nk * _topo->n2l;
    int gSize = _geom->nk * _topo->nDofs2G;
    int index;
    int *inds, *inds_g;
    Vec v_l, v_g;
    IS is_l, is_g;

    topo = _topo;
    geom = _geom;

    precon = _precon;

    elOrd = _topo->elOrd;
    nElsX = _topo->nElsX;
    inds2 = new int[elOrd2];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateSeq(MPI_COMM_SELF, lSize, &vl);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &x);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &b);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, lSize, lSize, gSize, gSize);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 11*elOrd2*elOrd2, PETSC_NULL, 11*elOrd2*elOrd2, PETSC_NULL);
    MatZeroEntries(M);

    if(precon) {
        MatCreate(MPI_COMM_WORLD, &P);
        MatSetSizes(P, lSize, lSize, gSize, gSize);
        MatSetType(P, MATMPIAIJ);
        MatMPIAIJSetPreallocation(P, 11*elOrd2*elOrd2, PETSC_NULL, 11*elOrd2*elOrd2, PETSC_NULL);
        MatZeroEntries(P);
    }

    KSPCreate(MPI_COMM_WORLD, &ksp);
    if(precon) {
        KSPSetOperators(ksp, M, P);
    } else {
        KSPSetOperators(ksp, M, M);
    }
    //KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    //KSPSetType(ksp, KSPGMRES);
    //KSPSetPCSide(ksp, PC_LEFT);
    //KSPGetPC(ksp, &pc);
    //PCSetType(pc, PCBJACOBI);
    //PCBJacobiSetTotalBlocks(pc, geom->nk, NULL);
/*
PCSetType(pc, PCFIELDSPLIT);
PCFieldSplitSetBlockSize(pc, geom->nk);
{
IS is;
int* local_inds = new int[topo->n2l];
char pc_name[] = "a";
for(int kk = 0; kk < geom->nk; kk++) {
  for(int ii = 0; ii < topo->n2l; ii++) {
    local_inds[ii] = rank*(geom->nk*topo->n2l) + kk*topo->n2l + ii;
  }
  ISCreateGeneral(MPI_COMM_WORLD, topo->n2l, local_inds, PETSC_COPY_VALUES, &is);
  PCFieldSplitSetIS(pc, pc_name, is);
  ISDestroy(&is);
  pc_name[0] += 1;
}
delete[] local_inds;
}
*/
    //KSPSetOptionsPrefix(ksp, "ksp_schur_");
    //KSPSetFromOptions(ksp);

    // create the scatter
    index = 0;
    inds_g = new int[lSize];
    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
#ifdef CONTIGUOUS_ELEMENT_DOFS
            inds = elInds2_g(ei%topo->nElsX, ei/topo->nElsX);
#else
            //inds = topo->elInds2_g(ei%topo->nElsX, ei/topo->nElsX);
            inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
#endif
            for(int ii = 0; ii < elOrd2; ii++) {
                //inds_g[index++] = kk*topo->nDofs2G + inds[ii];
                inds_g[index++] = rank * (geom->nk*topo->n2l) + kk*topo->n2l + inds[ii];
            }
        }
    }
    
    VecCreateSeq(MPI_COMM_SELF, lSize, &v_l);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &v_g);
    ISCreateStride(MPI_COMM_SELF, lSize, 0, 1, &is_l);
    ISCreateGeneral(MPI_COMM_WORLD, lSize, inds_g, PETSC_COPY_VALUES, &is_g);

    VecScatterCreate(v_g, is_g, v_l, is_l, &scat);

    delete[] inds_g;
    VecDestroy(&v_l);
    VecDestroy(&v_g);
    ISDestroy(&is_l);
    ISDestroy(&is_g);
}

void Schur::AddFromVertMat(int ei, Mat Az) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int nCols, row_g, cols_g[999];
#ifdef CONTIGUOUS_ELEMENT_DOFS
    int* inds = elInds2_g(ei%topo->nElsX, ei/topo->nElsX);
#else
    //int* inds = topo->elInds2_g(ei%topo->nElsX, ei/topo->nElsX);
    int* inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
#endif
    const int* cols;
    const double *vals;

    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ii = 0; ii < elOrd2; ii++) {
            //row_g = kk*topo->nDofs2G + inds[ii];
            row_g = rank * (geom->nk*topo->n2l) + kk*topo->n2l + inds[ii];

            MatGetRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
            for(int cc = 0; cc < nCols; cc++) {
                //cols_g[cc] = (cols[cc]/elOrd2)*topo->nDofs2G + inds[cols[cc]%elOrd2];
                cols_g[cc] = rank * (geom->nk*topo->n2l) + (cols[cc]/elOrd2)*topo->n2l + inds[cols[cc]%elOrd2];
            }
            MatSetValues(M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
            MatRestoreRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
        }
    }
}

void Schur::AddFromHorizMat(int kk, Mat Ax, Mat S) {
    int mi, mf, nCols, row_g, cols_g[999];
    const int* cols;
    const double *vals;
    int rank_i;

    MatGetOwnershipRange(Ax, &mi, &mf);

    for(int mm = mi; mm < mf; mm++) {
        MatGetRow(Ax, mm, &nCols, &cols, &vals);
        //row_g = kk*topo->nDofs2G + mm;
        //row_g = rank * (geom->nk*topo->n2l) + kk*topo->n2l + mm;
        rank_i = mm/topo->n2l;
        row_g = rank_i * (geom->nk*topo->n2l) + kk*topo->n2l + mm%topo->n2l;
        for(int cc = 0; cc < nCols; cc++) {
            //cols_g[cc] = kk*topo->nDofs2G + cols[cc];
            //cols_g[cc] = rank * (geom->nk*topo->n2l) + kk*topo->n2l + cols[cc];
            rank_i = cols[cc]/topo->n2l;
            cols_g[cc] = rank_i * (geom->nk*topo->n2l) + kk*topo->n2l + cols[cc]%topo->n2l;
        }
        //MatSetValues(M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
        MatSetValues(S, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
        MatRestoreRow(Ax, mm, &nCols, &cols, &vals);
    }
}

void Schur::RepackFromVert(Vec* vz, Vec v) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int ind_g;
    int* inds;
    PetscScalar *vzArray, *vArray;

    VecZeroEntries(vl);
    VecGetArray(vl, &vArray);
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
#ifdef CONTIGUOUS_ELEMENT_DOFS
        inds = elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
#else
        inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
#endif
        VecGetArray(vz[ei], &vzArray);
        for(int kk = 0; kk < geom->nk; kk++) {
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = kk*topo->n2l + inds[ii];
                vArray[ind_g] = vzArray[kk*elOrd2+ii];
            }
        }
        VecRestoreArray(vz[ei], &vzArray);
    }
    VecRestoreArray(vl, &vArray);

    // scatter to the global vector
    VecScatterBegin(scat, vl, v, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  scat, vl, v, ADD_VALUES, SCATTER_REVERSE);
}

// note: vx are assumed to be local vectors
void Schur::RepackFromHoriz(Vec* vx, Vec v) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int ind_g;
    int* inds;
//    int* inds_orig;
    PetscScalar *vxArray, *vArray;

    VecZeroEntries(vl);
    VecGetArray(vl, &vArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecGetArray(vx[kk], &vxArray);
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
#ifdef CONTIGUOUS_ELEMENT_DOFS
            inds_orig = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
            inds = elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
#else
            inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
#endif
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = kk*topo->n2l + inds[ii];
#ifdef CONTIGUOUS_ELEMENT_DOFS
                vArray[ind_g] = vxArray[inds_orig[ii]];
#else
                vArray[ind_g] = vxArray[inds[ii]];
#endif
            }
        }
        VecRestoreArray(vx[kk], &vxArray);
    }
    VecRestoreArray(vl, &vArray);

    // scatter to the global vector
    VecScatterBegin(scat, vl, v, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  scat, vl, v, ADD_VALUES, SCATTER_REVERSE);
}

// note: vx are assumed to be local vectors
void Schur::UnpackToHoriz(Vec v, Vec* vx) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int ind_g;
    int* inds;
//    int* inds_orig;
    PetscScalar *vxArray, *vArray;

    VecScatterBegin(scat, v, vl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  scat, v, vl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(vl, &vArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecGetArray(vx[kk], &vxArray);
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
#ifdef CONTIGUOUS_ELEMENT_DOFS
            inds_orig = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
            inds = elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
#else
            inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
#endif
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = kk*topo->n2l + inds[ii];
#ifdef CONTIGUOUS_ELEMENT_DOFS
                vxArray[inds_orig[ii]] = vArray[ind_g];
#else
                vxArray[inds[ii]] = vArray[ind_g];
#endif
            }
        }
        VecRestoreArray(vx[kk], &vxArray);
    }
    VecRestoreArray(vl, &vArray);
}

void Schur::Solve(L2Vecs* d_exner) {
    int nlocal, first_local;
    PC pc, subpc;
    KSP* subksp;

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);
    if(precon) {
        MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  P, MAT_FINAL_ASSEMBLY);
    }

    KSPGetPC(ksp, &pc);
    KSPSetType(ksp, KSPGMRES);
    KSPSetPCSide(ksp, PC_LEFT);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetLocalBlocks(pc, geom->nk, NULL);
    KSPSetUp(ksp);
    PCBJacobiGetSubKSP(pc, &nlocal, &first_local, &subksp);

    for(int ii = 0; ii < nlocal; ii++) {
        KSPGetPC(subksp[ii], &subpc);
        PCSetType(subpc, PCILU);
        KSPSetType(subksp[ii], KSPGMRES);
        KSPSetTolerances(subksp[ii], 1.e-16, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    }

    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetOptionsPrefix(ksp, "ksp_schur_");
    KSPSetFromOptions(ksp);

    KSPSolve(ksp, b, x);
    UnpackToHoriz(x, d_exner->vl);
}

Schur::~Schur() {
    delete[] inds2;
    VecDestroy(&vl);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&M);
    if(precon) MatDestroy(&P);
    KSPDestroy(&ksp);
    VecScatterDestroy(&scat);
}

int* Schur::elInds2_l(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd; ix++) {
            inds2[kk] = (ey*nElsX + ex)*elOrd*elOrd + iy*elOrd+ix;
            kk++;
        }
    }

    return inds2;
}

int* Schur::elInds2_g(int ex, int ey) {
    int ix, iy, kk;

    inds2 = elInds2_l(ex, ey);

    kk = 0;
    for(iy = 0; iy < elOrd; iy++) {
        for(ix = 0; ix < elOrd; ix++) {
            inds2[kk] += topo->loc2[0];
            kk++;
        }
    }

    return inds2;
}
