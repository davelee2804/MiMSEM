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
#include "ElMats.h"
#include "VertOps.h"
#include "Assembly.h"
#include "Schur.h"

using namespace std;

Schur::Schur(Topo* _topo, Geom* _geom) {
    int lSize = _geom->nk * _topo->n2l;
    int gSize = _geom->nk * _topo->nDofs2G;
    int index;
    int *inds_g;
    Vec v_l, v_g;
    IS is_l, is_g;

    topo = _topo;
    geom = _geom;

    elOrd = _topo->elOrd;
    nElsX = _topo->nElsX;
    elOrd2 = elOrd*elOrd;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateSeq(MPI_COMM_SELF, lSize, &vl);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &x);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &b);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &t);

    // create the scatter
    index = 0;
    inds_g = new int[lSize];
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        for(int kk = 0; kk < geom->nk; kk++) {
            for(int ii = 0; ii < elOrd2; ii++) {
                inds_g[index++] = rank*(geom->nk*topo->n2l) + ei*geom->nk*elOrd2 + kk*elOrd2 + ii;
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

    T = P = NULL;
}

void Schur::InitialiseMatrix() {
    int lSize = geom->nk * topo->n2l;
    int gSize = geom->nk * topo->nDofs2G;

    MatCreate(MPI_COMM_WORLD, &L_rho);
    MatSetSizes(L_rho, lSize, lSize, gSize, gSize);
    MatSetType(L_rho, MATMPIAIJ);
    MatMPIAIJSetPreallocation(L_rho, 11*elOrd2*elOrd2, PETSC_NULL, 11*elOrd2*elOrd2, PETSC_NULL);
    MatZeroEntries(L_rho);

    MatCreate(MPI_COMM_WORLD, &L_rt);
    MatSetSizes(L_rt, lSize, lSize, gSize, gSize);
    MatSetType(L_rt, MATMPIAIJ);
    MatMPIAIJSetPreallocation(L_rt, 11*elOrd2*elOrd2, PETSC_NULL, 11*elOrd2*elOrd2, PETSC_NULL);
    MatZeroEntries(L_rt);

    MatCreate(MPI_COMM_WORLD, &Q);
    MatSetSizes(Q, lSize, lSize, gSize, gSize);
    MatSetType(Q, MATMPIAIJ);
    MatMPIAIJSetPreallocation(Q, 11*elOrd2*elOrd2, PETSC_NULL, 11*elOrd2*elOrd2, PETSC_NULL);
    MatZeroEntries(Q);

    MatCreate(MPI_COMM_WORLD, &VISC);
    MatSetSizes(VISC, lSize, lSize, gSize, gSize);
    MatSetType(VISC, MATMPIAIJ);
    MatMPIAIJSetPreallocation(VISC, 11*elOrd2*elOrd2, PETSC_NULL, 11*elOrd2*elOrd2, PETSC_NULL);
    MatZeroEntries(VISC);

    KSPCreate(MPI_COMM_WORLD, &ksp_rt);
    KSPCreate(MPI_COMM_WORLD, &ksp_rho);
}

void Schur::DestroyMatrix() {
    MatDestroy(&L_rho);
    MatDestroy(&L_rt);
    MatDestroy(&Q);

    KSPDestroy(&ksp_rt);
    KSPDestroy(&ksp_rho);
}

void Schur::AddFromVertMat(int ei, Mat Az, Mat _M) {
    int nCols, row_g, cols_g[999];
    int lShift = rank*(geom->nk*topo->n2l) + ei*geom->nk*elOrd2;
    const int* cols;
    const double *vals;

    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ii = 0; ii < elOrd2; ii++) {
            row_g = lShift + kk*elOrd2 + ii;

            MatGetRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
            for(int cc = 0; cc < nCols; cc++) {
                cols_g[cc] = lShift + cols[cc];
            }
            MatSetValues(_M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
            MatRestoreRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
        }
    }
}

void Schur::AddFromHorizMat(int kk, Mat Ax, Mat _M) {
    int mi, mf, nCols, row_g, cols_g[999], rank_i;
    const int* cols;
    const double *vals;

    MatGetOwnershipRange(Ax, &mi, &mf);

    for(int mm = mi; mm < mf; mm++) {
        MatGetRow(Ax, mm, &nCols, &cols, &vals);

        rank_i = mm/topo->n2l;
        row_g = rank_i*geom->nk*topo->n2l + ((mm-mi)/elOrd2)*geom->nk*elOrd2 + kk*elOrd2 + (mm-mi)%elOrd2;
        if(mm<rank*topo->n2l || mm>=(rank+1)*topo->n2l) cout << "SCHUR: matrix assebly error [1]! " << rank << ", " << mm << endl;
	if(rank != rank_i                             ) cout << "SCHUR: matrix assebly error [2]! " << rank << ", " << mm << endl;

        for(int cc = 0; cc < nCols; cc++) {
            rank_i = cols[cc]/topo->n2l;
            cols_g[cc] = /*rank_i*geom->nk*topo->n2l +*/ (cols[cc]/elOrd2)*geom->nk*elOrd2 + kk*elOrd2 + cols[cc]%elOrd2;
        }
        MatSetValues(_M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
        MatRestoreRow(Ax, mm, &nCols, &cols, &vals);
    }
}

void Schur::RepackFromVert(Vec* vz, Vec v) {
    int ind_g;
    PetscScalar *vzArray, *vArray;

    VecZeroEntries(vl);
    VecGetArray(vl, &vArray);
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        VecGetArray(vz[ei], &vzArray);
        
        for(int kk = 0; kk < geom->nk; kk++) {
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = ei*geom->nk*elOrd2 + kk*elOrd2 + ii;
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
    int ind_g;
    int* inds;
    PetscScalar *vxArray, *vArray;

    VecZeroEntries(v);

    VecZeroEntries(vl);
    VecGetArray(vl, &vArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecGetArray(vx[kk], &vxArray);
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = ei*geom->nk*elOrd2 + kk*elOrd2 + ii;
                vArray[ind_g] = vxArray[inds[ii]];
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
    int ind_g;
    int* inds;
    PetscScalar *vxArray, *vArray;

    VecScatterBegin(scat, v, vl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  scat, v, vl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(vl, &vArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecGetArray(vx[kk], &vxArray);
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = ei*geom->nk*elOrd2 + kk*elOrd2 + ii;
                vxArray[inds[ii]] = vArray[ind_g];
            }
        }
        VecRestoreArray(vx[kk], &vxArray);
    }
    VecRestoreArray(vl, &vArray);
}

void Schur::Solve(KSP ksp, Mat _M, L2Vecs* d_exner) {
    int nlocal, first_local, size;
    PC pc, subpc;
    KSP* subksp;

    MatAssemblyBegin(_M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  _M, MAT_FINAL_ASSEMBLY);

    KSPSetOperators(ksp, _M, _M);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    KSPGetPC(ksp, &pc);
    KSPSetType(ksp, KSPGMRES);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetUp(ksp);
    PCBJacobiGetSubKSP(pc, &nlocal, &first_local, &subksp);

    for(int ii = 0; ii < nlocal; ii++) {
        KSPGetPC(subksp[ii], &subpc);
        PCSetType(subpc, PCLU);
    }
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetOptionsPrefix(ksp, "ksp_schur_");
    KSPSetFromOptions(ksp);

    KSPSolve(ksp, b, x);
    UnpackToHoriz(x, d_exner->vl);
}

/*
bool AddEntry(int n, int* array, int* buffer, int entry) {
    int start, i;

    for(i = 0; i < n; i++) {
        if(array[i] == entry) return false;
    }

    for(start = 0; start < n; start++) {
        if(array[start] < entry) buffer[start] = array[start];
        else break;
    }
    buffer[start++] = entry;

    for(i = start; i < n+1; i++) buffer[i] = array[i-1];
    for(i = 0; i < n+1; i++) array[i] = buffer[i];

    return true;
}

int FindIndex(int n, int* array, int entry) {
    for(int i = 0; i < n; i++) if(array[i] == entry) return i;
    return -1;
}

void Schur::Preallocate(HorizSolve* hs, VertSolve* vs, L1Vecs* velx, L2Vecs* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* pi, 
                        L2Vecs* theta, L1Vecs* F_u, L2Vecs* F_w, L2Vecs* F_rho, L2Vecs* F_rt, L2Vecs* F_pi, L1Vecs* gradPi) {
    int ProcMap[9], nProcs = 0;
    int** ProcRow;
    int*** ProcRowCol;
    int buffer[9999];
    int mi, mf, nCols, row_l, cols_g[999], rank_i, rank_j, lShift, ex, ey;
    const int* cols;
    const double *vals;
    int lSize = geom->nk * topo->n2l;

    for(int proc_i = 0; proc_i < 9; proc_i++) ProcMap[proc_i] = -1;

    ProcRow = new int*[9];
    ProcRowCol = new int**[9];
    for(int proc_i = 0; proc_i < 9; proc_i++) {
        ProcRow[proc_i] = new int[lSize];
        ProcRowCol[proc_i] = new int*[lSize];
        for(int row_i = 0; row_i < lSize; row_i++) {
            ProcRow[proc_i][row_i] = 0;
            ProcRowCol[proc_i][row_i] = new int[3*9*geom->nk*elOrd2];
        }
    }

    for(int lev = 0; lev < geom->nk; lev++) {
        if(!rank) cout << "doing preallocation from horiztonal matrix at level: " << lev << endl;
        hs->assemble_and_update(lev, theta->vl, velx->vl[lev], velx->vh[lev], rho->vl[lev], rt->vl[lev], pi->vl[lev],
                                F_u->vh[lev], F_rho->vh[lev], F_rt->vh[lev], F_pi->vh[lev], gradPi->vl[lev], &hs->_PCx);

        MatAssemblyBegin(hs->_PCx, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(  hs->_PCx, MAT_FINAL_ASSEMBLY);

        MatGetOwnershipRange(hs->_PCx, &mi, &mf);

        // first pass to build the processor to index map
        if(!lev) {
            for(int mm = mi; mm < mf; mm++) {
                MatGetRow(hs->_PCx, mm, &nCols, &cols, &vals);
                for(int cc = 0; cc < nCols; cc++) {
                    rank_i = cols[cc]/topo->n2l;
                    if(AddEntry(nProcs, ProcMap, buffer, rank_i)) {
                        nProcs++;
                    }
                }
                MatRestoreRow(hs->_PCx, mm, &nCols, &cols, &vals);
            }
        }

        for(int mm = mi; mm < mf; mm++) {
            MatGetRow(hs->_PCx, mm, &nCols, &cols, &vals);
            for(int cc = 0; cc < nCols; cc++) {
                rank_i = cols[cc]/topo->n2l;
                rank_j = FindIndex(nProcs, ProcMap, rank_i);

                cols_g[cc] = (cols[cc]/elOrd2)*geom->nk*elOrd2 + lev*elOrd2 + cols[cc]%elOrd2;

                if(AddEntry(ProcRow[rank_j][mm-mi], ProcRowCol[rank_j][mm-mi], buffer, cols_g[cc])) {
                    ProcRow[rank_j][mm-mi]++;
                }
            }
            MatRestoreRow(hs->_PCx, mm, &nCols, &cols, &vals);
        }

        MatDestroy(&hs->_PCx);
    }

    rank_j = FindIndex(nProcs, ProcMap, rank);
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        if(!rank) cout << "doing preallocation from vertical matrix at column: " << ei << endl;
        ex = ei%topo->nElsX;
        ey = ei/topo->nElsX;
        vs->assemble_and_update(ex, ey, theta->vz[ei], velz->vz[ei], rho->vz[ei], rt->vz[ei], pi->vz[ei],
                                F_w->vz[ei], F_rho->vz[ei], F_rt->vz[ei], F_pi->vz[ei]);

        lShift = rank*(geom->nk*topo->n2l) + ei*geom->nk*elOrd2;
        for(int kk = 0; kk < geom->nk; kk++) {
            for(int ii = 0; ii < elOrd2; ii++) {
                row_l = ei*geom->nk*elOrd2 + kk*elOrd2 + ii;

                MatGetRow(vs->_PCz, kk*elOrd2+ii, &nCols, &cols, &vals);
                for(int cc = 0; cc < nCols; cc++) {
                    cols_g[cc] = lShift + cols[cc];
                    if(AddEntry(ProcRow[rank_j][row_l], ProcRowCol[rank_j][row_l], buffer, cols_g[cc])) ProcRow[rank_j][row_l]++;
                }
                MatRestoreRow(vs->_PCz, kk*elOrd2+ii, &nCols, &cols, &vals);
            }
        }
    }

    //MatPreallocateLocations(Mat A,PetscInt row,PetscInt ncols,PetscInt *cols,PetscInt *dnz,PetscInt *onz)

    for(int proc_i = 0; proc_i < 9; proc_i++) {
        delete[] ProcRow[proc_i];
        for(int row_i = 0; row_i < lSize; row_i++) {
            delete[] ProcRowCol[proc_i][row_i];
        }
        delete[] ProcRowCol[proc_i];
    }
    delete[] ProcRow;
    delete[] ProcRowCol;
}
*/

Schur::~Schur() {
    VecDestroy(&vl);
    VecDestroy(&x);
    VecDestroy(&b);
    VecDestroy(&t);
    VecScatterDestroy(&scat);
}

// Row is in L2 and column is in H(div)
void Schur::AddFromVertMat_D(int ei, Mat Az, Mat _M) {
    int nCols, row_g, cols_g[999];
    int lShift_r = rank*(geom->nk*topo->n2l) + ei*geom->nk*elOrd2;
    int lShift_c = rank*(geom->nk*topo->n1l + (geom->nk-1)*topo->n2l);
    const int* cols;
    const double *vals;

    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ii = 0; ii < elOrd2; ii++) {
            row_g = lShift_r + kk*elOrd2 + ii;

            MatGetRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
            for(int cc = 0; cc < nCols; cc++) {
                cols_g[cc] = lShift_c + (cols[cc]/elOrd2)*(topo->n1l + topo->n2l) + topo->n1l + ei*elOrd2 + (cols[cc]%elOrd2);
            }
            MatSetValues(_M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
            MatRestoreRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
        }
    }
}

void Schur::AddFromHorizMat_D(int kk, Mat Ax, Mat _M) {
    int mi, mf, nCols, row_g, cols_g[999], rank_i;
    const int* cols;
    const double *vals;

    MatGetOwnershipRange(Ax, &mi, &mf);

    for(int mm = mi; mm < mf; mm++) {
        row_g = rank*geom->nk*topo->n2l + ((mm-mi)/elOrd2)*geom->nk*elOrd2 + kk*elOrd2 + (mm-mi)%elOrd2;

        MatGetRow(Ax, mm, &nCols, &cols, &vals);
        for(int cc = 0; cc < nCols; cc++) {
            rank_i = cols[cc]/topo->n1l;
            cols_g[cc] = rank_i*(geom->nk*topo->n1l + (geom->nk-1)*topo->n2l) + kk*(topo->n1l + topo->n2l) + cols[cc]%topo->n1l;
        }
        MatSetValues(_M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
        MatRestoreRow(Ax, mm, &nCols, &cols, &vals);
    }
}

// Row is in H(div) and column is in L2
void Schur::AddFromVertMat_G(int ei, Mat Az, Mat _M) {
    int nCols, row_g, cols_g[999];
    int lShift_r = rank*(geom->nk*topo->n1l + (geom->nk-1)*topo->n2l);
    int lShift_c = rank*(geom->nk*topo->n2l) + ei*geom->nk*elOrd2;
    const int* cols;
    const double *vals;

    for(int kk = 0; kk < geom->nk-1; kk++) {
        for(int ii = 0; ii < elOrd2; ii++) {
            row_g = lShift_r + kk*(topo->n1l + topo->n2l) + topo->n1l + ei*elOrd2 + ii;

            MatGetRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
            for(int cc = 0; cc < nCols; cc++) {
                cols_g[cc] = lShift_c + cols[cc];
            }
            MatSetValues(_M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
            MatRestoreRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
        }
    }
}

void Schur::AddFromHorizMat_G(int kk, Mat Ax, Mat _M) {
    int mi, mf, nCols, row_g, cols_g[999];
    const int* cols;
    const double *vals;

    MatGetOwnershipRange(Ax, &mi, &mf);

    for(int mm = mi; mm < mf; mm++) {
        row_g = rank*(geom->nk*topo->n1l + (geom->nk-1)*topo->n2l) + kk*(topo->n1l + topo->n2l) + (mm-mi);

        MatGetRow(Ax, mm, &nCols, &cols, &vals);
        for(int cc = 0; cc < nCols; cc++) {
            cols_g[cc] = (cols[cc]/elOrd2)*geom->nk*elOrd2 + kk*elOrd2 + cols[cc]%elOrd2;
        }
        MatSetValues(_M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
        MatRestoreRow(Ax, mm, &nCols, &cols, &vals);
    }
}

// Row is in H(div) and column is in H(div)
void Schur::AddFromVertMat_U(int ei, Mat Az, Mat _M) {
    int nCols, row_g, cols_g[999];
    int lShift = rank*(geom->nk*topo->n1l + (geom->nk-1)*topo->n2l);
    const int* cols;
    const double *vals;

    for(int kk = 0; kk < geom->nk-1; kk++) {
        for(int ii = 0; ii < elOrd2; ii++) {
            row_g = lShift + kk*(topo->n1l + topo->n2l) + topo->n1l + ei*elOrd2 + ii;

            MatGetRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
            for(int cc = 0; cc < nCols; cc++) {
                cols_g[cc] = lShift + (cols[cc]/elOrd2)*(topo->n1l + topo->n2l) + topo->n1l + ei*elOrd2 + (cols[cc]%elOrd2);
            }
            MatSetValues(_M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
            MatRestoreRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
        }
    }
}

void Schur::AddFromHorizMat_U(int kk, Mat Ax, Mat _M) {
    int mi, mf, nCols, row_g, cols_g[999], rank_i;
    const int* cols;
    const double *vals;

    MatGetOwnershipRange(Ax, &mi, &mf);

    for(int mm = mi; mm < mf; mm++) {
        row_g = rank*(geom->nk*topo->n1l + (geom->nk-1)*topo->n2l) + kk*(topo->n1l + topo->n2l) + (mm-mi);

        MatGetRow(Ax, mm, &nCols, &cols, &vals);
        for(int cc = 0; cc < nCols; cc++) {
            rank_i = cols[cc]/topo->n1l;
            cols_g[cc] = rank_i*(geom->nk*topo->n1l + (geom->nk-1)*topo->n2l) + kk*(topo->n1l + topo->n2l) + cols[cc]%topo->n1l;
        }
        MatSetValues(_M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
        MatRestoreRow(Ax, mm, &nCols, &cols, &vals);
    }
}

void Schur::RepackFromVert_U(Vec* uz, Vec _u) {
    int ind_g;
    int lShift = rank*(geom->nk*topo->n1l + (geom->nk-1)*topo->n2l);
    PetscScalar *uzArray, *uArray;

    VecGetArray(_u, &uArray);
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        VecGetArray(uz[ei], &uzArray);
        
        for(int kk = 0; kk < geom->nk-1; kk++) {
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = lShift + kk*(topo->n1l + topo->n2l) + topo->n1l + ei*elOrd2 + ii;
                uArray[ind_g] = uzArray[kk*elOrd2+ii];
            }
        }
        VecRestoreArray(uz[ei], &uzArray);
    }
    VecRestoreArray(_u, &uArray);
}

// note: ux are assumed to be local vectors
void Schur::RepackFromHoriz_U(Vec* ux, Vec _u) {
    int ind_g;
    int lShift = rank*(geom->nk*topo->n1l + (geom->nk-1)*topo->n2l);
    PetscScalar *uxArray, *uArray;

    VecGetArray(_u, &uArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecGetArray(ux[kk], &uxArray);
        for(int ii = 0; ii < topo->n1l; ii++) {
            ind_g = lShift + kk*(topo->n1l + topo->n2l) + ii;
            uArray[ind_g] = uxArray[ii];
        }
        VecRestoreArray(ux[kk], &uxArray);
    }
    VecRestoreArray(_u, &uArray);
}
