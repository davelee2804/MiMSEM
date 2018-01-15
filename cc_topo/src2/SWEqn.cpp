#include <iostream>

#include <mpi.h>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscksp.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
//#define RAD_SPHERE 1.0

using namespace std;
int step = 0;

SWEqn::SWEqn(Topo* _topo, Geom* _geom) {
    topo = _topo;
    geom = _geom;

    grav = 9.80616*(RAD_SPHERE/RAD_EARTH);
    omega = 7.292e-5;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // 0 form lumped mass matrix (vector)
    m0 = new Pvec(topo, geom, node);

    // 1 form mass matrix
    M1 = new Umat(topo, geom, node, edge);

    // 2 form mass matrix
    M2 = new Wmat(topo, geom, edge);

    // incidence matrices
    NtoE = new E10mat(topo);
    EtoF = new E21mat(topo);

    // adjoint differential operators (curl and grad)
    MatMatMult(NtoE->E01, M1->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E01M1);
    MatMatMult(EtoF->E12, M2->M, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &E12M2);

    // rotational operator
    R = new RotMat(topo, geom, node, edge);

    // mass flux operator
    F = new Uhmat(topo, geom, node, edge);

    // kinetic energy operator
    K = new WtQUmat(topo, geom, node, edge);

    // coriolis vector (projected onto 0 forms)
    coriolis();
}

// project coriolis term onto 0 forms
// assumes diagonal 0 form mass matrix
void SWEqn::coriolis() {
    int ii;
    PtQmat* PtQ = new PtQmat(topo, geom, node);
    PetscScalar *fArray;
    Vec fl, fxl, fxg, PtQfxg;

    // initialise the coriolis vector (local and global)
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fg);

    // evaluate the coriolis term at nodes
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &fxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &fxg);
    VecZeroEntries(fxg);
    VecGetArray(fxl, &fArray);
    for(ii = 0; ii < topo->n0; ii++) {
        //fArray[ii] = 2.0*omega;
        fArray[ii] = 2.0*omega*sin(geom->s[ii][1]);
    }
    VecRestoreArray(fxl, &fArray);

    // scatter array to global vector
    VecScatterBegin(topo->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, fxl, fxg, INSERT_VALUES, SCATTER_REVERSE);

    // project vector onto 0 forms
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PtQfxg);
    VecZeroEntries(PtQfxg);
    MatMult(PtQ->M, fxg, PtQfxg);
    // diagonal mass matrix as vector
    VecPointwiseDivide(fg, PtQfxg, m0->vg);
    
    delete PtQ;
    VecDestroy(&fl);
    VecDestroy(&fxl);
    VecDestroy(&fxg);
    VecDestroy(&PtQfxg);
}

// derive vorticity (global vector) as \omega = curl u + f
// assumes diagonal 0 form mass matrix
void SWEqn::diagnose_w(Vec u, Vec *w) {
    Vec du;

    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, w);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &du);

    VecZeroEntries(du);
    MatMult(E01M1, u, du);
    // diagonal mass matrix as vector
    VecPointwiseDivide(*w, du, m0->vg);
    // add the (0 form) coriolis vector
    VecAYPX(*w, 1.0, fg);

    VecDestroy(&du);
}

void SWEqn::diagnose_F(Vec u, Vec hl, KSP ksp, Vec* hu) {
    Vec Fu;

    // assemble the nonlinear rhs mass matrix (note that hl is a local vector)
    F->assemble(hl);

    // get the rhs vector
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Fu);
    VecZeroEntries(Fu);
    MatMult(F->M, u, Fu);

    // solve the linear system
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, hu);
    KSPSolve(ksp, Fu, *hu);
    VecDestroy(&Fu);
}

void SWEqn::solve(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save) {
    int rank;
    char fieldname[20];
    Vec wi, wj, uj, hj;
    Vec gh, uu, wv, hu;
    Vec Ui, Hi, Uj, Hj;
    Vec bu;
    Vec wl, ul, hl;
    KSP ksp;
    PC pc;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // initialize vectors
    VecCreateSeq(MPI_COMM_SELF, topo->n0, &wl);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &Uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Hi);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &Hj);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &bu);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &wv);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &gh);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &uu);

    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &uj);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &hj);

    // initialize the linear solver
    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetOptionsPrefix(ksp,"sw_");
    KSPSetFromOptions(ksp);

    /*** first step ***/
    if(!rank) cout << "half step..." << endl;

    // diagnose the initial vorticity
    diagnose_w(ui, &wi);

    // scatter initial vectors to local versions
    VecScatterBegin(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wi, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_2, hi, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, hi, hl, INSERT_VALUES, SCATTER_FORWARD);

    // momemtum equation
    if(!rank) cout << "\tmomentum eqn" << endl;
    K->assemble(ul);
    R->assemble(wl);

    VecZeroEntries(uu);
    MatMult(K->M, ui, uu);

    VecZeroEntries(gh);
    MatMult(M2->M, hi, gh);
    VecAYPX(gh, grav, uu);      // M2.(K + gh)
    
    VecZeroEntries(Ui);
    MatMult(EtoF->E12, gh, Ui); // E12.M2.(K + gh)

    VecZeroEntries(wv);
    MatMult(R->M, ui, wv);
    VecAYPX(Ui, 1.0, wv);       // M1.(w + f)Xu + E12.M2.(K + gh)

    VecZeroEntries(bu);
    MatMult(M1->M, ui, bu);     // M1.u
    VecAXPY(bu, -dt, Ui);       // M1.u - dt{ M1.(w + f)Xu + E12.M2.(K + gh) }

    // linear solve for uj
    VecZeroEntries(uj);
    KSPSolve(ksp, bu, uj);

    // mass equation
    if(!rank) cout << "\tcontinuity eqn" << endl;
    diagnose_F(ui, hl, ksp, &hu);
    
    VecZeroEntries(Hi);
    MatMult(EtoF->E21, hu, Hi);

    VecZeroEntries(hj);
    VecAXPY(hj, 1.0, hi);
    VecAXPY(hj, -dt, Hi);

    VecDestroy(&hu);

    /*** second step ***/
    if(!rank) cout << "full step..." << endl;

    // diagnose the half step vorticity
    diagnose_w(uj, &wj);

    // scatter half step vectors to local versions
    VecScatterBegin(topo->gtol_0, wj, wl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, wj, wl, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_1, uj, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, uj, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterBegin(topo->gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, hj, hl, INSERT_VALUES, SCATTER_FORWARD);

    // momentum equation
    if(!rank) cout << "\tmomentum eqn" << endl;
    K->assemble(ul);
    R->assemble(wl);

    VecZeroEntries(uu);
    MatMult(K->M, uj, uu);

    VecZeroEntries(gh);
    MatMult(M2->M, hj, gh);
    VecAYPX(gh, grav, uu);      // M2.(K + gh)
    
    VecZeroEntries(Uj);
    MatMult(EtoF->E12, gh, Uj); // E12.M2.(K + gh)

    VecZeroEntries(wv);
    MatMult(R->M, uj, wv);
    VecAYPX(Uj, 1.0, wv);       // M1.(w + f)Xu + E12.M2.(K + gh)

    VecZeroEntries(bu);
    MatMult(M1->M, uj, bu);     // M1.u
    VecAXPY(bu, -0.5*dt, Ui);   // M1.u - dt{ M1.(w + f)Xu + E12.M2.(K + gh) }
    VecAXPY(bu, -0.5*dt, Uj);

    // linear solve for uf
    VecZeroEntries(uf);
    KSPSolve(ksp, bu, uf);

    // mass equation
    if(!rank) cout << "\tcontinuity eqn" << endl;
    diagnose_F(uj, hl, ksp, &hu);
    
    VecZeroEntries(Hj);
    MatMult(EtoF->E21, hu, Hj);

    VecZeroEntries(hf);
    VecAXPY(hf, 1.0, hi);
    VecAXPY(hf, -0.5*dt, Hi);
    VecAXPY(hf, -0.5*dt, Hj);

    VecDestroy(&hu);

    if(!rank) cout << "...done." << endl;

    // write fields
    if(save) {
        step++;
        sprintf(fieldname, "vorticity");
        geom->write0(wi, fieldname, step);
        sprintf(fieldname, "velocity");
        geom->write1(uf, fieldname, step);
        sprintf(fieldname, "pressure");
        geom->write2(hf, fieldname, step);
    }

    // clean up
    VecDestroy(&wl);
    VecDestroy(&ul);
    VecDestroy(&hl);

    VecDestroy(&wi);
    VecDestroy(&wj);
    VecDestroy(&uj);
    VecDestroy(&hj);

    VecDestroy(&wv);
    VecDestroy(&uu);
    VecDestroy(&gh);

    VecDestroy(&Ui);
    VecDestroy(&Uj);
    VecDestroy(&Hi);
    VecDestroy(&Hj);

    VecDestroy(&bu);

    KSPDestroy(&ksp);
}

SWEqn::~SWEqn() {
    MatDestroy(&E01M1);
    MatDestroy(&E12M2);
    VecDestroy(&fg);

    delete m0;
    delete M1;
    delete M2;

    delete NtoE;
    delete EtoF;

    delete R;
    delete F;
    delete K;

    delete edge;
    delete node;
    delete quad;
}

void SWEqn::init0(Vec q, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int* inds0;
    PtQmat* PQ = new PtQmat(topo, geom, node);
    PetscScalar *bArray;
    Vec bl, bg, PQb;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &PQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);
    VecScatterBegin(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(PQ->M, bg, PQb);
    VecPointwiseDivide(q, PQb, m0->vg);

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&PQb);
    delete PQ;
}

void SWEqn::init1(Vec u, ICfunc* func_x, ICfunc* func_y) {
    int ex, ey, ii, mp1, mp12;
    int *inds0, *loc02;
    UtQmat* UQ = new UtQmat(topo, geom, node, edge);
    PetscScalar *bArray;
    KSP ksp;
    PC pc;
    Vec bl, bg, UQb;
    IS isl, isg;
    VecScatter scat;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, 2*topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, 2*topo->n0l, 2*topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &UQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[2*inds0[ii]+0] = func_x(geom->x[inds0[ii]]);
                bArray[2*inds0[ii]+1] = func_y(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);

    // create a new vec scatter object to handle vector quantity on nodes
    loc02 = new int[2*topo->n0];
    for(ii = 0; ii < topo->n0; ii++) {
        loc02[2*ii+0] = 2*topo->loc0[ii]+0;
        loc02[2*ii+1] = 2*topo->loc0[ii]+1;
    }
    ISCreateStride(MPI_COMM_WORLD, 2*topo->n0, 0, 1, &isl);
    ISCreateGeneral(MPI_COMM_WORLD, 2*topo->n0, loc02, PETSC_COPY_VALUES, &isg);
    VecScatterCreate(bg, isg, bl, isl, &scat);
    VecScatterBegin(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(scat, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(UQ->M, bg, UQb);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M1->M, M1->M);
    KSPGetPC(ksp,&pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, 2*topo->elOrd*(topo->elOrd+1), NULL);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOptionsPrefix(ksp,"init1_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, UQb, u);

    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&UQb);
    KSPDestroy(&ksp);
    ISDestroy(&isl);
    ISDestroy(&isg);
    VecScatterDestroy(&scat);
    delete UQ;
    delete[] loc02;
}

void SWEqn::init2(Vec h, ICfunc* func) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    PetscScalar *bArray;
    KSP ksp;
    Vec bl, bg, WQb;
    WtQmat* WQ = new WtQmat(topo, geom, edge);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &bl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &bg);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &WQb);
    VecZeroEntries(bg);

    VecGetArray(bl, &bArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            for(ii = 0; ii < mp12; ii++) {
                bArray[inds0[ii]] = func(geom->x[inds0[ii]]);
            }
        }
    }
    VecRestoreArray(bl, &bArray);
    VecScatterBegin(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, bl, bg, INSERT_VALUES, SCATTER_REVERSE);

    MatMult(WQ->M, bg, WQb);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M2->M, M2->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOptionsPrefix(ksp,"init2_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, WQb, h);

    delete WQ;
    KSPDestroy(&ksp);
    VecDestroy(&bl);
    VecDestroy(&bg);
    VecDestroy(&WQb);
}

double SWEqn::err0(Vec ug, ICfunc* fw, ICfunc* fu, ICfunc* fv) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    double det;
    double un[1], dun[2], ua[1], dua[2];
    double local[2], global[2]; // first entry is the H(rot) error, the second is the norm
    double **J;
    PetscScalar *array_0, *array_1;
    Vec ul, dug, dul;

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n1, &dul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &dug);

    VecScatterBegin(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    MatMult(NtoE->E10, ug, dug);
    VecScatterBegin(topo->gtol_1, dug, dul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, dug, dul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local[0] = local[1] = 0.0;

    VecGetArray(ul, &array_0);
    VecGetArray(dul, &array_1);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
                det = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
                geom->interp0(ex, ey, ii%mp1, ii/mp1, array_0, un);
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, dun, J);

                ua[0] = fw(geom->x[inds0[ii]]);
                dua[0] = fu(geom->x[inds0[ii]]);
                dua[1] = fv(geom->x[inds0[ii]]);

                local[0] += det*quad->w[ii%mp1]*quad->w[ii/mp1]*((un[0] - ua[0])*(un[0] - ua[0]) + 
                                                                 (dun[0] - dua[0])*(dun[0] - dua[0]) + 
                                                                 (dun[1] - dua[1])*(dun[1] - dua[1]));
                local[1] += det*quad->w[ii%mp1]*quad->w[ii/mp1]*(ua[0]*ua[0] + 
                                                                 dua[0]*dua[0] + 
                                                                 dua[1]*dua[1]);
            }
        }
    }
    VecRestoreArray(ul, &array_0);
    VecRestoreArray(dul, &array_1);

    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[] J[0];
    delete[] J[1];
    delete[] J;
    VecDestroy(&ul);
    VecDestroy(&dul);
    VecDestroy(&dug);

    return sqrt(global[0]/global[1]);
}

double SWEqn::err1(Vec ug, ICfunc* fu, ICfunc* fv, ICfunc* fp) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    double det;
    double un[2], dun[1], ua[2], dua[1];
    double local[2], global[2]; // first entry is the H(rot) error, the second is the norm
    double **J;
    PetscScalar *array_1, *array_2;
    Vec ul, dug, dul;

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, topo->n2, &dul);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &dug);

    VecScatterBegin(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    MatMult(EtoF->E21, ug, dug);
    VecScatterBegin(topo->gtol_2, dug, dul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, dug, dul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local[0] = local[1] = 0.0;

    VecGetArray(ul, &array_1);
    VecGetArray(dul, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
                det = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, un, J);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, dun, J);

                ua[0] = fu(geom->x[inds0[ii]]);
                ua[1] = fv(geom->x[inds0[ii]]);
                dua[0] = fp(geom->x[inds0[ii]]);

                local[0] += det*quad->w[ii%mp1]*quad->w[ii/mp1]*((un[0] - ua[0])*(un[0] - ua[0]) + 
                                                                 (un[1] - ua[1])*(un[1] - ua[1]) + 
                                                                 (dun[0] - dua[0])*(dun[0] - dua[0]));
                local[1] += det*quad->w[ii%mp1]*quad->w[ii/mp1]*(ua[0]*ua[0] + 
                                                                 ua[1]*ua[1] + 
                                                                 dua[0]*dua[0]);
            }
        }
    }
    VecRestoreArray(ul, &array_1);
    VecRestoreArray(dul, &array_2);

    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[] J[0];
    delete[] J[1];
    delete[] J;
    VecDestroy(&ul);
    VecDestroy(&dul);
    VecDestroy(&dug);

    return sqrt(global[0]/global[1]);
}

double SWEqn::err2(Vec ug, ICfunc* fu) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    double det;
    double un[1], ua[1];
    double local[2], global[2]; // first entry is the H(rot) error, the second is the norm
    double **J;
    PetscScalar *array_2;
    Vec ul;

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &ul);
    VecScatterBegin(topo->gtol_2, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local[0] = local[1] = 0.0;

    VecGetArray(ul, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);

            for(ii = 0; ii < mp12; ii++) {
                det = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, un, J);

                ua[0] = fu(geom->x[inds0[ii]]);

                local[0] += det*quad->w[ii%mp1]*quad->w[ii/mp1]*(un[0] - ua[0])*(un[0] - ua[0]);
                local[1] += det*quad->w[ii%mp1]*quad->w[ii/mp1]*ua[0]*ua[0];
            }
        }
    }
    VecRestoreArray(ul, &array_2);

    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[] J[0];
    delete[] J[1];
    delete[] J;
    VecDestroy(&ul);

    return sqrt(global[0]/global[1]);
}

double SWEqn::int0(Vec ug) {
    int ex, ey, ii, mp1, mp12;
    double det, uq, local, global;
    double **J;
    PetscScalar *array_0;
    Vec ul;

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ul);
    VecScatterBegin(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ul, &array_0);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
                geom->interp0(ex, ey, ii%mp1, ii/mp1, array_0, &uq);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*uq;
            }
        }
    }
    VecRestoreArray(ul, &array_0);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[] J[0];
    delete[] J[1];
    delete[] J;
    VecDestroy(&ul);

    return global;
}

double SWEqn::int2(Vec ug) {
    int ex, ey, ii, mp1, mp12;
    double det, uq, local, global;
    double **J;
    PetscScalar *array_2;
    Vec ul;

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &ul);
    VecScatterBegin(topo->gtol_2, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ul, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, &uq, J);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*uq;
            }
        }
    }
    VecRestoreArray(ul, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[] J[0];
    delete[] J[1];
    delete[] J;
    VecDestroy(&ul);

    return global;
}

double SWEqn::intE(Vec ug, Vec hg) {
    int ex, ey, ii, mp1, mp12;
    double det, hq, local, global;
    double uq[2];
    double **J;
    PetscScalar *array_1, *array_2;
    Vec ul, hl;

    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecScatterBegin(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, ug, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateSeq(MPI_COMM_SELF, topo->n2, &hl);
    VecScatterBegin(topo->gtol_2, hg, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, hg, hl, INSERT_VALUES, SCATTER_FORWARD);

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    local = 0.0;

    VecGetArray(ul, &array_1);
    VecGetArray(hl, &array_2);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            for(ii = 0; ii < mp12; ii++) {
                det = geom->jacDet(ex, ey, ii%mp1, ii/mp1, J);
                geom->interp1_g(ex, ey, ii%mp1, ii/mp1, array_1, uq, J);
                geom->interp2_g(ex, ey, ii%mp1, ii/mp1, array_2, &hq, J);

                local += det*quad->w[ii%mp1]*quad->w[ii/mp1]*(grav*hq + 0.5*(uq[0]*uq[0] + uq[1]*uq[1]));
            }
        }
    }
    VecRestoreArray(ul, &array_1);
    VecRestoreArray(hl, &array_2);

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    delete[] J[0];
    delete[] J[1];
    delete[] J;
    VecDestroy(&ul);
    VecDestroy(&hl);

    return global;
}
