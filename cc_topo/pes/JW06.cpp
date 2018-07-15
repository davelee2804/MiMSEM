#include <iostream>
#include <fstream>

#include <cmath>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"
#include "PrimEqns.h"

using namespace std;

#define RAD_EARTH 6371220.0
#define RAD_SPHERE 6371220.0
//#define NK 26
#define NK 14
//#define NK 4
#define P0 100000.0
#define U0 35.0
#define T0 288.0
#define RD 287.04
#define DELTA_T 480000.0
#define GAMMA 0.005
#define ETA_T 0.2
#define ETA_0 0.252
#define GRAVITY 9.80616
#define OMEGA 7.29212e-5
#define CP 1005.7
#define KAPPA (RD/CP)

//double A[27] = {0.002194067,0.004895209,0.009882418,0.01805201,0.02983724,0.04462334,0.06160587,0.07851243,0.07731271,
//                0.07590131, 0.07424086, 0.07228744, 0.06998933,0.06728574,0.06410509,0.06036322,0.05596111,0.05078225,
//                0.04468960, 0.03752191, 0.02908949, 0.02084739,0.01334443,0.00708499,0.00252136,0.0,       0.0       };
//double B[27] = {0.0,        0.0,        0.0,        0.0,       0.0,       0.0,       0.0,       0.0,       0.01505309,
//                0.03276228, 0.05359622, 0.07810627, 0.1069411, 0.1408637, 0.1807720, 0.2277220, 0.2829562, 0.3479364,
//                0.4243822,  0.5143168,  0.6201202,  0.7235355, 0.8176768, 0.8962153, 0.9534761, 0.9851122, 1.0       };
double A[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double B[15] = {0.05034551, 0.10365252, 0.16189536, 0.22606120, 0.29615005, 
                0.37413623, 0.45705824, 0.54392892, 0.63376111, 0.72260612, 
                0.80651530, 0.88153998, 0.94274432, 0.98519250, 1.00000000};
//double A[5] = {0.00,0.00,0.00,0.00,0.00};
//double B[5] = {0.00,0.25,0.50,0.75,1.00};

double t_bar(double eta) {
    if(eta < ETA_T) {
        return T0*pow(eta, RD*GAMMA/GRAVITY) + DELTA_T*pow(ETA_T - eta, 5.0);
    }
    else {
        return T0*pow(eta, RD*GAMMA/GRAVITY);
    }
}

// hybrid coordinate coefficients, A and B derived from:
//     Lauritzen, Jablonowski, Taylor and Nair, JAMES, 2010
double z_from_eta(double* x, int ki) {
    int kk;
    double pt, pb, ph, dp, temp, rho, eta_h, dz, z = 0.0;
    //double Ai[5] = {0.00,0.00,0.00,0.00,0.00};
    //double Bi[5] = {0.00,0.25,0.50,0.75,1.00};
    double Ai[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double Bi[15] = {0.05034551, 0.10365252, 0.16189536, 0.22606120, 0.29615005, 
                0.37413623, 0.45705824, 0.54392892, 0.63376111, 0.72260612, 
                0.80651530, 0.88153998, 0.94274432, 0.98519250, 1.00000000};
//    double Ai[27] = {0.002194067,0.004895209,0.009882418,0.01805201,0.02983724,0.04462334,0.06160587,0.07851243,0.07731271,
//                     0.07590131, 0.07424086, 0.07228744, 0.06998933,0.06728574,0.06410509,0.06036322,0.05596111,0.05078225,
//                     0.04468960, 0.03752191, 0.02908949, 0.02084739,0.01334443,0.00708499,0.00252136,0.0,       0.0       };
//    double Bi[27] = {0.0,        0.0,        0.0,        0.0,       0.0,       0.0,       0.0,       0.0,       0.01505309,
//                     0.03276228, 0.05359622, 0.07810627, 0.1069411, 0.1408637, 0.1807720, 0.2277220, 0.2829562, 0.3479364,
//                     0.4243822,  0.5143168,  0.6201202,  0.7235355, 0.8176768, 0.8962153, 0.9534761, 0.9851122, 1.0       };

    if(ki == 0) return z;

    for(kk = 0; kk < ki; kk++) {
        pt    = Ai[NK-kk-1]*P0 + Bi[NK-kk-1]*P0;
        pb    = Ai[NK-kk+0]*P0 + Bi[NK-kk+0]*P0;
        ph    = 0.5*(pt + pb);
        dp    = pb - pt;
        eta_h = ph/P0;
        temp  = t_bar(eta_h);
        rho   = ph/RD/temp;
        dz    = dp/rho/GRAVITY;
        z    += dz;
    }
    return z;
}

double f_topog(double* x) {
    return 0.0;
}

// initial condition given by:
//     Jablonowski and Williamson, QJRMS, 2006
double u_init(double* x, int ki) {
    double phi     = asin(x[2]/RAD_EARTH);
    double theta   = atan2(x[1], x[0]);
    //double Ai[5]   = {0.00,0.00,0.00,0.00,0.00};
    //double Bi[5]   = {0.00,0.25,0.50,0.75,1.00};
    double Ai[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double Bi[15] = {0.05034551, 0.10365252, 0.16189536, 0.22606120, 0.29615005, 
                0.37413623, 0.45705824, 0.54392892, 0.63376111, 0.72260612, 
                0.80651530, 0.88153998, 0.94274432, 0.98519250, 1.00000000};
    double eta     = 0.5*(Ai[NK-ki] + Bi[NK-ki] + Ai[NK-ki-1] + Bi[NK-ki-1]);
    double eta_v   = 0.5*(eta - ETA_0)*M_PI;
    double us      = U0*pow(cos(eta_v), 1.5)*sin(2.0*phi)*sin(2.0*phi);
    double theta_c = 1.0*M_PI/9.0;
    double phi_c   = 2.0*M_PI/9.0;
    double rad     = acos(sin(phi_c)*sin(phi) + cos(phi_c)*cos(phi)*cos(theta - theta_c));
    double up      = 1.0*exp(-rad*rad/10.0/10.0);
    
    return us + up;
}

double v_init(double* x, int ki) {
    return 0.0;
}

double t_init(double* x, int ki) {
    double phi     = asin(x[2]/RAD_EARTH);
    //double Ai[5]   = {0.00,0.00,0.00,0.00,0.00};
    //double Bi[5]   = {0.00,0.25,0.50,0.75,1.00};
    double Ai[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double Bi[15] = {0.05034551, 0.10365252, 0.16189536, 0.22606120, 0.29615005, 
                0.37413623, 0.45705824, 0.54392892, 0.63376111, 0.72260612, 
                0.80651530, 0.88153998, 0.94274432, 0.98519250, 1.00000000};
    double eta     = Ai[NK-ki] + Bi[NK-ki]; // temperature is defined at the half levels
    double eta_v   = 0.5*(eta - ETA_0)*M_PI;
    double t_avg   = t_bar(eta);
    double a       = 10.0/63.0 - 2.0*pow(sin(phi), 6.0)*(cos(phi)*cos(phi) + 1.0/3.0);
    double b       = 1.6*pow(cos(phi), 3.0)*(sin(phi)*sin(phi) + 2.0/3.0) - 0.25*M_PI;
    double temp    = t_avg + 0.75*eta*M_PI*U0/RD*sin(eta_v)*sqrt(cos(eta_v))*(a*2.0*U0*pow(cos(eta_v), 1.5) + b*RAD_EARTH*OMEGA);
    
    return temp;
}

double rho_init(double* x, int ki) {
    double tb  = t_init(x, ki    );
    double tt  = t_init(x, ki + 1);
    double th  = 0.5*(tb + tt);
    //double Ai[5] = {0.00,0.00,0.00,0.00,0.00};
    //double Bi[5] = {0.00,0.25,0.50,0.75,1.00};
    double Ai[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double Bi[15] = {0.05034551, 0.10365252, 0.16189536, 0.22606120, 0.29615005, 
                0.37413623, 0.45705824, 0.54392892, 0.63376111, 0.72260612, 
                0.80651530, 0.88153998, 0.94274432, 0.98519250, 1.00000000};
    double pb  = Ai[NK-ki+0]*P0 + Bi[NK-ki+0]*P0;
    double pt  = Ai[NK-ki-1]*P0 + Bi[NK-ki-1]*P0;
    double ph  = 0.5*(pb + pt);
    double rho = ph/RD/th;

    return rho;
}

double theta_init(double* x, int ki) {
    //double phi   = asin(x[2]/RAD_EARTH);
    //double Ac    = 10.0/63.0 - 2.0*pow(phi, 6.0)*(cos(phi)*cos(phi) + 1.0/3.0);
    //double Bc    = RAD_EARTH*OMEGA*(1.6*pow(cos(phi), 3.0)*(sin(phi)*sin(phi) + 2.0/3.0) - 0.25*M_PI);
    double temp  = t_init(x, ki);
    //double Ai[5] = {0.00,0.00,0.00,0.00,0.00};
    //double Bi[5] = {0.00,0.25,0.50,0.75,1.00};
    double Ai[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double Bi[15] = {0.05034551, 0.10365252, 0.16189536, 0.22606120, 0.29615005, 
                0.37413623, 0.45705824, 0.54392892, 0.63376111, 0.72260612, 
                0.80651530, 0.88153998, 0.94274432, 0.98519250, 1.00000000};
    double eta   = Ai[NK-ki] + Bi[NK-ki]; // theta is defined at the layer interfaces
    //double eta_v = 0.5*(eta - ETA_0)*M_PI;
    double Pi    = pow(eta*P0/P0, KAPPA);
    //double theta = (temp + 0.75*eta*M_PI*U0/RD*sin(eta_v)*sqrt(cos(eta_v))*(2.0*U0*Ac*pow(cos(eta_v), 1.5) + Bc))/pres;
    double theta = temp/Pi;

    return theta;
}

double rt_init(double* x, int ki) {
    double theta_t = theta_init(x, ki + 0);
    double theta_b = theta_init(x, ki + 1);
    double rho     = rho_init(x, ki);

    return 0.5*rho*(theta_b + theta_t);
}

double exner_init(double* x, int ki) {
    //double Ai[5] = {0.00,0.00,0.00,0.00,0.00};
    //double Bi[5] = {0.00,0.25,0.50,0.75,1.00};
    double Ai[15] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double Bi[15] = {0.05034551, 0.10365252, 0.16189536, 0.22606120, 0.29615005, 
                0.37413623, 0.45705824, 0.54392892, 0.63376111, 0.72260612, 
                0.80651530, 0.88153998, 0.94274432, 0.98519250, 1.00000000};
    double pres = 0.5*(Ai[NK-ki] + Bi[NK-ki] + Ai[NK-ki-1] + Bi[NK-ki-1])*P0;

    return CP*pow(pres/P0, RD/CP); // scaling by c_p
}

double theta_t_init(double* x, int ki) {
    return theta_init(x, 0);
}

double theta_b_init(double* x, int ki) {
    return theta_init(x, NK);
}

void LoadVecs(Vec* vecs, int nk, char* fieldname, int step, bool para) {
    int ki;
    char filename[100];
    PetscViewer viewer;

    for(ki = 0; ki < NK; ki++) {
        sprintf(filename, "output/%s_%.4u_%.4u.vec", fieldname, ki, step);
        if(para) {
            PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldname, FILE_MODE_READ, &viewer);
        }
        else {
            PetscViewerBinaryOpen(PETSC_COMM_SELF, fieldname, FILE_MODE_READ, &viewer);
        }
        VecLoad(vecs[ki], viewer);
        PetscViewerDestroy(&viewer);
    }
}

int main(int argc, char** argv) {
    int size, rank, step, ii, ki, n2;
    static char help[] = "petsc";
    //double vort_0, mass_0, ener_0;
    //double vort_n, mass_n, ener_n;
    char fieldname[50];//, filename[50];
    bool dump;
    int startStep = atoi(argv[1]);
    double dt = 80.0;    // for 16x3rd order elements
    int nSteps = 2;//10800;  // 10 days
    int dumpEvery = 1;//270; // 6 hours
    ofstream file;
    Topo* topo;
    Geom* geom;
    PrimEqns* pe;
    Vec *velx, *velz, *rho, *rt, *exner;
    PetscViewer viewer;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo, NK);
    // initialise the z coordinate layer heights
    geom->initTopog(f_topog, z_from_eta);
    pe   = new PrimEqns(topo, geom, dt);
    pe->step = startStep;

    n2 = topo->nElsX*topo->nElsX;

    velx  = new Vec[NK];
    rho   = new Vec[NK];
    rt    = new Vec[NK];
    exner = new Vec[NK];
    velz  = new Vec[n2];
    for(ki = 0; ki < NK; ki++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &velx[ki] );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rho[ki]  );
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &exner[ki]);
        VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &rt[ki]   );
    }
    for(ii = 0; ii < n2; ii++) {
        VecCreateSeq(MPI_COMM_SELF, (NK-1)*topo->elOrd*topo->elOrd, &velz[ii]);
        VecZeroEntries(velz[ii]);
    }

    // initialise the potential temperature top and bottom boundary conditions
    // these are 2d fields at the top and bottom, so re-set the thicknesses to
    // unity before initialising, then reset to correct values afterwards
    geom->initTopog(f_topog, NULL);
    pe->initTheta(pe->theta_b, theta_b_init);
    pe->initTheta(pe->theta_t, theta_t_init);
    geom->initTopog(f_topog, z_from_eta);

    if(startStep == 0) {
        pe->init1(velx, u_init, v_init);
        pe->init2(rho,   rho_init  );
        pe->init2(exner, exner_init);
        pe->init2(rt,    rt_init   );

        for(ki = 0; ki < NK; ki++) {
            sprintf(fieldname,"velocity_h");
            geom->write1(velx[ki],fieldname,0,ki);
            sprintf(fieldname,"density");
            geom->write2(rho[ki],fieldname,0,ki);
            sprintf(fieldname,"exner");
            geom->write2(exner[ki],fieldname,0,ki);
            sprintf(fieldname,"rhoTheta");
            geom->write2(rt[ki],fieldname,0,ki);
        }
        for(ii = 0; ii < n2; ii++) {
            sprintf(fieldname, "output/velocity_z_%.4u_%.4u.vec", ii, 0);
            PetscViewerBinaryOpen(MPI_COMM_SELF, fieldname, FILE_MODE_WRITE, &viewer);
            VecView(velz[ii], viewer);
            PetscViewerDestroy(&viewer);
        }
    } else {
        sprintf(fieldname,"density");
        LoadVecs(rho  , NK, fieldname, startStep, true );
        sprintf(fieldname,"velocity_h");
        LoadVecs(velx , NK, fieldname, startStep, true );
        sprintf(fieldname,"exner");
        LoadVecs(exner, NK, fieldname, startStep, true );
        sprintf(fieldname,"rhoTheta");
        LoadVecs(rt   , NK, fieldname, startStep, true );
        sprintf(fieldname,"velociyt_z");
        LoadVecs(velz , n2, fieldname, startStep, false);
    }

    //vort_0 = mass_0 = ener_0 = 0.0;
    //for(ki = 0; ki < NK; ki++) {
    //    pe->curl(velx[ki], &wi, 0, false);
    //    vort_0 += sw->int0(wi);
    //    mass_0 += sw->int2(rho[ki]);
    //    ener_0 += sw->intE(velx[ki], rho[ki]);
    //    VecDestroy(&wi);
    //}

    for(step = startStep*dumpEvery + 1; step <= nSteps; step++) {
        if(!rank) {
            cout << "doing step:\t" << step << ", time (days): \t" << step*dt/60.0/60.0/24.0 << endl;
        }
        dump = (step%dumpEvery == 0) ? true : false;
        //pe->SolveRK2(velx, velz, rho, rt, exner, dump);
        pe->SolveEuler(velx, velz, rho, rt, exner, dump);
        //if(dump) {
            //vort_n = mass_n = ener_n = 0.0;
            //for(ki = 0; ki < NK; ki++) {
                //pe->curl(velx[ki], &wi, 0, false);
                //vort_n += sw->int0(wi);
                //mass_n += sw->int2(rho[ki]);
                //ener_n += sw->intE(velx[ki], rho[ki]);
                //VecDestroy(&wi);

                //sprintf(filename, "output/conservation.dat");
                //file.open(filename, ios::out | ios::app);
                //file << (step*dt)/60.0/60.0/24.0 << "\t" << (mass_n-mass_0)/mass_0 
                //                                 << "\t" << (vort_n-vort_0) 
                //                                 << "\t" << (ener_n-ener_0)/ener_0 << endl;
                //file.close();
            //}
        //}
    }

    delete pe;
    delete geom;
    delete topo;

    for(ki = 0; ki < NK; ki++) {
        VecDestroy(&velx[ki] );
        VecDestroy(&rho[ki]  );
        VecDestroy(&rt[ki]   );
        VecDestroy(&exner[ki]);
    }
    for(ii = 0; ii < n2; ii++) {
        VecDestroy(&velz[ii]);
    }
    delete[] velx;
    delete[] rho;
    delete[] exner;
    delete[] rt;
    delete[] velz;

    PetscFinalize();

    return 0;
}
