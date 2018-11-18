typedef double (ICfunc3D) (double* xi, int ki);

class Euler {
    public:
        Euler(Topo* _topo, Geom* _geom, double _dt);
        ~Euler();
        double dt;
        double del2;
        bool do_visc;
        int rank;
        int step;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        Pvec* m0;
        Umat* M1;
        Wmat* M2;
        E10mat* NtoE;
        E21mat* EtoF;
        RotMat* R;
        Uhmat* F;
        WtQUmat* K;
        Ut_mat* M1t;
        UtQWmat* Rh;
        WtQdUdz_mat* Rz;
        Whmat* T;
        Vec* fg;                 // coriolis vector (global)
        bool firstStep;
        double k2i;              // kinetic to internal energy exchange
        double i2k;              // kinetic to internal energy exchange
        double k2i_z;            // kinetic to internal energy exchange
        double i2k_z;            // kinetic to internal energy exchange
        Vec theta_b;             // bottom potential temperature bc
        Vec theta_t;             // top potential temperature bc
        Vec theta_b_l;           // bottom potential temperature bc
        Vec theta_t_l;           // top potential temperature bc
        Vec* Kv;                 // kinetic energy vector for each vertical column
        Vec* Kh;                 // kinetic energy vector for each horiztontal layer
        Vec* gv;
        Vec* zv;
        Vec* uz;                 // dudz and dvdz vorticity components
        L2Vecs* uuz;             // u.dudz + v.dvdz vorticity velocity product
        Mat VA;
        Mat VB;
        KSP ksp1;
        KSP ksp2;
        KSP kspE;
        KSP kspColA;

        VertOps* vo;

        Wii* Q;
        M2_j_xy_i* W;
        double** Q0;
        double** Wt;
        double** WtQ;

        double viscosity();
        void coriolis();
        void initGZ();
        void grad(bool assemble, Vec phi, Vec* u, int lev);            // weak form grad operator
        void curl(bool assemble, Vec u, Vec* w, int lev, bool add_f);  // weak form curl operator
        void laplacian(bool assemble, Vec u, Vec* ddu, int lev);       // laplacian operator via helmholtz decomposition
        void massRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* Flux);
        void tempRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* rho_l, Vec* exner);
        void horizMomRHS(Vec ui, Vec* theta, Vec exner, int lev, Vec Fu, Vec Flux, Vec uzb, Vec uzt, Vec velz_b, Vec velz_t);
        void thetaBCVec(int ex, int ey, Mat A, Vec* bTheta);
        void diagTheta(Vec* rho, Vec* rt, Vec* theta);
        void diagThetaVert(int ex, int ey, Mat AB, Vec rho, Vec rt, Vec theta);
        void AssembleKEVecs(Vec* velx, Vec* velz);
        void VertToHoriz2(int ex, int ey, int ki, int kf, Vec pv, Vec* ph);
        void HorizToVert2(int ex, int ey, Vec* ph, Vec pv);
        void init0(Vec* q, ICfunc3D* func);
        void init1(Vec* u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* p, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);
        void HorizRHS(Vec* velx, Vec* rho, Vec* rt, Vec* exner, Vec* Fu, Vec* Fp, Vec* Ft, Vec* velz);
        void SolveExner(Vec* rt, Vec* Ft, Vec* exner_i, Vec* exner_f, double _dt);
        void StrangCarryover(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save);

        void VertSolve(Vec* velz, Vec* rho, Vec* rt, Vec* exner, Vec* velz_n, Vec* rho_n, Vec* rt_n, Vec* exner_n);
        void VertSolve_Explicit(Vec* velz, Vec* rho, Vec* rt, Vec* exner, Vec* velz_n, Vec* rho_n, Vec* rt_n, Vec* exner_n);
        void diagnostics(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner);

        void DiagExner(Vec* rtz, L2Vecs* exner);

        void HorizVort(Vec* velx);
        void AssembleVertMomVort(Vec* velx);

        L2Vecs* velz_prev;
        L2Vecs* rho_prev;
        L2Vecs* rt_prev;
        L2Vecs* exner_prev;
};
