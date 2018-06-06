typedef double (ICfunc3D) (double* xi, int ki);

class PrimEqns {
    public:
        PrimEqns(Topo* _topo, Geom* _geom, double _dt);
        ~PrimEqns();
        double dt;
        double grav;
        double omega;
        double del2;
        double vert_visc;
        bool do_visc;
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
        Vec fg;                                      // coriolis vector (global)
        Vec theta_b;                                 // bottom potential temperature bc
        Vec theta_t;                                 // top potential temperature bc
        Vec* Kv;                                     // kinetic energy vector for each horiztonal element
        Mat E01M1;
        Mat E12M2;
        Mat V01;                                     // vertical divergence operator
        Mat V10;                                     // vertical gradient operator
        KSP ksp1;
        KSP ksp2;
        double viscosity();
        double viscosity_vert();
        void coriolis();
        void vertOps();
        void grad(Vec phi, Vec* u, int lev);                        // weak form grad operator
        void curl(Vec u, Vec* w, int lev, bool add_f);              // weak form curl operator
        void laplacian(Vec u, Vec* ddu, int lev);                   // laplacian operator via helmholtz decomposition
        void AssembleConst(int ex, int ey, Mat A);                 // piecewise constant (in vertical) mass matrix
        void AssembleLinear(int ex, int ey, Mat B, bool add_g);    // piecewise linear (in vertical) mass matrix
        void AssembleLinCon(int ex, int ey, Mat AB);
        void AssembleLinearWithTheta(int ex, int ey, Vec* theta, Mat A);
        void AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A);
        void AssembleVertOps(int ex, int ey, Mat M0);
        void VertFlux(int ex, int ey, Vec* pi, Vec* ti, Mat Mp);    // vertical mass flux matrix
        void massRHS(Vec* uh, Vec* uv, Vec* pi, Vec **Fp);
        void tempRHS(Vec* uh, Vec* uv, Vec* pi, Vec* theta, Vec **Ft);
        void vertMomRHS(Vec* ui, Vec* wi, Vec* theta, Vec* exner, Vec **fw);
        void horizMomRHS(Vec ui, Vec* wi, Vec* theta, Vec exner, int lev, Vec *Fu);
        void thetaBCVec(int ex, int ey, Mat A, Vec* rho, Vec* bTheta);
        void diagTheta(Vec* rho, Vec* rt, Vec* theta);
        void progExner(Vec rt_i, Vec rt_f, Vec exner_i, Vec* exner_f, int lev);
        void UpdateKEVert(Vec ke, int lev);
        void VertConstMatInv(int ex, int ey, Mat M1inv);
        void VertToHoriz2(int ex, int ey, int nk, Vec pv, Vec* ph);
        void HorizToVert2(int ex, int ey, Vec* ph, Vec pv);
        void SolveRK2(Vec* velx, Vec* velw, Vec* rho, Vec* theta, Vec* exner, bool save);
        void init0(Vec* q, ICfunc3D* func);
        void init1(Vec* u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* p, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);
};
