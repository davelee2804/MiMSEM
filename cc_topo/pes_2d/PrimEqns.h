typedef double (ICfunc3D) (double xi, int ki);

class PrimEqns {
    public:
        PrimEqns(Topo* _topo, Geom* _geom, double _dt);
        ~PrimEqns();
        double dt;
        double grav;
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
        E21mat* EtoF;
        Uhmat* F;
        WtQUmat* K;
        Whmat* T;
        Vec theta_b;                                 // bottom potential temperature bc
        Vec theta_t;                                 // top potential temperature bc
        Vec* Kv;                                     // kinetic energy vector for each vertical column
        Vec* Kh;                                     // kinetic energy vector for each horiztontal layer
        Mat V01;                                     // vertical divergence operator
        Mat V10;                                     // vertical gradient operator
        Mat VA;
        Mat VB;
        KSP ksp1;
        KSP ksp2;
        KSP kspColA;
        double viscosity();
        double viscosity_vert();
        void vertOps();
        void grad(Vec phi, Vec* u, int lev);                      // weak form grad operator
        void curl(Vec u, Vec* w, int lev, bool add_f);            // weak form curl operator
        void laplacian(Vec u, Vec* ddu, int lev);                 // laplacian operator via helmholtz decomposition
        void AssembleConst(int ex, Mat A);  // piecewise constant (in vertical) mass matrix
        void AssembleLinear(int ex, Mat B); // piecewise linear (in vertical) mass matrix
        void AssembleLinCon(int ex, Mat AB);
        void AssembleLinearWithTheta(int ex, Vec* theta, Mat A);
        void AssembleConstWithTheta(int ex, Vec* theta, Mat A);
        void AssembleLinearWithRho(int ex, Vec* rho, Mat A);
        void AssembleVertLaplacian(int ex, Mat M0);
        void VertFlux(int ex, Vec* pi, Mat Mp);    // vertical mass flux matrix
        void massRHS(Vec* uh, Vec* uv, Vec* pi, Vec* Fh, Vec* Fv, Vec* Fp);
        void vertMomRHS(Vec* ui, Vec* wi, Vec* theta, Vec* exner, Vec *fw);
        void horizMomRHS(Vec ui, Vec* wi, Vec* theta, Vec exner, int lev, Vec *Fu);
        void thetaBCVec(int ex, Mat A, Vec* rho, Vec* bTheta);
        void diagTheta(Vec* rho, Vec* rt, Vec* theta);
        void progExner(Vec rt_i, Vec rt_f, Vec exner_i, Vec* exner_f, int lev);
        void AssembleKEVecs(Vec* velx, Vec* velz);
        void VertToHoriz2(int ex, int ki, int kf, Vec pv, Vec* ph);
        void HorizToVert2(int ex, Vec* ph, Vec pv);
        void SolveEuler(Vec* velx, Vec* velw, Vec* rho, Vec* theta, Vec* exner, bool save);
        void init0(Vec* q, ICfunc3D* func);
        void init1(Vec* u, ICfunc3D* func);
        void init2(Vec* p, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);
};
