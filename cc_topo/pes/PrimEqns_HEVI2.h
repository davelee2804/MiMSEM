typedef double (ICfunc3D) (double* xi, int ki);

class PrimEqns_HEVI2 {
    public:
        PrimEqns_HEVI2(Topo* _topo, Geom* _geom, double _dt);
        ~PrimEqns_HEVI2();
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
        Whmat* T;
        Vec* fg;                                     // coriolis vector (global)
        Vec theta_b;                                 // bottom potential temperature bc
        Vec theta_t;                                 // top potential temperature bc
        Vec* Kv;                                     // kinetic energy vector for each vertical column
        Vec* Kh;                                     // kinetic energy vector for each horiztontal layer
        Vec* gz;
        Vec* gv;
        Mat V01;                                     // vertical divergence operator
        Mat V10;                                     // vertical gradient operator
        Mat VA;
        Mat VB;
        KSP ksp1;
        KSP ksp2;
        KSP kspColA;
        KSP kspColB;
        double viscosity();
        double viscosity_vert();
        void coriolis();
        void vertOps();
        void initGZ();
        void grad(Vec phi, Vec* u, int lev);            // weak form grad operator
        void curl(Vec u, Vec* w, int lev, bool add_f);  // weak form curl operator
        void laplacian(Vec u, Vec* ddu, int lev);       // laplacian operator via helmholtz decomposition
        void AssembleConst(int ex, int ey, Mat A);      // piecewise constant (in vertical) mass matrix
        void AssembleLinear(int ex, int ey, Mat B);     // piecewise linear (in vertical) mass matrix
        void AssembleLinCon(int ex, int ey, Mat AB);
        void AssembleLinearWithTheta(int ex, int ey, Vec* theta, Mat A);
        void AssembleConstWithTheta(int ex, int ey, Vec* theta, Mat A);
        void AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A);
        void AssembleVertLaplacian(int ex, int ey, Mat M0);
        void VertFlux(int ex, int ey, Vec pi, Mat Mp);  // vertical mass flux matrix
        void massRHS(Vec* uh, Vec* pi, Vec* Fp);
        void vertMomRHS(Vec* ui, Vec* wi, Vec* theta, Vec* exner, Vec *fw);
        void horizMomRHS(Vec ui, Vec* theta, Vec exner, int lev, Vec Fu);
        void thetaBCVec(int ex, int ey, Mat A, Vec* rho, Vec* bTheta);
        void diagTheta(Vec* rho, Vec* rt, Vec* theta);
        void progExner(Vec rt_i, Vec DivH, Vec DivV, Vec exner_i, Vec* exner_f, int lev);
        void AssembleKEVecs(Vec* velx, Vec* velz);
        void VertToHoriz2(int ex, int ey, int ki, int kf, Vec pv, Vec* ph, bool assign);
        void HorizToVert2(int ex, int ey, Vec* ph, Vec pv);
        void init0(Vec* q, ICfunc3D* func);
        void init1(Vec* u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* p, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);
        void solveMass(double dt, int ex, int ey, Mat AB, Vec wz, Vec fv, Vec rho);
        void solveMom(double dt, int ex, int ey, Mat BA, Vec wz, Vec fv);
        void HorizRHS(Vec* velx, L2Vecs* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, Vec* Fu, Vec* Fp, Vec* Ft);
        void SolveStrang(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save);
};
