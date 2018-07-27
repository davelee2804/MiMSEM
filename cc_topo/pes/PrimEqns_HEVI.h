typedef double (ICfunc3D) (double* xi, int ki);

class PrimEqns_HEVI {
    public:
        PrimEqns_HEVI(Topo* _topo, Geom* _geom, double _dt);
        ~PrimEqns_HEVI();
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
        void grad(Vec phi, Vec* u, int lev);                      // weak form grad operator
        void curl(Vec u, Vec* w, int lev, bool add_f);            // weak form curl operator
        void laplacian(Vec u, Vec* ddu, int lev);                 // laplacian operator via helmholtz decomposition
        void AssembleConst(int ex, int ey, Mat A, double scale);  // piecewise constant (in vertical) mass matrix
        void AssembleLinear(int ex, int ey, Mat B, double scale); // piecewise linear (in vertical) mass matrix
        void AssembleLinCon(int ex, int ey, Mat AB, double scale);
        void AssembleLinearWithTheta(int ex, int ey, Vec* theta, Mat A, double scale);
        void AssembleConstWithTheta(int ex, int ey, Vec* theta, Mat A, double scale);
        void AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A, double scale);
        void AssembleVertLaplacian(int ex, int ey, Mat M0, double scale);
        void VertFlux(int ex, int ey, Vec* pi, Mat Mp, double scale);    // vertical mass flux matrix
        void massRHS(Vec* uh, Vec* uv, Vec* pi, Vec* Fh, Vec* Fv, Vec* Fp);
        void vertMomRHS(Vec* ui, Vec* wi, Vec* theta, Vec* exner, Vec *fw);
        void horizMomRHS(Vec ui, Vec* wi, Vec* theta, Vec exner, int lev, double scale, Vec *Fu);
        void thetaBCVec(int ex, int ey, Mat A, Vec* rho, Vec* bTheta, double scale);
        void diagTheta(Vec* rho, Vec* rt, Vec* theta);
        void progExner(Vec rt_i, Vec rt_f, Vec DG, Vec exner_i, Vec* exner_f, int lev);
        void AssembleKEVecs(Vec* velx, Vec* velz, double scale);
        void VertToHoriz2(int ex, int ey, int ki, int kf, Vec pv, Vec* ph);
        void HorizToVert2(int ex, int ey, Vec* ph, Vec pv);
        void SolveEuler(Vec* velx, Vec* velw, Vec* rho, Vec* theta, Vec* exner, bool save);
        void SolveRK2(Vec* velx, Vec* velw, Vec* rho, Vec* theta, Vec* exner, bool save);
        void init0(Vec* q, ICfunc3D* func);
        void init1(Vec* u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* p, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);
        void solveMass(double dt, int ex, int ey, double scale, Mat AB, Vec wz, Vec fv, Vec rho);
        void solveMom(double dt, int ex, int ey, double scale, Mat BA, Vec wz, Vec fv);
};