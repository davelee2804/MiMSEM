class PrimEqns {
    public:
        PrimEqns(Topo* _topo, Geom* _geom);
        ~PrimEqns();
        double grav;
        double omega;
        double del2;
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
        Mat E01M1;
        Mat E12M2;
        Mat V01;                                     // vertical divergence operator
        Mat V10;                                     // vertical gradient operator
        KSP ksp1;
        KSP ksp2;
        double viscosity();
        void coriolis();
        void vertOps();
        void grad(Vec phi, Vec* u, int lev);               // weak form grad operator
        void curl(Vec u, Vec* w, int lev, bool add_f);     // weak form curl operator
        void laplacian(Vec u, Vec* ddu, int lev);          // laplacian operator via helmholtz decomposition
        void AssembleConst(int ex, int ey, Mat M0);        // piecewise constant (in vertical) mass matrix
        void AssembleLinear(int ex, int ey, Mat M1);       // piecewise linear (in vertical) mass matrix
        void AssembleGrav(int ex, int ey, Mat Mg);         // vertical gravity gradient operator
        void VerticalKE(int ex, int ey, Vec* kh, Vec* kv); // kinetic energy vertical vector
        void VertFlux(int ex, int ey, Vec* pi, Vec wi, Mat Mp); // vertical mass flux matrix
        void VertVelRHS(Vec* ui, Vec* wi, Vec **fw);
        void solve_RK2(Vec wi, Vec di, Vec hi, Vec wf, Vec df, Vec hf, double dt, bool save);
        void horizMomRHS(Vec ui, Vec* wi, Vec theta, Vec exner, int lev, Vec *Fu);
};
