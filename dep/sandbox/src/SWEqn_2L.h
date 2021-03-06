typedef double (ICfunc) (double* xi);

class SWEqn_2L {
    public:
        SWEqn_2L(Topo* _topo, Geom* _geom);
        ~SWEqn_2L();
        double dt;
        double grav;
        double rho_t;
        double rho_b;
        double H_t;
        double H_b;
        double omega;
        double del2;
        bool do_visc;
        int step;
        int rank;
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
        Uhmat* M1h;
        WtQUmat* K;
        Vec fg;            // coriolis vector (global)
        Mat E01M1;
        Mat E12M2;
        KSP ksp;           // 1 form mass matrix linear solver
        VecScatter gtol_x;
        Vec u_ti;
        Vec h_ti;
        Vec u_tj;
        Vec h_tj;
        Vec u_bi;
        Vec h_bi;
        Vec u_bj;
        Vec h_bj;
        Mat A;
        void coriolis();
        void curl(Vec u, Vec* w);
        void diagnose_F(Vec u1, Vec u2, Vec h1, Vec h2, Vec* F);
        void diagnose_Phi(Vec u1, Vec u2, Vec h_t1, Vec h_t2, Vec h_b1, Vec h_b2, double grav_this, double grav_other, Vec* Phi);
        void diagnose_wxu(Vec u1, Vec u2, Vec* wxu);
        void init0(Vec q, ICfunc* func);
        void init1(Vec u, ICfunc* func_x, ICfunc* func_y);
        void init2(Vec h, ICfunc* func);
        void err0(Vec u, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms);
        void err1(Vec u, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms);
        void err2(Vec u, ICfunc* fu, double* norms);
        double int0(Vec u);
        double int2(Vec u);
        double intE(double gravity, Vec u, Vec h);
        void laplacian(Vec u, Vec* ddu);
        void writeConservation(double time, Vec u1, Vec u2, Vec h1, Vec h2, double mass0, double vort0, double ener0);
        void assemble_residual(Vec x, Vec f);
        void assemble_operator();
        void solve(Vec u1, Vec u2, Vec h1, Vec h2, double _dt, bool save);
        void solve_explicit(Vec u_tn, Vec u_bn, Vec h_tn, Vec h_bn, double _dt, bool save);
        double viscosity();
        void unpack(Vec x, Vec u_t, Vec h_t, Vec u_b, Vec h_b);
        void repack(Vec x, Vec u_t, Vec h_t, Vec u_b, Vec h_b);
};
