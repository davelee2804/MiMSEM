typedef double (ICfunc) (double* xi);

class AdvEqn {
    public:
        AdvEqn(Topo* _topo, Geom* _geom);
        ~AdvEqn();
        double dt;
        double grav;
        double omega;
        int step;
        int rank;
        bool precon_assembled;
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
        Uhmat* M1h;
        Vec fg;            // coriolis vector (global)
        Mat E01M1;
        Mat E12M2;
        KSP ksp;           // 1 form mass matrix linear solver
        Vec ui;
        Vec hi;
        Vec uj;
        Vec hj;
        IS is_u;           // field split index set for u (global)
        IS is_h;           // field split index set for h (global)
        Mat SC;            // schur complement matrix (for the fieldsplit preconditioner)
        void coriolis();
        void diagnose_F(Vec* F);
        void init0(Vec q, ICfunc* func);
        void init1(Vec u, ICfunc* func_x, ICfunc* func_y);
        void init2(Vec h, ICfunc* func);
        void err0(Vec u, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms);
        void err1(Vec u, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms);
        void err2(Vec u, ICfunc* fu, double* norms);
        double int0(Vec u);
        double int2(Vec u);
        double intE(Vec u, Vec h);
        void jfnk_vector(Vec x, Vec f);
        void jfnk_precon(Mat P);
        void solve(Vec u, Vec h, double _dt, bool save);
        void solve_explicit(Vec u, Vec h, double _dt, bool save);
};
