typedef double (ICfunc) (double* xi);

class SWEqn {
    public:
        SWEqn(Topo* _topo, Geom* _geom);
        ~SWEqn();
        double dt;
        double grav;
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
        Pmat* M0;
        Umat* M1;
        Wmat* M2;
        E10mat* NtoE;
        E21mat* EtoF;
        RotMat* R;
        Uhmat* M1h;
        Phmat* M0h;
        WtQUmat* K;
        Vec fg;            // coriolis vector (global)
        Vec fl;            // coriolis vector (local)
        Vec topog;
        Mat E01M1;
        Mat E12M2;
        KSP ksp;           // 1 form mass matrix linear solver
        KSP ksp0;          // 0 form mass matrix linear solver
        KSP ksp0h;         // 0 form mass matrix linear solver
        KSP ksp_rot;
        KSP ksp_helm;
        VecScatter gtol_x;
        Vec ui;
        Vec hi;
        Vec uj;
        Vec hj;
        Vec uil;
        Vec ujl;
        Vec u_prev;
        Mat A;
        Mat DM1inv;
        RotMat_up* R_up;
        void coriolis();
        void curl(Vec u, Vec* w);
        void diagnose_F(Vec* F);
        void diagnose_Phi(Vec* Phi);
        void diagnose_q(double _dt, Vec _ug, Vec _ul, Vec _h, Vec* qi);
        void init0(Vec q, ICfunc* func);
        void init1(Vec u, ICfunc* func_x, ICfunc* func_y);
        void init2(Vec h, ICfunc* func);
        void err0(Vec u, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms);
        void err1(Vec u, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms);
        void err2(Vec u, ICfunc* fu, double* norms);
        double int0(Vec u);
        double int2(Vec u);
        double intE(Vec u, Vec h);
        void laplacian(Vec u, Vec* ddu);
        void writeConservation(double time, Vec u, Vec h, double mass0, double vort0, double ener0);
        void assemble_residual(Vec x, Vec f);
        void assemble_operator(double dt);
        void assemble_operator_schur(double dt);
        void solve_schur(Vec Fu, Vec Fh, Vec _u, Vec _h, double imp_dt);
        void solve(Vec u, Vec h, double _dt, bool save);
        void solve_imex(Vec un, Vec hn, double _dt, bool save);
        void solve_implicit(Vec un, Vec hn, double _dt, bool save);
        void solve_rosenbrock(Vec un, Vec hn, double _dt, bool save);
        void rosenbrock_residuals(Vec _u, Vec _h, Vec _ul, Vec fu, Vec fh);
        void rosenbrock_solve(Vec _ui, Vec _uil, Vec _hi, Vec _uj, Vec _hj);
        void rhs_2ndOrd(Vec fu, Vec fh);
        double viscosity();
        void unpack(Vec x, Vec u, Vec h);
        void repack(Vec x, Vec u, Vec h);
        void coriolisMatInv(Mat A, Mat* Ainv);
};
