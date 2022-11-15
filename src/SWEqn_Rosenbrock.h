typedef double (ICfunc) (double* xi);

class SWEqn {
    public:
        SWEqn(Topo* _topo, Geom* _geom);
        ~SWEqn();
        double dt;
        double grav;
        double omega;
        int step;
        int rank;
        int size;
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
        Mat E01M1;
        Mat E12M2;
        KSP ksp;           // 1 form mass matrix linear solver
        KSP ksp0;          // 0 form mass matrix linear solver
        KSP ksp0h;         // 0 form mass matrix linear solver
        KSP kspA;
        KSP ksp1h;
        VecScatter gtol_x;
        Vec uj;
        Vec hj;
        Vec ujl;
        Mat A;
        Mat B;
        Mat DM1inv;
        RotMat_up* R_up;
        Mat Muf;
        Mat G;
        Mat D;
        Mat M1inv;
	Vec* ki;
	int n_rosenbrock;
	double gamma_0;
	double** alpha_ij;
	double** gamma_ij;
	double* bHat_j;
        void coriolis();
        void curl(Vec u, Vec* w);
        void diagnose_F(Vec _h, Vec _u, Vec F);
        void diagnose_Phi(Vec _h, Vec _u, Vec _ul, Vec Phi);
        void diagnose_q(double _dt, Vec _ug, Vec _ul, Vec _h, Vec qi);
        void init0(Vec q, ICfunc* func);
        void init1(Vec u, ICfunc* func_x, ICfunc* func_y);
        void init2(Vec h, ICfunc* func);
        void err0(Vec u, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms);
        void err1(Vec u, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms);
        void err2(Vec u, ICfunc* fu, double* norms);
        double int0(Vec u);
        double int2(Vec u);
        double intE(Vec u, Vec h);
        void writeConservation(double time, Vec u, Vec h, double mass0, double vort0, double ener0, double enst0);
        void assemble_residual(Vec x, Vec f);
        void assemble_operator(double dt);
        void solve(Vec u, Vec h, double _dt, bool save);
        void solve_cn(Vec un, Vec hn, double _dt, bool save, int n_its);
        void unpack(Vec x, Vec u, Vec h);
        void repack(Vec x, Vec u, Vec h);
};
