typedef double (ICfunc) (double* xi);

class ThermalSW_EEC_2 {
    public:
        ThermalSW_EEC_2(Topo* _topo, Geom* _geom);
        ~ThermalSW_EEC_2();
        double dt;
        double omega;
        int step;
        int rank;
        int size;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        Pmat* M0;
        Umat* M1;
        Wmat* M2;
        E10mat* NtoE;
        E21mat* EtoF;
        RotMat* R;
        Uhmat* M1h;
        Whmat* M2h;
        Phmat* M0h;
        WtQUmat* K;
        W_IP_mat* M2_ip;
        Vec fg;            // coriolis vector (global)
        Vec fl;            // coriolis vector (local)
        Vec M0fg;
        Mat E01M1;
        Mat E12M2;
        KSP ksp;           // 1 form mass matrix linear solver
        KSP ksp0;          // 0 form mass matrix linear solver
        KSP ksp0h;         // 0 form mass matrix linear solver
        KSP ksp2;
        KSP ksp1h;
        KSP ksp2h;
        Vec ui;
        Vec hi;
        Vec Si;
        Vec uj;
        Vec hj;
        Vec Sj;
        Vec fu;
        Vec fh;
        Vec fS;
        Vec uil;
        Vec ujl;
        Vec F;
        Vec Phi;
        Vec G;
        Vec wi;
	Vec ds_on_h;
	Vec ds_on_h_l;
	Vec S_on_h;
        RotMat_up* R_up;
        void coriolis();
        void curl(Vec u);
        void diagnose_q(Vec _ug, Vec _h, Vec* qi);
        void diagnose_G(Vec _s);
        void diagnose_ds(Vec _h, Vec _s);
        void diagnose_s(Vec _h, Vec _S);
        void init0(Vec q, ICfunc* func);
        void init1(Vec u, ICfunc* func_x, ICfunc* func_y);
        void init2(Vec h, ICfunc* func);
        void err0(Vec u, ICfunc* fw, ICfunc* fu, ICfunc* fv, double* norms);
        void err1(Vec u, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms);
        void err2(Vec u, ICfunc* fu, double* norms);
        double int2(Vec u);
        double intE(Vec ul, Vec hg, Vec sg);
        double intK(Vec dqg, Vec dsg);
        double intZrhs(Vec dqg, Vec dsg, Vec hh);
        void writeConservation(double time, double mass0, double vort0, double ener0, double enst0, double buoy0, double entr0);
        void grad(Vec phi, Vec* _u);
	//////////
        void solve_rk(double _dt, bool save);
        void diagnose_F(Vec _u, Vec _h);
        void diagnose_Phi(Vec _u, Vec _h, Vec _S, Vec _s, Vec _ul);
        void rhs_u(Vec _u, Vec _h, Vec _S, Vec _s, Vec _ul, double _dt);
        void rhs_S();
};
