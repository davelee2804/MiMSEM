typedef double (ICfunc) (double* xi);

class ThermalSW {
    public:
        ThermalSW(Topo* _topo, Geom* _geom);
        ~ThermalSW();
        double dt;
        double omega;
        int step;
        int rank;
        int size;
	bool first_step;
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
        //W_tau_mat* M2tau;
        Vec fg;            // coriolis vector (global)
        Vec fl;            // coriolis vector (local)
	Vec M0fg;
        Mat E01M1;
        Mat E12M2;
        KSP ksp;           // 1 form mass matrix linear solver
        KSP ksp0;          // 0 form mass matrix linear solver
        KSP ksp0h;         // 0 form mass matrix linear solver
        KSP ksp2;
        KSP kspA;
        KSP ksp1h;
        VecScatter gtol_x;
        Vec ui;
        Vec hi;
        Vec si;
        Vec uj;
        Vec hj;
        Vec sj;
        Vec fu;
        Vec fh;
        Vec fs;
        Vec u_prev;
        Vec h_prev;
        Vec s_prev;
        Vec uil;
        Vec ujl;
        Vec F;
        Vec Phi;
        Vec T;
        Vec ds;
        Vec dsl;
        Vec wi;
        Mat A;
        Mat B;
        RotMat_up* R_up;
        void coriolis();
        void curl(Vec u);
        void diagnose_F();
        void diagnose_Phi();
        void diagnose_T();
        void diagnose_q(Vec _ug, Vec _h, Vec* qi);
        void diagnose_ds(bool do_damping, double _dt, Vec _ul);
        void rhs_u(Vec uo, bool do_damping, double _dt, Vec _uhl);
        void rhs_h(Vec ho, double _dt);
        void rhs_s(Vec so, bool do_damping, double _dt, Vec _uhl);
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
        void writeConservation(double time, double mass0, double vort0, double ener0, double enst0, double buoy0);
        void assemble_operator(double dt);
        void solve(double _dt, bool save, int nits);
        void solve_rk2(double _dt, bool save);
        void unpack(Vec x, Vec u, Vec h, Vec s);
        void repack(Vec x, Vec u, Vec h, Vec s);
	////
        void diagnose_F_inst(Vec _u, Vec _h, Vec _F);
        void diagnose_Phi_inst(Vec _ul, Vec _ug, Vec _h, Vec _s, Vec _Phi);
        void diagnose_T_inst(Vec _h, Vec _T);
        void diagnose_ds_inst(Vec _h, Vec _s, Vec _ds);
        void rhs_u_inst(Vec _u, Vec _h, Vec _F, Vec _Phi, Vec _T, Vec _ds, Vec _q, Vec _ql, Vec _qil, Vec _ul, Vec _fu);
        void rhs_s_inst(Vec _F, Vec _dsl, Vec _s, Vec _fs);
        void solve_ssp_rk2(double _dt, bool save);

	double del2;
        double viscosity();
        void grad(Vec phi, Vec* _u);
        void laplacian(Vec _u, Vec* ddu);
};
