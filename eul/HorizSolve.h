typedef double (ICfunc3D) (double* xi, int ki);

class HorizSolve {
    public:
        HorizSolve(Topo* _topo, Geom* _geom, double _dt);
        ~HorizSolve();
        double dt;
        double del2;
        double k2i;
        bool do_visc;
        int rank;
        int size;
        int step;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        VertOps* vo;
        Pvec* m0;
        Phvec* m0h;
        Umat* M1;
        Wmat* M2;
        E10mat* NtoE;
        E21mat* EtoF;
        RotMat* R;
        Uhmat* F;
        WtQUmat* K;
        Ut_mat* M1t;
        UtQWmat* Rh;
        Uvec* m1;
        Wvec* m2;
        Vec* fg;                 // coriolis vector (global)
        Vec* fl;                 // coriolis vector (local)
        Vec* Fk;
        KSP ksp1;

        double viscosity();
        void coriolis();
        void grad(bool assemble, Vec phi, Vec* u, int lev); // weak form grad operator
        void curl(bool assemble, Vec u, Vec* w, int lev, bool add_f, Vec ul);      // weak form curl operator
        void laplacian(bool assemble, Vec u, Vec* ddu, int lev, Vec ul);       // laplacian operator via helmholtz decomposition

        void diagnose_Pi(int level, Vec rt1, Vec rt2, Vec Pi);

        void diagnose_fluxes(int level, Vec u1, Vec u2, Vec h1l, Vec h2l, Vec* theta_l, Vec _F, Vec _G, Vec u1l, Vec u2l);
        void advection_rhs(Vec* u1, Vec* u2, Vec* h1l, Vec* h2l, L2Vecs* theta, L2Vecs* dF, L2Vecs* dG, Vec* u1l, Vec* u2l, bool do_temp_visc);

        void diagnose_Phi(int level, Vec u1, Vec u2, Vec u1l, Vec u2l, Vec* Phi);
        void diagnose_q(int level, bool do_assemble, Vec rho, Vec vel, Vec* qi, Vec ul);
        void diagHorizVort(Vec* velx, Vec* dudz);

        void momentum_rhs(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                          Vec velx1, Vec velx2, Vec uil, Vec ujl, Vec rho1, Vec rho2, Vec fu, Vec* Fz);

        // for the semi-implicit horizontal solve
        Whmat* T;
        WmatInv* M2inv;
        WhmatInv* M2_rho_inv;
        N_rt_Inv* N2_rt;
        N_rt_Inv* N2_pi_inv;
        Mat M2D;
        Mat M2DM1_inv;
        Mat M1invDT;
        Mat M1invDTM2;
        Mat KT;
        Mat KTM2_inv;
        Mat G_pi;
        Mat G_rt;
        Mat D_rt;
        Mat D_rho;
        Mat Q_rt_rho;
        Mat M1_inv;
        Mat M_u_inv;
        Mat G_pi_C_pi_inv;
        Mat Q_rt_rho_M_rho_inv;
        Mat D_M_u_inv;
        Mat _D;
        Mat _G;
        Mat _PCx;
        KSP ksp_rt;
        KSP ksp_u;
        KSP kspColA2;
        void coriolisMatInv(Mat A, Mat* Ainv, MatReuse reuse);
        void diagTheta2(Vec* rho, Vec* rt, L2Vecs* theta);
        double MaxNorm(Vec dx, Vec x, double max_norm);
        void assemble_residual(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi,
                               Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec fu, Vec _F, Vec _G, Vec uil, Vec ujl, Vec grad_pi);
        void solve_schur(Vec* velx, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i);
        void solve_schur_level(int lev, Vec* theta, Vec velx_l, Vec velx_g, Vec rho, Vec rt, Vec pi, 
                               Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi, Vec dpil);
};
