typedef double (ICfunc3D) (double* xi, int ki);

class HorizSolve {
    public:
        HorizSolve(Topo* _topo, Geom* _geom, double _dt);
        ~HorizSolve();
        double dt;
        double del2;
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
        Phvec* mh0;
        Umat* M1;
        Wmat* M2;
        E10mat* NtoE;
        E21mat* EtoF;
        RotMat* R;
        Uhmat* F;
        WtQUmat* K;
        Whmat* T;
        EoSvec* eos;
        EoSmat* eos_mat;
        Ut_mat* M1t;
        UtQWmat* Rh;
        WmatInv* M2inv;
        WhmatInv* M2_rho_inv;
        N_rt_Inv* N2_pi_inv;
        N_rt_Inv* N2_rt;
        Vec* fg;                 // coriolis vector (global)
        Vec* fl;                 // coriolis vector (local)
        Vec* gv;                 // gravity vector
        Vec* zv;                 // level height vector
        KSP ksp1;
        KSP ksp2;
        KSP ksp_rt;
        KSP ksp_u;

        double viscosity();
        void coriolis();
        void initGZ();
        void grad(bool assemble, Vec phi, Vec* u, int lev); // weak form grad operator
        void curl(bool assemble, Vec u, Vec* w, int lev, bool add_f);      // weak form curl operator
        void laplacian(bool assemble, Vec u, Vec* ddu, int lev);       // laplacian operator via helmholtz decomposition

        void diagTheta2(Vec* rho, Vec* rt, Vec* theta);
        void diagHorizVort(Vec* velx, Vec* dudz);
        void diagnose_Pi(int level, Vec rt1, Vec rt2, Vec Pi);

        void diagnose_F(int level, Vec u1, Vec u2, Vec h1, Vec h2, Vec _F);
        void diagnose_Phi(int level, Vec u1, Vec u2, Vec u1l, Vec u2l, Vec* Phi);
        void diagnose_wxu(int level, Vec u1, Vec u2, Vec* wxu);

        double MaxNorm(Vec dx, Vec x, double max_norm);

        void assemble_residual(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi,
                               Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec fu, Vec _F, Vec _G, Vec uil, Vec ujl, Vec grad_pi);

        void coriolisMatInv(Mat A, Mat* Ainv);
        void assemble_biharmonic(int lev, MatReuse reuse, Mat* BVISC);
        void assemble_biharmonic_temp(int lev, Vec rho, MatReuse reuse, Mat* BVISC);

        void solve_schur_level(int lev, Vec* theta, Vec velx_l, Vec velx_g, Vec rho, Vec rt, Vec exner, 
                               Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi, Vec grad_pi);
        void assemble_and_update(int lev, Vec* theta, Vec velx_l, Vec velx_g, Vec rho, Vec rt, Vec exner, 
                                 Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec grad_pi);

        void solve_schur(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i);

        Mat _PCx;

    private:
        Mat M_u, M_u_inv, M0_inv, M1_inv, G_rt, G_pi, D_rho, D_rt, Q_rt_rho;
        Mat M1invDT, M1invDTM2, KT, KTM2_inv, M2D, M2DM1_inv;
        Mat D_rho_M_u_inv, D_rt_M_u_inv, L_rho_rt, L_rho_pi, L_rt_rt, L_rt_pi;
        Mat L_rho_pi_N_pi_inv, L_rt_pi_N_pi_inv, L_rho_pi_N_pi_inv_N_rt, L_rt_pi_N_pi_inv_N_rt, Q_rt_rho_M_rho_inv;
        Mat M2_invM2, CTM1, M0_invCTM1, M1_invDT_M2M2_invM2, M1_rhoM1_invDT_M2M2_invM2, M2_LAP_Theta, DT_LAP_Theta;
        Mat TEMP1;
};
