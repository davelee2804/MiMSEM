typedef double (ICfunc3D) (double* xi, int ki);

class Euler {
    public:
        Euler(Topo* _topo, Geom* _geom, double _dt);
        ~Euler();
        double dt;
        double del2;
        bool do_visc;
        int rank;
        int step;
        bool firstStep;
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
        EoSvec* eos;
        EoSmat* eos_mat;
        Ut_mat* M1t;
        UtQWmat* Rh;
        Vec* fg;                 // coriolis vector (global)
        Vec* fl;                 // coriolis vector (local)
        Vec* gv;                 // gravity vector
        Vec* zv;                 // level height vector
        VertOps* vo;
        KSP ksp1;
        KSP ksp2;
        KSP ksp_exner;
        KSP ksp_exner_x;
        KSP ksp_u;

        double viscosity();
        void coriolis();
        void initGZ();
        void grad(bool assemble, Vec phi, Vec* u, int lev);            // weak form grad operator
        void curl(bool assemble, Vec u, Vec* w, int lev, bool add_f);  // weak form curl operator
        void laplacian(bool assemble, Vec u, Vec* ddu, int lev);       // laplacian operator via helmholtz decomposition

        void diagTheta2(Vec* rho, Vec* rt, Vec* theta);
        void diagHorizVort(Vec* velx, Vec* dudz);
        void diagnose_Pi(int level, Vec rt1, Vec rt2, Vec Pi);

        void diagnose_F_x(int level, Vec u1, Vec u2, Vec h1, Vec h2, Vec _F);
        void diagnose_Phi_x(int level, Vec u1, Vec u2, Vec* Phi);
        void diagnose_wxu(int level, Vec u1, Vec u2, Vec* wxu);

        void diagnose_F_z(int ex, int ey, Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec _F);
        void diagnose_Phi_z(int ex, int ey, Vec velz1, Vec velz2, Vec Phi);

        void dump(Vec* velx, L2Vecs* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* theta, int num);

        void solve_vert_coupled(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save);
        void solve_vert_schur(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save);
        void solve_horiz_schur(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save);
        void solve_schur(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save);
        void assemble_operator(int ex, int ey, Vec velz, Vec theta, Vec rho, Vec rt, Vec exner, Mat* _PC);
        void assemble_operator_schur(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, 
                                     Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner);

        void assemble_schur_horiz(int lev, Vec* theta, Vec velx, Vec rho, Vec rt, Vec exner, 
                               Vec F_u, Vec F_rho, Vec F_rt, Vec F_exner, Vec du, Vec drho, Vec drt, Vec dexner);

        void assemble_residual_x(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi,
                                 Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec fu, Vec _F, Vec _G);
        void assemble_residual_z(int ex, int ey, Vec theta, Vec Pi, 
                                 Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fw, Vec _F, Vec _G);

        void init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* h, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);

        void coriolisMatInv(Mat A, Mat* Ainv);

        void repack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner);
        void unpack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner);

    private:
        // vertical vectors and matrices
        Vec _Phi_z;
        Vec _theta_h;
        Vec _tmpA1;
        Vec _tmpA2;
        Vec _tmpB1;
        Vec _tmpB2;
        Mat _V0_invV0_rt;
        // ...coupled preconditioner
        Mat pc_DTV1;
        Mat pc_V0_invDTV1;
        Mat pc_GRAD;
        Mat pc_V0_invV0_rt;
        Mat pc_DV0_invV0_rt;
        Mat pc_V1DV0_invV0_rt;
        Mat pc_V0_invV01;
        Mat pc_DV0_invV01;
        Mat pc_V1DV0_invV01;
        Mat pc_VB_rt_invVB_pi;
        Mat pc_VBVB_rt_invVB_pi;
        // .....schur preconditioner
        Mat pc_G;
        Mat pc_A_u;
        Mat pc_A_rt;
        Mat pc_D_rho;
        Mat pc_D_rt;
        Mat pc_M_u;
        Mat pc_M_u_inv;
        Mat pc_N_exner;
        Mat pc_N_exner_2;
        Mat pc_N_rt_inv;
        Mat pc_D_rt_M_u_inv;
        Mat pc_VB_N_rt_inv;
        Mat pc_A_u_VB_inv;
        Mat pc_A_rt_VB_inv;
        Mat pc_A_rt_VB_inv_D_rho;
        Mat pc_G_VB_rho_inv;
        Mat pc_M_rt;
        Mat pc_M_rt_inv;
        Mat pc_M_rt_VB_inv;

        Mat pc_V0_invV0_rt_DT;
        Mat pc_V0_invV0_rt_DT_VB_pi;
        Mat pc_V0_invV0_rt_DT_VB_pi_VB_inv;
        // ..... schur preconditioner (horizontal)
        Mat pcx_D;
        Mat pcx_G;
        Mat pcx_M1invD12;
        Mat pcx_M1invD12M2;
        Mat pcx_D_Mu_inv;
        Mat pcx_M1invF_rt;
        Mat pcx_D21M1invF_rt;
        Mat pcx_M2N_rt_inv;
        Mat pcx_M2N_rt_invN_pi;
        Mat pcx_LAP;
        // .....rho corrections
        Mat pcx_D_rho;
        Mat pcx_D_prime;
        Mat pcx_A_rtM2_inv;
        Mat pcx_M1invF_rho;
        Mat pcx_D21M1invF_rho;
        Mat pcx_M1_exner_M1_inv;
        Mat pcx_Au;
        Mat pcx_Au_M2_inv;
        Mat pcx_Mu_prime;

        Mat _PCz;
        Mat _PCx;
};
