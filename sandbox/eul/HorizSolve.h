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
        bool firstStep;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        VertOps* vo;
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

        void solve_schur(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i);

        void assemble_schur(int lev, Vec* theta, Vec velx, Vec rho, Vec rt, Vec exner, 
                            Vec F_u, Vec F_rho, Vec F_rt, Vec F_exner, Vec du, Vec drho, Vec drt, Vec dexner);

        void assemble_residual_x(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi,
                                 Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec fu, Vec _F, Vec _G);

        void coriolisMatInv(Mat A, Mat* Ainv);

        double MaxNorm(Vec dx, Vec x, double max_norm);

    private:
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
        // ..... rho corrections
        Mat pcx_D_rho;
        Mat pcx_D_prime;
        Mat pcx_A_rtM2_inv;
        Mat pcx_M1invF_rho;
        Mat pcx_D21M1invF_rho;
        Mat pcx_M1_exner_M1_inv;
        Mat pcx_Au;
        Mat pcx_Au_M2_inv;
        Mat pcx_Mu_prime;
        // ..... temperature equation viscosity 
        Mat pcx_M2_invM2;
        Mat pcx_M2M2_invM2;
        Mat pcx_DT_M2M2_invM2;
        Mat pcx_M1_invDT_M2M2_invM2;
        Mat pcx_M1_rhoM1_invDT_M2M2_invM2;
        Mat pcx_M1_invM1_rhoM1_invDT_M2M2_invM2;
        Mat pcx_LAP_Theta;
        Mat pcx_M2_LAP_Theta;
        Mat pcx_DT_LAP_Theta;
        Mat pcx_M1_invDT_LAP_Theta;
        Mat pcx_D_M1_invDT_LAP_Theta;
        Mat pcx_LAP2_Theta;

        Mat _PCx;
};
