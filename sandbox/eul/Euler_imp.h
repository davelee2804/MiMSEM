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
        Vec theta_b;             // bottom potential temperature bc
        Vec theta_t;             // top potential temperature bc
        Vec theta_b_l;           // bottom potential temperature bc
        Vec theta_t_l;           // top potential temperature bc

        double viscosity();
        void coriolis();
        void initGZ();
        void grad(bool assemble, Vec phi, Vec* u, int lev);            // weak form grad operator
        void curl(bool assemble, Vec u, Vec* w, int lev, bool add_f);  // weak form curl operator
        void laplacian(bool assemble, Vec u, Vec* ddu, int lev);       // laplacian operator via helmholtz decomposition

        void thetaBCVec(int ex, int ey, Mat A, Vec bTheta);
        void diagTheta(Vec* rho, Vec* rt, L2Vecs* theta);
        void diagHorizVort(Vec* velx, Vec* dudz);
        void diagnose_Pi(int level, Vec rt1, Vec rt2, Vec Pi);

        void diagnose_F_x(int level, Vec u1, Vec u2, Vec h1, Vec h2, Vec _F);
        void diagnose_Phi_x(int level, Vec u1, Vec u2, Vec* Phi);
        void diagnose_wxu(int level, Vec u1, Vec u2, Vec* wxu);

        void diagnose_F_z(int ex, int ey, Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec _F);
        void diagnose_Phi_z(int ex, int ey, Vec velz1, Vec velz2, Vec Phi);

        void solve(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, bool save);
        void solve_strang(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, bool save);
        void solve_vert(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, bool save);
        void solve_unsplit(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, bool save);
        void assemble_precon_z(int ex, int ey, Vec theta, Vec rt_i, Vec rt_j, Vec exner, Vec velz);
        void assemble_precon_x(int level, Vec* theta, Vec rt_i, Vec rt_j, Vec exner, Mat* _PC);

        void assemble_residual_x(int level, Vec* theta1, Vec* theta2, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi,
                                 Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fu, Vec _F, Vec _G);
        void assemble_residual_z(int ex, int ey, Vec theta1, Vec theta2, Vec Pi, 
                                 Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fw, Vec _F, Vec _G);

        void init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* h, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);

        void assemble_residual_u(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                                 Vec velx1, Vec velx2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fu);
        void assemble_residual_w(int ex, int ey, Vec theta, Vec Pi, Vec velz1, Vec velz2, Vec fw);

        double integrateTheta(Vec theta);
        void coriolisMatInv(Mat A, Mat* Ainv);
        void assemblePreconTheta(L2Vecs* theta, L2Vecs* rt, Vec* velx, Vec* velz);

        void repack_z(Vec x, Vec u, Vec rho, Vec rt);
        void unpack_z(Vec x, Vec u, Vec rho, Vec rt);
        void assemble_operator(int ex, int ey, Vec theta, Vec rho, Vec rt);

    private:
        // vertical vectors and matrices
        Vec _Phi_z;
        Vec _theta_h;
        Vec _tmpA1;
        Vec _tmpA2;
        Vec _tmpB1;
        Vec _tmpB2;
        Mat _DTV1;
        Mat _GRAD;
        Mat _DIV;
        Mat _V0_invDTV1;
        Mat _V0_invV0_rt;
        Mat _DV0_invV0_rt;
        Mat _V1_PiDV0_invV0_rt;
        Mat _V0_thetaV0_invDTV1;
        Mat _V0_invV0_thetaV0_invDTV1;
        Mat _DV0_invV0_thetaV0_invDTV1;
        Mat* PCz;
        // horiztonal vectors and matrices
        Mat _M1invM1;
        Mat _DM1invM1;
        Mat _PiDM1invM1;
        Mat _ThetaPiDM1invM1;
        Mat _M2ThetaPiDM1invM1;
        Mat _DM2ThetaPiDM1invM1;
        Mat _M1DM2ThetaPiDM1invM1;
        Mat* PCx;

        Mat _PCz, _Muu, _Muh, _Mhu, _Mhh;

        Mat _DTM2;
        Mat _M1invDTM2;
        Mat _M1thetaM1invDTM2;
        Mat _M1invM1thetaM1invDTM2;
        Mat _DM1invM1thetaM1invDTM2;
        Mat _KDT;
};
