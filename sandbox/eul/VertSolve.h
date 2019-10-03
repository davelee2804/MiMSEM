typedef double (ICfunc3D) (double* xi, int ki);

class VertSolve {
    public:
        VertSolve(Topo* _topo, Geom* _geom, double _dt);
        ~VertSolve();
        double dt;
        int step;
        int rank;
        bool firstStep;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        Vec* gv;                 // gravity vector
        Vec* zv;                 // level height vector
        VertOps* vo;
        KSP ksp_exner;
        KSP ksp_w;

        void initGZ();

        void diagTheta2(Vec* rho, Vec* rt, Vec* theta);

        void diagnose_F_z(int ex, int ey, Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec _F);
        void diagnose_Phi_z(int ex, int ey, Vec velz1, Vec velz2, Vec Phi);

        void solve_coupled(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i);
        void solve_schur(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i);
        void assemble_operator(int ex, int ey, Vec velz, Vec theta, Vec rho, Vec rt, Vec exner, Mat* _PC);
        void assemble_operator_schur(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, 
                                     Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner);

        void assemble_residual_z(int ex, int ey, Vec theta, Vec Pi, 
                                 Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fw, Vec _F, Vec _G);

        void repack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner);
        void unpack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner);

        void assemble_and_update(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner, bool eos_update);
        void det_deltas(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner,
                        Vec F_w, Vec F_rho, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner);

        double MaxNorm(Vec dx, Vec x, double max_norm);

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
        // .....density correction (velocity equation)
        Mat pc_V0_invV0_rt_DT;
        //Mat pc_V0_thetaV0_invV0_exner;
        //Mat pc_V0_thetaV0_invV0_exnerV0_inv;
        //Mat pc_V0_thetaV0_invV0_exnerV0_invDT;
        //Mat pc_A_u_2;

        Mat _PCz;
};
