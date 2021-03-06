typedef double (ICfunc3D) (double* xi, int ki);

class VertSolve {
    public:
        VertSolve(Topo* _topo, Geom* _geom, double _dt);
        ~VertSolve();
        double dt;
        double visc;
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

        void initGZ();
        void viscosity();

        void diagTheta2(Vec* rho, Vec* rt, Vec* theta);
        void diagExner(int ex, int ey, Vec rt, Vec pi);

        void diagnose_F_z(int ex, int ey, Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec _F);
        void diagnose_Phi_z(int ex, int ey, Vec velz1, Vec velz2, Vec Phi);

        void solve_coupled(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i);
        void solve_schur(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i);
        void assemble_operator(int ex, int ey, Vec velz, Vec theta, Vec rho, Vec rt, Vec exner, Mat* _PC);
        void assemble_operator_schur(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, 
                                     Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner);

        void assemble_residual(int ex, int ey, Vec theta, Vec Pi, 
                               Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fw, Vec _F, Vec _G);

        void repack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner);
        void unpack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner);

        void set_deltas(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner,
                        Vec F_w, Vec F_rho, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner, 
                        bool add_delta, bool neg_scale);

        void assemble_pc(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, bool eos_update);

        double MaxNorm(Vec dx, Vec x, double max_norm);

        void solve_schur_column(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi);
        void assemble_and_update(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                 Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Schur* schur);
        void update_deltas(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                           Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi);
        void update_delta_u(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                            Vec F_u, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi);
        void update_delta_u_2(int ex, int ey, Vec velz, Vec rho, Vec rt, Vec pi, 
                                    Vec F_u, Vec F_rho, Vec d_u, Vec d_rho);

        void update_delta_pi(int ex, int ey, Vec rho, Vec rt, Vec pi, Vec F_pi, Vec d_rho, Vec d_rt, Vec d_pi);

        void update_delta_pi_2(int ex, int ey, Vec rt, Vec pi, Vec F_pi, Vec d_rt, Vec d_pi);
        void assemble_and_update_2(int ex, int ey, Vec velz, Vec rho, Vec rt, Vec pi, 
                                    Vec F_u, Vec F_rho);

        void assemble_M_rho(int ex, int ey, Vec velz, Schur* schur);

        double LineSearch(Vec velz_i,  Vec velz_j, Vec d_velz, 
                          Vec rho_i,   Vec rho_j,  Vec d_rho, 
                          Vec rt_i,    Vec rt_j,   Vec d_rt, 
                          Vec pi_i,    Vec pi_j,   Vec d_pi, Vec pi_h,
                          Vec theta_i, Vec theta_h, int ei);

        void assemble_and_update_3(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                    Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec Du, Schur* schur, int itt);
        void update_delta_u_3(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                 Vec F_u, Vec F_rho, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi);

        Mat _PCz;
        Mat pc_LAP;

        Vec _tmpA1;
        Vec _tmpA2;
        Vec _tmpB1;
        Vec _tmpB2;

        Mat M_u_inv, M_rho, M_rt_inv, G_rho, G_pi, D_rho, D_rt, D_rho_M_u_inv, D_rt_M_u_inv, L_rho_rho, L_rho_pi, L_rt_rho, L_rt_pi, N_pi, N_rt;
        Mat N_rt_M_rt_inv, LL_inv, LL_invL, L_rho_rho_inv;
        Mat M_rt, G_rt, N_pi_inv, N_rt_inv, L_pi_rt, L_rt_pi_N_pi_inv, L_rho_rt, L_rt_rt, M_rho_inv, N_pi_invN_rt;

        Mat pc_V1_invV1, pc_V01V1_invV1, pc_V0_invV01V1_invV1, pc_DV0_invV01V1_invV1, pc_V1DV0_invV01V1_invV1, pc_V01VBA, pc_V1D;
        Mat pc_V1D_2, pc_V1_invV1_2, pc_V01V1_invV1_2, pc_V1DV0_invV01V1_invV1_2;
        Mat pc_V1_invV1_3, pc_V1V1_invV1, pc_V1_invV1V1_invV1, pc_V1V1_invV1V1_invV1;
        Mat Q_rt_rho, Q_rt_rho_M_rho_inv, L_rho_pi_N_pi_inv, L_rho_pi_N_pi_inv_N_rt, L_rt_pi_N_pi_inv_N_rt;
        Mat pc_VBtheta_VBinv, pc_VBtheta_VBinv_VBdu;
        Mat UdotGRAD;
        Mat L_rt_pi_N_pi_inv_N_rho, L_rt_pi_N_pi_inv_N_rho_M_inv;
        Mat L_rho_pi_N_pi_inv_N_rho;
        Mat G_pi_N_pi_inv, G_pi_N_pi_inv_N_rt;

        KSP ksp_pi, ksp_rho, ksp_w;

    //private:
        // vertical vectors and matrices
        Vec _Phi_z;
        Vec _theta_h;
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
        Mat pc_VISC;
        // .....density correction (velocity equation)
        Mat pc_V0_invV0_rt_DT;
};
