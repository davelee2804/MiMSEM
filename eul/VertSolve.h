typedef double (ICfunc3D) (double* xi, int ki);

class VertSolve {
    public:
        VertSolve(Topo* _topo, Geom* _geom, double _dt);
        ~VertSolve();
        double dt;
        double visc;
        int rank;
        int step;
        double k2i_z;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        Vec* gv;                 // gravity vector
        Vec* zv;                 // level height vector
        VertOps* vo;
        L2Vecs* theta_h;
        L2Vecs* exner_h;
        HorizSolve* horiz;

        void initGZ();
        void viscosity();

        void diagTheta2(Vec* rho, Vec* rt, Vec* theta);
        void diagTheta_up(Vec* rho, Vec* rt, Vec* theta, Vec* ul);
        void diagExner(int ex, int ey, Vec rt, Vec pi);

        void diagnose_F_z(int ex, int ey, Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec _F);
        void diagnose_Phi_z(int ex, int ey, Vec velz1, Vec velz2, Vec Phi);

        void solve_coupled(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i);
        void solve_schur(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, L2Vecs* udwdx
, double del2_x, Umat* M1, Wmat* M2, E21mat* EtoF, KSP ksp_x, L2Vecs* F_rho_o, L2Vecs* F_rt_o);
        void solve_schur_2(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, 
                              L2Vecs* udwdx, Vec* velx1, Vec* velx2, Vec* u1l, Vec* u2l, bool hs_forcing);
        void solve_schur_3(L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, 
                              L2Vecs* udwdx, Vec* velx1, Vec* velx2, Vec* u1l, Vec* u2l, bool hs_forcing,
                              L2Vecs* velz_j, L2Vecs* rho_j, L2Vecs* rt_j, L2Vecs* exner_j);
        void assemble_operator(int ex, int ey, Vec velz, Vec theta, Vec rho, Vec rt, Vec exner, Mat* _PC);
        void assemble_operator_schur(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, 
                                     Vec F_w, Vec F_rho, Vec F_rt, Vec F_exner, Vec dw, Vec drho, Vec drt, Vec dexner);

        void assemble_residual(int ex, int ey, Vec theta, Vec Pi, 
                               Vec velz1, Vec velz2, Vec rho1, Vec rho2, Vec rt1, Vec rt2, Vec fw, Vec _F, Vec _G);

        void repack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner);
        void unpack_z(Vec x, Vec u, Vec rho, Vec rt, Vec exner);

        void assemble_pc(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec exner, bool eos_update);

        double MaxNorm(Vec dx, Vec x, double max_norm);

        void solve_schur_column_3(int ex, int ey, Vec theta, Vec velz, Vec rho, Vec rt, Vec pi, 
                                   Vec F_u, Vec F_rho, Vec F_rt, Vec F_pi, Vec d_u, Vec d_rho, Vec d_rt, Vec d_pi, int ii);

        void AssembleVertMomVort(Vec* ul, L2Vecs* velz, KSP ksp1, Umat* M1, Wmat* M2, E21mat* EtoF, WtQdUdz_mat* Rz, L2Vecs* uuz);

        void solve_schur_vert(L2Vecs* velz_i, L2Vecs* velz_j, L2Vecs* velz_h, L2Vecs* rho_i, L2Vecs* rho_j, L2Vecs* rho_h, 
			      L2Vecs* rt_i, L2Vecs* rt_j, L2Vecs* rt_h, L2Vecs* exner_i, L2Vecs* exner_j, L2Vecs* _exner_h, 
                              L2Vecs* _theta_h, L2Vecs* udwdx, Vec* velx1, Vec* velx2, Vec* u1l, Vec* u2l, bool hs_forcing);
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
        Mat Q_rt_rho_M_rho_inv_D_rho;
        Mat VAB;

        KSP ksp_pi, ksp_rho, ksp_w;

        void assemble_operators(int ex, int ey, Vec theta, Vec rho, Vec rt, Vec pi, Vec velz);
    //private:
        // vertical vectors and matrices
        Vec _Phi_z;
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
