typedef double (ICfunc3D) (double* xi, int ki);

class Euler {
    public:
        Euler(Topo* _topo, Geom* _geom, double _dt);
        ~Euler();
        double dt;
        double del2;
        double vert_visc;
        bool do_visc;
        int step;
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
        Vec* fg;                                     // coriolis vector (global)
        bool firstStep;
        double k2i;                                  // kinetic to internal energy exchange
        double i2k;                                  // kinetic to internal energy exchange
        double k2i_z;                                // kinetic to internal energy exchange
        double i2k_z;                                // kinetic to internal energy exchange
        Vec theta_b;                                 // bottom potential temperature bc
        Vec theta_t;                                 // top potential temperature bc
        Vec theta_b_l;                               // bottom potential temperature bc
        Vec theta_t_l;                               // top potential temperature bc
        Vec* Kv;                                     // kinetic energy vector for each vertical column
        Vec* Kh;                                     // kinetic energy vector for each horiztontal layer
        Vec* gv;
        Vec* zv;
        Mat V01;                                     // vertical divergence operator
        Mat V10;                                     // vertical gradient operator
        Mat VA;
        Mat VB;
        KSP ksp1;
        KSP ksp2;
        KSP kspE;
        KSP kspColA;

        Wii* Q;
        M2_j_xy_i* W;
        double** Q0;
        double** QT;
        double** QB;
        double** Wt;
        double** WtQ;
        double** WtQW;
        double** WtQWinv;
        double* WtQWflat;

        double viscosity();
        double viscosity_vert();
        void coriolis();
        void vertOps();
        void initGZ();
        void grad(bool assemble, Vec phi, Vec* u, int lev);            // weak form grad operator
        void curl(bool assemble, Vec u, Vec* w, int lev, bool add_f);  // weak form curl operator
        void laplacian(bool assemble, Vec u, Vec* ddu, int lev);       // laplacian operator via helmholtz decomposition
        void massRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* Flux);
        void tempRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* exner);
        void horizMomRHS(Vec ui, Vec* theta, Vec exner, int lev, Vec Fu, Vec Flux);
        void thetaBCVec(int ex, int ey, Mat A, Vec* bTheta);
        void diagTheta(Vec* rho, Vec* rt, Vec* theta);
        void diagThetaVert(int ex, int ey, Mat AB, Vec rho, Vec rt, Vec theta);
        void AssembleKEVecs(Vec* velx, Vec* velz);
        void VertToHoriz2(int ex, int ey, int ki, int kf, Vec pv, Vec* ph);
        void HorizToVert2(int ex, int ey, Vec* ph, Vec pv);
        void init0(Vec* q, ICfunc3D* func);
        void init1(Vec* u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* p, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);
        void HorizRHS(Vec* velx, Vec* rho, Vec* rt, Vec* exner, Vec* Fu, Vec* Fp, Vec* Ft);
        void SolveStrang(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save);

        void AssembleConst(int ex, int ey, Mat A);      // piecewise constant (in vertical) mass matrix
        void AssembleLinear(int ex, int ey, Mat B);     // piecewise linear (in vertical) mass matrix
        void AssembleLinCon(int ex, int ey, Mat AB);
        void AssembleLinearWithTheta(int ex, int ey, Vec theta, Mat A);
        void AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A, bool do_internal);
        void AssembleLinearWithRT(int ex, int ey, Vec rt, Mat A, bool do_internal);
        void AssembleLinearInv(int ex, int ey, Mat A);
        void AssembleConstWithRhoInv(int ex, int ey, Vec theta, Mat B);
        void AssembleConstWithRho(int ex, int ey, Vec rho, Mat A);
        void AssembleConLinWithW(int ex, int ey, Vec velz, Mat BA);

        void diagnostics(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner);

        Mat VBA_w;
        Mat VA_inv;
        Mat VAB;
        Mat VA_theta;
        Mat VA_rho;
        Mat VB_Pi;
        Mat VB_rt;
        Mat VISC;

        Mat DW2;
        Mat DTVB;
        Mat VA_invDTVB;
        Mat GRAD;
        Mat VA_invVA_rt;
        Mat DVA_invVA_rt;
        Mat VB_PiDVA_invVA_rt;
        Mat DIV;
        Mat LAP;

        Vec pTmp;
        Vec pTmp2;
        Vec wTmp;
        Vec pGrad;
        Vec wRho;
        Vec wTheta;
        Vec dwTheta;
        Vec theta;
        Vec wNew;
        Vec fw;
        Vec bw;
        Vec rhoNew;
        Vec fRho;
        Vec bRho;
        Vec rtNew;
        Vec fRT;
        Vec bRT;
        Vec exnerNew;
        Vec fExner;
        Vec bExner;
        Vec rhoOld;
        Vec rtOld;
        Vec exnerOld;
        Vec eosRhs;
        void SetupVertOps();
        void DestroyVertOps();
        int eX;
        int eY;
        int iT;
        void VertSolve_JFNK(Vec* velz, Vec* rho, Vec* rt, Vec* exner, Vec* velz_n, Vec* rho_n, Vec* rt_n, Vec* exner_n);
        void AssemblePreconditioner(Mat P);
        void Assemble_EOS_RHS(int ex, int ey, Vec rt, Vec eos_rhs);
        void AssembleConstInv(int ex, int ey, Mat B);
        void DiagExner(L2Vecs* rt, L2Vecs* exner);
};