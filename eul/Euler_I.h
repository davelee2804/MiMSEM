typedef double (ICfunc3D) (double* xi, int ki);

class Euler_I {
    public:
        Euler_I(Topo* _topo, Geom* _geom, double _dt);
        ~Euler_I();
        double dt;
        bool hs_forcing;
        int rank;
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
        Ut_mat* M1t;
        UtQWmat* Rh;
        WtQdUdz_mat* Rz;
        Whmat* T;
        Mat KT;
        double k2i;              // kinetic to internal energy exchange
        double i2k;              // kinetic to internal energy exchange
        double k2i_z;            // kinetic to internal energy exchange
        double i2k_z;            // kinetic to internal energy exchange
        Vec* uz;                 // dudz and dvdz vorticity components
        Vec* uzl_i;
        Vec* uzl_j;
        Vec* uil;
        Vec* ujl;
        Vec* uhl;
        L2Vecs* uuz;             // u.dudz + v.dvdz vorticity velocity product
        Mat VA;
        Mat VB;
        KSP ksp1;
        KSP ksp2;

	Vec x;
	Vec dx;
	Vec b;
        Mat M; // global coupled matrix
	Umat_coupled*   M1c;
	RotMat_coupled* Rc;
	Wmat_coupled*   M2c;
	EoSmat_coupled* EoSc;
	Mat*            M1inv;
	Mat             GRADx;
	Mat             M1invGRADx;
	Mat             M1invM1;
	Mat             DM1invM1;
	Mat             Gpi;
	Mat             Grt;
	Mat             Dx;
	Mat             Qx;
	KSP             ksp_c;
	M2mat_coupled*  CM2;
	M3mat_coupled*  CM3;
	Kmat_coupled*   CK;
	E32_Coupled*    CE32;
	Mat             CM2inv;
        Mat             CE23M3;
        Mat             CM2invE23M3;
	Mat             CGRAD1;
	Mat             CGRAD2;
        Mat             CM2invM2;
        Mat             CM3invM3;
        Mat             CE32M2invM2;
        Mat             CDIV1;
        Mat             CDIV2;
        Mat             CQ;

        VertSolve* vert;

        Wii* Q;
        M2_j_xy_i* W;
        double* Q0;
        double* Wt;
        double* WtQ;

        void grad(bool assemble, Vec phi, Vec u, int lev);            // weak form grad operator
        void init0(Vec* q, ICfunc3D* func);
        void init1(Vec* u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* p, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);

        double int2(Vec ug);
        void diagnostics(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner);

        void HorizPotVort(Vec* velx, Vec* rho, Vec* uzl);
        void AssembleVertMomVort(Vec* ul, L2Vecs* velz);
        void VertMassFlux(L2Vecs* velz1, L2Vecs* velz2, L2Vecs* rho1, L2Vecs* rho2, L2Vecs* Fz);

	void CreateCoupledOperator();
        void AssembleCoupledOperator(L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* velz, L2Vecs* theta);

        void AssembleResidual(Vec* velx_i, Vec* velx_j,
                              L2Vecs* rho_i, L2Vecs* rho_j,
                              L2Vecs* rt_i, L2Vecs* rt_j,
                              L2Vecs* exner_i, L2Vecs* exner_j, L2Vecs* exner_h,
                              L2Vecs* velz_i, L2Vecs* velz_j, L2Vecs* theta_i, L2Vecs* theta_h,
                              L2Vecs* Fz, L2Vecs* dFx, L2Vecs* dGx, Vec* dwdx_i, Vec* dwdx_j, 
                              Vec* R_u, Vec* R_rho, Vec* R_rt, Vec* R_pi, Vec* R_w);

        void Solve(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save);
        void Solve_SNES(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save);
};

typedef struct {
    Euler_I* eul;
    Vec*     velx_i;
    Vec*     velx_j;
    Vec*     uil;
    Vec*     ujl;
    L2Vecs*  rho_i;
    L2Vecs*  rho_j;
    L2Vecs*  rho_h;
    L2Vecs*  rt_i;
    L2Vecs*  rt_j;
    L2Vecs*  rt_h;
    L2Vecs*  exner_i;
    L2Vecs*  exner_j;
    L2Vecs*  exner_h;
    L2Vecs*  velz_i;
    L2Vecs*  velz_j;
    L2Vecs*  velz_h;
    L2Vecs*  theta_i;
    L2Vecs*  theta_h;
    L2Vecs*  Fz;
    L2Vecs*  dFx;
    L2Vecs*  dGx;
    Vec*     dwdx_i;
    Vec*     dwdx_j;
    Vec*     R_u;
    Vec*     R_rho;
    Vec*     R_rt;
    Vec*     R_exner;
    Vec*     R_w;
} euler_ctx;
