typedef double (ICfunc3D) (double* xi, int ki);

class Euler_I {
    public:
        Euler_I(Topo* _topo, Geom* _geom, double _dt);
        ~Euler_I();
        double dt;
        double del2;
        bool do_visc;
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
        WmatInv* M2inv;
        Umat_ray* M1ray;
        Mat KT;
        Vec* fg;                 // coriolis vector (global)
        double k2i;              // kinetic to internal energy exchange
        double i2k;              // kinetic to internal energy exchange
        double k2i_z;            // kinetic to internal energy exchange
        double i2k_z;            // kinetic to internal energy exchange
        Vec* Kh;                 // kinetic energy vector for each horiztontal layer
        Vec* gv;
        Vec* zv;
        Vec* uz;                 // dudz and dvdz vorticity components
        Vec* uzl;
        Vec* uil;
        Vec* ujl;
        Vec* uhl;
        L2Vecs* uuz;             // u.dudz + v.dvdz vorticity velocity product
        Mat VA;
        Mat VB;
        KSP ksp1;
        KSP ksp2;
        KSP kspColA2; // for the diagnosis of theta without boundary conditions

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
	Mat             Gx;
	Mat             Dx;
	Mat             Qx;

        VertSolve* vert;

        Wii* Q;
        M2_j_xy_i* W;
        double* Q0;
        double* Wt;
        double* WtQ;

        double viscosity();
        void coriolis();
        void initGZ();
        void grad(bool assemble, Vec phi, Vec u, int lev);            // weak form grad operator
        void curl(bool assemble, Vec u, Vec* w, int lev, bool add_f); // weak form curl operator
        void laplacian(bool assemble, Vec u, Vec* ddu, int lev);      // laplacian operator via helmholtz decomposition
        void massRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* Flux);
        void tempRHS(Vec* uh, Vec* pi, Vec* Fp, Vec* rho_l, Vec* exner);
        void horizMomRHS(Vec ui, Vec* theta, Vec exner, int lev, Vec Fu, Vec Flux, Vec uzb, Vec uzt, Vec velz_b, Vec velz_t);
        void thetaBCVec(int ex, int ey, Mat A, Vec* bTheta);
        void diagTheta(Vec* rho, Vec* rt, Vec* theta);
        void diagTheta_av(Vec* rho, L2Vecs* rt, Vec* theta, L2Vecs* rhs, Vec* ul);
        void AssembleKEVecs(Vec* ul, Vec* velx);
        void init0(Vec* q, ICfunc3D* func);
        void init1(Vec* u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* p, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);
        void HorizRHS(Vec* velx, L2Vecs* rho, L2Vecs* rt, Vec* exner, Vec* Fu, Vec* Fp, Vec* Ft, Vec* velz);
        void SolveExner(Vec* rt, Vec* Ft, Vec* exner_i, Vec* exner_f, double _dt);

        double int2(Vec ug);
        void diagnostics(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner);

        void DiagExner(Vec* rtz, L2Vecs* exner);

        void HorizVort(Vec* velx);
        void HorizPotVort(Vec* velx, Vec* rho);
        void AssembleVertMomVort(Vec* ul, L2Vecs* velz);
        void VertMassFlux(L2Vecs* velz1, L2Vecs* velz2, L2Vecs* rho1, L2Vecs* rho2, L2Vecs* Fz);

	void CreateCoupledOperator();
	void AssembleCoupledOperator(Vec* rho_x, Vec* rt_x, Vec* exner_x, Vec* theta_x, 
			             Vec* rho_z, Vec* rt_z, Vec* exner_z, Vec* theta_z);
};
