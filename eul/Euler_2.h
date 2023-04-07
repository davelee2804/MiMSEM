typedef double (ICfunc3D) (double* xi, int ki);

class Euler {
    public:
        Euler(Topo* _topo, Geom* _geom, double _dt);
        ~Euler();
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
        Pmat* M0;
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
        bool firstStep;
        double k2i;              // kinetic to internal energy exchange
        double i2k;              // kinetic to internal energy exchange
        double k2i_z;            // kinetic to internal energy exchange
        double i2k_z;            // kinetic to internal energy exchange
        Vec* Kh;                 // kinetic energy vector for each horiztontal layer
        Vec* uz;                 // dudz and dvdz vorticity components
        Vec* uzl;
        Vec* uzl_prev;
        Vec* ul;
        Vec* ul_prev;
        Vec* u_curr;
        Vec* u_prev;
        L2Vecs* uuz;             // u.dudz + v.dvdz vorticity velocity product
        Mat VA;
        Mat VB;
        KSP ksp1;
        KSP ksp2;
        KSP kspColA2; // for the diagnosis of theta without boundary conditions

        VertSolve* vert;

        Wii* Q;
        M2_j_xy_i* W;
        double* Q0;
        double* Wt;
        double* WtQ;

        void diagTheta(Vec* rho, Vec* rt, Vec* theta);
        void AssembleKEVecs(Vec* velx);
        void init1(Vec* u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* p, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);
        void Trapazoidal(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save);
        void Strang(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save);
        void Strang_ec(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner, bool save);

        double int2(Vec ug);
        void diagnostics(Vec* velx, Vec* velz, Vec* rho, Vec* rt, Vec* exner);

        void HorizVort(Vec* velx);
        void HorizPotVort(Vec* velx, Vec* rho);
        void AssembleVertMomVort(L2Vecs* velz);
        void VertMassFlux(L2Vecs* velz1, L2Vecs* velz2, L2Vecs* rho1, L2Vecs* rho2, L2Vecs* Fz);
};
