typedef double (ICfunc) (double* xi);

class SWEqn {
    public:
        SWEqn(Topo* _topo, Geom* _geom);
        ~SWEqn();
        double grav;
        double omega;
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
        Vec fg;            // coriolis vector (global)
        Mat E01M1;
        Mat E12M2;
        void coriolis();
        void diagnose_w(Vec u, Vec* w);
        void diagnose_F(Vec u, Vec h, KSP ksp, Vec* hu);
        void solve(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save);
        void init0(Vec q, ICfunc* func);
        void init1(Vec u, ICfunc* func_x, ICfunc* func_y);
        void init2(Vec h, ICfunc* func);
        double err0(Vec u, ICfunc* fw, ICfunc* fu, ICfunc* fv);
        double err1(Vec u, ICfunc* fu, ICfunc* fv, ICfunc* fp);
        double err2(Vec u, ICfunc* fu);
        double int0(Vec u);
        double int2(Vec u);
        double intE(Vec u, Vec h);
        void solve_EEC(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save);
        void massEuler(Vec ui, Vec hi, Vec uj, Vec hj, Vec hf, KSP ksp, double dt);
        void momentumEuler(Vec ui, Vec hi, Vec uj, Vec hj, Vec hf, KSP ksp, double dt);
};
