typedef double (ICfunc) (double* xi);

class SWEqn {
    public:
        SWEqn(Topo* _topo, Geom* _geom);
        ~SWEqn();
        double grav;
        double omega;
        double del2;
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
        void diagnose_w(Vec u, Vec* w, bool add_f);
        void diagnose_F(Vec u, Vec h, KSP ksp, Vec* hu);
        void init0(Vec q, ICfunc* func);
        void init1(Vec u, ICfunc* func_x, ICfunc* func_y);
        void init2(Vec h, ICfunc* func);
        double err0(Vec u, ICfunc* fw, ICfunc* fu, ICfunc* fv);
        double err1(Vec u, ICfunc* fu, ICfunc* fv, ICfunc* fp);
        double err2(Vec u, ICfunc* fu);
        double int0(Vec u);
        double int2(Vec u);
        double intE(Vec u, Vec h);
        void solve_RK2(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save);
        void solve_RK2_SS(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save);
        void solve_EEC(Vec ui, Vec hi, Vec uf, Vec hf, double dt, bool save);
        void laplacian(Vec ui, KSP ksp, Vec* ddu);
        void writeConservation(double time, Vec ui, Vec hi, double mass0, double vort0, double ener0);
    private:
        double viscosity();
        void _massEuler(Vec ui, Vec hi, Vec uj, Vec hj, Vec hf, KSP ksp1, KSP ksp2, double dt);
        void _momentumEuler(Vec ui, Vec hi, Vec uj, Vec hj, Vec hf, KSP ksp, double dt);
        void _massEqn(Vec hi, Vec uj, Vec hj, Vec hf, KSP ksp, double dt);
        void _momentumEqn(Vec ui, Vec uj, Vec hj, Vec uf, KSP ksp, double dt);
        void _massTend(Vec ui, Vec hi, KSP ksp, Vec *Fh);
        void _momentumTend(Vec ui, Vec hi, KSP ksp, Vec *Fh);
};
