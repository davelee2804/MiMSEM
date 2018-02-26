class HPEqn {
    public:
        HPEqn(Topo* _topo, Geom* _geom, int _nLevs, double* _pBot, double* _pMid);
        ~HPEqn();
        Topo* topo;
        Geom* geom;
        SWEqn* sw;
        double R;
        double cp;
        int nLevs;
        double* pBot;  // vertical isobaric pressure level bottoms
        double* pMid;  // vertical isobaric pressure level middles
        Vec* ui;       // horizontal velocity field at each pressure level
        Vec* uh;
        Vec* uf;
        Vec* wi;       // vertical velocity field at each pressure half level
        Vec* wh;
        Vec* wf;
        Vec* Phii;     // geopotential field at each pressure level
        Vec* Phih;
        Vec* Phif;
        Vec* Ti;       // temperature field at each pressure level
        Vec* Th;
        Vec* Tf;
        void diagnose_vertVel(Vec* u, Vec* w);
        void diagnose_geoPot(Vec* T, Vec* Phi);
        void diagnose_wT(Vec* w, Vec* T, Vec* wt);
        void diagnose_wdudz(Vec* u);
        void solve(double dt);
};
