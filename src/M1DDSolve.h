typedef double (ICfunc) (double* xi);

class M1DDSolve {
    public:
        M1DDSolve(Topo* _topo, Geom* _geom);
        ~M1DDSolve();
        int rank;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        Umat* M1;
        E21mat* EtoF;
        KSP ksp;           // 1 form mass matrix linear solver
        void init1(Vec u, ICfunc* func_x, ICfunc* func_y);
        void err1(Vec u, ICfunc* fu, ICfunc* fv, ICfunc* fp, double* norms);
	Mat Mii;
	Mat Mid;
	Mat Mis;
	Mat Mdi;
	Mat Mdd;
	Mat Mds;
	Mat Msi;
	Mat Msd;
	Mat Mss;
	void assemble();
};
