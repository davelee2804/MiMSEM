class GMRES_constrained {
    public:
        GMRES_constrained(Topo* _topo, double _tol);
        ~GMRES_constrained();
        int rank;
        double tol;
        double** Hess;
        Vec* Q;
        Topo* topo;
        void solve(Mat A, Vec b, Vec c, Vec x);
};
