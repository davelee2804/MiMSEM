class GaussLobatto {
    public:
        GaussLobatto(int _n);
        ~GaussLobatto();
        int n;
        double* x;
        double* w;
};

class LagrangeNode {
    public:
        LagrangeNode(int _n, GaussLobatto* _q);
        ~LagrangeNode();
        int n;
        double* a;
        double** ljxi;
        double** ljxi_t;
        GaussLobatto* q;
        double eval(double x, int i);
        double evalDeriv(double x, int i);
    private:
        void polyMult(int n1, double* a1, int n2, double* a2, double* a3);
        void polyMultI(int i, double* X, double* pir);
};

class LagrangeEdge {
    public:
        LagrangeEdge(int _n, LagrangeNode* _l);
        ~LagrangeEdge();
        int n;
        double** ejxi;
        double** ejxi_t;
        LagrangeNode* l;
        double eval(double x, int i);
};
