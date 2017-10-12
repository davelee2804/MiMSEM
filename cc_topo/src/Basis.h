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
        LagrangeNode(int _n);
        ~LagrangeNode();
        int n;
        double* a;
        double** ljxi;
        double** ljxi_t;
        GaussLobatto* quad;
        double eval(double x, int i);
        double evalDeriv(double x, int i);
};

class LagrangeEdge {
    public:
        LagrangeEdge(int _n, LagrangeNode* _l);
        ~LagrangeEdge();
        int n;
        double** ejxi;
        double** ejxi_t;
        double eval(double x, int i);
}
