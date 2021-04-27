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
        double* x;
        double** ljxi;
        double** ljxi_t;
        GaussLobatto* q;
        double eval(double _x, int i);
        double eval_q(double _x, int i);
        double evalDeriv(double _x, int i);
        void test();
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
