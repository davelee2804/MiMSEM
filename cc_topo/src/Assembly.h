class Umat {
    public:
        Umat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e);
        ~Umat();
        Topo* topo;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class Wmat {
    public:
        Wmat(Topo* _topo, LagrangeEdge* _e);
        ~Wmat();
        Topo* topo;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class Pmat {
    public:
        Pmat(Topo* _topo, LagrangeNode* _l);
        ~Pmat();
        Topo* topo;
        LagrangeNode* l;
        Mat M;
        void assemble();
};
