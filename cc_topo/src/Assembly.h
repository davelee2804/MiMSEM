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
