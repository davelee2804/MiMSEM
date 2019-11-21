class Boundary {
    public:
        Boundary(Topo* _topo, Geom* _geom, LagrangeNode* _node, LagrangeEdge* _edge);
        ~Boundary();
        Topo* topo;
        Geom* geom;
        LagrangeNode* node;
        LagrangeEdge* edge;
        double** EQ;
        Vec ul;
        Vec hl;
        Vec ql;
        void Interp2FormTo0FormBndry(int lev, Vec u, Vec h, bool upwind);
        void _assembleGrad(int lev, Vec b);
        void AssembleGrad(int lev, Vec u, Vec h, Vec b, bool upwind);
};
