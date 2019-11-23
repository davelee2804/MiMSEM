class Boundary {
    public:
        Boundary(Topo* _topo, Geom* _geom, LagrangeNode* _node, LagrangeEdge* _edge);
        ~Boundary();
        Topo* topo;
        Geom* geom;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Vec ul;
        Vec hl;
        Vec ql;
        Vec bl;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        double** Ut;
        double** Vt;
        void Interp2To0Bndry(int lev, Vec u, Vec h, bool upwind);
        void Interp1To0Bndry(int lev, Vec u, bool upwind);
        void _assembleGrad(int lev, Vec b);
        void _assembleConv(int lev, Vec u, Vec b);
        void AssembleGrad(int lev, Vec u, Vec h, Vec b, bool upwind);
};
