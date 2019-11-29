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
        Vec qg;
        Vec bl;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        double** Ut;
        double** Vt;
        double* Qa;
        double* Qb;
        double* UtQa;
        double* VtQb;
        void Interp2To0Bndry(int lev, Vec u, Vec h, bool upwind);
        void _assembleGrad(int lev, Vec b);
        void AssembleGrad(int lev, Vec u, Vec h, Vec b, bool upwind);
};
