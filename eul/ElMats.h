class M1x_j_xy_i {
    public:
        M1x_j_xy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1x_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double* A;
        LagrangeNode* node;
        LagrangeEdge* edge;
};

class M1y_j_xy_i {
    public:
        M1y_j_xy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1y_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double* A;
        LagrangeNode* node;
        LagrangeEdge* edge;
};

class M2_j_xy_i {
    public:
        M2_j_xy_i(LagrangeEdge* _edge);
        ~M2_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double* A;
        LagrangeEdge* edge;
};

class M0_j_xy_i {
    public:
        M0_j_xy_i(LagrangeNode* _node);
        ~M0_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double* A;
        LagrangeNode* node;
};

class Wii {
    public:
        Wii(GaussLobatto* _quad, Geom* _geom);
        ~Wii();
        int nDofsI;
        int nDofsJ;
        double* A;
        GaussLobatto* quad;
        Geom* geom;
    private:
        void assemble();
};
