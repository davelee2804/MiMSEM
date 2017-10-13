void mult(int ni, int nj, int nk, double** A, double** B, double** C);
void tran(int ni, int nj, double**A, double** B);

class M1x_j_xy_i {
    public:
        M1x_j_xy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1x_j_xy_i();
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M1y_j_xy_i {
    public:
        M1y_j_xy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1y_j_xy_i();
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M1x_j_Cxy_i {
    public:
        M1x_j_Cxy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1x_j_Cxy_i();
        void assemble(double* c);
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M1y_j_Cxy_i {
    public:
        M1y_j_Cxy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1y_j_Cxy_i();
        void assemble(double* c);
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M1x_j_Exy_i {
    public:
        M1x_j_Exy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1x_j_Exy_i();
        void assemble(double* c);
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M1y_j_Exy_i {
    public:
        M1y_j_Exy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1y_j_Exy_i();
        void assemble(double* c);
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M1x_j_Dxy_i {
    public:
        M1x_j_Dxy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1x_j_Dxy_i();
        void assemble(double* c);
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M1y_j_Dxy_i {
    public:
        M1y_j_Dxy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1y_j_Dxy_i();
        void assemble(double* c);
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M1x_j_Fxy_i {
    public:
        M1x_j_Fxy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1x_j_Fxy_i();
        void assemble(double* c);
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M1y_j_Fxy_i {
    public:
        M1y_j_Fxy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M1y_j_Fxy_i();
        void assemble(double* c);
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class M2_j_xy_i {
    public:
        M2_j_xy_i(LagrangeEdge* _e);
        ~M2_j_xy_i();
        double** A;
        LagrangeEdge* e;
};

class M0_j_xy_i {
    public:
        M0_j_xy_i(LagrangeNode* _l);
        ~M0_j_xy_i();
        double** A;
        LagrangeNode* l;
};

class M0_j_Cxy_i {
    public:
        M0_j_Cxy_i(LagrangeNode* _l, LagrangeEdge* _e);
        ~M0_j_Cxy_i();
        void assemble(double* c);
        double** A;
        LagrangeNode* l;
        LagrangeEdge* e;
};

class Wii {
    public:
        Wii(GaussLobatto* _q);
        ~Wii();
        double** A;
        GaussLobatto* q;
};
