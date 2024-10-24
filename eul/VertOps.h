double compute_sigma(double phi, double z);

class VertOps {
    public:
        VertOps(Topo* _topo, Geom* _geom);
        ~VertOps();

        int   n2;
        Topo* topo;
        Geom* geom;

        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;

        Wii* Q;
        M2_j_xy_i* W;
        double* Q0;
        double* QT;
        double* QB;
        double* Wt;
        double* WtQ;
        double* WtQW;
        double* WtQWinv;
        double* WtQWflat;

        double* WtQW_2;
        double* WtQW_3;

        Mat V01;      // vertical gradient operator
        Mat V10;      // vertical divergence operator
        Mat V10_full; // vertical divergence operator with non-homogeneous bcs
        Mat VA;
        Mat VB;
        Mat VA_inv;
        Mat VB_inv;
        Mat VAB;
        Mat VBA;
        Mat VR;  // rayleigh friction operator
        Mat VA2; // vertical theta mass matrix (no boundary conditions)
        Mat VAB2;

        void vertOps();

        void AssembleConst(int ex, int ey, Mat A);      // piecewise constant (in vertical) mass matrix
        void AssembleLinear(int ex, int ey, Mat B);     // piecewise linear (in vertical) mass matrix
        void AssembleLinCon(int ex, int ey, Mat AB);
        void AssembleLinearWithTheta(int ex, int ey, Vec theta, Mat A);
        void AssembleLinearWithRT(int ex, int ey, Vec rt, Mat A, bool do_internal);
        void AssembleLinearInv(int ex, int ey, Mat A);
        void AssembleConstWithRhoInv(int ex, int ey, Vec theta, Mat B);
        void AssembleConstWithRho(int ex, int ey, Vec rho, Mat A);
        void AssembleConLinWithW(int ex, int ey, Vec velz, Mat BA);
        void AssembleRayleigh(int ex, int ey, Mat B);
        void AssembleConstInv(int ex, int ey, Mat B);
        void Assemble_EOS_RHS(int ex, int ey, Vec rt, Vec eos_rhs, double factor, double exponent);
        void AssembleConLin(int ex, int ey, Mat BA);
        void AssembleLinCon2(int ex, int ey, Mat AB);                // for the diagnosis of theta without boundary conditions
        void AssembleLinearWithRho2(int ex, int ey, Vec rho, Mat A); // for the diagnosis of theta without boundary conditions
        void AssembleConstWithTheta(int ex, int ey, Vec theta, Mat B);
        void Assemble_EOS_Residual(int ex, int ey, Vec rt, Vec exner, Vec eos_rhs);
        void Assemble_EOS_BlockInv(int ex, int ey, Vec rt, Vec theta, Mat B);
	// for the dry entropy preconditioning
        void Assemble_EOS_Block(int ex, int ey, Vec rt, Mat B);
        void AssembleConstWithLogThetaPlusEta(int ex, int ey, Vec theta, Vec eta, Vec rhs);
        void AssembleConstWithRhoExpEta(int ex, int ey, Vec rho, Vec eta, Vec rhs);
        void AssembleConLinWithRhodPi(int ex, int ey, Vec theta, Vec dpi, Mat BA);
        // for the density corrections to the schur complement solution
        void AssembleLinearWithRayleighInv(int ex, int ey, double dt_fric, Mat A);
        void AssembleLinearWithRho2_up(int ex, int ey, Vec rho, Mat A, double dt, Vec* uhl);
        void AssembleLinCon2_up(int ex, int ey, Mat AB, double dt, Vec* uhl);
        void AssembleTempForcing_HS(int ex, int ey, Vec exner, Vec theta, Vec rho, Vec vec);
};
