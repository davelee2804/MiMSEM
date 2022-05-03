class Topo {
    public:
        Topo();
        ~Topo();
        int  pi;      // processor index
        int  n0;      // number of local nodes
        int  n1;      // number of local edges
        int  n1x;     // number of local edges (x-normal)
        int  n1y;     // number of local edges (y-normal)
        int  n2;      // number of local faces
        int  n0l;     // number of local nodes (exclusing ghost values)
        int  n1l;     // number of local edges (exclusing ghost values)
        int  n1xl;    // number of local edges (x-normal) (exclusing ghost values)
        int  n1yl;    // number of local edges (y-normal) (exclusing ghost values)
        int  n2l;     // number of local faces (exclusing ghost values)
        int  elOrd;
        int  nElsX;
        int  nDofsX;
        int  nDofs0G;
        int  nDofs1G;
        int  nDofs2G;
        int* loc0;      // global indices of the nodes on this processor
        int* loc1;      // global indices of the edges on this processor
        int* loc1x;     // global indices of the x-normal edges on this processor
        int* loc1y;     // global indices of the y-normal edges on this processor
        int* loc2;      // global indices of the faces on this processor
        int* inds0_l;   // element indices for nodes on this processor
        int* inds1x_l;  // element indices for x-normal edges on this processor
        int* inds1y_l;  // element indices for y-normal edges on this processor
        int* inds2_l;   // element indices for faces on this processor
        int* inds0_g;   // element indices for nodes on this processor
        int* inds1x_g;  // element indices for x-normal edges on this processor
        int* inds1y_g;  // element indices for y-normal edges on this processor
        int* inds2_g;   // element indices for faces on this processor
	int* coarse_inds_x;
	int* coarse_inds_y;
	int* coarse_inds;
        IS is_l_0;
        IS is_g_0;
        IS is_l_1;
        IS is_g_1;
        VecScatter gtol_0;
        VecScatter gtol_1;
        void loadObjs(char* filename, int* inds);
        int* elInds0_l(int ex, int ey);
        int* elInds1x_l(int ex, int ey);
        int* elInds1y_l(int ex, int ey);
        int* elInds2_l(int ex, int ey);
        int* elInds0_g(int ex, int ey);
        int* elInds1x_g(int ex, int ey);
        int* elInds1y_g(int ex, int ey);
        int* elInds2_g(int ex, int ey);
        IS is_l_coarse;
        IS is_g_coarse;
        VecScatter gtol_coarse;
	void coarseInds();
	void skelIntlInds();
	int dd_n_skel_locl;
	int dd_n_intl_locl;
	int dd_n_skel_glob;
        int* dd_skel_locl_x;
        int* dd_skel_locl_y;
        int* dd_intl_locl_x;
        int* dd_intl_locl_y;
        int* dd_skel_global;
        int* inds_intl_x_l;
        int* inds_intl_y_l;
        int* inds_skel_x_l;
        int* inds_skel_y_l;
        int* inds_skel_x_g;
        int* inds_skel_y_g;
        int* elInds_intl_x_l(int ex, int ey);
        int* elInds_intl_y_l(int ex, int ey);
        int* elInds_skel_x_l(int ex, int ey);
        int* elInds_skel_y_l(int ex, int ey);
        int* elInds_skel_x_g(int ex, int ey);
        int* elInds_skel_y_g(int ex, int ey);
        IS is_skel_l, is_skel_g;
        VecScatter gtol_skel;
        int* dd_skel_locl_glob_map;
};
