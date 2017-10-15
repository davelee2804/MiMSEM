class Topo {
    public:
        Topo(int _pi, int _elOrd, int _nElsX);
        ~Topo();
        int  pi;      // processor index
        int  n0;      // number of local nodes
        int  n1x;     // number of local edges (x-normal)
        int  n1y;     // number of local edges (y-normal)
        int  n2;      // number of faces
        int  elOrd;
        int  nElsX;
        int  nDofsX;
        int  nDofs0G;
        int  nDofs1G;
        int  nDofs2G;
        int* loc0;    // global indices of the nodes on this processor
        int* loc1;    // global indices of the edges on this processor
        int* loc1x;   // global indices of the x-normal edges on this processor
        int* loc1y;   // global indices of the y-normal edges on this processor
        int* loc2;    // global indices of the faces on this processor
        int* inds0;   // element indices for nodes on this processor
        int* inds1x;  // element indices for x-normal edges on this processor
        int* inds1y;  // element indices for y-normal edges on this processor
        int* inds2;   // element indices for faces on this processor
        ISLocalToGlobalMapping map0;
        ISLocalToGlobalMapping map1;
        ISLocalToGlobalMapping map2;
        void loadObjs(char* filename, int* inds);
        int* elInds0(int ex, int ey);
        int* elInds1x(int ex, int ey);
        int* elInds1y(int ex, int ey);
        int* elInds2(int ex, int ey);
};
