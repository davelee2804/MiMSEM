class Topo {
    public:
        Topo(int _elOrd, int _nElsX);
        ~Topo();
        int  n0;      // number of local nodes
        int  n1;      // number of local edges
        int  n2;      // number of local faces
        int  elOrd;
        int  nElsX;
        int  nDofsX;
        int* loc0;      // global indices of the nodes on this processor
        int* loc1;      // global indices of the edges on this processor
        int* loc2;      // global indices of the faces on this processor
        int* inds0;   // element indices for nodes on this processor
        int* inds1;  // element indices for x-normal edges on this processor
        int* inds2;   // element indices for faces on this processor
        int* elInds0(int ex);
        int* elInds1(int ex);
        int* elInds2(int ex);
};
