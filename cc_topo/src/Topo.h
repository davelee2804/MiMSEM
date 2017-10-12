class Topo {
    public:
        Topo(int _pi);
        ~Topo();
        int  pi;
        int  n0;
        int  n1x;
        int  n1y;
        int  n2;
        int* inds0;   // global indices of the nodes on this processor
        int* inds1x;  // global indices of the x-normal edges on this processor
        int* inds1y;  // global indices of the y-normal edges on this processor
        int* inds2;   // global indices of the faces on this processor
        ISLocalToGlobalMapping map0;
        ISLocalToGlobalMapping map1x;
        ISLocalToGlobalMapping map1y;
        ISLocalToGlobalMapping map2;
        void LoadObjs(char* filename, int* inds);
};
