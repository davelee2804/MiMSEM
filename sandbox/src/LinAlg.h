double** Alloc2D(int ni, int nj);
void Free2D(int ni, double** A);
double* Flat2D(int ni, int nj, double** A);
void Flat2D_IP(int ni, int nj, double** A, double* Aflat);
double** Mult(int ni, int nj, int nk, double** A, double** B);
void Mult_IP(int ni, int nj, int nk, double** A, double** B, double** C);
double** Tran(int ni, int nj, double** A);
void Tran_IP(int ni, int nj, double** A, double** B);
bool ArcInt(double rad, double* ai, double* af, double* bi, double* bf, double* xo);
