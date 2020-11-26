# MiMSEM
A parallel framework for solving geophysical flow problems using mixed mimetic spectral elements.

Includes solvers (and test configurations) for:
* The 2D advection equation with explicit time integration
  * solid body rotation (on the sphere)
  * deformational flow (on the sphere)
* The rotating shallow water equations with semi-implicit time integration
  * Wilkinson2 steady flow (on the sphere)
  * Rossby-Haurwitz wave (on the sphere)
  * Barotropic instability (Galewsky) (on the sphere)
* The 3D compressible Euler equations with horizontally explicit/vertically implicit time integration
  * Baroclinic instability (on the sphere)
  * Non-hydrostatic gravity wave (on the sphere)
  * Warm bubble (doubly periodic planar)

Supports geometric configurations for:
* Doubly periodic box
* Cubed-sphere

Optional stabilisation terms include:
* Biharmonic viscosity
* Energetically consistent variational upwinding

## How to run the baroclinic instability test case on the sphere ##
0. Install the dependencies (C++, Python3, MPI, PETSc)
1. Clone the repository
2. Run the set up script: `./scr/Setup.py <polynomial_degree> <number_of_elements_per_dimension> <number_of_processors>` where
   * `polynomial_degree` is the polynomial order of the $L^2$ basis functions in the horizontal
   * `number_of_elements_per_dimension` is the number of elements in each dimension on each of the six faces of the cubed sphere
   * `number_of_processors` is the number of cores to run the code on. Note that this must be 6*n*n for integer n, ie: 6, 24, 54, etc, and `number_of_elements_per_dimension` must fit evenly into n
3. Build the code: `cd eul/; make mimsem`
4. Run the code: `mpirun -np <number_of_processors> ./mimsem <start_dump>`, where `0` indicates starting from the analytic initial condition rather than a start dump 
5. Plot an image: ``

## Picture Gallery ##
Baroclinic instability on the sphere, day 11
![](https://github.com/davelee2804/images/blob/master/euler_sphere/exner_000_0044_nh.pdf)
![](https://github.com/davelee2804/images/blob/master/euler_sphere/theta_0044_nh.pdf)
![](https://github.com/davelee2804/images/blob/master/euler_sphere/vorticity_004_0044_nh.pdf)
