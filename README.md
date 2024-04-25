# MiMSEM
A parallel framework for solving geophysical flow problems at both planetary and non-hydrostatic scales using mixed mimetic spectral elements. The code is based on spatial and temporal discretisations that preserve the exact balance of energy exchanges for an improved representation of dynamical processes.

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
* Doubly periodic box (3D)
* Cubed-sphere (2D/3D)

Optional stabilisation/forcing terms include:
* Biharmonic viscosity for horizontal momentum and temperature equations
* Energetically consistent variational upwinding for potential vorticity (in the horizontal) and potential temperature (in the vertical)
* Held-Suarez forcing of temperature and lower atmosphere Rayleigh friction

## How to run the baroclinic instability test case on the sphere ##
0. Install the dependencies (C++, Python3, OpenBLAS, MPI, PETSc)
1. Clone the repository
2. Run the set up script: `./scr/Setup.py <polynomial_degree> <number_of_elements_per_dimension> <number_of_processors>` where
   * `polynomial_degree` is the polynomial order of the ![](https://render.githubusercontent.com/render/math?math=L^2) basis functions in the horizontal
   * `number_of_elements_per_dimension` is the number of elements in each dimension on each of the six faces of the cubed sphere
   * `number_of_processors` is the number of cores to run the code on. Note that this must be  ![](https://render.githubusercontent.com/render/math?math=6n^2) for integer _n_, ie: 6, 24, 54, 96, etc, and `number_of_elements_per_dimension` must fit evenly into _n_
3. Build the code: `cd eul/; make mimsem`
4. Run the code: `mpirun -np <number_of_processors> ./mimsem <start_dump>`, where `0` indicates starting from the analytic initial condition rather than a start dump 
5. Plot an image: `../scr/WriteImage_NorthHemi.py <file_path> <field_name> <plot_contours> <vertical_level> <dump_time>`

## Picture Gallery ##
<img src="https://github.com/davelee2804/images/blob/master/euler_sphere/exner_000_0036_nh.png" height="360" width="270"><img src="https://github.com/davelee2804/images/blob/master/euler_sphere/theta_0036_nh.png" height="360" width="270"><img src="https://github.com/davelee2804/images/blob/master/euler_sphere/vorticity_004_0036_nh.png" height="360" width="270">

<sub>Baroclinic instability on the sphere, day 9: surface level Exner pressure, and potential temperature and vertical voricity component at z=1.5km</sub>

<img src="https://github.com/davelee2804/images/blob/master/euler_sphere/exner_000_0044_nh.png" height="360" width="270"><img src="https://github.com/davelee2804/images/blob/master/euler_sphere/theta_0044_nh.png" height="360" width="270"><img src="https://github.com/davelee2804/images/blob/master/euler_sphere/vorticity_004_0044_nh.png" height="360" width="270">

<sub>Baroclinic instability on the sphere, day 11: surface level Exner pressure, and potential temperature and vertical voricity component at z=1.5km</sub>

<img src="https://github.com/davelee2804/images/blob/master/held_suarez/exner_000_0040.png" height="300" width="270"><img src="https://github.com/davelee2804/images/blob/master/held_suarez/theta_003_0040.png" height="300" width="270"><img src="https://github.com/davelee2804/images/blob/master/held_suarez/vorticity_004_0040.png" height="300" width="270">

<sub>Global simulation with Held-Suarez forcing at day 40, surface level Exner pressure, potential temperature at z = 2.36km and vertical vorticity component at z = 3.14km</sub>

<img src="https://github.com/davelee2804/images/blob/master/sw_sphere/vorticity_0005_nh.png" height="360" width="270"><img src="https://github.com/davelee2804/images/blob/master/sw_sphere/vorticity_0006_nh.png" height="360" width="270"><img src="https://github.com/davelee2804/images/blob/master/sw_sphere/vorticity_0007_nh.png" height="360" width="270">

<sub>Vorticity field at days 5, 6 and 7 for the Galewsky test case for the rotating shallow water on the sphere using energetically balanced upwinding of potential vorticity and and energetically balanced IMEX time splitting without dissipation.</sub>

<img src="https://github.com/davelee2804/images/blob/master/euler_sphere/theta_0025_up.png" height="240" width="320"><img src="https://github.com/davelee2804/images/blob/master/euler_sphere/theta_0050_up.png" height="240" width="320">

<sub>Potential temperature for the 3D rising bubble test case in planar geometry without dissipation and with energetically consistent variational upwinding at times 200s (left) and 400s (right)</sub>

## References ##
Ricardo, K., Lee, D. and Duru, K. (2024) [Entropy and energy conservation for thermal atmospheric dynamics using mixed compatible finite elements](https://www.sciencedirect.com/science/article/abs/pii/S0021999123007003) _J. Comp. Phys._ 496 112605

Lee, D., Martin, A., Bladwell, C. and Badia, S. (2024) [A comparison of variational upwinding schemes for geophysical fluids, and their application to potential enstrophy conserving discretisations in space and time](https://www.sciencedirect.com/science/article/abs/pii/S0898122124001755) _Comput. Math. Appl._ 165 150-162

Lee, D. and Palha, A. (2021) [Exact spatial and temporal balance of energy exchanges within a horizontally explicit/vertically implicit non-hydrostatic atmosphere](https://www.sciencedirect.com/science/article/pii/S0021999121003272) _J. Comp. Phys._ 440 110432

Lee, D. (2021) [An energetically balanced, quasi-Newton integrator for non-hydrostatic vertical atmospheric dynamics](https://www.sciencedirect.com/science/article/pii/S0021999120307622) _J. Comp. Phys._ 429 109988

Lee, D. (2021) [Petrov-Galerkin flux upwinding for mixed mimetic spectral elements, and its application to geophysical flow problems](https://www.sciencedirect.com/science/article/pii/S0898122121000651) _Comput. Math. Appl._ 89 68-77

Lee, D. and Palha, A. (2020) [A mixed mimetic spectral element model of the 3D compressible Euler equations on the cubed sphere](https://www.sciencedirect.com/science/article/pii/S0021999119306989) _J. Comp. Phys._ 401 108993

Lee, D. and Palha, A. (2018) [A mixed mimetic spectral element model of the rotating shallow water equations on the cubed sphere](https://www.sciencedirect.com/science/article/pii/S0021999118305734) _J. Comp. Phys._ 375 240-262

Lee, D., Palha, A. and Gerritsma, M. (2018) [Discrete conservation properties for shallow water flows using mixed mimetic spectral elements](https://www.sciencedirect.com/science/article/pii/S0021999117309166) _J. Comp. Phys._ 357 282-304

## Comments? Questions? Like to Contribute? ##
Email Dave Lee at davelee2804@gmail.com
