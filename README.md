# Nonreciprocal characteristic mode computations

This repository contains code for computing characteristic modes using the finite element-boundary integral code [FE2MS](https://github.com/nwingren/fe2ms). The code can specifically be used for computations involving nonreciprocal systems.

The algorithms for computing the modes based on the scattering dyadic are described in the papers
- J. Lundgren, K. Schab, M. Capek, M. Gustafsson and L. Jelinek, "Iterative Calculation of Characteristic Modes Using Arbitrary Full-Wave Solvers," in IEEE Antennas and Wireless Propagation Letters, vol. 22, no. 4, pp. 799-803, April 2023, doi: [10.1109/LAWP.2022.3225706](https://doi.org/10.1109/LAWP.2022.3225706).
- M. Capek, J. Lundgren, M. Gustafsson, K. Schab and L. Jelinek, "Characteristic Mode Decomposition Using the Scattering Dyadic in Arbitrary Full-Wave Solvers," in IEEE Transactions on Antennas and Propagation, vol. 71, no. 1, pp. 830-839, Jan. 2023, doi: [10.1109/TAP.2022.3213945](https:/doi.org/10.1109/TAP.2022.3213945).

Other references of interest are
- M. Gustafsson, L. Jelinek, K. Schab and M. Capek, "Unified Theory of Characteristic Modes—Part I: Fundamentals," in IEEE Transactions on Antennas and Propagation, vol. 70, no. 12, pp. 11801-11813, Dec. 2022, doi: [10.1109/TAP.2022.3211338](https://doi.org/10.1109/TAP.2022.3211338).,
- M. Gustafsson, L. Jelinek, K. Schab and M. Capek, "Unified Theory of Characteristic Modes—Part II: Tracking, Losses, and FEM Evaluation," in IEEE Transactions on Antennas and Propagation, vol. 70, no. 12, pp. 11814-11824, Dec. 2022, doi: [10.1109/TAP.2022.3209264](https:/doi.org/10.1109/TAP.2022.3209264).
  
## Installation

The basis of this code is the finite element-boundary integral code [FE2MS](https://github.com/nwingren/fe2ms), and as such that needs to be installed according to the instructions at https://github.com/nwingren/fe2ms. The following assumes that the installation was done as such.

Some additional Python packages are necessary to run all provided scripts, and these are listed in [environment.yml](environment.yml). Packages already included in the FE2MS installation are not listed. These remaining packages are installed by
```bash
mamba env update
```

## License

Copyright (C) 2023 Niklas Wingren

This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Acknowledgements
This work was supported in part by the Swedish Armed Forces, in part by the Swedish Defence Materiel Administration, in part by the National Aeronautics Research Program and in part by the Swedish Governmental Agency for Innovation Systems.