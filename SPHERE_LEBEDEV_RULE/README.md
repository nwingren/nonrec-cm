SPHERE_LEBEDEV_RULE is a dataset directory which contains files defining Lebedev rules on the unit sphere, which can be used for quadrature, and have a known precision.

A Lebedev rule of precision p can be used to correctly integrate any polynomial for which the highest degree term x^i y^j z^k satisfies i+j+k <= p.

The approximation to the integral of f(x) has the form Integral f(x,y,z) = 4 * pi * sum ( 1 <= i < n ) wi * f(x_i,y_i,z_i) where

        x_i = cos ( theta_i ) * sin ( phi_i )
        y_i = sin ( theta_i ) * sin ( phi_i )
        z_i =                   cos ( phi_i )
      

The data file for an n point rule includes n lines, where the i-th line lists the values of

        theta_i phi_i w_i
      

The angles are measured in degrees, and chosen so that:

        - 180 <= theta_i <= + 180
            0 <= phi_i <= + 180
      

and the weights wi should sum to 1. 

Provided by John Burkardt at https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html under the GNU LGPL license.

The rules in the data files have the following numer of points and precision:

    lebedev_003.txt, 6 point rule, precision 3.
    lebedev_005.txt, 14 point rule, precision 5.
    lebedev_007.txt, 26 point rule, precision 7.
    lebedev_009.txt, 38 point rule, precision 9.
    lebedev_011.txt, 50 point rule, precision 11.
    lebedev_013.txt, 74 point rule, precision 13.
    lebedev_015.txt, 86 point rule, precision 15.
    lebedev_017.txt, 110 point rule, precision 17.
    lebedev_019.txt, 146 point rule, precision 19.
    lebedev_021.txt, 170 point rule, precision 21.
    lebedev_023.txt, 194 point rule, precision 23.
    lebedev_025.txt, 230 point rule, precision 25.
    lebedev_027.txt, 266 point rule, precision 27.
    lebedev_029.txt, 302 point rule, precision 29.
    lebedev_031.txt, 350 point rule, precision 31.
    lebedev_035.txt, 434 point rule, precision 35.
    lebedev_041.txt, 590 point rule, precision 41.
    lebedev_047.txt, 770 point rule, precision 47.
    lebedev_053.txt, 974 point rule, precision 53.
    lebedev_059.txt, 1202 point rule, precision 59.
    lebedev_065.txt, 1454 point rule, precision 65.
    lebedev_071.txt, 1730 point rule, precision 71.
    lebedev_077.txt, 2030 point rule, precision 77.
    lebedev_083.txt, 2354 point rule, precision 83.
    lebedev_089.txt, 2702 point rule, precision 89.
    lebedev_095.txt, 3074 point rule, precision 95.
    lebedev_101.txt, 3470 point rule, precision 101.
    lebedev_107.txt, 3890 point rule, precision 107.
    lebedev_113.txt, 4334 point rule, precision 113.
    lebedev_119.txt, 4802 point rule, precision 119.
    lebedev_125.txt, 5294 point rule, precision 125.
    lebedev_131.txt, 5810 point rule, precision 131.
