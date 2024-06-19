# Copyright (C) 2023 Niklas Wingren

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Functions for characteristic mode computations
"""

import os as _os
import numpy as _np
import numba as _nb
import scipy.sparse as _sparse
from scipy.constants import mu_0 as _mu0, epsilon_0 as _epsilon0

import fe2ms as _fe2ms

lebedev_deg_to_idx = {
    6:    "003",
    14:   "005",
    26:   "007",
    38:   "009",
    50:   "011",
    74:   "013",
    86:   "015",
    110:  "017",
    146:  "019",
    170:  "021",
    194:  "023",
    230:  "025",
    266:  "027",
    302:  "029",
    350:  "031",
    434:  "035",
    590:  "041",
    770:  "047",
    974:  "053",
    1202: "059",
    1454: "065",
    1730: "071",
    2030: "077",
    2354: "083",
    2702: "089",
    3074: "095",
    3470: "101",
    3890: "107",
    4334: "113",
    4802: "119",
    5294: "125",
    5810: "131"
}

LEBEDEV_DEGREES = _np.array(list(lebedev_deg_to_idx.keys()))

def get_lebedev(degree, which='above'):
    """
    Get lebedev quadrature rule of a given degree. Rule is transformed such that integrals can be
    computed by a sum over f(quad_point) * quad_weight for all (quad_point, quad_weight) pairs.

    Theta and phi angles are also changed from the original data to fit with the convention used in
    the fe2ms code, and they are changed to radians.

    Parameters
    ----------
    degree : int
        Degree of quadrature rule.
    which : {'above', 'near'}
        Which quadrature rule degree to use if 'degree' is not available, by default 'above'.

    Returns
    -------
    theta : array
        Polar angles of quadrature points in radians.
    phi : array
        Azimuth angles of quadrature points in radians.
    w : array
        Quadrature weights.
    """

    if degree not in lebedev_deg_to_idx:
        print(f'Lebedev rule of degree {degree} not available, ', end='')
        existing_degs = _np.array(list(lebedev_deg_to_idx.keys()))
        if which == 'above':
            degree = existing_degs[existing_degs - degree > 0]
            degree = degree[0]
        else:
            degree = existing_degs[_np.abs(existing_degs - degree) == _np.abs(existing_degs - degree).min()]
            degree = degree[-1]
        print(f'using {degree} instead')

    idx = lebedev_deg_to_idx[degree]

    file_dir = _os.path.dirname(__file__)
    data = _np.loadtxt(_os.path.join(file_dir, f'SPHERE_LEBEDEV_RULE/lebedev_{idx}.txt'))

    theta = _np.radians(data[:,1])
    phi = _np.radians(data[:,0])
    phi[phi < 0] += 2 * _np.pi
    w = data[:,2] * 4 * _np.pi

    return theta, phi, w


# FIXME: Change things in here to match eigenvector changes done in the iterative version
def compute_cm_full(system: _fe2ms.systems.FEBISystem, M_prec, lebedev_rule, n_modes=None):
    """
    Computation of characteristic modes by fully populating the scattering dyadic.

    Parameters
    ----------
    system : _fe2ms.systems.FEBISystem
        FEBI system for problem to compute modes for.
    M_prec : fun
        Function which preconditions a vector.
    lebedev_rule : tuple
        Tuple of (theta, phi, weights) corresponding to Lebedev rule, obtained from get_lebedev.
    n_modes : int, optional
        Number of modes to compute, by default None in which case it is the Lebedev degree.

    Returns
    -------
    modes : list
        Characteristic modes for all iterations, converged in modes[-1].
    eigenvectors : list
        Eigenvectors for all iterations, converged in eigenvectors[-1].
    Po : array
        Output powers of all modes.
    Po_r : array
        Reciprocal output powers of all modes.
    Po_nr : array
        Nonreciprocal output powers of all modes.
    -------
    """

    theta_leb, phi_leb, w_leb = lebedev_rule
    if _np.any(w_leb < 0):
        raise ValueError('Lebedev rule contains negative weights, choose another')

    # Get flip map before doubling the quadrature
    flip_map = lebedev_flip_map(theta_leb, phi_leb)

    # Double for two polarizations
    # The excitation vector is organized as [E_theta, E_phi].T
    N_leb = phi_leb.size * 2
    theta_leb = _np.concatenate((theta_leb, theta_leb))
    phi_leb = _np.concatenate((phi_leb, phi_leb))
    w_leb = _np.concatenate((w_leb, w_leb))

    # Create map from incident plane waves to BI right-hand side in FE-BI system
    rhs_from_inc = _np.zeros((system.spaces.bi_size, N_leb), dtype=_np.complex128)
    _compute_rhs_map(
        theta_leb, phi_leb, system._k0, system.spaces.bi_basisdata.basis,
        system.spaces.bi_basisdata.quad_points, system.spaces.bi_basisdata.quad_weights,
        system.spaces.bi_meshdata.edge2facet, system.spaces.bi_meshdata.facet_areas,
        rhs_from_inc
    )

    # For CFIE, H field contribution must be added. All values are contained in map for E field,
    # but at different locations
    if system._formulation == 'teth':
        rhs_from_inc_H = rhs_from_inc.copy()
        rhs_from_inc[:, :N_leb//2] += rhs_from_inc_H[:, N_leb//2:]
        rhs_from_inc[:, N_leb//2:] -= rhs_from_inc_H[:, :N_leb//2]
        rhs_from_inc *= 0.5

    # Create map from surface magnetic/electric currents to scattered far fields
    ff_from_M = _np.zeros((N_leb, system.spaces.bi_size), dtype=_np.complex128)
    ff_from_J = _np.zeros((N_leb, system.spaces.bi_size), dtype=_np.complex128)
    _compute_ff_map(
        theta_leb, phi_leb, system._k0, system.spaces.bi_basisdata.basis,
        system.spaces.bi_basisdata.quad_points, system.spaces.bi_basisdata.quad_weights,
        system.spaces.bi_meshdata.edge2facet, system.spaces.bi_meshdata.facet_areas,
        ff_from_M, ff_from_J
    )

    S_dyadic = _np.empty((N_leb, N_leb), dtype=_np.complex128)

    for m_inc in range(N_leb):

        E_inc = _np.zeros(N_leb, dtype=_np.complex128)
        E_inc[m_inc] = 1
        system._rhs = _np.concatenate((
            _np.zeros(system.spaces.fe_size), rhs_from_inc @ E_inc
        ))
        system.solve_iterative(preconditioner=M_prec)
        F_sc = ff_from_M @ -system.sol_E[-system.spaces.bi_size:] + ff_from_J @ system.sol_H
        S_dyadic[:, m_inc] = F_sc * system._k0 / (4 * _np.pi * 1j) #/ w_leb[m_inc]

    # Compute modes
    sqw = _np.sqrt(w_leb)[:,None]
    if n_modes is None:
        modes, eigenvectors = _np.linalg.eig(sqw.T * S_dyadic *sqw)
    else:
        modes, eigenvectors = _sparse.linalg.eigs(sqw.T * S_dyadic * sqw, n_modes)

    # Sort by eigenvalue magnitude (eigs does not sort them)
    id_sort = _np.argsort(_np.abs(modes))[::-1]
    modes = modes[id_sort]
    eigenvectors = eigenvectors[:, id_sort]

    # Compute nonreciprocity

    # Flip map is necessarily symmetric
    S_transpose = _np.block([
        [
            flip_map @ S_dyadic.T[:N_leb//2, :N_leb//2] @ flip_map,
            - flip_map @ S_dyadic.T[:N_leb//2, N_leb//2:] @ flip_map
        ],
        [
            - flip_map @ S_dyadic.T[N_leb//2:, :N_leb//2] @ flip_map,
            flip_map @ S_dyadic.T[N_leb//2:, N_leb//2:] @ flip_map
        ]
    ])

    eta0 = _np.sqrt(_mu0 / _epsilon0)
    Po = _np.empty(n_modes)
    Po_r = _np.empty_like(Po)
    Po_nr = _np.empty_like(Po)
    
    for m in range(n_modes):
        # Po[m] = (4 * _np.pi / system._k0)**2 / 2 / eta0 * _np.sum(
        #     _np.abs(2 * S_dyadic @ (eigenvectors[:,m] * w_leb) + eigenvectors[:,m])**2
        #     * w_leb
        # )
        # Po_r[m] = (4 * _np.pi / system._k0)**2 / 2 / eta0 * _np.sum(
        #     _np.abs((S_dyadic + S_transpose) @ (eigenvectors[:,m] * w_leb) + eigenvectors[:,m])**2
        #     * w_leb
        # )
        Po[m] = _np.sum(
            _np.abs(2 * (S_dyadic @ eigenvectors[:,m]) * _np.sqrt(w_leb) + eigenvectors[:,m])**2
        )
        Po_r[m] = _np.sum(
            _np.abs(
                ((S_dyadic + S_transpose) @ eigenvectors[:,m]) * _np.sqrt(w_leb)
                + eigenvectors[:,m]
            )**2
        )
    Po_nr = Po - Po_r

    return modes, eigenvectors, Po, Po_r, Po_nr


def compute_cm_iterative(
        system: _fe2ms.systems.FEBISystem, M_prec, lebedev_rule, n_modes, max_iters=100, error_thres=1e-4, sig_mode_thres=1e-2, n_convergence=2, matrix_free=False, full_eig=True, solve_direct=False
):
    """
    Computation of characteristic modes by iterative eigenvalue computations using the
    scattering dyadic.

    Parameters
    ----------
    system : _fe2ms.systems.FEBISystem
        FEBI system for problem to compute modes for.
    M_prec : fun
        Function which preconditions a vector, None if direct solution.
    lebedev_rule : tuple
        Tuple of (theta, phi, weights) corresponding to Lebedev rule, obtained from get_lebedev.
    n_modes : int
        Number of modes to compute.
    max_iters : int, optional
        Maximum number of iterations to perform, by default 100
    error_thres : float, optional
        Threshold for error to consider as converged, by default 1e-4
    sig_mode_thres : float, optional
        Threshold for a modal significance amplitude to be considered significant, by default 1e-2
    n_convergence : int, optional
        Number of concurrent iterations at stopping criterion before iteration stops, by default 2
    matrix_free : bool, optional
        Whether to compute eigenvalues in a matrix free way, by default False. This is only worth it
        if the Lebedev quadrature rule has a large number of points. This overrules full_eig.
    full_eig : bool, optional
        Whether to use full eigenvalue computation and truncate instead of using iterative method,
        by default True.
    solve_direct : bool, optional
        Whether to solve directly (with LU if that exists), will override iterative solver and preconditioner if True.

    Returns
    -------
    modes : list
        Characteristic modes for all iterations, converged in modes[-1].
    eigenvectors : list
        Eigenvectors for all iterations, converged in eigenvectors[-1].
    Po : array
        Output powers of all modes.
    Po_r : array
        Reciprocal output powers of all modes.
    Po_nr : array
        Nonreciprocal output powers of all modes.
    niters : int
        Number of iterations performed, -1 if not converged before max_iters.
    -------
    """
    
    rng = _np.random.default_rng()
    theta_leb, phi_leb, w_leb = lebedev_rule
    if _np.any(w_leb < 0):
        raise ValueError('Lebedev rule contains negative weights, choose another')

    # Get flip map before doubling the quadrature
    flip_map = lebedev_flip_map(theta_leb, phi_leb)

    # Double for two polarizations
    # The excitation vector is organized as [E_theta, E_phi].T
    N_leb = phi_leb.size * 2
    theta_leb = _np.concatenate((theta_leb, theta_leb))
    phi_leb = _np.concatenate((phi_leb, phi_leb))
    w_leb = _np.concatenate((w_leb, w_leb))
    sqw_leb = _np.sqrt(w_leb)[:,None]
    if matrix_free:
        sqw_rav = sqw_leb.ravel()

    # Create map from incident plane waves to BI right-hand side in FE-BI system
    rhs_from_inc = _np.zeros((system.spaces.bi_size, N_leb), dtype=_np.complex128)
    _compute_rhs_map(
        theta_leb, phi_leb, system._k0, system.spaces.bi_basisdata.basis,
        system.spaces.bi_basisdata.quad_points, system.spaces.bi_basisdata.quad_weights,
        system.spaces.bi_meshdata.edge2facet, system.spaces.bi_meshdata.facet_areas,
        rhs_from_inc
    )

    # For CFIE, H field contribution must be added. All values are contained in map for E field,
    # but at different locations
    if system._formulation == 'teth':
        rhs_from_inc_H = rhs_from_inc.copy()
        rhs_from_inc[:, :N_leb//2] += rhs_from_inc_H[:, N_leb//2:]
        rhs_from_inc[:, N_leb//2:] -= rhs_from_inc_H[:, :N_leb//2]
        rhs_from_inc *= 0.5

    # Create map from surface magnetic/electric currents to scattered far fields
    ff_from_M = _np.zeros((N_leb, system.spaces.bi_size), dtype=_np.complex128)
    ff_from_J = _np.zeros((N_leb, system.spaces.bi_size), dtype=_np.complex128)
    _compute_ff_map(
        theta_leb, phi_leb, system._k0, system.spaces.bi_basisdata.basis,
        system.spaces.bi_basisdata.quad_points, system.spaces.bi_basisdata.quad_weights,
        system.spaces.bi_meshdata.edge2facet, system.spaces.bi_meshdata.facet_areas,
        ff_from_M, ff_from_J
    )

    # Step 2
    E_inc = [rng.random(N_leb) + 0j]
    F_sc = []
    modes = []
    eigenvectors = []

    niters = -1
    n_below_error = 0

    S_approx = _np.zeros((N_leb, N_leb), dtype=_np.complex128)

    for m in range(min(max_iters, N_leb)):

        # Step 4 (normalize incident electric field vector)
        E_inc[-1] /= _np.sqrt(E_inc[-1].conj() @ (E_inc[-1] * w_leb))

        # Step 5 (compute scattered far field vector)
        system._rhs = _np.concatenate((
            _np.zeros(system.spaces.fe_size), rhs_from_inc @ (E_inc[-1] * w_leb)
        ))
        if solve_direct:
            system.solve_direct()
        else:
            system.solve_iterative(preconditioner=M_prec)
        F_sc.append((ff_from_M @ -system.sol_E[-system.spaces.bi_size:] + ff_from_J @ system.sol_H))

        # Step 6-7 (compute eigenvalues)
        S_approx += F_sc[-1][:,None] @ E_inc[-1][None,:].conj() * system._k0 / (4 * _np.pi * 1j)

        if matrix_free:
            S_matvec = _sparse.linalg.LinearOperator(
                shape = (N_leb, N_leb),
                matvec = lambda x: sqw_rav * _outprod_mvp(
                    _np.column_stack(F_sc),
                    _np.row_stack(E_inc).conj(),
                    sqw_rav * x
                ) * system._k0 / (4 * _np.pi * 1j),
                dtype = _np.complex128
            )
            mod, vec = _sparse.linalg.eigs(S_matvec, n_modes, v0=E_inc[0])
        else:
            if full_eig:
                mod, vec = _np.linalg.eig(sqw_leb * S_approx * sqw_leb.T)
            else:
                mod, vec = _sparse.linalg.eigs(sqw_leb * S_approx * sqw_leb.T, n_modes, v0=E_inc[0])

        # Sort by eigenvalue magnitude (eigs does not sort them)
        id_sort = _np.argsort(_np.abs(mod))[::-1]
        mod = mod[id_sort]
        vec = vec[:, id_sort]

        # Renormalize eigenvectors
        vec /= _np.sqrt(_np.sum(vec.conj() * (vec * w_leb[:,None]), axis=0, keepdims=True))

        if full_eig:
            mod = mod[:n_modes]
            vec = vec[:, :n_modes]
        modes.append(mod)
        eigenvectors.append(vec)

        # Compute error and check stopping criterion
        if m > 0:
            mode_errors = _np.abs(modes[-1] - modes[-2])/_np.abs(modes[-1])
            sig_modes = _np.abs(modes[-1]) > sig_mode_thres
            if _np.any(sig_modes):
                sig_mode_error = _np.max(mode_errors[sig_modes])
            else:
                sig_mode_error = 1

            if sig_mode_error < error_thres:
                n_below_error += 1
                if n_below_error == n_convergence:
                    niters = m + 1
                    break
            else:
                n_below_error = 0

        # Step 8-9 (compute next incidence, replaced by modified Gram-Schmidt)
        E_inc.append(F_sc[-1].copy())
        for p in range(m+1):
            E_inc[-1] -= E_inc[p] * (E_inc[p].conj() @ (E_inc[-1] * w_leb))

    # Compute nonreciprocity

    # Flip map is necessarily symmetric
    S_transpose = _np.block([
        [
            flip_map @ S_approx.T[:N_leb//2, :N_leb//2] @ flip_map,
            - flip_map @ S_approx.T[:N_leb//2, N_leb//2:] @ flip_map
        ],
        [
            - flip_map @ S_approx.T[N_leb//2:, :N_leb//2] @ flip_map,
            flip_map @ S_approx.T[N_leb//2:, N_leb//2:] @ flip_map
        ]
    ])

    E_vecs = eigenvectors[-1]
    Po = _np.empty(n_modes)
    Po_r = _np.empty_like(Po)
    Po_nr = _np.empty_like(Po)

    for m in range(n_modes):
        Po[m] = _np.sum(
            _np.abs((2*modes[-1][m] + 1) * E_vecs[:,m])**2 * w_leb
        )
        Po_r[m] = _np.sum(
            _np.abs(
                (sqw_leb * S_transpose * sqw_leb.T) @ E_vecs[:,m]
                + (modes[-1][m] + 1) * E_vecs[:,m]
            )**2 * w_leb
        )
    Po_nr = Po - Po_r

    return modes, eigenvectors, Po, Po_r, Po_nr, niters


def track_modes(prev_vectors, current_vectors):

    lebedev_deg = prev_vectors.shape[0] // 2
    w_leb = get_lebedev(lebedev_deg)[2]
    w_leb = _np.concatenate((w_leb, w_leb))

    mode_order = _np.full(prev_vectors.shape[1], -1, dtype=_np.int64)

    for m in range(prev_vectors.shape[1]):

        # Use mergesort to keep the relative order of equal correlations, which means prioritizing
        # higher modal significances due to the default ordering from compute_cm_full/iterative.
        correlating_vectors = _np.argsort(
            _np.abs((prev_vectors[:, m].conj() * w_leb) @ current_vectors),
            kind='mergesort'
        )[::-1]

        # Select the not yet selected vector with highest correlation
        for corr_idx in correlating_vectors:
            if corr_idx not in mode_order:
                mode_order[m] = corr_idx
                break

    return mode_order


@_nb.jit(nopython=True, fastmath=True, error_model='numpy', parallel=True)
def _compute_rhs_map(
    theta_leb, phi_leb, k0, basis, quad_points, quad_weights, edge2facet, facet_areas,
    rhs_from_inc
):
    for inc in _nb.prange(theta_leb.shape[0]):  # pylint: disable=not-an-iterable

        direction = _np.array((
            _np.sin(theta_leb[inc]) * _np.cos(phi_leb[inc]),
            _np.sin(theta_leb[inc]) * _np.sin(phi_leb[inc]),
            _np.cos(theta_leb[inc])
        ))
        if inc < theta_leb.shape[0]/2:
            polarization = _np.array((
                _np.cos(theta_leb[inc]) * _np.cos(phi_leb[inc]),
                _np.cos(theta_leb[inc]) * _np.sin(phi_leb[inc]),
                -_np.sin(theta_leb[inc])
            ))
        else:
            polarization = _np.array((
                -_np.sin(phi_leb[inc]),
                _np.cos(phi_leb[inc]),
                0
            ))

        for edge in range(edge2facet.shape[0]):
            for i_facet in range(2):

                jacobian = 2 * facet_areas[edge2facet[edge, i_facet]]

                for i_quad in range(quad_points.shape[1]):

                    rhs_from_inc[edge, inc] += (
                        basis[edge, i_facet, i_quad] @ polarization
                        * _np.exp(
                            -1j * k0
                            * (quad_points[edge2facet[edge, i_facet], i_quad] @ direction)
                        )
                        * quad_weights[i_quad] * jacobian
                    )


@_nb.jit(nopython=True, fastmath=True, error_model='numpy', parallel=True)
def _compute_ff_map(
    theta_leb, phi_leb, k0, basis, quad_points, quad_weights, edge2facet, facet_areas,
    ff_from_M, ff_from_J
):
    half_quad = int(theta_leb.shape[0]/2)
    for out in _nb.prange(half_quad): # pylint: disable=not-an-iterable

        direction = _np.array((
            _np.sin(theta_leb[out]) * _np.cos(phi_leb[out]),
            _np.sin(theta_leb[out]) * _np.sin(phi_leb[out]),
            _np.cos(theta_leb[out])
        ))
        theta_hat = _np.array((
            _np.cos(theta_leb[out]) * _np.cos(phi_leb[out]),
            _np.cos(theta_leb[out]) * _np.sin(phi_leb[out]),
            -_np.sin(theta_leb[out])
        ))
        phi_hat = _np.array((
            -_np.sin(phi_leb[out]),
            _np.cos(phi_leb[out]),
            0
        ))

        for edge in range(edge2facet.shape[0]):

            for i_facet in range(edge2facet.shape[1]):

                jacobian = 2 * facet_areas[edge2facet[edge, i_facet]]

                # No multiplication by eta0 since the solH coeffs are already scaled by that
                for i_quad in range(quad_points.shape[1]):
                    exp_r = 0j
                    scalar_prod_tJ = 0.
                    scalar_prod_pJ = 0.
                    scalar_prod_tM = 0.
                    scalar_prod_pM = 0.

                    for i_coord in range(3):
                        cross_component_M = (
                            direction[(i_coord+1)%3]
                            * basis[edge, i_facet, i_quad, (i_coord-1)%3]
                            - direction[(i_coord-1)%3]
                            * basis[edge, i_facet, i_quad, (i_coord+1)%3]
                        )
                        scalar_prod_tM += theta_hat[i_coord] * cross_component_M
                        scalar_prod_pM += phi_hat[i_coord] * cross_component_M
                        exp_r += (
                            direction[i_coord]
                            * quad_points[edge2facet[edge, i_facet], i_quad, i_coord]
                        )
                        scalar_prod_tJ += theta_hat[i_coord] * basis[edge, i_facet, i_quad, i_coord]
                        scalar_prod_pJ += phi_hat[i_coord] * basis[edge, i_facet, i_quad, i_coord]

                    exp_r = _np.exp(1j * k0 * exp_r)

                    ff_from_M[out, edge] += scalar_prod_tM * exp_r * quad_weights[i_quad] * jacobian
                    ff_from_M[half_quad + out, edge] += \
                        scalar_prod_pM * exp_r * quad_weights[i_quad] * jacobian

                    ff_from_J[out, edge] -= scalar_prod_tJ * exp_r * quad_weights[i_quad] * jacobian
                    ff_from_J[half_quad + out, edge] -= \
                        scalar_prod_pJ * exp_r * quad_weights[i_quad] * jacobian

    ff_from_M *= 1j * k0 / 4 / _np.pi
    ff_from_J *= 1j * k0 / 4 / _np.pi


@_nb.jit(nopython=True, fastmath=True, error_model='numpy')
def _outprod_mvp(left_mat, right_mat, x):
    """
    Performs MVP of outer product representing an mxn matrix.
    left_mat has shape (m,k)
    right_mat has shape (k,n)
    """
    y = _np.zeros_like(x)
    for i in range(right_mat.shape[0]):
        temp = right_mat[i,:] @ x

        # Multiply column i in left_mat by scalar
        for k in range(left_mat.shape[0]):
            y[k] += left_mat[k,i] * temp

    return y


def lebedev_flip_map(theta, phi, tol=1e-12):
    """
    Note! (theta, phi) should be original Lebedev points before doubling due to polarization.
    """

    rhat = _np.column_stack((
        _np.sin(theta) * _np.cos(phi),
        _np.sin(theta) * _np.sin(phi),
        _np.cos(theta)
    ))

    original_index = _np.arange(theta.size)
    flipped_index = _np.empty_like(original_index)

    # For each point, find the corresponding flipped point. Find where this is located in the
    # original array of points
    for i in range(theta.size):

        rhat_flipped = -rhat[i]
        flipped_index[i] = _np.nonzero(_np.linalg.norm(rhat - rhat_flipped, axis=1) < tol)[0][0]

    flip_map = _sparse.coo_array(
        (_np.ones(theta.size), (flipped_index, original_index)),
        dtype=_np.int64
    )

    return flip_map.tocsr()


#########################################################################
# Test below
# import time
# import gmsh
# import matplotlib.pyplot as plt
# from scipy.constants import speed_of_light as c0
# import fe2ms

# tag_sphere = 1
# tag_ext = 2

# ka = 2
# lam = 1.3
# a = ka / (2 * _np.pi / lam)
# freq = c0 / lam
# meshfactor = 0.05

# # Daniel's parameters
# delta = 0.1
# epsr = 1 + delta
# mur = epsr
# alphar = _np.sqrt(delta*(3 + delta))
# kappa = alphar
# chi = 0

# xi = alphar
# zeta = kappa + 1j * chi

# # mats = {tag_sphere: (epsr, mur, xi, zeta)}
# mats = {tag_sphere: (2, 1)}

# gmsh.initialize()
# gmsh.option.setNumber('General.Verbosity', 2)
# gmsh.model.add('sphere')
# # vol_sphere = gmsh.model.occ.addSphere(0, 0, 0, a)
# vol_sphere = gmsh.model.occ.addBox(0, 0, 0, a, a*1.3, a*0.8)
# gmsh.model.occ.synchronize()

# gmsh.model.addPhysicalGroup(3, [vol_sphere], tag_sphere)
# gmsh.model.addPhysicalGroup(
#     2, [b[1] for b in gmsh.model.getBoundary([(3, vol_sphere)], oriented=False)], tag_ext
# )

# # Generate 3D mesh
# gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lam * meshfactor)
# gmsh.model.mesh.generate(3)

# # Save mesh file
# gmsh.write('sphere.msh')
# gmsh.write('sphere.vtk')
# gmsh.finalize()

# cv = fe2ms.ComputationVolume('sphere.msh', mats, [tag_ext])
# system = fe2ms.FEBISystemACA(freq, cv, 'is-efie')

# system.connect_fe_bi()
# print(f'FE size: {system.spaces.fe_size}, BI size: {system.spaces.bi_size}')

# system.assemble()
# print('Assembled')
# M_prec = fe2ms.preconditioners.direct(system)
# print('Preconditioned')

# tick = time.perf_counter()
# modes, evec, Po, Po_r, Po_nr, niters = compute_cm_iterative(
#     system, M_prec, get_lebedev(50), 10, max_iters=300
# )
# print(f'{niters} iterations, {time.perf_counter()-tick:.1f} s')

# tick = time.perf_counter()
# modes_f, evec_f, Po_f, Po_r_f, Po_nr_f = compute_cm_full(
#     system, M_prec, get_lebedev(50), 10
# )
# print(f'Full done, {time.perf_counter()-tick:.1f} s')

# plt.figure()
# plt.plot(_np.abs(modes[-1]), 'b-')
# plt.plot(_np.abs(modes_f), 'r--')
# plt.legend(['Iterative', 'Full'])
# plt.show()

# sigmodes = modes[-1].copy()
# sigmodes[_np.abs(sigmodes) < 1e-4] = _np.nan
# changles = _np.angle(sigmodes)
# changles[changles < 0] += 2*_np.pi

# sigmodes_f = modes_f.copy()
# sigmodes_f[_np.abs(sigmodes_f) < 1e-4] = _np.nan
# changles_f = _np.angle(sigmodes_f)
# changles_f[changles_f < 0] += 2*_np.pi

# plt.figure()
# plt.plot(_np.degrees(changles), 'b-')
# plt.plot(_np.degrees(changles_f), 'r--')
# plt.legend(['Iterative', 'Full'])
# plt.show()

# plt.figure()
# plt.plot(Po, 'b-')
# plt.plot(Po_f, 'r-')
# plt.plot(Po_r,'b--')
# plt.plot(Po_r_f, 'r--')
# plt.legend(['Iter. total', 'Full total', 'Iter. rec.', 'Full rec.'])
# plt.show()
