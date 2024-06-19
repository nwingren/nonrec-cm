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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Script for computing characteristic modes of a rotating dielectric cylinder with
# adaptive frequency sampling
#
#
# Initialize
import time
import gc
import logging
from datetime import timedelta
from operator import itemgetter

import numpy as np
import gmsh
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.constants import speed_of_light as c0
from scipy import sparse

import ufl
import dolfinx

import adaptive

import fe2ms
from compute_cm import compute_cm_iterative, get_lebedev, track_modes

tag_cyl = 1
tag_ext = 2

f_min = 4e9
f_max = 8e9
lebquad = get_lebedev(302)
nmodes = 35

# Parameters defining when to end frequency sampling
# loss_goal = 0.005
loss_goal = 0.001
timeout = timedelta(hours=48)

# Benchmark cylinder
a = 5.25e-3
height = 4.6e-3
epsr_val = 38
mur_val = 1

lam_min = c0 / f_max
meshfactor = 0.08

beta_a_values = [0.01] # [0, 0.005, 0.01] # Omega * a / c0

savedir = 'modes_adaptive'
savefile = (
    f'cm_rotcyl_bench__'
    f'{time.strftime("%y%m%d_%Hh%M", time.localtime())}'
)
# savefile = None

# Logging things
logger = logging.getLogger('CM')
logger.propagate = False
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = logging.FileHandler(f'{savefile}.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

logger.info('Start simulation')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Make mesh
gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 2)
gmsh.model.add('cyl')
vol_cyl = gmsh.model.occ.addCylinder(0, 0, -height/2, 0, 0, height, a)
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(3, [vol_cyl], tag_cyl)
gmsh.model.addPhysicalGroup(
    2, [b[1] for b in gmsh.model.getBoundary([(3, vol_cyl)], oriented=False)], tag_ext
)

# Generate 3D mesh
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lam_min * meshfactor / np.sqrt(epsr_val))
gmsh.model.mesh.generate(3)

# Save mesh file
gmsh.write('cyl.msh')
gmsh.write('cyl.vtk')
gmsh.finalize()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run big loop over beta_a values

freqs_betas = {}
modes_betas = {}
vecs_betas = {}
nonrec_betas = {}
learners_betas = {}

for beta_a in beta_a_values:

    logger.info('***********************************************')
    logger.info(f'Starting simulations for beta_a = {beta_a:.1g}')
    logger.info('***********************************************')

    omega = beta_a * c0 / a

    ##########################################################################################
    ##########################################################################################
    # Define function that goes into adaptive learner

    def compute_for_freq(f):

        logger.info(f'Simulating at f = {f/1e9:.3g} GHz')
        tick_i = time.perf_counter()

        # CV is only used for mesh and external facet access in this case
        cv = fe2ms.ComputationVolume('cyl.msh', {tag_cyl: (epsr_val, mur_val)}, [tag_ext])

        system = fe2ms.FEBISystemFull(f, cv, 'teth')
        system.connect_fe_bi()

        # Now do assembly explicitly with modifications
        k0 = system._k0
        x = ufl.SpatialCoordinate(cv.mesh)
        xi = omega * (epsr_val * mur_val - 1) / c0 * ufl.as_matrix((
            (0, 0, x[0]),
            (0, 0, x[1]),
            (-x[0], -x[1], 0)
        ))
        zeta = omega * (epsr_val * mur_val - 1) / c0 * ufl.as_matrix((
            (0, 0, -x[0]),
            (0, 0, -x[1]),
            (x[0], x[1], 0)
        ))
        epsr = epsr_val * ufl.Identity(3)
        inv_mur = 1 / mur_val * ufl.Identity(3)

        # Finite element operator
        u = ufl.TrialFunction(system.spaces.fe_space)
        v = ufl.TestFunction(system.spaces.fe_space)

        # Generate and assemble K matrix
        form = dolfinx.fem.form(
            (
                ufl.inner(inv_mur * ufl.curl(u), ufl.curl(v))
                - k0**2 * ufl.inner((epsr - xi * inv_mur * zeta) * u, v)
                + 1j * k0 * ufl.inner((inv_mur * zeta) * u, ufl.curl(v))
                - 1j * k0 * ufl.inner((xi * inv_mur) * ufl.curl(u), v)
            )
            * ufl.dx
        )
        K_matrix = dolfinx.fem.assemble_matrix(form, bcs=[])

        K_matrix.finalize() # pylint: disable=no-member
        K_matrix = sparse.csr_array((K_matrix.data, K_matrix.indices, K_matrix.indptr)) # pylint: disable=no-member

        rows, cols, B_vals = fe2ms.assembly_nonsingular_full.assemble_B_integral(
            system.spaces.bi_basisdata.basis, system.spaces.bi_basisdata.quad_weights,
            system.spaces.bi_meshdata.facet2edge, system.spaces.bi_meshdata.edge2facet,
            system.spaces.bi_meshdata.facet_areas, system.spaces.bi_meshdata.facet_normals
        )
        B_matrix = 1j * k0 *sparse.coo_array((B_vals, (rows, cols)), shape=2*(system.spaces.bi_size,)).tocsr()

        # Boundary integral blocks
        P_matrix, Q_matrix, system.K_prec, system.L_prec = fe2ms.assembly.assemble_bi_blocks_full(
            k0, system.spaces.bi_meshdata, system.spaces.bi_basisdata, quad_order_singular=5
        )
        system._system_blocks = fe2ms.utility.FEBIBlocks(K_matrix, B_matrix, P_matrix, Q_matrix)
        system._system_lufactor = None

        M_prec = fe2ms.preconditioners.direct(system)

        modes_i, vecs_i, Po, Por, Ponr, niters = compute_cm_iterative(
            system, M_prec, lebquad, nmodes, max_iters=300
        )
        timesec = round(time.perf_counter() - tick_i)
        if niters > 0:
            logger.info(
                f'{niters} CM iterations, '
                f'time for freq. sample: {timedelta(seconds=timesec)} ({timesec} seconds)'
            )
            error = False
        else:
            logger.info(
                'No convergence! '
                f'Time for freq. sample: {timedelta(seconds=timesec)} ({timesec} seconds)'
            )
            error = True

        # Output gives full iteration progress, only keep the last ones
        modes_i = modes_i[-1]
        vecs_i = vecs_i[-1]

        # Quantity to compute loss for should only include significant modes
        modsig = np.abs(modes_i)
        modsig[modsig < 1e-2] = 0

        nonrec_i = Ponr / Po

        del system, M_prec
        gc.collect()

        if savefile is not None:
            try:
                data_modsig = np.loadtxt(
                    f'{savedir}/{savefile}__beta={beta_a:.1g}__plot_modsig.csv', delimiter=','
                )
                data_modsig = np.row_stack((
                    data_modsig,
                    np.column_stack((np.array((f,)), np.abs(modes_i)[None,:]))
                ))
                data_nonrec = np.loadtxt(
                    f'{savedir}/{savefile}__beta={beta_a:.1g}__plot_nonrec.csv', delimiter=','
                )
                data_nonrec = np.row_stack((
                    data_nonrec,
                    np.column_stack((np.array((f,)), np.abs(nonrec_i)[None,:]))
                ))
                data_modes = np.loadtxt(
                    f'{savedir}/{savefile}__beta={beta_a:.1g}__modes.csv', delimiter=',',
                    dtype=np.complex128
                )
                data_modes = np.row_stack((data_modes, modes_i[None,:]))
                data_vecs = np.loadtxt(
                    f'{savedir}/{savefile}__beta={beta_a:.1g}__vecs.csv', delimiter=',',
                    dtype=np.complex128
                )
                data_vecs = np.row_stack((data_vecs, vecs_i))
            except FileNotFoundError:
                data_modsig = np.column_stack((np.array((f,)), np.abs(modes_i)[None,:]))
                data_nonrec = np.column_stack((np.array((f,)), np.abs(nonrec_i)[None,:]))
                data_modes = modes_i[None,:]
                data_vecs = vecs_i

            hdr = f'Details: epsr={epsr_val:.1g} (temporary saved file)'
            np.savetxt(
                f'{savedir}/{savefile}__beta={beta_a:.1g}__plot_modsig.csv', data_modsig,
                delimiter=',',
                header=hdr+f'\f[GHz],Modal significances at all f for mode 0...mode {nmodes-1}'
            )
            np.savetxt(
                f'{savedir}/{savefile}__beta={beta_a:.1g}__plot_nonrec.csv', data_nonrec,
                delimiter=',',
                header=hdr+f'\nf[GHz],Nonreciprocities at all f for mode 0...mode {nmodes-1}'
            )
            np.savetxt(
                f'{savedir}/{savefile}__beta={beta_a:.1g}__modes.csv', data_modes,
                delimiter=',',
                header=hdr+f'\nComplex modes at all f for mode 0...mode {nmodes-1}'
            )
            np.savetxt(
                f'{savedir}/{savefile}__beta={beta_a:.1g}__vecs.csv', data_vecs,
                delimiter=',',
                header=hdr+f'\nReshaped eigenvectors at all f for mode 0...mode {nmodes-1}'
            )

        return {
            'modsig': modsig, 'modes': modes_i, 'vecs': vecs_i, 'nonrec': nonrec_i,
            'error': error
        }

    ##########################################################################################
    ##########################################################################################
    # Do simulations with adaptive sampling of ka

    # Previously used curvature loss function since most of the dynamics seem to happen near
    # high curvatures
    # Now back to default loss with more points instead to try to avoid some tracking issues and
    # such that might be due to large changes in modal significance that are resolved with only
    # a few points for curvature loss
    # _learner = adaptive.Learner1D(
    #     compute_for_freq, bounds=(f_min, f_max),
    #     loss_per_interval=adaptive.learner.learner1D.curvature_loss_function()
    # )
    _learner = adaptive.Learner1D(compute_for_freq, bounds=(f_min, f_max))
    learner = adaptive.DataSaver(_learner, arg_picker=itemgetter('modsig'))

    # Custom goal that combines a loss goal with a timeout
    timeout_goal = adaptive.runner.auto_goal(duration=timeout, learner=learner)
    def loss_timeout_goal(learner_0):
        logger.info(f'Current loss = {learner_0.loss():.3g}')
        if learner_0.loss() <= loss_goal:
            logger.info('Loss goal reached!')
            return True
        if timeout_goal(learner_0):
            logger.info('Timed out!')
            return True
        return False

    tick = time.perf_counter()

    # Use the simple runner to ensure reproducibility as well as not run anything in parallel here
    # (I use parallelization within fe2ms already so it shouldn't give anything here)
    adaptive.runner.simple(learner, goal=loss_timeout_goal)

    logger.info(
        f'Total time for beta_a = {beta_a:.1g}: '
        f'{timedelta(seconds=round(time.perf_counter() - tick))}'
    )

    ##########################################################################################
    ##########################################################################################
    # Process the results

    freqs = np.array(list(learner.extra_data))
    modes = np.array([d['modes'] for d in learner.extra_data.values()])
    vecs = np.array([d['vecs'] for d in learner.extra_data.values()])
    nonrec = np.array([d['nonrec'] for d in learner.extra_data.values()])

    # Sort the data
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    modes = modes[sort_idx]
    vecs = vecs[sort_idx, :]
    nonrec = nonrec[sort_idx]

    # DO NOT perform mode tracking to make sure that the data is raw
    # # Perform mode tracking
    # for i in range(1, freqs.size):
    #     mode_order = track_modes(vecs[i-1,:,:], vecs[i,:,:])
    #     modes[i,:] = modes[i,mode_order]
    #     vecs[i,:,:] = vecs[i][:,mode_order]
    #     nonrec[i,:] = nonrec[i,mode_order]

    freqs_betas[beta_a] = freqs
    modes_betas[beta_a] = modes
    vecs_betas[beta_a] = vecs
    nonrec_betas[beta_a] = nonrec
    learners_betas[beta_a] = learner

    if savefile is not None:

        hdr = f'Details: epsr={epsr_val:.1g} f_min={f_min:.1g} f_max={f_max:.1g} n_f={freqs.size}'
        np.savetxt(
            f'{savedir}/{savefile}__beta={beta_a:.1g}__plot_modsig.csv',
            np.column_stack((freqs/1e9, np.abs(modes))),
            delimiter=',',
            header=hdr+f'\f[GHz],Modal significances at all f for mode 0...mode {modes.shape[1]}'
        )
        np.savetxt(
            f'{savedir}/{savefile}__beta={beta_a:.1g}__plot_nonrec.csv',
            np.column_stack((freqs/1e9, nonrec)),
            delimiter=',',
            header=hdr+f'\nf[GHz],Nonreciprocities at all f for mode 0...mode {modes.shape[1]}'
        )
        np.savetxt(
            f'{savedir}/{savefile}__beta={beta_a:.1g}__modes.csv', modes,
            delimiter=',',
            header=hdr+f'\nComplex modes at all f for mode 0...mode {modes.shape[1]}'
        )
        np.savetxt(
            f'{savedir}/{savefile}__beta={beta_a:.1g}__vecs.csv',
            vecs.reshape(-1, vecs.shape[-1]),
            delimiter=',',
            header=hdr+f'\nReshaped eigenvectors at all f for mode 0...mode {modes.shape[1]}'
        )

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot results

multi_cycler = (
    cycler(marker=['', 'x', 'd'])
    * cycler(linestyle=['-', '--', ':'])
    * cycler(color=['r', 'g', 'b'])
)

plt.figure()
plt.gca().set_prop_cycle(multi_cycler)
plt.plot(freqs/1e9, np.abs(modes))
plt.title('Modal significance')
plt.xlabel('f [GHz]')
plt.ylabel('$|t_n|$')
plt.show()

sigmodes = modes.copy()
sigmodes[np.abs(sigmodes) < 1e-4] = np.nan
ch_angles = np.angle(sigmodes)
ch_angles[ch_angles < 0] += 2*np.pi

plt.figure()
plt.gca().set_prop_cycle(multi_cycler)
plt.plot(freqs/1e9, np.degrees(ch_angles))
plt.yticks(np.arange(90, 271, 30))
plt.title('Characteristic angles for $|t_n|>1e-4$')
plt.xlabel('f [GHz]')
plt.ylabel('$\\alpha_n$ [deg]')
plt.show()

plt.figure()
plt.gca().set_prop_cycle(multi_cycler)
plt.plot(freqs/1e9, nonrec)
plt.title('Nonreciprocity')
plt.xlabel('f [GHz]')
plt.ylabel('$P_o^{nr} / P_o$')
plt.show()
