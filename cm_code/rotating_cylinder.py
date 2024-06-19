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
# Script for computing characteristic modes of a rotating dielectric cylinder
#
#
# Initialize
import time
import gc
import numpy as np
import gmsh
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light as c0
from scipy import sparse
import ufl
import dolfinx
import fe2ms
from compute_cm import compute_cm_iterative, get_lebedev, track_modes

tag_cyl = 1
tag_ext = 2

beta_a = 0.03 # Omega * a / c0
ka_min = 0.5
ka_max = 2
nka = 31
lebquad = get_lebedev(350)
nmodes = 20

a = 1
height = a
omega = beta_a * c0 / a
epsr_val = 10
mur_val = 1

lam_min = 2 * np.pi * a / ka_max
meshfactor = 0.08

loadfile = None
save = None

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Make mesh

print('Start meshing')
tick = time.perf_counter()

gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 2)
gmsh.model.add('sphere')
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
print(f'Meshing time: {time.perf_counter()-tick:0.1f} s')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Iterate over frequencies

kas = np.linspace(ka_min, ka_max, nka)

allmodes = np.empty((kas.size, nmodes), dtype=np.complex128)
allvecs = np.empty((kas.size, 2*lebquad[-1].size, nmodes), dtype=np.complex128)

Po = np.empty((kas.size, nmodes))
Por = np.empty_like(Po)
Ponr = np.empty_like(Po)
nonrec = np.empty_like(Po)

if loadfile is None:

    tick0 = time.perf_counter()
    for i, ka in enumerate(kas):

        print('\n\n')
        print('*************************************************')
        print(f'ka = {ka}')
        print(f'Progress {time.strftime("%H:%M:%S", time.localtime())}: {i/len(kas)*100:.1f} %')
        if i > 0:
            print(f'Remaining: {((len(kas) - i) * (time.perf_counter() - tick0) / i)/60:.0f} min')
        print('*************************************************')

        # CV is only used for mesh and external facet access in this case
        cv = fe2ms.ComputationVolume('cyl.msh', {tag_cyl: (epsr_val, mur_val)}, [tag_ext])

        system = fe2ms.FEBISystemFull(ka * c0 / 2 / np.pi / a, cv, 'teth')
        system.connect_fe_bi()
        print(f'FE size: {system.spaces.fe_size}, BI size: {system.spaces.bi_size}')

        # Now do assembly explicitly with modifications
        tick = time.perf_counter()
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
        print(f'Assembly time: {time.perf_counter()-tick:.1f} s')

        tick = time.perf_counter()
        M_prec = fe2ms.preconditioners.direct(system)
        print(f'Preconditioner generation time: {time.perf_counter()-tick:0.1f} s')

        tick = time.perf_counter()
        modes, vecs, Po_i, Por_i, Ponr_i, niters = compute_cm_iterative(
            system, M_prec, lebquad, nmodes, max_iters=300
        )
        print(f'CM run time: {time.perf_counter()-tick:.1f} s')
        if niters > 0:
            print(f'{niters} iterations')
        else:
            print('!!!!!!!!!!!!!!!!!!!!')
            print('No convergence!')
            print('!!!!!!!!!!!!!!!!!!!!')
        
        # Output gives full iteration progress, only keep the last ones
        modes = modes[-1]
        vecs = vecs[-1]

        # Track modes after the first ka
        if i > 0:
            mode_order = track_modes(allvecs[i-1], vecs)
            allmodes[i,:] = modes[mode_order]
            allvecs[i,:,:] = vecs[:,mode_order]
            Po[i,:] = Po_i[mode_order]
            Por[i,:] = Por_i[mode_order]
            Ponr[i,:] = Ponr_i[mode_order]
        else:
            allmodes[i,:] = modes
            allvecs[i,:,:] = vecs
            Po[i,:] = Po_i
            Por[i,:] = Por_i
            Ponr[i,:] = Ponr_i
        
        # Fix garbage collection when using jupyter interactive console
        del system, M_prec
        gc.collect()

    print()
    print(f'Total time: {time.perf_counter()-tick0:.1f} s')

    nonrec = Ponr / Po

    if save:
        hdr = f'a={a}, height={height*1.13/a}a, beta_a={beta_a:.1g}, ka={kas[0]},{kas[1]},...,{kas[-1]}'
        np.savetxt(
            f'cm_rotcylinder_{time.strftime("%Y%m%d_%H%M")}.csv', allmodes, delimiter=',', header=hdr
        )

else:
    allmodes = np.loadtxt(loadfile, dtype=np.complex128, delimiter=',')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot

plt.figure()
plt.plot(kas, np.abs(allmodes))
plt.title('Modal significance')
plt.xlabel('ka')
plt.ylabel('$|t_n|$')
plt.show()

allsigmodes = allmodes.copy()
allsigmodes[np.abs(allsigmodes) < 1e-4] = np.nan
ch_angles = np.angle(allsigmodes)
ch_angles[ch_angles < 0] += 2*np.pi

plt.figure()
plt.plot(kas, np.degrees(ch_angles))
plt.yticks(np.arange(90, 271, 30))
plt.title('Characteristic angles for $|t_n|>1e-4$')
plt.xlabel('ka')
plt.ylabel('$\\alpha_n$ [deg]')
plt.show()

plt.figure()
line1 = plt.plot(kas, nonrec)
plt.title('Nonreciprocity')
plt.xlabel('ka')
plt.ylabel('$P_o^{nr} / P_o$')
plt.show()
