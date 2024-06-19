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

#%% Script for plotting far fields for a characteristic mode
import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.constants import speed_of_light as c0

import pyvista as pv
import gmsh
import dolfinx, ufl
import fe2ms

from compute_cm import get_lebedev, _compute_rhs_map

pv.set_jupyter_backend('static')

beta_a = 0.01
mode_index = 2
resonance_indices = {
    0:     [25, 82, 82, 115, 115, 244],
    0.005: [25, 88, 103, 136, 160, 245],
    0.01:  [25, 85, 107, 125, 175, 255]
}
freq_index = resonance_indices[beta_a][mode_index]

n_leb = 302
theta_leb, phi_leb, w_leb = get_lebedev(n_leb)

# Files should exist with postfix __modes, __vecs, __plot_modsig, __plot_nonrec
loaddir = '/mnt/c/Users/ni1863wi/Work Folders/Documents/NFFP-7/FE2MS/Figures/Characteristic modes'
loadfile = f'cm_rotcyl_bench__final__beta={beta_a}'

tick = time.perf_counter()
print('Load mode data')
allfreqs = np.loadtxt(f'{loaddir}/{loadfile}__plot_nonrec.csv', delimiter=',')[:,0] * 1e9
freq = allfreqs[freq_index]
vecs = np.loadtxt(f'{loaddir}/{loadfile}__vecs.csv', delimiter=',', dtype=np.complex128)
vecs = vecs.reshape(allfreqs.size, -1, vecs.shape[-1])
print(f'Data loaded, time {time.perf_counter()-tick:.1f} s')

# Do not remove mean phase, the +/- pi values need ot be handled in that case. Actually, it seems
# like numpy.linalg.eig synchronizes the phase just fine
vec_phasesync = vecs[freq_index, :, mode_index].copy()
# mean_phase = np.average(np.angle(vec_phasesync[np.abs(vec_phasesync) > 1e-2]))
# vec_phasesync *= np.exp(-1j * mean_phase)

# Benchmark cylinder
a = 5.25e-3
height = 4.6e-3
epsr_val = 38
mur_val = 1

tag_cyl = 1
tag_ext = 2

lam_min = c0 / freq
meshfactor = 0.08

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
# Simulate
tick0 = time.perf_counter()

omega = beta_a * c0 / a

# CV is only used for mesh and external facet access in this case
cv = fe2ms.ComputationVolume('cyl.msh', {tag_cyl: (epsr_val, mur_val)}, [tag_ext])

system = fe2ms.FEBISystemFull(freq, cv, 'teth')
system.connect_fe_bi()

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
B_matrix = 1j * k0 * sparse.coo_array((B_vals, (rows, cols)), shape=2*(system.spaces.bi_size,)).tocsr()

# Boundary integral blocks
P_matrix, Q_matrix, system.K_prec, system.L_prec = fe2ms.assembly.assemble_bi_blocks_full(
    k0, system.spaces.bi_meshdata, system.spaces.bi_basisdata, quad_order_singular=5
)
system._system_blocks = fe2ms.utility.FEBIBlocks(K_matrix, B_matrix, P_matrix, Q_matrix)
system._system_lufactor = None

# RHS for the current mode
rhs_from_inc = np.zeros((system.spaces.bi_size, n_leb*2), dtype=np.complex128)
_compute_rhs_map(
    theta_leb, phi_leb, system._k0, system.spaces.bi_basisdata.basis,
    system.spaces.bi_basisdata.quad_points, system.spaces.bi_basisdata.quad_weights,
    system.spaces.bi_meshdata.edge2facet, system.spaces.bi_meshdata.facet_areas,
    rhs_from_inc
)
Einc = vec_phasesync * (4 * np.pi * 1j / k0)
system._rhs = np.concatenate((
    np.zeros(system.spaces.fe_size), rhs_from_inc @ (Einc * np.concatenate(2*(w_leb, )))
))
print(f'Assembly time: {timedelta(seconds=round(time.perf_counter() - tick))}')
tick = time.perf_counter()
system.solve_iterative()
print(f'Solution time: {timedelta(seconds=round(time.perf_counter() - tick))}')

tick = time.perf_counter()
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 200)
theta_i, phi_i = np.meshgrid(theta, phi, indexing='ij')
theta_i = theta_i.ravel()
phi_i = phi_i.ravel()
farfield = system.compute_far_field(1., theta_i, phi_i) * np.exp(1j * k0)
print(f'Far field time: {timedelta(seconds=round(time.perf_counter() - tick))}')

print(f'Total simulation time: {timedelta(seconds=round(time.perf_counter() - tick0))}')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Prepare to visualize
farfield_1 = farfield.reshape((theta.size, phi.size, -1))
farfield_abs = np.linalg.norm(farfield_1, axis=-1)

theta_hat_i = np.column_stack(
    (np.cos(theta_i) * np.cos(phi_i), np.cos(theta_i) * np.sin(phi_i), -np.sin(theta_i))
)
phi_hat_i = np.column_stack(
    (-np.sin(phi_i), np.cos(phi_i), np.zeros_like(theta_i))
)
farfield_theta_i = np.sum(farfield * theta_hat_i, axis=-1)
farfield_phi_i = np.sum(farfield * phi_hat_i, axis=-1)

# Make farfields from eigenvectors (up to some factor common to all entries)
farfield_theta = vec_phasesync[:n_leb] / np.sqrt(w_leb)
farfield_phi = vec_phasesync[n_leb:] / np.sqrt(w_leb)
theta_hat = np.column_stack(
    (np.cos(theta_leb) * np.cos(phi_leb), np.cos(theta_leb) * np.sin(phi_leb), -np.sin(theta_leb))
)
phi_hat = np.column_stack(
    (-np.sin(phi_leb), np.cos(phi_leb), np.zeros_like(theta_leb))
)
farfield_leb = farfield_theta[:,None] * theta_hat + farfield_phi[:,None] * phi_hat
farfield_leb_abs = np.linalg.norm(farfield_leb, axis=-1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Actually plot
# Axes are theta down, phi right. But phi is x axis and theta is y axis

# farfield_theta_i = 1 / np.sqrt(2) * (farfield_theta_mode2 + 1j * farfield_theta_mode1)
# farfield_phi_i = 1 / np.sqrt(2) * (farfield_phi_mode2 + 1j * farfield_phi_mode1)
# farfield_abs = np.linalg.norm(np.column_stack((farfield_theta_i, farfield_phi_i)),axis=-1).reshape((theta.size,phi.size))

normval = farfield_abs.max()
plt.figure(figsize=(12,5))
phist = phi[1] - phi[0]
thetast = theta[1] - theta[0]
plt.imshow(
    farfield_abs/normval, interpolation='none',
    extent=(phi[0]-phist/2, phi[-1]+phist/2, theta[-1]+thetast/2, theta[0]-thetast/2),
    cmap='inferno'
)
plt.clim((0,1))
# plt.scatter(phi_leb, theta_leb, c=farfield_leb_abs/farfield_leb_abs.max(), cmap='inferno')
# plt.clim((0,1))
plt.colorbar(label='Far field amplitude (normalized)')
# plt.quiver(
#     phi_leb, theta_leb,
#     farfield_phi.real, farfield_theta.real,
#     pivot='middle', color='gray', angles='xy',
#     scale=50, headlength=3, headaxislength=3, headwidth=3, width=3e-3
# )
# plt.quiver(
#     phi_leb, theta_leb,
#     farfield_phi.imag, farfield_theta.imag,
#     pivot='middle', color='blue', angles='xy',
#     scale=50, headlength=3, headaxislength=3, headwidth=3, width=3e-3
# )
arrow_i = np.zeros_like(farfield_abs, dtype=np.bool8)
arrow_i[5::10, 5::10] = True
arrow_i = arrow_i.ravel()
plt.quiver(
    phi_i[arrow_i], theta_i[arrow_i],
    farfield_phi_i[arrow_i].real, farfield_theta_i[arrow_i].real,
    pivot='middle', color='k', angles='xy',
    scale=0.05, headlength=3, headaxislength=3, headwidth=3, width=5e-3
)
plt.quiver(
    phi_i[arrow_i], theta_i[arrow_i],
    farfield_phi_i[arrow_i].imag, farfield_theta_i[arrow_i].imag,
    pivot='middle', color='g', angles='xy',
    scale=0.05, headlength=3, headaxislength=3, headwidth=3, width=5e-3
)
pilabels = ['0', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$']
plt.xticks(np.linspace(0,2*np.pi,5), labels=pilabels)
plt.yticks(np.linspace(0,np.pi,3), labels=pilabels[:3])
plt.xlabel('$\\phi$')
plt.ylabel('$\\theta$')
plt.axis('equal')
plt.show()

# arrowscale = 4 * np.max((
#     np.sqrt(farfield_phi_i[arrow_i].real**2 + farfield_theta_i[arrow_i].real**2),
#     np.sqrt(farfield_phi_i[arrow_i].imag**2 + farfield_theta_i[arrow_i].imag**2)
# ))
# np.savetxt(
#     f'{loaddir}/farfield_beta={beta_a}_mode{mode_index}_arrows_Re.csv',
#     np.column_stack((
#         phi_i[arrow_i], theta_i[arrow_i],
#         farfield_phi_i[arrow_i].real / arrowscale,
#         farfield_theta_i[arrow_i].real / arrowscale
#     )),
#     delimiter=',', header='x,y,u,v', comments=''
# )
# np.savetxt(
#     f'{loaddir}/farfield_beta={beta_a}_mode{mode_index}_arrows_Im.csv',
#     np.column_stack((
#         phi_i[arrow_i], theta_i[arrow_i],
#         farfield_phi_i[arrow_i].imag / arrowscale,
#         farfield_theta_i[arrow_i].imag / arrowscale
#     )),
#     delimiter=',', header='x,y,u,v', comments=''
# )


# plt.imshow(
#     farfield_abs/normval, interpolation='none',
#     cmap='inferno'
# )
# plt.clim(0,1)
# plt.axis('equal')
# plt.savefig(f'{loaddir}/farfield_beta={beta_a}_mode{mode_index}_amplitude.pgf')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Pyvista interpolation on a unit sphere
save = False
x_i = np.sin(theta_i) * np.cos(phi_i)
y_i = np.sin(theta_i) * np.sin(phi_i)
z_i = np.cos(theta_i)
points = pv.PolyData(np.column_stack((x_i, y_i, z_i)))
points['F_Re'] = farfield_theta_i.real[:,None] * theta_hat_i \
    + farfield_theta_i.imag[:,None] * phi_hat_i
points['F_Im'] = farfield_phi_i.real[:,None] * theta_hat_i \
    + farfield_phi_i.imag[:,None] * phi_hat_i

# Amplitude and phase are computed after interpolation
points['F_Abs'] = np.linalg.norm(points['F_Re'] + 1j*points['F_Im'], axis=-1)

if save:
    points.save(
        f'{loaddir}/farfields/farfield__points__beta={beta_a}__mode_index={mode_index}.vtk'
    )