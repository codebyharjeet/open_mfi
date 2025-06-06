import numpy as np
import openfermion as of 
import string
import itertools

from mfi import *

x_dimension = 6
y_dimension = 1
n_qubits = x_dimension * y_dimension

H = of.hamiltonians.fermi_hubbard(x_dimension=x_dimension,y_dimension=y_dimension,tunneling=1.0,coulomb=2.0,chemical_potential=0.0,magnetic_field=0.0,periodic=False,spinless=True,particle_hole_symmetry=False)

H_sparse = of.linalg.get_sparse_operator(H, n_qubits=n_qubits)
E_exact, psi_exact = of.linalg.get_ground_state(H_sparse)
rho_exact = np.outer(psi_exact, psi_exact.conj())

print(f"Shape of fermi hubbard hamiltonian with x={x_dimension} and y={y_dimension} = {H_sparse.shape}")
print(f"Shape of exact rho = {rho_exact.shape}")
print(f"Exact energy from Exact Diag. = {E_exact:.8f} Hartree")
print(f"Exact energy from Tr(H@rho) = {(np.trace(H_sparse@rho_exact)).real:.8f} Hartree")

H_dense = H_sparse.toarray()

rho_rebuilt, rho_mf = cluster_expansion_rho(rho_exact, H_dense)

print("‖rho_exact - rho_mf‖  = ",np.linalg.norm(rho_exact - rho_mf))
print(f"Approx. energy from Tr(H@rho_mf) = {(np.trace(H_sparse@rho_mf)).real:.8f} Hartree")

print("‖rho_exact - rho_rebuilt‖  = ",np.linalg.norm(rho_exact - rho_rebuilt))
print(f"Approx. energy from Tr(H@rho_rebuilt) = {(np.trace(H_sparse@rho_rebuilt)).real:.8f} Hartree")

"""
Output:
Shape of fermi hubbard hamiltonian with x=6 and y=1 = (64, 64)
Shape of exact rho = (64, 64)
Exact energy from Exact Diag. = -2.90280999 Hartree
Exact energy from Tr(H@rho) = -2.90280999 Hartree
‖rho_exact - rho_mf‖  =  0.9868745152156453
Approx. energy from Tr(H@rho_mf) = 1.12263766 Hartree
‖rho_exact - rho_rebuilt‖  =  1.569824710173597
Approx. energy from Tr(H@rho_rebuilt) = -10.13948952 Hartree
"""
