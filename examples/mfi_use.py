import numpy as np
import openfermion as of 
import string
import itertools
import open_mfi
from open_mfi import *

import time
t0 = time.time()

x_dimension = 4
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

C = ClusterExpansion(rho_exact, H_dense, n_qubits=n_qubits, verbose=0)
rho_mf           = C.mean_field_state()
print("‖rho_exact - rho_mf‖  = ",np.linalg.norm(rho_exact - rho_mf))
print(f"Approx. energy from Tr(H@rho_mf) = {(np.trace(H_sparse@rho_mf)).real:.8f} Hartree")

#Two body
print("\n**************** - Two body - ****************")

rho_rebuilt, _   = C.cluster_expansion_rho(compute_3q_cumulants=False, compute_4q_cumulants=False)

print("‖rho_exact - rho_rebuilt‖  = ",np.linalg.norm(rho_exact - rho_rebuilt))
fidelity = np.trace(rho_exact @ rho_rebuilt)
print("(rho_exact | rho_rebuilt)  = %12.8f + %12.8fi" %(np.real(fidelity), np.imag(fidelity)))
print(f"Approx. energy from Tr(H@rho_rebuilt) = {(np.trace(H_sparse@rho_rebuilt)).real:.8f} Hartree")

# print("Eigenvalues of rho:")
# rho_eig_exact = np.linalg.eigvals(rho_exact)
# rho_eig_rebuilt = np.linalg.eigvals(rho_rebuilt)
# print(" %2s %12s %12s" %("", "Exact", "Rebuilt"))
# for i in range(len(rho_eig_exact)):
#     print(" %2i %12.8f %12.8f" %(i, np.real(rho_eig_exact[i]), np.real(rho_eig_rebuilt[i])))

print(" Deviation from Hermiticity  = %12.8f" %np.linalg.norm(rho_rebuilt - rho_rebuilt.conj().T))
print(" Trace(rho_exact)            = %12.8f" %np.abs(np.trace(rho_exact)))
print(" Trace(rho_rebuilt)          = %12.8f" %np.abs(np.trace(rho_rebuilt)))

#Three body
print("\n**************** - Three body - ***************")
rho_rebuilt, _   = C.cluster_expansion_rho(compute_3q_cumulants=True, compute_4q_cumulants=False)

print("‖rho_exact - rho_rebuilt‖  = ",np.linalg.norm(rho_exact - rho_rebuilt))
fidelity = np.trace(rho_exact @ rho_rebuilt)
print("(rho_exact | rho_rebuilt)  = %12.8f + %12.8fi" %(np.real(fidelity), np.imag(fidelity)))
print(f"Approx. energy from Tr(H@rho_rebuilt) = {(np.trace(H_sparse@rho_rebuilt)).real:.8f} Hartree")

# print("Eigenvalues of rho:")
# rho_eig_exact = np.linalg.eigvals(rho_exact)
# rho_eig_rebuilt = np.linalg.eigvals(rho_rebuilt)
# print(" %2s %12s %12s" %("", "Exact", "Rebuilt"))
# for i in range(len(rho_eig_exact)):
#     print(" %2i %12.8f %12.8f" %(i, np.real(rho_eig_exact[i]), np.real(rho_eig_rebuilt[i])))

print(" Deviation from Hermiticity  = %12.8f" %np.linalg.norm(rho_rebuilt - rho_rebuilt.conj().T))
print(" Trace(rho_exact)            = %12.8f" %np.abs(np.trace(rho_exact)))
print(" Trace(rho_rebuilt)          = %12.8f" %np.abs(np.trace(rho_rebuilt)))


#Four body
print("\n**************** - Four body - ****************")
rho_rebuilt, _   = C.cluster_expansion_rho(compute_3q_cumulants=True, compute_4q_cumulants=True)

print("‖rho_exact - rho_rebuilt‖  = ",np.linalg.norm(rho_exact - rho_rebuilt))
fidelity = np.trace(rho_exact @ rho_rebuilt)
print("(rho_exact | rho_rebuilt)  = %12.8f + %12.8fi" %(np.real(fidelity), np.imag(fidelity)))
print(f"Approx. energy from Tr(H@rho_rebuilt) = {(np.trace(H_sparse@rho_rebuilt)).real:.8f} Hartree")

# print("Eigenvalues of rho:")
# rho_eig_exact = np.linalg.eigvals(rho_exact)
# rho_eig_rebuilt = np.linalg.eigvals(rho_rebuilt)
# print(" %2s %12s %12s" %("", "Exact", "Rebuilt"))
# for i in range(len(rho_eig_exact)):
#     print(" %2i %12.8f %12.8f" %(i, np.real(rho_eig_exact[i]), np.real(rho_eig_rebuilt[i])))

print(" Deviation from Hermiticity  = %12.8f" %np.linalg.norm(rho_rebuilt - rho_rebuilt.conj().T))
print(" Trace(rho_exact)            = %12.8f" %np.abs(np.trace(rho_exact)))
print(" Trace(rho_rebuilt)          = %12.8f" %np.abs(np.trace(rho_rebuilt)))

# print(C.get_memory_usage())


t1 = time.time()
elapsed = t1 - t0
minutes, seconds = divmod(elapsed, 60)

print(f"Elapsed time: {int(minutes)} minutes {seconds:.4f} seconds")