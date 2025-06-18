import numpy as np
import openfermion as of 
import string
import itertools
import open_mfi
from open_mfi import *

import time
t0 = time.time()

x_dimension = 6
y_dimension = 1
n_qubits = x_dimension * y_dimension

H = of.hamiltonians.fermi_hubbard(x_dimension=x_dimension,y_dimension=y_dimension,tunneling=1.0,coulomb=2.0,chemical_potential=0.0,magnetic_field=0.0,periodic=False,spinless=True,particle_hole_symmetry=False)


H_sparse = of.linalg.get_sparse_operator(H, n_qubits=n_qubits)
H_dense = H_sparse.toarray()
E_exact, psi_exact = of.linalg.get_ground_state(H_sparse)
rho_exact = np.outer(psi_exact, psi_exact.conj())

dim = 2 ** n_qubits
rho_guess = np.eye(dim) / dim 

print(f"Shape of fermi hubbard hamiltonian with x={x_dimension} and y={y_dimension} = {H_sparse.shape}")
print(f"Shape of rho rebuilt = {rho_guess.shape}")
print(f"Exact energy from Exact Diag. = {E_exact:.8f} Hartree")
print(f"Initial energy from Tr(H@rho_guess) = {(np.trace(H_sparse@rho_guess)).real:.8f} Hartree")

#Two body
print("\n**************** - Two body - ****************")
energy_prev = None
energy_tol = 1e-7  # Energy convergence threshold
for i in range(50):
    print(f"\nIteration: {i}")
    C = ClusterExpansionApprox(rho_guess, H_dense, n_qubits=n_qubits, verbose=0)
    rho_rebuilt, rho_mf   = C.rho_expansion_approx(compute_3q_cumulants=True, compute_4q_cumulants=False)

    print("‖rho_exact - rho_rebuilt‖  = ",np.linalg.norm(rho_exact - rho_rebuilt))
    fidelity = np.trace(rho_exact @ rho_rebuilt)
    print("(rho_exact | rho_rebuilt)  = %12.8f + %12.8fi" %(np.real(fidelity), np.imag(fidelity)))
    energy_current = (np.trace(H_sparse @ rho_rebuilt)).real
    print(f"Approx. energy from Tr(H@rho_rebuilt) = {energy_current:.8f} Hartree")

    print(" Deviation from Hermiticity  = %12.8f" %np.linalg.norm(rho_rebuilt - rho_rebuilt.conj().T))
    print(" Trace(rho_exact)            = %12.8f" %np.abs(np.trace(rho_exact)))
    print(" Trace(rho_rebuilt)          = %12.8f" %np.abs(np.trace(rho_rebuilt)))

    # Check energy convergence
    if energy_prev is not None:
        energy_diff = abs(energy_current - energy_prev)
        print(f"Energy change: {energy_diff:.2e}")
        
        if energy_diff < energy_tol:
            print(f"Energy converged after {i+1} iterations (ΔE = {energy_diff:.2e})")
            break
    
    energy_prev = energy_current
    rho_guess = rho_rebuilt
    C.clear_cache()
exit()
# print("Eigenvalues of rho:")
# rho_eig_exact = np.linalg.eigvals(rho_exact)
# rho_eig_rebuilt = np.linalg.eigvals(rho_rebuilt)
# print(" %2s %12s %12s" %("", "Exact", "Rebuilt"))
# for i in range(len(rho_eig_exact)):
#     print(" %2i %12.8f %12.8f" %(i, np.real(rho_eig_exact[i]), np.real(rho_eig_rebuilt[i])))


#Three body
print("\n**************** - Three body - ***************")
rho_rebuilt, rho_mf   = C.rho_expansion_approx(compute_3q_cumulants=True, compute_4q_cumulants=False)

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
rho_rebuilt, rho_mf   = C.rho_expansion_approx(compute_3q_cumulants=True, compute_4q_cumulants=True)

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