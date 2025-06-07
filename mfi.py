import numpy as np
import openfermion as of 
import string
import itertools
from functools import reduce

def reshape_to_tensor(H_dense):
    """
    Reshape a (2^N x 2^N) array into a rank‐2N tensor of shape (2,)*N + (2,)*N.
    """
    dim = H_dense.shape[0]
    N = int(np.log2(dim))
    assert 2**N == dim 
    new_shape = (2,) * N + (2,) * N
    return H_dense.reshape(new_shape), N

def partial_trace(tensor_full, trace_out, full_space=False, verbose=False):

    N = tensor_full.ndim // 2
    t = len(trace_out)
    keep = [q for q in range(N) if q not in trace_out]

    letters, Letters = list(string.ascii_lowercase), list(string.ascii_uppercase)
    bra, ket = ([''] * N, [''] * N)

    for q in range(N):
        if q in trace_out:
            bra[q] = ket[q] = letters[q]
        else:
            bra[q] = letters[q]
            ket[q] = Letters[q]

    in_sub = ''.join(bra) + ''.join(ket)
    out_sub = ''.join(bra[k] for k in keep) + ''.join(ket[k] for k in keep)

    if verbose:
        print(f'tensor_reduced = np.einsum("{in_sub}->{out_sub}", tensor_full)')
    tensor_reduced = np.einsum(f'{in_sub}->{out_sub}', tensor_full)

    if not full_space:
        return tensor_reduced.reshape(2**len(keep), 2**len(keep)), keep

    I_t   = (np.eye(2**t) / 2**t).reshape((2,)*t + (2,)*t)

    bra_labels = [letters[q] for q in range(N)]
    ket_labels = [Letters[q] for q in range(N)]

    lhs_I   = ''.join(bra_labels[q] for q in trace_out) + ''.join(ket_labels[q] for q in trace_out)
    lhs_rho = ''.join(bra_labels[q] for q in keep) + ''.join(ket_labels[q] for q in keep)

    rhs = ''.join(bra_labels + ket_labels)

    if verbose:
        print(f"tensor_full = np.einsum('{lhs_I},{lhs_rho}->{rhs}', I_t, tensor_reduced)")
    tensor_full = np.einsum(f'{lhs_I},{lhs_rho}->{rhs}', I_t, tensor_reduced) 

    return tensor_full.reshape(2**N, 2**N), keep

def diagonalize_and_build_rho(H_ij):
    """
    Given a 4x4 Hermitian H_ij (two-qubit Hamiltonian),
    diagonalize it, pick the ground-state eigenvector psi0,
    and return rho0 = |psi0><psi0| (a 4x4 density matrix).
    """
    evals, evecs = np.linalg.eigh(H_ij)
    psi0 = evecs[:, 0]               # ground‐state (lowest eigenvalue) column
    return np.outer(psi0, psi0.conj())

def single_site_rhos(rho_full):
    """
    Input:
      - rho_dense:  a (2^N x 2^N) NumPy array representing the full N-qubit density matrix.
    Returns:
      - rho_dense:  a (2^N x 2^N) NumPy array representing the full N-qubit uncorrelated density matrix.        
    """
    rho_tensor, N = reshape_to_tensor(rho_full)      # rank-2N tensor
    singles = []
    for p in range(N):
        trace_out = [q for q in range(N) if q != p]
        rho_p, keep  = partial_trace(rho_tensor, trace_out, full_space=False)
        singles.append(rho_p)         
    return singles, N

def mean_field_state(singles):
    rho_mf = reduce(np.kron, singles)
    assert np.allclose(np.trace(rho_mf), 1.0)
    return rho_mf

def two_qubit_cumulants(rho_full, singles, verbose=1):
    """
    Returns a dict  lambdas[(i,j)]  = 4x4 matrix  rho_{ij} - rho_i⊗rho_j .
    """
    rho_tensor, N = reshape_to_tensor(rho_full)
    lam = {}
    indices = set(i for i in range(N))
    for (i, j) in itertools.combinations(range(N), 2):
        trace_out = indices.difference(set([i,j])) 
        rho_ij, keep = partial_trace(rho_tensor, trace_out, full_space=False)
        lam_ij = rho_ij - np.kron(singles[i], singles[j])
        lam[(i, j)] = lam_ij
        if verbose:
            tmp = np.linalg.norm(lam_ij)
            print("Norm of  λ(%i,%i) = %12.8f" %(i,j,tmp))

    return lam

def embed_pair_with_rest(lam_ij, i, j, singles, verbose=False):
    """
    Return the full (2^N x 2^N) operator that equals
        λ_{ij}   on qubits i and j
        rho_k      on every other qubit k.
    """
    if i == j:
        raise ValueError("i and j must differ")

    N = len(singles)

    lam_t       = lam_ij.reshape(2, 2, 2, 2)
    
    operands    = [lam_t] + [singles[q] for q in range(N) if q not in (i, j)]

    lc, uc = list(string.ascii_lowercase), list(string.ascii_uppercase)

    subs = [f"{lc[i]}{lc[j]}{uc[i]}{uc[j]}"]         
    subs += [f"{lc[q]}{uc[q]}" for q in range(N) if q not in (i, j)]

    rhs  = ''.join(lc[:N] + uc[:N])  

    full_t = np.einsum(','.join(subs) + '->' + rhs, *operands)
    if verbose:
        print(f"full_t = np.einsum({','.join(subs) + '->' + rhs}, *operands)")

    return full_t.reshape(2**N, 2**N)

def cluster_expansion_rho(rho_full, H_dense):
    singles, N   = single_site_rhos(rho_full)
    lam_dict     = two_qubit_cumulants(rho_full, singles)
    rho_mf       = mean_field_state(singles)

    rho_rebuilt  = rho_mf.copy()
    for (i, j), lam_ij in lam_dict.items():
        rho_rebuilt += embed_pair_with_rest(lam_ij, i, j, singles)
    return rho_rebuilt, rho_mf 

