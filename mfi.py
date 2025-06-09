import numpy as np
import itertools
import string
from functools import reduce
from typing import Dict, Tuple, List, Optional


class ClusterExpansion:
    """
    Cluster-expansion class for an N-qubit (or spin-orbital) density matrix.

    Parameters
    ----------
    rho_full : (2**N, 2**N) ndarray[complex]
        Exact many-body density matrix.  Must be Hermitian, unit-trace
        and positive semi-definite.
    n_qubits : int
        Number of physical qubits (spatial orbitals * spin).
    n_a : int, optional
        Number of alpha electrons.  Default = None.
    n_b : int, optional
        Number of beta  electrons.  Default = None.
    n_orb : int, optional
        Number of spatial orbitals.  Default = None.

    Notes
    -----
    *  The workflow is
        >>> C = ClusterDensity(rho_exact, n_qubits)
        >>> rho_mf           = C.mean_field_state()
        >>> rho_rebuilt      = C.cluster_expansion_rho()   # MF + all 2-body cumulants
    """    

    def __init__(
        self,
        rho_full: np.ndarray,
        n_qubits: int,
        n_a: int | None = None,
        n_b: int | None = None,
        n_orb: int | None = None,
        verbose: int = 0) -> None:
        # validation
        if rho_full.shape != (2**n_qubits, 2**n_qubits):
            raise ValueError("rho_full has wrong shape for n_qubits")
        if not np.allclose(rho_full, rho_full.conj().T):
            raise ValueError("rho_full must be Hermitian (rho = rho†)")
        if not np.isclose(np.trace(rho_full), 1.0):
            raise ValueError("Trace(rho) must be 1")
        if np.min(np.linalg.eigvalsh(rho_full)) < -1e-10:
            raise ValueError("rho_full must be positive semi-definite")

        self.rho_full = rho_full.astype(complex)
        self.N        = n_qubits
        self.n_a      = n_a
        self.n_b      = n_b
        self.n_orb    = n_orb
        self.verbose  = verbose

    @staticmethod
    def _reshape_to_tensor(mat: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Reshape a 2^Nx2^N matrix into a rank-2N tensor (2,)*N + (2,)*N.

        Returns
        -------
        tensor : ndarray
            Shape (2,)*N  + (2,)*N.
        N : int
            Number of qubits.
        """
        dim = mat.shape[0]
        N = int(np.log2(dim))
        if 2**N != dim:
            raise ValueError("Matrix dimension is not a power of two.")
        new_shape = (2,) * N + (2,) * N
        return mat.reshape(new_shape), N

    @staticmethod
    def _partial_trace(tensor_full: np.ndarray,trace_out: List[int],full_space: bool = False,verbose: int = 0) -> Tuple[np.ndarray, List[int]]:
        """
        Partial trace over qubits listed in `trace_out`.

        Parameters
        ----------
        tensor_full : rank-2N ndarray
        trace_out : list[int]
            Indices to trace away.
        full_space : bool, default False
            If True return the *embedded* (I / 2^t) ⊗ rho_keep matrix
            in the original 2^Nx2^N space; otherwise return the reduced
            density of shape (2^(N-t), 2^(N-t)).
        verbose : int, optional

        Returns
        -------
        tensor_reduced : ndarray
            Either the reduced density (full_space=False) or the embedded
            full-space matrix (full_space=True).
        keep : list[int]
            Complement of `trace_out`.
        """

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

    @staticmethod
    def diagonalize_and_build_rho(H_ij: np.ndarray) -> np.ndarray:
        """Projector onto the ground state of a 4x4 Hermitian."""
        evals, evecs = np.linalg.eigh(H_ij)
        psi0 = evecs[:, 0]               # ground‐state (lowest eigenvalue) column
        return np.outer(psi0, psi0.conj())

    def one_qubit_marginals(self) -> List[np.ndarray]:
        """Return list [rho₀, rho₁, …, rho_{N-1}] of (2x2) density matrices."""
        rho_tensor, N = self._reshape_to_tensor(self.rho_full)
        singles = []
        for p in range(N):
            trace_out = [q for q in range(N) if q != p]
            rho_p, keep  = self._partial_trace(rho_tensor, trace_out, full_space=False)
            singles.append(rho_p)         
        return singles 

    def mean_field_state(self) -> np.ndarray:
        """Return rho_mf = ⊗_{k=0}^{N-1} rho_k ."""
        singles = self.one_qubit_marginals()
        return reduce(np.kron, singles)

    def two_qubit_cumulants(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute λ_{ij} ≡ rho_{ij} - rho_i ⊗ rho_j for every pair i < j.

        Returns
        -------
        dict
            Keys (i, j) and 4x4 arrays λ_{ij} (zero-trace by construction).
        """  

        tens, N = self._reshape_to_tensor(self.rho_full)
        singles = self.one_qubit_marginals()
        lam: Dict[Tuple[int, int], np.ndarray] = {}

        indices = set(i for i in range(N))

        for i, j in itertools.combinations(range(N), 2):
            rho_ij, _ = self._partial_trace(tens, list(indices - {i, j}))
            lam_ij = rho_ij - np.kron(singles[i], singles[j])
            lam[(i, j)] = lam_ij
            if self.verbose:
                tmp = np.linalg.norm(lam_ij)
                print("Norm of  λ(%i,%i) = %12.8f" %(i,j,tmp))    

        return lam

    @staticmethod
    def _embed_pair(lam_ij: np.ndarray, i: int, j: int, singles: List[np.ndarray],verbose: int = 0) -> np.ndarray:
        """
        Embed λ_{ij} on qubits (i,j) and single-site rho_k on all others.

        Returns
        -------
        ndarray
            Full-space (2^N x 2^N) operator with correct qubit ordering.
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

    def cluster_expansion_rho(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build rho_{mf} + Σ_{i<j} λ_{ij} ⊗ ⊗_{k≠i,j} rho_k .

        Returns
        -------
        rho_rebuilt : ndarray
            Density matrix reconstructed from all one- and two-body clusters.
        rho_mf : ndarray
            Simple product (mean-field) state.
        """
        singles = self.one_qubit_marginals()
        lam     = self.two_qubit_cumulants()
        rho_mf  = self.mean_field_state()

        rho_rebuilt  = rho_mf.copy()
        for (i, j), lam_ij in lam.items():
            rho_rebuilt += self._embed_pair(lam_ij, i, j, singles)

        return rho_rebuilt, rho_mf 

