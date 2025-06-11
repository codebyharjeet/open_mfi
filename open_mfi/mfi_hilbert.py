"""Provide the primary functions."""

import numpy as np
import itertools
import string
from functools import reduce
from typing import Dict, Tuple, List, Optional, Literal


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
        hamiltonian: np.ndarray,
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
        self.ham = hamiltonian.astype(complex)
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
    def _partial_trace(tensor_full: np.ndarray,trace_out: List[int],full_space: bool = False, format: Literal["tensor", "matrix"] = "matrix",verbose: int = 0) -> Tuple[np.ndarray, List[int]]:
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
        
        verboses = False
        if verboses:
            print(f'tensor_reduced = np.einsum("{in_sub}->{out_sub}", tensor_full)')
        tensor_reduced = np.einsum(f'{in_sub}->{out_sub}', tensor_full)

        if not full_space:
            if format == "tensor":
                return tensor_reduced, keep 
            elif format == "matrix":
                return tensor_reduced.reshape(2**len(keep), 2**len(keep)), keep
            else:
                raise ValueError("Format must be 'tensor' or 'matrix'")

        # I_t   = (np.eye(2**t) / 2**t).reshape((2,)*t + (2,)*t)
        I_t   = (np.eye(2**t)).reshape((2,)*t + (2,)*t)

        bra_labels = [letters[q] for q in range(N)]
        ket_labels = [Letters[q] for q in range(N)]

        lhs_I   = ''.join(bra_labels[q] for q in trace_out) + ''.join(ket_labels[q] for q in trace_out)
        lhs_rho = ''.join(bra_labels[q] for q in keep) + ''.join(ket_labels[q] for q in keep)

        rhs = ''.join(bra_labels + ket_labels)

        if verbose:
            print(f"tensor_full = np.einsum('{lhs_I},{lhs_rho}->{rhs}', I_t, tensor_reduced)")
        tensor_full = np.einsum(f'{lhs_I},{lhs_rho}->{rhs}', I_t, tensor_reduced) 

        if format == "tensor":
            return tensor_full, keep 
        elif format == "matrix":
            return tensor_full.reshape(2**N, 2**N), keep
        else:
            raise ValueError("Format must be 'tensor' or 'matrix'")

    @staticmethod
    def diagonalize_and_build_rho(H_ij: np.ndarray) -> np.ndarray:
        """Projector onto the ground state of a 4x4 Hermitian."""
        evals, evecs = np.linalg.eigh(H_ij)
        psi0 = evecs[:, 0]               # ground‐state (lowest eigenvalue) column
        print("eigenvalue = ", evals[0])
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
        lam: Dict[Tuple[int, int], np.ndarray] = {}

        indices = set(i for i in range(N))

        for i, j in itertools.combinations(range(N), 2):
            rho_ij, _ = self._partial_trace(tens, list(indices - {i, j}))
            rho_ij_reshaped, _ = self._reshape_to_tensor(rho_ij)
            rho_i, _ = self._partial_trace(rho_ij_reshaped, [1])
            rho_j, _ = self._partial_trace(rho_ij_reshaped, [0])
            lam_ij = rho_ij - np.kron(rho_i, rho_j)
            lam[(i, j)] = lam_ij
            if self.verbose:
                trace_val = np.trace(lam_ij)
                norm_val = np.linalg.norm(lam_ij)
                print(f"λ({i},{j}) trace: {trace_val:.6f}, norm: {norm_val:.6f}")

        return lam

    def three_qubit_cumulants(self) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Compute λ_{ijk} ≡ rho_{ijk} - rho_{ij} ⊗ rho_k - rho_{ik} ⊗ rho_j - rho_{jk} ⊗ rho_i + 2 * rho_i ⊗ rho_j ⊗ rho_k
        for every triplet i < j < k.

        Returns
        -------
        dict
            Keys (i, j, k) and 8x8 arrays λ_{ijk} (zero-trace by construction).
        """
        tens, N = self._reshape_to_tensor(self.rho_full)

        lam: Dict[Tuple[int, int, int], np.ndarray] = {}
        
        indices = set(i for i in range(N))
        
        for i, j, k in itertools.combinations(range(N), 3):
            rho_ijk, _ = self._partial_trace(tens, list(indices - {i, j, k}))
            rho_ijk_tensor, _ = self._reshape_to_tensor(rho_ijk)

            rho_jk, _ = self._partial_trace(rho_ijk_tensor, [0], format="tensor")  # Trace out i
            rho_ik, _ = self._partial_trace(rho_ijk_tensor, [1], format="tensor")  # Trace out j
            rho_ij, _ = self._partial_trace(rho_ijk_tensor, [2], format="tensor")  # Trace out k
            
            rho_i, _ = self._partial_trace(rho_ijk_tensor, [1, 2])  # Trace out j, k
            rho_j, _ = self._partial_trace(rho_ijk_tensor, [0, 2])  # Trace out i, k
            rho_k, _ = self._partial_trace(rho_ijk_tensor, [0, 1])  # Trace out i, j
            
            lam_ijk = (rho_ijk 
                    - np.einsum('ijIJ, kK->ijkIJK', rho_ij, rho_k, optimize=True).reshape(8, 8)
                    - np.einsum('ikIK, jJ->ijkIJK', rho_ik, rho_j, optimize=True).reshape(8, 8)
                    - np.einsum('jkJK, iI->ijkIJK', rho_jk, rho_i, optimize=True).reshape(8, 8)
                    + 2 * np.kron(np.kron(rho_i, rho_j), rho_k))
            
            lam[(i, j, k)] = lam_ijk
            
            if self.verbose:
                tr = np.trace(lam_ijk)
                nrm = np.linalg.norm(lam_ijk)
                print(f"λ({i},{j},{k}) trace: {tr:.4f}, norm: {nrm:.6f}")
        
        return lam

    def four_qubit_cumulants(self) -> Dict[Tuple[int, int, int, int], np.ndarray]:
        """
        Compute λ_{ijkl} for every quadruplet i < j < k < l.
        
        The fourth-order cumulant expansion is:
        λ_{ijkl} = rho_{ijkl}
                - rho_{ijk} ⊗ rho_l - rho_{ijl} ⊗ rho_k - rho_{ikl} ⊗ rho_j - rho_{jkl} ⊗ rho_i
                - rho_{ij} ⊗ rho_{kl} - rho_{ik} ⊗ rho_{jl} - rho_{il} ⊗ rho_{jk}
                + 2(rho_{ij} ⊗ rho_k ⊗ rho_l + rho_{ik} ⊗ rho_j ⊗ rho_l + rho_{il} ⊗ rho_j ⊗ rho_k
                + rho_{jk} ⊗ rho_i ⊗ rho_l + rho_{jl} ⊗ rho_i ⊗ rho_k + rho_{kl} ⊗ rho_i ⊗ rho_j)
                - 6 * rho_i ⊗ rho_j ⊗ rho_k ⊗ rho_l
        Returns
        -------
        dict
            Keys (i, j, k, l) and 16x16 arrays λ_{ijkl} (zero-trace by construction).
        """
        tens, N = self._reshape_to_tensor(self.rho_full)
        lam: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        
        indices = set(i for i in range(N))
        
        for i, j, k, l in itertools.combinations(range(N), 4):
            # four-qubit marginal ρ_{ijkl}
            rho_ijkl, _ = self._partial_trace(tens, list(indices - {i, j, k, l}))
            rho_ijkl_tensor, _ = self._reshape_to_tensor(rho_ijkl)
            
            # three-qubit marginals
            rho_jkl, _ = self._partial_trace(rho_ijkl_tensor, [0], format="tensor")  # Trace out i
            rho_ikl, _ = self._partial_trace(rho_ijkl_tensor, [1], format="tensor")  # Trace out j
            rho_ijl, _ = self._partial_trace(rho_ijkl_tensor, [2], format="tensor")  # Trace out k
            rho_ijk, _ = self._partial_trace(rho_ijkl_tensor, [3], format="tensor")  # Trace out l
            
            # two-qubit marginals
            rho_kl, _ = self._partial_trace(rho_ijkl_tensor, [0, 1], format="tensor")  # Trace out i, j
            rho_jl, _ = self._partial_trace(rho_ijkl_tensor, [0, 2], format="tensor")  # Trace out i, k
            rho_jk, _ = self._partial_trace(rho_ijkl_tensor, [0, 3], format="tensor")  # Trace out i, l
            rho_il, _ = self._partial_trace(rho_ijkl_tensor, [1, 2], format="tensor")  # Trace out j, k
            rho_ik, _ = self._partial_trace(rho_ijkl_tensor, [1, 3], format="tensor")  # Trace out j, l
            rho_ij, _ = self._partial_trace(rho_ijkl_tensor, [2, 3], format="tensor")  # Trace out k, l
            
            # single-qubit marginals
            rho_i, _ = self._partial_trace(rho_ijkl_tensor, [1, 2, 3])  # Trace out j, k, l
            rho_j, _ = self._partial_trace(rho_ijkl_tensor, [0, 2, 3])  # Trace out i, k, l
            rho_k, _ = self._partial_trace(rho_ijkl_tensor, [0, 1, 3])  # Trace out i, j, l
            rho_l, _ = self._partial_trace(rho_ijkl_tensor, [0, 1, 2])  # Trace out i, j, k
            
            # Compute fourth-order cumulant
            lam_ijkl = (rho_ijkl
                        # Subtract three-qubit ⊗ single-qubit terms
                        - np.einsum('ijkIJK, lL->ijklIJKL', rho_ijk, rho_l, optimize=True).reshape(16, 16)
                        - np.einsum('ijlIJL, kK->ijklIJKL', rho_ijl, rho_k, optimize=True).reshape(16, 16)
                        - np.einsum('iklIKL, jJ->ijklIJKL', rho_ikl, rho_j, optimize=True).reshape(16, 16)
                        - np.einsum('jklJKL, iI->ijklIJKL', rho_jkl, rho_i, optimize=True).reshape(16, 16)
                        # Subtract two-qubit ⊗ two-qubit terms
                        - np.einsum('ijIJ, klKL->ijklIJKL', rho_ij, rho_kl, optimize=True).reshape(16, 16)
                        - np.einsum('ikIK, jlJL->ijklIJKL', rho_ik, rho_jl, optimize=True).reshape(16, 16)
                        - np.einsum('ilIL, jkJK->ijklIJKL', rho_il, rho_jk, optimize=True).reshape(16, 16)
                        # Add two-qubit ⊗ single ⊗ single terms
                        + 2 * np.einsum('ijIJ, kK, lL->ijklIJKL', rho_ij, rho_k, rho_l, optimize=True).reshape(16, 16)
                        + 2 * np.einsum('ikIK, jJ, lL->ijklIJKL', rho_ik, rho_j, rho_l, optimize=True).reshape(16, 16)
                        + 2 * np.einsum('ilIL, jJ, kK->ijklIJKL', rho_il, rho_j, rho_k, optimize=True).reshape(16, 16)
                        + 2 * np.einsum('jkJK, iI, lL->ijklIJKL', rho_jk, rho_i, rho_l, optimize=True).reshape(16, 16)
                        + 2 * np.einsum('jlJL, iI, kK->ijklIJKL', rho_jl, rho_i, rho_k, optimize=True).reshape(16, 16)
                        + 2 * np.einsum('klKL, iI, jJ->ijklIJKL', rho_kl, rho_i, rho_j, optimize=True).reshape(16, 16)
                        # Subtract 6 * single ⊗ single ⊗ single ⊗ single term
                        - 6 * np.kron(np.kron(np.kron(rho_i, rho_j), rho_k), rho_l))
            
            lam[(i, j, k, l)] = lam_ijkl
            
            if self.verbose:
                tr = np.trace(lam_ijkl)
                nrm = np.linalg.norm(lam_ijkl)
                print(f"λ({i},{j},{k},{l}) trace: {tr:.4f}, norm: {nrm:.6f}")
        
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

    @staticmethod
    def _embed_cluster(lam_k: np.ndarray,indices: Tuple[int, ...],singles: List[np.ndarray],verbose: bool = False) -> np.ndarray:
        """
        Embed a k-body cumulant lam_k on qubits `indices`,
        tensor-product with each single-qubit rho on the rest.
        """
        N = len(singles)
        k = len(indices)

        lam_t = lam_k.reshape((2,)*k + (2,)*k)

        operands = [lam_t] + [singles[q] for q in range(N) if q not in indices]

        lc, uc = list(string.ascii_lowercase), list(string.ascii_uppercase)

        lam_sub = ''.join(lc[i] for i in indices) + ''.join(uc[i] for i in indices)
        subscripts = [lam_sub]

        subscripts += [f"{lc[q]}{uc[q]}" for q in range(N) if q not in indices]

        out_sub = ''.join(lc[:N] + uc[:N])

        einsum_str = ','.join(subscripts) + '->' + out_sub
        if verbose:
            print("einsum:", einsum_str)

        full_t = np.einsum(einsum_str, *operands, optimize=True)
        return full_t.reshape(2**N, 2**N)

    def cluster_expansion_rho(self, compute_3q_cumulants: bool = False, compute_4q_cumulants: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build rho_{mf} + Σ_{i<j} λ_{ij} ⊗ ⊗_{k≠i,j} rho_k 
                       + Σ_{i<j<k} λ_{ijk} ⊗ ⊗_{k≠i,j,k} rho_k 
                       + Σ_{i<j<k<l} λ_{ijkl} ⊗ ⊗_{k≠i,j,k,l} rho_k.

        Returns
        -------
        rho_rebuilt : ndarray
            Density matrix reconstructed from all one- and two-body clusters.
        rho_mf : ndarray
            Simple product (mean-field) state.
        """
        singles = self.one_qubit_marginals()
        lam_2q  = self.two_qubit_cumulants()
        rho_mf  = self.mean_field_state()

        rho_rebuilt  = rho_mf.copy()
        for (i, j), lam_ij in lam_2q.items():
            # rho_rebuilt += self._embed_pair(lam_ij, i, j, singles)
            rho_rebuilt += self._embed_cluster(lam_ij, (i,j), singles)

        if compute_3q_cumulants:
            lam_3q     = self.three_qubit_cumulants()
            for (i, j, k), lam_ijk in lam_3q.items():
                rho_rebuilt += self._embed_cluster(lam_ijk, (i,j,k), singles)

        if compute_4q_cumulants:
            lam_4q     = self.four_qubit_cumulants()
            for (i, j, k, l), lam_ijkl in lam_4q.items():
                rho_rebuilt += self._embed_cluster(lam_ijkl, (i,j,k,l), singles)

        return rho_rebuilt, rho_mf 



if __name__ == "__main__":
    print()
