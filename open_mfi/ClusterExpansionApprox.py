"""Provide the primary functions."""

import numpy as np
import itertools
import string
from functools import reduce
from typing import Dict, Tuple, List, Optional, Literal
from opt_einsum import contract
import openfermion as of 

class ClusterExpansionApprox:
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
        n_a: Optional[int] = None,
        n_b: Optional[int] = None,
        n_orb: Optional[int] = None,
        verbose: int = 0) -> None:
        # validation
        if rho_full.shape != (2**n_qubits, 2**n_qubits):
            raise ValueError("rho_full has wrong shape for n_qubits")
        if not np.allclose(rho_full, rho_full.conj().T):
            raise ValueError("rho_full must be Hermitian (rho = rho†)")
        if not np.isclose(np.trace(rho_full), 1.0):
            raise ValueError("Trace(rho) must be 1")
        # if np.min(np.linalg.eigvalsh(rho_full)) < -1e-10:
        #     raise ValueError("rho_full must be positive semi-definite")

        self.rho_full = rho_full.astype(complex)
        self.ham = hamiltonian.astype(complex)
        self.N        = n_qubits
        self.n_a      = n_a
        self.n_b      = n_b
        self.n_orb    = n_orb
        self.verbose  = verbose

        # Cache for expensive computations
        self._rho_tensor = None
        self._ham_tensor = None
        self._marginals = {}  
        self._cumulants = {}  
        self._mean_field_state = None

    @property
    def rho_tensor(self) -> np.ndarray:
        """Cached tensor representation of the full density matrix."""
        if self._rho_tensor is None:
            self._rho_tensor = self.unfold(self.rho_full)
        return self._rho_tensor

    @property
    def ham_tensor(self) -> np.ndarray:
        """Cached tensor representation of the full density matrix."""
        if self._ham_tensor is None:
            self._ham_tensor = self.unfold(self.ham)
        return self._ham_tensor

    @staticmethod
    def unfold(mat: np.ndarray) -> np.ndarray:
        """
        Reshape a 2^N x 2^N matrix into a rank-2N tensor with shape (2,)*N + (2,)*N.
        
        Parameters
        ----------
        mat : ndarray
            Matrix of shape (2^N, 2^N)
            
        Returns
        -------
        tensor : ndarray
            Tensor of shape (2,)*N + (2,)*N
        """
        dim = mat.shape[0]
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("Matrix must be square")
        N = int(np.log2(dim))
        if 2**N != dim:
            raise ValueError("Matrix dimension is not a power of two.")
        new_shape = (2,) * N + (2,) * N
        return mat.reshape(new_shape)
    
    @staticmethod
    def fold(tensor: np.ndarray) -> np.ndarray:
        """
        Reshape a rank-2N tensor back into a 2^N x 2^N matrix.
        
        Parameters
        ----------
        tensor : ndarray
            Tensor of shape (2,)*N + (2,)*N
            
        Returns
        -------
        mat : ndarray
            Matrix of shape (2^N, 2^N)
        """
        if tensor.ndim % 2 != 0:
            raise ValueError("Tensor must have even rank (2N dimensions)")
        N = tensor.ndim // 2
        dim = 2**N
        return tensor.reshape(dim, dim)

    @staticmethod
    def _partial_trace(tensor_full: np.ndarray, trace_out: List[int], full_space: bool = False, format: Literal["tensor", "matrix"] = "matrix", verbose: int = 0) -> Tuple[np.ndarray, List[int]]:
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
            print(f'tensor_reduced = contract("{in_sub}->{out_sub}", tensor_full)')
        tensor_reduced = contract(f'{in_sub}->{out_sub}', tensor_full)

        if not full_space:
            if format == "tensor":
                return tensor_reduced, keep 
            elif format == "matrix":
                return tensor_reduced.reshape(2**len(keep), 2**len(keep)), keep
            else:
                raise ValueError("Format must be 'tensor' or 'matrix'")

        I_t   = (np.eye(2**t) / 2**t).reshape((2,)*t + (2,)*t)

        bra_labels = [letters[q] for q in range(N)]
        ket_labels = [Letters[q] for q in range(N)]

        lhs_I   = ''.join(bra_labels[q] for q in trace_out) + ''.join(ket_labels[q] for q in trace_out)
        lhs_rho = ''.join(bra_labels[q] for q in keep) + ''.join(ket_labels[q] for q in keep)

        rhs = ''.join(bra_labels + ket_labels)

        if verbose:
            print(f"tensor_full = contract('{lhs_I},{lhs_rho}->{rhs}', I_t, tensor_reduced)")
        tensor_full = contract(f'{lhs_I},{lhs_rho}->{rhs}', I_t, tensor_reduced) 

        if format == "tensor":
            return tensor_full, keep 
        elif format == "matrix":
            return tensor_full.reshape(2**N, 2**N), keep
        else:
            raise ValueError("Format must be 'tensor' or 'matrix'")

    @staticmethod
    def diagonalize_and_build_rho(ham_marginals: np.ndarray) -> np.ndarray:
        """Projector onto the ground state of a 4x4 Hermitian."""
        evals, evecs = np.linalg.eigh(ham_marginals)
        psi0 = evecs[:, 0]               # ground‐state (lowest eigenvalue) column
        return np.outer(psi0, psi0.conj())

    def _contract_with_ham(self, rho_p_bar: np.ndarray, keep: List[int], trace_out: List[int]) -> np.ndarray:
        """
        Contract rho_p_bar with Hamiltonian tensor, keeping specified qubits.
        """
        
        letters, Letters = list(string.ascii_lowercase), list(string.ascii_uppercase)
        
        bra_labels = [letters[q] for q in range(self.N)]
        ket_labels = [Letters[q] for q in range(self.N)]        

        sub1 = ''.join(bra_labels[q] for q in keep) + ''.join(ket_labels[q] for q in keep)
        
        sub2 = ''.join(bra_labels + ket_labels)
        
        sub3 = ''.join(bra_labels[q] for q in trace_out) + ''.join(ket_labels[q] for q in trace_out)
        
        verbose = False   
        if verbose:
            print(f"ham_p{trace_out} = contract('{sub1}, {sub2} -> {sub3}', rho_p_bar{keep}, self.ham)")
        
        ham_p = contract(f"{sub1}, {sub2} -> {sub3}", rho_p_bar, self.ham_tensor)
        
        rho_p = self.diagonalize_and_build_rho(self.fold(ham_p))

        return rho_p

    def _get_marginal(self, qubits: Tuple[int, ...]) -> np.ndarray:
        """Get marginal density matrix for specified qubits with caching."""
        qubits = tuple(sorted(qubits))
        
        if qubits not in self._marginals:
            trace_out = list(qubits)
            rho_marginal_bar, keep = self._partial_trace(self.rho_tensor, trace_out, full_space=False, format="tensor")
            rho_marginal = self._contract_with_ham(rho_marginal_bar, keep, trace_out)
            self._marginals[qubits] = rho_marginal 

        return self._marginals[qubits]

    def _get_cumulant(self, qubits: Tuple[int, ...]) -> np.ndarray:
        """Get cumulant for specified qubits with caching."""
        qubits = tuple(sorted(qubits))
        
        if qubits not in self._cumulants:
            if len(qubits) == 2:
                # Two-qubit cumulant: λ_{ij} = ρ_{ij} - ρ_i ⊗ ρ_j
                i, j = qubits
                rho_ij = self._get_marginal((i, j))
                rho_i = self._get_marginal((i,))
                rho_j = self._get_marginal((j,))
                lam_ij = rho_ij - np.kron(rho_i, rho_j)
                self._cumulants[qubits] = lam_ij
                
                if self.verbose:
                    trace_val = np.trace(lam_ij)
                    norm_val = np.linalg.norm(lam_ij)
                    print(f"λ{qubits} trace: {trace_val:.6f}, norm: {norm_val:.6f}")
                    
            elif len(qubits) == 3:
                # Three-qubit cumulant: λ_{ijk} = ρ_{ijk} - ρ_i⊗ρ_j⊗ρ_k - λ_{ij}⊗ρ_k - λ_{ik}⊗ρ_j - λ_{jk}⊗ρ_i
                i, j, k = qubits
                rho_ijk = self._get_marginal((i, j, k))
                rho_i = self._get_marginal((i,))
                rho_j = self._get_marginal((j,))
                rho_k = self._get_marginal((k,))
                
                lam_ij = self.unfold(self._get_cumulant((i, j)))
                lam_ik = self.unfold(self._get_cumulant((i, k)))
                lam_jk = self.unfold(self._get_cumulant((j, k)))
                
                lam_ijk = rho_ijk.copy()
                # Mean-field term
                lam_ijk -= contract('iI, jJ, kK->ijkIJK', rho_i, rho_j, rho_k).reshape(8, 8) 
                # λ_{ij} ⊗ ρ_k
                lam_ijk -= contract('ijIJ, kK->ijkIJK', lam_ij, rho_k).reshape(8, 8)
                # λ_{ik} ⊗ ρ_j  
                lam_ijk -= contract('ikIK, jJ->ijkIJK', lam_ik, rho_j).reshape(8, 8)
                # λ_{jk} ⊗ ρ_i
                lam_ijk -= contract('jkJK, iI->ijkIJK', lam_jk, rho_i).reshape(8, 8)

                self._cumulants[qubits] = lam_ijk
                
                if self.verbose:
                    tr = np.trace(lam_ijk)
                    nrm = np.linalg.norm(lam_ijk)
                    print(f"λ{qubits} trace: {tr:.4f}, norm: {nrm:.6f}")
                    
            elif len(qubits) == 4:
                # Four-qubit cumulant
                i, j, k, l = qubits
                rho_ijkl = self._get_marginal((i, j, k, l))
                rho_i = self._get_marginal((i,))
                rho_j = self._get_marginal((j,))
                rho_k = self._get_marginal((k,))
                rho_l = self._get_marginal((l,))
                
                # Two-body cumulants
                lam_ij = self.unfold(self._get_cumulant((i, j)))
                lam_ik = self.unfold(self._get_cumulant((i, k)))
                lam_il = self.unfold(self._get_cumulant((i, l)))
                lam_jk = self.unfold(self._get_cumulant((j, k)))
                lam_jl = self.unfold(self._get_cumulant((j, l)))
                lam_kl = self.unfold(self._get_cumulant((k, l)))
                
                # Three-body cumulants
                lam_ijk = self.unfold(self._get_cumulant((i, j, k)))
                lam_ijl = self.unfold(self._get_cumulant((i, j, l)))
                lam_ikl = self.unfold(self._get_cumulant((i, k, l)))
                lam_jkl = self.unfold(self._get_cumulant((j, k, l)))
                
                lam_ijkl = rho_ijkl.copy()
                # Mean-field term
                lam_ijkl -= contract('iI, jJ, kK, lL->ijklIJKL', rho_i, rho_j, rho_k, rho_l).reshape(16, 16)
                # Three-body cumulant terms
                lam_ijkl -= contract('ijkIJK, lL->ijklIJKL', lam_ijk, rho_l).reshape(16, 16)
                lam_ijkl -= contract('ijlIJL, kK->ijklIJKL', lam_ijl, rho_k).reshape(16, 16)
                lam_ijkl -= contract('iklIKL, jJ->ijklIJKL', lam_ikl, rho_j).reshape(16, 16)
                lam_ijkl -= contract('jklJKL, iI->ijklIJKL', lam_jkl, rho_i).reshape(16, 16)                
                # Two-body cumulant cross terms
                lam_ijkl -= contract('ijIJ, kK, lL->ijklIJKL', lam_ij, rho_k, rho_l).reshape(16, 16) 
                lam_ijkl -= contract('ikIK, jJ, lL->ijklIJKL', lam_ik, rho_j, rho_l).reshape(16, 16)
                lam_ijkl -= contract('ilIL, jJ, kK->ijklIJKL', lam_il, rho_j, rho_k).reshape(16, 16)
                lam_ijkl -= contract('jkJK, iI, lL->ijklIJKL', lam_jk, rho_i, rho_l).reshape(16, 16)
                lam_ijkl -= contract('jlJL, iI, kK->ijklIJKL', lam_jl, rho_i, rho_k).reshape(16, 16)
                lam_ijkl -= contract('klKL, iI, jJ->ijklIJKL', lam_kl, rho_i, rho_j).reshape(16, 16)
                
                self._cumulants[qubits] = lam_ijkl
                
                if self.verbose:
                    tr = np.trace(lam_ijkl)
                    nrm = np.linalg.norm(lam_ijkl)
                    print(f"λ{qubits} trace: {tr:.4f}, norm: {nrm:.6f}")
            else:
                raise ValueError("Only 2-4 qubit cumulants supported")
                
        return self._cumulants[qubits]

    def one_qubit_marginals(self) -> List[np.ndarray]:
        """Return list [rho₀, rho₁, …, rho_{N-1}] of (2x2) density matrices."""
        return [self._get_marginal((p,)) for p in range(self.N)]

    def mean_field_state(self) -> np.ndarray:
        """Return rho_mf = ⊗_{k=0}^{N-1} rho_k ."""
        if self._mean_field_state is None:
            singles = self.one_qubit_marginals()
            self._mean_field_state = reduce(np.kron, singles)
        return self._mean_field_state

    def two_qubit_cumulants(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute λ_{ij} ≡ rho_{ij} - rho_i ⊗ rho_j for every pair i < j.

        Returns
        -------
        dict
            Keys (i, j) and 4x4 arrays λ_{ij} (zero-trace by construction).
        """  
        result = {}
        for i, j in itertools.combinations(range(self.N), 2):
            result[(i, j)] = self._get_cumulant((i, j))
        return result

    def three_qubit_cumulants(self) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Compute λ_{ijk} ≡ rho_{ijk} - rho_i ⊗ rho_j ⊗ rho_k - λ_{ij} ⊗ rho_k - λ_{ik} ⊗ rho_j - λ_{jk} ⊗ rho_i
        for every triplet i < j < k.

        Returns
        -------
        dict
            Keys (i, j, k) and 8x8 arrays λ_{ijk} (zero-trace by construction).
        """
        result = {}
        for i, j, k in itertools.combinations(range(self.N), 3):
            result[(i, j, k)] = self._get_cumulant((i, j, k))
        return result

    def four_qubit_cumulants(self) -> Dict[Tuple[int, int, int, int], np.ndarray]:
        """
        Compute λ_{ijkl} for every quadruplet i < j < k < l.
        
        The fourth-order cumulant expansion is:
        λ_{ijkl} = rho_{ijkl}
                - rho_i ⊗ rho_j ⊗ rho_k ⊗ rho_l (mean-field)
                - λ_{ijk} ⊗ rho_l - λ_{ijl} ⊗ rho_k - λ_{ikl} ⊗ rho_j - λ_{jkl} ⊗ rho_i (3-body cumulant terms)
                - λ_{ij} ⊗ λ_{kl} - λ_{ik} ⊗ λ_{jl} - λ_{il} ⊗ λ_{jk} (2-body cumulant terms)
                
        Returns
        -------
        dict
            Keys (i, j, k, l) and 16x16 arrays λ_{ijkl} (zero-trace by construction).
        """
        result = {}
        for i, j, k, l in itertools.combinations(range(self.N), 4):
            result[(i, j, k, l)] = self._get_cumulant((i, j, k, l))
        return result

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

        full_t = contract(einsum_str, *operands)
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
            rho_rebuilt += self._embed_cluster(lam_ij, (i,j), singles)

        if compute_3q_cumulants:
            lam_3q = self.three_qubit_cumulants()
            for (i, j, k), lam_ijk in lam_3q.items():
                rho_rebuilt += self._embed_cluster(lam_ijk, (i,j,k), singles)

        if compute_4q_cumulants:
            lam_4q = self.four_qubit_cumulants()
            for (i, j, k, l), lam_ijkl in lam_4q.items():
                rho_rebuilt += self._embed_cluster(lam_ijkl, (i,j,k,l), singles)

        return rho_rebuilt, rho_mf 

    def rho_expansion_approx(self, compute_3q_cumulants: bool = False, compute_4q_cumulants: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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
        rho_mf  = self.mean_field_state()
        singles = self.one_qubit_marginals()
        lam_2q  = self.two_qubit_cumulants()
        
        rho_rebuilt  = rho_mf.copy()
        for (i, j), lam_ij in lam_2q.items():
            rho_rebuilt += self._embed_cluster(lam_ij, (i,j), singles)

        if compute_3q_cumulants:
            lam_3q = self.three_qubit_cumulants()
            for (i, j, k), lam_ijk in lam_3q.items():
                rho_rebuilt += self._embed_cluster(lam_ijk, (i,j,k), singles)

        if compute_4q_cumulants:
            lam_4q = self.four_qubit_cumulants()
            for (i, j, k, l), lam_ijkl in lam_4q.items():
                rho_rebuilt += self._embed_cluster(lam_ijkl, (i,j,k,l), singles)

        return rho_rebuilt, rho_mf 

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of cached objects in megabytes."""
        import sys
        
        # Calculate sizes in bytes first
        rho_size = sys.getsizeof(self._rho_tensor) if self._rho_tensor is not None else 0
        marginals_size = sum(sys.getsizeof(v) for v in self._marginals.values())
        cumulants_size = sum(sys.getsizeof(v) for v in self._cumulants.values())
        mean_field_size = sys.getsizeof(self._mean_field_state) if self._mean_field_state is not None else 0
        
        # Convert to megabytes (1 MB = 1024 * 1024 bytes)
        MB = 1024 * 1024
        
        memory_info = {
            'rho_tensor': rho_size / MB,
            'marginals': marginals_size / MB,
            'cumulants': cumulants_size / MB,
            'mean_field': mean_field_size / MB,
            'total_cached_objects': len(self._marginals) + len(self._cumulants)  # Keep as count
        }
        return memory_info

    def clear_cache(self, keep_marginals: bool = False):
        """Clear cached computations to free memory."""
        if not keep_marginals:
            self._marginals.clear()
        self._cumulants.clear()
        self._mean_field_state = None
        self._rho_tensor = None 

if __name__ == "__main__":
    print()
