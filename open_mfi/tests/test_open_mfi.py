"""
Unit and regression test for the open_mfi package.
"""
"""
Unit tests for the `ClusterExpansion` class.

Run with::
    pytest test_cluster_expansion.py       
"""

import sys
import numpy as np
import pytest
import open_mfi
from open_mfi import *
# from open_mfi.mfi_hilbert import mfi_hilbert
# from mfi_hilbert import ClusterExpansion  

def test_open_mfi_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "open_mfi" in sys.modules


def random_density(n_qubits: int, seed: int = 1234) -> np.ndarray:
    """Generate a random full-rank density matrix of dimension 2^N."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    A   = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = A @ A.conj().T
    return rho / np.trace(rho)

def random_hamiltonian(n_qubits: int, seed: int = 1234) -> np.ndarray:
    """
    Generate a random Hermitian Hamiltonian of dimension 2^N x 2^N.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (matrix dimension will be 2**n_qubits).
    seed : int, optional
        RNG seed for reproducibility.  Passing the same seed yields the same Hamiltonian.

    Returns
    -------
    H : np.ndarray
        Hermitian matrix suitable to serve as an N-qubit Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    A   = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    H   = (A + A.conj().T) / 2         
    return H

NQ = 3                           
RHO = random_density(NQ)
H = random_hamiltonian(NQ)

# ----------  reshape_to_tensor  -------------------------------------- #
def test_reshape_to_tensor_roundtrip():
    tens, n = ClusterExpansion._reshape_to_tensor(RHO)
    assert n == NQ
    back = tens.reshape(RHO.shape)
    assert np.allclose(back, RHO)


# ----------  class construction  ------------------------------------- #
def test_constructor_checks():

    ClusterExpansion(RHO, H, NQ)

    # non-Hermitian -> should raise
    bad = RHO.copy()
    bad[0, 1] += 0.3
    with pytest.raises(ValueError):
        ClusterExpansion(bad, H, NQ)

    # trace ≠ 1  -> should raise
    bad = RHO * 2.0
    with pytest.raises(ValueError):
        ClusterExpansion(bad, H, NQ)


# ----------  partial_trace  (single qubit) --------------------------- #
def test_partial_trace_one_qubit():
    C = ClusterExpansion(RHO, H, NQ)
    tens, _ = C._reshape_to_tensor(RHO)
    reduced, keep = C._partial_trace(tens, trace_out=[1], full_space=False)
    assert reduced.shape == (4, 4)      # 2^(N-1) = 4
    assert keep == [0, 2]
    assert np.isclose(np.trace(reduced), 1.0)


# ----------  one-qubit marginals ------------------------------------- #
def test_one_qubit_marginals():
    C = ClusterExpansion(RHO, H, NQ)
    singles = C.one_qubit_marginals()
    assert len(singles) == NQ
    for rho_k in singles:
        # (2×2), Hermitian, trace 1, positive
        assert rho_k.shape == (2, 2)
        assert np.allclose(rho_k, rho_k.conj().T)
        assert np.isclose(np.trace(rho_k), 1.0)
        # eigenvalues non-negative
        assert np.min(np.linalg.eigvalsh(rho_k)) >= -1e-10


# ----------  mean-field state ---------------------------------------- #
def test_mean_field_state():
    C = ClusterExpansion(RHO, H, NQ)
    singles = C.one_qubit_marginals()
    rho_mf = C.mean_field_state()
    assert rho_mf.shape == RHO.shape
    assert np.isclose(np.trace(rho_mf), 1.0)


# ----------  two-qubit cumulants ------------------------------------- #
def test_two_qubit_cumulants_trace_zero():
    C = ClusterExpansion(RHO, H, NQ)
    singles = C.one_qubit_marginals()
    lam = C.two_qubit_cumulants()
    # there are NQ*(NQ-1)/2 pairs
    assert len(lam) == NQ * (NQ - 1) // 2
    for (i, j), L in lam.items():
        assert L.shape == (4, 4)
        assert np.isclose(np.trace(L), 0.0)


# ----------  embed_pair  --------------------------------------------- #
def test_embed_pair_shape_trace():
    C = ClusterExpansion(RHO, H, NQ)
    singles = C.one_qubit_marginals()
    lam = C.two_qubit_cumulants()
    (i, j), lam_ij = next(iter(lam.items()))
    full = C._embed_pair(lam_ij, i, j, singles)
    assert full.shape == RHO.shape
    # trace should still be zero (λ trace 0, ∏ ρ_k trace 1)
    assert np.isclose(np.trace(full), 0.0)


# ----------  full cluster expansion ---------------------------------- #
def test_cluster_expansion_rho_properties():
    C = ClusterExpansion(RHO, H, NQ)
    rho_rebuilt, rho_mf = C.cluster_expansion_rho()
    # shapes
    assert rho_rebuilt.shape == RHO.shape
    # Hermitian & unit trace
    assert np.allclose(rho_rebuilt, rho_rebuilt.conj().T)
    assert np.isclose(np.trace(rho_rebuilt), 1.0)
    # rebuilt should be closer to exact than MF (norm check)
    err_mf  = np.linalg.norm(RHO - rho_mf)
    err_reb = np.linalg.norm(RHO - rho_rebuilt)
    assert err_reb <= err_mf + 1e-10 
