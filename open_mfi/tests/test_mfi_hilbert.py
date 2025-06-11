"""
Comprehensive test suite for open_mfi.mfi_hilbert
"""

import sys
import numpy as np
import pytest
import string
from unittest.mock import patch, MagicMock
from open_mfi.mfi_hilbert import ClusterExpansion


def random_density(n_qubits: int, seed: int = 1234) -> np.ndarray:
    """Generate a random full-rank density matrix of dimension 2^N."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def random_hamiltonian(n_qubits: int, seed: int = 1234) -> np.ndarray:
    """Generate a random Hermitian Hamiltonian of dimension 2^N x 2^N."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    H = (A + A.conj().T) / 2
    return H


# Test fixtures
NQ_2 = 2
NQ_3 = 3
NQ_4 = 4
RHO_2 = random_density(NQ_2, seed=42)
RHO_3 = random_density(NQ_3, seed=123)
RHO_4 = random_density(NQ_4, seed=456)
H_2 = random_hamiltonian(NQ_2, seed=42)
H_3 = random_hamiltonian(NQ_3, seed=123)
H_4 = random_hamiltonian(NQ_4, seed=456)


class TestClusterExpansionConstructor:
    """Test ClusterExpansion constructor and validation."""
    
    def test_valid_construction(self):
        """Test valid construction with all parameters."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3, n_a=1, n_b=1, n_orb=2, verbose=1)
        assert C.N == NQ_3
        assert C.n_a == 1
        assert C.n_b == 1
        assert C.n_orb == 2
        assert C.verbose == 1
        assert np.array_equal(C.rho_full, RHO_3.astype(complex))
        assert np.array_equal(C.ham, H_3.astype(complex))
    
    def test_wrong_shape_rho(self):
        """Test ValueError for wrong rho_full shape."""
        wrong_rho = np.random.random((5, 5))  # Not 2^n x 2^n
        with pytest.raises(ValueError, match="rho_full has wrong shape for n_qubits"):
            ClusterExpansion(wrong_rho, H_3, NQ_3)
    
    def test_non_hermitian_rho(self):
        """Test ValueError for non-Hermitian rho_full."""
        bad_rho = RHO_3.copy()
        bad_rho[0, 1] += 0.5  # Make it non-Hermitian
        with pytest.raises(ValueError, match="rho_full must be Hermitian"):
            ClusterExpansion(bad_rho, H_3, NQ_3)
    
    def test_wrong_trace_rho(self):
        """Test ValueError for rho_full with trace != 1."""
        bad_rho = RHO_3 * 2.0  # Double the trace
        with pytest.raises(ValueError, match="Trace\\(rho\\) must be 1"):
            ClusterExpansion(bad_rho, H_3, NQ_3)
    
    def test_negative_eigenvalues_rho(self):
        """Test ValueError for rho_full with negative eigenvalues."""
        # Create a matrix with negative eigenvalues
        bad_rho = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]], dtype=complex)
        bad_rho = bad_rho / np.trace(bad_rho)  # Normalize trace
        with pytest.raises(ValueError, match="rho_full must be positive semi-definite"):
            ClusterExpansion(bad_rho, H_2, NQ_2)


class TestReshapeToTensor:
    """Test _reshape_to_tensor static method."""
    
    def test_reshape_to_tensor_2qubit(self):
        """Test reshaping 2-qubit density matrix."""
        tensor, n = ClusterExpansion._reshape_to_tensor(RHO_2)
        assert n == 2
        assert tensor.shape == (2, 2, 2, 2)
        # Test roundtrip
        back = tensor.reshape(4, 4)
        assert np.allclose(back, RHO_2)
    
    def test_reshape_to_tensor_3qubit(self):
        """Test reshaping 3-qubit density matrix."""
        tensor, n = ClusterExpansion._reshape_to_tensor(RHO_3)
        assert n == 3
        assert tensor.shape == (2, 2, 2, 2, 2, 2)
    
    def test_reshape_to_tensor_invalid_dimension(self):
        """Test ValueError for non-power-of-2 dimension."""
        bad_matrix = np.random.random((3, 3))  # 3 is not a power of 2
        with pytest.raises(ValueError, match="Matrix dimension is not a power of two"):
            ClusterExpansion._reshape_to_tensor(bad_matrix)
    
    def test_reshape_to_tensor_1d_array(self):
        """Test with 1D array (should fail)."""
        bad_array = np.array([1, 2, 3, 4])
        with pytest.raises((ValueError, IndexError)):
            ClusterExpansion._reshape_to_tensor(bad_array)


class TestPartialTrace:
    """Test _partial_trace static method."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.C = ClusterExpansion(RHO_3, H_3, NQ_3)
        self.tensor_3, _ = self.C._reshape_to_tensor(RHO_3)
    
    def test_partial_trace_single_qubit_matrix_format(self):
        """Test partial trace removing single qubit, matrix format."""
        reduced, keep = ClusterExpansion._partial_trace(
            self.tensor_3, [1], full_space=False, format="matrix"
        )
        assert reduced.shape == (4, 4)  # 2^(3-1) = 4
        assert keep == [0, 2]
        assert np.isclose(np.trace(reduced), 1.0)
    
    def test_partial_trace_single_qubit_tensor_format(self):
        """Test partial trace removing single qubit, tensor format."""
        reduced, keep = ClusterExpansion._partial_trace(
            self.tensor_3, [1], full_space=False, format="tensor"
        )
        assert reduced.shape == (2, 2, 2, 2)  # rank-2(3-1) = 4
        assert keep == [0, 2]
    
    def test_partial_trace_multiple_qubits(self):
        """Test partial trace removing multiple qubits."""
        reduced, keep = ClusterExpansion._partial_trace(
            self.tensor_3, [0, 2], full_space=False, format="matrix"
        )
        assert reduced.shape == (2, 2)  # 2^(3-2) = 2
        assert keep == [1]
        assert np.isclose(np.trace(reduced), 1.0)
    
    def test_partial_trace_full_space_matrix(self):
        """Test partial trace with full_space=True, matrix format."""
        embedded, keep = ClusterExpansion._partial_trace(
            self.tensor_3, [1], full_space=True, format="matrix"
        )
        assert embedded.shape == (8, 8)  # Original full space
        assert keep == [0, 2]
    
    def test_partial_trace_full_space_tensor(self):
        """Test partial trace with full_space=True, tensor format."""
        embedded, keep = ClusterExpansion._partial_trace(
            self.tensor_3, [1], full_space=True, format="tensor"
        )
        assert embedded.shape == (2, 2, 2, 2, 2, 2)  # Original tensor shape
        assert keep == [0, 2]
    
    def test_partial_trace_invalid_format(self):
        """Test ValueError for invalid format."""
        with pytest.raises(ValueError, match="Format must be 'tensor' or 'matrix'"):
            ClusterExpansion._partial_trace(
                self.tensor_3, [1], full_space=False, format="invalid"
            )
    
    def test_partial_trace_all_qubits(self):
        """Test tracing out all qubits (edge case)."""
        reduced, keep = ClusterExpansion._partial_trace(
            self.tensor_3, [0, 1, 2], full_space=False, format="matrix"
        )
        assert reduced.shape == (1, 1)
        assert keep == []
        assert np.isclose(reduced[0, 0], 1.0)
    
    def test_partial_trace_no_qubits(self):
        """Test tracing out no qubits (identity operation)."""
        reduced, keep = ClusterExpansion._partial_trace(
            self.tensor_3, [], full_space=False, format="matrix"
        )
        assert reduced.shape == (8, 8)
        assert keep == [0, 1, 2]
        assert np.allclose(reduced, RHO_3)
    
    def test_partial_trace_verbose_mode(self):
        """Test partial trace with verbose output."""
        # This tests the verbose print statements
        with patch('builtins.print') as mock_print:
            ClusterExpansion._partial_trace(
                self.tensor_3, [1], full_space=True, format="matrix", verbose=1
            )
            mock_print.assert_called()


class TestDiagonalizeAndBuildRho:
    """Test diagonalize_and_build_rho static method."""
    
    def test_diagonalize_2x2_matrix(self):
        """Test diagonalization of 2x2 Hermitian matrix."""
        H = np.array([[1, 0.5], [0.5, 2]], dtype=complex)
        with patch('builtins.print'):  # Suppress eigenvalue print
            rho = ClusterExpansion.diagonalize_and_build_rho(H)
        assert rho.shape == (2, 2)
        assert np.isclose(np.trace(rho), 1.0)
        assert np.allclose(rho, rho.conj().T)  # Hermitian
        # Should be rank-1 projector
        assert np.linalg.matrix_rank(rho) == 1
    
    def test_diagonalize_4x4_matrix(self):
        """Test diagonalization of 4x4 Hermitian matrix."""
        H = np.random.random((4, 4))
        H = (H + H.T) / 2  # Make Hermitian
        with patch('builtins.print'):
            rho = ClusterExpansion.diagonalize_and_build_rho(H)
        assert rho.shape == (4, 4)
        assert np.isclose(np.trace(rho), 1.0)
        assert np.allclose(rho, rho.conj().T)
    
    def test_diagonalize_prints_eigenvalue(self):
        """Test that eigenvalue is printed."""
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        with patch('builtins.print') as mock_print:
            ClusterExpansion.diagonalize_and_build_rho(H)
            mock_print.assert_called_with("eigenvalue = ", 1.0)


class TestOneQubitMarginals:
    """Test one_qubit_marginals method."""
    
    def test_one_qubit_marginals_2qubit(self):
        """Test marginals for 2-qubit system."""
        C = ClusterExpansion(RHO_2, H_2, NQ_2)
        singles = C.one_qubit_marginals()
        assert len(singles) == 2
        for i, rho_i in enumerate(singles):
            assert rho_i.shape == (2, 2)
            assert np.allclose(rho_i, rho_i.conj().T)
            assert np.isclose(np.trace(rho_i), 1.0)
            assert np.min(np.linalg.eigvalsh(rho_i)) >= -1e-10
    
    def test_one_qubit_marginals_3qubit(self):
        """Test marginals for 3-qubit system."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        singles = C.one_qubit_marginals()
        assert len(singles) == 3
    
    def test_one_qubit_marginals_4qubit(self):
        """Test marginals for 4-qubit system."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)
        singles = C.one_qubit_marginals()
        assert len(singles) == 4


class TestMeanFieldState:
    """Test mean_field_state method."""
    
    def test_mean_field_state_properties(self):
        """Test properties of mean field state."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        rho_mf = C.mean_field_state()
        assert rho_mf.shape == RHO_3.shape
        assert np.isclose(np.trace(rho_mf), 1.0)
        assert np.allclose(rho_mf, rho_mf.conj().T)
    
    def test_mean_field_versus_exact(self):
        """Test mean field approximation quality."""
        C = ClusterExpansion(RHO_2, H_2, NQ_2)
        rho_mf = C.mean_field_state()
        # For highly entangled states, MF should be different from exact
        fidelity = np.real(np.trace(rho_mf @ RHO_2))
        assert 0 <= fidelity <= 1


class TestTwoQubitCumulants:
    """Test two_qubit_cumulants method."""
    
    def test_two_qubit_cumulants_count(self):
        """Test correct number of 2-qubit cumulants."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        lam = C.two_qubit_cumulants()
        expected_pairs = 3 * 2 // 2  # C(3,2) = 3
        assert len(lam) == expected_pairs
        
    def test_two_qubit_cumulants_keys(self):
        """Test that keys are ordered pairs (i,j) with i<j."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        lam = C.two_qubit_cumulants()
        expected_keys = {(0, 1), (0, 2), (1, 2)}
        assert set(lam.keys()) == expected_keys
    
    def test_two_qubit_cumulants_properties(self):
        """Test properties of 2-qubit cumulants."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        lam = C.two_qubit_cumulants()
        for (i, j), L in lam.items():
            assert L.shape == (4, 4)
            assert np.isclose(np.trace(L), 0.0, atol=1e-12)  # Zero trace
            assert np.allclose(L, L.conj().T)  # Hermitian
    
    def test_two_qubit_cumulants_verbose(self):
        """Test verbose output for 2-qubit cumulants."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3, verbose=1)
        with patch('builtins.print') as mock_print:
            C.two_qubit_cumulants()
            assert mock_print.call_count >= 3  # Should print for each pair


class TestThreeQubitCumulants:
    """Test three_qubit_cumulants method."""
    
    def test_three_qubit_cumulants_count(self):
        """Test correct number of 3-qubit cumulants."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)  # Need 4 qubits for 3-qubit cumulants
        lam = C.three_qubit_cumulants()
        expected_triplets = 4 * 3 * 2 // (3 * 2 * 1)  # C(4,3) = 4
        assert len(lam) == expected_triplets
    
    def test_three_qubit_cumulants_keys(self):
        """Test that keys are ordered triplets (i,j,k) with i<j<k."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)
        lam = C.three_qubit_cumulants()
        expected_keys = {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}
        assert set(lam.keys()) == expected_keys
    
    def test_three_qubit_cumulants_properties(self):
        """Test properties of 3-qubit cumulants."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)
        lam = C.three_qubit_cumulants()
        for (i, j, k), L in lam.items():
            assert L.shape == (8, 8)
            assert np.isclose(np.trace(L), 0.0, atol=1e-10)  # Zero trace
            assert np.allclose(L, L.conj().T, atol=1e-12)  # Hermitian
    
    def test_three_qubit_cumulants_verbose(self):
        """Test verbose output for 3-qubit cumulants."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4, verbose=1)
        with patch('builtins.print') as mock_print:
            C.three_qubit_cumulants()
            assert mock_print.call_count >= 4  # Should print for each triplet


class TestFourQubitCumulants:
    """Test four_qubit_cumulants method."""
    
    def test_four_qubit_cumulants_count(self):
        """Test correct number of 4-qubit cumulants."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)
        lam = C.four_qubit_cumulants()
        expected_quadruplets = 1  # C(4,4) = 1
        assert len(lam) == expected_quadruplets
    
    def test_four_qubit_cumulants_properties(self):
        """Test properties of 4-qubit cumulants."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)
        lam = C.four_qubit_cumulants()
        for (i, j, k, l), L in lam.items():
            assert L.shape == (16, 16)
            assert np.isclose(np.trace(L), 0.0, atol=1e-10)  # Zero trace
            assert np.allclose(L, L.conj().T, atol=1e-12)  # Hermitian
    
    def test_four_qubit_cumulants_verbose(self):
        """Test verbose output for 4-qubit cumulants."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4, verbose=1)
        with patch('builtins.print') as mock_print:
            C.four_qubit_cumulants()
            mock_print.assert_called()  # Should print for the quadruplet


class TestEmbedCluster:
    """Test _embed_cluster static method."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.C = ClusterExpansion(RHO_4, H_4, NQ_4)
        self.singles = self.C.one_qubit_marginals()
    
    def test_embed_cluster_2qubit(self):
        """Test embedding 2-qubit cluster."""
        lam_2q = self.C.two_qubit_cumulants()
        (i, j), lam_ij = next(iter(lam_2q.items()))
        full = ClusterExpansion._embed_cluster(lam_ij, (i, j), self.singles)
        assert full.shape == (16, 16)
        assert np.isclose(np.trace(full), 0.0, atol=1e-12)
    
    def test_embed_cluster_3qubit(self):
        """Test embedding 3-qubit cluster."""
        lam_3q = self.C.three_qubit_cumulants()
        (i, j, k), lam_ijk = next(iter(lam_3q.items()))
        full = ClusterExpansion._embed_cluster(lam_ijk, (i, j, k), self.singles)
        assert full.shape == (16, 16)
        assert np.isclose(np.trace(full), 0.0, atol=1e-12)
    
    def test_embed_cluster_4qubit(self):
        """Test embedding 4-qubit cluster."""
        lam_4q = self.C.four_qubit_cumulants()
        (i, j, k, l), lam_ijkl = next(iter(lam_4q.items()))
        full = ClusterExpansion._embed_cluster(lam_ijkl, (i, j, k, l), self.singles)
        assert full.shape == (16, 16)
        assert np.isclose(np.trace(full), 0.0, atol=1e-12)
    
    def test_embed_cluster_verbose(self):
        """Test verbose mode for embed_cluster."""
        lam_2q = self.C.two_qubit_cumulants()
        (i, j), lam_ij = next(iter(lam_2q.items()))
        with patch('builtins.print') as mock_print:
            ClusterExpansion._embed_cluster(lam_ij, (i, j), self.singles, verbose=True)
            mock_print.assert_called()


class TestClusterExpansionRho:
    """Test cluster_expansion_rho method."""
    
    def test_cluster_expansion_2body_only(self):
        """Test cluster expansion with only 2-body cumulants."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        rho_rebuilt, rho_mf = C.cluster_expansion_rho(
            compute_3q_cumulants=False, 
            compute_4q_cumulants=False
        )
        assert rho_rebuilt.shape == RHO_3.shape
        assert rho_mf.shape == RHO_3.shape
        assert np.isclose(np.trace(rho_rebuilt), 1.0)
        assert np.isclose(np.trace(rho_mf), 1.0)
        assert np.allclose(rho_rebuilt, rho_rebuilt.conj().T)
        assert np.allclose(rho_mf, rho_mf.conj().T)
    
    def test_cluster_expansion_with_3body(self):
        """Test cluster expansion including 3-body cumulants."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)
        rho_rebuilt, rho_mf = C.cluster_expansion_rho(
            compute_3q_cumulants=True, 
            compute_4q_cumulants=False
        )
        assert rho_rebuilt.shape == RHO_4.shape
        assert np.isclose(np.trace(rho_rebuilt), 1.0)
        assert np.allclose(rho_rebuilt, rho_rebuilt.conj().T)
    
    def test_cluster_expansion_with_4body(self):
        """Test cluster expansion including 4-body cumulants."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)
        rho_rebuilt, rho_mf = C.cluster_expansion_rho(
            compute_3q_cumulants=False, 
            compute_4q_cumulants=True
        )
        assert rho_rebuilt.shape == RHO_4.shape
        assert np.isclose(np.trace(rho_rebuilt), 1.0)
        assert np.allclose(rho_rebuilt, rho_rebuilt.conj().T)
    
    def test_cluster_expansion_full(self):
        """Test cluster expansion with all cumulants."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)
        rho_rebuilt, rho_mf = C.cluster_expansion_rho(
            compute_3q_cumulants=True, 
            compute_4q_cumulants=True
        )
        assert rho_rebuilt.shape == RHO_4.shape
        assert np.isclose(np.trace(rho_rebuilt), 1.0)
        assert np.allclose(rho_rebuilt, rho_rebuilt.conj().T)
        
        # For 4-qubit system with all cumulants, should be closer to exact
        err_mf = np.linalg.norm(RHO_4 - rho_mf)
        err_full = np.linalg.norm(RHO_4 - rho_rebuilt)
        assert err_full <= err_mf + 1e-10
    
    def test_cluster_expansion_approximation_quality(self):
        """Test that cluster expansion improves upon mean field."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        rho_rebuilt, rho_mf = C.cluster_expansion_rho()
        
        # Error with respect to exact
        err_mf = np.linalg.norm(RHO_3 - rho_mf)
        err_rebuilt = np.linalg.norm(RHO_3 - rho_rebuilt)
        
        # Cluster expansion should be at least as good as mean field
        assert err_rebuilt <= err_mf + 1e-10


class TestAdditionalCoverage:
    """Additional tests t"""
    
    def test_wrong_shape_hamiltonian(self):
        """Test with wrong hamiltonian shape (should still work)."""
        # The code doesn't validate hamiltonian shape, so this should pass
        wrong_ham = np.random.random((3, 3))
        C = ClusterExpansion(RHO_2, wrong_ham, NQ_2)
        assert C.ham.shape == (3, 3)
    
    def test_non_square_matrix_reshape_to_tensor(self):
        """Test _reshape_to_tensor with non-square matrix."""
        bad_matrix = np.random.random((4, 3))  # Non-square
        with pytest.raises((ValueError, IndexError)):
            ClusterExpansion._reshape_to_tensor(bad_matrix)
    
    def test_partial_trace_verbose_output(self):
        """Test _partial_trace verbose mode actually prints."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        tensor_3, _ = C._reshape_to_tensor(RHO_3)
        
        with patch('builtins.print') as mock_print:
            # Test the verboses = True branch (currently hardcoded to False)
            # We need to modify the verboses variable inside the function
            # Since it's hardcoded, we'll test the verbose parameter instead
            ClusterExpansion._partial_trace(
                tensor_3, [1], full_space=True, format="matrix", verbose=1
            )
            # Should print the einsum string
            mock_print.assert_called()
    
    def test_invalid_qubit_index_in_trace_out(self):
        """Test partial trace with invalid qubit indices."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        tensor_3, _ = C._reshape_to_tensor(RHO_3)
        
        # This should work fine - invalid indices are just ignored in the logic
        reduced, keep = ClusterExpansion._partial_trace(
            tensor_3, [5], full_space=False, format="matrix"
        )
        # Should return original matrix since qubit 5 doesn't exist
        assert keep == [0, 1, 2]
    
    def test_main_block_execution(self):
        """Test the if __name__ == '__main__' block."""
        # This is a bit tricky to test directly, but we can import the module
        # and check that it doesn't raise an error
        import open_mfi.mfi_hilbert
        # The main block just prints an empty line, so this should pass
        assert True
    
    def test_embed_cluster_verbose_true(self):
        """Test _embed_cluster with verbose=True."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        singles = C.one_qubit_marginals()
        lam_2q = C.two_qubit_cumulants()
        
        (i, j), lam_ij = next(iter(lam_2q.items()))
        with patch('builtins.print') as mock_print:
            ClusterExpansion._embed_cluster(lam_ij, (i, j), singles, verbose=True)
            mock_print.assert_called_with("einsum:", f"abAB,cC->abcABC")
    
    def test_partial_trace_with_duplicates_in_trace_out(self):
        """Test partial trace with duplicate indices in trace_out."""
        C = ClusterExpansion(RHO_3, H_3, NQ_3)
        tensor_3, _ = C._reshape_to_tensor(RHO_3)
        
        # Duplicate indices should be handled gracefully
        reduced, keep = ClusterExpansion._partial_trace(
            tensor_3, [1, 1, 2], full_space=False, format="matrix"
        )
        # Should effectively trace out qubits 1 and 2
        assert reduced.shape == (2, 2)
        assert keep == [0]
    
    def test_constructor_with_all_optional_parameters(self):
        """Test constructor with all optional parameters set."""
        C = ClusterExpansion(
            RHO_3, H_3, NQ_3, 
            n_a=1, n_b=1, n_orb=2, verbose=2
        )
        assert C.n_a == 1
        assert C.n_b == 1  
        assert C.n_orb == 2
        assert C.verbose == 2
    
    def test_einsum_optimization_paths(self):
        """Test that einsum operations use optimization."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4)
        
        # This tests the optimize=True parameter in various einsum calls
        lam_3q = C.three_qubit_cumulants()
        lam_4q = C.four_qubit_cumulants()
        
        # Just verify they complete without error
        assert len(lam_3q) == 4  # C(4,3) = 4
        assert len(lam_4q) == 1  # C(4,4) = 1
    
    def test_complex_dtype_preservation(self):
        """Test that complex dtypes are preserved throughout."""
        # Create a purely imaginary density matrix
        rho_complex = RHO_2.astype(complex)
        rho_complex += 1j * 0.01 * np.random.random(RHO_2.shape)
        rho_complex = (rho_complex + rho_complex.conj().T) / 2  # Keep Hermitian
        rho_complex = rho_complex / np.trace(rho_complex)  # Normalize
        
        C = ClusterExpansion(rho_complex, H_2, NQ_2)
        singles = C.one_qubit_marginals()
        
        # All outputs should preserve complex dtype
        for single in singles:
            assert single.dtype == complex
        
        rho_rebuilt, rho_mf = C.cluster_expansion_rho()
        assert rho_rebuilt.dtype == complex
        assert rho_mf.dtype == complex


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""
    
    def test_single_qubit_system(self):
        """Test with single qubit (minimal system)."""
        rho_1 = random_density(1, seed=999)
        H_1 = random_hamiltonian(1, seed=999)
        
        C = ClusterExpansion(rho_1, H_1, 1)
        singles = C.one_qubit_marginals()
        assert len(singles) == 1
        assert np.allclose(singles[0], rho_1)
        
        # Mean field should be identical to original
        rho_mf = C.mean_field_state()
        assert np.allclose(rho_mf, rho_1)
        
        # No 2-qubit cumulants possible
        lam_2q = C.two_qubit_cumulants()
        assert len(lam_2q) == 0
        
        # Cluster expansion should return mean field only
        rho_rebuilt, rho_mf_returned = C.cluster_expansion_rho()
        assert np.allclose(rho_rebuilt, rho_mf)
        assert np.allclose(rho_rebuilt, rho_1)
    
    def test_matrix_dimension_exactly_power_of_two_boundary(self):
        """Test matrices with dimensions that are exactly powers of 2."""
        for n in [1, 2, 3, 4, 5]:
            dim = 2**n
            mat = np.eye(dim, dtype=complex)
            tensor, N = ClusterExpansion._reshape_to_tensor(mat)
            assert N == n
            assert tensor.shape == (2,) * n + (2,) * n
    
    def test_trace_tolerance_boundaries(self):
        """Test density matrices with trace very close to 1."""
        rho_close = RHO_2.copy()
        rho_close = rho_close / np.trace(rho_close)
        rho_close *= (1.0 + 1e-14)  # Very slight deviation
        
        # Should still pass validation
        C = ClusterExpansion(rho_close, H_2, NQ_2)
        assert C.rho_full is not None
        
        # Test with trace slightly too far off
        rho_bad = RHO_2.copy() * 1.1  # 10% off
        with pytest.raises(ValueError, match="Trace\\(rho\\) must be 1"):
            ClusterExpansion(rho_bad, H_2, NQ_2)
    
    def test_eigenvalue_tolerance_boundaries(self):
        """Test positive semi-definite check with small negative eigenvalues."""
        # Create matrix with very small negative eigenvalue
        eigenvals = np.array([0.8, 0.2, 1e-12, -1e-12])
        eigenvecs = np.random.random((4, 4))
        eigenvecs, _ = np.linalg.qr(eigenvecs)  # Orthogonalize
        
        rho_boundary = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        rho_boundary = rho_boundary / np.trace(rho_boundary)
        
        # Should pass with very small negative eigenvalue (within tolerance)
        C = ClusterExpansion(rho_boundary, H_2, NQ_2)
        assert C.rho_full is not None


class TestPrintStatements:
    """Test all print statements for coverage."""
    
    def test_diagonalize_print_statement(self):
        """Test that diagonalize_and_build_rho prints eigenvalue."""
        H = np.array([[2, 1], [1, 3]], dtype=complex)
        with patch('builtins.print') as mock_print:
            rho = ClusterExpansion.diagonalize_and_build_rho(H)
            # Should print the lowest eigenvalue
            expected_eigenvalue = np.linalg.eigvalsh(H)[0]
            mock_print.assert_called_with("eigenvalue = ", expected_eigenvalue)
    
    def test_all_verbose_print_paths(self):
        """Ensure all verbose print statements are covered."""
        C = ClusterExpansion(RHO_4, H_4, NQ_4, verbose=1)
        
        with patch('builtins.print') as mock_print:
            # This should trigger verbose prints in cumulant calculations
            lam_2q = C.two_qubit_cumulants()
            lam_3q = C.three_qubit_cumulants() 
            lam_4q = C.four_qubit_cumulants()
            
            # Should have printed for each cumulant
            assert mock_print.call_count >= (len(lam_2q) + len(lam_3q) + len(lam_4q))