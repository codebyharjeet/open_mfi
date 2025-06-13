"""
Test suite for the ClusterExpansion class in mfi_hilbert.py
"""

import pytest
import numpy as np
import itertools
from typing import Dict, Tuple, List
from open_mfi.mfi_hilbert import ClusterExpansion


class TestClusterExpansion:
    """Test class for ClusterExpansion functionality."""
    
    def _create_random_hermitian(self, dim):
        """Helper method to create a random Hermitian matrix."""
        A = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        return A + A.conj().T
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create simple 2-qubit test cases
        self.n_qubits_2 = 2
        self.dim_2 = 2**self.n_qubits_2
        
        # Identity density matrix (maximally mixed state)
        self.rho_mixed_2 = np.eye(self.dim_2) / self.dim_2
        
        # Product state |00⟩⟨00|
        self.rho_00 = np.zeros((self.dim_2, self.dim_2), dtype=complex)
        self.rho_00[0, 0] = 1.0
        
        # Bell state (|00⟩ + |11⟩)/√2
        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        self.rho_bell = np.outer(bell_state, bell_state.conj())
        
        # Simple Hamiltonians
        self.ham_2 = self._create_random_hermitian(self.dim_2)
        
        # 3-qubit test cases
        self.n_qubits_3 = 3
        self.dim_3 = 2**self.n_qubits_3
        self.rho_mixed_3 = np.eye(self.dim_3) / self.dim_3
        self.ham_3 = self._create_random_hermitian(self.dim_3)
        
        # 4-qubit test cases  
        self.n_qubits_4 = 4
        self.dim_4 = 2**self.n_qubits_4
        self.rho_mixed_4 = np.eye(self.dim_4) / self.dim_4
        self.ham_4 = self._create_random_hermitian(self.dim_4)

    def test_init_valid_inputs(self):
        """Test ClusterExpansion initialization with valid inputs."""
        ce = ClusterExpansion(self.rho_mixed_2, self.ham_2, self.n_qubits_2)
        assert ce.N == self.n_qubits_2
        assert np.allclose(ce.rho_full, self.rho_mixed_2)
        assert np.allclose(ce.ham, self.ham_2)

    def test_init_with_optional_parameters(self):
        """Test initialization with optional parameters."""
        ce = ClusterExpansion(
            self.rho_mixed_2, self.ham_2, self.n_qubits_2,
            n_a=1, n_b=1, n_orb=2, verbose=1
        )
        assert ce.n_a == 1
        assert ce.n_b == 1
        assert ce.n_orb == 2
        assert ce.verbose == 1

    def test_init_invalid_shape(self):
        """Test initialization with invalid density matrix shape."""
        wrong_shape_rho = np.eye(3) / 3  # 3x3 matrix for 2-qubit system
        with pytest.raises(ValueError, match="rho_full has wrong shape"):
            ClusterExpansion(wrong_shape_rho, self.ham_2, self.n_qubits_2)

    def test_init_non_hermitian(self):
        """Test initialization with non-Hermitian density matrix."""
        non_hermitian = np.array([[1, 1j], [0, 0]], dtype=complex)
        with pytest.raises(ValueError, match="rho_full must be Hermitian"):
            ClusterExpansion(non_hermitian, self.ham_2, 1)

    def test_init_wrong_trace(self):
        """Test initialization with wrong trace."""
        wrong_trace = np.eye(self.dim_2) * 2  # Trace = 2 instead of 1
        with pytest.raises(ValueError, match="Trace\\(rho\\) must be 1"):
            ClusterExpansion(wrong_trace, self.ham_2, self.n_qubits_2)

    def test_init_negative_eigenvalues(self):
        """Test initialization with negative eigenvalues."""
        # Create a matrix with negative eigenvalues
        negative_eig = np.array([[0.5, 0.6], [0.6, 0.5]], dtype=complex)
        with pytest.raises(ValueError, match="rho_full must be positive semi-definite"):
            ClusterExpansion(negative_eig, np.eye(2), 1)

    def test_unfold_fold_consistency(self):
        """Test that unfold and fold are inverse operations."""
        # Test with 2-qubit system
        tensor_2 = ClusterExpansion.unfold(self.rho_mixed_2)
        reconstructed_2 = ClusterExpansion.fold(tensor_2)
        assert np.allclose(reconstructed_2, self.rho_mixed_2)
        
        # Test with 3-qubit system
        tensor_3 = ClusterExpansion.unfold(self.rho_mixed_3)
        reconstructed_3 = ClusterExpansion.fold(tensor_3)
        assert np.allclose(reconstructed_3, self.rho_mixed_3)

    def test_unfold_invalid_input(self):
        """Test unfold with invalid inputs."""
        # Non-square matrix
        with pytest.raises(ValueError, match="Matrix must be square"):
            ClusterExpansion.unfold(np.ones((2, 3)))
        
        # Dimension not power of 2
        with pytest.raises(ValueError, match="Matrix dimension is not a power of two"):
            ClusterExpansion.unfold(np.ones((3, 3)))

    def test_fold_invalid_input(self):
        """Test fold with invalid inputs."""
        # Odd rank tensor
        with pytest.raises(ValueError, match="Tensor must have even rank"):
            ClusterExpansion.fold(np.ones((2, 2, 2)))

    def test_rho_tensor_property(self):
        """Test the rho_tensor property and caching."""
        ce = ClusterExpansion(self.rho_bell, self.ham_2, self.n_qubits_2)
        
        # First access should compute and cache
        tensor1 = ce.rho_tensor
        assert tensor1.shape == (2, 2, 2, 2)
        
        # Second access should return cached version
        tensor2 = ce.rho_tensor
        assert tensor1 is tensor2  # Same object reference
        assert np.allclose(tensor1, tensor2)

    def test_partial_trace_single_qubit(self):
        """Test partial trace over single qubits."""
        ce = ClusterExpansion(self.rho_bell, self.ham_2, self.n_qubits_2)
        
        # Trace out qubit 0
        rho_1, keep = ClusterExpansion._partial_trace(
            ce.rho_tensor, [0], full_space=False
        )
        assert rho_1.shape == (2, 2)
        assert keep == [1]
        assert np.isclose(np.trace(rho_1), 1.0)
        
        # Trace out qubit 1
        rho_0, keep = ClusterExpansion._partial_trace(
            ce.rho_tensor, [1], full_space=False
        )
        assert rho_0.shape == (2, 2)
        assert keep == [0]
        assert np.isclose(np.trace(rho_0), 1.0)

    def test_partial_trace_full_space(self):
        """Test partial trace with full_space=True."""
        ce = ClusterExpansion(self.rho_bell, self.ham_2, self.n_qubits_2)
        
        rho_embedded, keep = ClusterExpansion._partial_trace(
            ce.rho_tensor, [0], full_space=True
        )
        assert rho_embedded.shape == (4, 4)
        assert keep == [1]
        assert np.isclose(np.trace(rho_embedded), 1.0)

    def test_partial_trace_tensor_format(self):
        """Test partial trace with tensor format."""
        ce = ClusterExpansion(self.rho_mixed_3, self.ham_3, self.n_qubits_3)
        
        # Trace out qubit 0, return as tensor
        tensor_reduced, keep = ClusterExpansion._partial_trace(
            ce.rho_tensor, [0], full_space=False, format="tensor"
        )
        assert tensor_reduced.shape == (2, 2, 2, 2)
        assert keep == [1, 2]

    def test_diagonalize_and_build_rho(self):
        """Test ground state projector construction."""
        # Create simple 4x4 Hamiltonian
        H = np.array([
            [1, 0, 0, 0],
            [0, 2, 0, 0], 
            [0, 0, 3, 0],
            [0, 0, 0, 4]
        ], dtype=complex)
        
        # Redirect stdout to capture print
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        rho_gs = ClusterExpansion.diagonalize_and_build_rho(H)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        assert rho_gs.shape == (4, 4)
        assert np.isclose(np.trace(rho_gs), 1.0)
        assert np.allclose(rho_gs, rho_gs.conj().T)  # Hermitian
        assert np.min(np.linalg.eigvalsh(rho_gs)) >= -1e-10  # Positive semi-definite

    def test_one_qubit_marginals(self):
        """Test computation of single-qubit marginals."""
        ce = ClusterExpansion(self.rho_mixed_3, self.ham_3, self.n_qubits_3)
        marginals = ce.one_qubit_marginals()
        
        assert len(marginals) == self.n_qubits_3
        for rho_i in marginals:
            assert rho_i.shape == (2, 2)
            assert np.isclose(np.trace(rho_i), 1.0)
            assert np.allclose(rho_i, rho_i.conj().T)

    def test_mean_field_state(self):
        """Test mean field state construction."""
        ce = ClusterExpansion(self.rho_mixed_2, self.ham_2, self.n_qubits_2)
        rho_mf = ce.mean_field_state()
        
        assert rho_mf.shape == (self.dim_2, self.dim_2)
        assert np.isclose(np.trace(rho_mf), 1.0)
        assert np.allclose(rho_mf, rho_mf.conj().T)
        
        # Test caching
        rho_mf2 = ce.mean_field_state()
        assert rho_mf is rho_mf2

    def test_mean_field_product_state(self):
        """Test mean field state for product states."""
        ce = ClusterExpansion(self.rho_00, self.ham_2, self.n_qubits_2)
        rho_mf = ce.mean_field_state()
        
        # For product state |00⟩⟨00|, mean field should equal original
        assert np.allclose(rho_mf, self.rho_00)

    def test_two_qubit_cumulants(self):
        """Test two-qubit cumulant computation."""
        ce = ClusterExpansion(self.rho_bell, self.ham_2, self.n_qubits_2)
        cumulants = ce.two_qubit_cumulants()
        
        assert len(cumulants) == 1  # Only (0,1) pair for 2 qubits
        assert (0, 1) in cumulants
        
        lam_01 = cumulants[(0, 1)]
        assert lam_01.shape == (4, 4)
        assert np.isclose(np.trace(lam_01), 0.0)  # Zero trace by construction

    def test_two_qubit_cumulants_product_state(self):
        """Test two-qubit cumulants for product states."""
        ce = ClusterExpansion(self.rho_00, self.ham_2, self.n_qubits_2)
        cumulants = ce.two_qubit_cumulants()
        
        lam_01 = cumulants[(0, 1)]
        # For product states, cumulants should be zero
        assert np.allclose(lam_01, 0, atol=1e-10)

    def test_three_qubit_cumulants(self):
        """Test three-qubit cumulant computation."""
        ce = ClusterExpansion(self.rho_mixed_3, self.ham_3, self.n_qubits_3)
        cumulants = ce.three_qubit_cumulants()
        
        assert len(cumulants) == 1  # Only (0,1,2) triplet for 3 qubits
        assert (0, 1, 2) in cumulants
        
        lam_012 = cumulants[(0, 1, 2)]
        assert lam_012.shape == (8, 8)
        assert np.isclose(np.trace(lam_012), 0.0, atol=1e-10)

    def test_four_qubit_cumulants(self):
        """Test four-qubit cumulant computation."""
        ce = ClusterExpansion(self.rho_mixed_4, self.ham_4, self.n_qubits_4)
        cumulants = ce.four_qubit_cumulants()
        
        assert len(cumulants) == 1  # Only (0,1,2,3) quadruplet for 4 qubits
        assert (0, 1, 2, 3) in cumulants
        
        lam_0123 = cumulants[(0, 1, 2, 3)]
        assert lam_0123.shape == (16, 16)
        assert np.isclose(np.trace(lam_0123), 0.0, atol=1e-10)

    def test_cumulant_caching(self):
        """Test that cumulants are properly cached."""
        ce = ClusterExpansion(self.rho_bell, self.ham_2, self.n_qubits_2)
        
        # First call should compute and cache
        lam1 = ce._get_cumulant((0, 1))
        # Second call should return cached version
        lam2 = ce._get_cumulant((0, 1))
        
        assert lam1 is lam2  # Same object reference
        assert np.allclose(lam1, lam2)

    def test_marginal_caching(self):
        """Test that marginals are properly cached."""
        ce = ClusterExpansion(self.rho_mixed_3, self.ham_3, self.n_qubits_3)
        
        # First call should compute and cache
        rho1 = ce._get_marginal((0, 1))
        # Second call should return cached version
        rho2 = ce._get_marginal((0, 1))
        
        assert rho1 is rho2  # Same object reference
        assert np.allclose(rho1, rho2)

    def test_embed_cluster_two_qubit(self):
        """Test embedding of two-qubit clusters."""
        ce = ClusterExpansion(self.rho_mixed_3, self.ham_3, self.n_qubits_3)
        singles = ce.one_qubit_marginals()
        
        # Get a 2-qubit cumulant
        lam_01 = ce._get_cumulant((0, 1))
        
        # Embed it in full 3-qubit space
        embedded = ClusterExpansion._embed_cluster(lam_01, (0, 1), singles)
        
        assert embedded.shape == (8, 8)
        assert np.allclose(embedded, embedded.conj().T)

    def test_embed_cluster_three_qubit(self):
        """Test embedding of three-qubit clusters."""
        ce = ClusterExpansion(self.rho_mixed_4, self.ham_4, self.n_qubits_4)
        singles = ce.one_qubit_marginals()
        
        # Get a 3-qubit cumulant
        lam_012 = ce._get_cumulant((0, 1, 2))
        
        # Embed it in full 4-qubit space
        embedded = ClusterExpansion._embed_cluster(lam_012, (0, 1, 2), singles)
        
        assert embedded.shape == (16, 16)
        assert np.allclose(embedded, embedded.conj().T)

    def test_cluster_expansion_rho_basic(self):
        """Test basic cluster expansion reconstruction."""
        ce = ClusterExpansion(self.rho_bell, self.ham_2, self.n_qubits_2)
        
        rho_rebuilt, rho_mf = ce.cluster_expansion_rho()
        
        assert rho_rebuilt.shape == self.rho_bell.shape
        assert rho_mf.shape == self.rho_bell.shape
        assert np.isclose(np.trace(rho_rebuilt), 1.0)
        assert np.isclose(np.trace(rho_mf), 1.0)
        
        # For entangled state, rebuilt should be closer to original than mean field
        error_mf = np.linalg.norm(self.rho_bell - rho_mf)
        error_rebuilt = np.linalg.norm(self.rho_bell - rho_rebuilt)
        assert error_rebuilt <= error_mf

    def test_cluster_expansion_rho_with_higher_order(self):
        """Test cluster expansion with 3 and 4-qubit cumulants."""
        ce = ClusterExpansion(self.rho_mixed_4, self.ham_4, self.n_qubits_4)
        
        rho_rebuilt, rho_mf = ce.cluster_expansion_rho(
            compute_3q_cumulants=True,
            compute_4q_cumulants=True
        )
        
        assert rho_rebuilt.shape == (16, 16)
        assert np.isclose(np.trace(rho_rebuilt), 1.0)
        assert np.allclose(rho_rebuilt, rho_rebuilt.conj().T)

    def test_cluster_expansion_exact_for_product_state(self):
        """Test that cluster expansion is exact for product states."""
        ce = ClusterExpansion(self.rho_00, self.ham_2, self.n_qubits_2)
        
        rho_rebuilt, rho_mf = ce.cluster_expansion_rho()
        
        # For product states, mean field should be exact
        assert np.allclose(rho_mf, self.rho_00, atol=1e-10)
        # And cluster expansion should not change anything
        assert np.allclose(rho_rebuilt, self.rho_00, atol=1e-10)

    def test_verbose_output(self):
        """Test verbose output functionality."""
        import io
        import sys
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        ce = ClusterExpansion(self.rho_bell, self.ham_2, self.n_qubits_2, verbose=1)
        ce.two_qubit_cumulants()  # This should trigger verbose output
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "trace:" in output  # Should contain trace information
        assert "norm:" in output   # Should contain norm information

    def test_invalid_cumulant_order(self):
        """Test error handling for unsupported cumulant orders."""
        ce = ClusterExpansion(self.rho_mixed_2, self.ham_2, self.n_qubits_2)
        
        with pytest.raises(ValueError, match="Only 2-4 qubit cumulants supported"):
            ce._get_cumulant((0, 1, 2, 3, 4))  # 5-qubit cumulant

    def test_comprehensive_properties(self):
        """Test comprehensive properties of the cluster expansion."""
        # Test with a known entangled state
        ce = ClusterExpansion(self.rho_bell, self.ham_2, self.n_qubits_2)
        
        # Test all components
        singles = ce.one_qubit_marginals()
        rho_mf = ce.mean_field_state()
        cumulants_2q = ce.two_qubit_cumulants()
        rho_rebuilt, _ = ce.cluster_expansion_rho()
        
        # Check that all single-qubit marginals are valid density matrices
        for rho_i in singles:
            assert np.isclose(np.trace(rho_i), 1.0)
            assert np.min(np.linalg.eigvalsh(rho_i)) >= -1e-10
        
        # Check that mean field is a valid density matrix
        assert np.isclose(np.trace(rho_mf), 1.0)
        assert np.min(np.linalg.eigvalsh(rho_mf)) >= -1e-10
        
        # Check that all cumulants have zero trace
        for lam in cumulants_2q.values():
            assert np.isclose(np.trace(lam), 0.0, atol=1e-10)
        
        # Check that rebuilt state is a valid density matrix
        assert np.isclose(np.trace(rho_rebuilt), 1.0)
        assert np.min(np.linalg.eigvalsh(rho_rebuilt)) >= -1e-10


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow_2_qubit(self):
        """Test complete workflow for 2-qubit system."""
        n_qubits = 2
        dim = 2**n_qubits
        
        # Create a random Hermitian Hamiltonian
        ham = self._create_random_hermitian(dim)
        
        # Get ground state
        evals, evecs = np.linalg.eigh(ham)
        psi0 = evecs[:, 0]
        rho_exact = np.outer(psi0, psi0.conj())
        
        # Run cluster expansion
        ce = ClusterExpansion(rho_exact, ham, n_qubits)
        rho_rebuilt, rho_mf = ce.cluster_expansion_rho()
        
        # Check that reconstruction is reasonable
        assert np.isclose(np.trace(rho_rebuilt), 1.0)
        assert np.min(np.linalg.eigvalsh(rho_rebuilt)) >= -1e-10
        
        # Error should be small for ground state of random Hamiltonian
        error = np.linalg.norm(rho_exact - rho_rebuilt)
        assert error < 1.0  # Reasonable bound

    def test_complete_workflow_3_qubit(self):
        """Test complete workflow for 3-qubit system with higher-order cumulants."""
        n_qubits = 3
        dim = 2**n_qubits
        
        # Use maximally mixed state
        rho_exact = np.eye(dim) / dim
        ham = self._create_random_hermitian(dim)
        
        # Run cluster expansion with 3-qubit cumulants
        ce = ClusterExpansion(rho_exact, ham, n_qubits)
        rho_rebuilt, rho_mf = ce.cluster_expansion_rho(compute_3q_cumulants=True)
        
        # For maximally mixed state, cluster expansion should be exact
        assert np.allclose(rho_rebuilt, rho_exact, atol=1e-10)

    def _create_random_hermitian(self, dim):
        """Helper method to create a random Hermitian matrix."""
        A = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
        return A + A.conj().T

    def test_performance_scaling(self):
        """Test that the code runs efficiently for different system sizes."""
        import time
        
        times = {}
        for n_qubits in [2, 3, 4]:
            dim = 2**n_qubits
            rho = np.eye(dim) / dim
            ham = self._create_random_hermitian(dim)
            
            start_time = time.time()
            ce = ClusterExpansion(rho, ham, n_qubits)
            rho_rebuilt, _ = ce.cluster_expansion_rho()
            end_time = time.time()
            
            times[n_qubits] = end_time - start_time
            
            # Basic sanity checks
            assert np.isclose(np.trace(rho_rebuilt), 1.0)
            assert np.min(np.linalg.eigvalsh(rho_rebuilt)) >= -1e-10
        
        # Just ensure it completes without error
        assert len(times) == 3


if __name__ == "__main__":
    # Run with pytest for full test suite, or run basic smoke test
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic smoke test...")
        
        # Basic smoke test
        test_ce = TestClusterExpansion()
        test_ce.setup_method()
        
        print("Testing initialization...")
        test_ce.test_init_valid_inputs()
        
        print("Testing unfold/fold...")
        test_ce.test_unfold_fold_consistency()
        
        print("Testing marginals...")
        test_ce.test_one_qubit_marginals()
        
        print("Testing cumulants...")
        test_ce.test_two_qubit_cumulants()
        
        print("Testing cluster expansion...")
        test_ce.test_cluster_expansion_rho_basic()
        
        print("All basic tests passed!")