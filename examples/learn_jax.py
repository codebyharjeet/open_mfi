import time
import numpy as np
import jax
import jax.numpy as jnp

def random_symmetric(n, seed=None):
    """Generate a random symmetric real matrix H of size nxn."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n)
    return (A + A.T) / 2

def exact_ground_state_energy(H):
    """Exact ground-state energy via full diagonalization."""
    w, _ = np.linalg.eig(H)
    return float(np.min(w))

def energy_from_X(X, H):
    """
    Parametrized energy:
      M = X @ X.T         (always PSD)
      rho = M / Tr(M)     (unit trace)
      E = Tr(rho @ H)
    """
    M = X @ X.T
    rho = M / jnp.trace(M)
    return jnp.trace(rho @ H)

def test(n=6, lr=1e-2, max_iter=2000, seed=0):

    H = random_symmetric(n, seed)
    print("Shape of H = ", H.shape)

    E_exact = exact_ground_state_energy(H)
    print(f"Exact ground-state energy:    {E_exact:20.8f}")

    energy_and_grad = jax.jit(jax.value_and_grad(energy_from_X, argnums=0))

    # Initialize X random
    key = jax.random.PRNGKey(seed)
    X = jax.random.normal(key, (n, n))

    print(f"Energy of Maximally mixed state:    {energy_from_X(X, H):20.8f}")

    # Gradient‐descent loop
    t0 = time.time()
    for i in range(max_iter):
        E_val, gX = energy_and_grad(X, H)
        X = X - lr * gX

    X.block_until_ready()
    t1 = time.time()

    E_opt = energy_from_X(X , H)
    print(f"JAX-GD optimized energy:       {E_opt:20.8f}")
    print(f"Gradient-descent runtime:      {t1 - t0:.3f} s")
    return H, X

if __name__ == "__main__":
    H, X = test(
        n=8,      # dimension
        lr=1e-0,  # learning rate
        max_iter=100,
        seed=2
    )



# Output - 
# Shape of H =  (8, 8)
# Exact ground‐state energy:             -3.37013258
# Energy of Maximally mixed state:             -0.58091283
# JAX-GD optimized energy:                -3.36777759
# Gradient-descent runtime:      0.095 s
