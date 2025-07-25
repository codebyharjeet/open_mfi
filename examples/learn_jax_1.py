import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

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

def energy_from_X(X_flat, H):
    """
    Parametrized energy:
      M = X @ X.T         (always PSD)
      rho = M / Tr(M)     (unit trace)
      E = Tr(rho @ H)
    """
    n = H.shape[0]
    X = X_flat.reshape((n, n))
    M = X @ X.T

    way_1 = False       
    if way_1:
        rho = M / jnp.trace(M)
        return jnp.trace(rho @ H)
    else:
        mu_trace = 1e2
        return jnp.trace(M @ H) + mu_trace * (jnp.trace(M) - 1.0)**2 

def test(n=6, seed=0):

    H = jnp.array(random_symmetric(n, seed))
    print("Shape of H = ", H.shape)

    E_exact = exact_ground_state_energy(H)
    print(f"Exact ground-state energy:    {E_exact:20.8f}")

    energy_jit = jax.jit(lambda x: energy_from_X(x, H))
    grad_jit   = jax.jit(jax.grad(lambda x: energy_from_X(x, H)))

    def energy_np(x_flat):
        return float(energy_jit(x_flat))

    def jac_np(x_flat):
        g = grad_jit(x_flat)
        g.block_until_ready()
        return np.array(g)
    
    hist = []
    def callback(rk):
        E = energy_np(rk)
        hist.append(E)
        print(f" SLSQP iter {len(hist):3d}, E = {E:.6f}")

    X = (np.eye(n) / n).flatten()

    _ = grad_jit(X)    
    print(f"Energy of Maximally mixed state:    {energy_jit(X):20.8f}")
    
    t0 = time.time()

    with_ad_grad = False     
    if with_ad_grad:
        res = minimize(
            fun    = energy_np,
            x0     = X,
            jac    = jac_np,
            method = "SLSQP",
            callback   = callback,    
            options= {"ftol": 1e-9, "disp": True}
        )
    else:
        res = minimize(
            fun    = energy_np,
            x0     = X,
            method = "SLSQP",
            callback   = callback,    
            options= {"ftol": 1e-9, "eps":  1e-4, "disp": True}
        )



    t1 = time.time()

    print(f"Optimized energy:       {res.fun:20.8f}")
    print(f"SLSQP runtime:      {t1 - t0:.3f} s")

    return None

if __name__ == "__main__":
    test(n=8,seed=2)



# Output - (Callback iterations removed for conciseness)

# Test 1 - (way1-normalization) and (With Jax-AD gradients)
# Shape of H =  (8, 8)
# Exact ground-state energy:             -3.37013268
# Energy of Maximally mixed state:             -0.29194617
# Optimization terminated successfully    (Exit mode 0)
#             Current function value: -3.3701329231262207
#             Iterations: 13
#             Function evaluations: 30
#             Gradient evaluations: 13
# Optimized energy:                -3.37013292
# SLSQP runtime:      0.008 s


# Test 2 - (way1-normalization) and (Without Jax-AD gradients)
# Shape of H =  (8, 8)
# Exact ground-state energy:             -3.37013268
# Energy of Maximally mixed state:             -0.29194617
# Optimization terminated successfully    (Exit mode 0)
#             Current function value: -3.370110034942627
#             Iterations: 39
#             Function evaluations: 2640
#             Gradient evaluations: 39
# Optimized energy:                -3.37011003
# SLSQP runtime:      0.196 s


# Test 3 - (way2-lagrange multiplier) and (With Jax-AD gradients)
# Shape of H =  (8, 8)
# Exact ground-state energy:             -3.37013268
# Energy of Maximally mixed state:             76.52600861
# Optimization terminated successfully    (Exit mode 0)
#             Current function value: -3.398527145385742
#             Iterations: 47
#             Function evaluations: 101
#             Gradient evaluations: 47
# Optimized energy:                -3.39852715
# SLSQP runtime:      0.022 s


# Test 4 - (way2-lagrange multiplier) and (Without Jax-AD gradients)
# Shape of H =  (8, 8)
# Exact ground-state energy:             -3.37013268
# Energy of Maximally mixed state:             76.52600861
# Optimization terminated successfully    (Exit mode 0)
#             Current function value: -3.398508071899414
#             Iterations: 71
#             Function evaluations: 4748
#             Gradient evaluations: 71
# Optimized energy:                -3.39850807
# SLSQP runtime:      0.357 s
