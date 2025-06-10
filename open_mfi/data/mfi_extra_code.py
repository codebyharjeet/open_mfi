


def embed_pair_with_rest(lam_ij, i, j, singles):
    """
    Parameters
    ----------
    lam_ij  :  4x4 numpy array, two-qubit operator
    i, j    :  qubit indices (0-based, arbitrary order)
    singles :  list [rho0,rho1,…] of 2x2 single-qubit density matrices

    Returns
    -------
    full_op :  (2^N x 2^N) operator   I⊗…⊗λ_ij⊗…⊗I
    """
    if i == j:
        raise ValueError("i and j must be different qubits")
    if i > j:                          # ensure i < j and swap qubit order inside λ
        i,  j  = j,  i
        lam_ij = lam_ij.reshape(2,2,2,2).transpose(1,0,3,2).reshape(4,4)
        
    blocks = []
    for q in range(len(singles)):
        if q == i:
            blocks.append(lam_ij)
        elif q == j:
            continue
        else:
            blocks.append(singles[q])

    full_op = reduce(np.kron, blocks)
    assert np.allclose(np.trace(full_op), 0.0)
    return full_op


    