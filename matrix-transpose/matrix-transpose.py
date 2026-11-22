import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """

    A_numpy = np.array(A)

    return A_numpy.T
    
