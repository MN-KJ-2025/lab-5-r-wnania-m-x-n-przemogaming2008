# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np


def spare_matrix_Abt(m: int, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(m, int) or not isinstance(n, int) or m <= 0 or n <= 0:
        return None
    t = np.linspace(0, 1, m)
    b = np.cos(4 * t)              # (m,)
    A = np.vander(t, n, increasing=True)  # (m,n)
    return A, b

def square_from_rectan(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray) or A.ndim != 2:
        return None
    m, n = A.shape
    b = b.ravel()
    if b.shape[0] != m:
        return None
    A_new = A.T @ A                # (n,n)
    b_new = A.T @ b                # (n,)
    return A_new, b_new

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    if A.ndim != 2 or x.ndim != 1 or b.ndim != 1:
        return None
    m, n = A.shape
    if x.shape[0] != n or b.shape[0] != m:
        return None
    r = A @ x - b
    return float(np.linalg.norm(r))
