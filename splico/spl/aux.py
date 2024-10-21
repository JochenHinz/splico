"""
Module providing helper functionality for use in ``kv.py`` and ``spline.py``,
such as prolonging to a refined knotvector.
"""

from ..util import frozen, np

from itertools import starmap
from functools import wraps, reduce, lru_cache
from typing import List

from numba import njit
from scipy import sparse


def freeze_csr(fn):
  """
  Decorator for freezing a :class:`sparse.csr_matrix` before it is returned
  to avoid accidental overwrites, for instance when returning from cached functions.

  If a mutable instance of the matrix is desired, the matrix has to be copied.
  """
  @wraps(fn)
  def wrapper(*args, **kwargs):
    ret = fn(*args, **kwargs)
    assert isinstance(ret, sparse.csr_matrix)
    for item in map(lambda x: getattr(ret, x), ('data', 'indices', 'indptr')):
      frozen(item)  # convert the array to read-only to avoid accidental overwrites
    return ret
  return wrapper


def sparse_kron(*_mats: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
  """
  Take the Kronecker product of several sparse matrices.
  """
  assert _mats
  mats: List[sparse.spmatrix] = list(map(sparse.csr_matrix, _mats))
  if len(mats) == 1:
    return mats[0]
  return reduce(lambda x, y: sparse.kron(x, y, format='csr'), mats)


@njit(cache=True)
def _univariate_prolongation_matrix(kvold, kvnew, p):
  """
  NURBS-Book implementation.
  """
  # XXX: implement for immediate sparse matrix output

  n, m = len(kvnew) - 1, len(kvold) - 1
  T = np.zeros((n, m), dtype=np.float64)

  for j in range(m):
    T[np.where(np.logical_and(kvold[j] <= kvnew[:n],
                              kvnew[:n] < kvold[j + 1]))[0], j] = 1

  for q in range(1, p + 1):
    T_new = np.zeros((n - q, m - q), dtype=np.float64)
    for i in range(n - q):
      for j in np.where(np.logical_or(T[i, : m - q] != 0,
                                      T[i, 1: m - q + 1] != 0))[0]:
        denom0 = kvold[j+q] - kvold[j]
        fac0 = (kvnew[i + q] - kvold[j]) / denom0 if denom0 != 0 else 0

        denom1 = kvold[j+q+1] - kvold[j+1]
        fac1 = (kvold[j+1+q] - kvnew[i+q]) / denom1 if denom1 != 0 else 0

        T_new[i, j] = fac0 * T[i, j] + fac1 * T[i, j + 1]

    T = T_new

  return T


@lru_cache(maxsize=8)
@freeze_csr
def univariate_prolongation_matrix(kvold,
                                   kvnew) -> sparse.csr_matrix:
  # XXX: support kvnew < kvold via Moore-penrose pseudo inverse
  assert kvold <= kvnew
  if kvold == kvnew:
    return sparse.eye(kvold.dim, format='csr')
  T = _univariate_prolongation_matrix(kvold.repeated_knots,
                                      kvnew.repeated_knots, kvold.degree)
  return sparse.csr_matrix(T)


@lru_cache(maxsize=8)
@freeze_csr
def tensorial_prolongation_matrix(kvold,
                                  kvnew) -> sparse.csr_matrix:
  # XXX: idem, kvnew < kvold should be managed using the pseudo inverse.
  #      I think in this case the matrix should be conditionally converted
  #      to sparse format.
  assert kvold <= kvnew
  # XXX: implement a variant that never explicitly carries out the kronecker product
  return sparse_kron(*starmap(univariate_prolongation_matrix, zip(kvold, kvnew)))
