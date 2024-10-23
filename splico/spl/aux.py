"""
Module providing helper functionality for use in ``kv.py`` and ``spline.py``,
such as prolonging to a refined knotvector.
"""

from ..util import frozen, np
from ._jit_spl import _univariate_prolongation_matrix

from itertools import starmap
from functools import wraps, reduce, lru_cache
from typing import List

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
