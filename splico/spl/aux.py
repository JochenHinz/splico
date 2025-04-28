"""
Module providing helper functionality for use in ``kv.py`` and ``spline.py``,
such as prolonging to a refined knotvector.

@author: Jochen Hinz
"""

from ..util import frozen, np
from ._jit_spl import _univariate_prolongation_matrix, _pseudo_inverse
from ..types import Int

from itertools import starmap
from functools import wraps, reduce, lru_cache
from typing import List, TYPE_CHECKING, Any

from scipy import sparse
from numpy.typing import NDArray


if TYPE_CHECKING:
  from .kv import UnivariateKnotVector, TensorKnotVector


def freeze_csr(fn):
  """
  Decorator for freezing a :class:`sparse.csr_matrix` before it is returned
  to avoid accidental overwrites, for instance when returning from cached
  functions.

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
def univariate_prolongation_matrix(kvold: 'UnivariateKnotVector',
                                   kvnew: 'UnivariateKnotVector') -> sparse.csr_matrix:
  """
  Compute the prolongation matrix from a coarser to a finer knotvector.
  Alternatively the restriction matrix from a finer to a coarser knotvector
  is computed. The restriction matrix is a pseudo-inverse of the prolongation
  matrix which employs the entries of the prolongation matrix to compute the
  entries of the restriction matrix.
  Either `kvold <= kvnew` or `kvnew <= kvold` must hold.
  """
  if kvold == kvnew:
    return sparse.eye(kvold.dim, format='csr')

  coarsen = False
  if not kvold <= kvnew:
    kvold, kvnew = kvnew, kvold
    coarsen = True

  assert kvold <= kvnew

  T = _univariate_prolongation_matrix(kvold.repeated_knots,
                                      kvnew.repeated_knots, kvold.degree)
  ret = sparse.csr_matrix(T)
  if not coarsen:
    return ret

  ret = ret.tocoo()
  data = ret.data
  coords = ret.coords

  return sparse.coo_matrix(_pseudo_inverse(data, *coords),
                                    shape=ret.shape[::-1]).tocsr()


@lru_cache(maxsize=8)
@freeze_csr
def tensorial_prolongation_matrix(kvold: 'TensorKnotVector',
                                  kvnew: 'TensorKnotVector') -> sparse.csr_matrix:
  # XXX: implement a variant that never explicitly carries out the kronecker product
  return sparse_kron(sparse.eye(1),
                     *starmap(univariate_prolongation_matrix, zip(kvold, kvnew)))


def denest_objarr(arr: NDArray):
  """
  Recursively flattens a numpy object array.
  Only flattens along entries that are themselves object arrays.

  Parameters
  ----------
  arr : np.ndarray
      A numpy array of dtype object.

  Returns
  -------
  np.ndarray
      A 1D numpy object array containing all non-array objects.
  """

  if not isinstance(arr, np.ndarray) or arr.dtype != object:
    raise TypeError("Input must be a numpy object array.")

  shape: List[Int] = []

  elem = arr
  while isinstance(elem, np.ndarray) and elem.dtype == object:
    shape.extend(elem.shape)
    elem = elem.ravel()[0]

  ret: List[Any] = []

  def recurse(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
      for item in x.flat:
        recurse(item)
    else:
      ret.append(x)

  recurse(arr)

  return np.array(ret, dtype=object).reshape(shape)
