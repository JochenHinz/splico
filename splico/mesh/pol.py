"""
Polynomial routines for evaluating the local element maps.
"""

from ..util import np, _, freeze
from ..types import Int, FloatArray, AnyIntSeq

from typing import Tuple, TYPE_CHECKING
from functools import lru_cache

from numba import njit

sl = slice(_)

if TYPE_CHECKING:
  from .mesh import Mesh

# While numpy provides routines for polynomial evaluation, they are not easily
# generalizable to N dimensions. In the long run, we would like to support
# meshes of higher dimensionality, hence the custom implementation.

# XXX: In the long run, nD polynomial evaluation should be based on Horner's
#      method, see https://en.wikipedia.org/wiki/Horner%27s_method.


@njit(cache=True)
def _derivative_weights(pol_order, dx):
  """
  Accumulated ``dx``-th derivative weights of a ``pol_order``-th polynomial.
  >>> _derivative_weights(5, 0)
      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  (identity)
  >>> _derivative_weights(5, 1)
      [1.0, 2.0, 3.0, 4.0, 5.0]
  >>> _derivative_weights(5, 2)
      [2.0, 6.0, 12.0, 20.0]
  >>> _derivative_weights(5, 3)
      [6.0, 24.0, 60.0]
  >>> _derivative_weights(5, 4)
      [24.0, 120.0]
  >>> _derivative_weights(5, 5)
      [120.0]
  >>> _derivative_weights(5, 6)
      [0.0]                           (null-function)
  """
  assert pol_order >= 0 and dx >= 0
  if dx > pol_order:
    return np.zeros((1,), dtype=np.float64)
  ret = np.ones((pol_order + 1,), dtype=np.float64)
  for i in range(dx):
    ret = (ret * np.arange(len(ret)))[1:]
  return ret


@freeze
def _nd_pol_derivative(weights: FloatArray, dx: AnyIntSeq) -> FloatArray:
  """
  Given a :class:`np.ndarray` of shape (p, q, r, ..., x, y, z, ...)
  respresening `(x, y, z, ...)` n-dependency polynomials of order
  (p, q, r, ...) and a length n array-like `dx` of positive integers
  representing partial derivative orders, return the polynomial weights
  of shape (p-dx[0], q-dx[1], r-dx[2], ..., x, y, z, ...)
  representing the derivatives of the input polynomials.

  Parameters
  ----------
  weights : :class:`np.ndarray`
      The polynomial weights of len(weights) polynomials, all of the same order.
  dx : :class:`np.ndarray` or Sequence of integers
      The derivative orders. Must be (non-strictly) positive. The first
      len(dx) axes of `weights` are interpreted as polynomial weight axes.

  Returns
  -------
  derivative_weights : :class:`np.ndarray`
      The polynomial weights of the derivative.
  """
  dx = np.asarray(dx, dtype=int)
  # get the number of derivative axes
  nder = len(dx)

  assert all(0 <= _dx < n for _dx, n in zip(dx, weights.shape[:nder], strict=True))
  ntot = weights.ndim

  _dweights = lambda shp, _dx: _derivative_weights(shp - 1, _dx)

  # shape (weights.shape[0] - dx[0], weights.shape[1] - dx[1], ...)
  dweights = np.broadcast_arrays(*np.meshgrid(*map(_dweights, weights.shape, dx),
                                              copy=False,
                                              sparse=True,
                                              indexing='ij') )

  # slice out first i = dx[j] entries in the j-th dimension of ``weights``
  # and multiply by the product of `dweights` with appropriate number of
  # one axes appended at the end.
  return weights[tuple(slice(i, _) for i in dx)] * \
         np.multiply.reduce(dweights)[(...,) + (_,) * (ntot - nder)]


@lru_cache(maxsize=8)
@freeze
def _compute_basis_weights(mesh: 'Mesh'):
  """
  The polynomial weights of the nodal basis functions in the reference
  element. Shape: (2,) * self.ndims + (nverts,)
  """

  # PointMesh ords have shape (1,) so we need to reshape to two dimensions
  ords = np.atleast_2d(mesh._local_ordinances(1).astype(int))

  # set up the matrix we need to solve
  X = np.stack([ np.multiply.reduce([_x ** i for _x, i in zip(ords.T, multi_index)])
                 for multi_index in ords ], axis=1)

  # solve for the nodal basis function's polynomial weights
  # and reshape them to tensorial (nfuncs, x, y, z, ...) shape

  # shape: (*(2,) * self.ndims, 2 ** self.ndims)
  # (basis_f_index, 2, 2, ...)
  basis_funcs = np.zeros((*(2,) * mesh.ndims, X.shape[0]), dtype=float)
  basis_funcs[*ords.T] = np.linalg.solve(X, np.eye(X.shape[0]))
  return basis_funcs


@lru_cache(maxsize=8)
def _compute_pol_weights(mesh: 'Mesh', dx: Tuple[Int, ...]) -> FloatArray:
  """
  Polynomial weights of each element's map.
  For `splico.mesh.Mesh.eval_local`.
  """

  assert len(dx) == mesh.ndims
  basis_funcs = _compute_basis_weights(mesh)

  # get the element-wise weights in tensorial layout
  # shape: (2 ** self.ndims, nelems, 3)
  elementwise_weights = mesh.points[mesh.elements.T]

  # (1, ..., 1, 2 ** ndims, nelems, 3) and (2, ..., 2, 2 **ndims, 1, 1 )
  # becomes (2, ..., 2, 2 ** ndims, nelems, 3).sum(-3) == (2, ..., 2, nelems, 3)
  ret = (elementwise_weights[(_,) * mesh.ndims] * basis_funcs[..., _, _]).sum(-3)
  return _nd_pol_derivative(ret, dx)


def eval_nd_polynomial_local(mesh: 'Mesh', points: FloatArray,
                                           dx: Int | AnyIntSeq = 0) -> FloatArray:
  """
  Evaluate `(x, y, z, ...)` n-dependency polynomials or their derivatives
  in `points`.

  Parameters
  ----------
  weights: :class:`np.ndarray`
      Array of shape (p, q, r, ..., x, y, z, ...) representing `(x, y, z, ...)`
      n-dimensional polynomials, all of the same order (p, q, r, ...).
  points : :class:`np.ndarray`
      Array of points. Must have shape (npoints, n), where n is the number
      of polynomial dependencies (x, y, z, ...).
  dx : :class:`np.ndarray[int]` or Sequence[int] or :class:`int` or None
      Derivative orders in each direction. If not array-like, assumed to be
      integer or None. If integer, the value is repeated n times. If None,
      defaults to zeros of appropriate length.

  Returns
  -------
  evaluations : :class:`np.ndarray`
      Polynomial (derivative) evaluations of shape (x, y, z, ..., npoints).
  """

  # in case points is one-dimensional (PointMesh)
  ndim, = (points := np.asarray(points)).shape[1:] or (0,)

  assert mesh.ndims == ndim, "The point array's shape doesn't match the mesh's" \
                             " dimensionality."

  # we have to convert to new variable to avoid mypy errors ...
  if isinstance(dx, Int):
    dx_tpl: Tuple[Int, ...] = (dx,) * ndim
  else:
    dx_tpl = tuple(dx)

  assert len(dx_tpl) == ndim

  # take derivative weights
  weights = _compute_pol_weights(mesh, dx_tpl)
  shape = weights.shape[:ndim]

  # compute all x, y, z, ... points raised to all powers
  # shape: (q, npoints) where q is the number of powers
  powers = [ mypoints[_] ** np.arange(dim)[:, _]
                            for mypoints, dim in zip(points.T, shape) ]

  # all powers are broadcast to this shape
  array_shape = weights.shape + (len(points),)
  m = len(array_shape)

  # broadcast to the following shapes:
  # suppose weight.shape == (p, q, r, ..., x, y, z, ...) and len(points) == npoints.
  # We create arrays of shape (p, 1, 1, ..., npoints), (1, q, 1, ..., npoints)
  # and (1, 1, r, 1, .., npoints) which are all broadcast to shape
  # (p, q, r, ..., x, y, z, ..., npoints)
  myshape = lambda j: (_,) * j + (sl,) + (_,) * (m - j - 2) + (sl,)
  reshaped_arrays = [ np.broadcast_to(mypower[myshape(i)], array_shape)
                                      for i, mypower in enumerate(powers) ]

  # return the result in shape (x, y, z, ..., n)
  return (weights[..., _] * np.multiply.reduce(reshaped_arrays)).sum(tuple(range(ndim)))
