from ..util import np, _

from typing import Sequence
from functools import lru_cache

sl = slice(_)


"""
  Polynomial routines for evaluating the local element maps.
"""

# While numpy provides routines for polynomial evaluation, they are not easily
# generalizable to N dimensions. In the long run, we would like to support
# meshes of higher dimensionality, hence the custom implementation.


def _nd_pol_derivative(weights: np.ndarray, dx: Sequence[int] | np.ndarray) -> np.ndarray:
  """
    Given a :class:`np.ndarray` of shape (p, q, r, ..., x, y, z) respresening
    `(x, y, z)` n-dependency polynomials of order (p, q, r, ...) and a length n
    array-like `dx` of positive integers representing partial derivative orders,
    return the polynomial weights of shape (p-dx[0], q-dx[1], r-dx[2], ..., x, y, z)
    representing the derivatives of the input polynomials.

    Parameters
    ----------
    weights : :class:`np.ndarray`
        The polynomial weights of len(weights) polynomials, all of the same order.
    dx : :class:`np.ndarray` or Sequence of integers.
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

  assert all( 0 <= _dx < n for _dx, n in zip(dx, weights.shape[:nder], strict=True) )

  if (dx == 0).all():
    return weights

  ntot = weights.ndim

  derivative_weights = np.meshgrid(*map(np.arange, weights.shape[:nder]),
                                   copy=False,
                                   sparse=True,
                                   indexing='ij')

  for i, (_dx, myweights) in enumerate(zip(dx, derivative_weights)):
    if _dx == 0: continue
    weights = weights * myweights[(...,) + (_,)*(ntot - nder)]
    weights = weights[(sl,) * i + (slice(1, _),)]

  return _nd_pol_derivative(weights, tuple(max(_dx - 1, 0) for _dx in dx))


@lru_cache(maxsize=32)
def _compute_basis_weights(mesh):
  """
    The polynomial weights of the nodal basis functions in the reference
    element.
    Shape: (2,) * self.ndims + (nverts,)
  """
  ords = mesh._local_ordinances(1).astype(int)
  # set up the matrix we need to solve
  X = np.stack([ np.multiply.reduce([_x ** i for _x, i in zip(ords.T, multi_index)])
                 for multi_index in ords ], axis=1)

  # solve for the nodal basis function's polynomial weights
  # and reshape them to tensorial (nfuncs, x, y, z, ...) shape

  # shape: (2 ** self.ndims, *(2,) * self.ndims)
  # (basis_f_index, 2, 2, ...)
  basis_funcs = np.zeros((*(2,) * mesh.ndims, X.shape[0]), dtype=float)
  basis_funcs[*ords.T] = np.linalg.solve(X, np.eye(X.shape[0]))
  return basis_funcs


@lru_cache(maxsize=32)
def _compute_pol_weights(mesh, dx) -> np.ndarray:
  """
    Polynomial weights of each element's map.
    For `mesh.eval_local`.
  """
  # XXX: caching doesn't re-use for instance the result of dx = (1, 0, 0) if
  #      dx = (2, 0, 0) is computed. Change the caching structure to improve this.

  assert len(dx) == mesh.ndims
  basis_funcs = _compute_basis_weights(mesh)

  # get the element-wise weights in tensorial layout
  # shape: (2 ** self.ndims, nelems, 3)
  elementwise_weights = mesh.points[mesh.elements.T]

  # (1, ..., 1, 2 ** ndims, nelems, 3) and (2, ..., 2, 2 **ndims, 1, 1 )
  # becomes (2, ..., 2, 2 ** ndims, nelems, 3).sum(-3) == (2, ..., 2, nelems, 3)
  ret = (elementwise_weights[(_,) * mesh.ndims] * basis_funcs[..., _, _]).sum(-3)
  return _nd_pol_derivative(ret, dx)


def eval_nd_polynomial_local(mesh, points, dx=None) -> np.ndarray:
  """
    Evaluate `(x, y, z)` n-dependency polynomials or their derivatives in `points`.

    Parameters
    ----------
    weights: :class:`np.ndarray`
        Array of shape (p, q, r, ..., x, y, z) representing `(x, y, z)` n-dimensional
        polynomials, all of the same order (p, q, r, ...).
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
        Polynomial (derivative) evaluations of shape (x, y, z, npoints).
  """
  ndim = points.shape[1]
  assert points.shape[1:] == (ndim,)

  if dx is None:
    dx = 0
  if np.isscalar(dx):
    dx = (dx,) * ndim

  assert len(dx) == ndim

  # take derivative weights
  weights = _compute_pol_weights(mesh, tuple(dx))

  shape = weights.shape[:len(dx)]

  # compute all x, y, z, ... points raised to all powers
  # shape: (mydim, len(points))
  powers = [ mypoints[_] ** np.arange(dim)[:, _]
                            for mypoints, dim in zip(points.T, shape) ]

  # all powers are broadcast to this shape
  array_shape = weights.shape + (len(points),)

  # broadcast to the following shapes:
  # suppose weight.shape == (p, q, r, ..., x, y, z), and len(points) == n.
  # We create arrays of shape (p, 1, 1, ..., 1), (1, q, 1, ..., 1) and
  # (1, 1, r, 1, .., 1) which are all broadcast to shape (p, q, r, ..., x, y, z, n)
  myshape = lambda j: (_,) * j + (sl,) + (_,) * (len(array_shape) - j - 2) + (sl,)
  reshaped_arrays = [ np.broadcast_to(mypower[myshape(i)], array_shape)
                      for i, mypower in enumerate(powers) ]

  # return the result in shape (x, y, z, n)
  return (weights[..., _] * np.multiply.reduce(reshaped_arrays)).sum(tuple(range(ndim)))
