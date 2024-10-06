from ..util import np, _

from typing import Sequence, Optional

sl = slice(_)


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

  for i, (_dx, dim) in enumerate(zip(dx, weights.shape)):
    if _dx == 0: continue
    weights = weights * np.arange(dim).reshape((1,) * i + (dim,) + (1,) * (ntot - i - 1))
    weights = weights[(sl,) * i + (slice(1, _),)]
  return _nd_pol_derivative(weights, tuple(max(_dx - 1, 0) for _dx in dx))


def _eval_nd_polynomial_local(weights: np.ndarray,
                              points: np.ndarray,
                              dx: Optional[Sequence[int] | np.ndarray | int] = None) -> np.ndarray:
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
  weights = _nd_pol_derivative(weights, dx)

  shape = weights.shape[:len(dx)]

  # compute all x, y, z, ... points raised to all powers
  # shape: (mydim, len(points))
  powers = [ mypoints[_] ** np.arange(dim)[:, _] for mypoints, dim in zip(points.T, shape) ]

  # all powers are broadcast to this shape
  array_shape = weights.shape + (len(points),)

  # broadcast to the following shapes:
  # suppose weight.shape == (p, q, r, ..., x, y, z), and len(points) == n. We create
  # arrays of shape (p, 1, 1, ..., 1), (1, q, 1, ..., 1) and (1, 1, r, 1, .., 1) which are
  # all broadcast to shape (p, q, r, ..., x, y, z, n)
  myshape = lambda j: (_,) * j + (sl,) + (_,) * (len(array_shape) - j - 2) + (sl,)
  reshaped_arrays = [ np.broadcast_to(mypower[myshape(i)], array_shape)
                      for i, mypower in enumerate(powers) ]

  # return the result in shape (x, y, z, n)
  return (weights[..., _] * np.multiply.reduce(reshaped_arrays)).sum(tuple(range(ndim)))
