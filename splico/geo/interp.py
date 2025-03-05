"""
Module for spline interpolation. Either linear or higher order interpolation.

@author: Jochen Hinz
"""

from splico.spl import NDSpline, NDSplineArray, UnivariateKnotVector, as_NDSplineArray
from ..util import np, _

from typing import Callable
from functools import wraps


Spline = NDSpline | NDSplineArray


def spline_or_array(f: Callable) -> Callable:
  """
  Decorator for functions that can either take a single :class:`splico.spl.NDSpline`
  or a :class:`splico.spl.NDSplineArray` as input. The function is then called
  with the input converted to a :class:`splico.spl.NDSplineArray` and the output
  is converted back to the input type.
  For the output to be converted to :class:`splico.spl.NDSpline`, both inputs
  must be of this type.
  """

  @wraps(f)
  def wrapper(spl0: Spline, spl1: Spline, *args, **kwargs):
    ret = f(as_NDSplineArray(spl0), as_NDSplineArray(spl1), *args, **kwargs)
    if isinstance(spl0, NDSpline) and isinstance(spl1, NDSpline):
      assert ret._shape == ()
      return ret.arr.ravel()[0]
    return ret

  return wrapper


@spline_or_array
def linear_interpolation(spl0: NDSplineArray,
                         spl1: NDSplineArray, zdegree: int = 1) -> NDSplineArray:
  """
  Perform a linear interpolation between two splines.
  This introduces a new parametric dependency which is appended at the end,
  i.e., spl(x0, x1, ..., xn) -> spl(x0, x1, ..., xn, xn+1).

  Parameters
  ----------
  spl0 : :class:`splico.spl.NDSpline` or :class:`splico.spl.NDSplineArray`
      The first input spline.
  spl1 : :class:`splico.spl.NDSpline` or :class:`splico.spl.NDSplineArray`
      The second input spline.
  zdegree : :class:`int`
      The spline basis order in the xn+1-direction. Must be >= 3.

  Returns
  -------
  spl : :class:`splico.spl.NDSpline`
      The resulting linearly interpolated spline satisfying
      `spl.nvars == spl0.nvars + 1`.
  """
  assert (zdegree := int(zdegree)) >= 1
  assert spl0.shape == spl1.shape
  assert (spl0.contract_all().knotvector == spl1.contract_all().knotvector).all()

  # make an at least linear knotvector without interior knots
  kvz = UnivariateKnotVector([0, 1], zdegree).to_tensor()

  # get the greville points
  z, = kvz.greville

  # we compute the hermite interpolation functions by fitting againts the
  # stable greville points. We need not stabilize and the fit should be exact.
  f0, f1 = kvz.fit([z], np.stack([1 - z, z], axis=1), lam0=0)

  return spl0 * f0 + spl1 * f1


@spline_or_array
def cubic_hermite_interpolation(spl0: NDSplineArray,
                                spl1: NDSplineArray,
                                t0: np.ndarray,
                                t1: np.ndarray,
                                zdegree: int = 3) -> NDSplineArray:
  """
  Perform a cubic hermite interpolation between two splines.
  This introduces a new parametric dependency which is appended at the end,
  i.e., spl(x0, x1, ..., xn) -> spl(x0, x1, ..., xn, xn+1).

  Parameters
  ----------
  spl0 : :class:`splico.spl.NDSpline` or :class:`splico.spl.NDSplineArray`
      The first input spline.
  spl1 : :class:`splico.spl.NDSpline` or :class:`splico.spl.NDSplineArray`
      The second input spline.
  t0 : :class:`np.ndarray`
      The desired tangent at xn+1 == 0.
  t1 : :class:`np.ndarray`
      The desired tangent at xn+1 == 1.
  zdegree : :class:`int`
      The spline basis order in the xn+1-direction. Must be >= 3.

  Returns
  -------
  spl : :class:`splico.spl.NDSpline`
      The resulting cubic Hermite interpolation spline satisfying
      `spl.nvars == spl0.nvars + 1`.
  """
  assert spl0.shape == spl1.shape
  assert (zdegree := int(zdegree)) >= 3
  if (t0 := np.asarray(t0)).shape != (3,) or (t1 := np.asarray(t1)).shape != (3,):
    raise NotImplementedError
  assert (spl0.contract_all().knotvector == spl1.contract_all().knotvector).all()

  # make an at least cubic knotvector without interior knots
  kvz = UnivariateKnotVector([0, 1], zdegree).to_tensor()
  z, = kvz.greville

  # as in `linear_interpolation` but this time we fit the cubic Hermite functions
  f0, f1, f2, f3 = kvz.fit([z], np.stack([2 * z**3 - 3 * z**2 + 1,
                                          z**3 - 2 * z**2 + z,
                                          -2 * z**3 + 3 * z**2,
                                          z**3 - z**2], axis=1), lam0=0)

  one = spl0.one(spl0.knotvector)

  # Interpolation along the last axis. One axes are appended to `t_i` automatically.
  return spl0 * f0 + one * t0 * f1 + spl1 * f2 + one * t1 * f3


# XXX: for Fabio - implement Hermite with nonconstant tangent vectors.
