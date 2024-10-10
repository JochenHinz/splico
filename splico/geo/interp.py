from splico.spl import NDSpline, UnivariateKnotVector, TensorKnotVector
from ..util import np, _


def linear_interpolation(spl0: NDSpline, spl1: NDSpline, zdegree: int = 1) -> NDSpline:
  """
    Perform a linear interpolation between two `splico.spl.NDSpline`s.
    This introduces a new parametric dependency which is appended at the end,
    i.e., spl(x0, x1, ..., xn) -> spl(x0, x1, ..., xn, xn+1).

    Parameters
    ----------
    spl0 : :class:`splico.spl.NDSpline`
        The first input spline.
    spl1 : :class:`splico.spl.NDSpline`
        The second input spline.
    zdegree : :class:`int`
        The spline basis order in the xn+1-direction. Must be >= 3.

    Returns
    -------
    spl : :class:`splico.spl.NDSpline`
        The resulting linearly interpolated spline satisfying
        `spl.nvars == spl0.nvars + 1`.
  """
  assert spl0.knotvector == spl1.knotvector and spl0.shape == spl1.shape
  assert (zdegree := int(zdegree)) >= 1

  # make an at least linear knotvector without interior knots
  kvz = TensorKnotVector([UnivariateKnotVector([0, 1], zdegree)])

  # get the greville points
  z, = kvz.greville

  # we compute the hermite interpolation functions by fitting againts the
  # stable greville points. We need not stabilize and the fit should be exact.
  f0, f1 = kvz.fit([z], np.stack([1 - z, z], axis=1), lam0=0)

  return spl0 * f0 + spl1 * f1


def cubic_hermite_interpolation(spl0: NDSpline,
                                spl1: NDSpline,
                                t0: np.ndarray,
                                t1: np.ndarray,
                                zdegree: int = 3) -> NDSpline:
  """
    Perform a cubic hermite interpolation between two `splico.spl.NDSpline`s.
    This introduces a new parametric dependency which is appended at the end,
    i.e., spl(x0, x1, ..., xn) -> spl(x0, x1, ..., xn, xn+1).

    Parameters
    ----------
    spl0 : :class:`splico.spl.NDSpline`
        The first input spline.
    spl1 : :class:`splico.spl.NDSpline`
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
  assert spl0.knotvector == spl1.knotvector and spl0.shape == spl1.shape
  assert (t0 := np.asarray(t0)).shape == (t1 := np.asarray(t1)).shape == (3,), \
      NotImplementedError
  assert (zdegree := int(zdegree)) >= 3

  # make an at least cubic knotvector without interior knots
  kvz = TensorKnotVector([UnivariateKnotVector([0, 1], zdegree)])
  z, = kvz.greville

  # as in `linear_interpolation`
  f0, f1, f2, f3 = kvz.fit([z], np.stack([2 * z**3 - 3 * z**2 + 1,
                                          z**3 - 2 * z**2 + z,
                                          -2 * z**3 + 3 * z**2,
                                          z**3 - z**2], axis=1), lam0=0)

  one = spl0.one(spl0.knotvector)

  return spl0 * f0 + ((one * t0[_]) * f1) + spl1 * f2 + ((one * t1[_]) * f3)
