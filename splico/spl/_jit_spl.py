#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..util import np, flat_meshgrid
from .._jit import arange_product, ravel_multi_index

from numba import njit, float64, int64, prange, config
import multiprocessing

config.NUMBA_NUM_THREADS = multiprocessing.cpu_count()


"""
  Various jit-compiled routines for spline evaluation.
"""


# XXX: this scipt requires a formatting overhaul and some docstrings need
#      need to be adjusted to the default format.
#      Some functions' positional arguments are in a confusing order.
#      Lastly, we need to write one function for N-D evaluation of splines
#      rather than dedicated routines for 1, 2 and 3D.
#      This can be accomplished by using the product routines from splico._jit.


@njit(cache=True)
def position_in_knotvector(t, x):
  # XXX: replace by binary search algorithm.
  """
  Return the position of ``x`` in the knotvector ``t``.
  If x equals t[-1], return the position before the first
  occurence of x in t.

  Parameters
  ----------
  t : :class:`np.ndarray`
      The knotvector with repeated knots.
  x : :class:`np.ndarray`
      The vector of positions.

  Returns
  -------
  ret : :class:`np.ndarray` comprised of integers
      The positions in the knotvector. Has the same length as `x`.
      If entry is not found, defaults to -1.
  """

  xlen = len(x)
  ret = np.empty(xlen, dtype=int64)

  for i in prange(xlen):
    for j in range(len(t) - 1):

      # if x equals the last knot, return this
      if x[i] == t[-1]:
        ret[i] = np.where(t == x[i])[0][0] - 1
        break

      if t[j] <= x[i] < t[j+1]:
        ret[i] = j
        break
    else:  # no break
        ret[i] = -1

  return ret


@njit(cache=True)
def nonzero_bsplines(mu, x, t, d):
  """
    Return the value of the d+1 nonzero basis
    functions at position ``x``.

    Parameters
    ----------
    mu : :class:`int`
        The position in `t` that contains `x`,
    x: :class:`float`
        The position,
    t: :class:`np.ndarray`
        The knotvector.
    d: :class:`int`
        The degree of the B-spline basis.

    Returns
    -------
    b : :class:`np.ndarray`
        The nonzero bsplines evalated in `x`
  """

  b = np.zeros(d + 1, dtype=float64)
  b[-1] = 1

  if x == t[-1]:
    return b

  for r in range(1, d + 1):

    k = mu - r + 1
    w2 = (t[k + r] - x) / (t[k + r] - t[k])
    b[d - r] = w2 * b[d - r + 1]

    for i in range(d - r + 1, d):
      k = k + 1
      w1 = w2
      w2 = (t[k + r] - x) / (t[k + r] - t[k])
      b[i] = (1 - w1) * b[i] + w2 * b[i + 1]

    b[d] = (1 - w2) * b[d]

  return b


@njit(cache=True)
def nonzero_bsplines_deriv(kv, p, x, dx):
  """
    Return the value of the d+1 nonzero basis
    functions and their derivatives up to order `dx` at position `x`.

    Parameters
    ----------
    x: position
    kv: knotvector
    p: degree of B-spline basis
    dx: max order of the derivative
  """
  # Initialize variables
  span = position_in_knotvector(kv, np.array([x], dtype=np.float64))[0]
  left = np.ones((p + 1,), dtype=np.float64)
  right = np.ones((p + 1,), dtype=np.float64)
  ndu = np.ones((p + 1, p + 1), dtype=np.float64)

  for j in range(1, p + 1):
    left[j] = x - kv[span + 1 - j]
    right[j] = kv[span + j] - x
    saved = 0.0
    r = 0
    for r in range(r, j):
      # Lower triangle
      ndu[j, r] = right[r + 1] + left[j - r]
      temp = ndu[r, j - 1] / ndu[j, r]
      # Upper triangle
      ndu[r, j] = saved + (right[r + 1] * temp)
      saved = left[j - r] * temp
    ndu[j, j] = saved

  # Load the basis functions
  # ders = [[0.0 for _ in range(p + 1)] for _ in range((min(p, dx) + 1))]
  ders = np.zeros((min(p, dx) + 1, p + 1), dtype=np.float64)
  for j in range(0, p + 1):
    ders[0, j] = ndu[j, p]

  # Start calculating derivatives
  # a = [[1.0 for _ in range(p + 1)] for _ in range(2)]
  a = np.ones((2, p + 1), dtype=np.float64)
  # Loop over function index
  for r in range(0, p + 1):
    # Alternate rows in array a
    s1 = 0
    s2 = 1
    a[0, 0] = 1.0
    # Loop to compute k-th derivative
    for k in range(1, dx + 1):
      d = 0.0
      rk = r - k
      pk = p - k
      if r >= k:
        a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
        d = a[s2, 0] * ndu[rk, pk]
      if rk >= -1:
        j1 = 1
      else:
        j1 = -rk
      if (r - 1) <= pk:
        j2 = k - 1
      else:
        j2 = p - r
      for j in range(j1, j2 + 1):
        a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
        d += (a[s2, j] * ndu[rk + j, pk])
      if r <= pk:
        a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
        d += (a[s2, k] * ndu[r, pk])
      ders[k, r] = d

      # Switch rows
      j = s1
      s1 = s2
      s2 = j

  # Multiply through by the the correct factors
  r = float(p)
  for k in range(1, dx + 1):
    for j in range(0, p + 1):
      ders[k, j] *= r
    r *= (p - k)

  # Return the basis function derivatives list
  return ders


@njit(cache=True)
def nonzero_bsplines_deriv_vectorized(kv, p, x, dx):
  """
    Vectorized (in x) version of `nonzero_bsplines_deriv`
    only returns the dx-th derivative though.
  """
  ret = np.empty((len(x), p+1), dtype=np.float64)
  for i in prange(len(ret)):
    ret[i] = nonzero_bsplines_deriv(kv, p, x[i], dx)[dx]
  return ret


@njit(cache=True)
def der_ith_basis_fun( kv, p, i, x, dx ):  # based on algorithm A2.5 from the NURBS-book

  """
    Return the N_ip(x) and its derivatives up to ``dx``,
    where N denotes the ``i``-th basis function of order
    ``p`` resulting from knotvector ``kv`` and x the position.

    Parameters
    ----------
    kv: knotvector
    p: degree of the basis
    i: index of the basis function
    x: position
    dx: highest-order derivative
  """

  if x == kv[-1]:
    x -= 1e-15

  basis_len = len(kv) - p - 1

  if x < kv[i] or x >= kv[i + p + 1]:
    if i != basis_len - 1 or x > kv[-1]:
      ''' x lies outside of the support of basis function or domain '''
      return np.zeros( (dx + 1, ), dtype=float64 )
    if i == basis_len - 1 and x == kv[-1]:
      '''
      special case: evaluation of the last basis function
      in the last point of the interval. Return a sequence
      (p / a_0) ** 0, (p / a_1) ** 1, ... (p / a_dx) ** dx
      '''
      # a = 1
      # ret = np.empty( (dx + 1, ), dtype=float64 )
      # for i in range( ret.shape[0] ):
      #     ret[i] = a
      #     if i != ret.shape[0] - 1:
      #         a *= p / ( kv[basis_len - 1 + p - i] - kv[basis_len - 1 - i] )
      # return ret
      x -= 1e-15

  ders = np.empty( (dx + 1, ), dtype=float64 )
  N = np.zeros( (p + 1, p + 1), dtype=float64 )

  for j in range(p + 1):
    if ( x >= kv[i + j] and x < kv[i + j + 1] ):
      N[j, 0] = 1.0
    else:
      N[j, 0] = 0.0

  for k in range(1, p + 1):
    saved = 0.0 if N[0, k - 1] == 0.0 else \
        (x - kv[i]) * N[0, k - 1] / (kv[i + k] - kv[i])
    for j in range(p - k + 1):
      Uleft, Uright = kv[i + j + 1], kv[i + j + k + 1]
      if N[j + 1, k - 1] == 0:
        N[j, k], saved = saved, 0
      else:
        temp = N[j + 1, k - 1] / (Uright - Uleft)
        N[j, k] = saved + (Uright - x) * temp
        saved = (x - Uleft) * temp

  ders[0] = N[0, p]
  ND = np.zeros( (k + 1, ), dtype=float64 )
  for k in range(1, dx + 1):
    for j in range(k + 1):
      ND[j] = N[j, p - k]
    for jj in range(1, k + 1):
      saved = 0.0 if ND[0] == 0.0 else ND[0] / (kv[i + p - k + jj] - kv[i])
      for j in range(k - jj + 1):
        # wrong in the NURBS book, -k is missing in Uright
        Uleft, Uright = kv[i + j + 1], kv[i + j + p - k + jj + 1]
        if ND[j + 1] == 0.0:
          ND[j], saved = (p - k + jj) * saved, 0.0
        else:
          temp = ND[j + 1] / (Uright - Uleft)
          ND[j] = (p - k + jj) * (saved - temp)
          saved = temp

    ders[k] = ND[0]
  return ders


@njit(cache=True, parallel=True)
def _call1D(xi, kv0, p0, x, dx):
  """
    Return function evaluations at positions xi.

    Parameters
    ----------
    xi: Vector of xi-values
    kv0: knotvector in xi-direction
    p0: degree in xi-direction
    x: vector of control points
    dx: derivative order
  """
  ret = np.zeros(xi.shape, dtype=float64)
  assert ret.ndim == 1
  element_indices0 = position_in_knotvector(kv0, xi)

  for i in prange(len(xi)):
    xi_calls = nonzero_bsplines_deriv(kv0, p0, xi[i], dx)[dx]
    for j in range(p0 + 1):
      a = xi_calls[j]
      global_index = element_indices0[i] - p0 + j
      ret[i] += x[global_index] * a

  return ret


@njit(cache=True, parallel=True)
def _callND(Xi, list_of_knotvectors, degrees, x, derivatives):
  assert Xi.shape[1:] == (len(list_of_knotvectors),)

  # make len(list_of_knotvectors) - shaped integer array with the ndofs per direction
  dims = np.empty(len(list_of_knotvectors), dtype=np.int64)
  for i, (kv, degree) in enumerate(zip(list_of_knotvectors, degrees)):
    dims[i] = kv.shape[0] - degree - 1

  ret = np.zeros(Xi.shape[0], dtype=np.float64)

  # make an outer product flat meshgrid with aranges from 0 to p + 1
  inner_loop_indices = arange_product(degrees + 1)

  # make integer array containing the positions in the knotvectors of the univariate
  # contributions in `Xi`
  element_indices = np.empty((len(Xi), len(list_of_knotvectors)), dtype=np.int64)
  for i, (mykv, xi) in enumerate(zip(list_of_knotvectors, Xi.T)):
    element_indices[:, i] = position_in_knotvector(mykv, xi)

  for i in prange(len(Xi)):

    # get all univariate local calls
    mycalls = [nonzero_bsplines_deriv(kv, p, xi, dx)[dx, :] for kv, p, xi, dx
               in zip(list_of_knotvectors, degrees, Xi[i], derivatives)]

    for multi_index in inner_loop_indices:
      # global index in x results from the multi_index + the element_indices minus the degrees
      # and the `dims` vector
      global_index = ravel_multi_index(element_indices[i] + multi_index - degrees, dims)

      # add product of all evaluations times the weight to the corresponding
      # position
      myval = x[global_index]
      for j, myindex in enumerate(multi_index):
        myval = myval * mycalls[j][myindex]

      ret[i] += myval

  return ret


def call(list_of_abscissae,
         list_of_knotvectors,
         list_of_degrees,
         controlpoints, dx=None):
  # XXX: docstring
  nvars = len(list_of_abscissae)
  assert 1 <= nvars <= 3
  if dx is None:
    dx = 0
  if np.isscalar(dx):
    dx = (dx,) * nvars

  degrees = np.asarray(list_of_degrees, dtype=int)
  dx = np.asarray(dx, dtype=int)
  Xi = np.stack(list_of_abscissae, axis=1)

  assert len(list_of_abscissae) == len(list_of_knotvectors) == len(list_of_degrees)

  return _callND(Xi,
                 list_of_knotvectors,
                 degrees,
                 controlpoints, dx)


@njit(cache=True, parallel=True)
def _tensor_call2D( xi, eta, kv0, kv1, p0, p1, x, dx, dy ):
  """
    Return function evaluations at all positions
    (xi_i, eta_i) in the outer product of univariate
    positions ``xi`` and ``eta``.
    Optimized because the bases are only evaluated in
    ``xi`` and ``eta`` once.
    Returns a matrix instead of a flat vector.

    Parameters
    ----------
    xi: Vector of xi-values
    eta: Vector of eta-values
    kv0: knotvector in xi-direction
    kv1: knotvector in eta-direction
    p0: degree in xi-direction
    p1: degree in eta-direction
    x: vector of control points
    dx: order of the derivative in the xi-direction
    dy: order of the derivative ins the eta-direction
  """

  mu0 = position_in_knotvector(kv0, xi)
  mu1 = position_in_knotvector(kv1, eta)

  n_eta = kv1.shape[0] - p1 - 1

  s0, s1 = p0 + 1, p1 + 1

  xi_evals = np.empty( s0 * xi.shape[0], dtype=np.float64 )
  eta_evals = np.empty( s1 * eta.shape[0], dtype=np.float64 )

  # XXX: for some reason, for p0 == dx, nonzero_bsplines_deriv gives the wrong outcome
  # for xi contained in the last knot span.
  for i in prange(len(xi)):
    xi_evals[i * s0: (i + 1) * s0] = nonzero_bsplines_deriv(kv0, p0, xi[i], dx)[dx, :]

  for i in prange(len(eta)):
    eta_evals[i * s1: (i + 1) * s1] = nonzero_bsplines_deriv(kv1, p1, eta[i], dy)[dy, :]

  ret = np.zeros(( len(xi), len(eta) ), dtype=np.float64)

  for i in prange(len(xi)):
    for j in prange(len(eta)):

      local_xi = xi_evals[i * s0: (i + 1) * s0]
      local_eta = eta_evals[j * s1: (j + 1) * s1]

      for k in range(s0):
        a = local_xi[k]
        for L in range(s1):
          b = local_eta[L]
          global_index = (mu0[i] - p0 + k) * n_eta + \
                          mu1[j] - p1 + \
                          L
          ret[i, j] += x[global_index] * a * b

  return ret


@njit(cache=True, parallel=True)
def _tensor_call3D( xi, eta, zeta, kv0, kv1, kv2, p0, p1, p2, x, dx, dy, dz ):

  mu0 = position_in_knotvector(kv0, xi)
  mu1 = position_in_knotvector(kv1, eta)
  mu2 = position_in_knotvector(kv2, zeta)

  n_eta = kv1.shape[0] - p1 - 1
  n_zeta = kv2.shape[0] - p2 - 1

  s0, s1, s2 = p0 + 1, p1 + 1, p2 + 1

  xi_evals = np.empty( s0 * xi.shape[0], dtype=np.float64 )
  eta_evals = np.empty( s1 * eta.shape[0], dtype=np.float64 )
  zeta_evals = np.empty( s2 * zeta.shape[0], dtype=np.float64 )

  for i in prange(len(xi)):
    xi_evals[i * s0: (i + 1) * s0] = nonzero_bsplines_deriv(kv0, p0, xi[i], dx)[dx, :]
  for i in prange(len(eta)):
    eta_evals[i * s1: (i + 1) * s1] = nonzero_bsplines_deriv(kv1, p1, eta[i], dy)[dy, :]
  for i in prange(len(zeta)):
    zeta_evals[i * s2: (i + 1) * s2] = nonzero_bsplines_deriv(kv2, p2, zeta[i], dz)[dz, :]

  ret = np.zeros((len(xi), len(eta), len(zeta)), dtype=np.float64)

  for i in prange(len(xi)):
    for j in prange(len(eta)):
      for k in prange(len(zeta)):

        local_xi = xi_evals[i * s0: (i + 1) * s0]
        local_eta = eta_evals[j * s1: (j + 1) * s1]
        local_zeta = zeta_evals[k * s2: (k + 1) * s2]

        for L in range(s0):
          a = local_xi[L]
          for m in range(s1):
            b = local_eta[m]
            for n in range(s2):
              c = local_zeta[n]

              global_index = (mu0[i] - p0 + L) * n_zeta * n_eta + \
                             (mu1[j] - p1 + m) * n_zeta + \
                             (mu2[k] - p2 + n)
              ret[i, j, k] += x[global_index] * a * b * c

  return ret


# XXX: implement _tensor_callND


def tensor_call(list_of_abscissae,
                list_of_knotvectors,
                list_of_degrees,
                controlpoints, dx=None):
  # XXX: docstring
  nvars = len(list_of_abscissae)
  assert 1 <= nvars <= 3
  if dx is None:
    dx = 0
  if np.isscalar(dx):
    dx = (dx,) * nvars

  assert len(list_of_abscissae) == len(list_of_knotvectors) \
                                == len(list_of_degrees) \
                                == len(dx)

  func = {1: _call1D,
          2: _tensor_call2D,
          3: _tensor_call3D}[nvars]

  return func(*list_of_abscissae,
              *list_of_knotvectors,
              *list_of_degrees,
              controlpoints, *dx).ravel()


def evaluate_multipatch(npoints, knotvector, degree, list_of_controlpoints, dx=0, dy=0):
  # XXX: docstring
  shape = list_of_controlpoints[0].shape
  assert shape[1:] == (2,)
  assert all( cp.shape == shape for cp in list_of_controlpoints )

  Xi = flat_meshgrid(*[np.linspace(0, 1, npoints)]*2, axis=1)

  return list(map(lambda cp: np.stack([_callND(Xi,
                                               [knotvector, knotvector],
                                               np.array([degree, degree], dtype=int),
                                               _cp,
                                               np.array([dx, dy], dtype=int)) for _cp in cp.T], axis=1), list_of_controlpoints))
