from splico.util import np, normalize, isincreasing
from splico.spl import NDSpline

from numba import njit
from typing import Optional, Sequence
import treelog as log


@njit(cache=True)
def _compute_notwistframe(T, n0, zero_thresh):

  # XXX: docstring

  t0 = T[0]
  binormal0 = np.cross(t0, n0)

  Rs = np.empty((len(T), 3, 3), dtype=np.float64)
  Rs[0] = np.stack((n0, binormal0, t0), axis=1)

  for i, (told, tnew) in enumerate(zip(T, T[1:]), 1):
    omega = np.cross(told, tnew)
    norm = (omega**2).sum()**.5
    if norm <= zero_thresh:
      binormal = Rs[i-1][:, 1]
      normal = Rs[i-1][:, 0]
    else:
      omega = omega / norm
      phi = np.arccos(np.dot(told, tnew))

      binormal_p = Rs[i-1][:, 1]
      binormal = binormal_p * np.cos(phi) + \
                 np.cross(omega, binormal_p) * np.sin(phi) + \
                 omega * np.dot(omega, binormal_p) * (1 - np.cos(phi))

      normal_p = Rs[i-1][:, 0]
      normal = normal_p * np.cos(phi) + \
               np.cross(omega, normal_p) * np.sin(phi) + \
               omega * np.dot(omega, normal_p) * (1 - np.cos(phi))

    Rs[i][:, 0] = normal
    Rs[i][:, 1] = binormal
    Rs[i][:, 2] = T[i]

  return Rs


def compute_notwistframe(tangents: np.ndarray,
                         n0: Optional[np.ndarray | Sequence[float]] = None,
                         zero_thresh=1e-8):

  # XXX: docstring

  assert (tangents := normalize(tangents)).shape[1:] == (3,)
  assert zero_thresh >= 0

  if n0 is None:
    n0 = np.array([-tangents[0, 2], 0, tangents[0, 0]])

  assert n0.shape == (3,)

  return _compute_notwistframe(tangents, n0, zero_thresh)


def compute_notwistframe_from_spline(spline: NDSpline,
                                     abscissae: np.ndarray | Sequence[float ] | int,
                                     refit=False,
                                     framekwargs=None,
                                     refitkwargs=None) -> np.ndarray | NDSpline:

  # XXX: docstring

  assert spline.nvars == 1 and spline.shape == (3,)
  if refit is False and refitkwargs:
    log.warning("Received refit keyword arguments but refit is set to false. They will be ignored.")

  (a, *ignore, b), = spline.knotvector.knots

  if np.isscalar(abscissae):
    abscissae = np.linspace(a, b, abscissae)

  assert a <= abscissae.min() < abscissae.max() <= b
  assert isincreasing(abscissae)

  tangents = spline(abscissae, dx=1)

  Rs = compute_notwistframe(tangents, **framekwargs or {})

  if refit:
    Rs = spline.knotvector.fit([abscissae], Rs, **refitkwargs)

  return Rs
