from splico.geo import interp, CrossSectionMaker
from splico.spl import NDSpline
from splico.mesh import rectilinear, mesh_union

import numpy as np


def main(spl0: NDSpline, spl1: NDSpline, t0: np.ndarray, t1=None) -> NDSpline:
  """
    Perform a linear interpolation and a cubic hermite interpolation between
    `spl0` and `spl1`.

    Parameters
    ----------
    spl0 : :class:`splico.spl.NDSpline`
        The first input spline.
    spl1 : :class:`splico.spl.NDSpline`
        The second input spline.
    t0 : :class:`np.ndarray`
        The desired tangent at xn+1 == 0.
    t1 : :class:`np.ndarray`
        The desired tangent at xn+1 == 1. If not passed, defaults to `t0`.
  """
  assert spl0.nvars == spl1.nvars == 2

  if t1 is None:
    t1 = t0
  t0, t1 = map(np.asarray, (t0, t1))

  sample_mesh = rectilinear([np.linspace(0, 1, 21)] * 3)

  linear_interp = interp.linear_interpolation(spl0, spl1, zdegree=1)
  mesh_union(*(myspline.sample_mesh(sample_mesh) for myspline in linear_interp),
             boundary=True).plot()

  hermite_interp = interp.cubic_hermite_interpolation(spl0, spl1, t0, t1)
  mesh_union(*(myspline.sample_mesh(sample_mesh) for myspline in hermite_interp),
             boundary=True).plot()


if __name__ == '__main__':
  maker = CrossSectionMaker(7)

  spl0 = maker.make_disc(1, 1, 0)
  spl1 = maker.make_disc(1, 1, 1) + np.array([[0, 0, 2]])

  main(spl0, spl1, [0, 1, 1])
