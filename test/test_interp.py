from splico.geo import CrossSectionMaker
from splico.geo.interp import cubic_hermite_interpolation
from splico.mesh import rectilinear, mesh_boundary_union
from splico.util import np

import unittest


class TestInterpolation(unittest.TestCase):

  def test_cubic_hermite(self):
    maker = CrossSectionMaker(4)

    spl0 = maker.make_disc(.5, .5, 0)
    spl1 = maker.make_disc(2, 2, 0) + np.array([0, 0, 3])[None]

    test = cubic_hermite_interpolation(spl0, spl1, np.array([0, 0, 3]), np.array([0, 0, 3]))

    sample_mesh = rectilinear([np.linspace(0, 1, 8),
                               np.linspace(0, 1, 8),
                               np.linspace(0, 1, 101)])

    mesh = mesh_boundary_union(*(mymesh.sample_mesh(sample_mesh) for mymesh in test))

    mesh.plot()


if __name__ == '__main__':
  unittest.main()
