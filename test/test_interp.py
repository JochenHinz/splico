from splico.geo import CrossSectionMaker
from splico.geo.interp import linear_interpolation, cubic_hermite_interpolation
from splico.mesh import rectilinear, mesh_union
from splico.util import np

import unittest


class TestInterpolation(unittest.TestCase):

  def test_interpolate(self):
    maker = CrossSectionMaker(4)

    spl0 = maker.make_disc(.5, .5, 0)
    spl1 = maker.make_disc(2, 2, 0) + np.array([0, 0, 3])[None]

    sample_mesh = rectilinear([np.linspace(0, 1, 8),
                               np.linspace(0, 1, 8),
                               np.linspace(0, 1, 101)])

    with self.subTest('Cubic Hermite interpolation.'):
      test = linear_interpolation(spl0, spl1)
      mesh = mesh_union(*(mymesh.sample_mesh(sample_mesh) for mymesh in test), boundary=True)
      mesh.plot()

    with self.subTest('Cubic Hermite interpolation.'):
      test = cubic_hermite_interpolation(spl0, spl1, np.array([0, 0, 3]), np.array([0, 0, 3]))
      mesh = mesh_union(*(mymesh.sample_mesh(sample_mesh) for mymesh in test), boundary=True)
      mesh.plot()


if __name__ == '__main__':
  unittest.main()
