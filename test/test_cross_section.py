from splico.geo import CrossSectionMaker
from splico.mesh import rectilinear, mesh_boundary_union

import unittest


class TestDisc(unittest.TestCase):

  def test_eval(self):
    dmaker = CrossSectionMaker(5)
    sol = dmaker.make_disc(4, 4, 0)

    rmesh = rectilinear([5, 5])

    all_meshes = [spl.sample_mesh(rmesh) for spl in sol]
    mesh = mesh_boundary_union(*all_meshes)

    mesh.plot()


if __name__ == '__main__':
  unittest.main()
