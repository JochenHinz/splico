from splico.geo import CrossSectionMaker
from splico.mesh import rectilinear, mesh_boundary_union

import unittest


class TestDisc(unittest.TestCase):

  def test_eval(self):
    dmaker = CrossSectionMaker(5)
    sol = dmaker.make_disc(4, 4, 0)

    for n in (3, 5, 7, 11, 13, 17, 19):
      with self.subTest('Evaluating in {n} x {n} points'):
        rmesh = rectilinear([n, n])

        all_meshes = [spl.sample_mesh(rmesh) for spl in sol]
        mesh = mesh_boundary_union(*all_meshes)

        # 5 faces, 12 edges, 8 vertices
        npoints = 5 * (n-2)**2 + 12 * (n-2) + 8

        self.assertEqual(len(mesh.points), npoints)


if __name__ == '__main__':
  unittest.main()