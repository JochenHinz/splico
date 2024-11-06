from test_mesh import unit_disc_triangulation
from splico.mesh.qual import aspect_ratio
from splico.mesh import rectilinear
from splico.mesh.skewness import skewness_2D

import unittest


class TestMeshQualityCriteria(unittest.TestCase):
             
    
    def test_skew(self):
        mesh_unstruct = unit_disc_triangulation()
        
        stats_unstruct = skewness_2D(mesh_unstruct)
        
        print(stats_unstruct)
        
        self.assertTrue( stats_unstruct[0] >= stats_unstruct[2])
        self.assertTrue( stats_unstruct[0] <= stats_unstruct[1])
        
       
if __name__ == '__main__':
  unittest.main()