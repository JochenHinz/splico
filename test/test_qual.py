from test_mesh import unit_disc_triangulation
import numpy as np
from splico.mesh.qual import vectorized_aspect_ratio
from splico.mesh import unitsquare


import unittest


class TestMeshQualityCriteria(unittest.TestCase):
             
    
    def test_aspect_ratio_unstruct(self):
        mesh = unit_disc_triangulation()
        
        stats = vectorized_aspect_ratio(mesh)
        
        print(stats)
        self.assertTrue( stats[0] >= stats[2])
        self.assertTrue( stats[0] <= stats[1])
        
       
if __name__ == '__main__':
  unittest.main()