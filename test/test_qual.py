from test_mesh import unit_disc_triangulation
import numpy as np
from splico.mesh.qual import vectorized_aspect_ratio, vectorized_aspectratio_2D_struct, vectorized_aspectratio_3D_struct
from splico.mesh import unitsquare


import unittest


class TestMeshQualityCriteria(unittest.TestCase):
             
    
    def test_aspect_ratio_unstruct(self):
        mesh = unit_disc_triangulation()
        
        stats = vectorized_aspect_ratio(mesh)
        
        #print(stats)
        self.assertTrue( stats[0] >= stats[2])
        self.assertTrue( stats[0] <= stats[1])
        
       
    def test_aspect_ratio_struct(self):
        mesh = unitsquare((4, 5, 6))
        
        stats = vectorized_aspectratio_3D_struct(mesh)
    
        print(stats)
        self.assertTrue( stats[0] >= stats[2])
        self.assertTrue( stats[0] <= stats[1])
            
       
       
if __name__ == '__main__':
  unittest.main()