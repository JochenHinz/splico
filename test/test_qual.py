from test_mesh import unit_disc_triangulation
import numpy as np
from splico.mesh.qual import aspectratio_unstruct, skewness_quality_2D_unstruct, aspectratio_2D_struct
from splico.mesh import unitsquare


import unittest


class TestMeshQualityCriteria(unittest.TestCase):
             
    
    def test_aspect_ratio_unstruct(self):
        mesh = unit_disc_triangulation()
        
        stats = aspectratio_unstruct(mesh)
        
        self.assertTrue( stats[0] >= stats[2])
        self.assertTrue( stats[0] <= stats[1])
        
    
    def test_skewness_2D_unstruct(self):
        mesh = unit_disc_triangulation()
    
        stats = skewness_quality_2D_unstruct(mesh)
        #print(stats)
        assert stats[0] <= stats[1], "mean bigger than max value"
        assert stats[0] >= stats[2], "mean lower than min value"


    def test_aspectratio_2D_struct(self):
      
        mesh = unitsquare((4, 5))
        #import pdb
        #pdb.set_trace()
        
        stats = aspectratio_2D_struct(mesh)
        print(stats)
        self.assertTrue( stats[0] >= stats[2])
        self.assertTrue( stats[0] <= stats[1])

       
if __name__ == '__main__':
  unittest.main()