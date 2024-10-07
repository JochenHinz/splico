from test_mesh import unit_disc_triangulation
import numpy as np
from splico.mesh import aspectratio_unstruct, skewness_quality_2D_unstruct

import unittest


class TestMeshQualityCriteria(unittest.TestCase):
             
    
    def test_aspect_ratio_unstruct(self):
        mesh = unit_disc_triangulation()
        
        stats = aspectratio_unstruct(mesh)
        
        self.assertTrue( stats[0] >= stats[2])
        self.assertTrue( stats[0] <= stats[1])
        
    
    def test_skewness_unstruct():
    mesh = unit_disc_triangulation()
    
    stats = skewness_quality_2D_unstruct(mesh)
    
    assert stats[0] <= stats[1], "mean bigger than max value"
    assert stats[0] >= stats[2], "mean lower than min value"
    
    
    
    
        
if __name__ == '__main__':
  unittest.main()