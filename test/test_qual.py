from test_mesh import unit_disc_triangulation
from splico.mesh.qual import aspect_ratio,skewness_metric
from splico.mesh import unitsquare

import unittest


class TestMeshQualityCriteria(unittest.TestCase):
             
    
    def test_aspect_ratio(self):
        mesh_unstruct = unit_disc_triangulation()
        mesh_struct = unitsquare((4,5,6))
        
        #stats_skew = skewness_metric(mesh_unstruct)
        
        stats_unstruct = aspect_ratio(mesh_unstruct)
        stats_struct = aspect_ratio(mesh_struct)
        
        print(stats_unstruct)
        print(stats_struct)
        
        self.assertTrue( stats_unstruct[0] >= stats_unstruct[2])
        self.assertTrue( stats_unstruct[0] <= stats_unstruct[1])
        
        self.assertTrue( stats_struct[0] >= stats_struct[2])
        self.assertTrue( stats_struct[0] <= stats_struct[1])
        
       
if __name__ == '__main__':
  unittest.main()