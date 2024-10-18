from test_mesh import unit_disc_triangulation
from splico.mesh.qual import aspect_ratio
from splico.mesh import rectilinear

import unittest


class TestMeshQualityCriteria(unittest.TestCase):
             
    
    def test_aspect_ratio(self):
        mesh_unstruct = unit_disc_triangulation()
        mesh_struct = rectilinear((17,25,13))
        
        mesh_struct.plot
        
        stats_unstruct = aspect_ratio(mesh_unstruct)
        #stats_unstruct = aspect_ratio(mesh_struct)
        
        print(stats_unstruct)
        
        self.assertTrue( stats_unstruct[0] >= stats_unstruct[2])
        self.assertTrue( stats_unstruct[0] <= stats_unstruct[1])
        
       
if __name__ == '__main__':
  unittest.main()