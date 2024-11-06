from numpy import linalg
from pyvista import Line

from ..util import np, _, frozen

from typing import Tuple

from functools import  lru_cache

from .mesh import LineMesh, Mesh 


IMPLEMENTED_MESH_TYPES = ('triangle', 'tetrahedron', 'quadrilateral', 'hexahedron')



def skewness_2D(mesh: Mesh) -> Tuple[np.float_,...]:
   
   # Check for the mesh validity
   assert mesh.is_valid(), "mesh is not valid" 
    
   if mesh.simplex_type in {'quadrilateral', 'hexahedron'}:
       ref_angle = np.pi/2
   else:
       ref_angle = np.pi/3
   
   mesh.elements = mesh.elements[:, mesh._submesh_indices]
   
   import ipdb
   ipdb.set_trace()
   
   elem_point = mesh.points[mesh.elements]
   
   diff = elem_point[...,1,:] - elem_point[...,0,:]
   
   norm = linalg.norm(diff,axis = -1)
   
   angles0 = np.arccos(np.sum(diff[:,0,:] * diff[:,1,:], axis=1))
   angles1 = np.arccos(np.sum(-diff[:,0,:] * diff[:,2,:], axis=1))
   
   angles = np.stack([angles0, angles1, np.pi - angles0 - angles1], axis = -1)

   max_angles = np.max(angles, axis = 1)
   min_angles = np.min(angles, axis = 1)
   
   skewness = np.maximum((max_angles - ref_angle)/ (np.pi - ref_angle), 
                         (ref_angle - min_angles)/ ref_angle)
   
   stats = np.mean(skewness), np.max(skewness), np.min(skewness)
 
   return stats