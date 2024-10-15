from numpy import linalg
from pyvista import Line

from ..util import np, _, frozen

from typing import Tuple

from functools import  lru_cache

from .mesh import LineMesh, Mesh 


IMPLEMENTED_MESH_TYPES = ('triangle', 'tetrahedron', 'quadrilateral', 'hexahedron')


def aspect_ratio(mesh: Mesh) -> Tuple[np.float_,...]:
   """Compute the aspect ratio of edges for each element in a given mesh.

    This function calculates the aspect ratio (AR) for each element in the mesh,
    defined as the ratio between the longest and shortest edge in that element.
    The aspect ratio is a measure of mesh quality:
    - AR close to 1 indicates a well-proportioned, high-quality mesh element
    - AR much greater than 1 indicates a stretched mesh element

   Args:
       mesh (Mesh): Mesh object (both 2D/3D unstructured and structured)

   Returns:
       np.ndarray: mean, maximum and minimum aspect ratio.

    Notes:
        - Edge lengths are typically computed using Euclidean distance.
        - For 2D elements (e.g., triangles, quads), the aspect ratio is based on edge lengths.
        - For 3D elements the aspect ratio is still based on edge lengths,
          not face diagonals or body diagonals.
      
   """      
   
   # Check for the mesh validity
   assert mesh.is_valid(), "mesh is not valid"

   submesh = mesh.submesh   
   
   mesh.elements = mesh.elements[:,mesh._submesh_indices]
   
   # 2D mesh
   if mesh._submesh_type == LineMesh:
      elem_point = mesh.points[mesh.elements]
      distances = linalg.norm(elem_point[...,1,:] - elem_point[...,0,:], axis = -1)
   else: # 3D mesh
      mesh.elements = mesh.elements[:,:,submesh._submesh_indices]
      elem_point = mesh.points[mesh.elements]
      distances = linalg.norm(elem_point[...,1,:] - elem_point[...,0,:], axis = -1).reshape(elem_point.shape[0],-1)

   dist_min = np.min(distances, axis = 1)
   dist_max = np.max(distances, axis = 1)
   
   # Calculate aspect ratios for each element
   aspect_ratios = dist_max / dist_min

   # Making a tuple
   stats = np.mean(aspect_ratios), np.max(aspect_ratios), np.min(aspect_ratios)

   return stats