from numpy import linalg

from ..util import np, _, frozen

from typing import Tuple

from functools import  lru_cache

from .mesh import Mesh 


IMPLEMENTED_MESH_TYPES = ('triangle', 'tetrahedron', 'quadrilateral', 'hexahedron')


@lru_cache(maxsize=32)
def invalid_edges(mesh: Mesh) -> int :
   """_summary_

   Args:
       mesh (Mesh): Mesh object (both 2D/3D unstructured and structured)

   Returns:
       int: number of invalid edges
       
   This function calculates the number of lines that are not valid edges in a geometric element
   (e.g., triangle, tetrahedron, hexahedron) when computing the lines between all points of the element. 
   For example in the case of a triangle, non_valid_edges = 1 since the distance between the same point
   which is not valid, but all the other lines are valid edge. For the quad, non_valid_edges = 2, since 
   the distance between the same point and the one on the other side of the diagonal (see class QuadMesh(BilinearMixin, Mesh)), i.e.,
   the line between vertices (0,3) is a not valid edge.
   
   """  
   
   assert mesh.simplex_type in IMPLEMENTED_MESH_TYPES, NotImplementedError
   
   non_valid_edges = 0
  
   if mesh.simplex_type in ['triangle','tetrahedron']:
     non_valid_edges = 1
   elif mesh.simplex_type == 'quadrilateral':
     non_valid_edges = 2   
   elif mesh.simplex_type == 'hexahedron':
     non_valid_edges = 5
  
   return frozen(non_valid_edges)

@lru_cache(maxsize=32)
def clean_distances(mesh: Mesh) -> np.ndarray :
   """_summary_

   Args:
       mesh (Mesh): Mesh object (both 2D/3D unstructured and structured)

   Returns:
       np.ndarray: For a given mesh element, this function creates a boolean matrix
       that identifies which componenet in the distance tensor represent valid 
       distances. It filters out invalid distances (inserting 1), such as those between a point
       and itself (on the diagonal of the distance tensor), and other non-edge distances
       specific to the mesh type. For all the mesh types, the diagonal of boolean matrix (mask)
       is filled with ones.
     """   
  
   assert mesh.simplex_type in IMPLEMENTED_MESH_TYPES, NotImplementedError
  
   # eliminate self-distances
   mask_self = np.eye(mesh.nverts)
  
   # eliminate non valid edges in the quad. Self distances are keeped for the moment.
   mask_quad = np.array(([0, 0, 0, 1], [0,0,1,0],[0,1,0,0],[1,0,0,0]), dtype= bool)
  
   # eliminate non valid edges in the hex
   mask_quad_to_quad = np.array(([0, 1, 1, 1], [1,0,1,1],[1,1,0,1],[1,1,1,0]), dtype= bool)
   
   if mesh.simplex_type in ['triangle','tetrahedron']:
      mask = np.copy(mask_self)   
   elif mesh.simplex_type in 'quadrilateral':
      mask = mask_self + mask_quad 
   elif mesh.simplex_type in 'hexahedron':
      mask = mask_self + np.block([[mask_quad, mask_quad_to_quad], [mask_quad_to_quad, mask_quad]])
         
   return frozen(mask)


# Tuple[np.ndarray[np.float_, 1], 3] means returns a tuple containing 3 elements of the type np.float.
def aspect_ratio(mesh: Mesh) -> Tuple[np.ndarray[np.float_, 1], 3]:
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

   element_points = mesh.points[mesh.elements]

   # Calculate pairwise distances for all elements
   distances = linalg.norm(element_points[:, :, _] - element_points[:, _, :], axis=-1)
   
   # Depending on the mesh type, mask cut the invalid distances
   mask = clean_distances(mesh)
   mask_vect = np.bool_(np.tile(mask , (distances.shape[0],1,1)))
      
   # Number of invalid edges considering that the distance is computed with respect all the points
   n_non_valid_edges = invalid_edges(mesh)
      
   valid_distances = distances[~mask_vect].reshape(distances.shape[0], mesh.nverts**2 - n_non_valid_edges*mesh.nverts )
   
   dist_min = np.min(valid_distances, axis = 1)
   dist_max = np.max(valid_distances, axis = 1)
      
   # Calculate aspect ratios for each element
   aspect_ratios = dist_max / dist_min

   # Freeze stats & making a tuple
   stats = np.stack(( np.mean(aspect_ratios), np.max(aspect_ratios), np.min(aspect_ratios)), axis = -1) 

   return (stats)