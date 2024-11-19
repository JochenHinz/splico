#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains functions to compute mesh quality metrics.
@author: Fabio Marcinno'
"""

from ..util import np
from .mesh import LineMesh, Mesh

from typing import Tuple

from numpy import linalg


IMPLEMENTED_MESH_TYPES = ('triangle', 'tetrahedron', 'quadrilateral', 'hexahedron')


def aspect_ratio(mesh: Mesh) -> Tuple[np.float_, ...]:
  """
  Compute the aspect ratio of edges for each element in a
  given mesh.

  This function calculates the aspect ratio (AR) for each
  element in the mesh, defined as the ratio between the
  longest and shortest edge in that element. The aspect
  ratio is a measure of mesh quality:
  - AR close to 1 indicates a well-proportioned,
    high-quality mesh element
  - AR much greater than 1 indicates a stretched
    mesh element

  Args:
   mesh (Mesh): Mesh object (both 2D/3D unstruct/struct)

  Returns:
    tuple with entries of type :class:`np.float_` representing, in order,
    the  mean, maximum and minimum aspect ratio.

  Notes:
   - Edge lengths are typically computed using Euclidean distance.
   - For 2D elements (e.g., triangles, quads), the aspect
     ratio is based on edge lengths.
   - For 3D elements the aspect ratio is still based on
     edge lengths, but excludes face diagonals or body diagonals.
  """

  assert mesh.is_valid(), "mesh is not valid"
  assert mesh.element_name in IMPLEMENTED_MESH_TYPES, \
      "This operation is currently not supported for this mesh type."

  elements = mesh.elements[:, mesh._submesh_indices]

  # Decomposing the mesh
  if mesh._submesh_type is LineMesh:  # 2D mesh case
     pass
  else:  # 3D mesh case
     elements = elements[:, :, mesh.submesh._submesh_indices]

  elem_point = mesh.points[elements]

  distances = linalg.norm(elem_point[..., 1, :] - elem_point[..., 0, :], axis=-1)
  distances = distances.reshape(distances.shape[0], -1)

  dist_min = np.min(distances, axis=1)
  dist_max = np.max(distances, axis=1)

  aspect_ratios = dist_max / dist_min

  stats = np.mean(aspect_ratios), np.max(aspect_ratios), np.min(aspect_ratios)

  return stats

def skewness(mesh: Mesh) -> Tuple[np.float_,...]:
   
   # Check for the mesh validity
   assert mesh.is_valid(), "mesh is not valid" 
   assert mesh.element_name in IMPLEMENTED_MESH_TYPES, \
      "This operation is currently not supported for this mesh type." 
   
   elements = mesh.elements[:, mesh._submesh_indices]
   
   # TODO: make ref_angle a property of the reference element 
   if mesh.element_name in {'triangle', 'quadrilateral'}: # 2D mesh
      if mesh.element_name in {'triangle'}:
          ref_angle = np.pi/3
      else:
          ref_angle = np.pi/2 
   elif mesh.element_name in {'tetrahedron',  'hexahedron'}:
      elements = elements[:, :, mesh.submesh._submesh_indices]
      elements = elements.reshape(elements.shape[0],elements.shape[1]*elements.shape[2], -1) 
      if mesh.element_name in {'tetrahedron'}:
          ref_angle = np.pi/3
      else:
          ref_angle = np.pi/2 
   
   skew = []
 
   for i in range(mesh.elements.shape[0]):
     skew.append(np.unique(np.concatenate([elements[i], elements[i,:, ::-1]]), axis=0))
     
   skew = np.asarray(skew)

   elem_point = mesh.points[skew]
   
   lines =  elem_point[...,1,:] - elem_point[...,0,:]
   
   lines = lines.reshape(lines.shape[0]*lines.shape[1],-1)

   scalar_product = np.sum(lines[::2]*lines[1::2], axis = 1)
   
   if mesh.element_name in {'tetrahedron',  'hexahedron'}:   
      angles = np.arccos(scalar_product).reshape(mesh.elements.shape[0], int(len(mesh._submesh_indices) * len(mesh.submesh._submesh_indices)/2))
   else:
      angles = np.arccos(scalar_product).reshape(mesh.elements.shape[0], len(mesh._submesh_indices))

   max_angles = np.max(angles, axis = 1)
   min_angles = np.min(angles, axis = 1)
   
   skewness = np.maximum((max_angles - ref_angle)/ (np.pi - ref_angle), 
                         (ref_angle - min_angles)/ ref_angle)
   
   stats = np.mean(skewness), np.max(skewness), np.min(skewness)
 
   return stats