from numpy import linalg
from gc import freeze
from turtle import end_fill
from ..util import np, _, frozen, HashMixin, frozen_cached_property, _round_array, isincreasing

from ._ref_structured import _refine_structured as ref_structured

import pyvista as pv
import vtk

from abc import abstractmethod

from typing import Callable, Sequence, Self, Tuple

from functools import cached_property, lru_cache
from itertools import count, product

import pickle

from .mesh import Mesh 


@lru_cache
def invalid_edges(mesh: Mesh):
  
  non_valid_edges = 0
  
  if mesh.simplex_type in ['triangle','tetrahedron']:
     non_valid_edges = 1
  
  if mesh.simplex_type == 'quadrilateral':
     non_valid_edges = 2   
  
  if mesh.simplex_type == 'hexahedron':
     non_valid_edges = 5
  
  
  assert non_valid_edges != 0, "Aspect ratio for the input mesh has not been implemented"

  
  return non_valid_edges

@lru_cache
def clean_distances(mesh: Mesh) -> int :
  
  # eliminate self-distances
  mask_self = np.eye(mesh.nverts)
  
  # eliminate non valid edges in the quad
  mask_quad = np.array(([0, 0, 0, 1], [0,0,1,0],[0,1,0,0],[1,0,0,0]), dtype= bool)
  
  # eliminate non valid edges in the hex
  mask_quad_to_quad = np.array(([0, 1, 1, 1], [1,0,1,1],[1,1,0,1],[1,1,1,0]), dtype= bool)
  
  if mesh.simplex_type in ['triangle','tetrahedron']:
     mask = np.copy(mask_self)
     
  if mesh.simplex_type in 'quadrilateral':
     mask = mask_self + mask_quad 
    
  if mesh.simplex_type in 'hexahedron':
     mask = mask_self + np.block([[mask_quad, mask_quad_to_quad], [mask_quad_to_quad, mask_quad]])
       

  return mask


# Tuple[np.ndarray[np.float_, 1], 3] means returns a tuple containing 3 elements of the type np.float.
def aspect_ratio(mesh: Mesh) -> Tuple[np.ndarray[np.float_, 1], 3]:
  '''
    Given a general mesh (defined by its elements and points), compute the aspect
    ratio (AR) of its edges defined by the ratio between the longest and shortest edge in a given element.
    AR close to 1 -> good mesh
    AR >> 1 -> stretched mesh 
  '''
  
  # Check for the mesh validity
  assert mesh.is_valid(), "mesh is not valid"

  element_points = mesh.points[mesh.elements]

  # Calculate pairwise distances for all elements
  distances = linalg.norm(element_points[:, :, _] - element_points[:, _, :], axis=-1)
  
  # Depending on the mesh type, mask cut the invalid distances
  mask = clean_distances(mesh)
  mask_vect = np.bool_(mask[_,:,:] * np.ones((distances.shape[0],1,1)))
    
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
