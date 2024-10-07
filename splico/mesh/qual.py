from numpy import linalg
from gc import freeze
from turtle import end_fill
from ..util import np, _, frozen, HashMixin, frozen_cached_property, _round_array, isincreasing

from ._ref_structured import _refine_structured as ref_structured

import pyvista as pv
import vtk

from abc import abstractmethod

from typing import Callable, Sequence, Self

from functools import cached_property, lru_cache
from itertools import count, product

import pickle

from .mesh import Mesh 


def arg_max_min(arr: np.ndarray | list ) -> np.ndarray | list :

    index_arg_max = np.argmax(arr)
    index_arg_min = np.argmin(arr)

    value_max = arr[index_arg_max]
    value_min = arr[index_arg_min]

    return value_max, value_min

def aspectratio_unstruct(Mesh):
  '''
  Efficiency problem. Some edges are checked 2 times because they are shared between elements.
  Not important the order (both in 2D and 3D) of the indeces in the elements, since the grid is unstructured.
  '''  

  aspect_ratio = []  
  
  # non-vectorized version
  for i in range(Mesh.elements.shape[0]):
    distance_list = []
    for k in range(Mesh.elements.shape[1]):
      for j in range(Mesh.elements.shape[1]):
        if j != k :
          distance = linalg.norm(Mesh.points[Mesh.elements[i, k]] - Mesh.points[Mesh.elements[i, j]]) 
          distance_list.append(distance)
          
    # np.argmin/max gives back the index, not the value!
    dist_min = np.argmin(distance_list)
    dist_max = np.argmax(distance_list)
    
    aspect_ratio_element = distance_list[dist_max]/distance_list[dist_min]
    aspect_ratio.append(aspect_ratio_element)

  # freeze the arrays
  max_aspect_ratio, min_aspect_ratio  = arg_max_min(aspect_ratio)

  mean_aspect_ratio = frozen(np.mean(aspect_ratio))   
  max_aspect_ratio = frozen(max_aspect_ratio)      
  min_aspect_ratio = frozen(min_aspect_ratio)
  
  # return as tuple, better the stats are immutable
  stats = (mean_aspect_ratio, max_aspect_ratio, min_aspect_ratio)
  
  return stats


def skewness_quality_2D_unstruct(Mesh):

  skewness = []  
  
  # non-vectorized version
  for i in range(Mesh.elements.shape[0]):
    angles = []
    for k in range(Mesh.elements.shape[1]):
      for j in range(Mesh.elements.shape[1]):
        if j != k :
          projection = np.dot(Mesh.points[Mesh.elements[i, k]], Mesh.points[Mesh.elements[i, j]])
          angles.append(np.arccos(projection))
    
    max_angle_element, min_angle_element = arg_max_min(angles)
    
    skewness_elem = [(max_angle_element - np.pi/3)/ (np.pi - np.pi/3), (np.pi/3 - min_angle_element)/(np.pi/3)]
    index_skew = np.argmax(skewness_elem)

    skewness.append(skewness_elem[index_skew])
  
  max_skewness, min_skewness = arg_max_min(skewness)
  
  mean_skewness = frozen(np.mean(skewness))   
  max_skewness = frozen(max_skewness)      
  min_skewness = frozen(min_skewness)
  
  # return as tuple, better the stats are immutable
  stats = (mean_skewness, max_skewness, min_skewness)
    
  return stats


def aspectratio_2D_struct(Mesh):
  '''
  Efficiency problem. Some edges are checked 2 times because they are shared between elements.
  Not important the order (both in 2D and 3D) of the indeces in the elements, since the grid is unstructured.
  '''
 
  # Mesh.elements = Mesh.lexsort_elements(Mesh.elements)
  aspect_ratio = []  
  
  # non-vectorized version
  for i in range(Mesh.elements.shape[0]):
    distance_list = []
    for k in range(Mesh.elements.shape[1]):
      for j in range(Mesh.elements.shape[1]):
        if j != k :
          if k == 0 and j != 3:
            distance = linalg.norm(Mesh.points[Mesh.elements[i, k]] - Mesh.points[Mesh.elements[i, j]]) 
            distance_list.append(distance)          
          if k == 1 and j == 3:
            distance = linalg.norm(Mesh.points[Mesh.elements[i, k]] - Mesh.points[Mesh.elements[i, j]]) 
            distance_list.append(distance)
          if k == 2 and j == 3:            
            distance = linalg.norm(Mesh.points[Mesh.elements[i, k]] - Mesh.points[Mesh.elements[i, j]]) 
            distance_list.append(distance)
         
    # np.argmin/max gives back the index, not the value!
    dist_min = np.argmin(distance_list)
    dist_max = np.argmax(distance_list)
    
    aspect_ratio_element = distance_list[dist_max]/distance_list[dist_min]
    aspect_ratio.append(aspect_ratio_element)

    #import pdb
    #pdb.set_trace()
  # freeze the arrays
  max_aspect_ratio, min_aspect_ratio  = arg_max_min(aspect_ratio)

  mean_aspect_ratio = frozen(np.mean(aspect_ratio))   
  max_aspect_ratio = frozen(max_aspect_ratio)      
  min_aspect_ratio = frozen(min_aspect_ratio)
  
  # return as tuple, better the stats are immutable
  stats = (mean_aspect_ratio, max_aspect_ratio, min_aspect_ratio)
  
  return stats