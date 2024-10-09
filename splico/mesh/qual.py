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


def vectorized_aspect_ratio(mesh: Mesh) -> Tuple[np.ndarray[np.float_, 1], 3]:
    '''
    Given an unstructred (2D or 3D) mesh (defined by its elements and points), compute the aspect
    ratio (AR) of its edges defined by the ratio between the longest and shortest edge in a given element.
    AR close to 1 -> good mesh
    AR >> 1 -> stretched mesh 
    '''  
    
    # Check for the mesh validity
    mesh.is_valid()

    # Use advanced indexing to get all points for all elements at once
    element_points = mesh.points[mesh.elements]

    # Calculate pairwise distances for all elements
    distances = linalg.norm(element_points[:, :, np.newaxis] - element_points[:, np.newaxis, :], axis=-1)
    
    # Create a mask to ignore self-distances (diagonal)
    mask = np.eye(mesh.nverts, dtype=bool)
    mask_vect = np.bool_(mask[np.newaxis,:,:] * np.ones((1,distances.shape[0]))[np.newaxis,:].T)

    
    valid_distances = distances[~mask_vect].reshape(distances.shape[0], mesh.nverts**2 - mesh.nverts)

    dist_min = np.min(valid_distances, axis = 1)
    dist_max = np.max(valid_distances, axis = 1)
    
    # Calculate aspect ratios for each element
    aspect_ratios = dist_max / dist_min

    # Freeze stats & making a tuple
    stats = np.stack( (frozen(np.mean(aspect_ratios)), frozen(np.max(aspect_ratios)), frozen(np.min(aspect_ratios))) , axis = -1) 

    return (stats)