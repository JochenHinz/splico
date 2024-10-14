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

from mesh import BilinearMixin, Mesh, Triangulation



class TetGen(BilinearMixin,Mesh):
    
    simplex_type = 'Tetraheda'
    ndims = 3
    nverts = 4
    nref = 4
    is_affine = False
    # to ask about being affine
    
    
# why _ in front of the function
@cached_property
def _submesh_indices(self):
    return tuple(map(frozen, [[0,1,2], [0,1,3], [1,2,3]] ))        
    
# reordering of the elements, why calling pvelements?
@cached_property
def pvelements(self):
    return self.elements[:, [0,1,2,3]]

@property
def _submesh_type(self):   
    return Triangulation
    
    
    














