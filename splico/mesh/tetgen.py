#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Fabio Marcinno'
"""

from ..util import frozen, frozen_cached_property
from functools import cached_property
from mesh import Triangulation, AffineMesh


class TetGen(AffineMesh):
  """

          1
          |\
          |  \
          |    \
          |      \
          |        \
        0 |__________\ 2
          /         /
         /        /
        /       /
       /     /
      /   /
     / /
    3

    the edge between vertices 3-1 is understood
  """

  simplex_type = 'tetrahedron'
  ndims = 3
  nverts = 4
  nref = 4

  @cached_property
  def _submesh_indices(self):
    return tuple(map(frozen, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))

  @frozen_cached_property
  def pvelements(self):
    return self.elements[:, [0, 3, 1, 2]]

  @property
  def _submesh_type(self):
    return Triangulation

  def subdivide_elements(self):
    raise NotImplementedError

  def _refine(self):
    raise NotImplementedError
