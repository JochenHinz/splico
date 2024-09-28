#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jochen Hinz
"""

from ..util import np, _, frozen, HashMixin, frozen_cached_property, _round_array, isincreasing

from ._ref_structured import _refine_structured as ref_structured

import pyvista as pv
import vtk

from abc import abstractmethod

from typing import Callable, Sequence, Self

from functools import cached_property, lru_cache
from itertools import count, product

import pickle


# all supported element types (for now)
SIMPLEX_TYPES = ('point', 'line', 'triangle',
                 'quadrilateral', 'tetrahedron' 'hexahedron')


class MissingVertexError(Exception):
  pass


class HasNoSubMeshError(Exception):
  pass


class HasNoBoundaryError(Exception):
  pass


@lru_cache
def _issubmesh(mesh0, mesh1):

  # XXX: docstring

  # mesh0 mesh has more vertices per element than mesh1 => False
  if mesh0.nverts > mesh1.nverts:
    return False

  # both meshes have the same class, check if mesh0's elements
  # are a subset of mesh1's and if the points are the same
  if mesh0.__class__ is mesh1.__class__:

    # mesh0 has more elements => mesh1 can't be a submesh of mesh0
    if not len(mesh0.elements) <= len(mesh1.elements):
      return False

    # get the set of the union of both meshe's elements
    all_unique_elements = np.unique(np.concatenate([mesh0.elements,
                                                    mesh1.elements]), axis=0)

    # the shape differs from mesh1.elements.shape => mesh0 can't be a submesh
    if not all_unique_elements.shape == mesh1.elements.shape:
      return False

    # the shape is the same => make sure that unique_elements and
    # mesh1.elements have the same indices when brought into lexigraphically sorted form
    if (all_unique_elements != np.unique(mesh1.elements, axis=0)).any():
      return False

    # all tests passed => check if the relevant points of mesh0
    # are equal to that of mesh1.
    indices = np.unique(mesh0.elements)
    return (mesh0.points[indices] == mesh1.points[indices]).all()
  # try to take mesh1's submesh and see if mesh0 is a submesh of that one
  try:
    submesh1 = mesh1.submesh
    return _issubmesh(mesh0, submesh1)
  # if mesh1 has no submesh, return False
  except HasNoSubMeshError:
    return False


def mesh_boundary_union(*meshes, **kwargs):
  from _jit import mesh_boundary_union as _mesh_boundary_union
  return _mesh_boundary_union(*meshes, **kwargs)


def mesh_union(*meshes):
  from ._jit import mesh_union as _mesh_union
  return _mesh_union(*meshes)


def mesh_difference(mesh0, mesh1):
  """ Take the difference of two meshes. """
  from ._jit import mesh_difference as _mesh_difference
  return _mesh_difference(mesh0, mesh1)


# XXX: write another parent class that allows for mixed element types.
#      The meshes that have already been implemented are then special cases
#      with only one element type.


class AffineMixin:

  """
    Mixin for affine mesh types (PointMesh, LineMesh, Triangulation).
    Provides implementations for the elementwise metric tensor (GK),
    its inverse (GKinv) the elementwise constant measure (detBK)
    and a function to check for validty (is_valid).

    Each derived class needs to implement `BK` in order for the others to work.
  """

  @property
  def BK(self) -> np.ndarray:
    raise NotImplementedError("Any derived mesh has to implement the `BK` attribute.")

  @frozen_cached_property
  def GK(self) -> np.ndarray:
    """ Metric Tensor on each element. """
    return (self.BK.swapaxes(-1, -2)[..., _] * self.BK[..., _, :, :]).sum(-2)

  @frozen_cached_property
  def GKinv(self) -> np.ndarray:
    """ Metric Tensor inverse per element. """
    return np.stack(list(map(np.linalg.inv, self.G)))

  @frozen_cached_property
  def detBK(self) -> np.ndarray:
    """ Jacobian determinant (measure) per element. """
    # the np.linalg.det function returns of an array ``x`` of shape
    # (n, m, m) the determinant taken along the last two axes, i.e.,
    # in this case an array of shape (nelems,) where the i-th entry is the
    # determinant of self.BK[i]
    BK = self.BK
    if len(BK.shape) == 2:
      BK = BK[..., _]
    return np.sqrt(np.abs(np.linalg.det((BK.swapaxes(-1, -2)[..., _] * BK[..., _, :, :]).sum(-2))))

  def is_valid(self) -> bool:
    return (np.linalg.det(self.GK) > 0).all()


# XXX: in the long run it would make more sense to implement the exact same
#      methods for both affine and bilinear maps, even though the routines for
#      affine maps are simpler and implementing them in a similar way as for
#      bilinear maps seems redundant.


class BilinearMixin:

  """
    Mixin for bilinear mesh types (QuadMesh, HexMesh).
    Provides implementations for the geometry map as well as its jacobian
    per element.
    Furthermore provides an implementation of `is_valid` using the elementwise
    Jacobian evaluated in the vertices.
    Finally provides a standard implementation of `_refine`.
  """

  # XXX: some routines may have to be jitted for acceptable performance in large
  #      meshes.
  #      Also, it makes more sense to evaluate all geometry maps at once
  #      in a jitted routine rather than element-by-element.
  #      Only evaluating a subset of elements can be accomplished by using
  #      self.take(elemindices).

  def geometry_map(self, ielem: int) -> Callable:
    """ Geometry map per element that works for general bilinear meshes. """
    mypoints = list(map(lambda x: x[_], self.points[self.elements[ielem]]))

    def gmap(x: np.ndarray, mypoints=mypoints) -> np.ndarray:
      assert x.shape[1:] == (self.ndims,)
      return np.add.reduce([ p * np.multiply.reduce(*fs) for p, fs
                             in zip(mypoints, product(*[ [1 - _x[:, _], _x[:, _]] for _x in x.T ])) ])

    return gmap

  def Jmap(self, ielem: int) -> Callable:
    """ Jacobian matrix per element that works for general bilinear meshes. """
    mypoints = list(map(lambda x: x[_], self.points[self.elements[ielem]]))

    def gmap(x: np.ndarray, mypoints=mypoints) -> np.ndarray:
      assert x.shape[1:] == (self.ndims,)
      ones = np.ones((x.shape[0], 1))
      ret = []
      for i in range(self.ndims):
        myval = np.add.reduce([ p * np.multiply.reduce(fs, axis=0) for p, fs
                                in zip(mypoints, product(*[ [-ones, ones] if j == i else [1 - _x[:, _], _x[:, _]] for j, _x in enumerate(x.T) ])) ])
        ret.append(myval)
      return np.stack(ret, axis=-1)

    return gmap

  def is_valid(self) -> bool:
    # XXX: vertex check may not always be sufficient.
    check_points = np.stack(list(map(np.ravel, np.meshgrid(*[[0, 1]] * self.ndims))), axis=1)
    for i in range(len(self.elements)):
      Jeval = self.Jmap(i)(check_points)
      G = (Jeval.swapaxes(-1, -2)[..., _] * Jeval[..., _, :, :]).sum(-2)
      if (np.linalg.det(G) < 0).any():
        return False
    return True

  def _refine(self) -> Self:
    return _refine_structured(self)


class Mesh(HashMixin):

  """
    Abstract Base Class for representing various mesh types.

    Parameters
    ----------
    elements : :class:`np.ndarray` of integers or Sequence of integers.
        Element index array of shape (nelems, nverts).
    points : :class:`np.ndarray` or any array-like of type float.
        Point index array of shape (npoints, 3).
        The element index array must satisfy:
            0 <= elements.min() <= elements.max() < len(points).

    Attributes
    ----------
    elements : :class:`np.ndarray`, frozen-array.
        The element index array.
    points : :class:`np.ndarray`, frozen-array.
        The point array.
    simplex_type : :class:`str`, class attribute.
        The name of the simplex associated with the mesh.
    ndims : :class:`int`.
        The spatial dimensionality of the reference element. Must satisfy
        0 <= ndims < 4 (for now).
    nverts : :class:`int`.
        Number of vertices per element. `self.elements.shape[1:] == (nverts,)`.
    nref : :class:`int`.
        Number of new elements each element is replaced by under refinement.
    is_affine : :class:`bool`.
        Boolean representing whether the mesh is affine or not
        (may be removed in the future).
  """

  # Derived classes NEED to overwrite this.

  # The simplex type's name as a string.
  # For instance, 'triangle'.
  simplex_type: str

  # spatial dimensionality of the local reference element
  ndims: int

  # Number of vertices per simplex.
  # For instance, simplex_type == 'triangle' means nverts = 3.
  nverts: int

  # Factor by which the number of elements grows if we perform refinement.
  # For instance, if nref == 3, the element with index 0 is replaced by elements
  # 0, 1, 2 (and so on).
  nref: int

  # A boolean that indicates whether the mesh is piecewise affine.
  # A piecewise affine mesh has a constant metric tensor on each element.
  is_affine: bool

  _items = 'elements', 'points'

  def save(self, filename):
    filename = str(filename)
    assert filename.endswith('.pkl'), 'The filename must end on .pkl'
    with open(filename, 'wb') as file:
      pickle.dump(file, self.tobytes)
    print("Mesh successfully saved under the filename {}.".format(filename))

  @classmethod
  def load(cls, filename):
    with open(filename, 'rb') as file:
      elements, points = pickle.load(file)
    return cls( np.frombuffer(elements, dtype=int).reshape(-1, cls.nverts),
                np.frombuffer(points, dtype=int).reshape(-1, 2) )

  @abstractmethod
  def _refine(self):
    """ Refine the mesh once.
        We assume that the new elements are ordered such that the element with index `i`
        is replaced by [self.nref * i, ..., self.nref * i + self.nref - 1] """
    # each derived class HAS to implement this method
    pass

  def issubmesh(self, other):
    """ Check if self is a submesh of other.
        A submesh is defined as a mesh with a subset or the same vertex indices
        and the same corresponding coordinates.
        If the two meshes are not of the same type, we check if self is
        a submesh of other.submesh and so on. """
    return _issubmesh(self, other)

  def __init__(self, elements: np.ndarray | Sequence[int], points: np.ndarray):
    assert hasattr(self, 'simplex_type') and hasattr(self, 'nverts') and hasattr(self, 'nref'), \
        'Derived classes need to implement their element type and the number of vertices per element as well as the number of refinement elements.'

    self.elements = frozen(elements, dtype=int)

    points = _round_array(points)

    self.points = frozen(points, dtype=float)

    # sanity checks
    assert self.points.shape[1:] == (3,), NotImplementedError("Meshes are assumed to be manifolds in R^3 by default.")
    assert self.elements.shape[1:] == (self.nverts,)
    assert 0 <= self.elements.min() <= self.elements.max() < len(self.points), 'Hanging node detected.'
    assert np.unique(self.elements, axis=0).shape == self.elements.shape, "Duplicate element detected."

    # an array containing the indices of all vertices in self.elements (in sorted order).
    self.vertex_indices = frozen(np.unique(self.elements), dtype=int)

  def lexsort_elements(self):
    """
      Reorder the elements in lexicographical ordering.
      >>> mesh.elements
          [[3, 2, 1], [1, 2, 0]]
      >>> mesh.lexsort_elements().elements
          [[1, 2, 0], [3, 2, 1]]
    """
    shuffle = np.lexsort(self.elements.T[::-1])
    return self._edit(elements=self.elements[shuffle])

  def get_points(self, vertex_indices):
    """
      Same as self.points[vertex_indices] with the difference that it first checks
      if `vertex_indices` is a subset of self.elements.
      The rationale behind this is that the self.points array may contain points
      that are not in self.elements.
    """
    vertex_indices = np.asarray(vertex_indices, dtype=int)
    diff = np.setdiff1d(vertex_indices, self.elements.ravel())
    if len(diff) != 0:
      raise MissingVertexError("Failed to locate the vertices with indices '{}'.".format(diff))
    return self.points[vertex_indices]

  def refine(self, n: int = 1):
    """
      Refine the mesh `n` times.
      Optionally return an array with rows that correspond to the element indices of
      elements that replace the old element.
    """
    assert (n := int(n)) >= 0
    if n == 0:
      ret = self
    else:
      ret = self._refine().refine(n=n-1)
    return ret

  def drop_points_and_renumber(self):
    # XXX: docstring
    unique_vertices = np.unique(self.elements)
    if len(unique_vertices) == unique_vertices[-1] + 1 == len(self.points):
      return self
    points = self.points[unique_vertices]
    map_old_index_new = dict(zip(unique_vertices, count()))
    elements = np.array([map_old_index_new[index] for index in self.elements.ravel()], dtype=int).reshape(self.elements.shape)
    return self._edit(elements=elements, points=points)

  @property
  def _submesh_indices(self):
    # XXX: docstring
    raise HasNoSubMeshError("A mesh of type '{}' has no submesh.".format(self.__class__.__name__))

  def _submesh(self):
    # XXX: docstring
    elements = np.concatenate([ self.elements[:, slce] for slce in self._submesh_indices ])

    sorted_elements = {}
    for elem in map(tuple, elements):
      sorted_elements.setdefault(tuple(sorted(elem)), []).append(elem)

    elements = sorted(map(min, sorted_elements.values()))
    return frozen(elements)

  @cached_property
  def _boundary_nonboundary_elements(self):
    # XXX: docstring

    # get all facets
    all_submesh_facets = np.concatenate([self.elements[:, indices] for indices in self._submesh_indices ])

    # create a unique identifier by sorting the indices
    sorted_edges = np.sort(all_submesh_facets, axis=1)

    # get the unique sorted_edges, their corresponding unique indices and the number of occurences of each
    _, unique_indices, counts = np.unique(sorted_edges, return_index=True, return_counts=True, axis=0)

    # keep only the ones that have been counted only once
    one_mask = counts == 1

    return tuple(map(frozen, (all_submesh_facets[unique_indices[mask]] for mask in (one_mask, ~one_mask))))

  @property
  def boundary(self):
    boundary_elements, _ = self._boundary_nonboundary_elements
    if len(boundary_elements) == 0:
      raise HasNoBoundaryError("The mesh has no boundary.")
    return self._submesh_type(boundary_elements, self.points)

  @cached_property
  def interfaces(self):
    try:
      dself = self.boundary
    except HasNoBoundaryError:
      return self.submesh
    if dself == self.submesh:
      raise NotImplementedError("The mesh has no interfaces.")
    return self.submesh - dself

  def subdivide_elements(self):
    """
      This function converts a mesh of one type to a mesh of another type
      through element subdivision. For instance a quadmesh to a triangulation.
      This function should be overwritten, if applicable.
    """
    raise NotImplementedError

  def __sub__(self, other):
    """
      Subtract `other` from `self` creating a new (usually smaller) mesh.
    """
    # XXX: avoid for loops using numpy
    assert other.__class__ is self.__class__, NotImplementedError()
    other_elems = set(map(lambda x: tuple(sorted(x)), other.elements))

    keep_elems = []
    for ielem, elem in enumerate(map(lambda x: tuple(sorted(x)), self.elements)):
      if elem in other_elems and (self.points[(_elem := list(elem))] == other.points[_elem]).all():
          continue
      keep_elems.append(ielem)

    return self.take(keep_elems)

  def __or__(self, other):
    # from _jit import union_of_two_meshes
    # return union_of_two_meshes(self, other)
    return mesh_union(self, other)

  @property
  def _submesh_type(self):
    raise HasNoSubMeshError("A mesh of type '{}' has no submesh.".format(self.__class__.__name__))

  @cached_property
  def submesh(self):
    return self._submesh_type(self._submesh(), self.points)

  @property
  def pvelements(self):
    return self.elements

  def points_iter(self):
    """
      An iterator that returns the three vertices of each element.

      Example
      -------

      for (a, b, c) in mesh.points_iter():
        # do stuff with vertices a, b and c

    """
    for indices in self.elements:
      yield self.points[indices]

  def plot(self):
    nelems = len(self.elements)
    cell_type = { 'point': vtk.VTK_POINTS,
                  'line': vtk.VTK_LINE,
                  'triangle': vtk.VTK_TRIANGLE,
                  'quadrilateral': vtk.VTK_QUAD,
                  'hexahedron': vtk.VTK_HEXAHEDRON }[self.simplex_type]

    self = self.drop_points_and_renumber()

    points = self.points
    elements = np.concatenate([np.full((nelems, 1), self.nverts, dtype=int), self.pvelements], axis=1).astype(int).copy()

    # for some reason I get segfaults sometimes if I don't copy self.points
    grid = pv.UnstructuredGrid(elements, np.array([cell_type] * nelems), points)
    grid.plot(show_edges=True, line_width=1, color="tan")

  def is_valid(self) -> bool:
    raise NotImplementedError

  @abstractmethod
  def geometry_map(self, ielem):
    """
      A Callable mapping from the reference element to the element
      with index `ielem`.
    """
    pass

  def geometry_map_iter(self):
    for ielem in range(len(self.elements)):
      yield self.geometry_map(ielem)

  @abstractmethod
  def _local_ordinances(self, order):
    # XXX: may potentially be removed
    pass

  def default_ordinances(self, order):
    points = self._local_ordinances(order)
    return tuple(_map(points) for _map in self.geometry_map_iter())

  def take(self, elemindices):
    return self._edit(elements=self.elements[elemindices])

  def take_elements(self, selecter: Callable, complement=False):
    if complement:
      _selecter = selecter
      selecter = lambda *args, **kwargs: not _selecter(*args, **kwargs)
    keep_elements = []
    for ielem, points in enumerate(self.points_iter()):
      if selecter(*points):
        keep_elements.append(ielem)
    return self.take(sorted(set(keep_elements)))


@lru_cache(maxsize=32)
def _refine_structured(self):
  elems, points = ref_structured(self.elements, self.points, self.ndims)
  return self.__class__(elems, points)


class HexMesh(BilinearMixin, Mesh):

  """
              3_______ 7
             /|      /|
           1 _|____ 5 |
           |  |____|__|
           | / 2   | / 6
           |/______|/
           0       4

    `_refine` and `is_valid` are implemented via `BilinearMixin`.
  """

  simplex_type = 'hexahedron'
  ndims = 3
  nverts = 8
  nref = 8
  is_affine = False

  @cached_property
  def _submesh_indices(self):
    return tuple(map(frozen, [ [0, 1, 2, 3],
                               [0, 1, 4, 5],
                               [0, 2, 4, 6],
                               [4, 5, 6, 7],
                               [1, 3, 5, 7],
                               [2, 3, 6, 7], ]))

  @cached_property
  def pvelements(self):
    return self.elements[:, [0, 4, 6, 2, 1, 5, 7, 3]]

  @property
  def _submesh_type(self):
    return QuadMesh

  def _local_ordinances(self, order):
    x = np.linspace(0, 1, order)
    return np.stack(list(map(np.ravel, np.meshgrid(x, x, x))), axis=1)


class QuadMesh(BilinearMixin, Mesh):

  """
     1 _____ 3
      |     |
      |     |
      |_____|
     0       2

    `_refine` and `is_valid` are implemented via `BilinearMixin`.
  """

  simplex_type = 'quadrilateral'
  ndims = 2
  nverts = 4
  nref = 4
  is_affine = False

  @cached_property
  def _submesh_indices(self):
    return tuple(map(frozen, [[0, 1], [0, 2], [2, 3], [1, 3]]))

  @property
  def _submesh_type(self):
    return LineMesh

  def _local_ordinances(self, order):
    x = np.linspace(0, 1, order)
    return np.stack(list(map(np.ravel, np.meshgrid(x, x))), axis=1)

  @cached_property
  def submesh(self):
    return LineMesh(self._submesh(), self.points)

  @cached_property
  def pvelements(self):
    return self.elements[:, [0, 2, 3, 1]]

  def subdivide_elements(self):
    elements = self.elements[:, [[0, 2, 1], [2, 3, 1]]].reshape(-1, 3)
    return Triangulation(elements, self.points)


class Triangulation(AffineMixin, Mesh):

  """
    `is_valid` is implemented via `AffineMixin`.
  """

  simplex_type = 'triangle'
  ndims = 2
  nverts = 3
  nref = 4
  is_affine = True

  @classmethod
  def from_polygon(cls, *args, **kwargs):
    """ See the docstring of `triangulation_from_polygon.` """
    return cls(*triangulation_from_polygon(*args, **kwargs))

  @cached_property
  def _submesh_indices(self):
    return tuple(map(frozen, [[0, 1], [1, 2], [0, 2]]))

  @property
  def _submesh_type(self):
    return LineMesh

  def _refine(self):
    """ See `_refine_Triangulation`. """
    return _refine_Triangulation(self)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @frozen_cached_property
  def BK(self):
    """
      Jacobi matrix per element of shape (nelems, 2, 2).
      mesh.BK[i, :, :] or, in short, mesh.BK[i] gives
      the Jacobi matrix corresponding to the i-th element.

      Example
      -------

      for i, BK in enumerate(mesh.BK):
        # do stuff with the Jacobi matrix BK corresponding to the i-th element.
    """
    a, b, c = self.points[self.elements.T]
    return np.stack([b - a, c - a], axis=2)

  def _local_ordinances(self, order):
    return _round_array(np.stack([[i, j] for i, j, k in product(*[range(order+1)]*3) if i + j + k == order ]) / order, 12)

  def default_ordinances(self, order):
    loc_ords = self._local_ordinances(order)
    return tuple( frozen(_round_array(loc_ords @ BK.T + a[_])) for (a, b, c), BK in zip(self.points_iter(), self.BK) )

  @cached_property
  def submesh(self):
    return LineMesh(self._submesh(), self.points)

  def geometry_map(self, ielem):
    # XXX: this should not be implemented by hand but should simply follow from
    #      a tri / tet - based abstraction
    p0, p1, p2 = map(lambda x: x[_], self.points[self.elements[ielem]])

    def gmap(x, p0=p0):
      return p0 + x @ self.BK[ielem].T

    return gmap


def abs_tuple(tpl):
  """
    [5, 6] -> (5, 6)
    [6, 5] -> (5, 6)
    (6, 5) -> (5, 6)
  """
  a, b = tpl
  if a > b: return b, a
  return tuple(tpl)


@lru_cache
def _refine_Triangulation(mesh: Triangulation) -> Triangulation:
  """
    Uniformly refine the entire mesh once.
          i2                             i2
          / \                            / \
         /   \                          /   \
        /     \         becomes      i20 ---- i12
       /       \                      / \   /  \
      /         \                    /   \ /    \
    i0 --------- i1                i0 -- i01 --- i1

    Returns
    -------
    The refined mesh of class `Triangulation`
  """

  # XXX: JIT compile

  points = mesh.points
  maxindex = len(points)
  slices = np.array([[0, 1], [1, 2], [2, 0]])
  all_edges = list(set(map(abs_tuple, np.concatenate(mesh.elements[:, slices]))))
  newpoints = points[np.array(all_edges)].sum(1) / 2
  map_edge_number = dict(zip(all_edges, count(maxindex)))

  triangles = []
  for tri in mesh.elements:
    i01, i12, i20 = [map_edge_number[edge] for edge in map(abs_tuple, tri[slices])]
    i0, i1, i2 = tri
    triangles.extend([
        [i0, i01, i20],
        [i01, i1, i12],
        [i01, i12, i20],
        [i20, i12, i2]
    ])

  elements = np.array(triangles, dtype=int)
  points = np.concatenate([points, newpoints])

  return Triangulation(elements, points)


def triangulation_from_polygon(points: np.ndarray, mesh_size=0.05) -> Triangulation:
  """
    create :class: ``Triangulation`` mesh from ordered set of boundary
    points.

    parameters
    ----------
    points: Array-like of shape points.shape == (npoints, 2) of boundary
            points ordered in counter-clockwise direction.
            The first point need not be repeated.
    mesh_size: Numeric value determining the density of cells.
               Smaller values => denser mesh.
      Can alternatively be a function of the form
        mesh_size = lambda dim, tag, x, y, z, _: target mesh size as function of x and y.
      For instance, mesh_size = lambda ... : 0.1 - 0.05 * np.exp(-20 * ((x - .5)**2 + (y - .5)**2))
      creates a denser mesh close to the point (x, y) = (.5, .5).
  """

  import pygmsh

  if np.isscalar(mesh_size):
    _mesh_size = mesh_size
    mesh_size = lambda *args, **kwargs: _mesh_size

  assert isinstance(type(mesh_size), Callable)

  points = np.asarray(points)
  assert points.shape[1:] == (2,)

  with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(points)
    geom.set_mesh_size_callback(mesh_size)
    mesh = geom.generate_mesh(algorithm=5)

  return mesh.cells_dict['triangle'], mesh.points


class LineMesh(AffineMixin, Mesh):

  simplex_type = 'line'
  ndims = 1
  nverts = 2
  nref = 2
  is_affine = True

  @cached_property
  def _submesh_indices(self):
    return (frozen([0]), frozen([1]))

  @property
  def _submesh_type(self):
    return PointMesh

  def _refine(self):
    return _refine_structured(self)

  @frozen_cached_property
  def BK(self):
    a, b = self.points[self.elements.T]
    return (b - a)[..., _]

  def _local_ordinances(self, order):
    return _round_array(np.linspace(0, 1, order+1))

  def default_ordinances(self, order):
    loc_ords = self._local_ordinances(order)
    a, b = self.points[self.elements.T]
    return tuple(map(frozen, (_round_array(_a[_] * (1 - loc_ords[:, _]) + _b[_] * loc_ords[:, _]) for _a, _b in zip(a, b))))

  @cached_property
  def submesh(self):
    return PointMesh(self._submesh(), self.points)

  def geometry_map(self, ielem):
    # XXX: this should not be implemented by hand but should simply follow from
    #      a tri / tet - based abstraction
    p0, p1 = map(lambda x: x[_], self.points[self.elements[ielem]])

    def gmap(x, p0=p0, p1=p1):
      return p0 * (1 - x) + p1 * x

    return gmap


class PointMesh(AffineMixin, Mesh):

  simplex_type = 'point'
  ndims = 0
  nverts = 1
  nref = 1
  is_affine = True

  def _refine(self):
    return self

  @frozen_cached_property
  def BK(self):
    return np.ones(len(self.elements)).reshape(-1, 1)

  def _local_ordinances(self, order):
    return np.zeros((1, 2))

  def default_ordinances(self, order):
    return self.points

  def plot(self):
    point_cloud = pv.PolyData(self.points[self.elements.ravel()])
    point_cloud.plot(eye_dome_lighting=True)

  def geometry_map(self, ielem):
    # XXX: this should not be implemented by hand but should simply follow from
    #      a tri / tet - based abstraction
    p0 = self.points[self.elements[ielem].ravel()][_]

    def gmap(x, p0=p0):
      return p0 + 0 * x

    return gmap


def unitsquare(_points: Sequence[int | np.ndarray]):

  assert (dim := len(_points)) <= 3, NotImplementedError

  # format to linspace if not already and make sure strictly monotone
  points = []
  for elem in _points:

    if np.isscalar(elem):
      elem = np.linspace(0, 1, elem)

    assert isincreasing((elem := np.asarray(elem, dtype=float)))
    points.append(elem)

  # keep track of the length of each vector of abscissae for later reshape
  lengths = list(map(len, points))

  # convert to meshgrid and augment by zeros if necessary
  points = np.stack(list(map(np.ravel, np.meshgrid(*points, indexing='ij'))), axis=-1)
  if points.shape[1] != 3:
    points = np.concatenate([points, np.zeros((points.shape[0], 3 - points.shape[1]))], axis=-1)

  element = np.asarray(list(product(*map(range, (2,)*dim)))).T
  indices = np.arange(len(points)).reshape(*lengths)
  ijk = np.stack(list(map(np.ravel, np.meshgrid(*(np.arange(n-1) for n in lengths), indexing='ij'))), axis=0)

  # 3, nelems, 8
  ijk = ijk[:, :, _] + element[:, _]

  elements = indices[ tuple(ijk) ]

  return {1: LineMesh, 2: QuadMesh, 3: HexMesh}[dim](elements, points)
