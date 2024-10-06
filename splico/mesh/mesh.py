#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jochen Hinz
"""

from ..util import np, _, frozen, HashMixin, frozen_cached_property, \
                   _round_array, round_result, isincreasing, flat_meshgrid, freeze

from ._jit_ref_structured import _refine_structured as ref_structured
from .pol import _eval_nd_polynomial_local

import pyvista as pv
import vtk
import treelog as log

from abc import abstractmethod

from typing import Callable, Sequence, Self, Tuple

from functools import cached_property, lru_cache
from itertools import count, product

import pickle


# all supported element types (for now)
SIMPLEX_TYPES = ('point', 'line', 'triangle',
                 'quadrilateral', 'tetrahedron', 'hexahedron')


class MissingVertexError(Exception):
  pass


class HasNoSubMeshError(Exception):
  pass


class HasNoBoundaryError(Exception):
  pass


@lru_cache
def _issubmesh(mesh0, mesh1):
  """
    Check if `mesh0` is a submesh of `mesh1`.
    A submesh is defined as a mesh that contains the same or a subset of the
    other mesh's points and elements. Alternatively, `mesh0` is also considered
    a submesh of `mesh1` if it is a submesh of `mesh1.submesh` or its submeshes.

    Parameters
    ----------
    mesh0 : :class:`Mesh`
        The submesh candidate.
    mesh1 : :class:`Mesh`
        The mesh we check if `mesh0` is a submesh of.

    Returns
    -------
    A boolean indicating whether `mesh0` is a submesh of `mesh1`.
  """

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
    # XXX: note that two elements can be the same even though the indices appear
    #      in a different order.
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
  """ Docstring: see _jit.py """
  from ._jit import multi_mesh_boundary_union as _mesh_boundary_union
  return _mesh_boundary_union(*meshes, **kwargs)


def mesh_union(*meshes):
  """ Docstring: see _jit.py """
  from ._jit import mesh_union as _mesh_union
  return _mesh_union(*meshes)


def mesh_difference(mesh0, mesh1):
  """ Take the difference of two meshes. """
  from ._jit import mesh_difference as _mesh_difference
  return _mesh_difference(mesh0, mesh1)


# XXX: write another parent class that allows for mixed element types.
#      The meshes that have already been implemented are then special cases
#      with only one element type.


class BilinearMixin:

  """
    Mixin for bilinear mesh types (QuadMesh, HexMesh).
    Provides implementations for the geometry map as well as its jacobian
    per element.
    Furthermore provides an implementation of `is_valid` using the elementwise
    Jacobian evaluated in the vertices.
    Finally provides a standard implementation of `_refine`.
  """

  @freeze
  @round_result
  def _local_ordinances(self, order):
    assert (order := int(order)) > 0
    x = np.linspace(0, 1, order+1)
    return flat_meshgrid(*[x] * self.ndims, axis=1)

  def _refine(self) -> Self:
    return _refine_structured(self)


class AffineMixin:

  """
    Mixin for affine mesh types (PointMesh, LineMesh, Triangulation).
    Provides implementations for the elementwise metric tensor (GK),
    its inverse (GKinv) the elementwise constant measure (detBK)
    and a function to check for validty (is_valid).

    Each derived class needs to implement `BK` in order for the others to work.
  """

  @freeze
  @round_result
  def _local_ordinances(self, order):
    ret = BilinearMixin._local_ordinances(self, order)
    return ret[ ret.sum(1) < 1 + 1e-8 ]


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

  @cached_property
  def active_indices(self):
    return np.unique(self.elements)

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
      return self
    return self._refine().refine(n=n-1)

  def drop_points_and_renumber(self):
    """
      Drop all points that are not used by `self.elements` and renumber
      `self.elements` to reflect the renumbering of the points from 0 to npoints-1.
    """
    unique_vertices = np.unique(self.elements)
    if len(unique_vertices) == unique_vertices[-1] + 1 == len(self.points):
      return self
    points = self.points[unique_vertices]
    elements = np.searchsorted(unique_vertices, self.elements)
    return self._edit(elements=elements, points=points)

  def JK(self, points):
    return np.stack([self.eval_local(points, dx=_dx) for _dx in np.eye(self.ndims)], axis=-1)

  def GK(self, points):
    JK = self.JK(points)
    return (JK.swapaxes(-1, -2)[..., _] * JK[..., _, :, :]).sum(-2)

  def is_valid(self, order=1) -> bool:
    points = self._local_ordinances(order)
    return (np.linalg.det(self.GK(points)) > 0).all()

  @frozen_cached_property
  def _pol_weights(self):
    """
      Polynomial weights of each element's map.
      For `self.eval_local`.
    """
    basis_funcs = self._basis_funcs

    # get the element-wise weights in tensorial layout
    # shape: (2 ** self.ndims, nelems, 3)
    elementwise_weights = self.points[self.elements.T]

    # (1, ..., 1, 2 ** ndims, nelems, 3) and (2, ..., 2, 2 **ndims, 1, 1 )
    # becomes (2, ..., 2, 2 ** ndims, nelems, 3).sum(-3) == (2, ..., 2, nelems, 3)
    return (elementwise_weights[(_,) * self.ndims] * basis_funcs[..., _, _]).sum(-3)

  @frozen_cached_property
  def _basis_funcs(self):
    """
      The polynomial weights of the nodal basis functions in the reference
      element.
      Shape: (2,) * self.ndims + (nverts,)
    """
    ords = self._local_ordinances(1).astype(int)
    # set up the matrix we need to solve
    X = np.stack([ np.multiply.reduce([_x ** i for _x, i in zip(ords.T, multi_index)])
                   for multi_index in ords ], axis=1)

    # solve for the nodal basis function's polynomial weights
    # and reshape them to tensorial (nfuncs, x, y, z, ...) shape

    # shape: (2 ** self.ndims, *(2,) * self.ndims)
    # (basis_f_index, 2, 2, ...)
    basis_funcs = np.zeros((*(2,) * self.ndims, X.shape[0]), dtype=float)
    basis_funcs[*ords.T] = np.linalg.solve(X, np.eye(X.shape[0]))
    return basis_funcs

  def eval_local(self, points: np.ndarray, dx=None) -> np.ndarray:
    """ Evaluate each element map locally in `points`. """
    pol_weights = self._pol_weights

    # pol_weights.shape == (2, 2, ..., nelems, 3)
    # _eval_nd_polynomial_local(pol_weights, points, dx=dx).shape == (nelems, 3, npoints)
    # we reshape to (nelems, npoints, 3)
    # XXX: try to avoid swapaxes
    return _eval_nd_polynomial_local(pol_weights, points, dx=dx).swapaxes(-1, -2)

  @property
  def _submesh_indices(self) -> Tuple[np.ndarray]:
    """
      The submesh indices are the columns of `self.elements` that have to be
      extracted to create the mesh's submesh. By default returns a `HasNoSubMeshError`
      but can be overwritten.
      Example: to go from a triangulation to a linmesh, we have to extract the
      columns ([0, 1], [1, 2], [2, 0]) (all edges of each triangle).
    """
    raise HasNoSubMeshError("A mesh of type '{}' has no submesh.".format(self.__class__.__name__))

  def _submesh(self):
    """
      Take the mesh's submesh, if applicable.
      Requres `self._submesh_indices` to tbe implemented by the class.
    """
    # XXX: jit-compile the element map

    elements = np.concatenate([ self.elements[:, slce] for slce in self._submesh_indices ])

    # iterate over all elements and map the sorted element indices to the element.
    sorted_elements = {}
    for elem in map(tuple, elements):
      sorted_elements.setdefault(tuple(sorted(elem)), []).append(elem)

    # for each list of equivalent elements (differing only by a permutation)
    # retain the minimum one. Example [(2, 3, 1), (1, 2, 3)] -> (1, 2, 3)
    elements = sorted(map(min, sorted_elements.values()))
    return frozen(elements)

  @cached_property
  def _boundary_nonboundary_elements(self):
    """
      Split all elements of the submesh into boundary and non-boundary elements.
    """

    # get all facets
    all_submesh_facets = np.concatenate([self.elements[:, indices] for indices in self._submesh_indices ])

    # create a unique identifier by sorting the indices
    sorted_edges = np.sort(all_submesh_facets, axis=1)

    # get the unique sorted_edges, their corresponding unique indices and the number of occurences of each
    _, unique_indices, counts = np.unique(sorted_edges, return_index=True, return_counts=True, axis=0)

    # keep the ones that have been counted only once
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
    assert self.__class__ is other.__class__, NotImplementedError
    return mesh_union(self, other)

  @property
  def _submesh_type(self):
    raise HasNoSubMeshError("A mesh of type '{}' has no submesh.".format(self.__class__.__name__))

  @cached_property
  def submesh(self):
    return self._submesh_type(self._submesh(), self.points)

  @property
  def pvelements(self):
    """
      PyVista may expect the elements in an order that differs from the chosen
      order. May need to be overwritten in order to properly work with PyVista.
    """
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
    """
      Given a Callable `selecter` iterator over all of the mesh's elements
      and keep only those elements for which selecter(*points) evaluates
      to `True`, where `points` are the element's points.
    """
    # XXX: re-implement `selecter` to apply to all elements at once to allow
    #      for proper vectorization (avoid element for loop).
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
  """ Refine a structured mesh (LineMesh, QuadMesh, HexMesh) """
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

  @cached_property
  def pvelements(self):
    return self.elements[:, [0, 2, 3, 1]]

  def subdivide_elements(self):
    elements = self.elements[:, [[0, 2, 1], [2, 3, 1]]].reshape(-1, 3)
    return Triangulation(elements, self.points)


class Triangulation(AffineMixin, Mesh):

  """
    `is_valid` is implemented via `AffineMixin`.

      1
      |\
      |  \
      |    \
      |      \
      |        \
      |__________\
     0            2
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

  @cached_property
  def pvelements(self):
    return self.elements[:, [0, 2, 1]]

  def _refine(self):
    """ See `_refine_Triangulation`. """
    return _refine_Triangulation(self)

  def default_ordinances(self, order):
    loc_ords = self._local_ordinances(order)
    return tuple( frozen(_round_array(loc_ords @ BK.T + a[_])) for (a, b, c), BK in zip(self.points_iter(), self.BK) )


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
          i1                             i1
          / \                            / \
         /   \                          /   \
        /     \         becomes      i10 ---- i21
       /       \                      / \   /  \
      /         \                    /   \ /    \
    i0 --------- i2                i0 -- i02 --- i2

    Returns
    -------
    The refined mesh of class `Triangulation`
  """

  # XXX: JIT compile

  points = mesh.points
  maxindex = len(points)
  slices = np.array([[0, 2], [2, 1], [1, 0]])
  all_edges = list(set(map(abs_tuple, np.concatenate(mesh.elements[:, slices]))))
  newpoints = points[np.array(all_edges)].sum(1) / 2
  map_edge_number = dict(zip(all_edges, count(maxindex)))

  triangles = []
  for tri in mesh.elements:
    i02, i21, i10 = [map_edge_number[edge] for edge in map(abs_tuple, tri[slices])]
    i0, i1, i2 = tri
    triangles.extend([
        [i0, i02, i10],
        [i02, i2, i21],
        [i02, i21, i10],
        [i10, i21, i1]
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

  try:
    import pygmsh
  except ModuleNotFoundError as ex:
    log.warning("The pygmsh library has not been found. Please install it via 'pip install pygmsh'.")
    raise ModuleNotFoundError from ex

  if np.isscalar((_mesh_size := mesh_size)):
    mesh_size = lambda *args, **kwargs: _mesh_size

  assert isinstance(type(mesh_size), Callable)

  points = np.asarray(points)
  assert points.shape[1:] == (2,)

  with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(points)
    geom.set_mesh_size_callback(mesh_size)
    mesh = geom.generate_mesh(algorithm=5)

  return mesh.cells_dict['triangle'][:, [0, 2, 1]], mesh.points


class LineMesh(AffineMixin, Mesh):

  simplex_type = 'line'
  ndims = 1
  nverts = 2
  nref = 2
  is_affine = True

  def _refine(self):
    return _refine_structured(self)

  @cached_property
  def _submesh_indices(self):
    return (frozen([0]), frozen([1]))

  @property
  def _submesh_type(self):
    return PointMesh

  def _local_ordinances(self, order):
    return _round_array(np.linspace(0, 1, order+1)[:, _])

  def default_ordinances(self, order):
    loc_ords = self._local_ordinances(order)
    a, b = self.points[self.elements.T]
    return tuple(map(frozen, (_round_array(_a[_] * (1 - loc_ords[:, _]) + _b[_] * loc_ords[:, _]) for _a, _b in zip(a, b))))


class PointMesh(Mesh):

  simplex_type = 'point'
  ndims = 0
  nverts = 1
  nref = 1
  is_affine = True

  def _refine(self):
    return self

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

  def is_valid(self, ischeme=None):
    return True


def rectilinear(_points: Sequence[int | np.ndarray]):
  """
    Rectilinear mesh in one, two or three dimensions.

    Parameters
    ----------
    _points : Sequence of integers or flat and strictly monotone np.ndarray.
        The dimensionality of the mesh follows from the length of the sequence.
        If an integer is encountered, it is converted to a linspace where the
        integer determines the number of steps.
        Else, it is assumed to be strictly monotone.

    Returns
    -------
    ret : :class:`LineMesh` or :class:`QuadMesh` or :class:`HexMesh`
        A rectilinear mesh whose dimensionality follows from the length of `_points`.
        The mesh vertices follow from a tensor product of all values generated
        by the conversion of `_points`.
  """

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

  return {1: LineMesh,
          2: QuadMesh,
          3: HexMesh}[dim](elements, points)
