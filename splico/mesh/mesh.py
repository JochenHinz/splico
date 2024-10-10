#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jochen Hinz
"""

from ..util import np, _, frozen, HashMixin, _round_array, round_result, \
                   isincreasing, flat_meshgrid, freeze, frozen_cached_property, \
                   augment_by_zeros
from ._refine import refine_structured, _refine_Triangulation
from .pol import eval_nd_polynomial_local
from .bool import _issubmesh, mesh_boundary_union, mesh_union, mesh_difference
from .aux import MissingVertexError, HasNoSubMeshError, HasNoBoundaryError

from abc import abstractmethod
from typing import Callable, Sequence, Self, Tuple
from functools import cached_property
from itertools import product
import pickle

import pyvista as pv
import vtk
import treelog as log
from numpy.typing import NDArray


# all supported element types (for now)
simplex_types = ('point', 'line', 'triangle',
                 'quadrilateral', 'tetrahedron', 'hexahedron')


# XXX: write another parent class that allows for mixed element types.
#      the meshes that have already been implemented are then special cases
#      with only one element type.


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

  # Spatial dimensionality of the local reference element
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

  def save(self, filename: str):
    filename = str(filename)
    assert filename.endswith('.pkl'), 'The filename must end on .pkl'
    with open(filename, 'wb') as file:
      pickle.dump(file, self.tobytes)
    print("Mesh successfully saved under the filename {}.".format(filename))

  @classmethod
  def load(cls, filename: str):
    with open(filename, 'rb') as file:
      elements, points = pickle.load(file)
    return cls( np.frombuffer(elements, dtype=int).reshape(-1, cls.nverts),
                np.frombuffer(points, dtype=int).reshape(-1, 3) )

  def __init__(self, elements: NDArray[np.int_] | Sequence[Sequence[int]], points: NDArray[np.float_]):
    assert hasattr(self, 'simplex_type') and hasattr(self, 'nverts') and hasattr(self, 'nref'), \
        'Derived classes need to implement their element type and the number of' \
        ' vertices per element as well as the number of refinement elements.'

    self.elements = frozen(elements, dtype=int)
    points = _round_array(points)
    self.points = frozen(points, dtype=float)

    # sanity checks
    assert self.points.shape[1:] == (3,), \
      NotImplementedError("Meshes are assumed to be manifolds in R^3 by default.")
    assert self.elements.shape[1:] == (self.nverts,)
    assert 0 <= self.elements.min() <= self.elements.max() < len(self.points), 'Hanging node detected.'
    assert np.unique(self.elements, axis=0).shape == self.elements.shape, "Duplicate element detected."

  def __repr__(self):
    return f"{self.__class__.__name__}[nelems: {len(self.elements)}, npoints: {len(self.points)}]"

  @abstractmethod
  def _refine(self):
    """
      Refine the entire mesh once.
      We assume that the new elements are ordered such that the element with index `i`
      is replaced by [self.nref * i, ..., self.nref * i + self.nref - 1]
    """
    # each derived class HAS to implement this method
    pass

  @abstractmethod
  def _local_ordinances(self, order: int):
    """
      Abstract method of the mesh's local ordinances.
      Given `order >= 1`, the local ordinances refer to the nodal points inside
      the mesh's reference element of a Lagrangian basis of order `order`.
      For `order == 1` this should default to the reference element's vertices.
    """
    pass

  def issubmesh(self, other: 'Mesh') -> bool:
    """
      Check if self is a submesh of other.
      A submesh is defined as a mesh with a subset or the same vertex indices
      and the same corresponding coordinates.
      If the two meshes are not of the same type, we check if self is
      a submesh of other.submesh and so on.
    """
    return _issubmesh(self, other)

  @frozen_cached_property
  def active_indices(self) -> NDArray[np.int_]:
    """ The indices of the points in self.points that are used in self.elements """
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

  def get_points(self, vertex_indices: Sequence[int] | NDArray[np.int_]) -> NDArray[np.int_]:
    """
      Same as self.points[vertex_indices] with the difference that it first checks
      if `vertex_indices` is a subset of self.elements.
      The rationale behind this is that the self.points array may contain points
      that are not in self.elements.
    """
    vertex_indices = np.asarray(vertex_indices, dtype=int)
    diff = np.setdiff1d(vertex_indices, self.active_indices)
    if len(diff) != 0:
      raise MissingVertexError("Failed to locate the vertices with indices '{}'.".format(diff))
    return self.points[vertex_indices]

  def refine(self, n: int = 1) -> Self:
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
    unique_vertices = self.active_indices
    if len(unique_vertices) == unique_vertices[-1] + 1 == len(self.points):
      return self
    points = self.points[unique_vertices]
    elements = np.searchsorted(unique_vertices, self.elements)
    return self._edit(elements=elements, points=points)

  def JK(self, points: NDArray[np.float_]) -> NDArray[np.float_]:
    """ Evaluation of the jacobian per element. """
    return np.stack([self.eval_local(points, dx=_dx)
                     for _dx in np.eye(self.ndims, dtype=int)], axis=-1)

  def GK(self, points: NDArray[np.float_]) -> NDArray[np.float_]:
    """ Evaluation of the metric tensor per element. """
    JK = self.JK(points)
    return (JK.swapaxes(-1, -2)[..., _] * JK[..., _, :, :]).sum(-2)

  def is_valid(self, order: int = 1, thresh=1e-8) -> bool:
    """
      Check if a mesh is valid.
      These are some standard checks that work for all mesh types.
      Can be overwritten for mesh-type specific validty checks.
    """
    points = self._local_ordinances(order)
    if self.ndims == 3:
      return (np.linalg.det(self.JK(points)) > 0).all()
    if self.ndims == 2:
      JK = self.JK(points)
      # shape (nelems, npoints, 3)
      crosses = np.cross(JK[..., 0], JK[..., 1])
      # first check if there are zeros
      if (np.linalg.norm(crosses, axis=-1) < thresh).any():
        return False
      if self.is_affine:
        # No ? Then check if the orientation of an element changes
        roots = crosses[:, :1]  # (nelems, 1, 3)
        # inner product negative ? => orientation change detected
        return ((crosses[:, 1:] * roots).sum(-1) > 0).all()
      else:
        # XXX: add another validity check for multilinear meshes here
        #      for now just do the metric tensor check
        pass
    # for a 1D mesh it's enough to check if the metric tensor is SPD
    return (np.linalg.det(self.GK(points)) > 0).all()

  def eval_local(self, points: NDArray[np.float_], dx=None) -> NDArray[np.float_]:
    """ Evaluate each element map locally in `points`. """

    # eval_nd_polynomial_local(self, points, dx=dx).shape == (nelems, 3, npoints)
    # we reshape to (nelems, npoints, 3)

    # XXX: try to avoid swapaxes to maintain contiguous memory layout
    return eval_nd_polynomial_local(self, points, dx=dx).swapaxes(-1, -2)

  @property
  def _submesh_indices(self) -> Tuple[NDArray[np.int_]]:
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
      Requres `self._submesh_indices` to be implemented by the class.
    """
    # XXX: jit-compile the element map

    elements = np.concatenate([ self.elements[:, list(slce)] for slce in self._submesh_indices ])

    # iterate over all elements and map the sorted element indices to the element.
    sorted_elements = {}
    for elem in map(tuple, elements):
      sorted_elements.setdefault(tuple(sorted(elem)), []).append(elem)

    # for each list of equivalent elements (differing only by a permutation)
    # retain the minimum one. Example [(2, 3, 1), (1, 2, 3)] -> (1, 2, 3)
    elements = sorted(map(min, sorted_elements.values()))
    return frozen(elements)

  @property
  def _submesh_type(self):
    raise HasNoSubMeshError(f"A mesh of type '{self.__class__.__name__}' has no submesh.")

  @cached_property
  def submesh(self):
    return self._submesh_type(self._submesh(), self.points)

  @cached_property
  def _boundary_nonboundary_elements(self):
    """
      Split all elements of the submesh into boundary and non-boundary elements.
    """

    # get all facets
    all_facets = np.concatenate([ self.elements[:, indices]
                                   for indices in self._submesh_indices ])

    # create a unique identifier by sorting the indices
    sorted_facets = np.sort(all_facets, axis=1)

    # get the unique sorted_edges, their corresponding unique indices and the
    # number of occurences of each
    _, unique_indices, counts = np.unique(sorted_facets, return_index=True,
                                                         return_counts=True,
                                                         axis=0)

    # keep the ones that have been counted only once
    one_mask = counts == 1

    return tuple(map(frozen, (all_facets[unique_indices[mask]] for mask in (one_mask, ~one_mask))))

  @property
  def boundary(self):
    boundary_elements, _ = self._boundary_nonboundary_elements
    if len(boundary_elements) == 0:
      raise HasNoBoundaryError("The mesh has no boundary.")
    return self._submesh_type(boundary_elements, self.points)

  def subdivide_elements(self):
    """
      This function converts a mesh of one type to a mesh of another type
      through element subdivision. For instance a quadmesh to a triangulation.
      This function should be overwritten, if applicable.
      Note that subdivision may not preserve the geometry exactly.
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

  @cached_property
  def interfaces(self):
    try:
      dself = self.boundary
    except HasNoBoundaryError:
      return self.submesh
    if dself == self.submesh:
      raise NotImplementedError("The mesh has no interfaces.")
    return self.submesh - dself

  @property
  def pvelements(self):
    """
      PyVista may expect the elements in an order that differs from the chosen
      order. May need to be overwritten in order to properly work with PyVista.
    """
    return self.elements

  def points_iter(self):
    """
      An iterator that returns the vertices of each element.

      Example
      -------
      For a triangulation:

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
    elements = np.concatenate([np.full((nelems, 1), self.nverts, dtype=int), self.pvelements], axis=1).astype(int)

    # for some reason I get segfaults sometimes if I don't copy self.points
    grid = pv.UnstructuredGrid(elements, np.array([cell_type] * nelems), points)
    grid.plot(show_edges=True, line_width=1, color="tan")

  def take(self, elemindices: Sequence[int] | NDArray[np.int_]):
    return self._edit(elements=self.elements[np.asarray(elemindices, dtype=int)])

  def take_elements(self, selecter: Callable, complement: bool = False):
    """
      Given a Callable `selecter` iterate over all of the mesh's elements
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


class MultilinearMesh(Mesh):
  """
    Derived class providing implementations for the `_local_ordinances` which is an
    abstract method of the `mesh` base class.
    Additionally, provides an implementation of the `_refine` abstract method
    which is accomplished using the _refine_structured routine, which works
    for multilinear mesh types of any dimensionality.
    Note that a one-dimensional multilinear mesh is simultaneously affine.
  """

  @freeze
  @round_result
  def _local_ordinances(self, order: int) -> NDArray[np.float_]:
    assert (order := int(order)) > 0
    x = np.linspace(0, 1, order+1)
    return flat_meshgrid(*[x] * self.ndims, axis=1)

  def _refine(self) -> Self:
    return refine_structured(self)


class AffineMesh(Mesh):
  """
    Mixin for affine mesh types.
    Provides an implementation for the _local_ordinances abstract method.
  """
  # XXX: currently affine meshes require special-tailored refinement methods.
  #      Write a method that can refine any affine mesh type, similar to
  #      _refine_structured. This should be possible by taking the same approach
  #      as in `MultilinearMixin` while restricting the attention to the plane
  #      x + y + z <= 1

  @freeze
  @round_result
  def _local_ordinances(self, order: int) -> NDArray[np.float_]:
    active_indices = \
        [i for i, mi in enumerate(product(*[range(2)]*self.ndims)) if sum(mi) <= 1]
    return MultilinearMesh._local_ordinances(self, order)[active_indices]

  def _refine(self) -> Self:
    raise NotImplementedError('Every affine mesh type needs to implement its own'
                              ' refinement method.')


class HexMesh(MultilinearMesh):

  """
              3_______ 7
             /|      /|
           1 _|____ 5 |
           |  |____|__|
           | / 2   | / 6
           |/______|/
           0       4

    `_refine` `_local_ordinances` are implemented via `MultilinearMixin`.
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

  @frozen_cached_property
  def pvelements(self):
    return self.elements[:, [0, 4, 6, 2, 1, 5, 7, 3]]

  @property
  def _submesh_type(self):
    return QuadMesh


class QuadMesh(MultilinearMesh):

  """
     1 _____ 3
      |     |
      |     |
      |_____|
     0       2

    `_refine` `_local_ordinances` are implemented via `MultilinearMixin`.
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

  @frozen_cached_property
  def pvelements(self):
    return self.elements[:, [0, 2, 3, 1]]

  def subdivide_elements(self):
    elements = self.elements[:, [[0, 1, 2], [2, 1, 3]]].reshape(-1, 3)
    return Triangulation(elements, self.points)


class Triangulation(AffineMesh):

  """
      1
      |\
      |  \
      |    \
      |      \
      |        \
      |__________\
     0            2

    `_local_ordinances` is implemented via `AffineMixin`
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
    return tuple(map(frozen, [[0, 2], [2, 1], [1, 0]]))

  @property
  def _submesh_type(self):
    return LineMesh

  @frozen_cached_property
  def pvelements(self):
    return self.elements[:, [0, 2, 1]]

  def _refine(self):
    """ See `_refine_Triangulation`. """
    return _refine_Triangulation(self)


def triangulation_from_polygon(points: NDArray[np.int_ | np.float_], mesh_size: float | int | Callable = 0.05):
  """
    create :class: ``Triangulation`` mesh from ordered set of boundary
    points.

    Parameters
    ----------
    points : Array-like of shape (npoints, 2)
        boundary points ordered in counter-clockwise direction.
        The first point need not be repeated.
    mesh_size : :class:`float` or :class:`int` or Callable
        Numeric value determining the density of cells.
        Smaller values => denser mesh.
        Can alternatively be a function of the form
        mesh_size = lambda dim, tag, x, y, z, _: target mesh size as a
        function of x and y.

        For instance,

        >>> mesh_size = lambda ... : 1-0.5*np.exp(-20*((x-.5)**2+(y-.5)**2))

        creates a denser mesh close to the point (x, y) = (.5, .5).

    Returns
    -------
    elements : :class:`np.ndarray[int, 3]`
        The mesh's element indices.
    points : :class:`np.ndarray[float, 3]`
        The mesh's points.
  """
  try:
    import pygmsh
  except ModuleNotFoundError as ex:
    log.warning("The pygmsh library has not been found. "
                "Please install it via 'pip install pygmsh'.")
    raise ModuleNotFoundError from ex

  if np.isscalar((_mesh_size := mesh_size)):
    mesh_size = lambda *args, **kwargs: _mesh_size

  points = np.asarray(points)
  assert points.shape[1:] == (2,)

  with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(points)
    geom.set_mesh_size_callback(mesh_size)
    mesh = geom.generate_mesh(algorithm=5)

  return mesh.cells_dict['triangle'][:, [0, 2, 1]], mesh.points


# XXX: TetMesh. Leave as an exercise for Fabio.


class LineMesh(AffineMesh):

  simplex_type = 'line'
  ndims = 1
  nverts = 2
  nref = 2
  is_affine = True

  def _refine(self):
    return refine_structured(self)

  @cached_property
  def _submesh_indices(self):
    return (frozen([0]), frozen([1]))

  @property
  def _submesh_type(self):
    return PointMesh


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

  def plot(self):
    point_cloud = pv.PolyData(self.points[self.elements.ravel()])
    point_cloud.plot(eye_dome_lighting=True)

  def is_valid(self, ischeme=None):
    return True


def rectilinear(_points: Sequence) -> LineMesh | QuadMesh | HexMesh:
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
  all_points = augment_by_zeros(flat_meshgrid(*points, axis=1), axis=1)

  element = np.asarray(list(product(*map(range, (2,)*dim)))).T
  indices = np.arange(len(all_points)).reshape(*lengths)
  ijk = flat_meshgrid(*(np.arange(n-1) for n in lengths), axis=0)

  # 3, nelems, 8
  ijk = ijk[:, :, _] + element[:, _]

  elements = indices[ tuple(ijk) ]

  return {1: LineMesh,
          2: QuadMesh,
          3: HexMesh}[dim](elements, all_points)
