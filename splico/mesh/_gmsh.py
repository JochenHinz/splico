"""
Gmsh routines for creating mesh types.
@author: Jochen Hinz
"""

from ..util import np
from ..types import FloatArray
from ..log import logging as log

from typing import Callable


try:
  import pygmsh
except ModuleNotFoundError as ex:
  log.warning("The pygmsh library has not been found. "
              "Please install it via 'pip install pygmsh'.")
  raise ModuleNotFoundError from ex


def triangulation_from_polygon(points: FloatArray, mesh_size: float | int | Callable = 0.05):
  """
  Create :class:`Triangulation` mesh from ordered set of boundary
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

  if np.isscalar((_mesh_size := mesh_size)):
    mesh_size = lambda *args, **kwargs: _mesh_size

  points = np.asarray(points)
  assert points.shape[1:] == (2,)

  with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(points)
    geom.set_mesh_size_callback(mesh_size)
    mesh = geom.generate_mesh(algorithm=5)

  return mesh.cells_dict['triangle'][:, [0, 2, 1]], mesh.points


ACCETABLE_FORMATS = 'stl'
# still needs a unit test.
def tet_gen_from_surface(points = None, elements= None, filename = None, mesh_size: int | float = 0.5):

  if filename is not None:
    assert filename.endswith in ACCETABLE_FORMATS, 'Format not valid'
    
    import trimesh
    mesh = trimesh.load_mesh('filename') 
    assert mesh.is_watertight, 'The surface is not closed'
    
    import meshio
    surface = meshio.read(filename) 
    points = surface.points
    elements = surface.cells_dict['tetra'][:, [0, 3, 1, 2]]
  else:
    assert elements.shape[1] == 3, 'The input surface is not a triangulation'

  import pygmsh
  geom = pygmsh.built_in.Geometry()

  for i in range(points.shape[0]):
    geom.add_point(points[i],mesh_size)

  for i in range(elements.shape[0]):
    geom.add_triangle(points[elements[i]])

  mesh_tet = pygmsh.generate_mesh(geom)
  
  import meshio
  meshio.write("tet_mesh.vtk", mesh.points, mesh.cells)

  return mesh_tet.cells_dict['tetra'][0, 3, 1, 2], mesh_tet.points




  

  
  