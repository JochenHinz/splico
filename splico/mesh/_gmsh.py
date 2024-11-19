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

from stl import mesh
import gmsh
from splico.err import HasNoBoundaryError


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


ACCETABLE_FORMATS = 'stl', 'msh', 'vtk'

def tet_gen_from_surface(points = None, elements= None, filename = None, mesh_size: int | float = 0.4, write_output = True):

  # Pass elements and points, or an stl file.  
  if filename is not None:
    assert (points is None and elements is None)
    assert filename.endswith(ACCETABLE_FORMATS), 'Format not valid'
  else:
    assert (points is not None and elements is not None) 
    assert np.asarray(elements).shape[1] == 3, 'The input surface is not a triangulation'
    filename = create_stl(points, elements, "output_surface.stl")
  
  #surface_mesh = trimesh.load_mesh(filename)
  #try:
  #  surface_mesh.boundary
  #  raise TypeError("The mesh is not watertight.")
  #except HasNoBoundaryError:
  #  pass
      
  gmsh.initialize()
   
  gmsh.open(filename)

  surface_tag = 1  # This is just an example; the actual tag depends on your geometry
  surface_loop_tag = gmsh.model.geo.addSurfaceLoop([surface_tag])

  gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
  gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)    
  
  gmsh.model.geo.addVolume([surface_loop_tag])
 
  gmsh.model.geo.synchronize()

  gmsh.model.mesh.generate(3)
  
  
  #gmsh.model.mesh.refine()
  _, node_coords, _ = gmsh.model.mesh.getNodes()
  p,v, element_nodes = gmsh.model.mesh.getElements()
  
  # Node_coords is a flattened list of node coordinates, needs to be reshaped to be compatible with our format
  points = node_coords.reshape(int(node_coords.shape[0]/3),3)
  
  # The final mesh is composed by triangles and tetraheda. element_nodes[0] contains 
  # the flattened connectivity of the the triangle, i.e., it must be rehsaped. element_nodes[1] contains 
  # the flattened connectivity of the the tet
  # Triangles
  elements = []
  # To be compatible with our format, I just have to pass the tet elements.
  # If we need to recognize boundary, it will be done later on.
  # Tet  
  elements.append(element_nodes[1].reshape(int(element_nodes[1].shape[0]/4),4))
  # I have to do (elements - 1), since gmsh starts to count from 1
  elements = (np.asarray(elements) -1).reshape(np.asarray(elements).shape[1],np.asarray(elements).shape[2])
  
  if write_output:    
    gmsh.write("/home/fabio/output.vtk")

  gmsh.finalize()
  
  return points, elements


def convert_stl_file(stl_file):
    # Load the STL file using numpy-stl
    gmsh.initialize()
    stl_mesh = mesh.Mesh.from_file(stl_file)
    
    # Extract unique vertices (points)
    vertices = stl_mesh.points.reshape(-1, 3)
    unique_vertices = np.unique(vertices, axis=0)
    
    # Create a mapping of points (to ensure unique point tags)
    point_tags = {}
    for i, pt in enumerate(unique_vertices):
        # Add the point to GMSH and store the tag
        point_tag = gmsh.model.geo.addPoint(*pt)
        point_tags[tuple(pt)] = point_tag
    
    # Extract the triangular elements (facets) using the points
    elements = []
    for facet in stl_mesh.vectors:
        # Each facet is a triangle defined by 3 points
        # Map the coordinates to their tags
        element = [point_tags[tuple(facet[0])], point_tags[tuple(facet[1])], point_tags[tuple(facet[2])]]
        elements.append(element)
    
    gmsh.finalize()
    
    # adding vertices number 0.
    elements = np.asarray(elements) - 1
    
    return unique_vertices, elements

def create_stl(points, elements, filename):
    # Create an empty mesh
    faces = []
    
    # Loop over elements to extract the faces (triangles)
    for element in elements:
        p1 = points[element[0]]
        p2 = points[element[1]]
        p3 = points[element[2]]
        
        # Add triangle face to the faces list (3 vertices per face)
        faces.append([p1, p2, p3])

    # Convert faces into a NumPy array for the mesh
    faces = np.array(faces)

    # Create the mesh object from faces
    model = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, f in enumerate(faces):
        for j in range(3):
            model.vectors[i][j] = f[j]
    
    # Write the mesh to an STL file
    model.save(filename)
    
    return filename
