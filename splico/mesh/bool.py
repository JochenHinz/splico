from ..util import np
from .aux import HasNoSubMeshError

from functools import lru_cache


"""
 This module's purpose is performing various boolean operations
 on meshes.
"""


@lru_cache(maxsize=32)
def _issubmesh(mesh0, mesh1):
  """
    Check if `mesh0` is a submesh of `mesh1`.
    A submesh is defined as a mesh that contains the same or a subset of the
    other mesh's points and elements. Alternatively, `mesh0` is also considered
    a submesh of `mesh1` if it is a submesh of `mesh1.submesh` or its submeshes.

    Parameters
    ----------
    mesh0 : :class:`splico.mesh.Mesh`
        The submesh candidate.
    mesh1 : :class:`splico.mesh.Mesh`
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
    #      in a different order. In the long run, detect this too.
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
  from ._bool import multi_mesh_boundary_union as _mesh_boundary_union
  return _mesh_boundary_union(*meshes, **kwargs)


def mesh_union(*meshes):
  """ Docstring: see _jit.py """
  from ._bool import mesh_union as _mesh_union
  return _mesh_union(*meshes)


def mesh_difference(mesh0, mesh1):
  """ Take the difference of two meshes. """
  from ._bool import mesh_difference as _mesh_difference
  return _mesh_difference(mesh0, mesh1)
