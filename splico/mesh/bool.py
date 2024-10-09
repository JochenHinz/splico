from ..util import np
from .aux import HasNoSubMeshError
from ._bool import make_numba_indexmap, renumber_elements_from_indexmap, \
                   _remap_elements, _match_active

from functools import lru_cache
from itertools import product

import treelog as log


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

    # the shape is the same => make sure that unique_elements and mesh1.elements
    # have the same indices when brought into lexigraphically sorted form
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


def mesh_union(*meshes):
  """
    Take the union of several meshes.
    Duplicate points and elements are detected using a hashmap. Therefore only
    points that match exactly are treated as equal. Very sensitive to numerical
    round-off errors. To reduce sensitivity, use in combination with the
    `splico.util.global_precision` context manager.
  """
  assert meshes

  if len(meshes) == 1:
    return meshes[0]

  assert all(mesh.__class__ is meshes[0].__class__ for mesh in meshes)

  # get all unique points
  allpoints = np.unique(np.concatenate([mesh.points for mesh in meshes]), axis=0)

  # map each unique point to an index
  indexmap = make_numba_indexmap(allpoints)

  # create all new elements (counting different orderings twice) by mapping element indices to new indices
  newelems = np.concatenate([renumber_elements_from_indexmap(mesh.elements,
                                                             mesh.points,
                                                             indexmap) for mesh in meshes])

  # keep get the indices of the unique elements, not counting different orderings twice
  _, unique_indices = np.unique(np.sort(newelems, axis=1), return_index=True, axis=0)

  # return new mesh
  return meshes[0].__class__(newelems[unique_indices], allpoints)


def mesh_difference(mesh0, mesh1):
  """ mesh0 - mesh1. """
  # XXX: some parts still need to be JIT compiled
  assert mesh0.__class__ is mesh1.__class__, NotImplementedError

  allpoints = np.unique(np.concatenate([mesh.points for mesh in (mesh0, mesh1)]), axis=0)

  indexmap = make_numba_indexmap(allpoints)

  elems0 = renumber_elements_from_indexmap(mesh0.elements, mesh0.points, indexmap)
  elems1 = renumber_elements_from_indexmap(mesh1.elements, mesh1.points, indexmap)

  identifiers0 = np.sort(elems0, axis=1)
  identifiers1 = np.sort(elems1, axis=1)

  setelems0 = set(map(tuple, identifiers0)) - set(map(tuple, identifiers1))

  keepindices = [i for i, identifier in enumerate(map(tuple, identifiers0)) if identifier in setelems0]

  return mesh0.__class__(mesh0.elements[keepindices], mesh0.points)


def mesh_boundary_union(*meshes, eps=1e-8, return_matches=False):
  """
    Take the union of several meshes at once, only matching points on the
    boundary. Optionally return the computed matched vertex pairs for re-use
    in other meshes with the same mutual conncetivity. Note that the matched
    vertex pairs are based on a global index which is the local index plus
    and offset that depends on the position in `meshes`.

    Parameters
    ----------
    meshes : :class:`splico.mesh.Mesh`
        The input meshes. All need to be of the same type and need to possess
        a boundary mesh.
    eps: :class:`float`
        Matching tolerance that is forwarded to `_match_active`.
    return_matches : :class:`bool`
        Boolean indicating whether the integer array of shape (nmatches, 2)
        containing matching pairs should be returned. This enables its re-use
        in case many meshes with the same topology need to be unified.

    Returns
    -------
    union_mesh : :class:`splico.mesh.Mesh`
        The mesh union.
    all_matches : :class:`np.ndarray[int, 2]
        Optionally return the matching integer array.
  """

  if any( len(mesh.active_indices) != len(mesh.points) for mesh in meshes ):
    log.warning("Warning, inactive points detected in at least one mesh,"
                " they will be removed.")

  meshes = [mesh.drop_points_and_renumber() for mesh in meshes]

  # the local patch index is offset by a certain amount to assign a global
  # index.
  offsets = np.array([0, *map(lambda x: len(x.points), meshes)]).cumsum()
  dmeshes = [mesh.boundary for mesh in meshes]

  # make all matchings between differing meshes (i < j)
  # XXX: find a more efficient solution
  all_matches = []
  for (i, dmesh0), (j, dmesh1) in product(enumerate(dmeshes), enumerate(dmeshes)):
    if j <= i: continue
    # add offset to the two columns of the matches to reflect global indexing
    all_matches.append( _match_active(dmesh0, dmesh1, eps) +
                        np.array([[offsets[i], offsets[j]]]) )

  # concatenate all matches into one array
  all_matches = np.concatenate(all_matches)

  # lexicographically sort all matches
  shuffle = np.lexsort(all_matches.T[::-1])
  all_matches = all_matches[shuffle]

  # concatenate all elements and add offset
  all_elements = np.concatenate([ mesh.elements + myoffset
                                  for mesh, myoffset in zip(meshes, offsets)])

  # concatenate all points
  all_points = np.concatenate([mesh.points for mesh in meshes])

  # remap the elements from the matches
  elements, points = _remap_elements(all_elements, all_points, all_matches)

  ret = meshes[0].__class__(elements, points)

  if return_matches:
    return ret, all_matches
  return ret
