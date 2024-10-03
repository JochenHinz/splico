from ..util import np
from ._jit import _make_matching

from numba import njit
import treelog as log


@njit(cache=True)
def _remap_elements(elems0, elems1, mindices0, mindices1, points0, points1):
  """
    Given two sets of elements `(elems0, elems1)`, two equally-sized
    index sets `(mindices0, mindices1)` representing matched points and
    the corresponding two point sets `(points0, points1)`,
    create a new set of elements from the old elements in which indices in
    mindices0 (elems0) and mindices1 (elems1) are mapped to a shared index which
    points to a point which is the average of points0[mindices0[i]] and
    points1[mindices1[i]].

    All other points are left unaffected. The indices of elems0 that are not
    in mindices0 are unchaged while those of elems1 that are not in mindices1
    only receive an offset, representing the number of indices in elems0 (excl mindices0).
  """

  # XXX: generalize to arbitrary number of meshes (see `mesh_boundary_union`)

  # get number of matches
  nmatches = len(mindices0)

  # make sure both index arrays have equal length
  assert len(mindices1) == nmatches

  # make sure we're dealing with same element types
  assert elems0.shape[1:] == elems1.shape[1:]

  # get total interior points of both meshes and the sum of them
  tot_int_points0, tot_int_points1 = len(points0) - nmatches, len(points1) - nmatches
  tot_int_points = tot_int_points0 + tot_int_points1

  # XXX: what follows can probably be parallelised

  # keep track of the matched indices from both meshes in a set
  indexset0, indexset1 = set(), set()
  for i0, i1 in zip(mindices0, mindices1):
    indexset0.add(i0)
    indexset1.add(i1)

  # compute the matched points as the average from both point sets
  matched_points = (points0[mindices0] + points1[mindices1]) / 2.0

  # allocate two integer arrays of len(points0) and len(points1), respectively
  # containing at the i-th position the new global index of the local i-th node
  remap0 = np.empty((len(points0),), dtype=np.int64)
  remap1 = np.empty((len(points1),), dtype=np.int64)

  # set the matched indices to an arange from the total internal points to
  # the total internal points + the number of matches
  remap0[mindices0] = np.arange(tot_int_points, tot_int_points + len(mindices0))
  remap1[mindices1] = np.arange(tot_int_points, tot_int_points + len(mindices0))

  # create new point array and fill it first with the unmatched points in
  # points0, then the unmatched ones in points1 and then with the matched points.
  points = np.empty( (tot_int_points + nmatches, 3), dtype=np.float64 )

  i = 0
  for mypoints, myindexset, myremap in zip((points0, points1), (indexset0, indexset1), (remap0, remap1)):

    for j, point in enumerate(mypoints):
      if j in myindexset:
        continue
      points[i] = point  # set the i-th entry to point
      myremap[j] = i  # set the j-th entry of the remap to i
      i += 1

  for point in matched_points:
    points[i] = point
    i += 1

  # remaps have been finished at this point

  elems = np.empty((len(elems0) + len(elems1), elems0.shape[1]), dtype=np.int64)

  # remap elems0 and elems1
  for myelems, myremap, myoffset in zip((elems0, elems1), (remap0, remap1), (0, len(elems0))):
    for i, row in enumerate(myelems):
      for j, val in enumerate(row):
        elems[i + myoffset, j] = myremap[myelems[i, j]]

  # return concatenation of both remapped element arrays and the points
  return elems, points


def mesh_boundary_union(*meshes, eps=1e-6):

  """
    Create a union of meshes without checking for duplicate interior points
    but only points that are on the boundary.
    As opposed to `mesh_union`, the points need not exactly match but are
    considered equal if their Euclidean distance is below `eps`.

    Parameters
    ----------
    meshes : :class:`Tuple[Mesh]`
        The input meshes whose union we take. Needs to contain at least one element.
    eps : :class:`float`
        The matching tolerance.

    Returns
    -------
    union_mesh : :class:`Mesh`
        The union of the input meshes, only considering boundary points.
  """

  # XXX: currently the boundary union takes place repeatedly in case more than
  #      two meshes have been passed. In the long run it would be better to
  #      take the union of all of them at once. This would also enable creating
  #      a global matching of all meshes which could then optionally be returned
  #      and reused for meshes with the same topology.

  assert meshes

  if len(meshes) == 1:
    return meshes[0]

  if any( len(mesh.active_indices) != len(mesh.points) for mesh in meshes ):
    log.warning("Warning, inactive points detected in at least one mesh,"
                " they will be removed.")

  meshes = [mesh.drop_points_and_renumber() for mesh in meshes]

  # for now all meshes need to be of the same type
  assert all(mesh.__class__ is meshes[0].__class__ for mesh in meshes)

  mesh0, mesh1, *tail = meshes

  # get boundaries
  dmesh0, dmesh1 = mesh0.boundary, mesh1.boundary

  # get sorted active indices
  active0, active1 = map(np.unique, (dmesh0.elements, dmesh1.elements))

  # retain only active points
  dmeshpoints = [dmesh.points[active] for dmesh, active in zip((dmesh0, dmesh1), (active0, active1))]

  # get the local matching of active points
  local_matching = _make_matching(*dmeshpoints, eps)

  # convert to global
  gmatch0, gmatch1 = [active[loc] for active, loc in zip((active0, active1), local_matching.T)]

  # create new elems, points from the matching data
  elems, points = _remap_elements(mesh0.elements,
                                  mesh1.elements,
                                  gmatch0,
                                  gmatch1,
                                  mesh0.points,
                                  mesh1.points)

  # take union of the new mesh with the remaining meshes
  # if tail is empty, the call simply returns the new mesh
  return mesh_boundary_union(mesh0.__class__(elems, points), *tail)
