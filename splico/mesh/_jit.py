# XXX: create module with jitted methods for use in `mesh.py` that improve performance

from numba import njit, prange

from .mesh import Mesh
from ..util import np

""" Routines for normal mesh union / difference """


@njit(cache=True)
def array_to_tuple1(arr):
  return (arr[0],)


@njit(cache=True)
def array_to_tuple2(arr):
  return (arr[0], arr[1])


@njit(cache=True)
def array_to_tuple3(arr):
  return (arr[0], arr[1], arr[2])


@njit(cache=True)
def array_to_tuple4(arr):
  return (arr[0], arr[1], arr[2], arr[3])


@njit(cache=True)
def array_to_tuple5(arr):
  return (arr[0], arr[1], arr[2], arr[3], arr[4])


def array_to_tuple_str(nelems):
  assert nelems >= 1
  funcstr = """def func(arr): return ({})""".format(', '.join(map('arr[{}]'.format, range(nelems))) + ',')
  local_namespace = {}

  # Execute the code in the given namespace
  exec(funcstr, globals(), local_namespace)

  # Extract the function from the namespace
  func = local_namespace['func']

  return njit(func)


@njit(cache=True)
def make_numba_indexmap(points):
  """
    Create a hashmap that maps each point in `points` to a unique running index.
    Assumes the points in `points` to already be unique.
  """
  map_coord_index = {}
  i = 0
  for point in points:
    map_coord_index[array_to_tuple3(point)] = i
    i += 1

  return map_coord_index


@njit(cache=True, parallel=True)
def renumber_elements_from_indexmap(elements, points, map_coord_index):
  # XXX: docstring
  newelems = np.empty(elements.shape, dtype=np.int64)
  for i in prange(len(elements)):
    myvertices = elements[i]
    for j, point in enumerate(points[myvertices]):
      newelems[i, j] = map_coord_index[array_to_tuple3(point)]
  return newelems


def mesh_union(*meshes: Mesh):
  assert meshes

  if len(meshes) == 1:
    return meshes[0]

  assert all(mesh.__class__ is meshes[0].__class__ for mesh in meshes)

  # get all unique points
  allpoints = np.unique(np.concatenate([mesh.points for mesh in meshes]), axis=0)

  # map each unique point to an index
  indexmap = make_numba_indexmap(allpoints)

  # create all new elements (counting different orderings twice) by mapping element indices to new indices
  newelems = np.concatenate([renumber_elements_from_indexmap(mesh.elements, mesh.points, indexmap) for mesh in meshes])

  # keep get the indices of the unique elements, not counting different orderings twice
  _, unique_indices = np.unique(np.sort(newelems, axis=1), return_index=True, axis=0)

  # return new mesh
  return meshes[0].__class__(newelems[unique_indices], allpoints)


def mesh_difference(mesh0: Mesh, mesh1: Mesh):
  """ mesh0 - mesh1. """
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


""" Routines for boundary mesh union """


# XXX: add boundary mesh difference


@njit(cache=True)
def _make_matching(points0, points1, eps):
  # XXX: docstring
  assert points0.ndim == 2 and points0.shape[1:] == points1.shape[1:]
  n = points0.shape[1]

  # make set of available right matches
  indices = np.arange(len(points1))

  match0, match1 = [], []

  # for all points in point0
  for i, point in enumerate(points0):

    # set the candidates to the currently available indices
    candidates = indices

    # retain only candidates that satisfy the necessary condition
    # abs(point0[i] - point1[i]) <= eps for all i in (0, n-1)
    candidates_found = True  # keep track if candidates were found
    for j in range(n):
      myslice = points1[:, j][candidates]
      candidates = candidates[ np.where(np.abs(myslice - point[j]) <= eps)[0] ]

      # no candidates left ? => candidates_found = False; break
      if len(candidates) == 0:
        candidates_found = False
        break

    # no candidates ? => continue
    if not candidates_found:
        continue

    # of the ones left behind, retain only those whose Euclidean distance is
    # truly less than eps

    # compute Euclidean distance (squared)
    # distances = ((points1[candidates, :] - point[None]) ** 2).sum((1,))

    distances = np.empty((len(candidates),), dtype=np.float64)

    for k, candidate in enumerate(candidates):
      distances[k] = ((points1[candidate] - point)**2).sum()

    # retain only indices that are below the threshold
    retain = np.where(distances <= eps ** 2)[0]

    # no indices left ? => continue
    if len(retain) == 0:
      continue

    # get the minimum distance global index
    j = candidates[np.argmin(distances)]

    match0.append(i)
    match1.append(j)

    # find local_index of j in `indices` and remove it from the available indices
    local_index = np.searchsorted(indices, j)
    indices = np.delete(indices, local_index)

  ret = np.empty((len(match0), 2), dtype=np.int64)

  for i in range(len(match0)):
    ret[i, 0] = match0[i]
    ret[i, 1] = match1[i]

  return ret


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

  # make two index maps, mapping the matching indices in mindices0, mindices1
  # to a shared new global index (at the end)
  indexmap0, indexmap1 = {}, {}
  for i, (i0, i1) in enumerate(zip(mindices0, mindices1)):
    indexmap0[i0] = tot_int_points + i
    indexmap1[i1] = tot_int_points + i

  # remap elems0 by only changing matched indices
  for i, row in enumerate(elems0):
    for j, val in enumerate(row):
      if val in indexmap0:
        elems0[i, j] = indexmap0[val]

  # remap elems1 by changing matched indices and giving unmatched ones an
  # appropriate offset
  for i, row in enumerate(elems1):
    for j, val in enumerate(row):
      if val in indexmap1:
        elems1[i, j] = indexmap1[val]
      else:
        elems1[i, j] += tot_int_points0 - nmatches

  # compute the matched points as the average from both point sets
  matched_points = (points0[mindices0] + points1[mindices1]) / 2.0

  # create new point array and fill it first with the unmatched points in
  # points0, then the unmatched ones in points1 and then with the matched points.
  points = np.empty( (tot_int_points + nmatches, 3), dtype=np.float64 )

  i = 0
  for j, point in enumerate(points0):
    if j in indexmap0:
      continue
    points[i] = point
    i += 1

  for j, point in enumerate(points1):
    if j in indexmap1:
      continue
    points[i] = point
    i += 1

  for point in matched_points:
    points[i] = point
    i += 1

  # return concatenation of both remapped element arrays and the points
  return np.concatenate((elems0, elems1)), points


def mesh_boundary_union(*meshes, eps=1e-6):

  # XXX: docstring

  assert meshes

  if len(meshes) == 1:
    return meshes[0]

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
  # XXX: try to avoid copying somehow
  elems, points = _remap_elements(mesh0.elements.copy(),
                                  mesh1.elements.copy(),
                                  gmatch0,
                                  gmatch1,
                                  mesh0.points,
                                  mesh1.points)

  # take union of the new mesh with the remaining meshes
  # if tail is empty, the call simply returns the new mesh
  return mesh_boundary_union(mesh0.__class__(elems, points), *tail)
