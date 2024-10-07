from .mesh import Mesh
from ..util import np

from itertools import product

from numba import njit, prange
import treelog as log


"""
  Module containing various JIT-compiled routines for use in `bool.py`
"""


""" JIT-compiled routines for normal mesh union / difference """


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
  """
    Given two sets of points and a threshold `eps`, create a correspondence
    between points in pairs if they are sufficiently close, i.e., if their
    Euclidean distance is below `eps`.
    If two points have been matched their indices will be removed from the
    set of indices that are still eligible for matching to avoid duplicate
    matching.

    Parameters
    ----------
    points0 : :class:`np.ndarray`
        The array containing the first mesh's points.
    points1 : :class:`np.ndarray`
        The array containing the second mesh's points.
    eps : :class:`float`
        The matching tolerance.

    Returns
    -------
    matching : :class:`np.ndarray` of integers
        An (nmatchings, 2) - shaped integer array containing the matches
        as rows.
  """
  assert points0.ndim == 2 and points0.shape[1:] == points1.shape[1:]
  assert eps >= 0
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

    # compute Euclidean distance (squared, for some reason I can't vectorize)
    distances = np.empty((len(candidates),), dtype=np.float64)

    for k, candidate in enumerate(candidates):
      distances[k] = ((points1[candidate] - point)**2).sum()

    # retain only indices that are below the threshold
    # XXX: for stability reasons it will probably be better in the long run
    #      to require that the number of retained indices is at most one.
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

  if len(match0) == 0:
    return np.empty((0, 2), dtype=np.int64)

  ret = np.empty((len(match0), 2), dtype=np.int64)

  for i in range(len(match0)):
    ret[i, 0] = match0[i]
    ret[i, 1] = match1[i]

  return ret


def _match_active(mesh0, mesh1, eps=1e-8):
  """
    Match a matching of two meshs' points based on proximity.
    As opposed to `_make_matching`, only the mesh's active points are matched
    and the global rather than local matching indices are returned.
  """
  active0, active1 = mesh0.active_indices, mesh1.active_indices
  return np.stack([active[ind] for active, ind
                   in zip((active0, active1),
                          _make_matching(mesh0.points[active0],
                                         mesh1.points[active1], eps).T)], axis=1)


@njit(cache=True)
def _remap_elements(all_elements, all_points, all_matches):
  """
    Given an array of elements, an array of corresponding points and an array
    of matched point pairs, create a new element array wherein the two distinct
    indices of pairs are replaced by a single index for both while only the
    corresponding point in `all_points` that corresponds to the lower of
    the two indices is kept.

    Parameters
    ----------
    all_elements : :class:`np.ndarray[int]`
        The element array containing all element indices before re-indexing.
    all_points : :class:`np.ndarray[float]`
        The point array. Is assumed to be exhaustive, i.e., there are exact
        as many points as there are unique element indices.
    all_matches : :class:`np.ndarray[int]`
        An (nmatches, 2) integer array containing matched point indices which
        are then coupled.

    Returns
    -------
    elements : :class:`np.ndarray[int]`
        The renumbered element array.
    points : :class:`np.ndarray[float]`
        The corresponding point array that no longer contains two distinct
        points for matched pairs.
  """

  # ascending index array
  merge = np.arange(len(all_points))
  while True:
    _merge = merge.copy()

    # iterate over all pairs and always assign the lower index of the
    # two to the corresponding position in `merge`
    for pair in all_matches:
      a, b = merge[pair[0]], merge[pair[1]]
      _min = min((a, b))
      merge[pair[0]] = merge[pair[1]] = _min

    # repeat until the merge array no longer changes
    if (_merge == merge).all():
      break

  # get all unique indices that are left
  active_indices = np.unique(merge)

  new_elements = np.empty(all_elements.shape, dtype=np.int64)
  for i, elem in enumerate(all_elements):
    for j, index in enumerate(elem):

      # the new index is the position of the mapped old index in `active_indices`
      new_elements[i, j] = np.searchsorted(active_indices, merge[index])

  # only keep active points
  all_points = all_points[active_indices]

  return new_elements, all_points


def multi_mesh_boundary_union(*meshes, eps=1e-8, return_matches=False):
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
        Matching tolerance that is forwarded to `make_mesh`.
    return_matches : :class:`bool`
        Boolean indicating whether the integer array of shape (nmatches, 2)
        containing matching pairs should be returned. This enables its re-use
        in case many meshes with the same topology need to be unified.

    Returns
    -------
    union_mesh : :class:`Mesh`
        The mesh union.
    all_matches : :class:`np.ndarray[int]
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
    all_matches.append(_match_active(dmesh0, dmesh1, eps) + np.array([[offsets[i], offsets[j]]]))

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
