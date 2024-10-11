from ..util import np

from numba import njit, prange


"""
  Module containing various JIT-compiled helper routines for use in `bool.py`
"""


# TODO: Make use of Numba parallelization


""" JIT-compiled routines for normal mesh union / difference """

# In Numba, tuples of differing length are differing types


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


def array_to_tupleN(nelems: int):
  """
    Dynamic code generator allowing for numba tuple conversion with an arbitrary
    number of entries.
  """
  # XXX: add unit test
  assert nelems >= 6
  funcstr = """def func(arr): return ({})""".\
            format(', '.join(map('arr[{}]'.format, range(nelems))) + ',')

  local_namespace = {}
  # Execute the code in the given namespace
  exec(funcstr, globals(), local_namespace)

  # Extract the function from the namespace
  func = local_namespace['func']

  return njit(func)


def default_converter(nelems):
  if 1 <= nelems <= 5:
     return {i: globals().get(f'array_to_tuple{i}') for i in range(1, 6)}[nelems]

  # dynamically create converter
  return array_to_tupleN(nelems)


def make_numba_indexmap(points):
  assert points.ndim == 2
  nargs = points.shape[1]
  if nargs == 3:
    return _make_numba_indexmap3(points)

  converter = default_converter(points.shape[1])
  return _make_numba_indexmap(points, converter)


@njit(cache=True)
def _make_numba_indexmap(points, point_converter):
  """
    Create a hashmap that maps each point in `points` to a unique running index.
    Assumes the points in `points` to already be unique.
  """
  map_coord_index = {}
  i = 0
  for point in points:
    map_coord_index[point_converter(point)] = i
    i += 1

  return map_coord_index


# XXX: Defining this function as
#      ``partial(_make_numba_indexmap, point_converter=array_to_tuple3)``
#      prevents function caching. Find way to write cached version of this function
#      without code copy.
#      This may be possible by dynamically creating the function string when
#      loading the script.
@njit(cache=True)
def _make_numba_indexmap3(points):
  """ ``partial(_make_numba_indexmap, point_converter=array_to_tuple3)`` """
  map_coord_index = {}
  i = 0
  for point in points:
    map_coord_index[array_to_tuple3(point)] = i
    i += 1

  return map_coord_index


def renumber_elements_from_indexmap(elements, points, map_coord_index):
  """
    Given an element array `elements` and corresponding points ``points``
    create a new element array wherein each entry ``newelements[i, j]`` is given
    by ``map_coord_index[points[elements[i, j]]]``.
    The indexmap must take ``points.shape[1]``-tuples of coordinates and return
    a new index.
    If an entry is not contained, undefined behavior results.
  """
  nargs = len(list(map_coord_index.keys())[0])
  assert points.shape[1:] == (nargs,)

  # more efficient non-dynamical routine
  if nargs == 3:
    return _renumber_elements_from_indexmap3(elements, points, map_coord_index)

  converter = default_converter(len(list(map_coord_index.keys())[0]))

  return _renumber_elements_from_indexmap(elements,
                                          points,
                                          map_coord_index,
                                          converter)


@njit(cache=True, parallel=True)
def _renumber_elements_from_indexmap(elements, points,
                                     map_coord_index, point_converter):
  newelems = np.empty(elements.shape, dtype=np.int64)
  for i in prange(len(elements)):
    myvertices = elements[i]
    for j, point in enumerate(points[myvertices]):
      newelems[i, j] = map_coord_index[point_converter(point)]
  return newelems


# XXX: Defining this function as
#      ``partial(_renumber_elements_from_indexmap, point_converter=array_to_tuple3)``
#      prevents function caching. Find way to write cached version of this function
#      without code copy.
@njit(cache=True, parallel=True)
def _renumber_elements_from_indexmap3(elements, points, map_coord_index):
  """
    ``partial(_renumber_elements_from_indexmap, point_converter=array_to_tuple3)``
  """
  newelems = np.empty(elements.shape, dtype=np.int64)
  for i in prange(len(elements)):
    myvertices = elements[i]
    for j, point in enumerate(points[myvertices]):
      newelems[i, j] = map_coord_index[array_to_tuple3(point)]
  return newelems


""" Routines for boundary mesh union """


@njit(cache=True)
def _make_matching(points0, points1, eps):
  """
    Given two sets of points and a threshold ``eps``, create a correspondence
    between points in pairs if they are sufficiently close, i.e., if their
    Euclidean distance is below ``eps``.
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
        An `(nmatchings, 2)` - shaped integer array containing the matches
        as rows.
  """
  # XXX: the efficiency of this routine would benefit greatly from parallelization.

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
    As opposed to ``_make_matching``, only the mesh's active points are matched
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
    corresponding point in ``all_points`` that corresponds to the lower of
    the two indices is kept.

    Parameters
    ----------
    all_elements : :class:`np.ndarray[int, 2]`
        The element array containing all element indices before re-indexing.
    all_points : :class:`np.ndarray[float, 2]`
        The point array. Is assumed to be exhaustive, i.e., there are exact
        as many points as there are unique element indices.
    all_matches : :class:`np.ndarray[int, 2]`
        An (nmatches, 2) integer array containing matched point indices which
        are then coupled.

    Returns
    -------
    elements : :class:`np.ndarray[int, 2]`
        The renumbered element array.
    points : :class:`np.ndarray[float, 2]`
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
      # normally this would be a one-liner but numba doesn't support the necessary
      # syntax.
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
