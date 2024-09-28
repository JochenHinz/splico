from ..util import np
from itertools import product
from typing import Sequence


"""
  Routines dedicated to refining quadrilateral meshes
  (linemesh, quadmesh and HexMesh).
"""

# XXX: Re-introduce Cython compilation


def _format_indices_weights(indices: np.ndarray, weights: np.ndarray) -> str:
  """
    Given a container of vertex indices and
    an equally-sized container of weights with respect to these indices,
    remove indices with weight 0 and argsort both remaining containers
    according to the remaining indices.
    >>> _format_indices_weights([5, 4, 1, 7], [.5, .5, 0, 0])
        (4, 5), (.5, .5)
  """
  mask = weights != 0
  indices, weights = indices[mask], weights[mask]
  argsort = np.argsort(indices)
  return str((tuple(indices[argsort]), tuple(weights[argsort])))


def mul_tuple(tpl: Sequence[int]) -> float:
  return np.multiply.reduce(tpl)


def _formatter(point: np.ndarray):
  return np.array([mul_tuple(weights) for weights in product(*[[1 - x, x] for x in point])])


def _refine_structured(elements: np.ndarray, points: np.ndarray, ndims: int):

  # map the range(len(self.points)) to existing points
  map_index_coord = {}
  for i in range(len(points)):
    map_index_coord[i] = points[i]

  index = len(map_index_coord)
  map_format_index = {}

  # map ((i,), (1.0,)) which represents 1 * self.points[i] to index i
  for i in range(len(points)):
    map_format_index['((i,), (1.0,))'] = i

  xi = np.linspace(0, 1, 3)

  # make a subdivision of [0, 1] ** n with 3 points in each direction
  _X = np.stack(list(map(np.ravel, np.meshgrid(*[xi] * ndims))), axis=1)

  # format X into the form (i0, i1, ...), (w0, w1, ...)
  X = tuple(map(_formatter, _X))

  # map tensor index to global index using `loc_indices`
  loc_indices = np.arange(3 ** ndims, dtype=np.int64).reshape(*[3]*ndims)

  elems = np.empty((int(2 ** ndims * elements.shape[0]), int(elements.shape[1])), dtype=np.int64)
  k = 0
  for ielem, indices in enumerate(elements):

    for j, weights in enumerate(X):
      # get index that corresponds to the formatted indices, weights
      # if not yet available, create new index via defaultdict constructor
      _hash = _format_indices_weights(indices, weights)
      if _hash not in map_format_index:
        map_format_index[_hash] = index
        map_index_coord[index] = (points[indices] * weights[:, None]).sum(0)
        index += 1

    # loop over all subdivision (hyber)cubes and look up their vertices' global indices.
    for my_multi_index in map(np.asarray, product(*[range(2)] * ndims)):
      my_elem_weights = [X[loc_indices[tuple(_index)]] for _index in [my_multi_index + myindex0 for myindex0 in map(np.asarray, product(*[range(2)]*ndims))]]
      for j, myweights in enumerate(my_elem_weights):
        elems[k, j] = map_format_index[_format_indices_weights(indices, myweights)]
      k += 1

  # for some reason I sometimes get segfaults when plotting if I don't copy the points
  points = np.stack([map_index_coord[i] for i in range(len(map_index_coord))])
  return np.asarray(elems), points
