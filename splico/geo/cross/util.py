from splico.util import np, _

from typing import Mapping

from numba import njit


@njit(cache=True)
def apply_pairs(ret, list_of_pairs):
  """
  Apply a list of pairs to a numpy array.
  For each index pair in the list, the minimum value of the pair is
  assigned to both indices in the array.
  """
  for pair in list_of_pairs:
    mask = np.empty( (len(pair),), dtype=np.int64 )
    for i in range(len(pair)):
      mask[i] = pair[i]
    _min = np.min( ret[mask] )
    ret[mask] = _min
  return ret


def reflect_kv_dict(kvdict: Mapping) -> Mapping:
  return {**kvdict, **{k[::-1]: -kv for k, kv in kvdict.items()}}


def get_patch_edges(patch):
  return (patch[:2], patch[2:], (patch[0], patch[2]), (patch[1], patch[3]))


def get_edges(list_of_vertices):
  assert len(list_of_vertices) == 4
  list_of_vertices = np.asarray(list_of_vertices).reshape([2, 2])
  return tuple(map(tuple, np.concatenate([list_of_vertices, list_of_vertices.T])))


OPPOSITE_SIDE = {0: 1, 1: 0, 2: 3, 3: 2}


def edge_neighbours(patches, edge):
  edge = tuple(edge)
  assert len(edge) == 2

  edges_per_patch = tuple(map(get_edges, patches))
  assert any( edge in patch_edges or edge[::-1] in patch_edges
                                          for patch_edges in edges_per_patch )

  neighbours = {edge}
  newneighbours = neighbours.copy()
  while True:
    for neigh in neighbours:
      for patchverts in edges_per_patch:
        reverse = False
        if neigh in patchverts:
          index = patchverts.index(neigh)
        elif neigh[::-1] in patchverts:
          index = patchverts.index(neigh[::-1])
          reverse = True
        else: continue
        newneighbours.update({tuple(patchverts[OPPOSITE_SIDE[index]])[{False: slice(_),
                                                                       True: slice(_, _, -1)}[reverse]]})
    if len(newneighbours) == len(neighbours): break
    neighbours = newneighbours.copy()

  return tuple(newneighbours - {edge})


def infer_knotvectors(patches, knotvectors):
  """ from a dictionary ``knotvectors`` of the form dict = {edge: knotvector}
      and a list of lists ``patches`` in standard nutils format, infer
      the knotvector corresponding to missing edges from the topology provided
      by ``patches`` and at it to knotvectors. If an edge is not missing but
      possesses a incompatible knotvector, raise and Error. """

  # make sure entries are in standard format
  newknotvectors = knotvectors.copy()

  # add missing entries
  while True:
    for edge, knotvector in knotvectors.items():
      myneighbours = edge_neighbours(patches, edge)  # all entries are in standard format
      for neighbour in myneighbours:
        otherknotvector = knotvectors.get(neighbour, None)
        if otherknotvector is None:
          newknotvectors[neighbour] = knotvector
          if neighbour[::-1] in newknotvectors:
            raise AssertionError
        else: assert otherknotvector == knotvector
    if len(newknotvectors) == len(knotvectors): break
    knotvectors = newknotvectors.copy()

  # check that entries are compatible
  for edge, knotvector in newknotvectors.items():
    myneighbours = edge_neighbours(patches, edge)
    for neighbour in myneighbours:
      if newknotvectors[neighbour] != knotvector:
        raise AssertionError('The knotvectors are incompatible.')

  return newknotvectors
