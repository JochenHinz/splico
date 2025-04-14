from splico.nutils.interface import NutilsInterface, infer_topo
from splico.nutils.topo import Topology
from splico.geo.disc import ellipse, PATCHES
from splico.spl import NDSplineArray
from splico.geo.interp import linear_interpolation
from splico.err import DuplicateOrientationError

import unittest

import numpy as np


class TestTopology(unittest.TestCase):

  def test_minterfaces(self):
    """ Forthcoming """
    topo = Topology(PATCHES).extrude()

    while topo.ndim > 0:
      minterfaces = {}
      for ipatch, intf in topo.patch_interface_flat_iter():
        minterfaces.setdefault(tuple(sorted(intf)), set()).add(intf)

      minterfaces = set( min(val) for val in minterfaces.values() )
      self.assertTrue(set(topo.minterfaces) == minterfaces)
      topo = topo.interfaces

  def test_valid_topo(self):
    topo = Topology([
        [0, 1, 3, 2],  # first square
        [2, 3, 5, 4],  # second square
    ])
    self.assertTrue(topo.ndim == 2)
    self.assertTrue(topo.shape == (2, 4))
    self.assertTrue(len(topo) == 2)
    self.assertTrue(bool(topo))

  def test_empty_topology(self):
    topo = Topology(np.zeros((0, 4), dtype=int))
    self.assertTrue(topo.ndim == 2)
    self.assertTrue(topo.shape == (0, 4))
    self.assertTrue(not topo)
    self.assertTrue(topo.interfaces.topo.shape == (0, 2))
    self.assertTrue(topo.extrude().topo.shape == (0, 8))

  def test_interfaces(self):
    topo = Topology(PATCHES).extrude()

    while topo.ndim > 0:
      interfaces = np.asarray(topo.minterfaces)
      topo = topo.interfaces
      self.assertTrue((topo.topo == interfaces).all())

    # taking the interfaces of a 0d topo gives a 0d topo without any patches
    self.assertTrue((y := topo.interfaces.topo).shape == (0, 1) and
                     y.dtype == np.int64)

  def test_valid_edge_orientations(self):
    topo = Topology(np.array([
        [0, 1, 3, 2],  # CCW
        [3, 2, 5, 4],  # CCW
    ]))
    self.assertTrue(topo.is_valid())

  def test_conflicting_orientations(self):
    topo = Topology(np.array([
        [0, 1, 3, 2],
        [2, 3, 1, 0],  # reversed order
    ]))
    self.assertTrue(not topo.is_valid())

  def test_interface_mapping(self):
    topo = Topology(np.array([
        [0, 1, 3, 2],
        [2, 3, 5, 4],
    ]))
    # Interface [2, 3] is shared
    self.assertTrue( topo.map_interface_patch[(2, 3)] == (0, 1) )

  def test_opposite_interface_chaine(self):
    topo = Topology([
        [0, 1],
        [1, 2],
        [2, 3],
    ])
    opp = topo.opposites
    # All interfaces should be transitively connected
    assert all( set([key, *val]) == {0, 1, 2, 3} for key, val in opp.items() )

  def test_extrude_topo(self):
    topo = Topology(np.array([
        [0, 1],
        [1, 2],
    ]))
    extruded = topo.extrude()
    self.assertTrue(extruded.ndim == 2)
    self.assertTrue(extruded.shape[1] == 4)

  def test_interfaces_special_cases(self):
    topo0d = Topology(np.zeros((0, 1), dtype=int))
    self.assertTrue(topo0d.interfaces.shape[1] == 1)

    topo1d = Topology( [[0, 1], [1, 2]] )
    self.assertTrue(topo1d.interfaces.shape[1] == 1)

  def test_patch_iterators_consistency(self):
    topo = Topology(np.array([
        [0, 1, 3, 2],
        [2, 3, 5, 4],
    ]))
    edges = list(topo.patch_edge_iter())
    interfaces = list(topo.patch_interface_iter())
    self.assertTrue(all(isinstance(e[1], tuple) for e in edges))
    self.assertTrue(all(isinstance(i[1], tuple) for i in interfaces))

  def test_orient(self):
    patches = list(PATCHES)
    patches[2] = [3, 5, 2, 4]

    topo = Topology(patches)
    self.assertTrue(topo.is_valid() is False)
    self.assertTrue(topo.orient().is_valid() is True)
    print(f"Before: {topo}, after {topo.orient()}.")

  def test_infer_topo(self):
    A = ellipse(1, 1, 4)
    topo = infer_topo(A)
    self.assertTrue(len(np.unique(topo.topo)) == len(np.unique(PATCHES)))
    self.assertTrue(topo.is_valid())
    A = linear_interpolation(A, A + np.array([0, 0, 1]), zdegree=1)
    topo = infer_topo(A)
    self.assertTrue(len(np.unique(topo.topo)) == 2 * len(np.unique(PATCHES)))
    self.assertTrue(topo.is_valid())


def add_zeros(ndarr: NDSplineArray, n=1):
  """ Extend last axis by zeros """
  if n == 0:
    return ndarr
  ndarr = ndarr.to_elemdim(1)
  newarr = []
  for _a in ndarr.arr.ravel():
    cps = _a.controlpoints
    ncps = np.concatenate([cps, np.zeros(cps.shape[:-1] + (n,))], axis=-1)
    newarr.append(_a._edit(controlpoints=ncps))
  return ndarr._edit(arr=np.asarray(newarr, dtype=object).reshape(ndarr.arr.shape))


class TestNutilsInterface(unittest.TestCase):

  def test2D(self):
    disc = ellipse(1, 1, 4).to_ndim(1)
    intf = NutilsInterface(disc, PATCHES)
    self.assertTrue(np.allclose(2 * disc.arr.ravel()[0].controlpoints,
                                intf.harmonic_transform(intf.geom * 2)
                                    .to_ndim(1)
                                    .arr
                                    .ravel()[0]
                                    .controlpoints))

    arr = disc.to_ndim(1).arr.ravel().copy()
    arr[2] = arr[2].refine(...)

    disc = disc._edit(arr=arr).contract_all()
    self.assertRaises(DuplicateOrientationError,
                      lambda: NutilsInterface(disc, PATCHES))

  def test3D(self):
    disc = ellipse(1, 1, 4).to_ndim(1)
    disc = linear_interpolation(disc, disc + np.array([0, 0, 1]), zdegree=1)
    topo = Topology(PATCHES).extrude()

    intf = NutilsInterface(disc, topo)
    self.assertTrue(np.allclose(2 * disc.arr.ravel()[0].controlpoints,
                                intf.harmonic_transform(intf.geom * 2)
                                    .arr
                                    .ravel()[0]
                                    .controlpoints))

    arr = disc.to_ndim(1).arr.ravel().copy()
    arr[2] = arr[2].refine(...)

    disc = disc._edit(arr=arr).contract_all()
    self.assertRaises(DuplicateOrientationError,
                      lambda: NutilsInterface(disc, topo))

  def test4D(self):
    disc = ellipse(1, 1, 4).to_ndim(1)
    disc = add_zeros(linear_interpolation(disc,
                                          disc + np.array([0, 0, 1]),
                                          zdegree=1))
    disc = linear_interpolation(disc, disc + np.array([0, 0, 0, 1]), zdegree=1)
    topo = Topology(PATCHES).extrude().extrude()
    intf = NutilsInterface(disc, topo)

    print("Solving a 4D problem, this may take a while...")
    self.assertTrue(np.allclose(2 * disc.arr.ravel()[0].controlpoints,
                                intf.harmonic_transform(intf.geom * 2)
                                    .arr
                                    .ravel()[0]
                                    .controlpoints))


if __name__ == '__main__':
  unittest.main()
