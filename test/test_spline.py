from splico.util import _
from splico.spl import UnivariateKnotVector, TensorKnotVector
from splico.spl import NDSpline, SplineCollection
from splico.mesh import rectilinear

import unittest

import numpy as np


class TestNDSpline(unittest.TestCase):

  def test1D(self):

    uknotvector = UnivariateKnotVector(np.linspace(0, 1, 51), 3)
    knotvector = TensorKnotVector([uknotvector])

    ndatapoints = 11
    xi = np.linspace(0, 1, ndatapoints) ** 2
    data = xi[:, _, _] + .4 * np.random.randn(*(ndatapoints,), 4, 3)
    spline = knotvector.fit([xi], data, lam0=1e-6)

    test = spline(xi)
    self.assertTrue(test.shape == (len(xi), 4, 3))

  def test2D(self):

    uknotvector = UnivariateKnotVector(np.linspace(0, 1, 21), 3)
    knotvector = uknotvector * uknotvector

    ndatapoints = 11
    xi = np.linspace(0, 1, ndatapoints)
    data = (xi[:, _, _, _] + .2 * np.random.randn(*(ndatapoints,)*2, 4, 3)).reshape(-1, 4, 3)
    spline = knotvector.fit([xi, xi], data, lam0=1e-6)

    test = spline(xi, xi)
    self.assertTrue(test.shape == (len(xi), 4, 3))


class TestFitSample(unittest.TestCase):

  def test_fit_spline2D(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 11))
    kv = kv * kv
    abscissae = np.linspace(0, 1, 21)
    x, y = map(np.ravel, np.meshgrid(abscissae, abscissae))
    z = np.zeros_like(x)
    spline = kv.fit([abscissae] * 2, np.stack([x, 1 + x + y, z], axis=1))
    mesh = rectilinear((21, 21))
    sampled_mesh = spline.sample_mesh( mesh )
    self.assertTrue( (np.abs(sampled_mesh.points[:, :2] - np.stack([x, 1 + x + y], axis=1)) < 1e-2).all() )

  def test_fit_spline3D(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 11))
    kv = kv * kv * kv
    abscissae = np.linspace(0, 1, 21)
    x, y, z = map(np.ravel, np.meshgrid(abscissae, abscissae, abscissae))
    spline = kv.fit([abscissae] * 3, np.stack([x, 1 + x + y, z], axis=1))
    mesh = rectilinear((21, 21, 21))
    sampled_mesh = spline.sample_mesh( mesh )
    self.assertTrue( (np.abs(sampled_mesh.points - np.stack([x, 1 + x + y, z], axis=1)) < 1e-2).all() )

  def test_sample_mesh(self):
    pass


class TestNDSPlineArithmetic(unittest.TestCase):

  def test_add(self):
    kv0, kv1 = [UnivariateKnotVector(np.linspace(0, 1, n)) for n in (5, 7)]
    kv = kv0 * kv1

    spl0 = NDSpline(kv, np.random.randn(kv.ndofs, 3, 2, 4))
    spl1 = NDSpline(kv, np.random.randn(kv.ndofs, 3, 2, 4))

    self.assertTrue( np.allclose((spl0 + spl1).controlpoints, spl0.controlpoints + spl1.controlpoints) )

    # test if addition is added to each DOF individually
    offset = np.random.randn(3, 2, 4)
    self.assertTrue( np.allclose((spl0 + offset).controlpoints, spl0.controlpoints + offset[_]) )

  def test_sum(self):
    kv0, kv1 = [UnivariateKnotVector(np.linspace(0, 1, n)) for n in (5, 7)]
    kv = kv0 * kv1

    spl0 = NDSpline(kv, np.random.randn(kv.ndofs, 3, 2, 4))

    # make sure but ways of summing are valid
    spl0_s = spl0.sum((0, 1))
    spl0_s = spl0.sum(0, 1)

    self.assertTrue(spl0_s.controlpoints.shape == (kv.ndofs, 4))
    self.assertTrue( np.allclose(spl0_s.controlpoints, spl0.controlpoints.sum((1, 2))) )

  def test_mul(self):
    kv0, kv1 = [UnivariateKnotVector(np.linspace(0, 1, n)) for n in (5, 7)]
    kv = kv0 * kv1

    spl0 = NDSpline(kv, np.random.randn(kv.ndofs, 3, 2, 4))

    offset = np.random.randn(3, 2, 4)
    self.assertTrue( np.allclose((spl0 * offset).controlpoints, spl0.controlpoints * offset[_]) )


class TestSplineCollection(unittest.TestCase):

  def test_new(self):
    xi = np.linspace(0, 1, 11)
    kv0 = UnivariateKnotVector(np.linspace(0, 1, 6))
    kv1 = UnivariateKnotVector(xi)
    kv = kv0 * kv1
    spline = NDSpline(kv, np.random.randn(kv.ndofs, 2, 3))
    spline0 = NDSpline(kv, np.random.randn(kv.ndofs, 2, 3, 4))

    splcollection = SplineCollection([ [spline, spline], [spline, spline] ])
    self.assertTrue(splcollection(xi, xi).shape == (11, *splcollection.shape, 2, 3))

    with self.assertRaises(AssertionError):
      test = SplineCollection([[spline, spline0], [spline, spline]])

    self.assertTrue( all(np.allclose( spl0.controlpoints, 2 * spl1.controlpoints ) for spl0, spl1 in zip((spline + spline).ravel(), spline.ravel())) )


if __name__ == '__main__':
  unittest.main()
