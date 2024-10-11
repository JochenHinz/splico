from splico.spl import UnivariateKnotVector, TensorKnotVector

import unittest

import numpy as np

_ = np.newaxis


class TestUnivariateKnotVector(unittest.TestCase):

  def test_flip(self):
    knotvalues = np.array([0.0, 0.1, 0.25, 0.5, 0.95, 1.0]) + 2
    knotvector = UnivariateKnotVector(knotvalues)
    self.assertTrue( np.allclose(np.array([0.0, 0.05, 0.5, 0.75, 0.9, 1.0]) + 2, knotvector.flip().knots) )

  def test_multiply(self):
    knotvalues = np.linspace(0, 2, 21)
    knotvector = UnivariateKnotVector(knotvalues)
    tknotvector = knotvector * knotvector
    self.assertTrue( len(tknotvector) == 2 )
    self.assertTrue( isinstance(tknotvector, TensorKnotVector) )
    self.assertTrue( isinstance(knotvector * tknotvector, TensorKnotVector) )
    self.assertTrue( isinstance(tknotvector * knotvector, TensorKnotVector) )

  def test_functionality(self):
    knotvector = UnivariateKnotVector(np.linspace(-1, 1, 21))
    with self.subTest('Test refine'):
      self.assertTrue(np.allclose(knotvector.refine().knots,
                                  np.linspace(-1, 1, 41)))
    with self.subTest('Test ref_by'):
      self.assertTrue(np.allclose(knotvector.ref_by(0).knots,
                                  np.insert(knotvector.knots, 1, -0.95)))
      self.assertTrue(np.allclose(knotvector.ref_by([0, 1, 2]).knots,
                                  np.insert(knotvector.knots, [1, 2, 3], [-0.95, -.85, -.75])))
      # refining beyond the last element should require indexing into
      # knotvector.knots[21] which should raise an IndexError
      self.assertRaises(IndexError, lambda: knotvector.ref_by(20))

    with self.subTest('Test add_knots'):
      knots = knotvector.knots
      # adding beyond the knotvector's bounds is prohibited and should raise a ValueError
      self.assertRaises(ValueError, lambda: knotvector.add_knots(-3))
      self.assertTrue( np.allclose(knotvector.refine().knots,
                                   np.unique([*knots, *(knots[1:] + knots[:-1])/2])))

    with self.subTest('Test raise_multiplicities'):
      self.assertTrue( np.allclose(
          knotvector.raise_multiplicities(np.arange(1, len(knots) - 1), 3).km,
          4 * np.ones(len(knots), dtype=int) ))


if __name__ == '__main__':
  unittest.main()
