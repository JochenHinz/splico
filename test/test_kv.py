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


if __name__ == '__main__':
  unittest.main()
