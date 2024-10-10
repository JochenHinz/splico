from splico.geo import CrossSectionMaker, compute_notwistframe_from_spline
from splico.mesh import rectilinear, mesh_boundary_union
from splico.spl import UnivariateKnotVector, TensorKnotVector
from splico.util import np, _

import unittest


class TestNoTwist(unittest.TestCase):

  def test_notwist(self):
    kv = UnivariateKnotVector(np.linspace(0, 1, 21))
    kv = TensorKnotVector([kv])
    radii = np.linspace(1, 2, 51)

    xi = np.linspace(0, 1, 101)
    centerline_points = np.stack([ 10 * xi, np.zeros_like(xi), 10 * xi**2 ], axis=1)
    X = kv.fit([xi], centerline_points)

    xi_notwist = np.linspace(0, 1, 51)
    Rs = compute_notwistframe_from_spline(X, xi_notwist)

    maker = CrossSectionMaker(4)

    disc = maker.make_disc(1, 1, 0)
    print(disc)
    points = disc.controlpoints.reshape(-1, 3)[:, :2]
    print(points.max())
    from matplotlib import pyplot as plt
    plt.scatter(*points.T)
    plt.show()
    rRs = kv.fit([xi_notwist], radii[:, _, _] * Rs)

    one = disc.__class__.one(disc.knotvector)

    rRs = rRs[_]
    vessel = (disc[:, _] * rRs).sum(-1)
    vessel = vessel + (one * X)[_]

    eval_mesh = rectilinear([5, 5, 101])

    mesh = mesh_boundary_union(*(v.sample_mesh(eval_mesh) for v in vessel))

    mesh.plot()

    # XXX: currently the test passes or fails by the `measure of eye`.
    #      Find a way to test whether the boundary union etc succeeds without
    #      having to look at it manually.


if __name__ == '__main__':
  unittest.main()
