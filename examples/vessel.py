from splico.geo import CrossSectionMaker, compute_notwistframe_from_spline
from splico.mesh import rectilinear, mesh_union
from splico.spl import UnivariateKnotVector, \
                       TensorKnotVector, NDSpline, NDSplineArray
from splico.util import np, _, clparam


def main(nelems_centerline, nelems_cross_section, radii, centerline_points):
  """
  Make a 5-patch spline of a vessel and convert to mesh for visualization.

  Parameters
  ----------
  nelems_centerline : :class:`int`
      The number of spline elements to use for the knotvector that fits
      the centerline.
  nelems_cross_section : :class:`int`
      The number of elements of the five cross section patches in each
      direction.
  radii : :class:`np.ndarray`
      Radius information at the centerline points.
      Must be of shape centerline_points.shape[:1] (and stritly positive).
  centerline_points : :class:`np.ndarray`
      The centerline point information. Must be of shape (npoints, 3).

  Version has been adapted to make use of the NDSplineArray class.
  """

  # make a knotvector with the specified number of elements
  kv = UnivariateKnotVector(np.linspace(0, 1, nelems_centerline))
  kv = TensorKnotVector([kv])

  assert centerline_points.shape[1:] == (3,)
  assert radii.shape == centerline_points.shape[:1]

  # compute the chord length parameter values of the centerline points
  xi = clparam(centerline_points)

  # fit a univariate spline to the data
  X = kv.fit([xi], centerline_points)

  # compute the notwistframe rotation matrices
  Rs = compute_notwistframe_from_spline(X, xi)

  maker = CrossSectionMaker(nelems_cross_section)

  # make the cross section of radius one
  disc = NDSplineArray(maker.make_disc(1, 1, 0))

  # fit a spline to radius and rotational frame information
  rRs = kv.fit([xi], radii[:, _, _] * Rs)

  one = NDSpline.one(disc.arr.ravel()[0].knotvector)

  # create the vessel spline using a tensor product
  # disc.shape == (5, 3) 5 patches 3 coordinates
  # rRs.shape == (3, 3) 3x3 matrix
  # disc[:, _] * rRs[_] = (5, 1, 3) * (1, 3, 3)
  # taking .sum(-1) => (5, 3), representing a matrix multiplication
  vessel = (disc[:, _] * rRs[_]).sum(-1) + (one * X)[_]

  # create dense mesh
  eval_mesh = rectilinear([2 * nelems_cross_section + 1,
                           2 * nelems_cross_section + 1,
                                                     301])

  # create a mesh sampled from each patch and take its boundary union
  # to provide one hexmesh for the entire geometry.
  mesh = mesh_union(*vessel.sample_mesh(eval_mesh), boundary=True)

  # plot
  mesh.plot()

  print(f'The mesh is valid ?: {mesh.is_valid()}.')


if __name__ == '__main__':
  xi = np.linspace(0, 1, 101)
  radii = 2 + .5 * np.sin(2 * np.pi * xi)

  r = 20
  centerline_points = np.stack([r * np.cos(4 * np.pi * xi),
                                r * np.sin(4 * np.pi * xi),
                                100 * xi], axis=1)
  main(40, 4, radii, centerline_points)
