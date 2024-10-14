from splico.util import np

from nutils import function, solver
import treelog as log


def get_all_vertices(domain, geom, patches):

  # XXX: docstring

  verts = {}

  for i, patch in enumerate(patches):
    for ii, j in enumerate(patch):
      if j in verts: continue
      side0, side1 = { 0: ('left', 'bottom'),
                       1: ('left', 'top'),
                       2: ('right', 'bottom'),
                       3: ('right', 'top') }[ii]
      verts[j] = domain._topos[i].boundary[side0]. \
                                  boundary[side1]. \
                                  sample('vertex', 0). \
                                  eval(geom).ravel()

  return np.stack([verts[key] for key in sorted(verts.keys())], axis=0)


def multipatch_trace_penalty_stab(domain, basis, geom, localgeom, patches, stabmat=None, mu=800):

  """
  Nutils implementation for reparameterizing the default covering
  of the unit disc toward more interface regularity.

  The stabilization for avoiding singularities at the vertices is tuned by
  `mu`.
  """

  assert mu >= 0

  if stabmat is None:
    stabmat = lambda x: function.eye(2)

  Jmu = function.normalized(geom.grad(localgeom), axis=0)
  D = function.matmat(Jmu, Jmu.T)

  x = basis.vector(2).dot(function.Argument('target', [len(basis)*2]))
  stabmat = stabmat(x)

  allverts = get_all_vertices(domain, geom, patches)
  blending = sum([ np.exp(-mu * ((vert - geom)**2).sum()) for vert in allverts ])

  D = (1 - blending) * D + blending * stabmat
  Dx = function.matmat(x.grad(geom), D)

  integrand = (basis.vector(2).grad(geom) * Dx[None]).sum([1, 2])
  res = domain.integral( integrand * function.J(geom), ischeme='gauss6')

  cons = domain.boundary.project(geom, onto=basis.vector(2), geometry=geom, ischeme='gauss6')

  controlmap = basis.vector(2).dot(solver.solve_linear('target', res, constrain=cons))

  return controlmap


def make_unit_disc(domain, basis, geom, localgeom, patches, reparam=False, **kwargs):

  """
  Create a multipatch covering of the unit disc from the topology.
  """

  if reparam is False and kwargs:
    log.warning('reparam is False so the keyword arguments will be ignored.')

  func = function.normalized(geom - np.array([.5, .5])) / np.sqrt(2) + np.array([.5, .5])
  cons = domain.boundary.project(func, geometry=geom, onto=basis.vector(2), ischeme='gauss6')
  for i, patch in enumerate(domain._topos):
    for side in ('left', 'right', 'bottom', 'top'):
      cons |= patch.boundary[side].project(geom, geometry=geom, onto=basis.vector(2), ischeme='gauss6')

  x = basis.vector(2).dot(function.Argument('target', [len(basis)*2]))
  J = x.grad(geom)
  G = function.matmat(J.T, J)

  costfunc = domain.integral(np.trace(G) * function.J(geom), degree=10)

  controlmap = basis.vector(2).dot(solver.optimize('target', costfunc, constrain=cons))

  if not reparam:
    return (controlmap - np.array([.5, .5])) * np.sqrt(2)

  return (multipatch_trace_penalty_stab(domain,
                                        basis,
                                        controlmap,
                                        localgeom,
                                        patches,
                                        **kwargs) - np.array([.5, .5])) * np.sqrt(2)
