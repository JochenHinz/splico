from .control import make_unit_disc
from .mul import multipatch
from splico.spl import UnivariateKnotVector, NDSpline
from splico.util import np, frozen, NanVec

from functools import lru_cache

from scipy.sparse import linalg as splinalg
from scipy import sparse
from nutils import function


@lru_cache(maxsize=32)
def rot_matrix(rot: float | int) -> np.ndarray:
  return np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])


def trampoline_template(inner_height: float = .5, inner_width: float = .5):
  """
    1 - - - - - 7
    | \   D   / |
    |  3 - - 5  |
    |A |  C  | E|
    |  |     |  |
    |  2-----4  |
    | /   B   \ |
    0 - - - - - 6

  """
  assert 0 < inner_height < 1
  assert 0 < inner_width < 1

  h, w = (1 - inner_height) / 2, (1 - inner_width) / 2

  # the vertices of the multipatch topology numbered from 0 to 7
  patchverts = ( (0, 0), (0, 1),
                 (w, h), (w, 1 - h),
                 (1 - w, h), (1 - w, 1 - h),
                 (1, 0), (1, 1) )

  # a patch is characterised by its patch vertices
  # patch = (0, 1, 2, 3) means it is oriented like this:
  """
            1 ----- 3
            |       |
            |       |
            |       |
            0-------2
  """

  # so the left side of the patch is given by edge 0 --- 1
  # right by 2 --- 3
  # bottom by 0 --- 2
  # top by 1 --- 3
  patches = ( (2, 0, 3, 1),  # patch A
              (2, 4, 0, 6),  # patch B
              (2, 3, 4, 5),  # etc
              (3, 1, 5, 7),
              (4, 5, 6, 7) )

  # patches and patchverts are tuple. Immutable objects.

  return patches, patchverts


def nutils_to_scipy(mat) -> sparse.csr_matrix:
  data = mat.export('csr')
  return sparse.csr_matrix(data, dtype=float)


class CrossSectionMaker:

  # XXX: docstring

  def __init__(self, nelems, degree=3, reparam=True, **templatekwargs):
    assert nelems > 0
    patches, patchverts = trampoline_template(**templatekwargs)

    # make a multipatch topology with the structure of ``trampoline_template``
    # and ``nelems`` elements per side
    # ``geom`` is the parameterisation of the parametric domain
    # ``localgeom`` maps each patch back onto the reference patch (0, 1)^2
    domain, geom, localgeom = multipatch(patches=patches, patchverts=patchverts, nelems=nelems)
    basis = domain.basis('spline', degree=degree)
    basis_disc = domain.basis('spline', degree=degree, patchcontinuous=False)

    self.domain = domain
    self.npatches = len(self.domain._topos)
    self.basis = basis
    self.basis_disc = basis_disc
    self.geom = geom
    self.localgeom = localgeom
    self.nelems = nelems
    self.degree = degree
    self.knotvector = UnivariateKnotVector( np.linspace(0, 1, nelems+1), degree=degree )
    self.tknotvector = self.knotvector * self.knotvector

    controlmap = make_unit_disc(domain, basis, geom, localgeom, patches, reparam=reparam)

    self.controlmap = controlmap

    # take the gradient of the basis with respect to the parametric domain ``geom``
    dbasis = basis.grad(controlmap)

    # assemble the stiffness matrix
    # convert it to csr for efficiency
    self.A = nutils_to_scipy(domain.integrate( (dbasis[:, None] * dbasis[None]).sum([2]) * function.J(controlmap), degree=10 ))

    # get a boolean array that gives True for indices of basis functions that are nonzero on the boundary
    self.freezemask = ~np.isnan( domain.boundary.project(1, geometry=geom, onto=basis, ischeme='gauss6') )

    # get the associated boolean array that is True for functions NOT on the boundary
    self.freemask = ~self.freezemask

    # compute the Cholesky decomposition of the stiffness matrix restricted
    # to the free indices (i.e. not on the boundary)
    self.CA = splinalg.splu(self.A.tolil()[:, self.freemask][self.freemask].tocsc())

    # take the basis restricted to the ones that are nonzero on the boundary
    self.basis_b = basis[self.freezemask]

    # assemble the mass matrix for projecting data onto the boundary DOFs
    self.M = nutils_to_scipy(domain.boundary.integrate(function.outer(self.basis_b) * function.J(geom), degree=10 ))

    # compute the Cholesky of it
    self.CM = splinalg.splu(self.M.tocsc())

    # compute the matrices necessary for the prolongation to the discontinuous basis
    self.M_disc, self.T_disc = map(nutils_to_scipy,
                                   domain.integrate([function.outer(self.basis_disc) * function.J(geom),
                                                     self.basis_disc[:, None] * self.basis[None] * function.J(geom)], degree=10))
    # unit disc
    self._reference_sol = frozen(self.domain.project(self.controlmap,
                                                     self.basis.vector(2),
                                                     ischeme='gauss10',
                                                     geometry=self.controlmap).reshape(-1, 2))

  @property
  def CM_disc(self):
    if not hasattr(self, '_CM_disc'):
      self._CM_disc = splinalg.splu(self.M_disc.tocsc())
    return self._CM_disc

  def boundary_correspondence(self, a, b, theta=0):
    x, y = self.geom
    theta = theta % (2 * np.pi)

    # Penalization factor.

    if b > a:
      alpha = np.arctan(b/a)
    elif b < a:
      alpha = np.arctan((b/a))
    elif b == a:
      alpha = np.pi/4

    basis_b = self.basis_b
    domain = self.domain

    # theta on each boundary as a function of x, y in the parametric domain
    # x and y both run from 0 to 1

    alpha = np.pi/4

    theta_right = (2 * alpha) * y - alpha + theta

    theta_top = (-np.pi + 2*alpha) * x + (np.pi - alpha) + theta

    theta_left = (np.pi + alpha) + (-2*alpha)*y + theta

    theta_bottom = (np.pi + alpha) + (np.pi - 2*alpha)*x + theta

    fcos = np.zeros(basis_b.shape)
    fsin = np.zeros(basis_b.shape)

    # take the right hand side as the sum of all integrals over all boundary edges
    # of the basis by the boundary times the boundary function
    # func = (a * cos(theta), b * sin(theta))
    J = function.J(self.geom)

    # the boundary sides are patch 1 right side, patch 4 right side, patch 3 top, patch0 top
    for (ipatch, side), theta in zip([ (1, 'right'),
                                       (4, 'right'),
                                       (3, 'top'),
                                       (0, 'top') ], [theta_bottom, theta_right, theta_top, theta_left] ):

      _domain = domain._topos[ipatch].boundary[side]
      _fcos, _fsin = _domain.integrate( [basis_b * a * np.cos(theta) * J,
                                         basis_b * b * np.sin(theta) * J], degree=10)
      fcos += _fcos
      fsin += _fsin

    return self.CM.solve(fcos), self.CM.solve(fsin)

  def break_apart(self, vec, split=False):
    ret = self.CM_disc.solve(self.T_disc @ vec)
    if split: return np.array_split(ret, self.npatches, axis=0)
    return ret

  def make_disc(self, a, b, theta=0, rot=0, return_type='NDSpline'):

    assert return_type in ('array', 'NDSpline'), NotImplementedError

    rotmat = rot_matrix(rot)

    # no rotation => just scale the reference solution by a in x and b in y-direction
    if theta == 0:
      solution = (self._reference_sol * np.array([[a, b]])) @ rotmat.T
    else:
      # get stiffness matrix and its factorisation
      A, CA = self.A, self.CA

      # assemble the Dirichlet data of the problem by projecting the ellipse
      # onto the basis by the boundary
      fcos, fsin = rotmat @ np.stack(self.boundary_correspondence(a, b, theta), axis=0)

      # solution is a list of the solution vector for the x and for the y coord
      solution = []
      for f in (fcos, fsin):
        # constraint vector that is initially empty
        mycons = NanVec(A.shape[:1])

        # on the positions that are frozen from the dirichlet data, substitute the
        # Dirichlet data
        mycons[self.freezemask] = f

        # make the right hand side corresponding to the elimination of the
        # DOFs that are frozen from the Dirichlet BC
        rhs = -(self.A @ (mycons | 0))[self.freemask]

        # solve for the inner DOFs
        mysol = CA.solve(rhs)

        # fill the solution vector with the solution at the indices that
        # correspond to DOfs
        mycons[self.freemask] = mysol

        # add to solution list
        solution.append(mycons.view(np.ndarray))
      solution = np.stack(solution, axis=1)

    solution = self.break_apart(solution, split=True)

    if return_type == 'array':
      return solution

    ret = np.stack([ np.concatenate([pnts, np.zeros((len(pnts), 1), dtype=float)], axis=1) for pnts in solution ], axis=1)
    return NDSpline(self.tknotvector, ret)
