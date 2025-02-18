from .control import make_unit_disc
from .mul import multipatch, basis_spline
from .util import infer_knotvectors

from splico.spl import UnivariateKnotVector, NDSpline, NDSplineArray
from splico.util import np, frozen, _
from splico.types import NanVec, Float, Int, Numeric, Singleton

from functools import lru_cache, cached_property
from typing import Optional

from scipy.sparse import linalg as splinalg
from scipy import sparse
from nutils import function


J = function.J
sl = slice(None)


PATCHES = (2, 0, 3, 1), \
          (2, 4, 0, 6), \
          (2, 3, 4, 5), \
          (3, 1, 5, 7), \
          (4, 5, 6, 7)

KNOTVECTOR_EDGES = (0, 1), (0, 6), (0, 2)


@lru_cache(maxsize=32)
def rot_matrix(rot: Float | Int) -> np.ndarray:
  return np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])


def trampoline_template(inner_height: Float = .5,
                        inner_width: Float = .5):
  r"""
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

  # patches and patchverts are tuple. Immutable objects.

  return PATCHES, patchverts


def nutils_to_scipy(mat) -> sparse.csr_matrix:
  data = mat.export('csr')
  return sparse.csr_matrix(data, dtype=float)


class CrossSectionGenerator(Singleton):

  # TODO: docstring

  def __init__(self, kv0: UnivariateKnotVector,
                     kv1: UnivariateKnotVector,
                     kv2: UnivariateKnotVector,
                     reparam: bool = True,
                     inner_height: Float = .5,
                     inner_width: Float = .5):
    """
    kv0: edge (0, 1)
    kv1: edge (0, 6)
    kv2: edge (0, 2)
    """
    patches, patchverts = trampoline_template(inner_height, inner_width)

    # make a multipatch topology with the structure of ``trampoline_template``
    # and ``nelems`` elements per side
    # ``geom`` is the parameterisation of the parametric domain
    # ``localgeom`` maps each patch back onto the reference patch (0, 1)^2
    knotvectors = {(0, 1): kv0, (0, 6): kv1, (0, 2): kv2}
    knotvectors = infer_knotvectors(PATCHES, knotvectors)
    domain, geom, localgeom = multipatch(patches=patches,
                                         knotvectors=knotvectors,
                                         patchverts=patchverts,
                                         space='XY')

    basis = basis_spline(domain, knotvectors)
    basis_disc = basis_spline(domain, knotvectors, patchcontinuous=False)

    self.domain = domain
    self.npatches = len(self.domain._topos)
    self.basis = basis
    self.basis_disc = basis_disc
    self.geom = geom
    self.localgeom = localgeom

    self.knotvectors = (kv0 * kv2.flip(), kv2.flip() * kv1,
                                          kv1 * kv0,
                                          kv1 * kv2.flip(),
                                          kv2.flip() * kv0)
    self.segments = frozen(np.array([kv.ndofs for kv in self.knotvectors]).cumsum()[:-1])

    controlmap = make_unit_disc(domain, basis,
                                        geom,
                                        localgeom,
                                        patches,
                                        reparam=reparam)

    self.controlmap = controlmap

    self.kv0, self.kv1, self.kv2 = kv0, kv1, kv2

    # take the gradient of the basis with respect to the parametric domain ``geom``
    dbasis = basis.grad(controlmap)

    # assemble the stiffness matrix
    # convert it to csr for efficiency
    self.A = nutils_to_scipy(domain.integrate((dbasis[:, _] *
                                               dbasis[_]).sum([2]) *
                                               function.J(controlmap), degree=10))

    # get a boolean array that gives True for indices of basis functions that
    # are nonzero on the boundary
    self.freezemask = ~np.isnan( domain.boundary.project(1,
                                                         onto=basis,
                                                         geometry=geom,
                                                         ischeme='gauss6') )

    # get the associated boolean array that is True for functions NOT on the boundary
    self.freemask = ~self.freezemask

    # compute the Cholesky decomposition of the stiffness matrix restricted
    # to the free indices (i.e. not on the boundary)
    self.CA = splinalg.splu(self.A.tolil()[:, self.freemask][self.freemask].tocsc())

    # take the basis restricted to the ones that are nonzero on the boundary
    self.basis_b = basis[self.freezemask]

    # assemble the mass matrix for projecting data onto the boundary DOFs
    self.M = nutils_to_scipy(domain.
                             boundary.
                             integrate(function.outer(self.basis_b) * function.J(geom),
                                                                            degree=10 ))

    # compute the Cholesky of it
    self.CM = splinalg.splu(self.M.tocsc())

    # compute the matrices necessary for the prolongation to the discontinuous basis
    self.M_disc, self.T_disc = map(nutils_to_scipy, domain.integrate(
                                   [function.outer(self.basis_disc) * function.J(geom),
                                    self.basis_disc[:, _] *
                                    self.basis[_] * function.J(geom)], degree=10))
    # unit disc
    self._reference_sol = frozen(self.domain.project(self.controlmap,
                                                     self.basis.vector(2),
                                                     ischeme='gauss10',
                                                     geometry=self.controlmap).reshape(-1, 2))

  @cached_property
  def is_NDSpline(self):
    """
    Check if the basis is a NDSpline.
    This means that the created ellipses can be represented as NDSplines.
    """
    return self.kv0 == self.kv1 == self.kv2

  @cached_property
  def CM_disc(self):
    return splinalg.splu(self.M_disc.tocsc())

  @cached_property
  def M_splu(self):
    return splinalg.splu(nutils_to_scipy(self.domain.integrate(
                         function.outer(self.basis) * J(self.controlmap), degree=10)).tocsc())

  def boundary_correspondence(self, a: Numeric, b: Numeric, theta=0):
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
    for (ipatch, side), theta in zip( [ (1, 'right'),
                                        (4, 'right'),
                                        (3, 'top'),
                                        (0, 'top') ],
                                      [theta_bottom, theta_right, theta_top, theta_left] ):

      _domain = domain._topos[ipatch].boundary[side]
      _fcos, _fsin = _domain.integrate( [basis_b * a * np.cos(theta) * J,
                                         basis_b * b * np.sin(theta) * J], degree=10)
      fcos += _fcos
      fsin += _fsin

    return self.CM.solve(fcos), self.CM.solve(fsin)

  def break_apart(self, vec: np.ndarray, split: bool = False):
    ret = self.CM_disc.solve(self.T_disc @ vec)
    if split: return np.array_split(ret, self.segments, axis=0)
    return ret

  def make_disc(self, a: Numeric, b: Numeric, theta: Numeric = 0,
                                              rot: Numeric = 0,
                                              return_type: Optional[str] = None):

    if return_type is None:
      if self.is_NDSpline:
        return_type = 'NDSpline'
      else:
        return_type = 'NDSplineArray'

    assert return_type in ('array', 'NDSpline', 'NDSplineArray'), NotImplementedError

    if return_type == 'NDSpline':
      if not self.is_NDSpline:
        raise ValueError("The knotvectors are not the same. Cannot return a single NDSpline.")

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

    elif return_type == 'NDSpline':
      solution = np.stack([ np.concatenate([pnts, np.zeros((len(pnts), 1), dtype=float)], axis=1) for pnts in solution ], axis=1)
      return NDSpline(self.kv0 * self.kv0, solution)

    return NDSplineArray([NDSpline(kv, np.concatenate([arr, np.zeros(len(arr))[:, _]], axis=1))
                          for kv, arr in zip(self.knotvectors, solution)
                          ]).contract_all()


def cross_section_generator(kv0: UnivariateKnotVector | Int,
                            kv1: UnivariateKnotVector | Int | None = None,
                            kv2: UnivariateKnotVector | Int | None = None,
                            **kwargs):
  if isinstance(kv0, Int):
    kv0 = UnivariateKnotVector(np.linspace(0, 1, kv0+1), degree=3)
  if isinstance(kv1, Int):
    kv1 = UnivariateKnotVector(np.linspace(0, 1, kv1+1), degree=3)
  elif kv1 is None:
    kv1 = kv0
  if isinstance(kv2, Int):
    kv2 = UnivariateKnotVector(np.linspace(0, 1, kv2+1), degree=3)
  elif kv2 is None:
    kv2 = kv0
  return CrossSectionGenerator(kv0, kv1, kv2, **kwargs)


@lru_cache(maxsize=32)
def ellipse(a: Numeric, b: Numeric, nelems: Int, degree: Int = 3,
                                                 reparam: bool = True):
  kv0 = kv1 = kv2 = UnivariateKnotVector(np.linspace(0, 1, nelems+1),
                                         degree=degree)
  maker = CrossSectionGenerator(kv0, kv1, kv2, reparam=reparam)
  return maker.make_disc(a, b)
