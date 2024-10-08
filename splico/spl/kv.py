from ..util import _round_array, isincreasing, np, HashMixin, frozen_cached_property, gauss_quadrature, _
from ._jit_spl import _call1D, nonzero_bsplines_deriv_vectorized

from itertools import starmap
from functools import partial, lru_cache, reduce
from typing import List, Sequence, Self, Any

from scipy import sparse
from scipy.sparse import linalg as splinalg


class UnivariateKnotVector(HashMixin):
  'Basic knot-vector object'

  _items = 'knotvalues', 'degree', 'knotmultiplicities'

  def __init__(self, knotvalues: Sequence[np.float_ | np.int_ | float | int],
                     degree: int = 3,
                     knotmultiplicities: Sequence[int] | None = None):
    knotvalues = _round_array(knotvalues)
    assert isincreasing(knotvalues), 'The knot sequence needs to be strictly increasing.'
    # XXX: failswitch in case knot sequence is too short for specified degree
    self.knotvalues = tuple(map(float, knotvalues))
    self.degree = int(degree)
    if knotmultiplicities is None:
      start = self.degree + 1
      knotmultiplicities = [start] + [1]*(len(self.knotvalues) - 2) + [start]
    self.knotmultiplicities = tuple(map(int, knotmultiplicities))
    assert all(i <= self.degree + 1 for i in self.knotmultiplicities)
    assert len(self.knotvalues) == len(self.knotmultiplicities)

  @frozen_cached_property
  def knots(self) -> np.ndarray:
    return np.asarray(self.knotvalues, dtype=float)

  @property
  def nelems(self) -> int:
    """ Return the number of elements. """
    return len(self.knots) - 1

  @property
  def dim(self) -> int:
    """ Amount of basis functions resulting from knot vector. """
    return np.sum(self.knotmultiplicities[:-1])

  def repeat_knots(self) -> np.ndarray:
    """ Repeat knots by their knotmultiplicity. """
    return np.repeat(self.knots, self.knotmultiplicities)

  @frozen_cached_property
  def dx(self) -> np.ndarray:
    """ Return the distance between distinct knots. """
    return np.diff(self.knots)

  def flip(self) -> Self:
    """
      Flip the knotvector.

      >>> knots = np.array([1.0, 1.2, 1.6, 2.0])
      >>> kv = UnivariateKnotVector(knots)
      >>> kv.knots
      >>> np.array([1.0, 1.2, 1.6, 2.0])
      >>> kv.flip().knots
      >>> np.array([1.0, 1.4, 1.8, 2.0])
    """
    return self.__class__(degree=self.degree,
                          knotmultiplicities=self.knotmultiplicities[::-1],
                          knotvalues=np.array([0, *self.dx[::-1]]).cumsum() + self.knots[0])

  def collocate(self, abscissae: Sequence[int | float] | np.ndarray, dx: int = 0):
    """
      Collocation matrix `X` over the abscissae `abscissae`.
      Generates a sparse matrix X such that the solution `x` of the system
      (X @ X.T) @ x = X @ data contains the control points with respect
      to the basis associated with `self` of the least squares problem of fitting
      the abscissae against `data`.

      Parameters
      ----------

      abscissae: Array-like of ndim == 1 containing the fitting abscissaer.
      dx: The basis derivative associated with the collocation matrix. If
          dx == 0, it's the ordinary collocation matrix.

      Returns
      -------

      X: Collocation matrix of type sparse.csr_matrix.
    """
    abscissae = np.asarray(abscissae, dtype=float)
    a, *ignore, b = self.knots
    assert a <= abscissae.min() <= abscissae.max() <= b

    # XXX: obviously we need to find a better solution for this.
    ret = _round_array(np.stack([_call1D(abscissae,
                                         self.repeat_knots(),
                                         self.degree, e, dx) for e in np.eye(self.dim)], axis=0))
    return sparse.csr_matrix(ret)

  def _refine(self):
    nknots = len(self.knots)
    knots = np.insert(self.knots, range(1, nknots), (self.knots[:-1] + self.knots[1:])/2.0)
    knotmultiplicities = np.insert(self.knotmultiplicities, range(1, nknots), [1]*(nknots-1))
    return UnivariateKnotVector(knots, degree=self.degree,
                                       knotmultiplicities=knotmultiplicities)

  def refine(self, n=1):
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self._refine().refine(n-1)

  def integrate(self, dx=0):
    """ See `univariate_integral`. """
    return univariate_integral(self, dx=dx).copy()

  @property
  def A(self):
    """ Parametric stiffness matrix. """
    return self.integrate(dx=1)

  @property
  def D(self):
    """ Parametric second-order equivalent of the stiffness matrix. """
    return self.integrate(dx=2)

  def __mul__(self, other):
    """
      Multiplying by `UnivariateKnotVector` or `TensorKnotVector`
      yields a `TensorKnotVector`.
    """
    if other.__class__ is self.__class__:
      return TensorKnotVector([self, other])
    if isinstance(other, TensorKnotVector):
      return TensorKnotVector([self, *other])
    raise TypeError

  def __rmul__(self, other):
    assert isinstance(other, TensorKnotVector)
    return TensorKnotVector([*other, self])


@lru_cache(maxsize=32)
def univariate_integral(uknotvector: UnivariateKnotVector, dx: int = 0) -> sparse.csr_matrix:
  """
    Compute the matrix with entries M_ij = \int_(a, b) phi_i^(dx) phi_j^(dx) dx,
    where uknotvector.knots == a, *ignore, b.

    Parameters
    ----------

    uknotvector : `UnivariateKnotVector`
      The univariate knotvector over which the phi_i are defined.
    dx : `int`
      The derivative order.
  """

  # XXX: jit-compile with Numba

  assert uknotvector.degree >= dx and np.asarray(uknotvector.knotmultiplicities, dtype=int)[1:-1].max() <= uknotvector.degree + 1 - dx
  knots, ext_knots = uknotvector.knots, uknotvector.repeat_knots()
  gauss = partial(gauss_quadrature, order=uknotvector.degree + 1)
  M = sparse.lil_matrix((uknotvector.dim,) * 2)

  for i, (weights, points) in enumerate(starmap(gauss, zip(knots, knots[1:]))):
    dofs = np.arange(i, i + uknotvector.degree + 1)
    shapeF = nonzero_bsplines_deriv_vectorized(ext_knots, uknotvector.degree, points, dx)
    M[np.ix_(dofs, dofs)] += (weights[:, _, _] * shapeF[..., _] * shapeF[:, _]).sum(0)

  return M.tocsr()


def as_UnivariateKnotVector(kv: UnivariateKnotVector | Any) -> UnivariateKnotVector:
  if isinstance(kv, UnivariateKnotVector):
    return kv
  return UnivariateKnotVector(*kv)


def sparse_kron(*_mats: Sequence[sparse.spmatrix | np.ndarray]) -> sparse.csr_matrix:
  assert len(_mats) >= 1
  mats: List[sparse.spmatrix] = list(map(sparse.csr_matrix, _mats))
  if len(mats) == 1:
    return mats[0]
  return reduce(lambda x, y: sparse.kron(x, y, format='csr'), mats)


class TensorKnotVector(HashMixin):

  # XXX: Maybe indirectly subclass np.ndarray via the __array_ufunc__ protocol.
  # XXX: docstring

  _items = 'knotvectors',

  def __init__(self, knotvectors: List[UnivariateKnotVector]):
    self.knotvectors = tuple(map(as_UnivariateKnotVector, knotvectors))

  def __iter__(self):
    """ By default we iterate over `self.knotvectors`. """
    yield from self.knotvectors

  # XXX: We may want to find a prettier solution for the following

  def _vectorize(name: str, return_type=None):
    """
      Vectorize an operation and apply it to each `UnivariateKnotVector`
      in `self.knotvectors`.

      Parameters
      ----------

      name: `str` the name of the method of `UnivariateKnotVector` that is to
            be applied to all `UnivariateKnotVector`'s in `self`.
      return_type: Container-type to iterate the result of the vectorization
                   into. Defaults to `self.__class__`.

      Returns
      -------

      func: `Callable` bound method that is added to the catalogue of functionality.
    """
    def wrapper(self, *args, **kwargs):
      rt = self.__class__ if return_type is None else return_type
      return rt([ getattr(e, name)(*(a[i] for a in args), **{k: v[i] for k, v in kwargs.items()}) for i, e in enumerate(self) ])
    return wrapper

  def _prop_wrapper(name: str, return_type=tuple):
    """
      Same as `_vectorize` but creates a property that doesn't take any arguments
      other than `self`.
    """
    @property
    def wrapper(self):
      return return_type([getattr(e, name) for e in self])
    return wrapper

  dx = _prop_wrapper('dx')
  knots = _prop_wrapper('knots')
  degree = _prop_wrapper('degree')
  knotmultiplicities = _prop_wrapper('knotmultiplicities')
  nelems = _prop_wrapper('nelems')
  dim = _prop_wrapper('dim')
  repeat_knots = _vectorize('repeat_knots', tuple)
  flip = _vectorize('flip')
  refine = _vectorize('refine')

  ###

  @property
  def ndim(self):
    """ Number of knotvectors. """
    return len(self.knotvectors)

  @property
  def ndofs(self):
    """ Total number of DOFs. """
    return np.prod(self.dim)

  def __len__(self):
    return self.ndim

  def collocate(self, *list_of_abscissae: Sequence | np.ndarray, dx: int | np.int_ | Sequence[int] | None = None):
    """
      Tensor-product version of UnivariateKnotVector.collocate.
    """
    if isinstance(dx, (int, np.int_)):
      dx = (int(dx),) * len(self)
    if dx is None:
      dx = (0,) * len(self)
    assert len( (dx := tuple(dx)) ) == len(list_of_abscissae) == len(self)
    mats = [ kv.collocate(absc, dx=_dx) for kv, absc, _dx, in zip(self, list_of_abscissae, dx) ]
    return sparse_kron(*mats)

  @property
  def A(self):
    """ Tensor-product version of UnivariateKnotVector.A """
    return sum( sparse_kron(*(kv.integrate(dx=i) for kv, i in zip(self, row))) for row in np.eye(len(self)).astype(int) )

  @property
  def D(self):
    """ Tensor-product version of UnivariateKnotVector.D """
    return sum( sparse_kron(*(kv.integrate(dx=i) for kv, i in zip(self, row))) for row in (2 * np.eye(len(self))).astype(int) )

  def fit(self, list_of_abscissae: Sequence[np.ndarray | Sequence[int | float]],
                data: Sequence | np.ndarray,
                lam0: float | int = 1e-5,
                lam1: float | int = 0                                           ):
    """
      Fit a spline to a set of points and vertices in the least squares sense
      with (optional) added regularisation using `self` as knotvector.

      Parameters
      ----------

      list_of_abscissae: Container-like of fitting abscissae in each parametric
                         direction. The number must match the dimensionality
                         of the knotvector. The vertices follow from a tensor
                         product.
      data: Array-like of data points. The shape must satisfy
            shape[0] == len(list_of_abscissae[0]) * len(list_of_abscissae[1]) * ...
            May also be tensorial, i.e., shape[1:] != () in which case each entry
            is fit individually.
      lam0: Added least squares first order smoothness regularisation.
            Defaults to 1e-5.
      lam1: Added least squares second order regularisation. Is ommitted
            by default.

      Returns
      -------
      ret: NDSpline of shape data.shape[1:] that follows from a least squares
           fit using `self` as a knotvector.

      >>> kv = UnivariateKnotVector(np.linspace(0, 1, 11))
      >>> kv = kv * kv  # two-dimensional tensor knotvector
      >>> absc = np.linspace(0, 1, 21)
      >>> x, y = map(np.ravel, np.meshgrid(*[np.linspace(0, 1, 11)]*2))
      >>> data = np.stack([x, 1 + x + y], axis=1)
      >>> spline = kv.fit([absc, absc], data)
      >>> (np.abs( spline(x, y) - data ) < 1e-2).all()
      >>> True
    """

    list_of_abscissae = list(map(np.asarray, list_of_abscissae))
    data = np.asarray(data)
    assert data.shape[:1] == (np.multiply.reduce(list(map(len, list_of_abscissae))),)
    assert all(lam >= 0 for lam in (lam0, lam1))
    X = self.collocate(*list_of_abscissae)
    M = X @ X.T
    if lam0 != 0:
      M += lam0 * self.A
    if lam1 != 0:
      M += lam1 * self.D
    rhs = X @ data.reshape((-1,) + (data.shape[1:] and (np.prod(data.shape[1:]),)))
    from .spline import NDSpline
    return NDSpline(self, splinalg.spsolve(M, rhs).reshape((-1,) + data.shape[1:]))

  def __mul__(self, other):
    """
      Multiplying by a `TensorKnotVector` or a `UnivariateKnotVector`
      simply gives a bigger knotvector.
    """
    if isinstance(other, UnivariateKnotVector):
      knotvectors = self.knotvectors + (other,)
    elif isinstance(other, TensorKnotVector):
      knotvectors = self.knotvectors + other.knotvectors
    else:
      return NotImplemented
    return TensorKnotVector(knotvectors)

  del _vectorize
  del _prop_wrapper


def as_TensorKnotVector(kv: TensorKnotVector | UnivariateKnotVector | Sequence[UnivariateKnotVector]) -> TensorKnotVector:
  if isinstance(kv, TensorKnotVector):
    return kv
  if isinstance(kv, UnivariateKnotVector):
    kv = kv,
  return TensorKnotVector(list(kv))
