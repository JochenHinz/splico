"""
Module defining knotvectors. The main objects are :class:`UnivariateKnotVector`
and :class:`TensorKnotVector`. The latter is a vectorized version of the former.
"""

from ..util import _round_array, isincreasing, np, _, \
                   frozen_cached_property, gauss_quadrature
from ..types import Immutable, ensure_same_class, ensure_same_length, Int, \
                    Numeric
from ..err import EmptyContainerError
from ._jit_spl import _call1D, nonzero_bsplines_deriv_vectorized
from .aux import freeze_csr, sparse_kron

from itertools import starmap
from functools import partial, lru_cache
from typing import List, Sequence, Self, Any, Optional, Tuple, Dict, cast
import operator

from scipy import sparse
from scipy.sparse import linalg as splinalg
from numpy.typing import NDArray


AnySequence = Sequence | Tuple | NDArray


# XXX: I would like to use functools.total_ordering but it is slightly out of
#      place here because two knotvectors can simultaneously satisfy a < b is
#      False and b < a is False.
class UnivariateKnotVector(Immutable):
  """
  Basic knot-vector object
  """
  # XXX: support for periodic knotvectors

  def __init__(self, knotvalues: AnySequence,
                     degree: Int = 3,
                     knotmultiplicities: Optional[AnySequence] = None):
    knotvalues = _round_array(knotvalues)
    assert isincreasing(knotvalues), 'The knot sequence needs to be strictly increasing.'

    # XXX: failswitch in case knot sequence is too short for specified degree
    self.knotvalues = tuple(map(float, knotvalues))
    assert len(self.knotvalues) >= 2

    self.degree = int(degree)
    assert self.degree > 0

    if knotmultiplicities is None:
      start = self.degree + 1
      knotmultiplicities = (start,) + (1,)*(len(self.knotvalues) - 2) + (start,)

    self.knotmultiplicities = tuple(map(int, knotmultiplicities))

    assert len(self.knotvalues) == len(self.knotmultiplicities)

    if self.knotmultiplicities[0] <= self.degree or \
       self.knotmultiplicities[-1] <= self.degree:
      raise NotImplementedError('Currently, only open knotvectors are supported.')

    assert all(0 < i <= self.degree + 1 for i in self.knotmultiplicities[1:-1])

  def __repr__(self):
    return f"{self.__class__.__name__}[degree: {self.degree}, nknots: {len(self.knots)}]"

  @frozen_cached_property
  def knots(self) -> NDArray[np.float_]:
    return np.asarray(self.knotvalues, dtype=float)

  @frozen_cached_property
  def km(self) -> NDArray[np.int_]:
    return np.asarray(self.knotmultiplicities, dtype=int)

  @frozen_cached_property
  def greville(self):
    """ Compute the Greville points. """
    knots = self.repeated_knots
    return knots[np.arange(self.degree, dtype=int)[_] +
                 np.arange(1, self.dim+1, dtype=int)[:, _]].sum(1) / self.degree

  @property
  def nelems(self) -> int:
    """ Return the number of elements. """
    return len(self.knots) - 1

  @property
  def dim(self) -> int:
    """ Amount of basis functions resulting from knot vector. """
    return np.sum(self.knotmultiplicities[:-1])

  @frozen_cached_property
  def repeated_knots(self) -> NDArray[np.float_]:
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

  __neg__ = flip

  def collocate(self, abscissae: Sequence[int | float] | np.ndarray, dx: int = 0):
    """
    Collocation matrix X over the abscissae ``abscissae``.
    Generates a sparse matrix X such that the solution x of the system
    (X @ X.T) @ x = X @ data contains the control points with respect
    to the basis associated with ``self`` of the least squares problem of fitting
    the abscissae against data.

    Parameters
    ----------
    abscissae : Array-like of ndim == 1
        The fitting abscissae.
    dx : :class:`int`
        The basis derivative associated with the collocation matrix. If
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
                                         self.repeated_knots,
                                         self.degree, e, dx) for e in np.eye(self.dim)], axis=0))
    return sparse.csr_matrix(ret)

  def _refine(self) -> Self:
    """ Uniformly refine the entire knotvector once. """
    nknots = len(self.knots)
    knots = np.insert(self.knots, range(1, nknots), (self.knots[:-1] + self.knots[1:])/2.0)
    knotmultiplicities = np.insert(self.knotmultiplicities, range(1, nknots), 1)
    return self.__class__(knots, degree=self.degree,
                                 knotmultiplicities=knotmultiplicities)

  def refine(self, n: int = 1) -> Self:
    """ Uniformly refine the entire knotvector ``n`` times. """
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self._refine().refine(n-1)

  def ref_by(self, indices: Numeric | Sequence[Any] | Tuple[Any] | np.ndarray) -> Self:
    """ Halve elements contained in ``indices``. """
    if np.isscalar(indices):
      indices = indices,
    indices = np.asarray(indices, dtype=int)
    add = (self.knots[indices + 1] + self.knots[indices]) / 2.0
    return self.__class__(knotvalues=np.insert(self.knots, indices+1, add),
                          degree=self.degree,
                          knotmultiplicities=np.insert(self.km, indices+1, 1))

  def add_knots(self, knotvalues: Numeric | AnySequence) -> Self:
    """
    Add new knots to the knotvector. Adding knots beyond the knotvector's
    limits is currently prohibited.
    """
    if np.isscalar(knotvalues):
      knotvalues = knotvalues,
    knotvalues = np.asarray(knotvalues, dtype=float)

    a, *inner_knots, b = self.knotvalues
    if (a > knotvalues).any() or (knotvalues > b).any():
      raise ValueError("Cannot add knots beyond the knotvector's domain.")

    new_knots = np.unique(np.concatenate([self.knots, knotvalues]))
    if len(new_knots) == len(self.knots):
      return self

    map_kv_km = dict(zip(self.knotvalues, self.knotmultiplicities, strict=True))
    new_km = [map_kv_km.get(val) or 1 for val in new_knots]
    return self.__class__(new_knots, degree=self.degree,
                                     knotmultiplicities=new_km)

  def raise_multiplicities(self, indices: Int | AnySequence,
                                 amounts: Int | AnySequence) -> Self:
    """
    Raise the knotmulitplicities corresponding to ``indices`` by ``amount``.
    """
    if np.isscalar(indices):
      indices = indices,
    indices = np.asarray(indices, dtype=int)

    if np.isscalar(amounts):
      amounts = (amounts,) * len(indices)

    km = np.asarray(self.knotmultiplicities, dtype=int)
    km[indices] += np.asarray(amounts, dtype=int)
    return self._edit(knotmultiplicities=km)

  def integrate(self, dx=0):
    """ See ``univariate_integral``. """
    return univariate_integral(self, dx=dx)

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
    Multiplying by :class:`UnivariateKnotVector` or :class:`TensorKnotVector`
    yields a :class:`TensorKnotVector`.
    """
    if other.__class__ is self.__class__:
      return TensorKnotVector([self, other])
    if isinstance(other, TensorKnotVector):
      return TensorKnotVector([self, *other])
    return NotImplemented

  def __rmul__(self, other: Any):
    assert isinstance(other, TensorKnotVector)
    return TensorKnotVector([*other, self])

  @ensure_same_class
  def __and__(self, other):
    """
    Create a knotvector from the shared knots.
    If a knot is shared, the larger of the two knotmultiplicities is taken.
    """
    knots = np.intersect1d(self.knots, other.knots)
    if self.degree != other.degree or not len(knots):
      raise EmptyContainerError("Found empty intersection.")
    km0, km1 = map(lambda x: x.km[np.searchsorted(x.knots, knots)], (self, other))
    km = np.max(np.stack([km0, km1], axis=0), axis=0)
    return UnivariateKnotVector(knots, degree=self.degree,
                                       knotmultiplicities=km)

  @ensure_same_class
  def __or__(self, other: Any):
    """
    Take the union of two :class:`UnivariateKnotVector`s.
    """
    if not other.degree == self.degree:
      raise NotImplementedError("Found differing polynomial degrees.")

    all_knots = np.concatenate([self.knots, other.knots])
    all_kms = np.concatenate([self.km, other.km])

    map_knots_km: Dict[np.float_, List[Int]] = {}
    for val, km in zip(all_knots, all_kms):
      map_knots_km.setdefault(val, []).append(km)

    # mypy still complains after recasting type ... nothing I can do.
    map_knots_km = cast(Dict[np.float_, Int],
                        {key: max(val) for key, val in map_knots_km.items()})

    knots = np.unique(all_knots)
    knotmultiplicities = np.array([map_knots_km[val] for val in knots], dtype=int)

    return UnivariateKnotVector(knots, degree=self.degree,
                                       knotmultiplicities=knotmultiplicities)

  @ensure_same_class
  def __lt__(self, other):
    """
      Check if the basis corresponding to ``self`` is contained in the
      basis of ``other`` but not equal.
    """
    if (dp := other.degree - self.degree) < 0 or len(np.setdiff1d(self.knots, other.knots)) != 0:
      return False
    if self == other:
      return False
    return (self.km + dp <= other.km[np.searchsorted(other.knots, self.knots)]).all()

  @ensure_same_class
  def __gt__(self, other):  # see if self contains the basis of other
    return other < self

  def __le__(self, other):
    return self == other or self < other

  def __ge__(self, other):
    return self == other or self > other


@lru_cache(maxsize=32)
@freeze_csr
def univariate_integral(uknotvector: UnivariateKnotVector, dx: int = 0) -> sparse.csr_matrix:
  """
  Compute the matrix with entries M_ij = \int_(a, b) phi_i^(dx) phi_j^(dx) dx,
  where ``uknotvector.knots == a, *ignore, b``.

  Parameters
  ----------

  uknotvector : :class:`UnivariateKnotVector`
    The univariate knotvector over which the phi_i are defined.
  dx : :class:`int`
    The derivative order.
  """

  # XXX: jit-compile with Numba using COO-format instead of lil.

  maxrep = np.asarray(uknotvector.knotmultiplicities, dtype=int)[1:-1].max()
  assert uknotvector.degree >= dx and maxrep <= uknotvector.degree + 1 - dx

  knots, ext_knots = uknotvector.knots, uknotvector.repeated_knots

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


class TensorKnotVector(Immutable):

  # XXX: Maybe indirectly subclass np.ndarray via the __array_ufunc__ protocol.
  # XXX: In the long run, UnivariateKnotVector, TensorKnotVector should be
  #      replaced by a more general NDKnotVector class.
  # XXX: docstring

  def __init__(self, knotvectors: Sequence[UnivariateKnotVector]):
    self.knotvectors = tuple(map(as_UnivariateKnotVector, knotvectors))
    if not self.knotvectors:
      raise EmptyContainerError("Cannot instantiate from empty Sequence.")

  def __iter__(self):
    """ By default we iterate over ``self.knotvectors``. """
    yield from self.knotvectors

  def __repr__(self):
    return f"{self.__class__.__name__}[degree: {self.degree}, nknots: {tuple(map(len, self.knots))}]"

  # XXX: We may want to find a prettier solution for the following

  @staticmethod
  def _vectorize_with_indices(name: str):
    """
    Vectorize an operation and apply it to each :class:`UnivariateKnotVector`
    in ``self.knotvectors``.

    Parameters
    ----------
    name: :class:`str`
    The name of the method of :class:`UnivariateKnotVector` that is to
    be applied to all :class:`UnivariateKnotVector`s in `self`.

    Returns
    -------
    func: :class:`Callable`
        Bound method that is added to the catalogue of functionality.

    The resulting function's syntax is the following:
    >>> kv
      UnivariateKnotVector[...]
    >>> tkv = kv * kv * kv
    >>> tkv.refine(..., n=[1, 0, 1])

    Here the ``...`` (or None) indicates that the operation should be applied to
    all :class:`UnivariateKnotVector`s in ``self``, where the i-th knotvector
    receives input ``n[i]``.

    The return type is always :class:`self.__class__`.

    Similarly, we may pass the indices explicitly, for instance:
    >>> tkv.refine([0, 2], [1, 1])
    >>> tkv.refine(..., n=[1, 0, 1]) == tkv.refine([0, 2], [1, 1])
      True

    Here the knotvector corresponding to the i-th entry in `indices` receives
    ``n[i]`` as input.
    """
    def wrapper(self, indices, *args, **kwargs):
      if np.isscalar(indices):
        indices = indices,
      elif indices is Ellipsis or indices is None:
        indices = range(len(self))
      indices = list(indices)
      assert all( -len(self) <= i < len(self) for i in indices )
      indices = [ i % len(self) for i in indices ]
      _self = list(self)
      assert all( len(a) == len(indices) for a in args ) and \
             all( len(val) == len(indices) for val in kwargs.values() )
      for j, i in enumerate(indices):
        _self[i] = getattr(_self[i], name)(*(a[j] for a in args),
                                           **{k: v[j] for k, v in kwargs.items()})
      return self.__class__(_self)

    return wrapper

  @staticmethod
  def _vectorize_operator(name: str, return_type=None):
    """
    Vectorize an operator operation. For instance __and__.
    >>> kv0
      UnivariateKnotVector[...]
    >>> kv1
      UnivariateKnotVector[...]
    >>> kv2 = kv0 & kv1
    >>> tkv0 = TensorKnotVector([kv0] * 3)
    >>> tkv1 = TensorKnotVector([kv1] * 3)
    >>> (tkv0 & tkv1) == TensorKnotVector([kv2] * 3)
        True

    We may optionally pass a return type that differs from ``None``
    in which case the return type defaults to ``self.__class__``.
    """
    op = getattr(operator, name)

    @ensure_same_length
    @ensure_same_class
    def wrapper(self, other):
      rt = return_type or self.__class__
      return rt([op(kv0, kv1) for kv0, kv1 in zip(self, other)])
    return wrapper

  @staticmethod
  def _prop_wrapper(name: str, return_type=tuple):
    """
    Vectorize a (cached) property. Optionally takes a return container-type
    argument which the properties are iterated into.
    For instance, :class:`list` or :class:`tuple`. Defaults to :class:`tuple`.
    """
    @property
    def wrapper(self):
      return return_type([getattr(e, name) for e in self])
    return wrapper

  # vectorized operations

  dx = _prop_wrapper('dx')
  knots = _prop_wrapper('knots')
  km = _prop_wrapper('km')
  degree = _prop_wrapper('degree')
  repeated_knots = _prop_wrapper('repeated_knots')
  nelems = _prop_wrapper('nelems')
  dim = _prop_wrapper('dim')
  greville = _prop_wrapper('greville')

  flip = _vectorize_with_indices('flip')
  refine = _vectorize_with_indices('refine')
  ref_by = _vectorize_with_indices('ref_by')
  add_knots = _vectorize_with_indices('add_knots')
  raise_multiplicities = _vectorize_with_indices('raise_multiplicities')

  __and__ = _vectorize_operator('__and__')
  __or__ = _vectorize_operator('__or__')

  # all pairs have to pass operation to be true
  __lt__ = _vectorize_operator('__lt__', all)
  __gt__ = _vectorize_operator('__gt__', all)
  __le__ = _vectorize_operator('__le__', all)
  __ge__ = _vectorize_operator('__le__', all)

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

  def collocate(self, *list_of_abscissae, dx: Int | Sequence[Int] | None = None):
    """
    Tensor-product version of ``UnivariateKnotVector.collocate``.
    """
    if isinstance(dx, (int, np.int_)):
      dx = (int(dx),) * len(self)
    if dx is None:
      dx = (0,) * len(self)
    assert len( (dx := tuple(dx)) ) == len(list_of_abscissae) == len(self)
    mats = [ kv.collocate(absc, dx=_dx)
                      for kv, absc, _dx, in zip(self, list_of_abscissae, dx) ]
    return sparse_kron(*mats)

  @property
  def A(self):
    """ Tensor-product version of ``UnivariateKnotVector.A`` """
    return sum( sparse_kron(*(kv.integrate(dx=i) for kv, i in zip(self, row)))
                for row in np.eye(len(self)).astype(int) )

  @property
  def D(self):
    """ Tensor-product version of ``UnivariateKnotVector.D`` """
    return sum( sparse_kron(*(kv.integrate(dx=i) for kv, i in zip(self, row)))
                for row in (2 * np.eye(len(self))).astype(int) )

  def fit(self, list_of_abscissae: Sequence[np.ndarray | Sequence[int | float]],
                data: Sequence | np.ndarray,
                lam0: float | int = 1e-5,
                lam1: float | int = 0                                           ):
    """
    Fit a spline to a set of points and vertices in the least squares sense
    with (optional) added regularisation using ``self`` as knotvector.

    Parameters
    ----------

    list_of_abscissae: Container-like of fitting abscissae in each parametric direction
        The number must match the dimensionality of the knotvector.
        The vertices follow from a tensor product.
    data: Sequence of Array-likes of data points
        The shape must satisfy

        ``self.shape[0] == len(list_of_abscissae[0]) * len(list_of_abscissae[1]) * ...``

        May also be tensorial, i.e., ``self.shape[1:] != ()`` in which case
        each entry is fit individually.
    lam0: :class:`float` or equivalent
        Added least squares first order smoothness regularisation.
        Defaults to 1e-5.
    lam1: :class:`float` or equivalent
        Added least squares second order regularisation. Is ommitted by default.

    Returns
    -------
    ret: NDSpline of shape ``data.shape[1:]`` that follows from a least squares
         fit using ``self`` as a knotvector.

    >>> kv = UnivariateKnotVector(np.linspace(0, 1, 11))
    >>> kv = kv * kv  # two-dimensional tensor knotvector
    >>> absc = np.linspace(0, 1, 21)
    >>> x, y = map(np.ravel, np.meshgrid(*[np.linspace(0, 1, 11)]*2))
    >>> data = np.stack([x, 1 + x + y], axis=1)
    >>> spline = kv.fit([absc, absc], data)
    >>> (np.abs(spline(x, y) - data) < 1e-2).all()
        True
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
    Multiplying by a :class:`TensorKnotVector` or a
    :class:`UnivariateKnotVector` simply gives a bigger knotvector.
    """
    if isinstance(other, UnivariateKnotVector):
      knotvectors = self.knotvectors + (other,)
    elif isinstance(other, TensorKnotVector):
      knotvectors = self.knotvectors + other.knotvectors
    else:
      return NotImplemented
    return TensorKnotVector(knotvectors)

  del _vectorize_with_indices
  del _vectorize_operator
  del _prop_wrapper


KnotVectorType = UnivariateKnotVector | TensorKnotVector | \
                 Sequence[UnivariateKnotVector]


def as_TensorKnotVector(kv: KnotVectorType) -> TensorKnotVector:
  if isinstance(kv, TensorKnotVector):
    return kv
  if isinstance(kv, UnivariateKnotVector):
    kv = kv,
  return TensorKnotVector(list(kv))
