"""
Module defining knotvectors. The main objects are :class:`UnivariateKnotVector`
and :class:`TensorKnotVector`. The latter is a vectorized version of the former.
"""

from ..util import _round_array, isincreasing, np, _, \
                   frozen_cached_property, gauss_quadrature
from ..types import Immutable, ensure_same_class, Int, Numeric, AnyIntSeq, \
                    AnyNumericSeq, FloatArray, IntArray, AnySequence, \
                    NumericArray
from ..err import EmptyContainerError
from .meta import TensorKnotVectorMeta
from ._jit_spl import _call1D, nonzero_bsplines_deriv_vectorized
from .aux import freeze_csr, sparse_kron

from itertools import starmap
from functools import partial, lru_cache
from typing import List, Sequence, Self, Any, Optional, Dict, cast, Callable

from scipy import sparse
from scipy.sparse import linalg as splinalg


# XXX: I would like to use functools.total_ordering but it is slightly out of
#      place here because two knotvectors can simultaneously satisfy a < b is
#      False and b < a is False.
#      Couldn't quite get it to work as intended with total_ordering.
class UnivariateKnotVector(Immutable):
  """
  Basic knot-vector object.

  Parameters
  ----------
  knotvalues : Array-like of Int / Float
    Strictly increasing one-dimensional sequence of numbers representing the
    knotvector's knots without repretitions.
  degree : :class:`int`
    Integer value representing the knotvector's polynomial degree.
  knotmultiplicities : Array-like of integers, optional
    Positive sequence of length len(self.knotvalues) representing each knot's
    multiplicity. Must represent an open knotvector, i.e,
        knotmultiplicities[0] == knotmultiplicities[-1] == self.degree + 1.
    If not passed, defaults to an open knotvector without interior knot
    repretitions.
  """
  # XXX: support for periodic knotvectors

  def __init__(self, knotvalues: AnyNumericSeq,
                     degree: Int = 3,
                     knotmultiplicities: Optional[AnyIntSeq] = None):
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

    if any( self.knotmultiplicities[i] != self.degree + 1 for i in (0, -1) ):
      raise NotImplementedError('Currently, only open knotvectors are supported.')

    assert (0 < self.km[1:-1]).all() and (self.km[1:-1] <= self.degree + 1).all()

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}[degree: {self.degree}, nknots: {len(self.knots)}]"

  @frozen_cached_property
  def knots(self) -> FloatArray:
    return np.asarray(self.knotvalues, dtype=float)

  @frozen_cached_property
  def km(self) -> IntArray:
    return np.asarray(self.knotmultiplicities, dtype=int)

  @frozen_cached_property
  def greville(self) -> FloatArray:
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
  def repeated_knots(self) -> FloatArray:
      """ Repeat knots by their knotmultiplicity. """
      return np.repeat(self.knots, self.knotmultiplicities)

  @frozen_cached_property
  def dx(self) -> FloatArray:
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

  def collocate(self, abscissae: AnyNumericSeq, dx: Int = 0) -> sparse.csr_matrix:
    """
    Collocation matrix X over the abscissae ``abscissae``.
    Generates a sparse matrix X such that the solution x of the system
    (X @ X.T) @ x = X @ data contains the control points with respect
    to the basis associated with ``self`` of the least squares problem of
    fitting the abscissae against data.

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
                                         self.degree, e, dx)
                                 for e in np.eye(self.dim)], axis=0))
    return sparse.csr_matrix(ret)

  def _refine(self) -> Self:
    """ Uniformly refine the entire knotvector once. """
    nknots = len(self.knots)
    knots = np.insert(self.knots, range(1, nknots), (self.knots[:-1] + self.knots[1:])/2.0)
    knotmultiplicities = np.insert(self.knotmultiplicities, range(1, nknots), 1)
    return self.__class__(knots, degree=self.degree,
                                 knotmultiplicities=knotmultiplicities)

  def refine(self, n: Int = 1) -> Self:
    """ Uniformly refine the entire knotvector ``n`` times. """
    assert (n := int(n)) >= 0
    if n == 0:
      return self
    return self._refine().refine(n-1)

  def ref_by(self, indices: Int | AnyIntSeq) -> Self:
    """ Halve elements contained in ``indices``. """
    if isinstance(indices, Int):
      indices = indices,
    indices = np.asarray(indices, dtype=int)
    add = (self.knots[indices + 1] + self.knots[indices]) / 2.0
    return self.__class__(knotvalues=np.insert(self.knots, indices+1, add),
                          degree=self.degree,
                          knotmultiplicities=np.insert(self.km, indices+1, 1))

  def add_knots(self, knotvalues: Numeric | AnyNumericSeq) -> Self:
    """
    Add new knots to the knotvector. Adding knots beyond the knotvector's
    limits is currently prohibited.
    """
    if isinstance(knotvalues, Numeric):
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

  def raise_multiplicities(self, indices: Int | AnyIntSeq,
                                 amounts: Int | AnyIntSeq) -> Self:
    """
    Raise the knotmulitplicities corresponding to ``indices`` by ``amount``.
    """
    if isinstance(indices, Int):
      indices = indices,
    indices = np.asarray(indices, dtype=int)

    if isinstance(amounts, Int):
      amounts = (amounts,) * len(indices)

    km = np.asarray(self.knotmultiplicities, dtype=int)
    km[indices] += np.asarray(amounts, dtype=int)
    return self._edit(knotmultiplicities=km)

  def integrate(self, dx: Int = 0):
    """ See ``univariate_integral``. """
    return univariate_integral(self, dx=dx)

  @property
  def M(self):
    return self.integrate(dx=0)

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
def univariate_integral(uknotvector: UnivariateKnotVector, dx: Int = 0) -> sparse.csr_matrix:
  r"""
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

  maxrep = uknotvector.km[1:-1].max()
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


def _empty_csr(ndofs):
  return sparse.csr_matrix((ndofs, ndofs))


class TensorKnotVector(Immutable, metaclass=TensorKnotVectorMeta):

  """
  Class representing a tensorial knotvector.
  Takes as input a sequence of :class:`UnivariateKnotVector` and vectorizes
  all relevant properties / methods of its inputs.

  Most of the vectorization is implemented via the :type:`TensorKnotVectorMeta`
  metaclass.
  """

  # XXX: In the long run, UnivariateKnotVector, TensorKnotVector should be
  #      replaced by a more general NDKnotVector class.

  # through metaclass vectorization inferred properties
  dx: tuple
  km: tuple
  dim: tuple
  knots: tuple
  degree: tuple
  nelems: tuple
  greville: tuple
  repeated_knots: tuple

  # through metaclass vectorization inferred methods
  flip: Callable
  refine: Callable
  ref_by: Callable
  add_knots: Callable
  raise_multiplicities: Callable

  __or__: Callable
  __and__: Callable

  __lt__: Callable  # all comparisons must hold for ``True``
  __gt__: Callable
  __le__: Callable
  __ge__: Callable

  def __init__(self, knotvectors: AnySequence[UnivariateKnotVector]):
    self.knotvectors = tuple(map(as_UnivariateKnotVector, knotvectors))

  def __iter__(self):
    """ By default we iterate over ``self.knotvectors``. """
    yield from self.knotvectors

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}[degree: {self.degree}, nknots: {tuple(map(len, self.knots))}]"

  def __bool__(self):
    return bool(len(self))

  @property
  def ndim(self) -> int:
    """ Number of knotvectors. """
    return len(self.knotvectors)

  @property
  def ndofs(self) -> np.int_:
    """ Total number of DOFs. """
    return np.prod(self.dim).astype(int)

  def __len__(self) -> int:
    return self.ndim

  def collocate(self, *list_of_abscissae, dx: AnyIntSeq = ()) -> sparse.csr_matrix:
    """
    Tensor-product version of ``UnivariateKnotVector.collocate``.
    """
    if isinstance(dx, Int):
      assert self
      dx = (dx,) * len(self)
    if len(dx) == 0:
      dx = (0,) * len(self)
    assert len( (dx := tuple(dx)) ) == len(list_of_abscissae) == len(self)
    mats = tuple( kv.collocate(absc, dx=_dx)
                  for kv, absc, _dx, in zip(self, list_of_abscissae, dx) )
    return sparse_kron(sparse.eye(1), *mats)

  @property
  def M(self):
    """ Tensor-product version of ``UnivariateKnotVector.A`` """
    return sparse_kron(sparse.eye(1), *(kv.M for kv in self))

  @property
  def A(self):
    """ Tensor-product version of ``UnivariateKnotVector.A`` """
    return sum( [sparse_kron(*(kv.integrate(dx=i) for kv, i in zip(self, row)))
                for row in np.eye(len(self)).astype(int)],
                start=_empty_csr(self.ndofs) )

  @property
  def D(self):
    """ Tensor-product version of ``UnivariateKnotVector.D`` """
    return sum( [sparse_kron(*(kv.integrate(dx=i) for kv, i in zip(self, row)))
                for row in (2 * np.eye(len(self))).astype(int)],
                start=_empty_csr(self.ndofs) )

  def fit(self, list_of_abscissae: Sequence[AnyNumericSeq],
                data: NumericArray,
                lam0: Numeric = 1e-5,
                lam1: Numeric = 0                         ):
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
    data = np.asarray(data, dtype=float)

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

  def __mul__(self, other: Any):
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


KnotVectorType = UnivariateKnotVector | TensorKnotVector | \
                 Sequence[UnivariateKnotVector]


def as_TensorKnotVector(kv: KnotVectorType) -> TensorKnotVector:
  if isinstance(kv, TensorKnotVector):
    return kv
  if isinstance(kv, UnivariateKnotVector):
    kv = kv,
  return TensorKnotVector(tuple(kv))
