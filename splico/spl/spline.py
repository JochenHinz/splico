"""
Module defining the NDSpline class and related functions.
@author: Jochen Hinz
"""

from ..util import _round_array, np, frozen, augment_by_zeros, _
from ..types import Immutable, FloatArray, Index, MultiIndex, NumericArray, \
                    Int, AnyIntSeq, AnyFloatSeq, LockableDict, Numeric
from ..mesh.mesh import Mesh
from ._jit_spl import call, tensor_call
from .kv import UnivariateKnotVector, TensorKnotVector, as_TensorKnotVector, \
                KnotVectorType
from .aux import tensorial_prolongation_matrix
from .meta import NDSplineMeta

from typing import Sequence, Callable, Tuple, Self, List
from types import GenericAlias
from functools import reduce, cached_property

from scipy import interpolate
from numpy.lib.mixins import NDArrayOperatorsMixin


sl = slice(_)


IMPLEMENTED_UFUNCS = (np.add, np.subtract, np.multiply, np.divide)
HANDLED_NDSPLINE_FUNCTIONS = LockableDict()


def try_broadcast_shapes(*shapes: AnyIntSeq) -> Tuple[int, ...]:
  """
  Try to broadcast the shapes of the input arrays. Throws a ValueError if
  broadcasting fails.
  """
  try:
    final_shape = np.broadcast_shapes(*shapes)
  except ValueError as ex:
    raise ValueError("Broadcasting failed with error message '{}'".format(ex))
  return final_shape


class __NDSpline_implementations__:
  """
  Dummy class for implementing various `NDSpline` ufuncs.
  They are added to `HANDLED_NDSPLINE_FUNCTIONS` to be used in the `NDSpline`
  __array_ufunc__ protocol.
  """

  @staticmethod
  def implements(np_function: np.ufunc):
    def wrapper(func):
      HANDLED_NDSPLINE_FUNCTIONS[np_function] = func
      return func
    return wrapper

  @implements(np.add)
  def add(self, other):
    """
    Add two :class:`NDSpline`s sharing the same knotvector. The controlpoints
    will be added.
    """
    assert self.knotvector == other.knotvector
    final_shape = try_broadcast_shapes(self.shape, other.shape)
    ndims = len(final_shape)
    n, m = map(len, (self.shape, other.shape))
    # if the knotvectors are the same, simply add the controlpoints together
    return NDSpline(self.knotvector,
                    self.controlpoints[(sl,) + (_,) * (ndims - n)]
                    + other.controlpoints[(sl,) + (_,) * (ndims - m)])

  @implements(np.subtract)
  def sub(self, other):
    return self + -other

  @implements(np.multiply)
  def mul(self, other):
    """
    Multiply two :class:`NDSpline`s. Creates a tensor product of the
    knotvector and controlpoints. The number of dependencies of the resulting
    spline is the sum of the dependencies of the input splines.
    """

    # tensor knotvector
    knotvector = self.knotvector * other.knotvector

    n, m = map(len, (self.shape, other.shape))

    # prepend ones to the shorter shaped NDSpline (default numpy behavior)
    myshape = (1,) * (m - n) + self.shape
    othershape = (1,) * (n - m) + other.shape

    final_shape = try_broadcast_shapes(myshape, othershape)

    # take outer product and reshape to (ncontrolpoints,) + shape
    cps = (self.controlpoints.reshape(-1, 1, *myshape) *
           other.controlpoints.reshape(1, -1, *othershape)).reshape(-1, *final_shape)

    return NDSpline(knotvector, cps)


# lock to prevent unwanted modifications
HANDLED_NDSPLINE_FUNCTIONS.lock()


class NDSpline(Immutable, NDArrayOperatorsMixin, metaclass=NDSplineMeta):
  """
  Class representing an N-dimensional spline.

  Parameters
  ----------
  knotvector : :class:`splico.spl.kv.TensorKnotVector` or
               :class:`splico.spl.kv.UnivariateKnotVector`
      The (tensorial) knotvector associated with the spline. If not yet
      tensorial, will be converted to tensorial.
  controlpoints : :class:`np.ndarray` or any array-like
      Numpy array or any array-like that can be converted to Numpy array
      representing the controlpoints. Upon conversion, ``self.controlpoints``
      must satisfy ``self.controlpoints.shape[0] == self.knotvector.ndofs``.
      Here, ``self.controlpoints[i]`` denotes the (potentially tensorial)
      control point assigned to the i-th spline function in the basis that
      corresponds to ``self.knotvector``. Is rounded to the current precision
      as set by the `splico.util.global_precision` context manager and then
      frozen. Here, self.controlpoints.shape[1:] can be anything and then
      simply represents a vectorial / tensorial set of splines all with the
      same knotvector.

  Attributes
  ----------
  knotvector: :class:`splico.spl.kv.TensorKnotVector`
      The knotvector.
  controlpoints: :class:`np.ndarray`
      The controlpoints.


  When performing default ``numpy`` arithmetic on an NDSpline, the operations
  are by default performed on ``self.controlpoints[i]`` for all i. Hence, when
  performing an airthmetic operation with a :class:`np.ndarray` ``A``,
  ``A.shape`` has to be broadcastable with ``self.controlpoints.shape[1:]``
  rather than ``self.controlpoints.shape``.

  >>> spline
  ... NDSpline<4, 5, 3>
  >>> spline.controlponts.shape
  ... (10, 4, 5, 3)
  >>> type(A)
  ... np.ndarray
  >>> A.shape
  ... (2, 4, 1, 3)
  >>> (spline + A).shape  # (4, 5, 3) becomes (1, 4, 5, 3)
  ... (2, 4, 5, 3)
  ... # this won't work because (10, 4, 5, 3) + (2, 4, 1, 3) is not broadcastable
  >>> spline.controlpoints + A
  ... ERROR
  """

  # methods inherited from `self.knotvector` via the metaclass
  refine: Callable
  ref_by: Callable
  add_knots: Callable
  raise_multiplicities: Callable

  # properties inherited from `self.knotvector`
  km: Tuple
  knots: Tuple
  degree: Tuple
  greville: Tuple
  repeated_knots: Tuple

  @classmethod
  def one(cls, knotvector):
    """
    Constant one function. Helpful when a function of one variable
    has to be made a function of the other variables as well.
    """
    knotvector = as_TensorKnotVector(knotvector)
    return cls(knotvector, np.ones(knotvector.ndofs))

  @classmethod
  def from_exact_interpolation(cls, verts: AnyFloatSeq,
                                     data: AnyFloatSeq, **scipyargs):
    """
    Classmethod for creating a an exact interpolation of a verts, data pair
    using scipy. It is then wrapped as an NDSpline.

    Scipy only supports 1D interpolation for now but is quite sophisticated.
    We may add our own interpolation routines in the future.
    """

    # XXX: parameters

    verts = np.asarray(verts, dtype=float)
    data = np.asarray(data, dtype=float)

    # For now only 1D
    assert verts.ndim == data.ndim == 1, NotImplementedError

    k = scipyargs.setdefault('k', 3)
    spline = interpolate.InterpolatedUnivariateSpline(verts, data, **scipyargs)

    coeffs, knots = spline.get_coeffs(), spline.get_knots()
    knotvector = TensorKnotVector([UnivariateKnotVector(knots, degree=k)])
    return cls(knotvector, coeffs)

  @staticmethod
  def from_least_squares(knotvector: KnotVectorType, *args, **kwargs):
    """
    Docstring: see `splico.spl.kv.TensorKnotVector.fit`.
    """
    # Contrary to ``from_exact_interpolation`` this one works for any dimension
    # convert to TensorKnotVector if not alreay
    knotvector = as_TensorKnotVector(knotvector)
    # forward to `TensorKnotVector.fit` routine
    return knotvector.fit(*args, **kwargs)

  @classmethod
  def __class_getitem__(cls, dimension):
    assert isinstance(dimension, (int, type(...)))
    return GenericAlias(cls, dimension)

  def __init__(self, knotvector: KnotVectorType, controlpoints: AnyFloatSeq):
    self.knotvector = as_TensorKnotVector(knotvector)
    self.controlpoints = frozen(_round_array(controlpoints), dtype=float)
    assert self.controlpoints.shape[0] == self.knotvector.ndofs

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}<{', '.join(map(str, self.shape))}>"

  @property
  def shape(self) -> Tuple[int, ...]:
    """ The shape of the vectorial spline. """
    return self.controlpoints.shape[1:]

  @property
  def nvars(self) -> int:
    """ Number of dependencies. """
    return self.knotvector.ndim

  @property
  def tcontrolpoints(self) -> FloatArray:
    """
    Represent the controlpoints tensorially.
    >>> spline.controlpoints.shape
    ... (36, 2, 3)
    >>> spline.knotvector.dim
    ... (3, 3, 4)
    >>> spline.tcontrolpoints.shape
    ... (3, 3, 4, 2, 3)
    """
    return self.controlpoints.reshape(self.knotvector.dim + self.shape)

  def prolong_to(self, knotvector_to: TensorKnotVector) -> Self:
    """
    Prolong the spline to a refined knotvector.
    """
    T = tensorial_prolongation_matrix(self.knotvector, knotvector_to)
    n, m = T.shape
    controlpoints = T @ self.controlpoints.reshape(m, -1)
    return self._edit(knotvector=knotvector_to,
                      controlpoints=controlpoints.reshape((n,) + self.shape))

  def __call__(self, *positions: FloatArray,
                     tensor: bool = False,
                     dx: Int | AnyIntSeq = ()):
    """
    Evaluate the spline in a set of points.

    Parameters
    ----------
    positions : :class:`tuple` of :class:`np.ndarray` or any array-like
        Positional arguments of the x, y, z, ... coordinates all represented
        by a flat array-like, such as :class:`np.ndarray`.
        Must satisfy all(len(pos) == len(positions[0]) for pos in positions)
        and len(positions) == len(self.knotvector) unless ``tensor`` is True,
        see below.
    tensor : :class:`bool`
        If true, the spline is evaluated over a meshgrid of the positions, i.e.,
        positions is converted to tuple(map(np.ravel, np.meshgrid(*positions))).
    dx : `Int` or sequence of `Int`
        Derivative order of the evaluation. Must satisfy:
        len(dx) == len(self.knotvector).
        If given by a :class:`int` it is repeated len(self.knotvector) times.

    Returns
    -------
    ret : :class:`np.ndarray``
        The evaluation array. Has shape
          (len(positions[0]),) + self.controlpoints.shape[1:]
        if tensor is False and
          (np.prod([len(po) for pos in positions]),) + self.controlpoints.shape[1:]
        else.
    """
    if isinstance(dx, Int):
      assert self.nvars  # 0-D spline doesn't have any derivatives
      dx = (dx,) * self.nvars
    dx = dx or (0,) * self.nvars
    assert len((dx := tuple(map(int, dx)))) == len(positions) == self.nvars

    if not positions:  # 0-D spline
      return self.controlpoints[0].copy()

    positions = tuple(map(np.asarray, positions))
    assert all( len((y := pos.shape)) == 1 for pos in positions )

    # if not evaluated tensorially, make sure all positions have equal length
    if not tensor:
      assert all(pos.shape == y for pos in positions)

    # reshape to matrix shape, if self.shape == (), np.prod((), dtype=int) == 1
    controlpoints = self.controlpoints.reshape(-1, np.prod(self.shape, dtype=int))

    function = {False: call, True: tensor_call}[tensor]
    ret = [function(positions, self.repeated_knots, self.degree, x, dx)
                                                      for x in controlpoints.T]

    # XXX: np.stack makes a copy, find better solution
    # (possibly do this directly in numba)
    return np.stack(ret, axis=1).reshape(-1, *self.shape)

  def tensorcall(self, *args, **kwargs):
    """ partial(self, tensor=True) """
    return self(*args, tensor=True, **kwargs)

  def __getitem__(self, item: Index | MultiIndex):
    """
    Same as np.ndarray.__getitem__ with the difference that the zeroth axis
    is ignored and the broadcasting etc. is applied to all axes with index > 0.
    >>> spline.controlpoints.shape
    ... (54, 2, 3)
    >>> spline[_].controlpoints.shape
    ... (54, 1, 2, 3)
    """
    if not isinstance(item, tuple):  # single integer or slices
      item = item,
    return self._edit(controlpoints=self.controlpoints[(sl,) + item])

  @property
  def ndim(self):
    """ Dimensionality of the spline (not the number of dependencies). """
    return self.controlpoints.ndim - 1

  def __iter__(self):
    """ Iterate over the each sub-spline the same way as a numpy array. """
    if not self.shape:
      raise TypeError('iteration over 0-d array')
    yield from (self._edit(controlpoints=controlpoints)
                        for controlpoints in self.controlpoints.swapaxes(0, 1))

  def ravel(self) -> Self:
    """ Ravel the tensorial spline. """
    shape = self.controlpoints.shape[:1] + (self.shape and (-1,))
    return self._edit(controlpoints=self.controlpoints.reshape(shape))

  def __neg__(self) -> Self:
    return -1 * self

  def sum(self, *args, axis=None) -> Self:
    """
    Same as np.sum but applied to the tail of self.controlpoints.
    >>> type(spl0)
    ... splico.spl.spline.NDSpline
    >>> spl0.shape
    ... (2, 3)
    >>> spl0.controlpoints.shape
    ... (54, 2, 3)
    >>> spl1 = spl0.sum(1)
    >>> spl1.shape
    ... (2,)
    >>> spl1.controlpoints.shape
    ... (54, 2)
    """
    if args:
      assert axis is None
      axis = args[0] if len(args) == 1 else args
    # if axis is None sum over all axes
    if axis is None:
      axis = tuple(range(self.controlpoints.ndim-1))
    if isinstance(axis, Int):
      axis = axis,
    assert all(ax > -self.ndim for ax in axis)

    # increment all summation axes by one in order to apply summation to
    # self.controlpoint's tail.
    axis = tuple(ax % self.ndim + 1 for ax in axis)
    assert len(axis) <= self.ndim
    return self._edit(controlpoints=np.sum(self.controlpoints, axis=axis))

  @property
  def T(self) -> Self:
    """
    >>> spl0.shape
    ... (2, 3, 4)
    >>> spl0.controlpoints.shape
    ... (54, 2, 3, 4)
    >>> spl1 = spl0.T
    >>> spl1.shape
    ... (4, 3, 2)
    >>> spl1.controlpoints.shape
    ... (54, 4, 3, 2)
    """
    return self.__class__(self.knotvector, np.moveaxis(self.controlpoints.T, -1, 0))

  def sample_mesh(self, mesh: Mesh):
    """
    Sample a mesh from `self`.

    Parameters
    ----------
    mesh : :class:`splico.mesh.Mesh`
        The mesh's points serve as the evaluation points to `self` for sampling
        a mesh from a spline with target space R^3. The connectivity, elements
        and element types of the sampled mesh follow directly from `mesh`.
        Must satisfy mesh.ndims == len(self). If mesh.ndims < 3, only the first
        mesh.ndims columns of mesh.points are utilized for sampling and augmented
        by zeros to be a manifold in R^3.

    Returns
    -------
    sampled_mesh: :class:`splico.mesh.Mesh`
        The sampled mesh. Has the same type as `mesh`.
    """
    assert self.shape == (3,), 'Mesh export requires the target space to be R^3.'
    assert 0 < self.nvars == mesh.ndims <= 3
    points = self(*mesh.points.T[:self.nvars])
    if points.shape[1:] != (3,):
      points = augment_by_zeros(points, axis=1)
    return mesh._edit(points=points)

  def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs) -> Self | 'NDSplineArray':
    """
    Numpy arithmetic is handled via the __array_ufunc__ protocol.
    When the inputs are all of type :class:`NDSpline`, the arithmetic operation
    is delegated to the corresponding function in :class:`LockableDict`
    ``HANDLED_NDSPLINE_FUNCTIONS``.

    In case arithmetic between ``self`` and some :class:`np.ndarray` is
    performed, the operation is handled as described in this class's docstring.

    If any of the inputs is of type :class:`SplineCollection`, the operation
    is will be handled in largely the same way but the return type will be
    :class:`SplineCollection`, not :class:`NDSpline`.
    """
    # XXX: add support for kwargs. This is tricky because some cannot be
    #      supported by immutable classes.
    if method != '__call__' or kwargs or ufunc not in IMPLEMENTED_UFUNCS:
      return NotImplemented

    # Split into instances of the same class and of other classes.
    # If of other class type, needs to be convertable to np.ndarray or
    # of be an instantiation of class `SplineArray`.
    myclass: List[Self] = []
    notmyclass: List[Numeric | NumericArray | 'NDSplineArray'] = []
    for inp in inputs:
      (myclass if isinstance(inp, NDSpline) else notmyclass).append(inp)

    # get the ufunc
    func = getattr(ufunc, method)

    # If any of the inputs is of type `SplineArray`, let SplineArray
    # handle the operation using the default numpy arithmetic protocol by
    # converting all inputs of type `NDSpline` to `NDSplineArray`.
    if any(isinstance(inp, NDSplineArray) for inp in inputs):
      return func( *(NDSplineArray(inp) if isinstance(inp, NDSpline) else inp
                   for inp in inputs), **kwargs )

    if len(myclass) > 1:  # this block handles interactions between NDSplines
      # since we only handle __call__ for now, the length of `inputs` is
      # two. If both are `NDSpline`, handle them using the implemented protocols.
      if ufunc not in HANDLED_NDSPLINE_FUNCTIONS:
        return NotImplemented
      func = HANDLED_NDSPLINE_FUNCTIONS[ufunc]
      # use a reduce for future handling of more than two inputs
      return reduce(lambda x, y: func(x, y), myclass)

    # What follows handles NDSpline and np.ndarray-like

    # for now only allow the handling of IMPLEMENTED_UFUNCS
    if ufunc not in IMPLEMENTED_UFUNCS:
      return NotImplemented

    # at this point there's at most one NDSpline but it need not be the
    # first entry. However to correctly determine the shape of the output
    # the shape of the NDSpline's controlpoints array needs to be known.
    # We find the index `pivot` of the NDSpline in `inputs` and pass it to the
    # `_vectorize_first_axis` function

    pivot = inputs.index(myclass.pop())

    # operate on `args` to maintain the correct order of the inputs
    args = [inp.controlpoints if isinstance(inp, NDSpline)
                              else np.asarray(inp) for inp in inputs]

    # apply operation with input to controlpoints along the first axis

    controlpoints = _vectorize_first_axis(func,
                                          pivot,
                                          *args,
                                          **kwargs)

    return self.__class__(self.knotvector, controlpoints)


def _vectorize_first_axis(op: Callable, pivot: Int,
                                        *args: NumericArray, **kwargs):
  """
  Vectorization along the first axis.

  Inputs are broadcast to a form that the operation is performed
  len(controlpoints) times along all axes in controlpoints.shape[1:].

  Here, `controlpoints` refers to the array contained in `args` at
  position `pivot`.
  """
  assert pivot < len(args)

  head, (controlpoints, *tail) = list(args[:pivot]), args[pivot:]

  if not controlpoints.shape:  # 0-D case, no broadcasting required
    return op(*args, **kwargs)

  n, *shape_tail = controlpoints.shape

  # find the output shape by taking the pivot's tail and then the shapes of all
  # other inputs
  output_shape_tail = try_broadcast_shapes(shape_tail,
                                           *map(lambda x: x.shape, head + tail))

  # add artificial axes after the first axis if necessary
  p, q = len(output_shape_tail), len(shape_tail)
  controlpoints = controlpoints[(sl,) + (_,) * (p - q)]
  controlpoints = np.broadcast_to(controlpoints, (n,) + output_shape_tail)

  _args = []
  for i, arg in enumerate(args):
    if i == pivot:
      _args.append(controlpoints)
    else:
      # first broadcast to shape (1,) + output_shape_tail, then to
      # (n,) + output_shape_tail
      _args.append( np.broadcast_to(np.broadcast_to(arg, output_shape_tail)[_],
                                                    (n,) + output_shape_tail) )

  return op(*_args, **kwargs)


class NDSplineArrayDelegate(NDArrayOperatorsMixin):
  """
  Helper class for delegating operations to the elements of an NDSplineArray.
  The entire range of Numpy ufuncs that is supported for NDSpline is applied
  to each element of the NDSplineArray in a for loop.

  Performing a Numpy arithmetic operation creates a new instance of
  NDSplineArray with the resulting NDSpline elements.

  An instantiation of this class is added as an attribute to NDSplineArray
  upon creation. The attribute is named `elements` and Numpy operations are
  can be delegated to the elements of the NDSplineArray by calling them on
  the `elements` attribute.

  The operations are performed on the `raveled` array of NDSpline elements.
  Note that this will also work for 0-D `NDSplineArray`s because they are
  broadcast to shape `(1,)` under `np.ravel`.
  """

  def _delegate_to_all_elements(op: str):
    """
    Function that creates a method that delegates an operation to all elements
    of the NDSplineArray using a string to identify which operation to perform.
    """
    # this may lead to problems if the elements of the NDSplineArray are not
    # of type NDSpline but of some derived class of NDSpline that overwrites
    # the method. In this case, the overwritten method will not be called.
    # For now this is not a problem since there is currently no need to
    # derive from NDSpline. We may change this in the future using a metaclass.
    operator = getattr(NDSpline, op)

    def wrapper(self, *args, **kwargs):
      lst = [ operator(elem, *args, **kwargs) for elem in self.arr.ravel() ]
      return self.arr.__class__(lst).reshape(self.arr.shape)

    return wrapper

  def __init__(self, arr: 'NDSplineArray'):
    self.arr = arr

  @cached_property
  def shape(self):
    return self.arr.ravel()[0].shape

  # these methods are not handled by `NDArrayOperatorsMixin` and need to be
  # implemented explicitly
  __getitem__ = _delegate_to_all_elements('__getitem__')
  sum = _delegate_to_all_elements('sum')

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    pivot, = [i for i, inp in enumerate(inputs) if inp is self]
    head, tail = inputs[:pivot], inputs[pivot+1:]
    return NDSplineArray([ elem.__array_ufunc__(ufunc,
                                                method,
                                                *head,
                                                elem,
                                                *tail, **kwargs)
                           for elem in self.arr.ravel() ]).reshape(self.arr.shape)

  def apply(self, func: Callable):
    return NDSplineArray([ func(elem) for elem in self.arr.ravel() ]) \
                          .reshape(self.arr.shape)


class NDSplineArray(np.ndarray):
  """
  Array of splines.
  The knotvectors of all splines may differ but each spline must have the same
  shape.

  Derives from :class:`np.ndarray` and is intended to be used as a container
  for multiple :class:`NDSpline`s. The class is read-only to ensure that
  hashing is safe.

  As a subclass of :class:`np.ndarray`, the class inherits all the methods of
  :class:`np.ndarray` and can be used as such. However, the class also provides
  additional methods that are not available in :class:`np.ndarray`.
  For instance, the class is endowed with an attribute `elements` that delegates
  operations to the elements of the array.

  Furthermore the class provides a method `expand` that expands elements of the
  array into a new instantiation of :class:`NDSplineArray`. So, if each element
  of the array has shape (n0, n1, n2, ...) and the array has shape
  (m0, m1, m2, ...), the expanded array will have shape (m0, m1, m2, ..., n0)
  while the elements will have shape (n1, n2, ...).

  The array additionally provides the `expanded_shape` attribute that returns
  a tuple of the form (shp, *elshp) where `shp` is the shape of the array and
  `elshp` is the shape of each element.

  Example
  -------

  >>> A = ellipse(1, 1, 10)  # NDSpline of shape (5, 3)
  >>> B = NDSplineArray(A)  # NDSplineArray of shape ()
  >>> B.shape
  ... ()
  >>> B.elements.shape
  ... (5, 3)
  >>> B = B.expand()  # absorb the first axis of the elements into the array
  >>> B.shape
  ... (5,)
  >>> B.elements.shape
  ... (3,)
  >>> B.sum().shape  # sum all NDSpline elements
  ... ()
  >>> B.sum(0).elements.shape  # the shape of the elements shouldn't change
  ... (3,)
  >>> B.elements.sum().elements.shape  # sum each component of each element separately
  ... ()

  Arithmetic operations can either be performed between instances of
  the NDSplineArray class or between an instance of NDSplineArray and an
  instance of NDSpline. In the latter case, the NDSpline is first coerced into
  an NDSplineArray with a single element.
  Also, operations between NDSplineArray and np.ndarray or types that can be
  converted to np.ndarray are supported. In this case, the operation is
  performed on the controlpoints of the NDSplineArray elements as implemented
  by the __array_ufunc__ protocol of the NDSpline class.

  >>> B.shape
  ... ()
  >>> (B + B).ravel()[0].controlpoints == 2 * B.ravel()[0].controlpoints
  ... [True, True, True, ..., True]

  Parameters
  ----------
  array_of_splines : Array-like of `NDSpline`s or just a single `NDSpline`.
      Input array-like containing `NDSpline`s.

  ----------

  NOTE THAT THIS CLASS IS AN EXPERIMENTAL FEATURE AND MAY NOT BEHAVE AS
  EXPECTED IN ALL SITUATIONS. IT IS SUBJECT TO CONTINUOUS DEVELOPMENT AND
  IMPROVEMENT.
  """

  # TODO: In the long run it will probably be safer to indirectly derive from
  #       :class:`np.ndarray` via the `__array_ufunc__` protocol. Directly
  #       inheriting from :class:`np.ndarray` is pretty tricky and associated
  #       with a lot of side effects.
  #       We keep the current implementation for now since it is straightforward
  #       and works well on the considered test cases. However, experience
  #       shows that it is not always easy to predict how numpy will behave
  #       when overwriting / extending its methods.
  #       It is likely that the current implementation will not behave as
  #       expected in some edge cases. We will have to keep an eye on this.

  def __new__(cls, array_of_splines: Sequence[Sequence | NDSpline] | NDSpline):
    if array_of_splines.__class__ is cls:
      return array_of_splines
    if not isinstance(array_of_splines, NDSpline):
      array_of_splines = list(iter(array_of_splines))
    ret = np.array(array_of_splines, dtype=NDSpline)
    # assert ret.ndim > 0
    el0, *ignore = ret_raveled = ret.ravel()
    if not all(isinstance(element, NDSpline) for element in ret_raveled):
      raise TypeError("All entries need to be an instantiation of `NDSpline`.")
    # fail switch that ensures that each NDSpline's shape is such that
    # the evaluation can be reshaped properly
    # XXX: not sure if this catches all situations
    assert all((element.nvars, element.shape) == (el0.nvars, el0.shape)
               for element in ret_raveled)
    ret = ret.view(cls)
    ret.flags.writeable = False  # make read-only to allow for safe hashing
    return ret

  def __array_finalize__(self, obj):
    if obj is None:
      return
    self.elements = NDSplineArrayDelegate(self)

  @cached_property
  def expanded_shape(self) -> Tuple:
    return (self.shape, *self.elements.shape)

  def expand(self) -> Self:
    """
    Peel off one layer of the elements of the NDSplineArray and absorb it
    into the shape of the array. The dimension of the element's target space
    will be reduced by one while the dimension of the NDSplineArray is increased
    by one.
    """
    return self.__class__([ list(iter(elem)) for elem in self.ravel() ]).reshape(*self.shape, -1)

  def _call(self, *args, tensor=False, **kwargs) -> FloatArray:
    # XXX: np.stack makes a copy. Find better solution.
    evals = [ elem(*args, tensor=tensor, **kwargs) for elem in self.ravel() ]
    shape0, *tail = evals[0].shape
    return np.stack(evals, axis=1).reshape((shape0,) + self.shape + tuple(tail))

  def __call__(self, *args, **kwargs) -> FloatArray:
    return self._call(*args, tensor=False, **kwargs)

  def tensorcall(self, *args, **kwargs) -> FloatArray:
    return self._call(*args, tensor=True, **kwargs)

  def __hash__(self) -> int:
    if not hasattr(self, '_hash'):  # hashing is safe thanks to read-only flag
      self._hash = hash((tuple(self.ravel()), self.shape))
    return self._hash

  def __eq__(self, other) -> bool:
    """
    We have to overwrite the __eq__ method since the default implementation
    of np.ndarray.__eq__ is not suitable for the purpose of hashing.
    """
    # XXX: We have to thoroughly study what the side-effects of overwriting
    #      __eq__ are when it comes to performing arithmetic operations.
    #      It is likely that this will lead to unexpected behavior in some
    #      edge cases.
    return self.__class__ is other.__class__ and \
           self.shape == other.shape and \
           bool((super().__eq__(other)).all())  # convert to bool to avoid ambiguity

  def sample_mesh(self, sample_meshes: Mesh | np.ndarray | Sequence):
    """
    Sample a mesh from each element of the NDSplineArray.

    If `sample_meshes` is a single mesh, the same mesh is used for all elements.
    If `sample_meshes` is a sequence of meshes, each element is sampled with the
    corresponding mesh. If the shape (after coercion to np.ndarray) of
    ``sample_meshes`` differs from ``self.shape``, the ``sample_meshes`` array
    is broadcast to the shape of ``self.shape``.
    """
    sample_meshes = np.broadcast_to(np.asarray(sample_meshes,
                                                 dtype=Mesh), self.shape)
    sampled_meshes = [elem.sample_mesh(mesh) for elem, mesh
                                   in zip(self.ravel(), sample_meshes.ravel())]
    return np.array(sampled_meshes, dtype=Mesh).reshape(self.shape)
