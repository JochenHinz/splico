from ..util import _round_array, np, HashMixin, frozen, _
from ._jit_spl import call, tensor_call
from ..mesh.mesh import Mesh
from .kv import UnivariateKnotVector, TensorKnotVector, as_TensorKnotVector

from typing import List, Sequence, Callable, Tuple, Any
from scipy import interpolate
from numpy.lib.mixins import NDArrayOperatorsMixin

from functools import reduce


IMPLEMENTED_UFUNCS = (np.add, np.subtract, np.multiply, np.divide)

HANDLED_NDSPLINE_FUNCTIONS = {}


class __NDSpline_implementations__:
  """
    Dummy class for implementing various `NDSpline` ufuncs.
    They are added to `HANDLED_NDSPLINE_FUNCTIONS` to be used in the `NDSpline`
    __array_ufunc__ protocol.
  """

  def implements(np_function):
    def decorator(func):
      HANDLED_NDSPLINE_FUNCTIONS[np_function] = func
      return func
    return decorator

  @implements(np.add)
  def add(self, other):
    assert self.knotvector == other.knotvector
    # if the knotvectors are the same, simply add the controlpoints together
    return NDSpline(self.knotvector, self.controlpoints + other.controlpoints)

  @implements(np.multiply)
  def mul(self, other):

    # tensor knotvector
    knotvector = self.knotvector * other.knotvector

    myshape, othershape = self.shape, other.shape

    # prepend ones to the shorter shaped NDSpline (default numpy behavior)
    n, m = map(len, (myshape, othershape))
    if m > n:
      myshape = (1,) * (m - n) + myshape
    elif n > m:
      othershape = (1,) * (n - m) + othershape
    assert all(i == j or 1 in (i, j) for i, j in zip(myshape, othershape))

    # determine final shape of controlpoints after multiplication
    shape = tuple(map(max, zip(myshape, othershape)))

    # take outer product and reshape to (ncontrolpoints,) + shape
    cps = (self.controlpoints.reshape(-1, 1, *myshape) *
           other.controlpoints.reshape(1, -1, *othershape)).reshape(-1, *shape)

    return NDSpline(knotvector, cps)


class NDSpline(HashMixin, NDArrayOperatorsMixin):

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
        representing the controlpoints. Upon conversion, self.controlpoints
        must satisfy self.controlpoints.shape[0] == self.knotvector.ndofs.
        Is rounded to the current precision as set by `splico.util.GlobalPrecision`
        and then frozen.
        self.controlpoints.shape[1:] can be anything and then simply represents
        a vectorial / tensorial set of splines all with the same knotvector.

    Attributes
    ----------
    knotvector: :class:`splico.spl.kv.TensorKnotVector`
        The knotvector.
    controlpoints: :class:`np.ndarray`
        The controlpoints.
  """

  _items = 'knotvector', 'controlpoints'

  @classmethod
  def one(cls, knotvector):
    """
      Constant one function. Helpful when a function of one variable
      has to be made a function of the other variables as well.
    """
    if isinstance(knotvector, UnivariateKnotVector):
      knotvector = TensorKnotVector([knotvector])
    return cls(knotvector, np.ones(knotvector.ndofs))

  @classmethod
  def from_exact_interpolation(cls, verts: Sequence | np.ndarray, data: Sequence | np.ndarray, **scipyargs):
    """
      Classmethod for creating a an exact interpolation of a verts, data pair
      using scipy. It is then wrapped as a NDSpline.
    """

    # XXX: parameters

    verts = np.asarray(verts)
    data = np.asarray(data)

    assert verts.ndim == data.ndim == 1, NotImplementedError

    k = scipyargs.setdefault('k', 3)
    spline = interpolate.InterpolatedUnivariateSpline(verts, data, **scipyargs)

    coeffs, knots = spline.get_coeffs(), spline.get_knots()
    knotvector = TensorKnotVector([UnivariateKnotVector(knots, degree=k)])
    return cls(knotvector, coeffs)

  @staticmethod
  def from_least_squares(knotvector: UnivariateKnotVector | TensorKnotVector, *args, **kwargs):
    """
      Docstring: see `splico.spl.kv.TensorKnotVector.fit`.
    """
    # convert to TensorKnotVector if not alreay
    knotvector = as_TensorKnotVector(knotvector)
    # forward to `TensorKnotVector.fit` routine
    return knotvector.fit(*args, **kwargs)

  def __init__(self, knotvector: UnivariateKnotVector | TensorKnotVector | Sequence[UnivariateKnotVector], controlpoints: np.ndarray | Sequence):
    self.knotvector = as_TensorKnotVector(knotvector)

    # for now we only allow tensorial knotvectors of up to length 3
    assert 1 <= len(self.knotvector) <= 3, NotImplementedError

    self.controlpoints = frozen(_round_array(controlpoints))
    assert self.controlpoints.shape[0] == self.knotvector.ndofs

  @property
  def shape(self):
    """ The shape of the vectorial spline. """
    return self.controlpoints.shape[1:]

  @property
  def nvars(self):
    """ Number of dependencies. """
    return self.knotvector.ndim

  @property
  def tcontrolpoints(self):
    """
      Represent the controlpoints tensorially.
      >>> spline.controlpoints.shape
      >>> (36, 2, 3)
      >>> spline.knotvector.dim
      >>> (3, 3, 4)
      >>> spline.tcontrolpoints.shape
      >>> (3, 3, 4, 2, 3)
    """
    return self.controlpoints.reshape(self.knotvector.dim + self.shape)

  def __call__(self, *positions: np.ndarray, tensor: bool = False,
                                             dx: Sequence[int] | int | np.int_ | None = None):
    """
      Evaluate the spline in a set of points.

      Parameters
      ----------
      positions : :class:`tuple` of :class:`np.ndarray` or any array-like
          Positional arguments of the x, y, z, ... coordinates all represented
          by a flat array-like, such as :class:`np.ndarray`.
          Must satisfy all(len(pos) == len(positions[0]) for pos in positions)
          and len(positions) == len(self.knotvector)
      dx : :class:`Sequence[int]` or :class:`int`
          Derivative order of the evaluation. Must satisfy len(dx) == len(self.knotvector).
          If given by a :class:`int` it is repeated len(self.knotvector) times.
      tensor : :class:`bool`
          If true, the spline is evaluated over a meshgrid of the positions, i.e.,
          positions is converted to tuple(map(np.ravel, np.meshgrid(*positions))).

      Returns
      -------

      ret : :class:`np.ndarray``
          The evaluation array. Has shape
            (len(positions[0]),) + self.controlpoints.shape[1:]
          if tensor is False and
            (np.prod([len(po) for pos in positions]),) + self.controlpoints.shape[1:]
          else.
    """
    if dx is None:
      dx = 0
    if isinstance(dx, (int, np.int_)):
      dx = (dx,) * self.nvars

    positions = tuple(map(np.asarray, positions))
    assert all( len((y := pos.shape)) == 1 for pos in positions ), NotImplementedError

    # if not evaluated tensorially, make sure all positions have equal length
    if not tensor:
      assert all(pos.shape == y for pos in positions)

    controlpoints = self.controlpoints if self.shape else self.controlpoints[..., _]

    function = {False: call, True: tensor_call}[tensor]
    ret = [function(positions, self.knotvector.repeat_knots(), self.knotvector.degree, x, dx)
                               for x in controlpoints.reshape(-1, np.prod(controlpoints.shape[1:])).T]

    # XXX: np.stack makes a copy, find better solution (possibly do this in numba)
    return np.stack(ret, axis=1).reshape(-1, *self.shape)

  def tensorcall(self, *args, **kwargs):
    """ partial(self, tensor=True) """
    return self(*args, tensor=True, **kwargs)

  def __getitem__(self, item: int | List[int] | slice | None | Tuple[int | List[int] | slice | None]):
    """
      Same as np.ndarray.__getitem__ with the difference that the zeroth axis
      is ignored and the broadcasting etc. is applied to all axes with index > 0.
      >>> spline.controlpoints.shape
      >>> (54, 2, 3)
      >>> spline[_].controlpoints.shape
      >>> (54, 1, 2, 3)
    """
    if not isinstance(item, tuple):
      item = item,
    return self._edit(controlpoints=np.ndarray.__getitem__(self.controlpoints, (slice(None),) + item))

  @property
  def ndim(self):
    """ Dimensionality of the spline (not the number of dependencies). """
    return self.controlpoints.ndim - 1

  def __iter__(self):
    """ Iterate over the each sub-spline as a numpy array. """
    if not self.shape:
      raise TypeError('iteration over 0-d array')
    yield from (self._edit(knotvector=self.knotvector, controlpoints=controlpoints) for controlpoints in self.controlpoints.swapaxes(0, 1))

  def ravel(self):
    """ Ravel the tensorial spline. """
    return self._edit(controlpoints=self.controlpoints.reshape(self.controlpoints.shape[:1] + (self.shape and (-1,))))

  def __matmul__(self, other):
    # XXX: docstring
    # XXX: maybe remove this one
    if isinstance(other, np.ndarray):
      return self._edit(controlpoints=self.controlpoints @ other)
    assert isinstance(other, NDSpline)
    kv1 = other.knotvector
    other = NDSpline.one(self.knotvector) * other
    self = self * NDSpline.one(kv1)
    return self._edit(controlpoints=(self.tcontrolpoints @ other.tcontrolpoints).reshape(-1, *self.shape))

  def sum(self, *args, axis=None):
    """
      Same as np.sum but applied to the tail of self.controlpoints.
      >>> type(spl0)
          splico.spl.spline.NDSpline
      >>> spl0.shape
          (2, 3)
      >>> spl0.controlpoints.shape
          (54, 2, 3)
      >>> spl1 = spl0.sum(1)
      >>> spl1.shape
          (2,)
      >>> spl1.controlpoints.shape
          (54, 2)
    """
    if args:
      assert axis is None
      if len(args) == 1:
        args, = args
        if isinstance(args, tuple):
          axis = args
        else:
          axis = (args,)
      else:
        axis = args
    # if axis is None sum over all axes
    if axis is None:
      axis = tuple(range(self.controlpoints.ndim-1))
    assert all(ax > -self.ndim for ax in axis)

    # increment all summation axes by one in order to apply summation to
    # self.controlpoint's tail.
    axis = tuple(ax % self.ndim + 1 for ax in axis)
    assert len(axis) <= self.ndim
    return self._edit(controlpoints=np.sum(self.controlpoints, axis=axis))

  @property
  def T(self):
    """
      >>> spl0.shape
          (2, 3, 4)
      >>> spl0.controlpoints.shape
          (54, 2, 3, 4)
      >>> spl1 = spl0.T
      >>> spl1.shape
          (4, 3, 2)
      >>> spl1.controlpoints.shape
          (54, 4, 3, 2)
    """
    return NDSpline(self.knotvector, np.moveaxis(self.controlpoints.T, -1, 0))

  def sample_mesh(self, mesh: Mesh, axes: Sequence[int] | None = None):
    """
      Sample a mesh from `self`.

      Parameters
      ----------
      mesh : :class:`splico.mesh.Mesh`
          The mesh's points serve as the evaluation points to `self` for sampling
          a mesh from a spline with target space R^3. The connectivity, elements
          and element types of the sampled mesh follow directly from `mesh`.
      axes : Sequence[int] or None
          The axes a mesh is sampled from. Only applicable if self.nvars < 3.
          For instance, if `mesh` has points [ [a0, b0, 0], [a1, b1, 0], ... ]
          it would make sense to disregard the z-coordinates of the points for
          the sampling.
          If None, defaults to (0, 1, ..., self.nvars).

      Returns
      -------
      sampled_mesh: :class:`splico.mesh.Mesh`
          The sampled mesh. Has the same type as `mesh`.
    """
    # XXX: find more elegant solution than `axes` argument.
    assert self.shape == (3,), 'Mesh export requires the target space to be R^3.'
    assert self.nvars == mesh.ndims <= 3
    if axes is None:
      axes = tuple(range(self.nvars))
    assert len(axes) == self.nvars and 0 <= min(axes) <= max(axes) < self.nvars
    points = _round_array( self(*mesh.points.T[list(axes)]) )
    return mesh._edit(points=points)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """
      All other numpy methods are implemented using the `__array_ufunc__`
      protocol in combination with a for loop for now.
    """
    if method != '__call__':
      return NotImplemented

    if ufunc not in IMPLEMENTED_UFUNCS:
      return NotImplemented

    # split into instances of the same class and of other classes
    # if of other class type, needs to be broadcastable to np.ndarray
    myclass, notmyclass = [], []
    for inp in inputs:
      if isinstance(inp, NDSpline):
        myclass.append(inp)
      else:
        notmyclass.append(np.asarray(inp))

    # this block handles interactions between NDSpline and NDSpline
    if len(myclass) > 1:
      if notmyclass or kwargs or ufunc not in HANDLED_NDSPLINE_FUNCTIONS:
        return NotImplemented
      try:
        func = HANDLED_NDSPLINE_FUNCTIONS[ufunc]
        return reduce(lambda x, y: func(x, y), myclass)
      except Exception as ex:
        raise Exception("Failed with unknown error '{}'.".format(ex))

    # What follows handles NDSpline and np.ndarray-like

    # for now only allow the handling of IMPLEMENTED_UFUNCS
    if ufunc not in IMPLEMENTED_UFUNCS:
      return NotImplemented

    spl, = myclass

    # get the ufunc
    func = getattr(ufunc, method)

    # apply operation with input to controlpoints along the first axis in
    # a for loop (for now)

    # XXX: remove for loop and replace by dedicated vectorized routines.
    #      Using Numba is another option.
    controlpoints = _vectorize_numpy_operation(func, spl.controlpoints, *notmyclass, **kwargs)

    return NDSpline(self.knotvector, controlpoints)


def _vectorize_numpy_operation(op: Callable, input_arr: np.ndarray, *args: Sequence[Any], **kwargs):
  return np.stack([ op(subarr, *args, **kwargs) for subarr in input_arr ], axis=0)


def as_NDSpline(spln) -> NDSpline:
  """
    Convert to NDSpline if not already.
  """
  if isinstance(spln, NDSpline):
    return spln
  elems = np.asarray(spln, dtype=NDSpline)
  elem0, *elems_rav = (y := elems.ravel())
  assert all(elem.knotvector == elem0.knotvector for elem in elems_rav)
  controlpoints = np.stack([_y.controlpoints for _y in y], axis=1).reshape(elem0.controlpoints.shape[0], *elems.shape, *elem0.controlpoints.shape[1:])
  return NDSpline(elem0.knotvector, controlpoints)


# Not sure if the `SplineCollection` class is really necessary.
# Maybe there's a more elegant solution.

class SplineCollection(np.ndarray):

  """
    Array of splines.
    Differs from a tensorial instantiation of `NDSpline` in that the knotvectors
    of the splines may differ.

    Parameters
    ----------
    array_of_splines : Array-like of `NDSpline`s or just a single `NDSpline`.
        Input array-like containing `NDSpline`s.
  """

  # XXX: potentially make this an indirect ndarray subclass using the __array_ufunc__ protocol

  def __new__(cls, array_of_splines: Sequence[Sequence | NDSpline] | NDSpline):
    if isinstance(array_of_splines, NDSpline):
      array_of_splines = list(iter(array_of_splines))
    ret = np.array(array_of_splines, dtype=NDSpline)
    assert ret.ndim > 0, NotImplementedError
    ret_ravel = ret.ravel()
    if not all(isinstance(element, NDSpline) for element in ret_ravel):
      raise TypeError("All entries need to be an instantiation of `NDSpline`.")
    # fail switch that ensures that each NDSpline's shape is such that the evaluation can be reshaped properly
    # XXX: not sure if this catches all situations
    assert all((element.nvars, element.shape) == (ret_ravel[0].nvars, ret_ravel[0].shape) for element in ret_ravel)
    ret = ret.view(cls)
    ret.flags.writeable = False
    return ret

  def _call(self, *args, tensor=False, **kwargs):
    # XXX: np.stack makes a copy. Find better solution.
    rav = self.ravel()
    evals = [ elem(*args, tensor=tensor, **kwargs) for elem in rav ]
    shape0, *tail = evals[0].shape
    return np.stack(evals, axis=1).reshape((shape0,) + self.shape + tuple(tail))

  def __call__(self, *args, **kwargs):
    return self._call(*args, tensor=False, **kwargs)

  def tensorcall(self, *args, **kwargs):
    return self._call(*args, tensor=True, **kwargs)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash((tuple(self.ravel()), self.shape))
    return self._hash

  def __eq__(self, other):
    return self.__class__ is other.__class__ and \
           self.shape == other.shape and \
           (super().__eq__(other)).all()


del __NDSpline_implementations__
