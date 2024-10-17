"""
Metaclasses specific to the `splico.spl` package.
"""

from ..types import ImmutableMeta, ensure_same_length, ensure_same_class
from ..util import np

import operator


def add_vectorizations(cls, *args, **kwargs):
  """
  Decorator for adding vectorized versions of various :class:`UnivariateKnotVector`
  methods and properties to the :class:`TensorKnotVector` class.
  """
  # `_vectorize_X` methods available from the metaclass

  # add all vectorized properties
  for name in ('dx', 'knots', 'km', 'degree',
               'repeated_knots', 'nelems', 'dim', 'greville'):
    cls._vectorize_property(name)

  # add all vectorization with arguments
  for name in ('flip', 'refine', 'ref_by', 'add_knots', 'raise_multiplicities'):
    cls._vectorize_with_indices(name)

  # add all operator vectorizations without custom return type
  for name in ('__and__', '__or__'):  # operator is inferred from ``name``
    cls._vectorize_operator(name)

  # add all operator vectorizations with custom return type
  for name in ('__lt__', '__gt__', '__le__', '__ge__'):
    cls._vectorize_operator(name, all)

  return cls


class TensorKnotVectorMeta(ImmutableMeta):
  """
  Metaclass for :class:`splico.spl.kv.TensorKnotVector`.
  Adds a number of methods to the class that can be used for vectorizing
  various :class:`splico.spl.kv.UnivariateKnotVector` methods and properties.

  >>> kv = UnivariateKnotVector(np.linspace(0, 1, 3))
  >>> class MyClass(Immutable, metaclass=TensorKnotVectorMeta):
          def __init__(self, knotvectors):
            self.knotvectors = tuple(knotvectors)
  >>> MyClass._vectorize_property('knots')
  >>> kv.knots
      [0.0, 0.5, 1.0]
  >>> A = MyClass([kv, kv, kv])
  >>> A.knots
      ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])

  """

  # XXX: this should become a general metaclass that allows for vectorization
  #      of arbitrary attribute's methods.

  def __new__(mcls, *args, **kwargs):
    return add_vectorizations(super().__new__(mcls, *args, **kwargs))

  def _vectorize_with_indices(cls, name: str):
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

    setattr(cls, name, wrapper)

  def _vectorize_operator(cls, name: str, return_type=None):
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

    setattr(cls, name, wrapper)

  def _vectorize_property(cls, name: str, return_type=tuple):
    """
    Vectorize a (cached) property. Optionally takes a return container-type
    argument which the properties are iterated into.
    For instance, :class:`list` or :class:`tuple`. Defaults to :class:`tuple`.
    """
    @property
    def wrapper(self):
      return return_type([getattr(e, name) for e in self])

    setattr(cls, name, wrapper)


class NDSplineMeta(ImmutableMeta):
  """
  Metaclass for the NDSpline class.
  Is equivalent to :type:`ImmutableMeta` but automatically inherits a number
  of functions from ``splico.spl.TensorKnotVector`` that create a new knotvector.
  The spline is then automatically prolonged to the newly created knotvector.
  Similarly, a number of properties are inherited. They can then directly be
  called from the spline rather than delegating them to the spline's knotvector.
  For example:

  >>> spl
      NDSpline<(4, 5)>
  >>> spl.knotvector.knots
      ([0, 0.1, 0.2, 0.4, 0.8, 1.0],)
  >>> spl.knots
      ([0, 0.1, 0.2, 0.4, 0.8, 1.0],)
  """
  def __new__(mcls, *args, **kwargs):
    cls = super().__new__(mcls, *args, **kwargs)

    def forward_and_refine(name):
      # create a new function that forwards input to the knotvector
      # and then prolongs to the new knotvector.
      def wrapper(self, *args, **kwargs):
        new_knotvector = getattr(self.knotvector, name)(*args, **kwargs)
        return self.prolong_to(new_knotvector)
      return wrapper

    def wrap_property(name):
      # create a new property that returns what the instantiation's knotvector
      # property of the same name would return.
      return property(lambda self: getattr(self.knotvector, name))

    # create the methods that are inherited from the knotvector
    # and subsequently prolonged to the new knotvector
    for name in ('refine', 'ref_by', 'add_knots', 'raise_multiplicities'):
      setattr(cls, name, forward_and_refine(name))

    # add the properties that are inherited from the knotvector
    for name in ('knots', 'km', 'degree', 'repeated_knots', 'greville'):
      setattr(cls, name, wrap_property(name))

    return cls
