"""
Module declaring various types that find applications throughout the library.
In particular, this module introduces :class:`HashMixin`, a general purpose
base class for immutable and hashable classes.
"""


from .util import serialize_array, deserialize_array, serialized_array, np
from .err import UnequalLengthError

from functools import wraps
from collections import ChainMap
from collections.abc import Hashable
from typing import Any, Self, Tuple, List

import treelog as log
from numpy.typing import NDArray


IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float_]


def ensure_same_class(fn):
  """
  A decorator ensuring that a function of the form ``fn(self, other)``
  immediately returns ``NotImplemented`` in case
  ``self.__class__ is not other.__class__``.

  >>> @ensure_same_class
      def __le__(self, other):
        return self.attr <= other.attr

  >>> @ensure_same_class
      def __eq__(self, other):
        return self.attr == other.attr

  >>> a = MyClass(attr=5)  # derives from HashMixin
  >>> b = MyClass(attr=5)
  >>> a <= b
      True
  >>> a == b
      True
  >>> c = MyClass(attr=4)
  >>> a <= c
      False
  >>> a <= 'foo'  # inequality not resolved
      ERROR
  >>> a == 'foo'  # equality not resolved
      False
  >>> c = MyOtherClass(attr=5)  # derives from HashMixin
  >>> a <= c
      ERROR
  >>> a == c
      False
  """
  @wraps(fn)
  def wrapper(self, other: Any) -> Self:
    if self.__class__ is not other.__class__:
      return NotImplemented
    return fn(self, other)
  return wrapper


def ensure_same_length(fn):
  """
  Decorator for making sure that sized objects have equal length.
  The types of ``self`` and ``other`` have to match.
  """
  @wraps(fn)
  def wrapper(self, other):
    if len(self) != len(other):
      raise UnequalLengthError("Can only perform operation on instances with"
                               " equal length.")
    return fn(self, other)
  return wrapper


class HashMixin(Hashable):
  """
  Generic Mixin for hashability.
  Requires each derived class to implement the ``_items`` class attribute.
  The ``_items`` attribute is a tuple of strings where each string represents
  the name of a class attribute (typically set in ``__init__``) that contributes
  to the class's hash.
  Each element in ``_items`` needs to refer to a hashable type with the exception
  of :class:`np.ndarray` which is serialized using ``serialize_array``.

  The Mixin then implements the ``__hash__`` and the ``__eq__`` dunder methods
  in the obvious way. For this, the ``__hash__` and ``__eq__`` dunder methods
  make use of the ``tobytes`` cached property which serializes all relevant
  attributes and returns them as a tuple.

  The same Mixin implements the ``_lib`` method that returns a dictionary of
  all (relevant) attributes implemented by ``_items``, i.e.,
      {item: getattr(self, item) for item in self._items}.
  The ``_edit`` method then allows for editing single or several attributes
  while keeping all others intact and instantiates a new instance of the
  same class with the updated attributes.

  It is of paramount importance that the derived class is immutable.
  This means that all relevant attributes are immutable and hashable with
  the exception of :class:`np.ndarray`s which need to be frozen using
  ``frozen`` or ``freeze``.

  >>> class MyClass(HashMixin):
        _items = 'a', 'b'
        def __init__(a: np.ndarray, b: Tuple[int, ...]):
          self.a = frozen(a, dtype=float)
          self.b = tuple(map(int, b))

  >>> A = MyClass( np.linspace(0, 1, 11), (1, 2, 3) )
  >>> hash(A)
      5236462403644277461

  >>> B = MyClass( np.linspace(0, 1, 11), (1, 2, 3) )
  >>> A == B
      True

  >>> B = MyClass( np.linspace(1, 2, 11), (1, 2, 3) )
  >>> A == B
      False

  >>> A._edit(a=np.linspace(1, 2, 11)) == B
      True
  """

  _items: Tuple[str, ...]

  def _edit(self, **kwargs):
    return self.__class__(**ChainMap(kwargs, self._lib))

  @property
  def _lib(self) -> dict:
    return {item: getattr(self, item) for item in self._items}

  @property
  def tobytes(self) -> Tuple[Hashable, ...]:
    ret: List[Hashable | serialized_array] = []
    for i, attr in enumerate(map(self.__getattribute__, self._items)):
      if isinstance(attr, np.ndarray):

        if attr.flags.writeable is True:
          log.warning(f"Warning, attempting to hash the attribute `{self._items[i]}`"
                      " which is a writeable `np.ndarray`."
                       "Ensure that all `np.ndarray` attributes are non-writeable"
                       " using `util.frozen` or `util.freeze`.")

        ret.append(serialize_array(attr))
      elif isinstance(attr, Hashable):
        ret.append(attr)
      else:
        raise AssertionError(f"Attribute of type '{str(type(attr))}' cannot be hashed.")
    return tuple(ret)

  def __getstate__(self):
    """ For pickling. """
    return self.tobytes

  def __setstate__(self, state):
    """ For unpickling. """
    args = {}
    for key, item in zip(self._items, state):
      if isinstance(item, serialized_array):
        item = deserialize_array(item)
      args[key] = item
    self.__init__(**args)

  def __hash__(self) -> int:
    if not hasattr(self, '_hash'):
      self._hash = hash(self.tobytes)
    return self._hash

  def __eq__(self, other: Any) -> bool:
    """
    Default implementation of __eq__ for comparison between same types.
    The behavior when ``other.__class__ is not self.__class__``, see
    `ensure_same_class`.
    """
    if self.__class__ is not other.__class__:
      return NotImplemented
    for item0, item1 in zip(self.tobytes, other.tobytes):
      if item0 != item1: return False
    return True


class NanVec(np.ndarray):
  """
  Vector of dtype float initilized to np.nan.
  I used to implement Dirichlet boundary conditions.
  """

  @classmethod
  def from_indices_data(cls, length, indices, data):
    'Instantiate NanVec x of length ``length`` satisfying x[indices] = data.'
    vec = cls(length)
    vec[np.asarray(indices)] = np.asarray(data)
    return vec

  def __new__(cls, length):
    vec = np.empty(length, dtype=float).view(cls)
    vec[:] = np.nan
    return vec

  @property
  def where(self):
    """
    Return boolean mask ``mask`` of shape self.shape with mask[i] = True
    if self[i] != np.nan and False if self[i] == np.nan.
    >>> vec = NanVec(5)
    >>> vec[[1, 2, 3]] = 7
    >>> vec.where
        [False, True, True, True, False]
    """
    return ~np.isnan(self.view(np.ndarray))

  def __ior__(self, other):
    """
    Set self[~self.where] to other[~self.where].
    Other is either an array-like of self.shape or a scalar.
    If it is a scalar set self[~self.where] = other
    >>> vec = NanVec(5)
    >>> vec[[0, 4]] = 5
    >>> vec |= np.array([0, 1, 2, 3, 4])
    >>> print(vec)
        [5, 1, 2, 3, 5]

    >>> vec = NanVec(5)
    >>> vec[[0, 4]] = 5
    >>> vec |= 0
    >>> print(vec)
        [5, 0, 0, 0, 5]
    """
    wherenot = ~self.where
    self[wherenot] = other if np.isscalar(other) else other[wherenot]
    return self

  def __or__(self, other):
    """
    Same as self.__ior__ but the result is cast into a new vector, i.e.,
    z = self | other.
    """
    return self.copy().__ior__(other)
