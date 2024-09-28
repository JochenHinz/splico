#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jochen Hinz
"""

import numpy as np
import sys

from functools import wraps, cached_property
from collections import namedtuple, ChainMap
from collections.abc import Hashable
from typing import Tuple, Sequence

import treelog as log  # installed via the nutils dependency


# alias for vectorisation.
# >>> arr = np.ones((10,), dtype=int)
# >>> print(arr.shape)
#     (10,)
# >>> print(arr[:, np.newaxis].shape)
#     (10, 1)
# >>> print(arr[:, _].shape)
#     (10, 1)
_ = np.newaxis


# the mesh's points are rounded to this number of significant digits
GLOBAL_PRECISION = 12


BAR_LENGTH = 100


def _round_array(arr, precision=None):
  """
    Round an array to `precision` which defaults to `GLOBAL_PRECISION`
    if not passed.
  """
  if precision is None:
    precision = GLOBAL_PRECISION
  return np.round(arr, precision)


class GlobalPrecision:
  """
    Context manager for locally adjusting the machine precision.
    Applies to all attributes / arrays that are truncated using the
    `_round_array` method.

    Example
    -------

    >>> class MyClass:
        def __init__(self, a: np.ndarray):
          self.a = frozen(_round_array(a))  # round and freeze array

    >>> A = MyClass(np.linspace(0, 1, 4))
    >>> A.a
        [0, 0.333333333333, 0.666666666667, 1]

    >>> with GlobalPrecision(4):
    >>>   A = MyClass(np.linspace(0, 1, 4))
    >>> A.a
        [0, 0.3333, 0.6667, 1]
  """

  def __init__(self, precision):
    self.oldprecision = int(GLOBAL_PRECISION)
    self.precision = int(precision)
    assert 0 < self.precision < 16

  def __enter__(self):
    global GLOBAL_PRECISION
    GLOBAL_PRECISION = self.precision

  def __exit__(self, *args, **kwargs):
    global GLOBAL_PRECISION
    GLOBAL_PRECISION = self.oldprecision


def frozen_cached_property(fn):
  """
    Combined decorator for `freeze` and `cached_property`.
    Equivalent to:

    >>> @cached_property
    >>> @freeze
    >>> def myfunc(self):
          return np.arange(self.n)
  """
  return wraps(fn)(cached_property(freeze(fn)))


def frozen(array: np.ndarray, dtype=None) -> np.ndarray:
  """
    Freeze a vector inplace and return it.

    Example
    -------

    >>> arr = np.zeros((10,), dtype=int)
    >>> print(arr[0])
        0
    >>> arr[0] = 1
    >>> print(arr[0])
        1
    >>> arr = np.zeros((10,), dtype=int)
    >>> arr = frozen(arr)
    >>> arr[0] = 1
        ERROR

    Both in and out of place will work.
    >>> arr = np.zeros((10,), dtype=int)
    >>> frozen(arr)
    >>> arr[0] = 1
        ERROR
  """
  array = np.asarray(array, dtype=dtype)
  array.flags.writeable = False
  return array


def freeze(fn, dtype=None):
  """
    Decorator that freezes the returned array inplace.

    Example
    -------

    def multiply(arr, val):
      return val * arr

    >>> arr = np.ones((5,), dtype=int)
    >>> new_arr = multiply(arr, 2)
    >>> print(new_arr)
        [2, 2, 2, 2, 2]
    >>> new_arr[0] = 10
    >>> print(new_arr)
        [10, 2, 2, 2, 2]

    @freeze
    def multiply(arr, val):
      return val * arr

    >>> arr = np.ones((5,), dtype=int)
    >>> new_arr = multiply(arr, 2)
    >>> print(new_arr)
        [2, 2, 2, 2, 2]
    >>> new_arr[0] = 10
        ERROR
  """
  @wraps(fn)
  def wrapper(*args, **kwargs):
    return frozen(fn(*args, **kwargs), dtype=dtype)
  return wrapper


# named tuple with fields corresponding to a serialized array (for hashing)
serialized_array = namedtuple('serialized_array', ('shape', 'dtype', 'bytes'))


# serialize and array for hashing
def serialize_array(arr: np.ndarray):
  return serialized_array(arr.shape, arr.dtype.str.encode(), arr.tobytes())


# convert serialized array back to ordinary np.ndarray
def deserialize_array(serial: serialized_array):
  shape, dtype, _bytes = serial
  return np.frombuffer(_bytes, dtype=np.dtype(dtype.decode())).reshape(shape)


def serialize_input(fn):
  @wraps(fn)
  def wrapper(arr, *args, **kwargs):
    return fn(serialize_array(arr), *args, **kwargs)
  return wrapper


class HashMixin(Hashable):
  """
    Generic Mixin for hashability.
    Requires each derived class to implement the `_items` class attribute.
    The `_item` attribute is a tuple of strings where each string represents
    the name of a class attribute (typically set in __init__) that contributes
    to the class's hash.
    Each element in `_items` needs to refer to a hashable type with the exception
    of `np.ndarray` which is serialized using `serialize_array`.

    The Mixin then implements the `__hash__` and the `__eq__` dunder methods in
    the obvious way. For this, the `__hash__` and `__eq__` dunder methods
    make use of the `tobytes` cached property which serializes all relevant
    attributes and returns them as a tuple.

    The same Mixin implements the `_lib` method that returns a dictionary of
    all (relevant) attributes implemented by `_items`, i.e.,
        {item: getattr(self, item) for item in self._items}.
    The `_edit` method then allows for editing single or several attributes
    while keeping all others intact and instantiates a new instance of the
    same class with the updated attributes.

    It is of paramount importance that the derived class is immutable.
    This means that all relevant attributes are immutable and hashable with
    the exception of `np.ndarray`s which need to be frozen using `frozen` or
    `freeze`.

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
    return self.__class__(**dict(ChainMap(kwargs, self._lib)))

  @property
  def _lib(self):
    return {item: getattr(self, item) for item in self._items}

  @cached_property
  def tobytes(self):
    ret = []
    for i, attr in enumerate(map(self.__getattribute__, self._items)):
      if isinstance(attr, np.ndarray):
        if attr.flags.writeable is True:
          log.warning(f"Warning, attempting to hash the attribute `{self._items[i]}` which is a writeable `np.ndarray`."
                       "Ensure that all `np.ndarray` attributes are non-writeable using `util.frozen` or `util.freeze`.")
        ret.append(serialize_array(attr))
      elif isinstance(attr, Hashable):
        ret.append(attr)
      else:
        raise AssertionError("Attribute of type '{}' cannot be hashed.".format(str(type(attr))))
    return tuple(ret)

  def __hash__(self):
    if not hasattr(self, '_hash'):
      self._hash = hash(self.tobytes)
    return self._hash

  def __eq__(self, other):
    if self.__class__ is not other.__class__: return False
    for item0, item1 in zip(self.tobytes, other.tobytes):
      if item0 != item1: return False
    return True


class FrozenDict(dict):
  """
    Frozen dictionary. Once instantiated, cannot be changed.
  """

  # XXX: find more elegant solution using collections.abc.mapping

  def _immutable(self, *args, **kwargs):
    raise TypeError('Cannot change object - object is immutable')

  __setitem__ = _immutable
  __delitem__ = _immutable
  pop = _immutable
  popitem = _immutable
  clear = _immutable
  update = _immutable
  setdefault = _immutable

  del _immutable


class NanVec(np.ndarray):
  """
     Vector of dtype float initilized to np.nan.
     Used in `solve_with_dirichlet_data`.
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


class ProgressBar:
  """ For monitoring the progress of a loop. """

  def __init__(self, prefix='Task', suffix='completed'):
    self.prefix = str(prefix).strip()
    self.suffix = str(suffix).strip()

  def __call__(self, factor):
    filled_up_Length = int(round(BAR_LENGTH * factor))
    percentage = round(100.0 * factor, 1)
    bar = '=' * filled_up_Length + '-' * (BAR_LENGTH - filled_up_Length)
    sys.stdout.write('%s [%s] %s%s %s.\r' % (self.prefix, bar, percentage, '%', self.suffix))
    sys.stdout.flush()

  def __del__(self):
    sys.stdout.write('\n')


def isincreasing(arr: Sequence | np.ndarray):
  """
    Return True if an array-like is strictly increasing.
    Else return False.
  """
  arr = np.asarray(arr)
  assert arr.ndim == 1
  return (np.diff(arr) > 0).all()


def gauss_quadrature(a: int | float, b: int | float, order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
  """
   Given the element boundaries `(a, b)`, return the weights and evaluation points
   corresponding to a gaussian quadrature scheme of order `order`.

    Parameters
    ----------
    a : :class:`float`
      The left boundary of the element.
    b : :class:`float`
      The right boundary of the element.
    order : :class:`int`
      The order of the Gaussian quadrature scheme.

    Returns
    -------
    weights : :class:`np.ndarray`
      The weights of the quadrature scheme.
    points : :class:`np.ndarray`
      The points (abscissae) over (a, b).
  """
  assert b > a
  points, weights = np.polynomial.legendre.leggauss(order)
  points = (points + 1) / 2
  return (b - a) / 2 * weights, a + points * (b - a)


def clparam(points: Sequence | np.ndarray):
  # XXX: docstring
  points = np.asarray(points)
  assert points.ndim
  if points.ndim == 1:
    points = points[:, _]

  ret = np.array([0, *np.linalg.norm(points, axis=1).cumsum()])
  return ret / ret[-1]
