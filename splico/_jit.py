from .util import np

from typing import Sequence
import math

from numba import njit


""" Routines for use in JIT-compiled functions. """


""" itertools-equivalent Numba implementations. """


@njit(cache=True)
def _product(arr0: np.ndarray, list_of_linspaces: Sequence[np.ndarray]):
  """
    Given :class:`np.ndarray` `arr0` and :class:`list` of :class:`np.ndarray`s
    ``list_of_linspaces``,  create a column tensor product with ``arr0`` and
    all arrays in ``list_of_linspaces``.
    The input ``arr0`` is assumed to be two-dimensional, i.e., in the case of a
    single array, the shape needs to be ``(npoints, 1)``.
  """
  while True:
    lin0, list_of_linspaces = list_of_linspaces[0], list_of_linspaces[1:]
    n, m = len(lin0), len(arr0)
    m, ndims = arr0.shape[0], arr0.shape[1]

    ret = np.empty((len(arr0) * n, ndims + 1), dtype=arr0.dtype)
    for i in range(n):
      ret[i * m: (i+1) * m, -ndims:] = arr0

    counter = 0
    for i in range(n):
      myval = lin0[i]
      for j in range(arr0.shape[0]):
        ret[counter, 0] = myval
        counter += 1
    if len(list_of_linspaces) == 0:
      return ret
    arr0 = ret


@njit(cache=True)
def product(list_of_arrays):
  """
    Numba equivalent of the ``itertools.product`` iterator with the difference
    that it can be used inside of Numba, works only with array inputs and
    creates all products at once.
  >>> linspaces = [np.linspace(0, 1, i) for i in (4, 5, 6)]
  >>> X = np.stack(list(map(np.ravel, np,meshgrid(*linspaces))), axis=1)
  >>> Y = _product(linspaces)
  >>> np.allclose(X, Y)
      True
  """
  # XXX: make a Numba equivalent of consecutive iterator creation
  assert len(list_of_arrays) >= 1
  if len(list_of_arrays) == 1:
    return list_of_arrays[0][:, None]
  list_of_arrays = list_of_arrays[::-1]
  return _product(list_of_arrays[0][:, None], list_of_arrays[1:])


@njit(cache=True)
def linspace_product(array_of_steps):
  """
    Convenience function for creating a product of linspaces
    over [0, 1] from an array of integers representing the steps.
  """
  list_of_arrays = []
  for i in array_of_steps:
    list_of_arrays.append(np.linspace(0, 1, i))
  return product(list_of_arrays)


@njit(cache=True)
def arange_product(array_of_integers):
  """
    Convenience function for creating a product of aranges
    from [0, i) from an array of integers containing the `i`.
  """
  list_of_arrays = []
  for i in array_of_integers:
    list_of_arrays.append(np.arange(i).astype(np.int64))
  return product(list_of_arrays)


"""
  Formatting to strings for homogeneous string-based Numba hashing.
"""


@njit(cache=True)
def cut_trail(f_str):
  # XXX: docstring
  cut = 0
  for c in f_str[::-1]:
    if c == "0":
      cut += 1
    else:
      break
  if cut == 0:
    for c in f_str[::-1]:
      if c == "9":
        cut += 1
      else:
        cut -= 1
        break
  if cut > 0:
    f_str = f_str[:-cut]
  if f_str == "":
    f_str = "0"
  return f_str


@njit(cache=True)
def float2str(value):
  # XXX: docstring
  if math.isnan(value):
    return "nan"
  elif value == 0.0:
    return "0.0"
  elif value < 0.0:
    return "-" + float2str(-value)
  elif math.isinf(value):
    return "inf"
  else:
    max_digits = 16
    min_digits = -4
    e10 = math.floor(math.log10(value)) if value != 0.0 else 0
    if min_digits < e10 < max_digits:
      i_part = math.floor(value)
      f_part = math.floor((1 + value % 1) * 10.0 ** max_digits)
      i_str = str(i_part)
      f_str = cut_trail(str(f_part)[1:max_digits - e10])
      return i_str + "." + f_str
    else:
      m10 = value / 10.0 ** e10
      i_part = math.floor(m10)
      f_part = math.floor((1 + m10 % 1) * 10.0 ** max_digits)
      i_str = str(i_part)
      f_str = cut_trail(str(f_part)[1:max_digits])
      e_str = str(e10)
      if e10 >= 0:
        e_str = "+" + e_str
      return i_str + "." + f_str + "e" + e_str


""" np.ufunc equivalents """


@njit(cache=True)
def mul_reduce(tpl):
  ret = 1
  for fac in tpl:
    ret *= fac
  return ret


@njit(cache=True)
def add_reduce(tpl):
  ret = 0
  for item in tpl:
    ret += item
  return ret


"""
  Various custom implementations of numpy functions not yet supported in Numba
"""


@njit(cache=True)
def ravel_multi_index(multi_index, dims):
  flat_index = 0
  stride = 1

  # Loop through dimensions in reverse order to calculate the flat index
  for i in range(len(dims) - 1, -1, -1):
    flat_index += multi_index[i] * stride
    stride *= dims[i]

  return flat_index
