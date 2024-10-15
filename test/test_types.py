from splico.types import Immutable, ensure_same_class

import unittest


class A(Immutable):
  _items = 'a',

  def __init__(self, a):
    self.a = float(a)

  @ensure_same_class
  def __le__(self, other):
    return self.a <= other.a


class B(Immutable):
  _items = 'a',

  def __init__(self, a):
    self.a = float(a)

  @ensure_same_class
  def __le__(self, other):
    return self.a <= other.a


class C:
  def __init__(self, a):
    self.a = float(a)

  def __eq__(self, other):
    if self.__class__ is not other.__class__:
      return NotImplemented
    return self.a == other.a

  def __le__(self, other):
    if self.__class__ is not other.__class__:
      return False
    return self.a <= other.a

  def __ge__(self, other):
    if self.__class__ is not other.__class__:
      return False
    return self.a >= other.a


class TestHashMixinEquality(unittest.TestCase):

  def test_equal_hashable(self):
    a = A(50)
    b = A(5)
    c = A(50)
    self.assertFalse(a == b)
    self.assertTrue(a == c)

  def test_unequal_hashable(self):
    a = A(50)
    b = B(50)
    self.assertFalse(a == b)
    self.assertRaises(TypeError, lambda: a <= b)

  def test_unequal(self):
    a = A(50)
    c = C(50)

    # with the implementation of C also returning NotImplemented when the
    # types don't match, we should get False in this case
    self.assertFalse(a == c)

    # since `C` implements `__ge__` and returns False in case the classes
    # are not the same, since the first comparison returns NotImplemented,
    # Python should fall back on `__ge__` which should return False.
    self.assertFalse(a <= c)


if __name__ == '__main__':
  unittest.main()
