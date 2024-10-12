"""
Module defining various custom errors.
"""


""" For use in splico.mesh.mesh """


class MissingVertexError(Exception):
  pass


class HasNoSubMeshError(Exception):
  pass


class HasNoBoundaryError(Exception):
  pass


class EmptyMeshError(Exception):
  pass


""" Container-type  / sized-type operations """


class UnequalLengthError(Exception):
  """ For operations that require two iterables to have equal length. """
  pass


class EmptyContainerError(Exception):
  """ For preventing instantiations of empty container-types. """
  pass
