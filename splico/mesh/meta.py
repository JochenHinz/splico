"""
Module containing the :type:`MeshMeta` metaclass for use in
:class:`splico.mesh.Mesh`
"""

from ..types import ImmutableMeta


class MeshMeta(ImmutableMeta):

  def __call__(cls, *args, **kwargs):
    # make sure that class implements all necessary class-level attributes
    # before instantiation.
    for item in ('simplex_type', 'ndims', 'nverts', 'is_affine'):
      if not hasattr(cls, item):
        raise TypeError("Cannot instantiate mesh class that does not implement"
                        f" the {item} class-level attribute.")
    return ImmutableMeta.__call__(cls, *args, **kwargs)
