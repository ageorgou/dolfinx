# -*- coding: utf-8 -*-
# Copyright (C) 2009-2019 Johan Hake, Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Finite element functions"""

import typing
from functools import singledispatch

import numpy as np
from petsc4py import PETSc

import ufl
from dolfin import common, cpp, function
from dolfin.function import functionspace


class Function(ufl.Coefficient):
    """A finite element function that is represented by a function
    space (domain, element and dofmap) and a vector holding the
    degrees-of-freedom

    """

    def __init__(self,
                 V: functionspace.FunctionSpace,
                 x: typing.Optional[PETSc.Vec] = None,
                 name: typing.Optional[str] = None):
        """Initialize finite element Function."""

        # Create cpp Function
        if x is not None:
            self._cpp_object = cpp.function.Function(V._cpp_object, x)
        else:
            self._cpp_object = cpp.function.Function(V._cpp_object)

        # Initialize the ufl.FunctionSpace
        super().__init__(V.ufl_function_space(), count=self._cpp_object.id)

        # Set name
        if name is None:
            self.name = "f_{}".format(self.count())
        else:
            self.name = name

        # Store DOLFIN FunctionSpace object
        self._V = V

    def function_space(self) -> function.functionspace.FunctionSpace:
        """Return the FunctionSpace"""
        return self._V

    def value_rank(self) -> int:
        return self._cpp_object.value_rank()

    def value_dimension(self, i) -> int:
        return self._cpp_object.value_dimension(i)

    def value_shape(self):
        return self._cpp_object.value_shape

    def __call__(self, x: np.ndarray, bb_tree: cpp.geometry.BoundingBoxTree) -> np.ndarray:
        """Evaluate Function at points x, where x has shape (num_points, gdim)"""
        _x = np.asarray(x, dtype=np.float)
        num_points = _x.shape[0] if len(_x.shape) > 1 else 1
        _x = np.reshape(_x, (num_points, -1))
        if _x.shape[1] != self.geometric_dimension():
            raise ValueError("Wrong geometric dimension for coordinate(s).")

        value_size = ufl.product(self.ufl_element().value_shape())
        if common.has_petsc_complex:
            values = np.empty((num_points, value_size), dtype=np.complex128)
        else:
            values = np.empty((num_points, value_size))

        # Call the evaluation
        self._cpp_object.eval(values, _x, bb_tree)
        if num_points == 1:
            values = np.reshape(values, (-1, ))

        return values

    def eval_cell(self, u, x, cell):
        return self._cpp_object.eval(u, x, cell)

    def eval(self, u, x, bb_tree: cpp.geometry.BoundingBoxTree):
        return self._cpp_object.eval(u, x, bb_tree)

    def interpolate(self, u) -> None:
        """Interpolate an expression"""

        @singledispatch
        def _interpolate(u):
            try:
                self._cpp_object.interpolate(u)
            except TypeError:
                self._cpp_object.interpolate(u._cpp_object)

        @_interpolate.register(int)
        def _(u, verbose=False):
            self._cpp_object.interpolate_ptr(u)

        _interpolate(u)

    def compute_point_values(self, mesh=None):
        if mesh is not None:
            return self._cpp_object.compute_point_values(mesh)
        else:
            return self._cpp_object.compute_point_values()

    def copy(self):
        """Return a copy of the Function. The FunctionSpace is shared and the
        degree-of-freedom vector is copied.

        """
        return function.Function(self.function_space(),
                                 self._cpp_object.vector().copy())

    def vector(self):
        """Return the vector holding Function degrees-of-freedom."""
        return self._cpp_object.vector()

    def name(self) -> str:
        """Return name of the Function."""
        return self._cpp_object.name

    @property
    def id(self) -> int:
        """Return object id index."""
        return self._cpp_object.id

    def __str__(self):
        """Return a pretty print representation of it self."""
        return self.name

    def sub(self, i: int):
        """Return a sub function.

        The sub functions are numbered from i = 0..N-1, where N is the
        total number of sub spaces.

        """
        return Function(
            self._V.sub(i), self.vector(), name="{}-{}".format(str(self), i))

    def split(self):
        """Extract any sub functions.

        A sub function can be extracted from a discrete function that
        is in a mixed, vector, or tensor FunctionSpace. The sub
        function resides in the subspace of the mixed space.

        """
        num_sub_spaces = self.function_space().num_sub_spaces()
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        return tuple(self.sub(i) for i in range(num_sub_spaces))

    def collapse(self):
        u_collapsed = self._cpp_object.collapse()
        V_collapsed = functionspace.FunctionSpace(None, self.ufl_element(),
                                                  u_collapsed.function_space())
        return Function(V_collapsed, u_collapsed.vector())


class Constant(ufl.Coefficient):
    def __init__(self, value: typing.Union[float, list], mesh: cpp.mesh.Mesh, name: str = None):
        self._value = np.asarray(value)

        # Prepare a FE Function on Real element
        if self._value.ndim == 0:
            el = ufl.FiniteElement("R", mesh.ufl_cell(), 0)
        elif self._value.ndim == 1:
            el = ufl.VectorElement("R", mesh.ufl_cell(), 0, dim=self._value.shape[0])
        elif self._value.ndim >= 2:
            el = ufl.TensorElement("R", mesh.ufl_cell(), 0, shape=self._value.shape)

        V = function.FunctionSpace(mesh, el)
        self._cpp_object = cpp.function.Function(V._cpp_object)

        # Initialize the ufl.FunctionSpace
        super().__init__(V.ufl_function_space(), count=self._cpp_object.id)

        # Store DOLFIN FunctionSpace object
        self._V = V

        # Set public interface value to be the numpy array
        self.value = self._value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = np.asarray(value)

        def constant_eval(values, x):
            values[:, :] = self._value.flatten()

        self._cpp_object.interpolate(constant_eval)
