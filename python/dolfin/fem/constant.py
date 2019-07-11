# -*- coding: utf-8 -*-
# Copyright (C) 2019 Chris N. Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
import dolfin
import numpy

class Constant(ufl.Coefficient):

    def __init__(self, domain, val, count=None):
        domain = ufl.as_domain(domain)
        element = ufl.FiniteElement("Real", domain.ufl_cell(), 0)
        fs = ufl.FunctionSpace(domain, element)
        self._value = dolfin.cpp.fem.Constant([val])
        super().__init__(fs, count=count)

    def __setattr__(self, name, value):
        if name == 'value':
            arr = self._value.array()
            arr[:] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name != 'value':
            raise AttributeError("No such attribute")
        else:
            arr = self._value.array()
            return arr[0, 0]

class VectorConstant(ufl.Coefficient):

    def __init__(self, domain, val, count=None):
        dim = len(val)
        domain = ufl.as_domain(domain)
        element = ufl.VectorElement("Real", domain.ufl_cell(), 0, dim)
        fs = ufl.FunctionSpace(domain, element)
        self._value = dolfin.cpp.fem.Constant([val])
        super().__init__(fs, count=count)

    def __setattr__(self, name, value):
        if name == 'value':
            arr = self._value.array()
            arr[:] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name != 'value':
            raise AttributeError("No such attribute")
        else:
            arr = self._value.array()
            return arr[0]
