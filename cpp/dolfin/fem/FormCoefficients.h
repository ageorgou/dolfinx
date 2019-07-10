// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <petscsys.h>
#include <string>
#include <vector>

namespace dolfin
{

namespace function
{
class Function;
}

namespace fem
{
class Constant;
class FiniteElement;

/// Storage for the coefficients of a Form consisting of Function and
/// the Element objects they are defined on.

class FormCoefficients
{
public:
  /// Initialise the FormCoefficients, using tuples of
  /// (original_coeff_position, name, shared_ptr<function::Function>). The
  /// shared_ptr<Function> may be a nullptr and assigned later.
  FormCoefficients(
      const std::vector<std::tuple<int, std::string, int>>& coefficients);

  /// Get number of coefficients
  int size() const;

  /// Offset for each coefficient expansion array on a cell. Use to pack
  /// data for multiple coefficients in a flat array. The last entry is
  /// the size required to store all coefficients.
  const std::vector<int>& offsets() const;

  /// Set coefficient with index i to be a Function
  void set(int i, std::shared_ptr<const function::Function> coefficient);

  /// Set coefficient with name to be a Function
  void set(std::string name,
           std::shared_ptr<const function::Function> coefficient);

  /// Get the Function coefficient i
  std::shared_ptr<const function::Function> get(int i) const;

  /// Set constant coefficient i
  void set_const(int i, std::shared_ptr<fem::Constant> constant);

  /// Get constant coefficient i
  //  Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>>
  //  get_const(int i) const;

  /// Original position of coefficient in UFL form
  int original_position(int i) const;

  /// Get index from name of coefficient
  int get_index(std::string name) const;

  /// Get name from index of coefficient
  std::string get_name(int index) const;

  // Return an array of sufficient size to contain all coefficients and
  // constants, prefilled with any constant values.
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> array() const;

private:
  // Functions for the coefficients
  std::vector<std::shared_ptr<const function::Function>> _coefficients;

  // Constant coefficients
  std::vector<std::shared_ptr<fem::Constant>> _constants;

  // Copy of 'original positions' in UFL form
  std::vector<int> _original_pos;

  // Names of coefficients
  std::vector<std::string> _names;

  std::vector<int> _offsets;
};
} // namespace fem
} // namespace dolfin
