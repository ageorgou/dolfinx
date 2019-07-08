// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormCoefficients.h"
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <memory>
#include <string>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FormCoefficients::FormCoefficients(
    const std::vector<std::tuple<int, std::string,
                                 std::shared_ptr<function::Function>>>& coeffs)
{
  for (const auto& coeff : coeffs)
  {
    _original_pos.push_back(std::get<0>(coeff));
    _names.push_back(std::get<1>(coeff));
    _coefficients.push_back(std::get<2>(coeff));
  }
  _constants.resize(_coefficients.size());
}
//-----------------------------------------------------------------------------
int FormCoefficients::size() const { return _coefficients.size(); }
//-----------------------------------------------------------------------------
std::vector<int> FormCoefficients::offsets() const
{
  std::vector<int> n = {0};

  // Same list size for coefficients and constants: any which are not set
  // should be nullptr.
  assert(_coefficients.size() == _constants.size());

  for (std::size_t i = 0; i < _coefficients.size(); ++i)
  {
    if (_coefficients[i])
    {
      assert(!_constants[i]);
      n.push_back(
          n.back()
          + _coefficients[i]->function_space()->element->space_dimension());
    }
    else if (_constants[i])
      n.push_back(n.back() + _constants[i]->size());
    else
      throw std::runtime_error(
          "Not all form coefficients/constants have been set.");
  }
  return n;
}
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, 1>
FormCoefficients::array(const std::vector<int>& offsets) const
{
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Copy constants into array
  // FIXME: why copy? maybe we just store the "coeff_array" here prefilled?
  for (std::size_t i = 0; i < _constants.size(); ++i)
  {
    if (_constants[i])
    {
      std::copy(_constants[i]->data(),
                _constants[i]->data() + _constants[i]->size(),
                coeff_array.data() + offsets[i]);
    }
  }

  return coeff_array;
}
//-----------------------------------------------------------------------------
void FormCoefficients::set(
    int i, std::shared_ptr<const function::Function> coefficient)
{
  if (i >= (int)_coefficients.size())
  {
    _coefficients.resize(i + 1);
    _constants.resize(i + 1);
  }

  _coefficients[i] = coefficient;
}
//-----------------------------------------------------------------------------
void FormCoefficients::set(
    std::string name, std::shared_ptr<const function::Function> coefficient)
{
  int i = get_index(name);
  if (i >= (int)_coefficients.size())
  {
    _coefficients.resize(i + 1);
    _constants.resize(i + 1);
  }

  _coefficients[i] = coefficient;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const function::Function> FormCoefficients::get(int i) const
{
  assert(i < (int)_coefficients.size());
  return _coefficients[i];
}
//-----------------------------------------------------------------------------
void FormCoefficients::set_const(
    int i, std::shared_ptr<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>>
               constant)
{
  if (i >= (int)_constants.size())
  {
    _coefficients.resize(i + 1);
    _constants.resize(i + 1);
  }

  _constants[i] = constant;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>>
FormCoefficients::get_const(int i) const
{
  assert(i < (int)_constants.size());
  return _constants[i];
}
//-----------------------------------------------------------------------------
int FormCoefficients::original_position(int i) const
{
  assert(i < (int)_original_pos.size());
  return _original_pos[i];
}
//-----------------------------------------------------------------------------
int FormCoefficients::get_index(std::string name) const
{
  auto it = std::find(_names.begin(), _names.end(), name);
  if (it == _names.end())
    throw std::runtime_error("Cannot find coefficient name:" + name);

  return std::distance(_names.begin(), it);
}
//-----------------------------------------------------------------------------
std::string FormCoefficients::get_name(int i) const
{
  if (i >= (int)_names.size())
    throw std::runtime_error("Invalid coefficient index");

  return _names[i];
}
//-----------------------------------------------------------------------------
