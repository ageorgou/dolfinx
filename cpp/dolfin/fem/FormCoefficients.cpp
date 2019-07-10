// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FormCoefficients.h"

#include <dolfin/fem/Constant.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <memory>
#include <string>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
FormCoefficients::FormCoefficients(
    const std::vector<std::tuple<int, std::string, int>>& coeffs)
    : _coefficients(coeffs.size()), _constants(coeffs.size()), _offsets(1, 0)
{
  for (const auto& coeff : coeffs)
  {
    _original_pos.push_back(std::get<0>(coeff));
    _names.push_back(std::get<1>(coeff));
    _offsets.push_back(_offsets.back() + std::get<2>(coeff));
  }
}
//-----------------------------------------------------------------------------
int FormCoefficients::size() const { return _coefficients.size(); }
//-----------------------------------------------------------------------------
const std::vector<int>& FormCoefficients::offsets() const { return _offsets; }
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, 1> FormCoefficients::array() const
{
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(_offsets.back());

  // Copy constants into array
  for (std::size_t i = 0; i < _constants.size(); ++i)
  {
    if (_constants[i])
    {
      std::copy(_constants[i]->value.data(),
                _constants[i]->value.data() + _constants[i]->value.size(),
                coeff_array.data() + _offsets[i]);
    }
  }

  return coeff_array;
}
//-----------------------------------------------------------------------------
void FormCoefficients::set(
    int i, std::shared_ptr<const function::Function> coefficient)
{
  int coeff_size = coefficient->function_space()->element->space_dimension();

  if (i > (int)_coefficients.size())
    throw std::runtime_error("Cannot add coefficient");
  else if (i == (int)_coefficients.size())
  {
    _coefficients.push_back(coefficient);
    _constants.push_back(nullptr);
    _offsets.push_back(_offsets.back() + coeff_size);
    return;
  }

  if (_offsets[i + 1] - _offsets[i] != coeff_size)
    throw std::runtime_error("Invalid coefficient size");

  _coefficients[i] = coefficient;
  _constants[i] = nullptr;
}
//-----------------------------------------------------------------------------
void FormCoefficients::set(
    std::string name, std::shared_ptr<const function::Function> coefficient)
{
  int i = get_index(name);
  this->set(i, coefficient);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const function::Function> FormCoefficients::get(int i) const
{
  assert(i < (int)_coefficients.size());
  return _coefficients[i];
}
//-----------------------------------------------------------------------------
void FormCoefficients::set_const(int i, std::shared_ptr<fem::Constant> constant)

{
  if (i > (int)_constants.size())
    throw std::runtime_error("Cannot add constant");
  else if (i == (int)_constants.size())
  {
    _coefficients.push_back(nullptr);
    _constants.push_back(constant);
    _offsets.push_back(_offsets.back() + constant->value.size());
    return;
  }

  if (_offsets[i + 1] - _offsets[i] != constant->value.size())
    throw std::runtime_error("Invalid constant size");

  _constants[i] = constant;
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
