// This file is part of primal-dual-toolbox.
//
// Copyright (C) 2018 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
// Institute of Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/research/team-pock/
//
// primal-dual-toolbox is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// primal-dual-toolbox is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include "optimizer/tgvoptimizer.h"
#include "optimizer/tvoptimizer.h"
#include "operator/operatorbase.h"
#include <iu/iupython.h>
#include <boost/python/overloads.hpp>
#include <string>

/** Set noisy input data for optimizer */
template<template <typename, typename> class TOptimizer, typename TInput, typename TOutput>
void setNoisyOptimizerData(bp::object& self, bp::object& py_arr)
{
  TOptimizer<TInput, TOutput>& optimizer = bp::extract<
      TOptimizer<TInput, TOutput>&>(self);
  iu::LinearHostMemory<typename TOutput::pixel_type, TOutput::ndim> hostmem(
      py_arr);
  std::shared_ptr<TOutput> devicemem(new TOutput(hostmem.size()));
  iu::copy(&hostmem, devicemem.get());
  optimizer.setNoisyData(devicemem);
}

/** Set initial input for optimizer */
template<template <typename, typename> class TOptimizer, typename TInput, typename TOutput>
void setOptimizerInput0(bp::object& self, bp::object& py_arr)
{
  TOptimizer<TInput, TOutput>& optimizer = bp::extract<
      TOptimizer<TInput, TOutput>&>(self);
  iu::LinearHostMemory<typename TInput::pixel_type, TInput::ndim> hostmem(
      py_arr);
  TInput devicemem(hostmem.size());
  iu::copy(&hostmem, &devicemem);
  optimizer.setInput0(devicemem);
}

/** Set operator for optimizer */
template<template <typename, typename> class TOptimizer, template <typename, typename> class TOperator, typename TInput, typename TOutput>
void setOptimizerOperator(
    bp::object& self,
    std::shared_ptr<TOperator<TInput, TOutput>> op)
{
  TOptimizer<TInput, TOutput>& optimizer = bp::extract<
      TOptimizer<TInput, TOutput>&>(self);
  optimizer.setOperator(op);
}

/** Get result from optimizer */
template<template <typename, typename> class TOptimizer, typename TInput, typename TOutput>
PyObject* getOptimizerResult(bp::object& self)
{
  TOptimizer<TInput, TOutput>& optimizer = bp::extract<
      TOptimizer<TInput, TOutput>&>(self);
  return iu::python::PyArray_from_LinearDeviceMemory(*optimizer.getResult());
}

//==============================================================================
// boost python overloads
//==============================================================================
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(solve_optimizer_overloads, solve, 0, 1)

/** Extract OpConfigDict from python dict */
void mapFromPyObject(const bp::object &py_ob, OpConfigDict &out_map)
{
  bp::dict py_dict = bp::extract<bp::dict>(py_ob);
  boost::python::list keys = py_dict.keys();
  for (int i = 0; i < bp::len(keys); ++i)
  {
    // workaround - extract char* instead of std::string. otherwise I get following error:
    // 'TypeError: No registered converter was able to produce a C++ rvalue of type std::string from this Python object of type str'
    std::string key = std::string(bp::extract<char*>(keys[i]));
    bp::object py_val_fun = py_dict[key].attr("__str__");
    bp::object py_val = py_val_fun();
    std::string value = std::string(bp::extract<char*>(py_val));
    out_map[key] = value;
  }
}
