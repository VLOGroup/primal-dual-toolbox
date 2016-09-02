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

#define BOOST_PYTHON_STATIC_LIB

#include <memory>
#include <boost/python/make_constructor.hpp>

#include "moduleutils.h"
#include "operator/mricartesianoperator.h"
#include "operator/mrisamplingoperator.h"
#ifdef WITH_GPUNUFFT
  #include "operator/mriradialoperator.h"
#endif

#include "optimizer/tvoptimizerwithop.h"
#include "optimizer/tgvoptimizerwithop.h"
#include "optimizer/tgvmrioptimizer.h"

namespace bp = boost::python;

//==============================================================================
// Definitions
//==============================================================================
typedef float floatX;
typedef iu::type_trait<floatX>::complex_type complexX;
typedef iu::LinearDeviceMemory<complexX, 2> DataType2c;
typedef iu::LinearDeviceMemory<complexX, 3> DataType3c;

//==============================================================================
// MRI Operator
//==============================================================================
/** Set Mask (real, 2D) */
template<template <typename, typename> class TOperator, typename TInput, typename TOutput>
void setMask(bp::object& self, bp::object& py_arr)
{
  TOperator<TInput, TOutput>& op = bp::extract<
      TOperator<TInput, TOutput>&>(self);
  iu::LinearHostMemory<floatX, 2> hostmem(py_arr);
  op.addConstant(hostmem);
}

/** Set coil sensitivities (complex, 3D) */
template<template <typename, typename> class TOperator, typename TInput, typename TOutput>
void setCoilSens(bp::object& self, bp::object& py_arr)
{
  TOperator<TInput, TOutput>& op = bp::extract<
      TOperator<TInput, TOutput>&>(self);
  iu::LinearHostMemory<complexX, 3> hostmem(py_arr);
  op.addConstant(hostmem);
}

/** Apply forward operation */
template<template <typename, typename> class TOperator, typename TInput, typename TOutput>
PyObject* forward(bp::object& self, bp::object& py_arr)
{
  TOperator<TInput, TOutput>& op = bp::extract<
      TOperator<TInput, TOutput>&>(self);
  iu::LinearHostMemory<typename TInput::pixel_type, TInput::ndim> src_h(py_arr);
  TInput src_d(src_h.size());
  iu::copy(&src_h, &src_d);
  TOutput dst_d(op.getOutputSize(src_d));
  op.forward(src_d, dst_d);
  return iu::python::PyArray_from_LinearDeviceMemory(dst_d);
}

/** Apply adjoint operation */
template<template <typename, typename> class TOperator, typename TInput, typename TOutput>
PyObject* adjoint(bp::object& self, bp::object& py_arr)
{
  TOperator<TInput, TOutput>& op = bp::extract<
      TOperator<TInput, TOutput>&>(self);
  iu::LinearHostMemory<typename TOutput::pixel_type, TOutput::ndim> src_h(py_arr);
  TOutput src_d(src_h.size());
  iu::copy(&src_h, &src_d);
  TInput dst_d(op.getInputSize(src_d));
  op.adjoint(src_d, dst_d);
  return iu::python::PyArray_from_LinearDeviceMemory(dst_d);
}

#ifdef WITH_GPUNUFFT
/** Define custom operator init for MriRadialOperator, taking a dict as input */
template<typename DataType2c, typename DataType3c>
std::shared_ptr<MriRadialOperator<DataType2c, DataType3c>> MriRadialOperator_Init(const bp::object & pyconfig)
{
  OpConfigDict config;
  mapFromPyObject(pyconfig, config);

  // Initialize op
  std::shared_ptr<MriRadialOperator<DataType2c, DataType3c> > op(new MriRadialOperator<DataType2c, DataType3c>(config));
  return op;
}
#endif

//==============================================================================
// create python module
//==============================================================================
BOOST_PYTHON_MODULE(pymrireconstruction)  // name must (!) be the same as the resulting *.so file
// get python ImportError about missing init function otherwise
// probably best to sort it out in cmake...
{
  import_array();                   // initialize numpy c-api
  bp::register_exception_translator<iu::python::Exc>(
      &iu::python::ExcTranslator);

  // TGV MRI parameters
  bp::class_<TgvMriParameters<floatX>>("TgvMriParameters", bp::init<>())
      .def(bp::self_ns::str(bp::self))  // allow debug printing
      .def_readwrite("reduction", &TgvMriParameters<floatX>::reduction)
      .def_readwrite("max_iter", &TgvMriParameters<floatX>::max_iter)
      .def_readwrite("check", &TgvMriParameters<floatX>::check)
      .def_readwrite("alpha0", &TgvMriParameters<floatX>::alpha0)
      .def_readwrite("alpha1", &TgvMriParameters<floatX>::alpha1);

  // TGV parameters
  bp::class_<TgvParameters<floatX>>("TgvParameters", bp::init<>())
      .def(bp::self_ns::str(bp::self))  // allow debug printing
      .def_readwrite("Lambda", &TgvParameters<floatX>::lambda)
      .def_readwrite("max_iter", &TgvParameters<floatX>::max_iter)
      .def_readwrite("check", &TgvParameters<floatX>::check)
      .def_readwrite("alpha0", &TgvParameters<floatX>::alpha0)
      .def_readwrite("alpha1", &TgvParameters<floatX>::alpha1);

  // TV parameters
  bp::class_<TvParameters<floatX>>("TvParameters", bp::init<>())
      .def(bp::self_ns::str(bp::self))  // allow debug printing
      .def_readwrite("Lambda", &TvParameters<floatX>::lambda)
      .def_readwrite("max_iter", &TvParameters<floatX>::max_iter)
      .def_readwrite("check", &TvParameters<floatX>::check);

  // Cartesian MRI operator
  bp::class_<MriCartesianOperator<DataType2c, DataType3c>,
      std::shared_ptr<MriCartesianOperator<DataType2c, DataType3c>>,
      boost::noncopyable>("MriCartesianOperator", bp::init<>())
      .def(bp::self_ns::str(bp::self))  // allow debug printing
      .def("setMask", setMask<MriCartesianOperator, DataType2c, DataType3c>)
      .def("setCoilSens", setCoilSens<MriCartesianOperator, DataType2c, DataType3c>)
      .def("forward", forward<MriCartesianOperator, DataType2c, DataType3c>)
      .def("adjoint", adjoint<MriCartesianOperator, DataType2c, DataType3c>);

  // Cartesian MRI operator, including removal of read-out oversampling.
  bp::class_<MriCartesianRemoveROOSOperator<DataType2c, DataType3c>,
      std::shared_ptr<MriCartesianRemoveROOSOperator<DataType2c, DataType3c>>,
      boost::noncopyable>("MriCartesianRemoveROOSOperator", bp::init<>())
      .def(bp::self_ns::str(bp::self))  // allow debug printing
      .def("setMask", setMask<MriCartesianRemoveROOSOperator, DataType2c, DataType3c>)
      .def("setCoilSens", setCoilSens<MriCartesianRemoveROOSOperator, DataType2c, DataType3c>)
      .def("forward", forward<MriCartesianRemoveROOSOperator, DataType2c, DataType3c>)
      .def("adjoint", adjoint<MriCartesianRemoveROOSOperator, DataType2c, DataType3c>);

#ifdef WITH_GPUNUFFT
  // Radial MRI operator
  bp::class_<MriRadialOperator<DataType2c, DataType2c>,
      std::shared_ptr<MriRadialOperator<DataType2c, DataType2c>>,
      boost::noncopyable>("MriRadialOperator", bp::no_init)
      .def("__init__", bp::make_constructor(&MriRadialOperator_Init<DataType2c, DataType2c>))
      .def(bp::self_ns::str(bp::self))  // allow debug printing
      .def("setTrajectory", setMask<MriRadialOperator, DataType2c, DataType2c>)
      .def("setDcf", setMask<MriRadialOperator, DataType2c, DataType2c>)
      .def("setCoilSens", setCoilSens<MriRadialOperator, DataType2c, DataType2c>)
      .def("forward", forward<MriRadialOperator, DataType2c, DataType2c>)
      .def("adjoint", adjoint<MriRadialOperator, DataType2c, DataType2c>);
#endif

  // Sampling MRI operator
  bp::class_<MriSamplingOperator<DataType2c, DataType2c>,
      std::shared_ptr<MriSamplingOperator<DataType2c, DataType2c>>,
      boost::noncopyable>("MriSamplingOperator", bp::init<>())
      .def(bp::self_ns::str(bp::self))  // allow debug printing
      .def("setMask", setMask<MriSamplingOperator, DataType2c, DataType2c>)
      .def("forward", forward<MriSamplingOperator, DataType2c, DataType2c>)
      .def("adjoint", adjoint<MriSamplingOperator, DataType2c, DataType2c>);

  // TGV MRI optimizer
  bp::class_<TgvMriOptimizer<DataType2c, DataType3c>, boost::noncopyable>(
      "TgvMriOptimizer_2c3c", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TgvMriOptimizer, DataType2c, DataType3c>)
      .def("setInput0", setOptimizerInput0<TgvMriOptimizer, DataType2c, DataType3c>)
      .def("setOperator", setOptimizerOperator<TgvMriOptimizer, MriCartesianOperator, DataType2c, DataType3c>)
      .def("setOperator", setOptimizerOperator<TgvMriOptimizer, MriCartesianRemoveROOSOperator, DataType2c, DataType3c>)
      .def("solve", &TgvMriOptimizer<DataType2c, DataType3c>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TgvMriOptimizer, DataType2c, DataType3c>)
      .def("getParameters", &TgvMriOptimizer<DataType2c, DataType3c>::getParameters, bp::return_internal_reference<>());

  // TGV optimizer - MriCartesianOperator, MriCartesianRemoveROOSOperator
  bp::class_<TgvOptimizerWithOp<DataType2c, DataType3c>, boost::noncopyable>(
      "TgvOptimizer_2c3c", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TgvOptimizerWithOp, DataType2c, DataType3c>)
      .def("setOperator", setOptimizerOperator<TgvOptimizerWithOp, MriCartesianOperator, DataType2c, DataType3c>)
      .def("setOperator", setOptimizerOperator<TgvOptimizerWithOp, MriCartesianRemoveROOSOperator, DataType2c, DataType3c>)
      .def("solve", &TgvOptimizerWithOp<DataType2c, DataType3c>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TgvOptimizerWithOp, DataType2c, DataType3c>)
      .def("getParameters", &TgvOptimizerWithOp<DataType2c, DataType3c>::getParameters, bp::return_internal_reference<>());

  // TV optimizer - MriCartesianOperator, MriCartesianRemoveROOSOperator
  bp::class_<TvOptimizerWithOp<DataType2c, DataType3c>, boost::noncopyable>(
      "TvOptimizerWithOp_2c3c", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TvOptimizerWithOp, DataType2c, DataType3c>)
      .def("setInput0", setOptimizerInput0<TvOptimizerWithOp, DataType2c, DataType3c>)
      .def("setOperator", setOptimizerOperator<TvOptimizerWithOp, MriCartesianOperator, DataType2c, DataType3c>)
      .def("setOperator", setOptimizerOperator<TvOptimizerWithOp, MriCartesianRemoveROOSOperator, DataType2c, DataType3c>)
      .def("solve", &TvOptimizerWithOp<DataType2c, DataType3c>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TvOptimizerWithOp, DataType2c, DataType3c>)
      .def("getParameters", &TvOptimizerWithOp<DataType2c, DataType3c>::getParameters, bp::return_internal_reference<>());

  // TGV MRI optimizer - MriRadialOperator, MriSamplingOperator
  bp::class_<TgvMriOptimizer<DataType2c, DataType2c>, boost::noncopyable>(
      "TgvMriOptimizer_2c2c", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TgvMriOptimizer, DataType2c, DataType2c>)
      .def("setInput0", setOptimizerInput0<TgvMriOptimizer, DataType2c, DataType2c>)
#ifdef WITH_GPUNUFFT
      .def("setOperator", setOptimizerOperator<TgvMriOptimizer, MriRadialOperator, DataType2c, DataType2c>)
#endif
      .def("setOperator", setOptimizerOperator<TgvMriOptimizer, MriSamplingOperator, DataType2c, DataType2c>)
      .def("solve", &TgvMriOptimizer<DataType2c, DataType2c>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TgvMriOptimizer, DataType2c, DataType2c>)
      .def("getParameters", &TgvMriOptimizer<DataType2c, DataType2c>::getParameters, bp::return_internal_reference<>());

  // TGV optimizer - MriRadialOperator, MriSamplingOperator
  bp::class_<TgvOptimizerWithOp<DataType2c, DataType2c>, boost::noncopyable>(
      "TgvOptimizer_2c2c", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TgvOptimizerWithOp, DataType2c, DataType2c>)
#ifdef WITH_GPUNUFFT
      .def("setOperator", setOptimizerOperator<TgvOptimizerWithOp, MriRadialOperator, DataType2c, DataType2c>)
#endif
      .def("setOperator", setOptimizerOperator<TgvOptimizerWithOp, MriSamplingOperator, DataType2c, DataType2c>)
      .def("solve", &TgvOptimizerWithOp<DataType2c, DataType2c>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TgvOptimizerWithOp, DataType2c, DataType2c>)
      .def("getParameters", &TgvOptimizerWithOp<DataType2c, DataType2c>::getParameters, bp::return_internal_reference<>());

  // TV optimizer - MriRadialOperator, MriSamplingOperator
  bp::class_<TvOptimizerWithOp<DataType2c, DataType2c>, boost::noncopyable>(
      "TvOptimizer_2c2c", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TvOptimizerWithOp, DataType2c, DataType2c>)
#ifdef WITH_GPUNUFFT
      .def("setOperator", setOptimizerOperator<TvOptimizerWithOp, MriRadialOperator, DataType2c, DataType2c>)
#endif
      .def("setOperator", setOptimizerOperator<TvOptimizerWithOp, MriSamplingOperator, DataType2c, DataType2c>)
      .def("solve", &TvOptimizerWithOp<DataType2c, DataType2c>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TvOptimizerWithOp, DataType2c, DataType2c>)
      .def("getParameters", &TvOptimizerWithOp<DataType2c, DataType2c>::getParameters, bp::return_internal_reference<>());
}
