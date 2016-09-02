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

#include "moduleutils.h"
#include "optimizer/tgvoptimizer.h"
#include "optimizer/tvoptimizer.h"

namespace bp = boost::python;

//==============================================================================
// Definitions
//==============================================================================
typedef float floatX;
typedef iu::type_trait<floatX>::complex_type complexX;
typedef iu::LinearDeviceMemory<floatX, 2> DataType2f;
typedef iu::LinearDeviceMemory<complexX, 2> DataType2c;
typedef iu::LinearDeviceMemory<complexX, 3> DataType3c;

//==============================================================================
// create python module
//==============================================================================

BOOST_PYTHON_MODULE(pydenoising)  // name must (!) be the same as the resulting *.so file
// get python ImportError about missing init function otherwise
// probably best to sort it out in cmake...
{
  import_array();                   // initialize numpy c-api
  bp::register_exception_translator<iu::python::Exc>(
      &iu::python::ExcTranslator);

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

  // TGV optimizer - 2D denoising
  bp::class_<TgvOptimizer<DataType2f, DataType2f>, boost::noncopyable>(
      "TgvOptimizer_2f", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TgvOptimizer, DataType2f, DataType2f>)
      .def("solve", &TgvOptimizer<DataType2f, DataType2f>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TgvOptimizer, DataType2f, DataType2f>)
      .def("getParameters", &TgvOptimizer<DataType2f, DataType2f>::getParameters, bp::return_internal_reference<>());

  // TV optimizer - 2D denoising
  bp::class_<TvOptimizer<DataType2f, DataType2f>, boost::noncopyable>(
      "TvOptimizer_2f", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TvOptimizer, DataType2f, DataType2f>)
      .def("solve", &TvOptimizer<DataType2f, DataType2f>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TvOptimizer, DataType2f, DataType2f>)
      .def("getParameters", &TvOptimizer<DataType2f, DataType2f>::getParameters, bp::return_internal_reference<>());

  // TGV optimizer - 2D complex denoising
  bp::class_<TgvOptimizer<DataType2c, DataType2c>, boost::noncopyable>(
      "TgvOptimizer_2c", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TgvOptimizer, DataType2c, DataType2c>)
      .def("solve", &TgvOptimizer<DataType2c, DataType2c>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TgvOptimizer, DataType2c, DataType2c>)
      .def("getParameters", &TgvOptimizer<DataType2c, DataType2c>::getParameters, bp::return_internal_reference<>());

  // TV optimizer - 2D complex denoising
  bp::class_<TvOptimizer<DataType2c, DataType2c>, boost::noncopyable>(
      "TvOptimizer_2c", bp::init<>())
      .def("setNoisyData", setNoisyOptimizerData<TvOptimizer, DataType2c, DataType2c>)
      .def("solve", &TvOptimizer<DataType2c, DataType2c>::solve, solve_optimizer_overloads())
      .def("getResult", getOptimizerResult<TvOptimizer, DataType2c, DataType2c>)
      .def("getParameters", &TvOptimizer<DataType2c, DataType2c>::getParameters, bp::return_internal_reference<>());
}
