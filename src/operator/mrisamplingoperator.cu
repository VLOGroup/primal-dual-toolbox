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

#include "mrisamplingoperator.h"

#include <iu/iumath.h>
#include <iu/iuhelpermath.h>
#include <iu/iudefs.h>

#include "definitions.h"
#include "iurandom.h"

template<typename InputType, typename OutputType>
MriSamplingOperator<InputType, OutputType>::MriSamplingOperator() :
    OperatorBase<InputType, OutputType>(1, "MriSamplingOperator")
{
}

template<typename InputType, typename OutputType>
MriSamplingOperator<InputType, OutputType>::~MriSamplingOperator()
{
}

template<typename InputType, typename OutputType>
void MriSamplingOperator<InputType, OutputType>::sizeCheck(
    const InputType & src, const OutputType & dst)
{
  IU_SIZE_CHECK(dst.size(), (this->template getConstant<real_type, 2>(0))->size());
  IU_SIZE_CHECK(src.size(), (this->template getConstant<real_type, 2>(0))->size());
}

template<typename InputType, typename OutputType>
iu::Size<InputType::ndim> MriSamplingOperator<InputType, OutputType>::getInputSize(const OutputType & output)
{
  return output.size();
}

template<typename InputType, typename OutputType>
iu::Size<OutputType::ndim> MriSamplingOperator<InputType, OutputType>::getOutputSize(const InputType & input)
{
  return input.size();
}

template<typename InputType, typename OutputType>
void MriSamplingOperator<InputType, OutputType>::executeForward(const InputType & src,
                                                          OutputType & dst)
{
  // extract constants
  auto mask = this->template getConstant<real_type, 2>(0);

  // centered fft
  iu::math::fft::fft2c(src, dst, true);

  // apply sampling mask
  iu::math::complex::multiply(dst, *mask, dst);
}

template<typename InputType, typename OutputType>
void MriSamplingOperator<InputType, OutputType>::executeAdjoint(
    const OutputType & src, InputType & dst)
{
  // extract constants
  auto mask = this->template getConstant<real_type, 2>(0);

  // allocate temporary memory
  InputType tmp(src.size());
  iu::copy(&src, &tmp);

  // apply sampling mask
  iu::math::complex::multiply(tmp, *mask, tmp);

  // centered ifft
  iu::math::fft::ifft2c(tmp, dst, true);
}

template<typename InputType, typename OutputType>
void MriSamplingOperator<InputType, OutputType>::adjointnessCheck()
{
  std::cout << "Test adjointness of operator: " << *this << std::endl;

  InputType u(this->template getConstant<real_type, 2>(0)->size());
  OutputType p(u.size());
  iu::random::fillRandomFloatingNumbers(u);
  iu::random::fillRandomFloatingNumbers(p);
  OutputType Au(p.size());
  InputType Atp(u.size());

  executeForward(u, Au);
  executeAdjoint(p, Atp);

  TEST_ADJOINTNESS(u, Au, p, Atp);
}

// explicit template instantiations
template class MriSamplingOperator<iu::LinearDeviceMemory<float2, 2>,
    iu::LinearDeviceMemory<float2, 2>> ;
template class MriSamplingOperator<iu::LinearDeviceMemory<double2, 2>,
    iu::LinearDeviceMemory<double2, 2>> ;
