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

#include "tvoptimizerwithop.h"

#include <iu/iucutil.h>
#include <iu/iumath.h>
#include <iu/iuhelpermath.h>

#include "definitions.h"
#include "tgv_kernels.cuh"

template<typename InputType, typename OutputType>
TvOptimizerWithOp<InputType, OutputType>::TvOptimizerWithOp() :
    op_(NULL)
{
}

template<typename InputType, typename OutputType>
TvOptimizerWithOp<InputType, OutputType>::~TvOptimizerWithOp()
{
}

template<typename InputType, typename OutputType>
InputType* TvOptimizerWithOp<InputType, OutputType>::getResult()
{
	if (u_ == nullptr)
	{
		// throw exception
		std::stringstream ss;
		ss << "There is no result yet!";
		throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
	}
	return u_.get();
}

template<typename InputType, typename OutputType>
void TvOptimizerWithOp<InputType, OutputType>::setOperator(
    const std::shared_ptr<OperatorBase<InputType, OutputType> >& op)
{
  op_ = op;
}

template<typename InputType, typename OutputType>
void TvOptimizerWithOp<InputType, OutputType>::setInput0(const InputType &input)
{
  // primal variables
  u_.reset(new InputType(input.size()));
  u__.reset(new InputType(input.size()));

  // dual variables
  p_.init(input.size());

  // copy input to u
  iu::copy(&input, u_.get());
  iu::copy(&input, u__.get());

}

template<typename InputType, typename OutputType>
void TvOptimizerWithOp<InputType, OutputType>::setNoisyData(
    const std::shared_ptr<OutputType> &f)
{
  // set data
  f_ = f;

  // init dual variable r
  r_.reset(new OutputType(f->size()));
  iu::math::fill(*r_, iu::type_trait<typename OutputType::pixel_type>::make(0));
}

template<typename InputType, typename OutputType>
void TvOptimizerWithOp<InputType, OutputType>::solve(bool verbose)
{
  if (op_ == nullptr)
  {
    // throw exception
    std::stringstream ss;
    ss << "No operator is set!";
    throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  if (f_ == nullptr)
  {
    // throw exception
    std::stringstream ss;
    ss << "No noisy data is set!";
    throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  if (u_ == nullptr)
  {
    std::cout << "Initialization with A^T(f)" << std::endl;
    InputType input0(op_->getInputSize(*f_));
    op_->adjoint(*f_, input0);
    this->setInput0(input0);
  }

  dim3 dimBlock(COMMON_BLOCK_SIZE_2D_X, COMMON_BLOCK_SIZE_2D_Y);
  dim3 dimGrid(iu::divUp(u_->size()[0], dimBlock.x),
               iu::divUp(u_->size()[1], dimBlock.y));

  dim3 dimBlock1d(COMMON_BLOCK_SIZE_2D_X * COMMON_BLOCK_SIZE_2D_Y);
  dim3 dimGrid1d(iu::divUp(r_->size().numel(), dimBlock.x));

  real_type tau = 1.0 / 8.0;
  real_type sigma = 1.0 / 8.0;
  real_type theta = 1.0;

  iu::IuCudaTimer timer;
  timer.start();

  std::cout << params_ << std::endl;

  for (int k = 0; k <= params_.max_iter; k++)
  {
    if (!(k % params_.check))
    {
      std::cout << "iter=" << k << " lambda=" << params_.lambda << std::endl;
    }

    // pre-compute K(u_)
    OutputType tmp(f_->size());
    op_->forward(*u__, tmp);

    // Compute K(u_) - f_ and store it in tmp
    iu::math::addWeighted(tmp, iu::type_trait < output_pixel_type > ::make(1.0),
                          *f_,
                          iu::type_trait < output_pixel_type > ::make(-1.0),
                          tmp);

    // dual step and projection on r
    TGV_prox_r_kernel<OutputType> <<<dimGrid1d, dimBlock1d>>>(*r_, tmp, sigma, params_.lambda);
    IU_CUDA_CHECK;

    // dual step and projection on p
    TV_dual_p_kernel<InputType> <<<dimGrid, dimBlock>>>(p_, *u__, sigma);
    IU_CUDA_CHECK;

    // pre-compute K*(r) and store it in u__
    op_->adjoint(*r_, *u__);

    // primal step on u
    TGV_primal_u_noprox_kernel<InputType> <<<dimGrid, dimBlock>>>(*u_, *u__, p_,
                                                                  tau, theta);
    IU_CUDA_CHECK;
  }

  std::cout << "reconstruction time " << timer.elapsed() << std::endl;
}

// explicit template instantiations
template class TvOptimizerWithOp<iu::LinearDeviceMemory<float2, 2>,
    iu::LinearDeviceMemory<float2, 3>> ;
template class TvOptimizerWithOp<iu::LinearDeviceMemory<float2, 2>,
    iu::LinearDeviceMemory<float2, 2>> ;
template class TvOptimizerWithOp<iu::LinearDeviceMemory<float, 2>,
    iu::LinearDeviceMemory<float, 3>> ;
template class TvOptimizerWithOp<iu::LinearDeviceMemory<double2, 2>,
    iu::LinearDeviceMemory<double2, 3>> ;
template class TvOptimizerWithOp<iu::LinearDeviceMemory<double2, 2>,
    iu::LinearDeviceMemory<double2, 2>> ;
template class TvOptimizerWithOp<iu::LinearDeviceMemory<float, 2>,
    iu::LinearDeviceMemory<float, 2>> ;
template class TvOptimizerWithOp<iu::LinearDeviceMemory<double, 2>,
    iu::LinearDeviceMemory<double, 2>> ;
template class TvOptimizerWithOp<iu::LinearDeviceMemory<double, 2>,
    iu::LinearDeviceMemory<double, 3>> ;
