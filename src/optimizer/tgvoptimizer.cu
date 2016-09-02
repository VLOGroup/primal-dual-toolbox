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

#include "tgvoptimizer.h"

#include <iu/iucutil.h>
#include <iu/iuhelpermath.h>

#include "definitions.h"
#include "tgv_kernels.cuh"

template<typename PixelType>
TgvParameters<PixelType>::TgvParameters() :
    alpha0(1), alpha1(1), lambda(1), max_iter(1), check(100)
{
}

template<typename PixelType>
TgvParameters<PixelType>::~TgvParameters()
{
}

template<typename InputType, typename OutputType>
TgvOptimizer<InputType, OutputType>::TgvOptimizer()
{
}

template<typename InputType, typename OutputType>
TgvOptimizer<InputType, OutputType>::~TgvOptimizer()
{
}

template<typename InputType, typename OutputType>
InputType* TgvOptimizer<InputType, OutputType>::getResult()
{
  return u_.get();
}

template<typename InputType, typename OutputType>
void TgvOptimizer<InputType, OutputType>::setInput0(const InputType &input)
{
  // primal variables
  u_.reset(new InputType(input.size()));
  u__.reset(new InputType(input.size()));
  v_.init(input.size());
  v__.init(input.size());

  // dual variables
  p_.init(input.size());
  q_.init(input.size());

  // copy input to u
  iu::copy(&input, u_.get());
  iu::copy(&input, u__.get());

}

template<typename InputType, typename OutputType>
void TgvOptimizer<InputType, OutputType>::setNoisyData(
    const std::shared_ptr<OutputType> &f)
{
  // set data
  f_ = f;
}

template<typename InputType, typename OutputType>
void TgvOptimizer<InputType, OutputType>::solve(bool verbose)
{
  if (f_ == nullptr)
  {
    // throw exception
    std::stringstream ss;
    ss << "No noisy data is set!";
    throw IuException(ss.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  if (u_ == nullptr)
  {
    std::cout << "Initialization with f" << std::endl;
    iu::Size<2> size(f_->size()[0], f_->size()[1]);
    this->setInput0(*f_);
  }

  dim3 dimBlock(COMMON_BLOCK_SIZE_2D_X, COMMON_BLOCK_SIZE_2D_Y);
  dim3 dimGrid(iu::divUp(u_->size()[0], dimBlock.x),
               iu::divUp(u_->size()[1], dimBlock.y));

  real_type tau = 1.0 / sqrt(12.0);
  real_type sigma = 1.0 / sqrt(12.0);
  real_type theta = 1.0;

  iu::IuCudaTimer timer;
  timer.start();

  std::cout << params_ << std::endl;

  for (unsigned int k = 0; k <= params_.max_iter; k++)
  {
    if (!(k % params_.check))
    {
      std::cout << "iter=" << k << std::endl;
    }

    // primal step on u
    TV_primal_u_kernel<InputType> <<<dimGrid, dimBlock>>>(*u_, *u__, *f_, p_, tau,
                                                          params_.lambda, theta);

    IU_CUDA_CHECK;

    // primal step on v
    TGV_primal_v_kernel<InputType> <<<dimGrid, dimBlock>>>(v_, v__, p_, q_, tau,
                                                           theta);
    IU_CUDA_CHECK;

    // dual step and projection on p
    TGV_dual_p_kernel<InputType> <<<dimGrid, dimBlock>>>(p_, *u__, v__, sigma,
                                                         params_.alpha1);
    IU_CUDA_CHECK;

    // dual step and prox on q
    TGV_dual_q_kernel<InputType> <<<dimGrid, dimBlock>>>(q_, v__, sigma,
                                                         params_.alpha0);

  }

  std::cout << "reconstruction time " << timer.elapsed() << std::endl;
}

// explicit template instantiations
template class TgvOptimizer<iu::LinearDeviceMemory<float2, 2>,
    iu::LinearDeviceMemory<float2, 2>> ;
template class TgvOptimizer<iu::LinearDeviceMemory<double2, 2>,
    iu::LinearDeviceMemory<double2, 2>> ;
template class TgvOptimizer<iu::LinearDeviceMemory<float, 2>,
    iu::LinearDeviceMemory<float, 2>> ;
template class TgvOptimizer<iu::LinearDeviceMemory<double, 2>,
    iu::LinearDeviceMemory<double, 2>> ;

template class TgvParameters<float>;
template class TgvParameters<double>;
