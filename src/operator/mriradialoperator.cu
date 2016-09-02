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

#include "mriradialoperator.h"

#include <iu/iumath.h>
#include <iu/iuhelpermath.h>
#include <iu/iudefs.h>

#include "definitions.h"
#include "iurandom.h"

template<typename InputType, typename OutputType>
MriRadialOperator<InputType, OutputType>::MriRadialOperator(const OpConfigDict &config) :
    OperatorBase<InputType, OutputType>(3, "MriRadialOperator", config, 4, 4)
{
  osf_ = this->getConfigDouble("osf");
  sector_width_ = this->getConfigInt("sector_width");
  kernel_width_ = this->getConfigInt("kernel_width");
  img_dim_ = this->getConfigInt("img_dim");
}

template<typename InputType, typename OutputType>
MriRadialOperator<InputType, OutputType>::~MriRadialOperator()
{
}

template<typename InputType, typename OutputType>
void MriRadialOperator<InputType, OutputType>::sizeCheck(
    const InputType & src, const OutputType & dst)
{
  IU_SIZE_CHECK(src.size(), iu::Size<2>({img_dim_, img_dim_}));
  const int nCh = dst.size()[1];
  const int kspace_dim = dst.size()[0];
  IU_SIZE_CHECK((this->template getConstant<real_type, 2>(0))->size(), iu::Size<2>({kspace_dim, 2}));
  IU_SIZE_CHECK((this->template getConstant<real_type, 2>(1))->size(), iu::Size<2>({kspace_dim, 1}));
  IU_SIZE_CHECK((this->template getConstant<complex_type, 3>(2))->size(), iu::Size<3>({img_dim_,  img_dim_, nCh}));
}

template<typename InputType, typename OutputType>
iu::Size<InputType::ndim> MriRadialOperator<InputType, OutputType>::getInputSize(const OutputType& output)
{
  iu::Size<2> input_size({img_dim_, img_dim_});
  return input_size;
}

template<typename InputType, typename OutputType>
iu::Size<OutputType::ndim> MriRadialOperator<InputType, OutputType>::getOutputSize(const InputType& input)
{
  auto iu_trajectory = this->template getConstant<real_type, 2>(0);
  auto iu_sensitivities = this->template getConstant<complex_type, 3>(2);
  const int kspace_dim = iu_trajectory->size()[0];
  const int nCh = iu_sensitivities->size()[2];
  iu::Size<2> output_size({kspace_dim, nCh});
  return output_size;
}

template<typename InputType, typename OutputType>
void MriRadialOperator<InputType, OutputType>::createNufftOperator()
{
  // extract constants
  auto iu_trajectory = this->template getConstant<real_type, 2>(0);
  auto iu_dcf = this->template getConstant<real_type, 2>(1);
  auto iu_coil_sens = this->template getConstant<complex_type, 3>(2);

  gpuNUFFT::Dimensions img_dims;
  img_dims.width = img_dim_;
  img_dims.height = img_dim_;
  img_dims.depth = 0;

  gpuNUFFT::Array<real_type> trajectory;
  trajectory.dim.length = iu_trajectory->size()[0];
  trajectory.data = iu_trajectory->data();

  gpuNUFFT::Array<real_type> dcf;
  dcf.dim.length = iu_dcf->size()[0];
  dcf.data = iu_dcf->data();

  gpuNUFFT::Array<complex_type> coil_sens;
  coil_sens.dim = img_dims;
  coil_sens.dim.channels = iu_coil_sens->size()[2];
  coil_sens.data = iu_coil_sens->data();

  gpuNUFFT::GpuNUFFTOperatorFactory factory(true,true,true);
  nufft_op_.reset(factory.createGpuNUFFTOperator(trajectory, dcf, coil_sens, kernel_width_, sector_width_, osf_, img_dims));
}

template<typename InputType, typename OutputType>
void MriRadialOperator<InputType, OutputType>::executeForward(const InputType & src,
                                                          OutputType & dst)
{
  if (nufft_op_ == nullptr)
  {
    createNufftOperator();
  }

  gpuNUFFT::Dimensions img_dims;
  img_dims.width = img_dim_;
  img_dims.height = img_dim_;
  img_dims.depth = 0;

  gpuNUFFT::Array<complex_type> img;
  img.dim = img_dims;
  img.data = const_cast<complex_type*>(src.data());

  gpuNUFFT::Array<complex_type> kspace;
  kspace.dim.length = dst.size()[0];
  kspace.dim.channels = dst.size()[1];
  kspace.data = dst.data();

  nufft_op_->performForwardGpuNUFFT(img, kspace);
}

template<typename InputType, typename OutputType>
void MriRadialOperator<InputType, OutputType>::executeAdjoint(
    const OutputType & src, InputType & dst)
{
  if (nufft_op_ == nullptr)
  {
    createNufftOperator();
  }

  gpuNUFFT::Dimensions img_dims;
  img_dims.width = img_dim_;
  img_dims.height = img_dim_;
  img_dims.depth = 0;

  gpuNUFFT::Array<complex_type> img;
  img.dim = img_dims;
  img.data = dst.data();

  gpuNUFFT::Array<complex_type> kspace;
  kspace.dim.length = src.size()[0];
  kspace.dim.channels = src.size()[1];
  kspace.data = const_cast<complex_type*>(src.data());

  nufft_op_->performGpuNUFFTAdj(kspace, img);
}

template<typename InputType, typename OutputType>
void MriRadialOperator<InputType, OutputType>::adjointnessCheck()
{
  iu::Size<2> u_size({img_dim_, img_dim_});
  iu::Size<2> p_size({this->template getConstant<real_type, 2>(1)->size()[0], this->template getConstant<complex_type, 3>(2)->size()[2]});
  InputType u(u_size);
  OutputType p(p_size);
  iu::random::fillRandomFloatingNumbers(u);
  iu::random::fillRandomFloatingNumbers(p);
  OutputType Au(p.size());
  InputType Atp(u.size());

  executeForward(u, Au);
  executeAdjoint(p, Atp);

  std::cout << "Test adjointness of operator:" << *this << std::endl;
  TEST_ADJOINTNESS(u, Au, p, Atp);
}

// explicit template instantiations
template class MriRadialOperator<iu::LinearDeviceMemory<float2, 2>,
    iu::LinearDeviceMemory<float2, 2>> ;
//template class MriRadialOperator<iu::LinearDeviceMemory<double2, 2>,
//    iu::LinearDeviceMemory<double2, 2>> ;
