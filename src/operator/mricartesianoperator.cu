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

#include "mricartesianoperator.h"

#include <iu/iumath.h>
#include <iu/iuhelpermath.h>
#include <iu/iudefs.h>

#include "definitions.h"
#include "iurandom.h"

/** Preparations before FFT: Apply coil sensitivities and perform ifftshift. */
template<typename PixelType>
__global__ void prefft_kernel(
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::complex_type, 3>::KernelData coil_sens,
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::complex_type, 2>::KernelData img,
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::complex_type, 3>::KernelData dst)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int c = threadIdx.z + blockIdx.z * blockDim.z;

  const int height = coil_sens.size_[1];
  const int width = coil_sens.size_[0];

  if (x < width && y < height && c < coil_sens.size_[2])
  {
    // for ifftshift
    int x_mid = (width + 1.f) / 2.f;
    int y_mid = (height + 1.f) / 2.f;

    // ifftshift to get destination idx
    int x_dst = (x + x_mid) % width;
    int y_dst = (y + y_mid) % height;

    dst(x_dst, y_dst, c) = complex_multiply < PixelType > (img(x, y), coil_sens(x, y, c));
  }
}

/** Multiply a 3D complex array with a 2D real-valued array. */
template<typename PixelType>
__global__ void applyMask_kernel(
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::complex_type, 3>::KernelData Au,
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::real_type, 2>::KernelData mask)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int c = threadIdx.z + blockIdx.z * blockDim.z;

  if (x < Au.size_[0] && y < Au.size_[1] && c < Au.size_[2])
  {
    Au(x, y, c) = Au(x, y, c) * mask(x, y);
  }
}

/** Preparations before IFFT: Apply sampling mask and perform ifftshift. */
template<typename PixelType>
__global__ void preifft_kernel(
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::complex_type, 3>::KernelData f,
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::real_type, 2>::KernelData mask,
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::complex_type, 3>::KernelData dst)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int c = threadIdx.z + blockIdx.z * blockDim.z;

  const int height = f.size_[1];
  const int width = f.size_[0];

  if (x < width && y < height && c < f.size_[2])
  {
    // for ifftshift
    int x_mid = (width + 1.f) / 2.f;
    int y_mid = (height + 1.f) / 2.f;

    // ifftshift to get destination idx
    int x_dst = (x + x_mid) % width;
    int y_dst = (y + y_mid) % height;

    dst(x_dst, y_dst, c) = f(x, y, c) * mask(x, y);
  }
}

/** Combine image with coil sensitivity maps (complex conjugate) and store it in a destination image. */
template<typename PixelType>
__global__ void combineImg_kernel(
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::complex_type, 3>::KernelData img,
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::complex_type, 3>::KernelData coil_sens,
    struct iu::LinearDeviceMemory<typename iu::type_trait<PixelType>::complex_type, 2>::KernelData dst)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  const int coils = coil_sens.size_[2];
  const int height = coil_sens.size_[1];
  const int width = coil_sens.size_[0];

  if (x < width && y < height)
  {
    typename iu::type_trait<PixelType>::complex_type sum =
        iu::type_trait<PixelType>::make_complex(0);

    for (int c = 0; c < coils; c++)
    {
      sum += complex_multiply_conjugate < PixelType
          > (img(x, y, c), coil_sens(x, y, c));
    }

    dst(x, y) = sum;
  }
}

/** Crop image and store it in a new variable according to given indices in
  x (phase-encoding) and y (frequency-encoding) direction. */
template<typename InputType>
__global__ void cropFOV_kernel(
    struct InputType::KernelData img,
    struct InputType::KernelData cropped,
    unsigned int FE_start_idx, unsigned int PE_start_idx)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < cropped.size_[0] && y < cropped.size_[1])
  {
    cropped(x, y) = img(x + PE_start_idx, y + FE_start_idx);
  }
}

/** Copy image into a padded image according to given indices in
  x (phase-encoding) and y (frequency-encoding) direction. */
template<typename InputType>
__global__ void padFOV_kernel(
    struct InputType::KernelData img,
    struct InputType::KernelData padded,
    unsigned int FE_start_idx, unsigned int PE_start_idx)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < img.size_[0] && y < img.size_[1])
  {
    padded(x + PE_start_idx, y + FE_start_idx) = img(x, y);
  }
}

template<typename InputType, typename OutputType>
MriCartesianOperator<InputType, OutputType>::MriCartesianOperator() :
    OperatorBase<InputType, OutputType>(2, "MriCartesianOperator")
{
}

template<typename InputType, typename OutputType>
MriCartesianOperator<InputType, OutputType>::~MriCartesianOperator()
{
}

template<typename InputType, typename OutputType>
void MriCartesianOperator<InputType, OutputType>::sizeCheck(
    const InputType & src, const OutputType & dst)
{
  IU_SIZE_CHECK(dst.size(), (this->template getConstant<complex_type, 3>(0))->size());
  IU_SIZE_CHECK(src.size(), (this->template getConstant<real_type, 2>(1))->size());
  IU_SIZE_CHECK(src.size(), iu::Size<2>( { dst.size()[0], dst.size()[1] }));
}

template<typename InputType, typename OutputType>
iu::Size<InputType::ndim> MriCartesianOperator<InputType, OutputType>::getInputSize(const OutputType& output)
{
  iu::Size<2> input_size({output.size()[0], output.size()[1]});
  return input_size;
}

template<typename InputType, typename OutputType>
iu::Size<OutputType::ndim> MriCartesianOperator<InputType, OutputType>::getOutputSize(const InputType& input)
{
  iu::Size<OutputType::ndim> size = this->template getConstant<complex_type, 3>(0)->size();
  return size;
}

template<typename InputType, typename OutputType>
void MriCartesianOperator<InputType, OutputType>::executeForward(const InputType & src,
                                                          OutputType & dst)
{
  // extract constants
  auto coil_sens = this->template getConstant<complex_type, 3>(0);
  auto mask = this->template getConstant<real_type, 2>(1);

  // temporary variable
  iu::LinearDeviceMemory<complex_type, 3> kspace(dst.size());

  // output = A(u)
  dim3 dimBlock(COMMON_BLOCK_SIZE_3D_X, COMMON_BLOCK_SIZE_3D_Y,
  COMMON_BLOCK_SIZE_3D_Z);
  dim3 dimGrid(iu::divUp(dst.size()[0], dimBlock.x),
               iu::divUp(dst.size()[1], dimBlock.y),
               iu::divUp(dst.size()[2], dimBlock.z));

  // perform multiplication with coil sensitivity profiles and ifftshift2
  prefft_kernel<pixel_type> <<<dimGrid, dimBlock>>>(*coil_sens, src, dst);
  IU_CUDA_CHECK;

  // perform fft2 and scale with 1/sqrt(elements)
  iu::math::fft::fft2(dst, kspace, true);

  // perform fftshift2
  iu::math::fft::fftshift2(kspace, dst);

  // apply sampling mask
  applyMask_kernel<pixel_type> <<<dimGrid, dimBlock>>>(dst, *mask);
  IU_CUDA_CHECK;
}

template<typename InputType, typename OutputType>
void MriCartesianOperator<InputType, OutputType>::executeAdjoint(
    const OutputType & src, InputType & dst)
{
  // extract constants
  auto coil_sens = this->template getConstant<complex_type, 3>(0);
  auto mask = this->template getConstant<real_type, 2>(1);

  // temporary variables
  iu::LinearDeviceMemory<complex_type, 3> kspace1(src.size());
  iu::LinearDeviceMemory<complex_type, 3> kspace2(src.size());

  // output = A^H(f)
  dim3 dimBlock(COMMON_BLOCK_SIZE_3D_X, COMMON_BLOCK_SIZE_3D_Y,
  COMMON_BLOCK_SIZE_3D_Z);
  dim3 dimGrid(iu::divUp(src.size()[0], dimBlock.x),
               iu::divUp(src.size()[1], dimBlock.y),
               iu::divUp(src.size()[2], dimBlock.z));

  // perform ifftshift and apply sampling mask
  preifft_kernel<pixel_type> <<<dimGrid, dimBlock>>>(src, *mask, kspace1);
  IU_CUDA_CHECK;

  // perform ifft2 and scale with 1/sqrt(elements)
  iu::math::fft::ifft2(kspace1, kspace2, true);

  // perform fftshift2
  iu::math::fft::fftshift2(kspace2, kspace1);

  // multiply kspace result with coil sensitivity profiles and add up the
  // single channels
  dimGrid = dim3(iu::divUp(src.size()[0], dimBlock.x),
                 iu::divUp(src.size()[1], dimBlock.y), 1);
  combineImg_kernel<pixel_type> <<<dimGrid, dimBlock>>>(kspace1, *coil_sens,
                                                            dst);
  IU_CUDA_CHECK;
}

template<typename InputType, typename OutputType>
void MriCartesianOperator<InputType, OutputType>::adjointnessCheck()
{
  InputType u(this->template getConstant<real_type, 2>(1)->size());
  OutputType p(this->template getConstant<complex_type, 3>(0)->size());
  iu::random::fillRandomFloatingNumbers(u);
  iu::random::fillRandomFloatingNumbers(p);
  OutputType Au(p.size());
  InputType Atp(u.size());

  executeForward(u, Au);
  executeAdjoint(p, Atp);

  std::cout << "Test adjointness of operator:" << *this << std::endl;
  TEST_ADJOINTNESS(u, Au, p, Atp);
}

////////////////////////////////////////////////////////////////////////////////
template<typename InputType, typename OutputType>
MriCartesianRemoveROOSOperator<InputType, OutputType>::MriCartesianRemoveROOSOperator() :
    OperatorBase<InputType, OutputType>(2, "MriCartesianRemoveROOSOperator")
{
}

template<typename InputType, typename OutputType>
MriCartesianRemoveROOSOperator<InputType, OutputType>::~MriCartesianRemoveROOSOperator()
{
}

template<typename InputType, typename OutputType>
iu::Size<InputType::ndim> MriCartesianRemoveROOSOperator<InputType, OutputType>::getInputSize(const OutputType& output)
{
  iu::Size<2> input_size({output.size()[0], output.size()[1]/2});
  return input_size;
}

template<typename InputType, typename OutputType>
iu::Size<OutputType::ndim> MriCartesianRemoveROOSOperator<InputType, OutputType>::getOutputSize(const InputType& input)
{
  iu::Size<OutputType::ndim> size = this->template getConstant<complex_type, 3>(0)->size();
  return size;
}

template<typename InputType, typename OutputType>
void MriCartesianRemoveROOSOperator<InputType, OutputType>::sizeCheck(
    const InputType & src, const OutputType & dst)
{
  iu::Size<2> size = src.size();
  size[1] *= 2;
  IU_SIZE_CHECK(dst.size(), (this->template getConstant<complex_type, 3>(0))->size());
  IU_SIZE_CHECK(size, (this->template getConstant<real_type, 2>(1))->size());
}

template<typename InputType, typename OutputType>
void MriCartesianRemoveROOSOperator<InputType, OutputType>::executeForward(const InputType & src,
                                                          OutputType & dst)
{
  // extract constants
  auto coil_sens = this->template getConstant<complex_type, 3>(0);
  auto mask = this->template getConstant<real_type, 2>(1);

  // temporary variable
  iu::LinearDeviceMemory<complex_type, 3> kspace(dst.size());
  iu::Size<2> size = src.size();
  size[1] *= 2;
  InputType src_pad(size);

  // init padded image with zeros
  iu::math::fill(src_pad, iu::type_trait<typename InputType::pixel_type>::make(0));

  // Pad in read-out direction
  dim3 dimBlockSmall(COMMON_BLOCK_SIZE_3D_X, COMMON_BLOCK_SIZE_3D_Y);
  dim3 dimGridSmall(iu::divUp(src.size()[0], dimBlockSmall.x),
                    iu::divUp(src.size()[1], dimBlockSmall.y));
  unsigned int FE_idx = size[1] * 0.25f + 1;
  padFOV_kernel<InputType><<<dimGridSmall, dimBlockSmall>>>(src, src_pad, FE_idx, 0);
  IU_CUDA_CHECK;

  // output = A(u)
  dim3 dimBlock(COMMON_BLOCK_SIZE_3D_X, COMMON_BLOCK_SIZE_3D_Y,
  COMMON_BLOCK_SIZE_3D_Z);
  dim3 dimGrid(iu::divUp(dst.size()[0], dimBlock.x),
               iu::divUp(dst.size()[1], dimBlock.y),
               iu::divUp(dst.size()[2], dimBlock.z));

  // perform multiplication with coil sensitivity profiles and ifftshift2
  prefft_kernel<pixel_type> <<<dimGrid, dimBlock>>>(*coil_sens, src_pad, dst);
  IU_CUDA_CHECK;

  // perform fft2 and scale with 1/sqrt(elements)
  iu::math::fft::fft2(dst, kspace, true);

  // perform fftshift2
  iu::math::fft::fftshift2(kspace, dst);

  // apply sampling mask
  applyMask_kernel<pixel_type> <<<dimGrid, dimBlock>>>(dst, *mask);
  IU_CUDA_CHECK;
}

template<typename InputType, typename OutputType>
void MriCartesianRemoveROOSOperator<InputType, OutputType>::executeAdjoint(
    const OutputType & src, InputType & dst)
{
  // extract constants
  auto coil_sens = this->template getConstant<complex_type, 3>(0);
  auto mask = this->template getConstant<real_type, 2>(1);

  // temporary variables
  iu::LinearDeviceMemory<complex_type, 3> kspace1(src.size());
  iu::LinearDeviceMemory<complex_type, 3> kspace2(src.size());
  iu::Size<2> size = dst.size();
  size[1] *= 2;
  InputType dst_pad(size);
  const unsigned int FE_idx = size[1] * 0.25f + 1;

  // output = A^H(f)
  dim3 dimBlock(COMMON_BLOCK_SIZE_3D_X, COMMON_BLOCK_SIZE_3D_Y,
  COMMON_BLOCK_SIZE_3D_Z);
  dim3 dimGrid(iu::divUp(src.size()[0], dimBlock.x),
               iu::divUp(src.size()[1], dimBlock.y),
               iu::divUp(src.size()[2], dimBlock.z));

  // perform ifftshift and apply sampling mask
  preifft_kernel<pixel_type> <<<dimGrid, dimBlock>>>(src, *mask, kspace1);
  IU_CUDA_CHECK;

  // perform ifft2 and scale with 1/sqrt(elements)
  iu::math::fft::ifft2(kspace1, kspace2, true);

  // perform fftshift2
  iu::math::fft::fftshift2(kspace2, kspace1);

  // multiply kspace result with coil sensitivity profiles and add up the
  // single channels
  dimGrid = dim3(iu::divUp(src.size()[0], dimBlock.x),
                 iu::divUp(src.size()[1], dimBlock.y), 1);
  combineImg_kernel<pixel_type> <<<dimGrid, dimBlock>>>(kspace1, *coil_sens,
                                                            dst_pad);
  IU_CUDA_CHECK;

  // Remove read-out oversampling
  dimBlock = dim3(COMMON_BLOCK_SIZE_3D_X, COMMON_BLOCK_SIZE_3D_Y, 1);
  dimGrid = dim3(iu::divUp(dst.size()[0], dimBlock.x),
                 iu::divUp(dst.size()[1], dimBlock.y), 1);
  cropFOV_kernel<InputType><<<dimGrid, dimBlock>>>(dst_pad, dst, FE_idx, 0);
  IU_CUDA_CHECK;
}

template<typename InputType, typename OutputType>
void MriCartesianRemoveROOSOperator<InputType, OutputType>::adjointnessCheck()
{
  std::cout << "Test adjointness of operator: " << *this << std::endl;

  InputType u(getInputSize(*this->template getConstant<complex_type, 3>(0)));
  OutputType p(this->template getConstant<complex_type, 3>(0)->size());
  iu::random::fillRandomFloatingNumbers(u);
  iu::random::fillRandomFloatingNumbers(p);
  OutputType Au(p.size());
  InputType Atp(u.size());

  executeForward(u, Au);
  executeAdjoint(p, Atp);

  TEST_ADJOINTNESS(u, Au, p, Atp);
}


// explicit template instantiations
template class MriCartesianOperator<iu::LinearDeviceMemory<float2, 2>,
    iu::LinearDeviceMemory<float2, 3>> ;
template class MriCartesianOperator<iu::LinearDeviceMemory<double2, 2>,
    iu::LinearDeviceMemory<double2, 3>> ;

template class MriCartesianRemoveROOSOperator<iu::LinearDeviceMemory<float2, 2>,
    iu::LinearDeviceMemory<float2, 3>> ;
template class MriCartesianRemoveROOSOperator<iu::LinearDeviceMemory<double2, 2>,
    iu::LinearDeviceMemory<double2, 3>> ;

