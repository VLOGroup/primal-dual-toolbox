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

#include <iostream>

#include "iurandom.h"
#include <gpuNUFFT_operator_factory.hpp>

int main(int argc, char *argv[])
{
  std::cout << "*****************************" << std::endl;
  std::cout << "Test gpuNUFFT"<< std::endl;
  std::cout << "*****************************" << std::endl;

  const unsigned int nFE = 512;
  const unsigned int nSpokes = 64;
  const unsigned int osf = 2;
  const unsigned int sector_width = 3;
  const unsigned int kernel_width = 3;
  const unsigned int nCh = 32;

  iu::LinearDeviceMemory<float, 2> iu_kSpaceTraj({nFE * nSpokes, 2});
  iu::random::fillRandomFloatingNumbers(iu_kSpaceTraj);

  iu::LinearDeviceMemory<float, 1> iu_dcf({nFE * nSpokes});
  iu::random::fillRandomFloatingNumbers(iu_dcf);

  iu::LinearDeviceMemory<float2, 3> iu_sens({nFE/osf, nFE/osf, nCh});
  iu::random::fillRandomFloatingNumbers(iu_sens);

  iu::LinearDeviceMemory<float2, 2> iu_img({nFE/osf, nFE/osf});
  iu::random::fillRandomFloatingNumbers(iu_img);

  iu::LinearDeviceMemory<float2, 3> iu_kspace({nFE, nSpokes, nCh});
  iu::random::fillRandomFloatingNumbers(iu_kspace);

  gpuNUFFT::Array<float> kSpaceTraj;
  kSpaceTraj.dim.length = nFE * nSpokes;
  kSpaceTraj.data = iu_kSpaceTraj.data();

  gpuNUFFT::Array<float> dcf;
  dcf.dim.length = nFE * nSpokes;
  dcf.data = iu_dcf.data();

  gpuNUFFT::Dimensions img_dims;
  img_dims.width = nFE/osf;
  img_dims.height = nFE/osf;
  img_dims.depth = 0;

  gpuNUFFT::Array<float2> sens;
  sens.dim = img_dims;
  sens.dim.channels = nCh;
  sens.data = iu_sens.data();

  gpuNUFFT::Array<float2> img;
  img.dim = img_dims;
  img.data = iu_img.data();

  gpuNUFFT::Array<float2> kspace;
  kspace.dim.length = nFE * nSpokes;
  kspace.dim.channels = nCh;
  kspace.data = iu_kspace.data();

  gpuNUFFT::GpuNUFFTOperatorFactory factory(false,true,true);
  gpuNUFFT::GpuNUFFTOperator * nufft_op = factory.createGpuNUFFTOperator(
      kSpaceTraj, dcf, sens, kernel_width, sector_width, osf, img_dims);

  nufft_op->performForwardGpuNUFFT(img, kspace);
  nufft_op->setSens(sens);
  nufft_op->performForwardGpuNUFFT(img, kspace);

  delete nufft_op;
  return 0;
}
