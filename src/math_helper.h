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

#ifdef NVCC

#include <cuda.h>
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
inline __device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull, assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  }
  while (assumed != old);
  return __longlong_as_double(old);
}
#endif

#endif
