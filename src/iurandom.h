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

#include <iostream>
#include <type_traits>
#include <random>

#include <iu/iumath.h>
#include <iu/iucore.h>
#include <iu/iumath/typetraits.h>

namespace iu {
namespace random {

/** Global seeding parameter. The seed is changed every time a random number generating function is called.*/
static unsigned int SEED = 42;

/** Fill LinearHostMemory for arbitrary PixelType except float2 and double2 with random integer numbers.
 * @param src LinearHostMemory array that should be filled with random int numbers.
 * @param lb Lower bound
 * @param ub Upper bound
 */
template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<
    !std::is_same<PixelType, float2>::value
        && !std::is_same<PixelType, double2>::value, ResultType>::type fillRandomIntNumbers(
    iu::LinearHostMemory<PixelType, Ndim>& src, int lb = 0, int ub = 255)
{
  std::default_random_engine generator(SEED);
  std::uniform_int_distribution<> dis(lb, ub);

  for (int i = 0; i < src.size().numel(); i++)
  {
    src.data()[i] = dis(generator);
  }

  SEED += src.size().numel();
}

/** Fill LinearHostMemory for float2 or double2 with random integer numbers.
 * @param src LinearHostMemory array that should be filled with random int numbers.
 * @param lb Lower bound
 * @param ub Upper bound
 */
template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<
    std::is_same<PixelType, float2>::value
        || std::is_same<PixelType, double2>::value, ResultType>::type fillRandomIntNumbers(
    iu::LinearHostMemory<PixelType, Ndim>& src, int lb = 0, int ub = 255)
{
  std::default_random_engine generator(SEED);
  std::uniform_int_distribution<> dis(lb, ub);

  for (int i = 0; i < src.size().numel(); i++)
  {
    src.data()[i] = iu::type_trait<PixelType>::make_complex(dis(generator),
                                                            dis(generator));
  }

  SEED += src.size().numel();
}

/** Fill LinearDeviceMemory for arbitrary PixelType with random integer numbers.
 * @param src LinearDeviceMemory array that should be filled with random int numbers.
 * @param lb Lower bound
 * @param ub Upper bound
 */
template<typename PixelType, unsigned int Ndim>
void fillRandomIntNumbers(iu::LinearDeviceMemory<PixelType, Ndim>& src_d,
                          int lb = 0, int ub = 255)
{
  iu::LinearHostMemory < PixelType, Ndim > src_h(src_d.size());
  fillRandomIntNumbers(src_h, lb, ub);
  iu::copy(&src_h, &src_d);
}

/** Fill LinearHostMemory for arbitrary PixelType except float2 and double2 with random floating point numbers.
 * @param src LinearHostMemory array that should be filled with random floating point numbers.
 * @param lb Lower bound
 * @param ub Upper bound
 */
template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<
    !std::is_same<PixelType, float2>::value
        && !std::is_same<PixelType, double2>::value, ResultType>::type fillRandomFloatingNumbers(
    iu::LinearHostMemory<PixelType, Ndim>& src, int lb = 0, int ub = 1)
{
  std::default_random_engine generator(SEED);
  std::uniform_real_distribution<> dis(lb, ub);

  for (int i = 0; i < src.size().numel(); i++)
  {
    src.data()[i] = dis(generator);
  }

  SEED += src.size().numel();
}

/** Fill LinearHostMemory for float2 or double2 with random floating point numbers.
 * @param src LinearHostMemory array that should be filled with random floating Point numbers.
 * @param lb Lower bound
 * @param ub Upper bound
 */
template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<
    std::is_same<PixelType, float2>::value
        || std::is_same<PixelType, double2>::value, ResultType>::type fillRandomFloatingNumbers(
    iu::LinearHostMemory<PixelType, Ndim>& src, int lb = 0, int ub = 1)
{
  std::default_random_engine generator(SEED);
  std::uniform_real_distribution<> dis(lb, ub);

  for (int i = 0; i < src.size().numel(); i++)
  {
    src.data()[i] = iu::type_trait<PixelType>::make_complex(dis(generator),
                                                            dis(generator));
  }

  SEED += src.size().numel();
}

/** Fill LinearDeviceMemory for arbitrary PixelType with random floating point numbers.
 * @param src LinearDeviceMemory array that should be filled with random floating point numbers.
 * @param lb Lower bound
 * @param ub Upper bound
 */
template<typename PixelType, unsigned int Ndim>
void fillRandomFloatingNumbers(iu::LinearDeviceMemory<PixelType, Ndim>& src_d,
                          int lb = 0, int ub = 1)
{
  iu::LinearHostMemory < PixelType, Ndim > src_h(src_d.size());
  fillRandomFloatingNumbers(src_h, lb, ub);
  iu::copy(&src_h, &src_d);
}

}
}

