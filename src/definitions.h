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

#include <iu/iumath.h>
#include <type_traits>

#define COMMON_BLOCK_SIZE_2D_X 32
#define COMMON_BLOCK_SIZE_2D_Y 16

#define COMMON_BLOCK_SIZE_3D_X 16
#define COMMON_BLOCK_SIZE_3D_Y 8
#define COMMON_BLOCK_SIZE_3D_Z 4

#define FILTER_SIZE_MAX 11

namespace utils{
/** Test adjointness for real-valued images (not float2/double2). */
template<typename InputType, typename OutputType, typename ResultType = void>
typename std::enable_if<!
    std::is_same<typename InputType::pixel_type, float2>::value
        && !std::is_same<typename InputType::pixel_type, double2>::value, ResultType>::type testAdjointness(InputType& u, OutputType& Au, OutputType& p, InputType& Atp)
{
  typename InputType::pixel_type lhs;
  typename InputType::pixel_type rhs;
  iu::math::dotProduct(Au, p, lhs);
  iu::math::dotProduct(u, Atp, rhs);

  std::cout << "<Au,p>=" << lhs << std::endl;
  std::cout << "<u,Atp>=" << rhs << std::endl;
  std::cout << "diff= " << abs(lhs - rhs) << std::endl;
  if(abs(lhs - rhs) < 1e-12)
  {
    std::cout << "TEST PASSED" << std::endl;
  }
  else
  {
    std::cout << "TEST FAILED" << std::endl;
  }
  std::cout << std::endl;
}

/** Test adjointness for complex-valued images (float2/double2). */
template<typename InputType, typename OutputType, typename ResultType = void>
typename std::enable_if<
    std::is_same<typename InputType::pixel_type, float2>::value
        || std::is_same<typename InputType::pixel_type, double2>::value, ResultType>::type testAdjointness(InputType& u, OutputType& Au, OutputType& p, InputType& Atp)
{
  typename InputType::pixel_type lhs;
  typename InputType::pixel_type rhs;
  iu::math::complex::dotProduct(Au, p, lhs);
  iu::math::complex::dotProduct(u, Atp, rhs);

  std::cout << "<Au,p>=" << lhs << std::endl;
  std::cout << "<u,Atp>=" << rhs << std::endl;
  std::cout << "diff: x= " << abs(lhs.x - rhs.x) << " y=" << abs(lhs.y - rhs.y) << std::endl;

  if(abs(lhs.x - rhs.x) < 1e-12 && abs(lhs.y - rhs.y) < 1e-12)
  {
    std::cout << "TEST PASSED" << std::endl;
  }
  else
  {
    std::cout << "TEST FAILED" << std::endl;
  }
  std::cout << std::endl;
}
}

#define TEST_ADJOINTNESS(u, Au, p, Atp) utils::testAdjointness(u, Au, p, Atp)
