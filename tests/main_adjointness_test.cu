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
#include "operator/mricartesianoperator.h"
#include "operator/mrisamplingoperator.h"

#ifdef WITH_GPUNUFFT
  #include "operator/mriradialoperator.h"
#endif

#include "iurandom.h"

void testMriCartesianOperator()
{
  // Create constants needed for operator
  iu::LinearHostMemory<double2, 3> h_coilsens({10,10,5});
  iu::LinearHostMemory<double, 2> h_mask({10,10});
  iu::random::fillRandomFloatingNumbers(h_coilsens);
  iu::random::fillRandomIntNumbers(h_mask, 0, 1);

  // Create operator, add constants and check for adjointness
  MriCartesianOperator<iu::LinearDeviceMemory<double2, 2>, iu::LinearDeviceMemory<double2, 3>> op;
  op.addConstant(h_coilsens);
  op.addConstant(h_mask);
  op.adjointnessCheck();
}

void testMriCartesianRemoveROOSOperator()
{
  // Create constants needed for operator
  iu::LinearHostMemory<double2, 3> h_coilsens({10,10,5});
  iu::LinearHostMemory<double, 2> h_mask({10,10});
  iu::random::fillRandomFloatingNumbers(h_coilsens);
  iu::random::fillRandomIntNumbers(h_mask, 0, 1);

  // Create operator, add constants and check for adjointness
  MriCartesianRemoveROOSOperator<iu::LinearDeviceMemory<double2, 2>, iu::LinearDeviceMemory<double2, 3>> op;
  op.addConstant(h_coilsens);
  op.addConstant(h_mask);
  op.adjointnessCheck();
}

void testMriSamplingOperator()
{
  // Create operator & check adjointness
  iu::LinearHostMemory<double, 2> h_mask({10,10});
  iu::random::fillRandomIntNumbers(h_mask, 0, 1);

  MriSamplingOperator<iu::LinearDeviceMemory<double2, 2>, iu::LinearDeviceMemory<double2, 2>> op;
  op.addConstant(h_mask);
  op.adjointnessCheck();
}

void testMriRadialOperator()
{
#ifdef WITH_GPUNUFFT
  // Setup config parameters
  OpConfigDict config;
  config["img_dim"] = "256";
  config["osf"] = "2";
  config["kernel_width"] = "3";
  config["sector_width"] = "5";

  // Constants
  iu::LinearHostMemory<float2, 3> h_coilsens({256,256, 5});
  iu::LinearHostMemory<float, 2> h_trajectory({256*64, 2});
  iu::LinearHostMemory<float, 2> h_dcf({256*64, 1});

  iu::random::fillRandomFloatingNumbers(h_trajectory);
  iu::random::fillRandomFloatingNumbers(h_dcf);
  iu::random::fillRandomFloatingNumbers(h_coilsens);

  // Create operator & check adjointness
  MriRadialOperator<iu::LinearDeviceMemory<float2, 2>, iu::LinearDeviceMemory<float2, 2>> op(config);
  op.addConstant(h_trajectory);
  op.addConstant(h_dcf);
  op.addConstant(h_coilsens);
  op.adjointnessCheck();
#endif
}

int main(int argc, char *argv[])
{
  std::cout << "*****************************" << std::endl;
  std::cout << "Test adjointness of operators"<< std::endl;
  std::cout << "*****************************" << std::endl;

  testMriCartesianOperator();
  testMriCartesianRemoveROOSOperator();
  testMriSamplingOperator();
  testMriRadialOperator();
  return 0;
}
