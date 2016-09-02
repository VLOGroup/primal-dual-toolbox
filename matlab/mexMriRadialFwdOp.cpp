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

#include <mex.h>

// system includes
#include <iostream>
using namespace std;
#include <math.h>
#include <stdlib.h>
#include <list>
#include <vector>
#include <cuda_runtime.h>
#include <sstream>
// #ifdef _WIN32
// #  include <windows.h>
// #endif

#include <cuda.h>
#include "operator/mriradialoperator.h"
#include "optimizer/tgvmrioptimizer.h"

#include <iu/iucore.h>
#include <iu/iumath/typetraits.h>

#include <iu/iumatlab.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  typedef float real_type;
  typedef iu::type_trait<real_type>::complex_type complex_type;
  typedef iu::LinearDeviceMemory<complex_type, 2> InputType;
  typedef iu::LinearDeviceMemory<complex_type, 2> OutputType;

  // This function does not check a single thing, it just assumes two images
  // as parameters, does something with them on the GPU and returns the result
  char err_msg[128];

  // Checking number of arguments
  if (nrhs != 5)
    mexErrMsgIdAndTxt(
        "MATLAB:gpuMriRadialFwd:invalidNumInputs",
        "Five inputs required (trajectory, dcf, sensitivities, img, nufft_params)");
  if (nlhs > 1)
    mexErrMsgIdAndTxt("MATLAB:gpuMriRadialFwd:maxlhs",
                      "Too many output arguments.");

  // Check datatypes
  if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) )
  {
      mexErrMsgIdAndTxt("MATLAB:gpuMriRadialFwd:wrongDataType",
                        "Input 1 (trajectory) must be double!");
  }
  if( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) )
  {
      mexErrMsgIdAndTxt("MATLAB:gpuMriRadialFwd:wrongDataType",
                        "Input 2 (dcf) must be double!");
  }
  if( !mxIsDouble(prhs[2]) || !mxIsComplex(prhs[2]) )
  {
      mexErrMsgIdAndTxt("MATLAB:gpuMriRadialFwd:wrongDataType",
                        "Input 3 (sensitivities) must be double complex!");
  }
  if( !mxIsDouble(prhs[3]) || !mxIsComplex(prhs[3]) )
  {
      mexErrMsgIdAndTxt("MATLAB:gpuMriRadialFwd:wrongDataType",
                        "Input 4 (img) must be double complex!");
  }

  // NUFFT parameters
  const mxArray *mat_nufft_params = prhs[4];

  if (!mxIsStruct(mat_nufft_params))
    mexErrMsgTxt("Expect struct with nufft params!");

  OpConfigDict config;
  config["kernel_width"] = std::to_string(mxGetScalar(mxGetField(mat_nufft_params, 0, "kernel_width")));
  config["sector_width"] = std::to_string(mxGetScalar(mxGetField(mat_nufft_params, 0, "sector_width")));
  config["osf"] = std::to_string(mxGetScalar(mxGetField(mat_nufft_params, 0, "osf")));
  config["img_dim"] = std::to_string(mxGetScalar(mxGetField(mat_nufft_params, 0, "img_dim")));

  // Convert Matlab to C
  iu::LinearHostMemory<real_type, 2> h_trajectory(*prhs[0],false);
  iu::LinearHostMemory<real_type, 2> h_dcf(*prhs[1],false);
  iu::LinearHostMemory<complex_type, 3> h_sensitivities(*prhs[2],false);
  iu::LinearHostMemory<complex_type, 2> h_img(*prhs[3],false);

  // do something magical

  // Init operator
  std::shared_ptr < MriRadialOperator<InputType, OutputType>
      > op(new MriRadialOperator<InputType, OutputType>(config));

  // Add coil sensitivities and mask to operator
  op->addConstant(h_trajectory);
  op->addConstant(h_dcf);
  op->addConstant(h_sensitivities);

  // Copy host to device
  iu::LinearDeviceMemory<complex_type, 2> d_img(h_img.size());
  iu::copy(&h_img, &d_img);

   // Get result
  iu::Size<2> output_size = op->getOutputSize(d_img);

  iu::LinearDeviceMemory<complex_type, 2> d_output(output_size);
  iu::LinearHostMemory<complex_type, 2> h_output(output_size);

  op->forward(d_img, d_output);

  iu::copy(&d_output, &h_output);

  // Convert to MATLAB Output
  iu::matlab::convertCToMatlab(h_output, &plhs[0], false);
}
