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

#include <iu/iumath/typetraits.h>

#include "operatorbase.h"
#include "primaldualtoolbox_api.h"

#include <gpuNUFFT_operator_factory.hpp>
#include <memory>

/** \brief MRI 2D radial sampling operator.
 *
 * This class implements a 2D MRI radial sampling operator.
 * @param constants[0]: trajectory (2D array, real-valued)
 * @param constants[1]: density compensation function (2D array, real-valued)
 * @param constants[2]: coil sensitivity maps (3D array, complex-valued)
 */
template<typename InputType, typename OutputType>
class PRIMALDUALTOOLBOX_DLLAPI MriRadialOperator : public OperatorBase<InputType, OutputType> {
public:
  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;
  typedef typename iu::type_trait<pixel_type>::complex_type complex_type;

  /** Special constructor requiring an OpConfigDict including the fields 'osf' (oversampling factor),
   * 'sector_width', 'kernel_width', and 'img_dim'. */
  MriRadialOperator(const OpConfigDict &config);

  /** Destructor. */
  virtual ~MriRadialOperator();

  /** Perform forward operation: Based on gpuNUFFT
   * @param[in] src complex-valued data in image domain
   * @param[out] dst complex-valued data in Fourier domain
   */
  void executeForward(const InputType & src, OutputType & dst);

  /** Perform adjoint operation: Based on gpuNUFFT
   * @param[in] src complex-valued data in Fourier domain
   * @param[out] dst complex-valued data in image domain
   */
  void executeAdjoint(const OutputType & src, InputType & dst);

  /** Get size of the input data */
  iu::Size<InputType::ndim> getInputSize(const OutputType & output);

  /** Get size of the output data */
  iu::Size<OutputType::ndim> getOutputSize(const InputType & input);

  /** Check if the forward/adjoint operators are implemented correctly. */
  void adjointnessCheck();

private:
  /** Check if source and destination dimensions match. */
  void sizeCheck(const InputType & src, const OutputType & dst);

  /** Initialize gpuNUFFT operator. */
  void createNufftOperator();

  /** Over-samplign factor. */
  real_type osf_;

  /** Kernel width. */
  int kernel_width_;

  /** Sector width. */
  int sector_width_;

  /** Image dimension. */
  int img_dim_;

  /** Interface to gpuNUFFT. */
  std::unique_ptr<gpuNUFFT::GpuNUFFTOperator> nufft_op_;
};

/** \brief MRI 2D radial sampling operator for singlecoil data (no coil sensitivities).
 *
 * This class implements a 2D MRI radial sampling operator.
 * @param constants[0]: trajectory (2D array, real-valued)
 * @param constants[1]: density compensation function (2D array, real-valued)
 */
template<typename InputType, typename OutputType>
class PRIMALDUALTOOLBOX_DLLAPI MriRadialSinglecoilOperator : public OperatorBase<InputType, OutputType> {
public:
  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;
  typedef typename iu::type_trait<pixel_type>::complex_type complex_type;

  /** Special constructor requiring an OpConfigDict including the fields 'osf' (oversampling factor),
   * 'sector_width', 'kernel_width', and 'img_dim'. */
  MriRadialSinglecoilOperator(const OpConfigDict &config);

  /** Destructor. */
  virtual ~MriRadialSinglecoilOperator();

  /** Perform forward operation: Based on gpuNUFFT
   * @param[in] src complex-valued data in image domain
   * @param[out] dst complex-valued data in Fourier domain
   */
  void executeForward(const InputType & src, OutputType & dst);

  /** Perform adjoint operation: Based on gpuNUFFT
   * @param[in] src complex-valued data in Fourier domain
   * @param[out] dst complex-valued data in image domain
   */
  void executeAdjoint(const OutputType & src, InputType & dst);

  /** Get size of the input data */
  iu::Size<InputType::ndim> getInputSize(const OutputType & output);

  /** Get size of the output data */
  iu::Size<OutputType::ndim> getOutputSize(const InputType & input);

  /** Check if the forward/adjoint operators are implemented correctly. */
  void adjointnessCheck();

private:
  /** Check if source and destination dimensions match. */
  void sizeCheck(const InputType & src, const OutputType & dst);

  /** Initialize gpuNUFFT operator. */
  void createNufftOperator();

  /** Over-samplign factor. */
  real_type osf_;

  /** Kernel width. */
  int kernel_width_;

  /** Sector width. */
  int sector_width_;

  /** Image dimension. */
  int img_dim_;

  /** Interface to gpuNUFFT. */
  std::unique_ptr<gpuNUFFT::GpuNUFFTOperator> nufft_op_;
};
