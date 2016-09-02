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

/** \brief MRI 2D Cartesian sampling operator with application of coil sensitivity maps.
 *
 * This class implements an MRI (Cartesian) sampling operator with application of coil sensitivity maps.
 * @param constants[0]: coil sensitivity maps (3D, complex-valued), third dimensions is the number of channels.
 * @param constants[1]: sampling mask (2D, real-valued)
 */
template<typename InputType, typename OutputType>
class PRIMALDUALTOOLBOX_DLLAPI MriCartesianOperator : public OperatorBase<InputType, OutputType> {
public:
  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;
  typedef typename iu::type_trait<pixel_type>::complex_type complex_type;

  /** Constructor */
	MriCartesianOperator();

	/** Destructor */
	virtual ~MriCartesianOperator();

  /** Perform forward operation: application of coil sensitivity maps, centered FFT followed by
   * application of the sampling mask.
   * @param src[in] complex-valued data in image domain
   * @param dst[out] complex-valued data in Fourier domain
   */
  void executeForward(const InputType & src, OutputType & dst);

  /** Perform forward operation: application of the sampling mask, centered IFFT followed by a
   * sensitivity-weighted combination of the coils.
   * @param src[in] complex-valued data in Fourier domain
   * @param dst[out] complex-valued data in image domain
   */
  void executeAdjoint(const OutputType & src, InputType & dst);

  /** Get size of the input data */
  iu::Size<InputType::ndim> getInputSize(const OutputType & output);

  /** Get size of the output data */
  iu::Size<OutputType::ndim> getOutputSize(const InputType & dst);

  /** Check if the forward/adjoint operators are implemented correctly. */
  void adjointnessCheck();

private:
  /** Check if source and destination dimensions match. */
  void sizeCheck(const InputType & src, const OutputType & dst);
};

/** \brief MRI 2D Cartesian sampling operator with application of coil sensitivity maps and additional removal
 * of read-out oversampling (ROOS).
 *
 * This class implements an MRI (Cartesian) sampling operator with application of coil sensitivity maps.
 * @param constants[0]: coil sensitivity maps (3D, complex-valued), third dimensions is the number of channels.
 * @param constants[1]: sampling mask (2D, real-valued)
 */
template<typename InputType, typename OutputType>
class PRIMALDUALTOOLBOX_DLLAPI MriCartesianRemoveROOSOperator : public OperatorBase<InputType, OutputType> {
public:
  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;
  typedef typename iu::type_trait<pixel_type>::complex_type complex_type;

  /** Constructor */
  MriCartesianRemoveROOSOperator();

  /** Destructor */
  virtual ~MriCartesianRemoveROOSOperator();

  /** Perform forward operation: Pad in RO-direction, application of coil sensitivity maps, centered FFT
   * followed by application of the sampling mask.
   * @param[in] src complex-valued data in image domain
   * @param[out] dst complex-valued data in Fourier domain
   */
  void executeForward(const InputType & src, OutputType & dst);

  /** Perform forward operation: application of the sampling mask, centered IFFT followed by a
   * sensitivity-weighted combination of the coils. Remove ROOS.
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
};
