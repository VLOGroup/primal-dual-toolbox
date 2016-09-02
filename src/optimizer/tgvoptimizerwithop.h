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

#include <memory>

#include <iu/iucore.h>
#include <iu/iumath.h>

#include "operator/operatorbase.h"
#include "primaldualtoolbox_api.h"
#include "optimizer_helper.h"
#include "tgvoptimizer.h"

/** \brief Second-order TGV optimizer for arbitrary applications.
 *
 *   Applications are defined by an operator.
 *   According to:
 *   K. Bredies, K. Kunisch and T. Pock:
 *   Total generalized variation.
 *   SIAM Journal on Imaging Sciences 3 (3), 492-526 (2010).
 */
template<typename InputType, typename OutputType>
class PRIMALDUALTOOLBOX_DLLAPI TgvOptimizerWithOp
{
private:
  typedef typename InputType::pixel_type pixel_type;
  typedef typename OutputType::pixel_type output_pixel_type;

  static const unsigned int ndim = InputType::ndim;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;
  typedef typename iu::type_trait<pixel_type>::complex_type complex_type;

public:
  /** Constructor. */
  TgvOptimizerWithOp();

  /** Destructor. */
  virtual ~TgvOptimizerWithOp();

  /** Run optimizer. */
  void solve(bool verbose = false);

  /** Set operator. */
  void setOperator(const std::shared_ptr<OperatorBase<InputType, OutputType> > & op);

  /** Set initial input. */
  void setInput0(const InputType &input);

  /** Set noisy data. */
  void setNoisyData(const std::shared_ptr<OutputType> &f);

  /** Get result of the optimizer. */
  InputType* getResult();

  /** Get optimizer parameters of type TgvParameters. */
  TgvParameters<real_type>& getParameters()
  {
    return params_;
  }

  /** No copies are allowed. */
  TgvOptimizerWithOp(TgvOptimizerWithOp const&) = delete;

  /** No assignments are allowed. */
  void operator=(TgvOptimizerWithOp const&) = delete;

private:
  /** Optimizer parameters of type TgvParameters. */
  TgvParameters<real_type> params_;

  /** Operator. */
  std::shared_ptr<OperatorBase<InputType, OutputType> > op_;

  /** Noisy input data. */
  std::shared_ptr<OutputType> f_;

  /** Primal variable. */
  std::unique_ptr<InputType> u_;

  /** Overrelaxation of primal variable. */
  std::unique_ptr<InputType> u__;

  /** Primal variable. */
  Variable2<InputType> v_;

  /** Overrelaxation of primal variable. */
  Variable2<InputType> v__;

  /** Additional dual variable to handle operator. */
  std::unique_ptr<OutputType> r_;

  /** Dual variable. */
  Variable2<InputType> p_;

  /** Dual variable. */
  Variable2sym<InputType> q_;
};
