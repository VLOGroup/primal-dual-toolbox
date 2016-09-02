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

#include "primaldualtoolbox_api.h"

#include <iu/iumath/typetraits.h>
#include <iu/iumath.h>
#include <iu/iucore.h>

/** \brief Variable2 stores an x and y component that can be used for T(G)V optimization.
*/
template<typename InputType>
class PRIMALDUALTOOLBOX_DLLAPI Variable2
{
public:
  /* Constructor. */
  Variable2()
  {
  }

  /* Destructor. */
  ~Variable2()
  {
  }

  /** No copies are allowed. */
  Variable2(Variable2 const&) = delete;

  /** No assignments are allowed. */
  void operator=(Variable2 const&) = delete;

  /** Set size of x and y component and initialize them with zeros. 
   *  @param size Size of the x/y components. 
   */
  void init(const iu::Size<InputType::ndim>& size)
  {
    // set size
    x_.reset(new InputType(size));
    y_.reset(new InputType(size));

    // init with zeros
    iu::math::fill(*x_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
    iu::math::fill(*y_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
  }

  /** Get raw pointer of x component. */
  InputType * x()
  {
    return x_.get();
  }

  /** Get raw pointer of y component. */
  InputType * y()
  {
    return y_.get();
  }

  /** Get raw pointer of x component (const). */
  const InputType * x() const
  {
    return x_.get();
  }

  /** Get raw pointer of y component (const). */
  const InputType * y() const
  {
    return y_.get();
  }

  /** \brief Struct pointer KernelData that can be used in CUDA kernels.
  *
  *  This struct provides the data pointer for the x and y component.
  */
  struct KernelData
  {
    /** KernelData struct of the x component. */
    typename InputType::KernelData x_;
    /** KernelData struct of the y component. */
    typename InputType::KernelData y_;

    /** Constructor. */
    __host__ KernelData(const Variable2<InputType> &mem) :
        x_(*mem.x()),
        y_(*mem.y())
    {

    }
  };

private:
  /** x component. */
  std::unique_ptr<InputType> x_;
  /** y component. */
  std::unique_ptr<InputType> y_;
};

/** \brief Variable2sym stores an xx, yy and xy component that can be used for T(G)V optimization.
*/
template<typename InputType>
class PRIMALDUALTOOLBOX_DLLAPI Variable2sym
{
public:
  /** Constructor. */
  Variable2sym()
  {
  }

  /** Destructor.*/
  ~Variable2sym()
  {
  }

  /** No copies are allowed. */
  Variable2sym(Variable2sym const&) = delete;

  /** No assignments are allowed. */
  void operator=(Variable2sym const&) = delete;

  /** Set size of xx, xy and yy component and initialize them with zeros. 
   *  @param size Size of the xx/yy/xy components.
   */
  void init(const iu::Size<InputType::ndim>& size)
  {
    // set size
    xx_.reset(new InputType(size));
    yy_.reset(new InputType(size));
    xy_.reset(new InputType(size));

    // fill with zeros
    iu::math::fill(*xx_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
    iu::math::fill(*xy_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
    iu::math::fill(*yy_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
  }

  /** Get raw pointer of xx component. */
  InputType * xx()
  {
    return xx_.get();
  }

  /** Get raw pointer of xy component. */
  InputType * xy()
  {
    return xy_.get();
  }

  /** Get raw pointer of yy component. */
  InputType * yy()
  {
    return yy_.get();
  }

  /** Get raw pointer of xx component (const). */
  const InputType * xx() const
  {
    return xx_.get();
  }

  /** Get raw pointer of xy component (const). */
  const InputType * xy() const
  {
    return xy_.get();
  }

  /** Get raw pointer of yy component (const). */
  const InputType * yy() const
  {
    return yy_.get();
  }

  /** \brief Struct pointer KernelData that can be used in CUDA kernels.
  *
  *  This struct provides the data pointer for the x and y component.
  */
  struct KernelData
  {
    /** KernelData struct of the xx component. */
    typename InputType::KernelData xx_;
    /** KernelData struct of the yy component. */
    typename InputType::KernelData yy_;
    /** KernelData struct of the xy component. */
    typename InputType::KernelData xy_;

    /** Constructor. */
    __host__ KernelData(const Variable2sym<InputType> &mem) :
        xx_(*mem.xx()),
        yy_(*mem.yy()),
        xy_(*mem.xy())
    {

    }
  };

private:
  /** xx component. */
  std::unique_ptr<InputType> xx_;
  /** xy component. */
  std::unique_ptr<InputType> xy_;
  /** yy component. */
  std::unique_ptr<InputType> yy_;
};


/** \brief Variable3 stores an x,y and z component that can be used for T(G)V optimization.
*/
template<typename InputType>
class PRIMALDUALTOOLBOX_DLLAPI Variable3
{
public:
  /* Constructor. */
  Variable3(): dx_(1), dy_(1), dz_(1)
  {
  }

  /* Destructor. */
  ~Variable3()
  {
  }

  /** No copies are allowed. */
  Variable3(Variable3 const&) = delete;

  /** No assignments are allowed. */
  void operator=(Variable3 const&) = delete;

  /** Set size of x and y component and initialize them with zeros.
   *  @param size Size of the x/y components.
   */
  void init(const iu::Size<InputType::ndim>& size)
  {
    // set size
    x_.reset(new InputType(size));
    y_.reset(new InputType(size));
    z_.reset(new InputType(size));

    // init with zeros
    iu::math::fill(*x_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
    iu::math::fill(*y_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
    iu::math::fill(*z_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
  }

  /** Set size of x and y component and initialize them with zeros.
   *  @param size Size of the x/y components.
   */
  void init(const iu::Size<InputType::ndim>& size,
            typename InputType::pixel_type dx,
            typename InputType::pixel_type dy,
            typename InputType::pixel_type dz)
  {
    // set size
    x_.reset(new InputType(size));
    y_.reset(new InputType(size));
    z_.reset(new InputType(size));

    // init with zeros
    iu::math::fill(*x_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
    iu::math::fill(*y_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));
    iu::math::fill(*z_,
                   iu::type_trait<typename InputType::pixel_type>::make(0));

    // set the spacing
    dx_ = dx;
    dy_ = dy;
    dz_ = dz;
  }

  /** Get raw pointer of x component. */
  InputType * x()
  {
    return x_.get();
  }

  /** Get raw pointer of y component. */
  InputType * y()
  {
    return y_.get();
  }

  /** Get raw pointer of z component. */
  InputType * z()
  {
    return z_.get();
  }

  /** Get raw pointer of x component (const). */
  const InputType * x() const
  {
    return x_.get();
  }

  /** Get raw pointer of y component (const). */
  const InputType * y() const
  {
    return y_.get();
  }

  /** Get raw pointer of y component (const). */
  const InputType * z() const
  {
    return z_.get();
  }

  /** \brief Struct pointer KernelData that can be used in CUDA kernels.
  *
  *  This struct provides the data pointer for the x and y component.
  */
  struct KernelData
  {
    /** KernelData struct of the x component. */
    typename InputType::KernelData x_;
    /** KernelData struct of the y component. */
    typename InputType::KernelData y_;
    /** KernelData struct of the z component. */
    typename InputType::KernelData z_;

    /** spacing of x volume */
    typename InputType::pixel_type dx_;
    /** spacing of y volume */
    typename InputType::pixel_type dy_;
    /** spacing of z volume */
    typename InputType::pixel_type dz_;

    /** Constructor. */
    __host__ KernelData(const Variable3<InputType> &mem) :
        x_(*mem.x()),
        y_(*mem.y()),
        z_(*mem.z()),
        dx_(mem.dx_), dy_(mem.dy_), dz_(mem.dz_)
    {

    }
  };

private:
  /** x component. */
  std::unique_ptr<InputType> x_;
  /** y component. */
  std::unique_ptr<InputType> y_;
  /** z component. */
  std::unique_ptr<InputType> z_;

  /** x spacing */
  typename InputType::pixel_type dx_;
  /** y spacing */
  typename InputType::pixel_type dy_;
  /** z spacing */
  typename InputType::pixel_type dz_;
};
