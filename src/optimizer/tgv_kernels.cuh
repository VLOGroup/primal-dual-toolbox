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
#include <iu/iuhelpermath.h>

#include "optimizer_helper.h"

template<typename InputType>
__device__ void dp_sym(typename Variable2<InputType>::KernelData &v,
                       unsigned int x, unsigned int y,
                       typename InputType::pixel_type& grad_v_xx,
                       typename InputType::pixel_type& grad_v_yy,
                       typename InputType::pixel_type& grad_v_xy)
{
  const unsigned int xp = x + (x < v.x_.size_[0] - 1);
  const unsigned int yp = y + (y < v.x_.size_[1] - 1);

  grad_v_xx = v.x_(xp, y) - v.x_(x, y);
  grad_v_xy = 0.5 * (v.x_(x, yp) - v.x_(x, y) + v.y_(xp, y) - v.y_(x, y));
  grad_v_yy = v.y_(x, yp) - v.y_(x, y);
}

template<typename InputType>
__device__ void dp(typename InputType::KernelData &u, unsigned int x,
                   unsigned int y, typename InputType::pixel_type& grad_u_x,
                   typename InputType::pixel_type& grad_u_y)
{
  const unsigned int xp = x + (x < u.size_[0] - 1);
  const unsigned int yp = y + (y < u.size_[1] - 1);

  grad_u_x = u(xp, y) - u(x, y);
  grad_u_y = u(x, yp) - u(x, y);
}

template<typename InputType>
__device__ void dp(typename InputType::KernelData &u, unsigned int x,
                   unsigned int y, unsigned int z,
                   typename InputType::pixel_type& grad_u_x,
                   typename InputType::pixel_type& grad_u_y,
                   typename InputType::pixel_type& grad_u_z)
{
  const unsigned int xp = x + (x < u.size_[0] - 1);
  const unsigned int yp = y + (y < u.size_[1] - 1);
  const unsigned int zp = z + (z < u.size_[2] - 1);

  grad_u_x = u(xp, y, z) - u(x, y, z);
  grad_u_y = u(x, yp, z) - u(x, y, z);
  grad_u_z = u(x, y, zp) - u(x, y, z);
}

template<typename InputType>
__device__ void dm(typename Variable2<InputType>::KernelData &p, unsigned int x,
                   unsigned int y, typename InputType::pixel_type& div_p)
{
  div_p =
      (x > 0) ?
          ((x < p.x_.size_[0] - 1) ?
              p.x_(x, y) - p.x_(x - 1, y) : -p.x_(x - 1, y)) :
          p.x_(x, y);
  div_p +=
      (y > 0) ?
          ((y < p.x_.size_[1] - 1) ?
              p.y_(x, y) - p.y_(x, y - 1) : -p.y_(x, y - 1)) :
          p.y_(x, y);
}

template<typename InputType>
__device__ void dm(typename Variable3<InputType>::KernelData &p, unsigned int x,
                   unsigned int y, unsigned int z,
                   typename InputType::pixel_type& div_p)
{
  div_p =
      -( (x > 0) ?
        ((x < p.x_.size_[0] - 1) ? p.x_(x, y, z) - p.x_(x - 1, y, z) : -p.x_(x - 1, y, z)) :
        p.x_(x, y, z) ) / p.dx_;
  div_p -=
      ( (y > 0) ?
          ((y < p.x_.size_[1] - 1) ?
              p.y_(x, y, z) - p.y_(x, y - 1, z) : -p.y_(x, y - 1, z)) :
          p.y_(x, y, z) ) / p.dy_;
  div_p -=
      ( (z > 0) ?
          ((z < p.x_.size_[2] - 1) ?
              p.z_(x, y, z) - p.z_(x, y, z - 1) : -p.z_(x, y, z - 1)) :
          p.z_(x, y, z) ) / p.dz_;
}

template<typename InputType>
__device__ void dm_sym(typename Variable2sym<InputType>::KernelData &q,
                       unsigned int x, unsigned int y,
                       typename InputType::pixel_type& div_q_x,
                       typename InputType::pixel_type& div_q_y)
{
  typedef typename InputType::pixel_type pixel_type;
  pixel_type div_q_xx_x =
      (x > 0) ?
          ((x < q.xx_.size_[0] - 1) ?
              q.xx_(x, y) - q.xx_(x - 1, y) : -q.xx_(x - 1, y)) :
          q.xx_(x, y);
  pixel_type div_q_yy_y =
      (y > 0) ?
          ((y < q.xx_.size_[1] - 1) ?
              q.yy_(x, y) - q.yy_(x, y - 1) : -q.yy_(x, y - 1)) :
          q.yy_(x, y);
  pixel_type div_q_xy_x =
      (x > 0) ?
          ((x < q.xx_.size_[0] - 1) ?
              q.xy_(x, y) - q.xy_(x - 1, y) : -q.xy_(x - 1, y)) :
          q.xy_(x, y);
  pixel_type div_q_xy_y =
      (y > 0) ?
          ((y < q.xx_.size_[1] - 1) ?
              q.xy_(x, y) - q.xy_(x, y - 1) : -q.xy_(x, y - 1)) :
          q.xy_(x, y);

  div_q_x = div_q_xx_x + div_q_xy_y;
  div_q_y = div_q_yy_y + div_q_xy_x;
}

template<typename PixelType>
__device__ typename iu::type_trait<PixelType>::real_type norm_sym(
    PixelType q_xx, PixelType q_yy, PixelType q_xy)
{
  return sqrt(
      sqr(iu::type_trait < PixelType > ::abs(q_xx))
          + sqr(iu::type_trait < PixelType > ::abs(q_yy))
          + 2.0 * sqr(iu::type_trait < PixelType > ::abs(q_xy)));
}

template<typename PixelType>
__device__ typename iu::type_trait<PixelType>::real_type norm(PixelType u_x,
                                                              PixelType u_y)
{
  return sqrt(
      sqr(iu::type_trait < PixelType > ::abs(u_x))
          + sqr(iu::type_trait < PixelType > ::abs(u_y)));
}

template<typename PixelType>
__device__ typename iu::type_trait<PixelType>::real_type norm(PixelType u_x,
                                                              PixelType u_y,
                                                              PixelType u_z)
{
  return sqrt(
      sqr(iu::type_trait < PixelType > ::abs(u_x))
          + sqr(iu::type_trait < PixelType > ::abs(u_y))
          + sqr(iu::type_trait < PixelType > ::abs(u_z)));
}

template<typename InputType>
__global__ void TV_dual_overrelax(
    typename Variable3<InputType>::KernelData p_bar,
    typename Variable3<InputType>::KernelData p,
    typename Variable3<InputType>::KernelData p_old,
    typename iu::type_trait<typename InputType::pixel_type>::real_type theta = 1.0)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < p.x_.size_[0] && y < p.x_.size_[1] && z < p.x_.size_[2])
  {
    p_bar.x_(x,y,z) = (1+theta)*p.x_(x,y,z) - theta*p_old.x_(x,y,z);
    p_bar.y_(x,y,z) = (1+theta)*p.y_(x,y,z) - theta*p_old.y_(x,y,z);
    p_bar.z_(x,y,z) = (1+theta)*p.z_(x,y,z) - theta*p_old.z_(x,y,z);
  }
}

template<typename InputType>
__global__ void TV_primal_nabla(
    typename Variable3<InputType>::KernelData p,
    typename InputType::KernelData u)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < u.size_[0] && y < u.size_[1] && z < u.size_[2])
  {
    pixel_type grad_u_x, grad_u_y, grad_u_z;
    dp<InputType>(u, x, y, z, grad_u_x, grad_u_y, grad_u_z);

    p.x_(x,y,z) = grad_u_x / p.dx_;
    p.y_(x,y,z) = grad_u_y / p.dy_;
    p.z_(x,y,z) = grad_u_z / p.dz_;
  }
}

template<typename OutputType>
__global__ void TGV_prox_r_kernel(
    typename OutputType::KernelData r, typename OutputType::KernelData f,
    typename iu::type_trait<typename OutputType::pixel_type>::real_type sigma,
    typename iu::type_trait<typename OutputType::pixel_type>::real_type lambda = 1.0)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < f.numel_)
  {
    r(idx) = (r(idx) + sigma * f(idx)) / (1 + sigma * lambda);
  }
}

template<typename InputType>
__global__ void TGV_dual_q_kernel(
    typename Variable2sym<InputType>::KernelData q,
    typename Variable2<InputType>::KernelData v,
    typename iu::type_trait<typename InputType::pixel_type>::real_type sigma,
    typename iu::type_trait<typename InputType::pixel_type>::real_type alpha0)
{
  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < v.x_.size_[0] && y < v.x_.size_[1])
  {
    // compute symmetric gradient of v
    pixel_type grad_v_xx, grad_v_yy, grad_v_xy;
    dp_sym<InputType>(v, x, y, grad_v_xx, grad_v_yy, grad_v_xy);

    // gradient ascent in the dual variable
    pixel_type q_xx = q.xx_(x, y) + sigma * grad_v_xx;
    pixel_type q_yy = q.yy_(x, y) + sigma * grad_v_yy;
    pixel_type q_xy = q.xy_(x, y) + sigma * grad_v_xy;

    // projection of q
    real_type q_norm = norm_sym(q_xx, q_yy, q_xy);
    real_type scale = 1.0 / iu::type_trait<real_type>::max(1.0, q_norm / alpha0);

    // write back
    q.xx_(x, y) = q_xx * scale;
    q.yy_(x, y) = q_yy * scale;
    q.xy_(x, y) = q_xy * scale;
  }
}

template<typename InputType>
__global__ void TGV_dual_p_kernel(
    typename Variable2<InputType>::KernelData p,
    typename InputType::KernelData u,
    typename Variable2<InputType>::KernelData v,
    typename iu::type_trait<typename InputType::pixel_type>::real_type sigma,
    typename iu::type_trait<typename InputType::pixel_type>::real_type alpha1)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < v.x_.size_[0] && y < v.x_.size_[1])
  {
    // compute gradient of u
    pixel_type grad_u_x, grad_u_y;
    dp<InputType>(u, x, y, grad_u_x, grad_u_y);

    // gradient ascent in the dual variable
    pixel_type p_x = p.x_(x, y) + sigma * (grad_u_x - v.x_(x, y));
    pixel_type p_y = p.y_(x, y) + sigma * (grad_u_y - v.y_(x, y));

    // projection of p
    real_type p_norm = norm(p_x, p_y);
    real_type scale = 1.0 / iu::type_trait<real_type>::max(1.0, p_norm / alpha1);

    // write back
    p.x_(x, y) = p_x * scale;
    p.y_(x, y) = p_y * scale;
  }
}

template<typename InputType>
__global__ void TV_dual_p_kernel(
    typename Variable2<InputType>::KernelData p,
    typename InputType::KernelData u,
    typename iu::type_trait<typename InputType::pixel_type>::real_type sigma)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < u.size_[0] && y < u.size_[1])
  {
    // compute gradient of u
    pixel_type grad_u_x, grad_u_y;
    dp<InputType>(u, x, y, grad_u_x, grad_u_y);

    // gradient ascent in the dual variable
    pixel_type p_x = p.x_(x, y) + sigma * grad_u_x;
    pixel_type p_y = p.y_(x, y) + sigma * grad_u_y;

    // projection of p
    real_type p_norm = norm(p_x, p_y);
    real_type scale = 1.0 / iu::type_trait<real_type>::max(1.0, p_norm);

    // write back
    p.x_(x, y) = p_x * scale;
    p.y_(x, y) = p_y * scale;
  }
}

template<typename InputType>
__global__ void TV_dual_p_kernel(
    typename Variable3<InputType>::KernelData p,
    typename InputType::KernelData u,
    typename InputType::pixel_type sigma, typename InputType::pixel_type lambda)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  typedef typename InputType::pixel_type pixel_type;

  if (x < u.size_[0] && y < u.size_[1] && z < u.size_[2])
  {
    // compute gradient of u
    pixel_type grad_u_x, grad_u_y, grad_u_z;
    dp<InputType>(u, x, y, z, grad_u_x, grad_u_y, grad_u_z);

    // gradient ascent in the dual variable
    pixel_type p_x = p.x_(x, y, z) + sigma * (grad_u_x / p.dx_);
    pixel_type p_y = p.y_(x, y, z) + sigma * (grad_u_y / p.dy_);
    pixel_type p_z = p.z_(x, y, z) + sigma * (grad_u_z / p.dz_);

    // projection of p
    pixel_type p_norm = norm(p_x, p_y, p_z);
    pixel_type scale = 1.0 / iu::type_trait<pixel_type>::max(1.0, p_norm/lambda);

    // write back
    p.x_(x, y, z) = p_x * scale;
    p.y_(x, y, z) = p_y * scale;
    p.z_(x, y, z) = p_z * scale;
  }
}

template<typename InputType>
__global__ void TV_dual_p_kernel(
    typename Variable3<InputType>::KernelData p,
    typename Variable3<InputType>::KernelData p_old,
    typename InputType::KernelData u,
    typename InputType::pixel_type sigma, typename InputType::pixel_type lambda)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  typedef typename InputType::pixel_type pixel_type;

  if (x < u.size_[0] && y < u.size_[1] && z < u.size_[2])
  {
    // compute gradient of u
    pixel_type grad_u_x, grad_u_y, grad_u_z;
    dp<InputType>(u, x, y, z, grad_u_x, grad_u_y, grad_u_z);

    // gradient ascent in the dual variable
    pixel_type p_x = p_old.x_(x, y, z) + sigma * (grad_u_x / p_old.dx_);
    pixel_type p_y = p_old.y_(x, y, z) + sigma * (grad_u_y / p_old.dy_);
    pixel_type p_z = p_old.z_(x, y, z) + sigma * (grad_u_z / p_old.dz_);

    // projection of p
    pixel_type p_norm = norm(p_x, p_y, p_z);
    pixel_type scale = 1.0 / iu::type_trait<pixel_type>::max(1.0, p_norm/lambda);

    // store the old value
    p_old.x_(x, y, z) = p.x_(x, y, z);
    p_old.y_(x, y, z) = p.y_(x, y, z);
    p_old.z_(x, y, z) = p.z_(x, y, z);
    // write back
    p.x_(x, y, z) = p_x * scale;
    p.y_(x, y, z) = p_y * scale;
    p.z_(x, y, z) = p_z * scale;
  }
}


template<typename InputType>
__global__ void TV_dual_p_l1_kernel(
    typename Variable2<InputType>::KernelData p,
    typename InputType::KernelData u,
    typename iu::type_trait<typename InputType::pixel_type>::real_type sigma,
    typename iu::type_trait<typename InputType::pixel_type>::real_type alpha0)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < u.size_[0] && y < u.size_[1])
  {
    // compute gradient of u
    pixel_type grad_u_x, grad_u_y;
    dp<InputType>(u, x, y, grad_u_x, grad_u_y);

    // gradient ascent in the dual variable
    pixel_type p_x = p.x_(x, y) + sigma * grad_u_x;
    pixel_type p_y = p.y_(x, y) + sigma * grad_u_y;

    // projection of p
    real_type scale = 1.0 / (1.0 + sigma/alpha0);

    // write back
    p.x_(x, y) = p_x * scale;
    p.y_(x, y) = p_y * scale;
  }
}

template<typename InputType>
__global__ void TGV_primal_u_noprox_kernel(
    typename InputType::KernelData u, typename InputType::KernelData u_,
    typename Variable2<InputType>::KernelData p,
    typename iu::type_trait<typename InputType::pixel_type>::real_type tau,
    typename iu::type_trait<typename InputType::pixel_type>::real_type theta =
        1.0)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < u.size_[0] && y < u.size_[1])
  {
    // remember old u
    pixel_type u_old = u(x, y);

    // extract precomputed f from u_
    pixel_type f = u_(x, y);

    // compute divergence of p
    pixel_type div_p;
    dm<InputType>(p, x, y, div_p);

    // gradient descent in the primal variable
    pixel_type u_new = u_old + tau * (div_p - f);

    // write back
    u(x, y) = u_new;

    // overrelaxation
    u_(x, y) = (1 + theta) * u_new - theta * u_old;
  }
}

template<typename InputType>
__global__ void TGV_primal_u_noprox_kernel(
    typename InputType::KernelData u, typename InputType::KernelData u_,
    typename Variable3<InputType>::KernelData p,
    typename iu::type_trait<typename InputType::pixel_type>::real_type tau,
    typename iu::type_trait<typename InputType::pixel_type>::real_type theta =
        1.0)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < u.size_[0] && y < u.size_[1] && z < u.size_[2])
  {
    // remember old u
    pixel_type u_old = u(x, y, z);

    // extract precomputed f from u_
    pixel_type f = u_(x, y, z);

    // compute divergence of p
    pixel_type div_p;
    dm<InputType>(p, x, y, z, div_p);

    // gradient descent in the primal variable
    pixel_type u_new = u_old + tau * (div_p - f);

    // write back
    u(x, y, z) = u_new;

    // overrelaxation
    u_(x, y, z) = (1 + theta) * u_new - theta * u_old;
  }
}

template<typename InputType>
__global__ void TV_primal_u_noprox_kernel(
    typename InputType::KernelData u, typename InputType::KernelData u_,
    typename InputType::KernelData grad_u,
    typename Variable3<InputType>::KernelData p,
    typename iu::type_trait<typename InputType::pixel_type>::real_type tau,
    typename iu::type_trait<typename InputType::pixel_type>::real_type theta =
        1.0)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < u.size_[0] && y < u.size_[1] && z < u.size_[2])
  {
    // remember old u
    pixel_type u_old = u(x, y, z);

    // compute divergence of p
    pixel_type div_p;
    dm<InputType>(p, x, y, z, div_p);

    // gradient descent in the primal variable
    pixel_type u_new = u_old - tau * (div_p + grad_u(x,y,z));

    // write back
    u(x, y, z) = u_new;

    // overrelaxation
    u_(x, y, z) = (1 + theta) * u_new - theta * u_old;
  }
}

template<typename InputType>
__global__ void TV_primal_u_noprox_noover_kernel(
    typename InputType::KernelData u,
    typename InputType::KernelData u_old, typename InputType::KernelData grad_u,
    typename Variable3<InputType>::KernelData p,
    typename iu::type_trait<typename InputType::pixel_type>::real_type tau)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < u.size_[0] && y < u.size_[1] && z < u.size_[2])
  {
    // compute divergence of p
    pixel_type div_p;
    dm<InputType>(p, x, y, z, div_p);

    // write back
    u(x, y, z) = u_old(x, y, z) - tau * (div_p + grad_u(x,y,z));
  }
}

template<typename InputType>
__global__ void TV_primal_u_kernel(
    typename InputType::KernelData u, typename InputType::KernelData u_,
    typename InputType::KernelData f,
    typename Variable2<InputType>::KernelData p,
    typename iu::type_trait<typename InputType::pixel_type>::real_type tau,
    typename iu::type_trait<typename InputType::pixel_type>::real_type lambda,
    typename iu::type_trait<typename InputType::pixel_type>::real_type theta =
        1.0)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < u.size_[0] && y < u.size_[1])
  {
    // remember old u
    pixel_type u_old = u(x, y);

    // compute divergence of p
    pixel_type div_p;
    dm<InputType>(p, x, y, div_p);

    // gradient descent in the primal variable
    pixel_type u_new = u_old + tau * (div_p + lambda*f(x, y));

    // Compute prox
    u_new /= (1 + tau * lambda);

    // write back
    u(x, y) = u_new;

    // overrelaxation
    u_(x, y) = (1 + theta) * u_new - theta * u_old;
  }
}

template<typename InputType>
__global__ void TV_primal_u_kernel(
    typename InputType::KernelData u, typename InputType::KernelData u_,
    typename InputType::KernelData f,
    typename Variable3<InputType>::KernelData p,
    typename iu::type_trait<typename InputType::pixel_type>::real_type tau,
    typename iu::type_trait<typename InputType::pixel_type>::real_type lambda,
    typename iu::type_trait<typename InputType::pixel_type>::real_type theta =
        1.0)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < u.size_[0] && y < u.size_[1] && z < u.size_[2])
  {
    // remember old u
    pixel_type u_old = u(x, y, z);

    // compute divergence of p
    pixel_type div_p;
    dm<InputType>(p, x, y, z, div_p);

    // gradient descent in the primal variable
    pixel_type u_new = u_old + tau * (div_p + lambda*f(x, y));

    // Compute prox
    u_new /= (1 + tau * lambda);

    // write back
    u(x, y) = u_new;

    // overrelaxation
    u_(x, y) = (1 + theta) * u_new - theta * u_old;
  }
}

template<typename InputType>
__global__ void TGV_primal_v_kernel(
    typename Variable2<InputType>::KernelData v,
    typename Variable2<InputType>::KernelData v_,
    typename Variable2<InputType>::KernelData p,
    typename Variable2sym<InputType>::KernelData q,
    typename iu::type_trait<typename InputType::pixel_type>::real_type tau,
    typename iu::type_trait<typename InputType::pixel_type>::real_type theta =
        1.0)
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

  typedef typename InputType::pixel_type pixel_type;
  typedef typename iu::type_trait<pixel_type>::real_type real_type;

  if (x < v.x_.size_[0] && y < v.x_.size_[1])
  {
    // remember old v
    pixel_type v_x_old = v.x_(x, y);
    pixel_type v_y_old = v.y_(x, y);

    // compute symmetric divergence of q
    pixel_type div_q_x, div_q_y;
    dm_sym<InputType>(q, x, y, div_q_x, div_q_y);

    // gradient descent in the primal variable
    pixel_type v_x = v_x_old + tau * (div_q_x + p.x_(x, y));
    pixel_type v_y = v_y_old + tau * (div_q_y + p.y_(x, y));

    // write back
    v.x_(x, y) = v_x;
    v.y_(x, y) = v_y;

    // overrelaxation
    v_.x_(x, y) = (1 + theta) * v_x - theta * v_x_old;
    v_.y_(x, y) = (1 + theta) * v_y - theta * v_y_old;

  }
}

