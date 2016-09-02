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

//-----------------------------------------------------------------------------
/* Shared lib macros for windows dlls
*/
#ifdef WIN32
#pragma warning( disable : 4251 ) // disable the warning about exported template code from stl
#pragma warning( disable : 4231 ) // disable the warning about nonstandard extension in e.g. istream

// mri reconstruction module
#ifdef PRIMALDUALTOOLBOX_USE_STATIC
#define PRIMALDUALTOOLBOX_DLLAPI
#else
  #ifdef PRIMALDUALTOOLBOX_EXPORTS
#define PRIMALDUALTOOLBOX_DLLAPI __declspec(dllexport)
  #else
#define PRIMALDUALTOOLBOX_DLLAPI __declspec(dllimport)
  #endif
#endif

#else
  #define PRIMALDUALTOOLBOX_DLLAPI
#endif
