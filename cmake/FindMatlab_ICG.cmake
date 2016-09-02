# - this module looks for Matlab
#
# Extension (Manuel Werlberger <werlberger@icg.tugraz.at>:
#   - really find matlab independend to the version number
#   - Therefore set env variable: MATLAB_ROOT=/path/to/matlab
# Extension (Christian Reinbacher <reinbacher@icg.tugraz.at>:
#   - report correct suffix of mex file depending on architecture
#
#
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
#  MATLAB_LIBRARIES:   required libraries: libmex, etc
#  MATLAB_MEX_LIBRARY: path to libmex.lib
#  MATLAB_MX_LIBRARY:  path to libmx.lib
#  MATLAB_ENG_LIBRARY: path to libeng.lib
#  MATLAB_SUFFIX:      correct suffix depending on arch

#=============================================================================
# Copyright 2005-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distributed this file outside of CMake, substitute the full
#  License text for the above reference.)

set(MATLAB_ROOT $ENV{MATLAB_ROOT})

message(STATUS "You are using the ICG FindMatlab_ICG.cmake script.")
message("searching in ${MATLAB_ROOT}")

set(MATLAB_FOUND 0)
if(WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 4) # je olde 32 bit
	  set(MATLAB_SEARCH_SUFFIX "extern/lib/win32/microsoft")
	  if(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
		set(MATLAB_SEARCH_SUFFIX "extern/lib/win32/microsoft/msvc60")
	  else(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
		if(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
		  # Assume people are generally using 7.1,
		  # if using 7.0 need to link to: ../extern/lib/win32/microsoft/msvc70
		  set(MATLAB_SEARCH_SUFFIX "extern/lib/win32/microsoft/msvc71")
		else(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
		  if(${CMAKE_GENERATOR} MATCHES "Borland")
		    # Same here, there are also: bcc50 and bcc51 directories
		    set(MATLAB_SEARCH_SUFFIX "extern/lib/win32/microsoft/bcc54")
		  else(${CMAKE_GENERATOR} MATCHES "Borland")
		    if(MATLAB_FIND_REQUIRED)
		      MESSAGE(FATAL_ERROR "Generator not compatible: ${CMAKE_GENERATOR}")
		    endif(MATLAB_FIND_REQUIRED)
		  endif(${CMAKE_GENERATOR} MATCHES "Borland")
		endif(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
	  endif(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
  else(CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(MATLAB_SEARCH_SUFFIX "extern/lib/win64/microsoft")
	  if(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
		set(MATLAB_SEARCH_SUFFIX "extern/lib/win64/microsoft/msvc60")
	  else(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
		if(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
		  # Assume people are generally using 7.1,
		  # if using 7.0 need to link to: ../extern/lib/win32/microsoft/msvc70
		  set(MATLAB_SEARCH_SUFFIX "extern/lib/win64/microsoft/msvc71")
		else(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
		  if(${CMAKE_GENERATOR} MATCHES "Borland")
		    # Same here, there are also: bcc50 and bcc51 directories
		    set(MATLAB_SEARCH_SUFFIX "extern/lib/win64/microsoft/bcc54")
		  else(${CMAKE_GENERATOR} MATCHES "Borland")
		    if(MATLAB_FIND_REQUIRED)
		      MESSAGE(FATAL_ERROR "Generator not compatible: ${CMAKE_GENERATOR}")
		    endif(MATLAB_FIND_REQUIRED)
		  endif(${CMAKE_GENERATOR} MATCHES "Borland")
		endif(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
	  endif(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
  endif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  find_library(MATLAB_MEX_LIBRARY
    libmex
    PATHS ${MATLAB_ROOT}
    PATH_SUFFIXES ${MATLAB_SEARCH_SUFFIX}
    )
  find_library(MATLAB_MX_LIBRARY
    libmx
    PATHS ${MATLAB_ROOT}
    PATH_SUFFIXES ${MATLAB_SEARCH_SUFFIX}
    )
  find_library(MATLAB_ENG_LIBRARY
    libeng 
    PATHS ${MATLAB_ROOT}
    PATH_SUFFIXES ${MATLAB_SEARCH_SUFFIX}
    )
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    # Regular x86
      set(MATLAB_SUFFIX .mexw32)		
  else(CMAKE_SIZEOF_VOID_P EQUAL 4)
    # AMD64:
      set(MATLAB_SUFFIX .mexw64)		
  endif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  # FIXME:
  FIND_PATH(MATLAB_INCLUDE_DIR
    "mex.h"
    "${MATLAB_ROOT}/extern/include"
    )
else( WIN32 )
  ## NOT WINDOWS: 
  
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    # Regular x86
    set(MATLAB_SEARCH_SUFFIX bin/glnx86/ bin/maci86)
    if(APPLE)
      set(MATLAB_SUFFIX .mexmaci)
    else(APPLE)
      set(MATLAB_SUFFIX .mexglx)		
    endif(APPLE)
  else(CMAKE_SIZEOF_VOID_P EQUAL 4)
    # AMD64:
    set(MATLAB_SEARCH_SUFFIX bin/glnxa64/ bin/maci64)
    if(APPLE)
      set(MATLAB_SUFFIX .mexmaci64)
    else(APPLE)
      set(MATLAB_SUFFIX .mexa64)		
    endif(APPLE)
  endif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  
  find_library(MATLAB_MEX_LIBRARY
    NAMES mex
    PATHS ${MATLAB_ROOT}
    PATH_SUFFIXES ${MATLAB_SEARCH_SUFFIX}
    NO_DEFAULT_PATH
    )
  #message("MATLAB_MEX_LIBRARY=${MATLAB_MEX_LIBRARY}")
  find_library(MATLAB_MX_LIBRARY
    NAMES mx
    PATHS ${MATLAB_ROOT}
    PATH_SUFFIXES ${MATLAB_SEARCH_SUFFIX}
    NO_DEFAULT_PATH
    )
  find_library(MATLAB_ENG_LIBRARY
    NAMES eng
    PATHS ${MATLAB_ROOT}
    NO_DEFAULT_PATH
    PATH_SUFFIXES ${MATLAB_SEARCH_SUFFIX}
    )
  FIND_PATH(MATLAB_INCLUDE_DIR
    "mex.h"
    ${MATLAB_ROOT}/extern/include
    )
endif(WIN32)

# This is common to UNIX and Win32:
set(MATLAB_LIBRARIES
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MX_LIBRARY}
  ${MATLAB_ENG_LIBRARY}
)
message("Determined Suffix: ${MATLAB_SUFFIX}")
message("Found Libraries: ${MATLAB_LIBRARIES}")
if(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  set(MATLAB_FOUND 1)
endif(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

MARK_AS_ADVANCED(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_ENG_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
  MATLAB_ROOT
  MATLAB_SUFFIX
)

