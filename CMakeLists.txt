project(PrimalDualToolbox)

cmake_minimum_required(VERSION 2.8)

##----------------------------------------------------------------------------
## CMake definitions
##----------------------------------------------------------------------------
set(CMAKE_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

##-----------------------------------------------------------------------------
## Library output directory
##-----------------------------------------------------------------------------
if(WIN32)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin)
else(WIN32)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
endif(WIN32)


##-----------------------------------------------------------------------------
## ImageUtilities
##-----------------------------------------------------------------------------
set(ImageUtilities_DIR $ENV{IMAGEUTILITIES_ROOT})
find_package(ImageUtilities REQUIRED COMPONENTS iucore iumath)
include_directories(${IMAGEUTILITIES_INCLUDE_DIR})

##-----------------------------------------------------------------------------
## gpuNUFFT
##-----------------------------------------------------------------------------
option(WITH_GPUNUFFT "Use gpuNUFFT for radial MRI operator. Requires GPUNUFFT_ROOT to be set." OFF)

if(WITH_GPUNUFFT)
    message(STATUS "Use gpuNUFFT (required for radial MRI operator)")
    find_package(GPUNUFFT REQUIRED)
    include_directories(${GPUNUFFT_INCLUDE_DIRS})
else(WITH_GPUNUFFT)
    message(STATUS "Disable gpuNUFFT (required for radial MRI operator)")
endif(WITH_GPUNUFFT)

include_directories(src)
add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(tests)
add_subdirectory(matlab)

##-----------------------------------------------------------------------------
## Documentation - doxygen
##-----------------------------------------------------------------------------
SET(DOXYGEN_OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/html/index.html)
SET(DOXYGEN_EXECUTABLE doxygen)
SET(DOXYGEN_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxconfig)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/docs/doxconfig.in ${DOXYGEN_INPUT})

ADD_CUSTOM_TARGET( apidoc
 COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
 COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
 COMMAND ${CMAKE_COMMAND} -E echo "Done."
 WORKING_DIRECTORY ${CMAKE_CACHEFILE_DIR}
 DEPENDS ${DOXYGEN_INPUT}
 )
