##----------------------------------------------------------------------------
## CMake definitions
##----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
include("../cmake/macros.cmake")

##----------------------------------------------------------------------------
## Python (for wrappers)
##----------------------------------------------------------------------------
## Find correct version if e.g. Anaconda is used.
find_package(PythonInterp) # run this such that lib and interpreter path match.
find_package(PythonLibs 2.7 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
message(STATUS "Python libraries: " ${PYTHON_LIBRARIES})

# Extract boost root dir. Should be the same as the Python lib dir.
get_filename_component(PYTHON_LIBRARY_DIR ${PYTHON_LIBRARIES} DIRECTORY)
get_filename_component(BOOST_ROOT ${PYTHON_LIBRARY_DIR} DIRECTORY)

message(STATUS "Boost root: " ${BOOST_ROOT})

find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS python)
message(STATUS "Boost libraries:" ${Boost_LIBRARIES})

if(NOT (${PYTHON_LIBRARY_DIR} MATCHES  ${Boost_LIBRARY_DIRS}))
    message(FATAL_ERROR "Boost library dir does not match python library dir. Is boost installed? "
                        "If you use anaconda, type `conda install boost`. "
                        "If you use system python (ubuntu), type `sudo apt-get install libboost-python`.")
endif(NOT (${PYTHON_LIBRARY_DIR} MATCHES  ${Boost_LIBRARY_DIRS}))

include_directories(${Boost_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

##----------------------------------------------------------------------------
## Generate lib for MRI Reconstruction
##----------------------------------------------------------------------------
if(${WITH_GPUNUFFT})
    add_definitions("-DWITH_GPUNUFFT")
endif(${WITH_GPUNUFFT})
cuda_add_library(pymrireconstruction SHARED  pyMriReconstruction.cpp)
finish_python_lib(pymrireconstruction)

##----------------------------------------------------------------------------
## Generate lib for denoising
##----------------------------------------------------------------------------
cuda_add_library(pydenoising SHARED pyDenoising.cpp)
finish_python_lib(pydenoising)
