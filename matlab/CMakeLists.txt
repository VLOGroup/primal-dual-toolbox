##----------------------------------------------------------------------------
## CMake definitions
##----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
include("../cmake/macros.cmake")

##----------------------------------------------------------------------------
## Matlab
##----------------------------------------------------------------------------
FIND_PACKAGE(Matlab_ICG)
if(MATLAB_FOUND)
    INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

    CUDA_ADD_LIBRARY(gpuTgvMriOptimizer_cartesian SHARED  mexMriCartesianReconstructTgv.cpp OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
    finish_mex_lib(gpuTgvMriOptimizer_cartesian)

    CUDA_ADD_LIBRARY(gpuMriCartesianFwd SHARED  mexMriCartesianFwdOp.cpp OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
    finish_mex_lib(gpuMriCartesianFwd)

    CUDA_ADD_LIBRARY(gpuMriCartesianAdj SHARED  mexMriCartesianAdjOp.cpp OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
    finish_mex_lib(gpuMriCartesianAdj)
    
    CUDA_ADD_LIBRARY(gpuTgvMriOptimizer_cartesianRemoveROOS SHARED  mexMriCartesianRemoveROOSReconstructTgv.cpp OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
    finish_mex_lib(gpuTgvMriOptimizer_cartesianRemoveROOS)

    CUDA_ADD_LIBRARY(gpuMriCartesianRemoveROOSFwd SHARED  mexMriCartesianRemoveROOSFwdOp.cpp OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
    finish_mex_lib(gpuMriCartesianRemoveROOSFwd)

    CUDA_ADD_LIBRARY(gpuMriCartesianRemoveROOSAdj SHARED  mexMriCartesianRemoveROOSAdjOp.cpp OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
    finish_mex_lib(gpuMriCartesianRemoveROOSAdj)

    if(WITH_GPUNUFFT)
        CUDA_ADD_LIBRARY(gpuTgvMriOptimizer_radial SHARED mexMriRadialReconstructTgv.cpp OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
        finish_mex_lib(gpuTgvMriOptimizer_radial)

        CUDA_ADD_LIBRARY(gpuMriRadialFwd SHARED mexMriRadialFwdOp.cpp OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
        finish_mex_lib(gpuMriRadialFwd)

        CUDA_ADD_LIBRARY(gpuMriRadialAdj SHARED mexMriRadialAdjOp.cpp OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
        finish_mex_lib(gpuMriRadialAdj)
    endif(WITH_GPUNUFFT)
else(MATLAB_FOUND)
    message("   [-] Omitting matlab module. Matlab not found.")
endif(MATLAB_FOUND)
