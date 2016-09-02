##-----------------------------------------------------------------------------
## Create Tests
##-----------------------------------------------------------------------------
cuda_add_executable(randomTest main_random_test.cu)
target_link_libraries(randomTest primaldualtoolbox ${IMAGEUTILITIES_LIBRARIES})
add_dependencies(randomTest primaldualtoolbox)
if(${WITH_GPUNUFFT})
add_definitions("-DWITH_GPUNUFFT")
endif(${WITH_GPUNUFFT})
cuda_add_executable(adjointnessTest main_adjointness_test.cu)
target_link_libraries(adjointnessTest primaldualtoolbox ${IMAGEUTILITIES_LIBRARIES} ${GPUNUFFT_LIBRARIES})
add_dependencies(adjointnessTest primaldualtoolbox)

IF(WITH_GPUNUFFT)
    cuda_add_executable(gpunufftTest main_gpunufft_test.cu)
    target_link_libraries(gpunufftTest primaldualtoolbox ${IMAGEUTILITIES_LIBRARIES} ${GPUNUFFT_LIBRARIES})
    add_dependencies(gpunufftTest primaldualtoolbox)
endif(WITH_GPUNUFFT)