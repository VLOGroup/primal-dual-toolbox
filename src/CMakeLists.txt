##-----------------------------------------------------------------------------
## Files
##-----------------------------------------------------------------------------
set(COMMON_HEADER
    definitions.h
    primaldualtoolbox_api.h
    iurandom.h
    math_helper.h
)

set(COMMON_SOURCE
)

set(OPERATOR_HEADER
    operator/operatorbase.h
    operator/mricartesianoperator.h
    operator/mrisamplingoperator.h
)

set(OPERATOR_SOURCE
    operator/mricartesianoperator.cu
    operator/mrisamplingoperator.cu
)

set(OPTIMIZER_HEADER
	optimizer/tgvmrioptimizer.h
    optimizer/tvoptimizer.h
    optimizer/tgvoptimizer.h
    optimizer/optimizer_helper.h
    optimizer/tgvoptimizerwithop.h
    optimizer/tvoptimizerwithop.h
)

set(OPTIMIZER_SOURCE
    optimizer/tgvmrioptimizer.cu
    optimizer/tgv_kernels.cuh
    optimizer/tvoptimizer.cu
    optimizer/tgvoptimizer.cu
    optimizer/tgvoptimizerwithop.cu
    optimizer/tvoptimizerwithop.cu
)

if(WITH_GPUNUFFT)
    set(OPERATOR_HEADER ${OPERATOR_HEADER} operator/mriradialoperator.h)
    set(OPERATOR_SOURCE ${OPERATOR_SOURCE} operator/mriradialoperator.cu)
endif(WITH_GPUNUFFT)

set(HEADERS
    ${COMMON_HEADER}
    ${OPTIMIZER_HEADER}
    ${OPERATOR_HEADER}
)

set(SOURCES
    ${COMMON_SOURCE}
    ${OPTIMIZER_SOURCE}
    ${OPERATOR_SOURCE}
)

##-----------------------------------------------------------------------------
## Core lib
##-----------------------------------------------------------------------------
cuda_add_library(primaldualtoolbox SHARED ${HEADERS} ${SOURCES} OPTIONS "-DPRIMALDUALTOOLBOX_EXPORTS")
set_target_properties(primaldualtoolbox PROPERTIES DEFINE_SYMBOL PRIMALDUALTOOLBOX_EXPORTS)
target_link_libraries(primaldualtoolbox ${IMAGEUTILITIES_LIBRARIES} ${GPUNUFFT_LIBRARIES})
