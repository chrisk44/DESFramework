find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(OpenMP REQUIRED)

set(MODULE_NAME DES)

add_library(${MODULE_NAME} framework.cu
                           framework.h
                           kernels.cu
                           kernels.h
                           utilities.cpp
                           utilities.h)

target_link_libraries(${MODULE_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${MODULE_NAME} ${MPI_LIBRARIES})
target_link_libraries(${MODULE_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${MODULE_NAME} nvidia-ml)

target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

set_property(TARGET ${MODULE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
