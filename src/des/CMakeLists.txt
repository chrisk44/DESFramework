find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(OpenMP REQUIRED)

set(MODULE_NAME DES)

add_library(${MODULE_NAME} framework.cpp
                           framework.h
                           computeThread.cu
                           coordinatorThread.cpp
                           masterProcess.cpp
                           slaveProcess.cpp
                           messaging.cpp
                           cpuKernel.h
                           cpuKernel.cpp
                           gpuKernel.h
                           gpuKernel.cu
                           utilities/stopwatch.h
                           utilities/stopwatch.cpp
                           utilities/utilities.cpp
                           utilities/utilities.h)

file(GLOB DES_SRC CONFIGURE_DEPENDS "*.cpp")
set_source_files_properties(${DES_SRC} PROPERTIES LANGUAGE CUDA)

target_link_libraries(${MODULE_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${MODULE_NAME} ${MPI_LIBRARIES})
target_link_libraries(${MODULE_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${MODULE_NAME} nvidia-ml)

target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       -Xcompiler "-fopenmp -O3 -g"
                       >)

set_property(TARGET ${MODULE_NAME} PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET ${MODULE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
