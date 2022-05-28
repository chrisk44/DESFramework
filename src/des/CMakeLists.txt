find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(OpenMP REQUIRED)

add_library(Stopwatch stopwatch.h stopwatch.cpp)
add_library(Utilities utilities.h utilities.cpp)

add_library(GpuKernel gpuKernel.h gpuKernel.cu)
target_compile_options(GpuKernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       -Xcompiler "-O3 -g -Wall -Wextra"
                       >)

add_library(CpuKernel cpuKernel.h cpuKernel.cpp)
target_link_libraries(CpuKernel OpenMP::OpenMP_CXX)

add_library(ComputeThread computeThread.h computeThread.cu)
target_link_libraries(ComputeThread Stopwatch)
target_link_libraries(ComputeThread OpenMP::OpenMP_CXX)
target_link_libraries(ComputeThread ${CUDA_LIBRARIES})
target_link_libraries(ComputeThread nvidia-ml)

add_library(Coordinator coordinatorThread.h coordinatorThread.cpp)
target_link_libraries(Coordinator ComputeThread)

set(MODULE_NAME DES)

add_library(${MODULE_NAME} desf.cpp
                           desf.h
                           masterProcess.cpp
                           slaveProcess.cpp
                           messaging.cpp
                           definitions.h
                           scheduler/scheduler.h
                           scheduler/constantScheduler.h
                           scheduler/uniformScheduler.h
                           scheduler/hrdlsScheduler.h
                           scheduler/hrslsScheduler.h
                           types.cpp
                           types.h)

file(GLOB DES_SRC CONFIGURE_DEPENDS "*.cpp")
set_source_files_properties(${DES_SRC} PROPERTIES LANGUAGE CUDA)

target_link_libraries(${MODULE_NAME} Coordinator ComputeThread Stopwatch CpuKernel GpuKernel Utilities)

target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       -Xcompiler "-O3 -g -Wall -Wextra"
                       >)

set_property(TARGET ${MODULE_NAME} PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET ${MODULE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
