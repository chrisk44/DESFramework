find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(OpenMP REQUIRED)

add_subdirectory(models)
add_subdirectory(des)

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME} DES Mogi Okada)

target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} nvidia-ml)

target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)
