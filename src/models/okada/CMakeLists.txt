find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(Okada okada_common.h
                  okada.h
                  okada1.cpp
                  okada2.cpp
                  okada3.cpp)
