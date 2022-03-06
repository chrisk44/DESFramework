find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(Mogi mogi_common.h
                 mogi.h
                 mogi1.cpp
                 mogi2.cpp)
                 
