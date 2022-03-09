cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(desframework LANGUAGES CXX CUDA)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

add_subdirectory(src)
