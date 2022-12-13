cmake_minimum_required(VERSION 3.24)
project(cuda_practice CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_practice main.cu)

set_target_properties(cuda_practice PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


add_executable(VectorAdd VectorAdd.cu)