cmake_minimum_required(VERSION 4.1)
project(cuda CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

#if(MSVC)
#    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
#endif()

set(CMAKE_CUDA_FLAGS "-O3")

add_executable(cuda main.cu)
