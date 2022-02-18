cmake_minimum_required(VERSION 3.18)

project(Assignment02 LANGUAGES CUDA CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(CUDAToolkit)
find_package(OpenMP)


set(SOURCES 
    src/main.cpp
    src/sdt_cpu.cpp
    src/sdt_gpu.cpp
    src/sdt_gpu_kernels.cu
    )

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES 
                      CUDA_SEPERABLE_COMPILATION ON
                      CUDA_ARCHITECTURES 60)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                      --keep # Keep PTX code 
                      >)
if(OpenMP_CXX_FOUND)
	target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
