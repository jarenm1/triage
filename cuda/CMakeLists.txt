cmake_minimum_required(VERSION 3.27)

project(sim_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)
get_property(CUDA_CUDART_LIBRARY CACHE CUDA_CUDART PROPERTY VALUE)
get_filename_component(CUDA_RUNTIME_LIBRARY_DIR "${CUDA_CUDART_LIBRARY}" DIRECTORY)
get_filename_component(CUDA_RUNTIME_ROOT "${CUDA_RUNTIME_LIBRARY_DIR}" DIRECTORY)
if(EXISTS "${CUDA_RUNTIME_ROOT}/include/cuda_runtime.h")
  set(SIM_CUDA_INCLUDE_DIR "${CUDA_RUNTIME_ROOT}/include")
else()
  set(SIM_CUDA_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}")
endif()

add_library(sim_cuda_runtime STATIC
  src/physics.cu
  src/reference_physics.cpp
  src/runtime.cu
)
target_include_directories(sim_cuda_runtime PUBLIC src "${SIM_CUDA_INCLUDE_DIR}")
target_compile_options(sim_cuda_runtime PUBLIC
  "$<$<COMPILE_LANGUAGE:CUDA>:-I${SIM_CUDA_INCLUDE_DIR}>"
)
target_link_options(sim_cuda_runtime INTERFACE "-L${CUDA_RUNTIME_ROOT}/lib")
target_link_libraries(sim_cuda_runtime PUBLIC CUDA::cudart)
set_target_properties(sim_cuda_runtime PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
)

include(CTest)
if(BUILD_TESTING)
  add_executable(sim_cuda_runtime_smoke tests/runtime_smoke.cu)
  target_link_libraries(sim_cuda_runtime_smoke PRIVATE sim_cuda_runtime)
  add_test(NAME sim_cuda_runtime_smoke COMMAND sim_cuda_runtime_smoke)
  add_executable(sim_cuda_reference_physics_test tests/reference_physics_test.cpp)
  target_link_libraries(sim_cuda_reference_physics_test PRIVATE sim_cuda_runtime)
  add_test(NAME sim_cuda_reference_physics_test
           COMMAND sim_cuda_reference_physics_test)

  add_executable(sim_cuda_physics_batch_test tests/physics_batch_test.cu)
  target_link_libraries(sim_cuda_physics_batch_test PRIVATE sim_cuda_runtime)
  add_test(NAME sim_cuda_physics_batch_test COMMAND sim_cuda_physics_batch_test)
endif()
