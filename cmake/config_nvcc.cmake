# prepend -Xcompiler infront of -fexceptions when compiling w/ cuda - may be resolved in future CMake
message(STATUS "config_nvcc.cmake -----------------------------------------")
get_target_property(mpi_compile_options MPI::MPI_CXX INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-fexceptions" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >-fexceptions" 
  mpi_compile_options "${mpi_compile_options}")
set_target_properties(MPI::MPI_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS "${mpi_compile_options}")

unset(cuda_arch_flags)
if(NOT cuda_arch)
  # default to Auto if none is provided
  set(cuda_arch "Auto")
endif()

# Use the findCUDA functions to find the arch but don't use find_package(CUDA)
include( "FindCUDA/select_compute_arch" ) 
cuda_select_nvcc_arch_flags(cuda_arch_flags ${cuda_arch})
message(STATUS "config_nvcc.cmake -- cuda_arch_flags=${cuda_arch_flags}")

list(APPEND CMAKE_CUDA_FLAGS "--relocatable-device-code=true" )
list(APPEND CMAKE_CUDA_FLAGS "--expt-extended-lambda" )
list(APPEND CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr" ) 
list(APPEND CMAKE_CUDA_FLAGS "-cudart static" ) 
list(APPEND CMAKE_CUDA_FLAGS "${cuda_arch_flags}" ) 

string( REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" )

message(STATUS "config_nvcc.cmake -- CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
set_target_properties(MonteRay PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MonteRay PROPERTIES POSITION_INDEPENDENT_CODE ON)
# set_target_properties(MonteRay PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(MonteRay INTERFACE cuda)
message(STATUS "-----------------------------------------------------------")

