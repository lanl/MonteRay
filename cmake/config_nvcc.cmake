# prepend -Xcompiler infront of -pthread when compiling w/ cuda
get_target_property(mpi_compile_options MPI::MPI_CXX INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-pthread" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >-pthread" 
  mpi_compile_options "${mpi_compile_options}")
set_target_properties(MPI::MPI_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS ${mpi_compile_options})

if(cuda_arch)
else()
  set(cuda_arch 52) # default to 52 if not specified
endif()
set(cuda_compute "compute_${cuda_arch}")
set(cuda_code "sm_${cuda_arch}")
list(APPEND CMAKE_CUDA_FLAGS "--relocatable-device-code=true" )
list(APPEND CMAKE_CUDA_FLAGS "--expt-extended-lambda" )
list(APPEND CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr" ) 
list(APPEND CMAKE_CUDA_FLAGS "-cudart static -gencode arch=${cuda_compute},code=${cuda_code}")
string( REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" )

set_target_properties(MonteRay PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MonteRay PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(MonteRay INTERFACE cuda)


