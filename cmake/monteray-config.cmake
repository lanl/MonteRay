include(CMakeFindDependencyMacro)

find_dependency(MPI COMPONENTS CXX)
find_dependency(mpark_variant)

get_target_property(mpi_compile_options MPI::MPI_CXX INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-fexceptions" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >-fexceptions" 
  mpi_compile_options "${mpi_compile_options}")
set_target_properties(MPI::MPI_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS "${mpi_compile_options}")

include("${CMAKE_CURRENT_LIST_DIR}/monteray-targets.cmake")
