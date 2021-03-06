# Cmake 3.14 is needed for proper CUDA + MPI support
CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
message( STATUS " -- CMakeList.txt -- Configuring MonteRay ")

if(DEFINED PROJECT_NAME)
  set(subproject ON)
else()
  set(subproject OFF)
endif()

PROJECT(MonteRay
  VERSION 0.1.0
  DESCRIPTION "A library for computing a ray casting estimator"
  HOMEPAGE_URL "https://xcp-stash.lanl.gov/projects/MR/repos/monteray"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.cmake")

add_library(MonteRay STATIC)
target_compile_features(MonteRay PUBLIC cxx_std_14)

find_package(mpark_variant)
target_link_libraries(MonteRay PUBLIC mpark_variant)

if( enable_cuda )
  message( "Base CmakeLists.txt - enabling CUDA." ) 
  message( STATUS "-- Building MonteRay with CUDA.")
  message( STATUS "-- Loading config_nvcc.cmake")

  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message( FATAL_ERROR "Requested to build with CUDA, but CMake could not find a CUDA compiler.")
  endif()
  set_target_properties(MonteRay PROPERTIES LINKER_LANGUAGE CUDA) 
else()
  message( STATUS "-- Building without CUDA.")
  # TODO: if compiler is GNU do this
  target_compile_options(MonteRay INTERFACE "-xc++")
  set_target_properties(MonteRay PROPERTIES LINKER_LANGUAGE CXX) 
endif()

set_target_properties(MonteRay PROPERTIES POSITION_INDEPENDENT_CODE ON)

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(CTest)
CMAKE_DEPENDENT_OPTION(MonteRay.tests
  "Build the MonteRay tests and integrate with ctest" ON 
  "BUILD_TESTING; NOT subproject" OFF)

find_package(MPI REQUIRED COMPONENTS CXX) # TODO: MPI_C
target_link_libraries(MonteRay PUBLIC MPI::MPI_CXX)


if(MonteRay.tests)
  include(Warnings)
  include(Sanitizers)

  set(CMAKE_CXX_EXTENSIONS OFF)
  add_library(MonteRay::testing IMPORTED INTERFACE)

  add_library(UnitTest++ STATIC IMPORTED)
  set_target_properties(UnitTest++ PROPERTIES IMPORTED_LOCATION $ENV{UNITTEST_ROOT}/lib/libUnitTest++.a)
  #  target_include_directories(UnitTest++ INTERFACE $ENV{UNITTEST_ROOT}/include/UnitTest++) # UnitTest version 1.61
  target_include_directories(UnitTest++ INTERFACE $ENV{UNITTEST_ROOT}/include)# UnitTest version 1.4
  target_link_libraries(MonteRay::testing INTERFACE UnitTest++)

  set_target_properties(MonteRay::testing PROPERTIES
    INTERFACE_WARN_ALL ON
    INTERFACE_WARN_ERROR ON
    INTERFACE_LLVM_DISABLED_WARNINGS "missing-braces")

  if(enable_cuda)
    set_property(TARGET MonteRay::testing APPEND PROPERTY INTERFACE_GNU_DISABLED_WARNINGS "old-style-cast" "pedantic" "suggest-override" "subobject-linkage" "format")
  endif()

  set_property(TARGET MonteRay::testing APPEND PROPERTY 
    COMPATIBLE_INTERFACE_BOOL
    WARN_ALL WARN_ERROR
    )

  set_property(TARGET MonteRay::testing APPEND PROPERTY COMPATIBLE_INTERFACE_STRING
    GNU_DISABLED_WARNINGS
    CUDA_DISABLED_WARNINGS
    LLVM_DISABLED_WARNINGS)

  target_link_libraries(MonteRay::testing INTERFACE 
    MonteRay
    # shacl::cmake::Warnings_CXX
    # shacl::cmake::Sanitizers_CXX
  )

endif()

if(CMAKE_INSTALL_LIBDIR)
else()
  set(CMAKE_INSTALL_LIBDIR lib)
endif()

if(CMAKE_INSTALL_BINDIR)
else()
  set(CMAKE_INSTALL_BINDIR bin)
endif()

if (CMAKE_INSTALL_INCLUDEDIR)
else()
  set(CMAKE_INSTALL_INCLUDEDIR include)
endif()

string(CONCAT prefix
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_include_directories(MonteRay PUBLIC "${prefix}")

install(DIRECTORY src/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.hh"
  PATTERN "*test*" EXCLUDE
)

install(FILES
  "${PROJECT_SOURCE_DIR}/.cmake/shacl-config.cmake"
  DESTINATION share/cmake/monteray
)

# use a shared library to force device code linking to occur 
# otherwise host code will need to use NVCC during the linking stage
add_library(monteray SHARED)
target_link_libraries(monteray PUBLIC MonteRay)

install(TARGETS monteray EXPORT monteray DESTINATION ${MonteRayINSTALL_DIR} )
install(TARGETS monteray MonteRay EXPORT monterayTargets)
install(EXPORT monterayTargets
   FILE monteray-targets.cmake
   DESTINATION share/cmake/monteray
)
write_basic_package_version_file("monteray-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  "${PROJECT_SOURCE_DIR}/cmake/monteray-config.cmake"
  "${PROJECT_BINARY_DIR}/monteray-config-version.cmake"
  DESTINATION share/cmake/monteray
)

if(NOT subproject)
  include(InstallRequiredSystemLibraries)
  set(CPACK_PACKAGE_VENDOR "Los Alamos National Laboratory")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
  set(CPACK_PACKAGE_CONTACT "jsweezy@lanl.gov")
  include(CPack)
endif()

# add_subdirectory( apps )

include(cmake/FindMonteRayTestFiles.cmake)
# If not using CUDA, make all CUDA files C++ files
include(cmake/config_tests.cmake)
if(enable_cuda)
  include(cmake/config_nvcc.cmake)
else()
  file(GLOB_RECURSE cuda_srcs "*.cu")
  foreach(file ${cuda_srcs})
    set_source_files_properties( ${file} PROPERTIES LANGUAGE "CXX")
    set_source_files_properties( ${file} PROPERTIES COMPILE_FLAGS "-xc++")
  endforeach()
endif()

# add_subdirectory( apps )
add_subdirectory( src )
