set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )

string( REGEX REPLACE "\\." ";" CXX_VERSION ${CMAKE_CXX_COMPILER_VERSION} ) 
list( GET CXX_VERSION 0 Clang_MAJOR_VERSION ) 
list( GET CXX_VERSION 1 Clang_MINOR_VERSION ) 
list( GET CXX_VERSION 2 Clang_REVISION_VERSION ) 
message(STATUS "Clang version is ${Clang_MAJOR_VERSION}.${Clang_MINOR_VERSION}.${Clang_REVISION_VERSION}" )

#Needed for install releases. Has to be done in each indivdual toolchain.
set(compiler_install_prefix "Clang-${Clang_MAJOR_VERSION}.${Clang_MINOR_VERSION}" CACHE STRING "Compiler Name Prefix used in naming the install directory")
message(STATUS "Compiler Install Prefix is [ ${compiler_install_prefix} ]" )

find_package( Threads )

# ================================
# Sanitizers
# http://clang.llvm.org/docs/UsersManual.html
#
# Enables various analysis tools available under LLVM. (NOTE: Some are x86_64 ONLY)
# leak    : memory leak analysis
# address : memory access violations AND memory leak analysis
#           to disable setenv ASAN_OPTIONS detect_leaks=0
# memory  : Enables the unitialized memory usage analysis
# thread  : Enables thread checking
if( CLANG_ANALYZE )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=${CLANG_ANALYZE}" )
#    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=${CLANG_ANALYZE} -fsanitize-blacklist=${CMAKE_SOURCE_DIR}/cmake_files/Unix/sanitizer_suppress.txt" )
    if( CLANG_ANALYZE STREQUAL "memory" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -pie -fno-omit-frame-pointer -fsanitize-memory-track-origins=2" )
#        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize-recover=memory -fPIE -pie -fno-omit-frame-pointer -fsanitize-memory-track-origins=2" )
    endif()
endif()
