# Checks for C++11 features
#  CXX11_FEATURE_LIST - a list containing all supported features
#  HAS_CXX11_AUTO               - auto keyword
#  HAS_CXX11_NULLPTR            - nullptr
#  HAS_CXX11_LAMBDA             - lambdas
#  HAS_CXX11_STATIC_ASSERT      - static_assert()
#  HAS_CXX11_RVALUE_REFERENCES  - rvalue references
#  HAS_CXX11_DECLTYPE           - decltype keyword
#  HAS_CXX11_CSTDINT_H          - cstdint header
#  HAS_CXX11_LONG_LONG          - long long signed & unsigned types
#  HAS_CXX11_VARIADIC_TEMPLATES - variadic templates
#  HAS_CXX11_CONSTEXPR          - constexpr keyword
#  HAS_CXX11_SIZEOF_MEMBER      - sizeof() non-static members
#  HAS_CXX11_FUNC               - __func__ preprocessor constant
#
# Original script by Rolf Eike Beer
# Modifications by Andreas Weis and Steve Nolen
#
function(CXX11_CHECK_FEATURE FEATURE_NAME FEATURE_NUMBER RESULT_VAR)
  if(DEFINED ${RESULT_VAR})
    return()
  endif()

  set( bindir "${CMAKE_CURRENT_BINARY_DIR}/cxx11/cxx11_${FEATURE_NAME}")
  
  if(${FEATURE_NUMBER})
    set( SrcFileBase ${CMAKE_CURRENT_LIST_DIR}/cxx11/c++11-test-${FEATURE_NAME}-N${FEATURE_NUMBER})
    set( LogName "\"${FEATURE_NAME}\" (N${FEATURE_NUMBER})")
  else()
    set( SrcFileBase ${CMAKE_CURRENT_LIST_DIR}/cxx11/c++11-test-${FEATURE_NAME})
    set( LogName "\"${FEATURE_NAME}\"")
  endif()
  
  set( SrcFile "${SrcFileBase}.cpp")
  if(NOT CROSS_COMPILING)
    try_run( RunExitCode CompileSuccessful "${bindir}" "${SrcFile}")

    if( CompileSuccessful AND RunExitCode EQUAL 0 )
      set(${RESULT_VAR} TRUE)
    else()
      set(${RESULT_VAR} FALSE)
    endif()
  else()
    # cross compiles can ONLY check compilation
    try_compile(${RESULT_VAR} "${bindir}" "${SrcFile}")
    if(${RESULT_VAR} AND EXISTS ${SrcFile_FAIL})
      try_compile(${RESULT_VAR} "${bindir}_fail" "${SrcFile_FAIL}")
    endif()
  endif()

  # See if the feature is behaving as expected by checking compile time assertions
  set( SrcFile_FAIL_COMPILE "${SrcFileBase}_fail_compile.cpp")
  if(${RESULT_VAR} AND EXISTS ${SrcFile_FAIL_COMPILE})
    try_compile( CompileSuccessful "${bindir}_fail_compile" "${SrcFile_FAIL_COMPILE}")
    set( ${RESULT_VAR} NOT ${CompileSuccessful} )
  endif()
    
  if(${RESULT_VAR})
    message( STATUS "Checking C++11 support for ${LogName} -- works")
    set_property(GLOBAL APPEND PROPERTY cxx11FeaturesSupported ${RESULT_VAR} )
  else()
    message(STATUS "Checking C++11 support for ${LogName} -- not supported")
  endif()

  # add the result to the cache
  set(${RESULT_VAR} ${${RESULT_VAR}} CACHE INTERNAL "C++11 support for ${LogName}")

endfunction()

set_property(GLOBAL PROPERTY cxx11FeaturesSupported )

CXX11_CHECK_FEATURE("auto"               2546 HAS_CXX11_AUTO)
CXX11_CHECK_FEATURE("nullptr"            2431 HAS_CXX11_NULLPTR)
CXX11_CHECK_FEATURE("lambda"             2927 HAS_CXX11_LAMBDA)
CXX11_CHECK_FEATURE("static_assert"      1720 HAS_CXX11_STATIC_ASSERT)
CXX11_CHECK_FEATURE("rvalue_references"  2118 HAS_CXX11_RVALUE_REFERENCES)
CXX11_CHECK_FEATURE("decltype"           2343 HAS_CXX11_DECLTYPE)
CXX11_CHECK_FEATURE("cstdint"            ""   HAS_CXX11_CSTDINT_H)
CXX11_CHECK_FEATURE("long_long"          1811 HAS_CXX11_LONG_LONG)
CXX11_CHECK_FEATURE("variadic_templates" 2555 HAS_CXX11_VARIADIC_TEMPLATES)
CXX11_CHECK_FEATURE("constexpr"          2235 HAS_CXX11_CONSTEXPR)
CXX11_CHECK_FEATURE("sizeof_member"      2253 HAS_CXX11_SIZEOF_MEMBER)
CXX11_CHECK_FEATURE("__func__"           2340 HAS_CXX11_FUNC)
CXX11_CHECK_FEATURE("initializer_list"   2672 HAS_CXX11_INITIALIZER_LIST )
CXX11_CHECK_FEATURE("unique_ptr"         ""   HAS_CXX11_UNIQUE_PTR )

get_property( CXX11_FEATURE_LIST GLOBAL PROPERTY cxx11FeaturesSupported )
MARK_AS_ADVANCED(FORCE CXX11_FEATURE_LIST)

# To add flags...
#
# set_property(TARGET target-name APPEND PROPERTY COMPILE_DEFINITIONS ${CXX11_FEATURE_LIST} )

if( NOT HAS_CXX11_NULLPTR )
    add_definitions( -Dnullptr=0 )
endif()
if( HAS_CXX11_INITIALIZER_LIST )
    add_definitions( -DHAS_CXX11_INITIALIZER_LIST )
endif()
