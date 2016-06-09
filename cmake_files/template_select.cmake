get_filename_component( current_dir ${CMAKE_CURRENT_SOURCE_DIR} NAME )

list( FIND TestDirNames ${current_dir} Index )

if( Index EQUAL -1 )
    set( UnitName ${current_dir} )
    unset( ${UnitName}_packages )
    unset( ${UnitName}_includes )
    set( Cmake_template library_template )
else()
    # Should be using value for UnitName inherited from parent directory
    if( NOT DEFINED UnitName )
        message( FATAL_ERROR "Unable to determine generic name for directives" )
    endif()
    set( Cmake_template common_test_template )
endif()


