set( source_file "${SOURCE_MAKE_PATH}/Makefile.in" )
set( target_file "${TARGET_MAKE_PATH}/Makefile" )

configure_file( ${source_file} ${target_file} ESCAPE_QUOTES IMMEDIATE @ONLY )