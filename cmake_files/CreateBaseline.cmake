file( STRINGS @testName@.out lines NEWLINE_CONSUME )

#string( REGEX REPLACE "^Timing*" "" lines ${lines} )

file( WRITE @CMAKE_CURRENT_SOURCE_DIR@/Baseline ${lines} )
