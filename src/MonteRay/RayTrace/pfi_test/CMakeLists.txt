include( template_select )

list( APPEND ${UnitName}_packages MPI )

set( ToolkitLibs Tally ThirdParty Utilities OutputControls CrossSection Material Geometry Particle Math_Functions IPComm GPU_Utilities )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

