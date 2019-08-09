include( template_select )

set( ToolkitLibs Utilities CrossSection Material Particle Geometry Math_Functions Tally IPComm GPU_Utilities ThirdParty)

list( APPEND ${UnitName}_packages MPI )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()
