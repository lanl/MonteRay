include( template_select )

set( ToolkitLibs Utilities CrossSection 
                 Material Geometry lnk3dnt RayTrace 
                 Tally Math_Functions IPComm GPU_Utilities )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()