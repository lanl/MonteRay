include( template_select )

set( ToolkitLibs Utilities CrossSection Material lnk3dnt GPU_Utilities RayTrace Geometry )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()