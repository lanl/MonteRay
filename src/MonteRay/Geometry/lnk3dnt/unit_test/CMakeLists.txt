include( template_select )

list( APPEND ${UnitName}_packages )

set( ToolkitLibs Utilities lnk3dnt )

set( TestFiles2Link SuperSmall.lnk3dnt 
                    2by2by2_box_lev5.lnk3dnt
                    2by2by2_box.lnk3dnt
                    2by2by2_box_neg_density.lnk3dnt
                    2by2by2_box_neg_matID.lnk3dnt
   )

if( NOT DEFINED Cmake_template )
    message( FATAL_ERROR "Unknown directory type." )
else()
    include( ${Cmake_template} )
endif()

