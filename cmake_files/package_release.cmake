# Handle a formal release
#
#  ) check for existence of current release - 
#      True - terminate and prompt to remove current or change version number
#  ) 'svn copy' the current trunk into a new dir
#  ) clean out any previous builds
#  ) build 2 or 3 variants (i.e. gnu and intel)
#  ) generate documentation for toolkit and API
#  ) install under project directory (optional: create soft link to/from appropriate r#### )
#  ) export and package tarball
#  ) sync repository on xcp lan and at livermore

#string(REGEX REPLACE "^(.*\n)?Repository Root: ([^\n]+).*"
#       "\\2" SvnRoot "${mcatk_WC_INFO}" )
set( SvnRoot ${mcatk_WC_ROOT} )
       
if( UNIX )
add_custom_target( archive_minor
                   COMMAND ${Subversion_SVN_EXECUTABLE} export ${SvnRoot}/mcatk/trunk ${ReleaseName}
                   COMMAND tar cfz mcatk_${ReleaseName}.tgz ${ReleaseName}
                   COMMAND push mcatk_${ReleaseName}.tgz 
                   COMMAND rm -rf ${ReleaseName} mcatk_${ReleaseName}.tgz 
                   DEPENDS mcatk
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMENT "Archiving mcatk_${ReleaseName}.tgz"
                   VERBATIM
                 )
add_custom_target( archive_major
                   COMMAND ${Subversion_SVN_EXECUTABLE} cp ${SvnRoot}/mcatk/trunk ${SvnRoot}/release/${ToolkitVersion} -m "RELEASE - ${ToolkitVersion}"
                   COMMAND ${Subversion_SVN_EXECUTABLE} export ${SvnRoot}/release/${ToolkitVersion}
                   COMMAND tar cfz mcatk_${ToolkitVersion}.tgz ${ToolkitVersion}
                   COMMAND push mcatk_${ToolkitVersion}.tgz 
                   COMMAND rm -rf ${ToolkitVersion} mcatk_${ToolkitVersion}.tgz 
                   DEPENDS mcatk
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMENT "Tagging repository and pushing mcatk_${ToolkitVersion}.tgz"
                   VERBATIM
                 )
                 
endif()
