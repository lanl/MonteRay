#
# Write a mcatk.sh and mcatk.csh file that:
#
#   defines MCATK_COMPILER_NAME
#
#   defines MCATK_SYSTEM_NAME
#
#   defines MCATK_CONFIG_FILEPATH
#
#   defines MCATK_ROOT
#
#   adds the ${CMAKE_INSTALL_PREFIX}/bin/${MCATK_SYSTEM}/${MCATK_COMPILER} to the PATH
#
#   adds the ${CMAKE_INSTALL_PREFIX}/lib/${MCATK_SYSTEM}/${MCATK_COMPILER} to the {DY}LD_LIBRARY_PATH
#
#

if(NOT UNIX)
    return()
endif()


set(SYSTEM_LIB_PATH LD_LIBRARY_PATH)
if(APPLE)
    set(SYSTEM_LIB_PATH DYLD_LIBRARY_PATH)
endif()

#------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------#
function(configure_for_sh)

    set(SHELL_SPECIFICATION "#!/bin/sh" PARENT_SCOPE)
    
    set(SCRIPT_NAME "mcatk.sh")
    
    set(SELF_LOCATE "
#-----------------------------------------------------------------------
# Locate directory of self
#
# Self locate script when sourced
if [ -z \"\$BASH_VERSION\" ]; then
    # Not bash, so rely on sourcing from correct location
    if [ ! -f ${SCRIPT_NAME} ]; then
        echo 'ERROR: ${SCRIPT_NAME} could NOT self-locate MCATK installation'
        echo 'This is most likely because you are using ksh, zsh or similar'
        echo 'To fix this issue, cd to the directory containing this script'
        echo 'and source it in that directory.'
        return 1
    fi
    scriptloc=\$\(pwd\)
else
    scriptloc=\$\(cd `dirname \"\${BASH_SOURCE[0]}\"` && pwd\)
fi

scriptloc=\"`cd \$scriptloc > /dev/null ; pwd`\" 

    "
    PARENT_SCOPE)
    
    
    set(SET_COMPILER_NAME "
export MCATK_COMPILER_NAME=\"\$\(basename \$scriptloc\)\"
    "
    PARENT_SCOPE)
    
    set(SET_SYSTEM_NAME "
export MCATK_SYSTEM_NAME=\"\$\(basename \$\(dirname $scriptloc\)\)\"
    "
    PARENT_SCOPE)
    
    set(SET_ROOT "
export MCATK_ROOT=\"\$\(dirname \$\(dirname \$\(dirname \$scriptloc\)\)\)\"
    "
    PARENT_SCOPE)
    
    set(SET_CONFIG_FILEPATH "
export MCATK_CONFIG_FILEPATH=\"\$scriptloc/mcatk-config\"
    "
    PARENT_SCOPE)
        
    set(ADD_TO_PATH "
if test \"x\$PATH\" = \"x\" ; then
  export PATH=\"\${MCATK_ROOT}/bin/\${MCATK_SYSTEM_NAME}/\${MCATK_COMPILER_NAME}\"
else
  export PATH=\"\${MCATK_ROOT}/bin/\${MCATK_SYSTEM_NAME}/\${MCATK_COMPILER_NAME}\":\${PATH}
fi
    "
    PARENT_SCOPE)
    
    set(ADD_TO_LIBRARY_PATH "
if test \"x\$${SYSTEM_LIB_PATH}\" = \"x\" ; then
  export ${SYSTEM_LIB_PATH}=\"\${MCATK_ROOT}/lib/\${MCATK_SYSTEM_NAME}/\${MCATK_COMPILER_NAME}\"
else
  export ${SYSTEM_LIB_PATH}=\"\${MCATK_ROOT}/lib/\${MCATK_SYSTEM_NAME}/\${MCATK_COMPILER_NAME}\":\${${SYSTEM_LIB_PATH}}
fi
    "
    PARENT_SCOPE)
        
endfunction(configure_for_sh)

#------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------#
function(configure_for_csh )

    set(SHELL_SPECIFICATION "#!/bin/csh" PARENT_SCOPE)
    
    set(SCRIPT_NAME "mcatk.csh")
    
    set(SELF_LOCATE "
#-----------------------------------------------------------------------
# Locate directory of self
#
# Self locate script when sourced
# If sourced interactively, we can use $_ as this should be
#
#   source path_to_script_dir/${SCRIPT_NAME}
#
unset scriptloc

set ARGS=($_)
if (\"$ARGS\" != \"\") then
    if (\"$ARGS[2]\" =~ */${SCRIPT_NAME}) then
        set scriptloc=\"`dirname \${ARGS[2]}`\"
    endif
endif

if (! \$?scriptloc) then
    # We were sourced non-interactively. This means that $_
    # won't be set, so we need an external source of information on
    # where the script is located.
    # We obtain this in one of two ways:
    #   1) Current directory:
    #     cd script_dir ; source ${SCRIPT_NAME}
    #
    #   2) Supply the directory as an argument to the script:
    #     source script_dir/${SCRIPT_NAME} script_dir
    #
    if ( -e ${SCRIPT_NAME} ) then
        set scriptloc=\"`pwd`\"
    else if ( \"\$1\" != \"\" )  then
        if ( -e \${1}/${SCRIPT_NAME} ) then
            set scriptloc=\${1}
        else
            echo \"ERROR \${1} does not contain a MCATK installation\"
        endif
    endif
endif

if (! \$?scriptloc) then
    echo \"ERROR: ${SCRIPT_NAME} could NOT self-locate MCATK installation\"
    echo \"because it was sourced (i.e. embedded) in another script.\"
    echo \"This is due to limitations of (t)csh but can be worked around by providing\"
    echo \"the directory where ${SCRIPT_NAME} is located\"
    echo \"to it, either via cd-ing to the directory before sourcing:\"
    echo \"  cd where_script_is ; source ${SCRIPT_NAME}\"
    echo \"or by supplying the directory as an argument to the script:\"
    echo \"  source where_script_is/${SCRIPT_NAME} where_script_is\"
    echo \" \"
    exit 1
endif

set scriptloc=\"`cd \$scriptloc > /dev/null ; pwd`\" 
    "
    PARENT_SCOPE)
    
    
    set(SET_COMPILER_NAME "
setenv MCATK_COMPILER_NAME `basename \$scriptloc`
    "
    PARENT_SCOPE)
    
    set(SET_SYSTEM_NAME "
set d1=`dirname \$scriptloc`
setenv MCATK_SYSTEM_NAME `basename \$d1`
unset d1
    "
    PARENT_SCOPE)

    set(SET_ROOT "
set d1=`dirname \$scriptloc`
set d2=`dirname \$d1`
setenv MCATK_ROOT `dirname \$d2`
unset d1
unset d2
    "
    PARENT_SCOPE)
    
    set(SET_CONFIG_FILEPATH "
setenv MCATK_CONFIG_FILEPATH \$scriptloc/mcatk-config
    "
    PARENT_SCOPE)
      
      
    set(ADD_TO_PATH "
if ( ! \${?PATH} ) then
  setenv PATH \"\${MCATK_ROOT}/bin/\${MCATK_SYSTEM_NAME}/\${MCATK_COMPILER_NAME}\"
else
  setenv PATH \"\${MCATK_ROOT}/bin/\${MCATK_SYSTEM_NAME}/\${MCATK_COMPILER_NAME}\":\${PATH}
endif
    "
    PARENT_SCOPE)
    
    set(ADD_TO_LIBRARY_PATH "
if ( ! \${?${SYSTEM_LIB_PATH}} ) then
  setenv ${SYSTEM_LIB_PATH} \"\${MCATK_ROOT}/lib/\${MCATK_SYSTEM_NAME}/\${MCATK_COMPILER_NAME}\"
else
  setenv ${SYSTEM_LIB_PATH} \"\${MCATK_ROOT}/lib/\${MCATK_SYSTEM_NAME}/\${MCATK_COMPILER_NAME}\":\${${SYSTEM_LIB_PATH}}
endif
    "
    PARENT_SCOPE)  
endfunction()

#------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------#

configure_for_sh()
configure_file(${CMAKE_SOURCE_DIR}/cmake_files/templates/env.in
               ${CMAKE_BINARY_DIR}/InstallTreeFiles/mcatk.sh
               @ONLY
)

#------------------------------------------------------------------------------#
#
#
#
#------------------------------------------------------------------------------#

configure_for_csh()
configure_file(${CMAKE_SOURCE_DIR}/cmake_files/templates/env.in
               ${CMAKE_BINARY_DIR}/InstallTreeFiles/mcatk.csh
               @ONLY
)

install(FILES
            ${CMAKE_BINARY_DIR}/InstallTreeFiles/mcatk.sh
            ${CMAKE_BINARY_DIR}/InstallTreeFiles/mcatk.csh
        DESTINATION 
            ${CMAKE_INSTALL_PREFIX}/${binary_install_prefix}
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE            
)


