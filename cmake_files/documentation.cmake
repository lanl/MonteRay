# - Run Doxygen
#
# Adds a doxygen target that runs doxygen to generate the html
# and optionally the LaTeX API documentation.
# The doxygen target is added to the doc target as dependency.
# i.e.: the API documentation is built with:
#  make doc
#
# USAGE: GLOBAL INSTALL
#
# Install it with:
#  cmake ./ && sudo make install
# Add the following to the CMakeLists.txt of your project:
#  include(UseDoxygen OPTIONAL)
# Optionally copy Doxyfile.in in the directory of CMakeLists.txt and edit it.
#
# USAGE: INCLUDE IN PROJECT
#
#  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
#  include(UseDoxygen)
# Add the Doxyfile.in and UseDoxygen.cmake files to the projects source directory.
#
#
# Variables you may define are:
#  DOXYFILE_OUTPUT_DIR - Path where the Doxygen output is stored. Defaults to "doc".
#
#  DOXYFILE_LATEX_DIR - Directory where the Doxygen LaTeX output is stored. Defaults to "latex".
#
#  DOXYFILE_HTML_DIR - Directory where the Doxygen html output is stored. Defaults to "html".
#

#
#  Copyright (c) 2009, 2010 Tobias Rautenkranz <tobias@rautenkranz.ch>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

# Create a target called "doc_name"_docs that generates the doxygen documentation including the pdf version
function( generate_doc_rule doc_name file.in directory )

    find_package( Doxygen )
    
    if( NOT DOXYGEN_FOUND )
        # Doxygen was NOT found so don't do anything
        return()
    endif()

    set( template_search_path "${CMAKE_SOURCE_DIR}/${directory}" )

    find_file( DOXYFILE.${doc_name} 
               NAMES ${file.in}
               PATHS ${template_search_path}
               NO_DEFAULT_PATH )
               
    if( NOT DOXYFILE.${doc_name} )
        message( FATAL_ERROR "File: ${file.in} was not found.  Searching in ${template_search_path}." )
    endif()
    

    set(DOXYFILE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/${directory}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYFILE_OUTPUT_DIR} )
    
    set( DocumentDepends ${CMAKE_BINARY_DIR}/docs/html/index.html )

    add_custom_command( OUTPUT ${DocumentDepends} 
                        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
                        COMMAND chmod -R a+rX  ${DOXYFILE_OUTPUT_DIR}
                        COMMAND chmod -R g+w   ${DOXYFILE_OUTPUT_DIR}
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                      )
    add_custom_target( createDocs DEPENDS ${CMAKE_BINARY_DIR}/docs/html/index.html )
     
    # Create HTML output
    set(DOXYFILE_HTML_DIR "html")

    # Add the html directory to the list of those to delete during a 'clean'
    set_property(DIRECTORY APPEND PROPERTY
                 ADDITIONAL_MAKE_CLEAN_FILES "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_HTML_DIR}")

    # Turn off latex output unless we can find the needed programs
    set(DOXYFILE_LATEX "YES")
    set(DOXYFILE_PDFLATEX "YES")
    
    set(DOXYFILE_DOT "NO")
    if( DOXYGEN_DOT_FOUND )
        set( DOXYFILE_DOT "YES" )
    endif()

     # FindLATEX.cmake is pretty rudimentary -- we're going to do our own for now
 #    find_package(LATEX)
    set( LatexBinPaths /usr/lanl/bin ${package_dir}/bin /usr/bin )
    find_program( LATEX_COMPILER
                  NAMES latex
                  PATHS ${LatexBinPaths} NO_DEFAULT_PATH
     )
     get_filename_component( LatexPath ${LATEX_COMPILER} PATH )
     
     find_program( PDFLATEX_COMPILER
                   NAMES pdflatex
                   PATHS ${LatexBinPaths} NO_DEFAULT_PATH
     )
     find_program(MAKEINDEX_COMPILER
                  NAMES makeindex
                  PATHS ${LatexBinPaths} NO_DEFAULT_PATH
     )
 
    if(LATEX_COMPILER AND MAKEINDEX_COMPILER )
        set(DOXYFILE_LATEX "YES")
        set(DOXYFILE_LATEX_DIR "latex")

        set_property( DIRECTORY APPEND PROPERTY
                      ADDITIONAL_MAKE_CLEAN_FILES
                      "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}" )

        if(PDFLATEX_COMPILER)
            set(DOXYFILE_PDFLATEX "YES")
        endif()
        if(DOXYGEN_DOT_EXECUTABLE)
            set(DOXYFILE_DOT "YES")
        endif()

        # Add ability to invoke doxygen's make system for building latex/pdf stuff
        set( PDFManual ${CMAKE_BINARY_DIR}/Docs/latex/refman.pdf )
        add_custom_command( OUTPUT ${PDFManual}
                            COMMAND ${CMAKE_MAKE_PROGRAM} PATH=${LatexPath}:$ENV{PATH}
                            COMMAND chmod -R a+rX  ${DOXYFILE_OUTPUT_DIR}
                            COMMAND chmod -R g+w   ${DOXYFILE_OUTPUT_DIR}
                            DEPENDS ${DocumentDepends}
                            WORKING_DIRECTORY "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}"
                          )
        add_custom_target( manual_generator DEPENDS ${PDFManual} )                  
        add_dependencies( createDocs manual_generator )
        set( DocumentDepends ${DocumentDepends} ${PDFManual} )

    endif()


    configure_file( ${DOXYFILE.${doc_name}} Doxyfile ESCAPE_QUOTES IMMEDIATE @ONLY )

    # Create a target for building installing the documentation manually.
    #get_target_property( DOC_TARGET install_docs TYPE )
    if(NOT TARGET install_docs )
       add_custom_target( install_docs 
                          COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOXYFILE_OUTPUT_DIR} ${CMAKE_INSTALL_PREFIX}/docs
                          COMMENT "Installing documentation for ${doc_name} in ${CMAKE_INSTALL_PREFIX}/docs"
                          DEPENDS ${DocumentDepends}
                        ) 
    endif()
   
    set( DocDependencies ${DocumentDepends} PARENT_SCOPE )
endfunction()
