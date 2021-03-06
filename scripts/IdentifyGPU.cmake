function( IdentifyGPU RESULT COMPUTECAPABILITY)

execute_process( COMMAND ${CMAKE_SOURCE_DIR}/scripts/IdentifyGPU.sh
                 OUTPUT_VARIABLE output_value
                 OUTPUT_STRIP_TRAILING_WHITESPACE
               )
               
if( DEFINED output_value) 
    message( STATUS "IndentifyGPU.cmake -- raw GPU type = ${output_value}"  )
else()
    message( ERROR "IndentifyGPU.cmake --No Nvidia GPU was found." )
endif()  

if( output_value STREQUAL "GK107GL" )
  add_definitions(-DK420_GPU)
  set( ${RESULT}  "K420" PARENT_SCOPE )
  set( ${COMPUTECAPABILITY} "-arch=sm_30" PARENT_SCOPE )
  return()
endif() 

# Tesla K40m
if( output_value STREQUAL "GK110BGL" )
  add_definitions(-DTESLA_K40_GPU)
  set( ${RESULT} "TESLA_K40" PARENT_SCOPE )
  set( ${COMPUTECAPABILITY} "-arch=sm_35" PARENT_SCOPE )
  return()
endif()   

if( output_value STREQUAL "GM200" )
  add_definitions(-DTITANX_MAXWELL_GPU)
  set( ${RESULT} "TITANX_MAXWELL" PARENT_SCOPE )
  set( ${COMPUTECAPABILITY} "-arch=sm_52" PARENT_SCOPE )
  return()
endif()   

if( output_value STREQUAL "GP100GL" )
  add_definitions(-DP100_GPU)
  set( ${RESULT} "P100" PARENT_SCOPE )
  set( ${COMPUTECAPABILITY} "--gpu-architecture=compute_60 -code=compute_60,sm_60" PARENT_SCOPE  )
  return()
endif()   

if( output_value STREQUAL "GP102" )
  add_definitions(-DP102_GPU)
  set( ${RESULT} "P102" PARENT_SCOPE )
  set( ${COMPUTECAPABILITY} "-arch=sm_61" PARENT_SCOPE )
  return()
endif()  

if( output_value STREQUAL "GP104" )
  add_definitions(-DP104_GPU)
  set( ${RESULT} "P104" PARENT_SCOPE )
  set( ${COMPUTECAPABILITY} "-arch=sm_61" PARENT_SCOPE )
  return()
endif() 

if( output_value STREQUAL "GV100GL" )
  add_definitions(-DV100_GPU)
  set( ${RESULT} "V100" PARENT_SCOPE )
  set( ${COMPUTECAPABILITY} "--gpu-architecture=compute_70 -code=compute_70,sm_70" PARENT_SCOPE  )
  return()
endif() 

message( WARNING "MonteRay scripts/IndentifyGPU.cmake -- Unknown GPU Type, add your card, to support customized testing." )     

endfunction()
