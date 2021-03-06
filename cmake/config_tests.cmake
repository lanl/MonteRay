macro(setCudaFilesToCppFiles)
  if(enable_cuda)
  else()
    file(GLOB_RECURSE cuda_srcs "*.cu")
    foreach(file ${cuda_srcs})
      set_source_files_properties( ${file} PROPERTIES LANGUAGE "CXX")
      set_source_files_properties( ${file} PROPERTIES COMPILE_FLAGS "-xc++")
    endforeach()
  endif()
endmacro()
