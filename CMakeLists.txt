cmake_minimum_required(VERSION 3.20)
project(trivial_simd LANGUAGES C)

add_library(trivial_simd SHARED native/trivial_simd.c)
set_target_properties(trivial_simd PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  WINDOWS_EXPORT_ALL_SYMBOLS YES)
if(MSVC)
  target_compile_options(trivial_simd PRIVATE /O2)
else()
  target_compile_options(trivial_simd PRIVATE -O3)
endif()

foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(trivial_simd PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${config} "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_${config} "${PROJECT_BINARY_DIR}")
endforeach()
