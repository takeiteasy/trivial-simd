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
  target_link_libraries(trivial_simd PRIVATE m)
endif()

foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(trivial_simd PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${config} "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_${config} "${PROJECT_BINARY_DIR}")
endforeach()

include(CTest)
if(BUILD_TESTING)
  foreach(target native_kernel_test native_kernel_scalar_test)
    add_executable(${target} tests/native-kernel.c)
    set_target_properties(${target} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES)
    if(NOT MSVC)
      target_link_libraries(${target} PRIVATE m)
    endif()
  endforeach()
  target_compile_definitions(native_kernel_scalar_test PRIVATE TS_SCALAR)
  add_test(NAME native-kernel COMMAND native_kernel_test)
  add_test(NAME native-kernel-scalar COMMAND native_kernel_scalar_test)
endif()
