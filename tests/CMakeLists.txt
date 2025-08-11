include_directories(${CMAKE_HOME_DIRECTORY}/src)

set(BITSET_TEST_SRCS
    test_functionality.cpp
)

# set_source_files_properties(test_functionality.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq")
# set_source_files_properties(test_functionality.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mavx -mfma")

# set_source_files_properties(test_functionality.cpp PROPERTIES COMPILE_FLAGS "-mcpu=neoverse-v1")

find_package(GTest REQUIRED)

include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)

check_cxx_compiler_flag("-march=armv8-a+sve" COMPILER_SUPPORTS_SVE)
check_include_file_cxx("arm_sve.h" COMPILER_HAS_ARM_SVE_HEADER)

if(COMPILER_SUPPORTS_SVE)
    message(STATUS "Compiler supports SVE. Adding SVE compile options")
else()
    message(STATUS "Compiler does NOT support SVE")
endif()

if(COMPILER_HAS_ARM_SVE_HEADER)
    message(STATUS "Compiler has arm_sve.h")
else()
    message(STATUS "Compiler has no arm_sve.h")
endif()

add_executable(tests)

if (COMPILER_SUPPORTS_SVE AND COMPILER_HAS_ARM_SVE_HEADER)
    message(STATUS "SVE support for the library is enabled")
    target_compile_definitions(tests PRIVATE BITSET_ENABLE_SVE_SUPPORT=1)

    set_source_files_properties(test_functionality.cpp PROPERTIES COMPILE_FLAGS "-march=native")
else()
    message(STATUS "SVE support for the library is disabled")
endif()

target_sources(tests PRIVATE ${BITSET_TEST_SRCS})

target_link_libraries(tests 
    GTest::gtest_main 
    pthread 
    bitset
)
