include_directories(${CMAKE_HOME_DIRECTORY}/src)

set(BITSET_TEST_SRCS
    test_functionality.cpp
)

# set_source_files_properties(test_functionality.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mavx -mfma")

find_package(GTest REQUIRED)

add_executable(tests ${BITSET_TEST_SRCS})

target_link_libraries(tests 
    GTest::gtest_main 
    pthread 
    bitset
)
