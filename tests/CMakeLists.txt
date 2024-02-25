include_directories(${CMAKE_HOME_DIRECTORY}/src)

set(BITSET_TEST_SRCS
    test_functionality.cpp
)

find_package(GTest REQUIRED)

add_executable(tests ${BITSET_TEST_SRCS})

target_link_libraries(tests 
    GTest::gtest_main 
    pthread 
    bitset
)
