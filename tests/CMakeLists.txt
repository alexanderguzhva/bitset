include_directories(${CMAKE_HOME_DIRECTORY}/src)

set(BITSET_TEST_SRCS
    test_avx512.cpp
)

find_package(GTest REQUIRED)

add_executable(tests ${BITSET_TEST_SRCS})

target_link_libraries(tests 
    GTest::gtest_main 
    gtest_main 
    pthread 
    bitset
)
