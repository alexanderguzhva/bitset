cmake_minimum_required(VERSION 3.0)

project(bitset CXX)

set (CMAKE_CXX_STANDARD 17)

# add_compile_options(-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -fsanitize=address -g)
# add_link_options(-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -fsanitize=address -g)

# add_compile_options(-march=native -g)

add_subdirectory(src)
add_subdirectory(tests)