cmake_minimum_required(VERSION 3.0)

project(bitset CXX)

add_compile_options(-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -fsanitize=address -g)
add_link_options(-fno-stack-protector -fno-omit-frame-pointer -fno-var-tracking -fsanitize=address -g)

add_subdirectory(src)
add_subdirectory(tests)