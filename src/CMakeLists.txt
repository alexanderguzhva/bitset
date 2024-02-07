set(BITSET_SRCS
    avx512.cpp
)

set_source_files_properties(avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl")

add_library(bitset ${BITSET_SRCS})
