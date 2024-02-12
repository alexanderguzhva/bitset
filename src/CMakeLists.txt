set(BITSET_SRCS
    avx512.cpp
    detail/platform/dynamic.cpp
    detail/platform/x86/avx2.cpp
    detail/platform/x86/instruction_set.cpp
)

set_source_files_properties(avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl")
set_source_files_properties(detail/platform/x86/avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mavx -mfma")

add_library(bitset ${BITSET_SRCS})
