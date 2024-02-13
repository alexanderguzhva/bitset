set(BITSET_SRCS
    detail/platform/dynamic.cpp
)

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    list(APPEND BITSET_SRCS
        detail/platform/x86/avx2.cpp
        detail/platform/x86/avx512.cpp
        detail/platform/x86/instruction_set.cpp
    )

    set_source_files_properties(detail/platform/x86/avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl")
    set_source_files_properties(detail/platform/x86/avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mavx -mfma")
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm*")
    list(APPEND BITSET_SRCS
        detail/platform/arm/neon.cpp
    )
endif()

add_library(bitset ${BITSET_SRCS})
