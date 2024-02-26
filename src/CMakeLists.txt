set(BITSET_SRCS
    detail/platform/dynamic.cpp
)

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    list(APPEND BITSET_SRCS
        detail/platform/x86/avx2-inst.cpp
        detail/platform/x86/avx512-inst.cpp
        detail/platform/x86/instruction_set.cpp
    )

    set_source_files_properties(detail/platform/x86/avx512-inst.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq")
    set_source_files_properties(detail/platform/x86/avx2-inst.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mavx -mfma")
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm*")
    list(APPEND BITSET_SRCS
        detail/platform/arm/neon.cpp
        detail/platform/arm/sve.cpp
    )

    # let dynamic.cpp know that SVE is available
    # comment it out for now
    # set_source_files_properties(detail/platform/dynamic.cpp PROPERTIES COMPILE_FLAGS "-mcpu=neoverse-512tvb")

    set_source_files_properties(detail/platform/arm/sve.cpp PROPERTIES COMPILE_FLAGS "-mcpu=neoverse-512tvb")
endif()

add_library(bitset ${BITSET_SRCS})
