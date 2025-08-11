set(BITSET_SRCS
    detail/platform/dynamic.cpp
)

add_library(bitset)

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    list(APPEND BITSET_SRCS
        detail/platform/x86/avx2-inst.cpp
        detail/platform/x86/avx512-inst.cpp
        detail/platform/x86/instruction_set.cpp
    )

    set_source_files_properties(detail/platform/x86/avx512-inst.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd -mbmi")
    set_source_files_properties(detail/platform/x86/avx2-inst.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mavx -mfma -mbmi")

    # set_source_files_properties(detail/platform/dynamic.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq")
    # set_source_files_properties(detail/platform/dynamic.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mavx -mfma")
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm*")
    list(APPEND BITSET_SRCS
        detail/platform/arm/neon-inst.cpp
        detail/platform/arm/sve-inst.cpp
        detail/platform/arm/instruction_set.cpp
    )

    # Perform a test for the ARM SVE command set.
    # Sometimes, a compiler may support it, but there are no headers available.
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

    # targeting AWS graviton, 
    #  https://github.com/aws/aws-graviton-getting-started/blob/main/c-c%2B%2B.md

    if (COMPILER_SUPPORTS_SVE AND COMPILER_HAS_ARM_SVE_HEADER)
        message(STATUS "SVE support for the library is enabled")
        target_compile_definitions(bitset PRIVATE BITSET_ENABLE_SVE_SUPPORT=1)

        set_source_files_properties(detail/platform/arm/sve-inst.cpp PROPERTIES COMPILE_FLAGS "-march=native")
    endif()
endif()

target_sources(bitset PRIVATE ${BITSET_SRCS})
