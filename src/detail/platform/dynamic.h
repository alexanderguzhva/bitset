#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common.h"

namespace milvus {
namespace bitset {
namespace detail {

//
struct VectorizedDynamic {
    // Fills a bitmask by comparing two arrays element-wise.
    // API requirement: size % 8 == 0
    template<typename T, typename U, CompareType Op>
    static bool op_compare_column(
        uint8_t* const __restrict output, 
        const T* const __restrict t,
        const U* const __restrict u,
        const size_t size
    );

    // Fills a bitmask by comparing elements of a given array to a
    //   given value.
    // API requirement: size % 8 == 0
    template<typename T, CompareType Op>
    static bool op_compare_val(
        uint8_t* const __restrict output,
        const T* const __restrict t,
        const size_t size,
        const T value
    );
};

}
}
}