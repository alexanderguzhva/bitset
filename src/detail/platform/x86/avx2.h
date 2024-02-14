#pragma once

#include <cstddef>
#include <cstdint>

#include "common.h"

namespace milvus {
namespace bitset {
namespace detail {
namespace x86 {

void
AndAVX2(void* const left, const void* const right, const size_t size);

void
OrAVX2(void* const left, const void* const right, const size_t size);

//
template <typename T, CompareType Op>
void CompareValAVX2(
    const T* const __restrict src, 
    const size_t size, 
    const T val, 
    uint8_t* const __restrict res
);

//
template <typename T, CompareType Op>
void CompareColumnAVX2(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    uint8_t* const __restrict res
);

//
template<typename T, RangeType Op>
void WithinRangeAVX2(
    const T* const __restrict lower, 
    const T* const __restrict upper, 
    const T* const __restrict values, 
    const size_t size, 
    uint8_t* const __restrict res
);

//
struct VectorizedAvx2 {
    // API requirement: size % 8 == 0
    template<typename T, typename U, CompareType Op>
    static inline bool op_compare_column(
        uint8_t* const __restrict output, 
        const T* const __restrict t,
        const U* const __restrict u,
        const size_t size
    ) {
        // same data types for both t and u?
        if constexpr(std::is_same_v<T, U>) {
            CompareColumnAVX2<T, Op>(t, u, size, output);
            return true;
        }

        // technically, it is possible to add T != U cases by 
        // utilizing SIMD data conversion functions

        return false;
    }

    template<typename T, CompareType Op>
    static inline bool op_compare_val(
        uint8_t* const __restrict output, 
        const T* const __restrict t,
        const size_t size,
        const T value
    ) {
        CompareValAVX2<T, Op>(t, size, value, output);
        return true;
    }

    // API requirement: size % 8 == 0
    template<typename T, RangeType Op>
    static bool op_within_range(
        uint8_t* const __restrict data, 
        const T* const __restrict lower,
        const T* const __restrict upper,
        const T* const __restrict values,
        const size_t size
    ) {
        WithinRangeAVX2<T, Op>(lower, upper, values, size, data);
        return true;
    }
};

}
}
}
}
