#pragma once

#include <cstddef>
#include <cstdint>

#include "common.h"

namespace milvus {
namespace bitset {
namespace detail {
namespace x86 {

void
AndAVX512(void* const left, const void* const right, const size_t size);

void
OrAVX512(void* const left, const void* const right, const size_t size);

template <typename T>
void
EqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
LessValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
GreaterValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
NotEqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
LessEqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
GreaterEqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
EqualColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
LessColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
LessEqualColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
GreaterColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
GreaterEqualColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
NotEqualColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template<typename T, RangeType Op>
void WithinRangeAVX512(const T* const __restrict lower, const T* const __restrict upper, const T* const __restrict values, const size_t size, uint8_t* const __restrict res);

//
struct VectorizedAvx512 {
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
             op_compare_column_same<T, Op>(output, t, u, size);
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
        if constexpr(Op == CompareType::EQ) {
            EqualValAVX512(t, size, value, output);
            return true;
        } else if constexpr(Op == CompareType::GE) {
            GreaterEqualValAVX512(t, size, value, output);
            return true;
        } else if constexpr(Op == CompareType::GT) {
            GreaterValAVX512(t, size, value, output);            
            return true;
        } else if constexpr(Op == CompareType::LE) {
            LessEqualValAVX512(t, size, value, output);            
            return true;
        } else if constexpr(Op == CompareType::LT) {
            LessValAVX512(t, size, value, output);            
            return true;
        } else if constexpr(Op == CompareType::NEQ) {
            NotEqualValAVX512(t, size, value, output);
            return true;
        } else {
            // unimplemented
            return false;
        }
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
        WithinRangeAVX512<T, Op>(lower, upper, values, size, data);
        return true;
    }

private:
    template<typename T, CompareType Op>
    static inline void op_compare_column_same(
        uint8_t* const __restrict output, 
        const T* const __restrict t,
        const T* const __restrict u,
        const size_t size
    ) {
        if constexpr(Op == CompareType::EQ) {
            EqualColumnAVX512<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::GE) {
            GreaterEqualColumnAVX512<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::GT) {
            GreaterColumnAVX512<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::LE) {
            LessEqualColumnAVX512<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::LT) {
            LessColumnAVX512<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::NEQ) {
            NotEqualColumnAVX512<T>(t, u, size, output);
        } else {
            // unimplemented
        }
    }
};

}
}
}
}
