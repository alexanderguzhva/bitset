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

template <typename T>
void
EqualValAVX2(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
LessValAVX2(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
GreaterValAVX2(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
NotEqualValAVX2(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
LessEqualValAVX2(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
GreaterEqualValAVX2(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
EqualColumnAVX2(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
LessColumnAVX2(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
LessEqualColumnAVX2(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
GreaterColumnAVX2(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
GreaterEqualColumnAVX2(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
NotEqualColumnAVX2(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

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
            EqualValAVX2(t, size, value, output);
            return true;
        } else if constexpr(Op == CompareType::GE) {
            GreaterEqualValAVX2(t, size, value, output);
            return true;
        } else if constexpr(Op == CompareType::GT) {
            GreaterValAVX2(t, size, value, output);            
            return true;
        } else if constexpr(Op == CompareType::LE) {
            LessEqualValAVX2(t, size, value, output);            
            return true;
        } else if constexpr(Op == CompareType::LT) {
            LessValAVX2(t, size, value, output);            
            return true;
        } else if constexpr(Op == CompareType::NEQ) {
            NotEqualValAVX2(t, size, value, output);
            return true;
        } else {
            // unimplemented
            return false;
        }
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
            EqualColumnAVX2<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::GE) {
            GreaterEqualColumnAVX2<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::GT) {
            GreaterColumnAVX2<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::LE) {
            LessEqualColumnAVX2<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::LT) {
            LessColumnAVX2<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::NEQ) {
            NotEqualColumnAVX2<T>(t, u, size, output);
        } else {
            // unimplemented
        }
    }
};

}
}
}
}
