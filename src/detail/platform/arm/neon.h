#pragma once

#include <cstddef>
#include <cstdint>

#include "common.h"

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {

void
AndNeon(void* const left, const void* const right, const size_t size);

void
OrNeon(void* const left, const void* const right, const size_t size);

template <typename T>
void
EqualValNeon(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
LessValNeon(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
GreaterValNeon(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
NotEqualValNeon(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
LessEqualValNeon(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
GreaterEqualValNeon(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
EqualColumnNeon(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
LessColumnNeon(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
LessEqualColumnNeon(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
GreaterColumnNeon(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
GreaterEqualColumnNeon(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
NotEqualColumnNeon(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

//
struct VectorizedNeon {
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
            EqualValNeon(t, size, value, output);
            return true;
        } else if constexpr(Op == CompareType::GE) {
            GreaterEqualValNeon(t, size, value, output);
            return true;
        } else if constexpr(Op == CompareType::GT) {
            GreaterValNeon(t, size, value, output);            
            return true;
        } else if constexpr(Op == CompareType::LE) {
            LessEqualValNeon(t, size, value, output);            
            return true;
        } else if constexpr(Op == CompareType::LT) {
            LessValNeon(t, size, value, output);            
            return true;
        } else if constexpr(Op == CompareType::NEQ) {
            NotEqualValNeon(t, size, value, output);
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
            EqualColumnNeon<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::GE) {
            GreaterEqualColumnNeon<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::GT) {
            GreaterColumnNeon<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::LE) {
            LessEqualColumnNeon<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::LT) {
            LessColumnNeon<T>(t, u, size, output);
        } else if constexpr(Op == CompareType::NEQ) {
            NotEqualColumnNeon<T>(t, u, size, output);
        } else {
            // unimplemented
        }
    }
};

}
}
}
}