#pragma once

#include <immintrin.h>

#include <type_traits>

#include "../../../common.h"

namespace milvus {
namespace bitset {
namespace detail {
namespace x86 {

template<typename T, CompareType type>
struct ComparePredicate {};

template<typename T>
struct ComparePredicate<T, CompareType::EQ> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_EQ_OQ : _MM_CMPINT_EQ; 
};

template<typename T>
struct ComparePredicate<T, CompareType::LT> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_LT_OQ : _MM_CMPINT_LT; 
};

template<typename T>
struct ComparePredicate<T, CompareType::LE> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_LE_OQ : _MM_CMPINT_LE; 
};

template<typename T>
struct ComparePredicate<T, CompareType::GT> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_GT_OQ : _MM_CMPINT_NLE; 
};

template<typename T>
struct ComparePredicate<T, CompareType::GE> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_GE_OQ : _MM_CMPINT_NLT; 
};

template<typename T>
struct ComparePredicate<T, CompareType::NEQ> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_NEQ_OQ : _MM_CMPINT_NE; 
};


}
}
}
}