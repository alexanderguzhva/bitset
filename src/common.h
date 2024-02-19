#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace milvus {
namespace bitset {

#define CHECK_SUPPORTED_TYPE(T, Message)                                     \
    static_assert(                                                           \
        std::is_same<T, bool>::value || std::is_same<T, int8_t>::value ||    \
            std::is_same<T, int16_t>::value ||                               \
            std::is_same<T, int32_t>::value ||                               \
            std::is_same<T, int64_t>::value ||                               \
            std::is_same<T, float>::value || std::is_same<T, double>::value, \
        Message);

enum class CompareOpType {
    GT = 1,
    GE = 2,
    LT = 3,
    LE = 4,
    EQ = 5,
    NEQ = 6,
};

template<CompareOpType Op>
struct CompareOperator {};

template<>
struct CompareOperator<CompareOpType::EQ> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t == u);
    }
};

template<>
struct CompareOperator<CompareOpType::GE> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t >= u);
    }
};

template<>
struct CompareOperator<CompareOpType::GT> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t > u);
    }
};

template<>
struct CompareOperator<CompareOpType::LE> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t <= u);
    }
};

template<>
struct CompareOperator<CompareOpType::LT> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t < u);
    }
};

template<>
struct CompareOperator<CompareOpType::NEQ> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t != u);
    }
};

//
enum class RangeType {
    // [a, b]
    IncInc,
    // [a, b)
    IncExc,
    // (a, b]
    ExcInc,
    // (a, b)
    ExcExc
};

template<RangeType Op>
struct RangeOperator {};

template<>
struct RangeOperator<RangeType::IncInc> {
    template<typename T>
    static inline bool within_range(const T& lower, const T& upper, const T& value) {
        return (lower <= value && value <= upper);
    }
};

template<>
struct RangeOperator<RangeType::IncExc> {
    template<typename T>
    static inline bool within_range(const T& lower, const T& upper, const T& value) {
        return (lower <= value && value < upper);
    }
};

template<>
struct RangeOperator<RangeType::ExcInc> {
    template<typename T>
    static inline bool within_range(const T& lower, const T& upper, const T& value) {
        return (lower < value && value <= upper);
    }
};

template<>
struct RangeOperator<RangeType::ExcExc> {
    template<typename T>
    static inline bool within_range(const T& lower, const T& upper, const T& value) {
        return (lower < value && value < upper);
    }
};

// The following operation is Milvus-specific
enum class ArithOpType {
    Add,
    Sub,
    Mul,
    Div,
    Mod
};

template<typename T>
using ArithHighPrecisionType = 
    std::conditional_t<std::is_integral_v<T> && !std::is_same_v<bool, T>, int64_t, T>;

template<ArithOpType AOp, CompareOpType CmpOp>
struct ArithCompareOperator {};

template<CompareOpType CmpOp>
struct ArithCompareOperator<ArithOpType::Add, CmpOp> {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(left + right, value);
    }
};

template<CompareOpType CmpOp>
struct ArithCompareOperator<ArithOpType::Sub, CmpOp> {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(left - right, value);
    }
};

template<CompareOpType CmpOp>
struct ArithCompareOperator<ArithOpType::Mul, CmpOp> {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(left * right, value);
    }
};

template<CompareOpType CmpOp>
struct ArithCompareOperator<ArithOpType::Div, CmpOp> {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(left / right, value);
    }
};

template<CompareOpType CmpOp>
struct ArithCompareOperator<ArithOpType::Mod, CmpOp> {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(fmod(left, right), value);
    }
};

}
}
