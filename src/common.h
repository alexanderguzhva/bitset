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

enum class CompareType {
    GT = 1,
    GE = 2,
    LT = 3,
    LE = 4,
    EQ = 5,
    NEQ = 6,
};

template<CompareType Op>
struct CompareOperator {};

template<>
struct CompareOperator<CompareType::EQ> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t == u);
    }
};

template<>
struct CompareOperator<CompareType::GE> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t >= u);
    }
};

template<>
struct CompareOperator<CompareType::GT> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t > u);
    }
};

template<>
struct CompareOperator<CompareType::LE> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t <= u);
    }
};

template<>
struct CompareOperator<CompareType::LT> {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        return (t < u);
    }
};

template<>
struct CompareOperator<CompareType::NEQ> {
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
enum class ArithType {
    Add,
    Sub,
    Mul,
    Div,
    Mod
};

template<typename T, typename U>
struct HighestPrecisionType {};
template<typename T>
struct HighestPrecisionType<T, T> { using type = T; };
template<>
struct HighestPrecisionType<int8_t, int64_t> { using type = int64_t; };
template<>
struct HighestPrecisionType<int16_t, int64_t> { using type = int64_t; };
template<>
struct HighestPrecisionType<int32_t, int64_t> { using type = int64_t; };
template<>
struct HighestPrecisionType<int64_t, int64_t> { using type = int64_t; };

template<typename T, typename U>
using arith_highest_precision_t = typename HighestPrecisionType<T, U>::type;

template<typename T>
using ArithHighPrecisionType = 
    std::conditional_t<std::is_integral_v<T> && !std::is_same_v<bool, T>, int64_t, T>;

template<ArithType AOp, CompareType CmpOp>
struct ArithCompareOperator {};

template<CompareType CmpOp>
struct ArithCompareOperator<ArithType::Add, CmpOp> {
    // template<typename T, typename U>
    // static inline bool compare(const T& left, const U& right, const arith_highest_precision_t<T, U>& value) {
    //     using hp_type = arith_highest_precision_t<T, U>;
    //     return CompareOperator<CmpOp>::compare(hp_type(left) + hp_type(right), value);
    // }

    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(left + right, value);
    }
};

template<CompareType CmpOp>
struct ArithCompareOperator<ArithType::Sub, CmpOp> {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(left - right, value);
    }
};

template<CompareType CmpOp>
struct ArithCompareOperator<ArithType::Mul, CmpOp> {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(left * right, value);
    }
};

template<CompareType CmpOp>
struct ArithCompareOperator<ArithType::Div, CmpOp> {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(left / right, value);
    }
};

template<CompareType CmpOp>
struct ArithCompareOperator<ArithType::Mod, CmpOp> {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        return CompareOperator<CmpOp>::compare(fmod(left, right), value);
    }
};

}
}
