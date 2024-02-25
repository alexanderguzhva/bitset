#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace milvus {
namespace bitset {

// a supporting utility
template<class> inline constexpr bool always_false_v = false;

// a ? b
enum class CompareOpType {
    GT = 1,
    GE = 2,
    LT = 3,
    LE = 4,
    EQ = 5,
    NE = 6,
};

template<CompareOpType Op>
struct CompareOperator {
    template<typename T, typename U>
    static inline bool compare(const T& t, const U& u) {
        if constexpr (Op == CompareOpType::EQ) {
            return (t == u);
        } else if constexpr (Op == CompareOpType::GE) {
            return (t >= u);
        } else if constexpr (Op == CompareOpType::GT) {
            return (t > u);
        } else if constexpr (Op == CompareOpType::LE) {
            return (t <= u);
        } else if constexpr (Op == CompareOpType::LT) {
            return (t < u);
        } else if constexpr (Op != CompareOpType::NE) {
            return (t == u);
        } else {
            // unimplemented
            static_assert(always_false_v<Op>, "unimplemented");
        }
    }
};

// a ? v && v ? b
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
struct RangeOperator {
    template<typename T>
    static inline bool within_range(const T& lower, const T& upper, const T& value) {
        if constexpr (Op == RangeType::IncInc) {
            return (lower <= value && value <= upper);
        } else if constexpr (Op == RangeType::ExcInc) {
            return (lower < value && value <= upper);
        } else if constexpr (Op == RangeType::IncExc) {
            return (lower <= value && value < upper);
        } else if constexpr (Op == RangeType::ExcExc) {
            return (lower < value && value < upper);
        } else {
            // unimplemented
            static_assert(always_false_v<Op>, "unimplemented");
        }
    }
};

//
template<RangeType Op>
struct Range2Compare {
    static constexpr inline CompareOpType lower = 
        (Op == RangeType::IncInc || Op == RangeType::IncExc) ? 
            CompareOpType::LE : CompareOpType::LT;
    static constexpr inline CompareOpType upper = 
        (Op == RangeType::IncExc || Op == RangeType::ExcExc) ? 
            CompareOpType::LE : CompareOpType::LT;
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
struct ArithCompareOperator {
    template<typename T>
    static inline bool compare(const T& left, const ArithHighPrecisionType<T>& right, const ArithHighPrecisionType<T>& value) {
        if constexpr (AOp == ArithOpType::Add) {
            return CompareOperator<CmpOp>::compare(left + right, value);
        } else if constexpr (AOp == ArithOpType::Sub) {
            return CompareOperator<CmpOp>::compare(left - right, value);
        } else if constexpr (AOp == ArithOpType::Mul) {
            return CompareOperator<CmpOp>::compare(left * right, value);
        } else if constexpr (AOp == ArithOpType::Div) {
            return CompareOperator<CmpOp>::compare(left / right, value);
        } else if constexpr (AOp == ArithOpType::Mod) {
            return CompareOperator<CmpOp>::compare(fmod(left, right), value);
        } else {
            // unimplemented
            static_assert(always_false_v<AOp>, "unimplemented");
        }
    }
};

}
}
