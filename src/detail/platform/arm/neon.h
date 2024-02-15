#pragma once

#include <cstddef>
#include <cstdint>

#include "common.h"

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {

namespace neon {

// a facility to run through all acceptable data types
#define ALL_DATATYPES_1(FUNC) \
    FUNC(int8_t); \
    FUNC(int16_t); \
    FUNC(int32_t); \
    FUNC(int64_t); \
    FUNC(float); \
    FUNC(double);


// the default implementation
template<typename T, typename U, CompareType Op>
struct OpCompareColumnImpl {
    static bool op_compare_column(
        uint8_t* const __restrict bitmask, 
        const T* const __restrict t,
        const U* const __restrict u,
        const size_t size
    ) {
        return false;
    }
};

#define DECLARE_PARTIAL_OP_COMPARE_COLUMN(TTYPE) \
    template<CompareType Op> \
    struct OpCompareColumnImpl<TTYPE, TTYPE, Op> { \
        static bool op_compare_column( \
            uint8_t* const __restrict bitmask, \
            const TTYPE* const __restrict t, \
            const TTYPE* const __restrict u, \
            const size_t size \
        ); \
    };

ALL_DATATYPES_1(DECLARE_PARTIAL_OP_COMPARE_COLUMN)

#undef DECLARE_PARTIAL_OP_COMPARE_COLUMN


// the default implementation
template<typename T, CompareType Op>
struct OpCompareValImpl {
    static inline bool op_compare_val(
        uint8_t* const __restrict bitmask,
        const T* const __restrict t,
        const size_t size,
        const T& value
    ) {
        return false;
    }
};

#define DECLARE_PARTIAL_OP_COMPARE_VAL(TTYPE) \
    template<CompareType Op> \
    struct OpCompareValImpl<TTYPE, Op> { \
        static bool op_compare_val( \
            uint8_t* const __restrict bitmask, \
            const TTYPE* const __restrict t, \
            const size_t size, \
            const TTYPE& value \
        ); \
    };

ALL_DATATYPES_1(DECLARE_PARTIAL_OP_COMPARE_VAL)

#undef DECLARE_PARTIAL_OP_COMPARE_VAL


// the default implementation
template<typename T, RangeType Op>
struct OpWithinRangeColumnImpl {
    static inline bool op_within_range_column(
        uint8_t* const __restrict bitmask, 
        const T* const __restrict lower,
        const T* const __restrict upper,
        const T* const __restrict values,
        const size_t size
    ) {
        return false;
    }
};

#define DECLARE_PARTIAL_OP_WITHIN_RANGE_COLUMN(TTYPE) \
    template<RangeType Op> \
    struct OpWithinRangeColumnImpl<TTYPE, Op> { \
        static bool op_within_range_column( \
            uint8_t* const __restrict bitmask, \
            const TTYPE* const __restrict lower, \
            const TTYPE* const __restrict upper, \
            const TTYPE* const __restrict values, \
            const size_t size \
        ); \
    };

ALL_DATATYPES_1(DECLARE_PARTIAL_OP_WITHIN_RANGE_COLUMN)

#undef DECLARE_PARTIAL_OP_WITHIN_RANGE_COLUMN


// the default implementation
template<typename T, RangeType Op>
struct OpWithinRangeValImpl {
    static inline bool op_within_range_val(
        uint8_t* const __restrict bitmask, 
        const T& lower,
        const T& upper,
        const T* const __restrict values,
        const size_t size
    ) {
        return false;
    }
};

#define DECLARE_PARTIAL_OP_WITHIN_RANGE_VAL(TTYPE) \
    template<RangeType Op> \
    struct OpWithinRangeValImpl<TTYPE, Op> { \
        static bool op_within_range_val( \
            uint8_t* const __restrict bitmask, \
            const TTYPE& lower, \
            const TTYPE& upper, \
            const TTYPE* const __restrict values, \
            const size_t size \
        ); \
    };

ALL_DATATYPES_1(DECLARE_PARTIAL_OP_WITHIN_RANGE_VAL)

#undef DECLARE_PARTIAL_OP_WITHIN_RANGE_VAL

//

#undef ALL_DATATYPES_1

}

//
struct VectorizedNeon {
    // API requirement: size % 8 == 0
    template<typename T, typename U, CompareType Op>
    static bool op_compare_column(
        uint8_t* const __restrict bitmask, 
        const T* const __restrict t,
        const U* const __restrict u,
        const size_t size
    ) {
        return neon::OpCompareColumnImpl<T, U, Op>::op_compare_column(bitmask, t, u, size);
    }

    // Fills a bitmask by comparing elements of a given array to a
    //   given value.
    // API requirement: size % 8 == 0
    template<typename T, CompareType Op>
    static bool op_compare_val(
        uint8_t* const __restrict bitmask,
        const T* const __restrict t,
        const size_t size,
        const T& value
    ) {
        return neon::OpCompareValImpl<T, Op>::op_compare_val(bitmask, t, size, value);
    }

    // API requirement: size % 8 == 0
    template<typename T, RangeType Op>
    static bool op_within_range_column(
        uint8_t* const __restrict bitmask, 
        const T* const __restrict lower,
        const T* const __restrict upper,
        const T* const __restrict values,
        const size_t size
    ) {
        return neon::OpWithinRangeColumnImpl<T, Op>::op_within_range_column(bitmask, lower, upper, values, size);
    }

    // API requirement: size % 8 == 0
    template<typename T, RangeType Op>
    static bool op_within_range_val(
        uint8_t* const __restrict bitmask, 
        const T& lower,
        const T& upper,
        const T* const __restrict values,
        const size_t size
    ) {
        return neon::OpWithinRangeValImpl<T, Op>::op_within_range_val(bitmask, lower, upper, values, size);
    }
};

}
}
}
}
