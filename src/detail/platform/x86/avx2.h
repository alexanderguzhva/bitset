#pragma once

#include <cstddef>
#include <cstdint>

#include "../../../common.h"

#include "avx2-decl.h"

#ifdef BITSET_HEADER_ONLY
#include "avx2-impl.h"
#endif

namespace milvus {
namespace bitset {
namespace detail {
namespace x86 {

///////////////////////////////////////////////////////////////////////////

//
struct VectorizedAvx2 {
    template<typename T, typename U, CompareOpType Op>
    static constexpr inline auto op_compare_column = avx2::OpCompareColumnImpl<T, U, Op>::op_compare_column;

    template<typename T, CompareOpType Op>
    static constexpr inline auto op_compare_val = avx2::OpCompareValImpl<T, Op>::op_compare_val;

    template<typename T, RangeType Op>
    static constexpr inline auto op_within_range_column = avx2::OpWithinRangeColumnImpl<T, Op>::op_within_range_column;

    template<typename T, RangeType Op>
    static constexpr inline auto op_within_range_val = avx2::OpWithinRangeValImpl<T, Op>::op_within_range_val;

    template<typename T, ArithOpType AOp, CompareOpType CmpOp>
    static constexpr inline auto op_arith_compare = avx2::OpArithCompareImpl<T, AOp, CmpOp>::op_arith_compare;
};

}
}
}
}

