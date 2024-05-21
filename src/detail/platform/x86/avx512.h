#pragma once

#include <cstddef>
#include <cstdint>

#include "../../../common.h"

#include "avx512-decl.h"

#ifdef BITSET_HEADER_ONLY
#include "avx512-impl.h"
#endif

namespace milvus {
namespace bitset {
namespace detail {
namespace x86 {

///////////////////////////////////////////////////////////////////////////

//
struct VectorizedAvx512 {
    template<typename T, typename U, CompareOpType Op>
    static constexpr inline auto op_compare_column = avx512::OpCompareColumnImpl<T, U, Op>::op_compare_column;

    template<typename T, CompareOpType Op>
    static constexpr inline auto op_compare_val = avx512::OpCompareValImpl<T, Op>::op_compare_val;

    template<typename T, RangeType Op>
    static constexpr inline auto op_within_range_column = avx512::OpWithinRangeColumnImpl<T, Op>::op_within_range_column;

    template<typename T, RangeType Op>
    static constexpr inline auto op_within_range_val = avx512::OpWithinRangeValImpl<T, Op>::op_within_range_val;

    template<typename T, ArithOpType AOp, CompareOpType CmpOp>
    static constexpr inline auto op_arith_compare = avx512::OpArithCompareImpl<T, AOp, CmpOp>::op_arith_compare;

    template<typename ElementT>
    static constexpr inline auto forward_op_and = avx512::ForwardOpsImpl<ElementT>::op_and;

    template<typename ElementT>
    static constexpr inline auto forward_op_and_multiple = avx512::ForwardOpsImpl<ElementT>::op_and_multiple;

    template<typename ElementT>
    static constexpr inline auto forward_op_or = avx512::ForwardOpsImpl<ElementT>::op_or;

    template<typename ElementT>
    static constexpr inline auto forward_op_or_multiple = avx512::ForwardOpsImpl<ElementT>::op_or_multiple;

    template<typename ElementT>
    static constexpr inline auto forward_op_xor = avx512::ForwardOpsImpl<ElementT>::op_xor;

    template<typename ElementT>
    static constexpr inline auto forward_op_sub = avx512::ForwardOpsImpl<ElementT>::op_sub;
};

}
}
}
}

