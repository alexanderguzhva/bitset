#pragma once

#include <cstddef>
#include <cstdint>

#include "../../../common.h"

#include "neon-decl.h"

#ifdef BITSET_HEADER_ONLY
#include "neon-impl.h"
#endif

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {

///////////////////////////////////////////////////////////////////////////

//
struct VectorizedNeon {
    template<typename T, typename U, CompareOpType Op>
    static constexpr inline auto op_compare_column = neon::OpCompareColumnImpl<T, U, Op>::op_compare_column;

    template<typename T, CompareOpType Op>
    static constexpr inline auto op_compare_val = neon::OpCompareValImpl<T, Op>::op_compare_val;

    template<typename T, RangeType Op>
    static constexpr inline auto op_within_range_column = neon::OpWithinRangeColumnImpl<T, Op>::op_within_range_column;

    template<typename T, RangeType Op>
    static constexpr inline auto op_within_range_val = neon::OpWithinRangeValImpl<T, Op>::op_within_range_val;

    template<typename T, ArithOpType AOp, CompareOpType CmpOp>
    static constexpr inline auto op_arith_compare = neon::OpArithCompareImpl<T, AOp, CmpOp>::op_arith_compare;

    template<typename ElementT>
    static constexpr inline auto forward_op_and = neon::ForwardOpsImpl<ElementT>::op_and;

    template<typename ElementT>
    static constexpr inline auto forward_op_and_multiple = neon::ForwardOpsImpl<ElementT>::op_and_multiple;

    template<typename ElementT>
    static constexpr inline auto forward_op_or = neon::ForwardOpsImpl<ElementT>::op_or;

    template<typename ElementT>
    static constexpr inline auto forward_op_or_multiple = neon::ForwardOpsImpl<ElementT>::op_or_multiple;

    template<typename ElementT>
    static constexpr inline auto forward_op_xor = neon::ForwardOpsImpl<ElementT>::op_xor;

    template<typename ElementT>
    static constexpr inline auto forward_op_sub = neon::ForwardOpsImpl<ElementT>::op_sub;
};

}
}
}
}

