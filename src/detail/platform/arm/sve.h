#pragma once

#include <cstddef>
#include <cstdint>

#include "../../../common.h"

#include "sve-decl.h"

#ifdef BITSET_HEADER_ONLY
#include "sve-impl.h"
#endif

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {

///////////////////////////////////////////////////////////////////////////

//
struct VectorizedSve {
    template<typename T, typename U, CompareOpType Op>
    static constexpr inline auto op_compare_column = sve::OpCompareColumnImpl<T, U, Op>::op_compare_column;

    template<typename T, CompareOpType Op>
    static constexpr inline auto op_compare_val = sve::OpCompareValImpl<T, Op>::op_compare_val;

    template<typename T, RangeType Op>
    static constexpr inline auto op_within_range_column = sve::OpWithinRangeColumnImpl<T, Op>::op_within_range_column;

    template<typename T, RangeType Op>
    static constexpr inline auto op_within_range_val = sve::OpWithinRangeValImpl<T, Op>::op_within_range_val;

    template<typename T, ArithOpType AOp, CompareOpType CmpOp>
    static constexpr inline auto op_arith_compare = sve::OpArithCompareImpl<T, AOp, CmpOp>::op_arith_compare;

    template<typename ElementT>
    static constexpr inline auto forward_op_and = sve::ForwardOpsImpl<ElementT>::op_and;

    template<typename ElementT>
    static constexpr inline auto forward_op_and_multiple = sve::ForwardOpsImpl<ElementT>::op_and_multiple;

    template<typename ElementT>
    static constexpr inline auto forward_op_or = sve::ForwardOpsImpl<ElementT>::op_or;

    template<typename ElementT>
    static constexpr inline auto forward_op_or_multiple = sve::ForwardOpsImpl<ElementT>::op_or_multiple;

    template<typename ElementT>
    static constexpr inline auto forward_op_xor = sve::ForwardOpsImpl<ElementT>::op_xor;

    template<typename ElementT>
    static constexpr inline auto forward_op_sub = sve::ForwardOpsImpl<ElementT>::op_sub;
};

}
}
}
}

