#pragma once

#include "../../../common.h"

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {

//
template<RangeType Op>
struct Range2Compare {};

template<>
struct Range2Compare<RangeType::IncInc> {
    static constexpr inline CompareOpType lower = CompareOpType::LE;  
    static constexpr inline CompareOpType upper = CompareOpType::LE;  
};

template<>
struct Range2Compare<RangeType::IncExc> {
    static constexpr inline CompareOpType lower = CompareOpType::LE;  
    static constexpr inline CompareOpType upper = CompareOpType::LT;  
};

template<>
struct Range2Compare<RangeType::ExcInc> {
    static constexpr inline CompareOpType lower = CompareOpType::LT;  
    static constexpr inline CompareOpType upper = CompareOpType::LE;  
};

template<>
struct Range2Compare<RangeType::ExcExc> {
    static constexpr inline CompareOpType lower = CompareOpType::LT;  
    static constexpr inline CompareOpType upper = CompareOpType::LT;  
};

}
}
}
}
