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
    static constexpr inline CompareType lower = CompareType::LE;  
    static constexpr inline CompareType upper = CompareType::LE;  
};

template<>
struct Range2Compare<RangeType::IncExc> {
    static constexpr inline CompareType lower = CompareType::LE;  
    static constexpr inline CompareType upper = CompareType::LT;  
};

template<>
struct Range2Compare<RangeType::ExcInc> {
    static constexpr inline CompareType lower = CompareType::LT;  
    static constexpr inline CompareType upper = CompareType::LE;  
};

template<>
struct Range2Compare<RangeType::ExcExc> {
    static constexpr inline CompareType lower = CompareType::LT;  
    static constexpr inline CompareType upper = CompareType::LT;  
};

}
}
}
}
