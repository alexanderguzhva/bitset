#pragma once

#include <cstddef>
#include <cstdint>

#include "../../common.h"

namespace milvus {
namespace bitset {
namespace detail {

//
struct VectorizedDynamic {
    // size is in bytes
    template<typename T, typename U, CompareType Op>
    static bool op_compare(
        uint8_t* const __restrict data, 
        const T* const __restrict t,
        const U* const __restrict u,
        const size_t size
    );
};

}
}
}