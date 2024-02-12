#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../../common.h"

namespace milvus {
namespace bitset {
namespace detail {

// The default reference vectorizer. 
// Its every function returns a boolean value whether a vectorized implementation
//   exists and was invoked. If not, then the caller code will use a default 
//   non-vectorized implementation. 
// The default vectorizer provides no vectorized implementation, forcing the
//   caller to use a defaut non-vectorized implementation every time.
struct VectorizedRef {
    // size is in bytes
    template<typename T, typename U, CompareType Op>
    static bool op_compare(
        uint8_t* const __restrict data, 
        const T* const __restrict t,
        const U* const __restrict u,
        const size_t size
    ) {
        return false;
    }
};

}
}
}
