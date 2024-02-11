#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace milvus {
namespace bitset {

#define CHECK_SUPPORTED_TYPE(T, Message)                                     \
    static_assert(                                                           \
        std::is_same<T, bool>::value || std::is_same<T, int8_t>::value ||    \
            std::is_same<T, int16_t>::value ||                               \
            std::is_same<T, int32_t>::value ||                               \
            std::is_same<T, int64_t>::value ||                               \
            std::is_same<T, float>::value || std::is_same<T, double>::value, \
        Message);

enum class CompareType {
    GT = 1,
    GE = 2,
    LT = 3,
    LE = 4,
    EQ = 5,
    NEQ = 6,
};

}
}