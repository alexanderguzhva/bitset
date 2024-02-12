#pragma once

#include <cstdint>

namespace milvus {
namespace bitset {
namespace detail {

// returns 8 * sizeof(T) for 0
// returns 1 for 0b10
// returns 2 for 0b100
template<typename T>
struct CtzHelper {};

template<>
struct CtzHelper<uint8_t> {
    static inline auto ctz(const uint8_t value) {
        return __builtin_ctz(value);
    }
};

template<>
struct CtzHelper<uint32_t> {
    static inline auto ctz(const uint32_t value) {
        return __builtin_ctz(value);
    }
};

template<>
struct CtzHelper<uint64_t> {
    static inline auto ctz(const uint64_t value) {
        return __builtin_ctzll(value);
    }
};

}
}
}