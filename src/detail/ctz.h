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
struct CtzHelper<unsigned int> {
    static inline auto ctz(const unsigned int value) {
        return __builtin_ctz(value);
    }
};

template<>
struct CtzHelper<unsigned long> {
    static inline auto ctz(const unsigned long value) {
        return __builtin_ctzl(value);
    }
};

template<>
struct CtzHelper<unsigned long long> {
    static inline auto ctz(const unsigned long long value) {
        return __builtin_ctzll(value);
    }
};

}
}
}
