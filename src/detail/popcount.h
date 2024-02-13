#pragma once

#include <cstdint>

namespace milvus {
namespace bitset {
namespace detail {

//
template<typename T>
struct PopCountHelper {};

//
template<>
struct PopCountHelper<unsigned long long> {
    static inline unsigned long long count(const unsigned long long v) {
        return __builtin_popcountll(v);
    }
};

template<>
struct PopCountHelper<unsigned long> {
    static inline unsigned long count(const unsigned long v) {
        return __builtin_popcountl(v);
    }
};

template<>
struct PopCountHelper<unsigned int> {
    static inline unsigned int count(const unsigned int v) {
        return __builtin_popcount(v);
    }
};

template<>
struct PopCountHelper<uint8_t> {
    static inline uint8_t count(const uint8_t v) {
        return __builtin_popcount(v);
    }
};

}
}
}
