#pragma once

#include <cstdint>

namespace milvus {
namespace bitset {
namespace detail {

//
template<typename T>
struct PopCntHelper {};

//
template<>
struct PopCntHelper<uint64_t> {
    static inline uint64_t count(const uint64_t v) {
        return __builtin_popcountll(v);
    }
};

template<>
struct PopCntHelper<uint32_t> {
    static inline uint32_t count(const uint32_t v) {
        return __builtin_popcount(v);
    }
};

template<>
struct PopCntHelper<uint8_t> {
    static inline uint8_t count(const uint8_t v) {
        return __builtin_popcount(v);
    }
};

}
}
}
