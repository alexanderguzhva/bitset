#include "avx512.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <immintrin.h>

namespace milvus {
namespace bitset {

// count is expected to be in range [0, 64)
inline uint64_t get_mask(const size_t count) {
    return (uint64_t(1) << count) - uint64_t(1);
}

void
AndAVX512(void* const left, const void* const right, const size_t size) {
    uint8_t* const __restrict left_u8 = reinterpret_cast<uint8_t*>(left);
    const uint8_t* const __restrict right_u8 = reinterpret_cast<const uint8_t*>(right);

    const size_t size64 = (size / 64) * 64;
    for (size_t i = 0; i < size64; i += 64) {
        const __m512i left_v = _mm512_loadu_si512(left_u8 + i);
        const __m512i right_v = _mm512_loadu_si512(right_u8 + i);
        const __m512i res = _mm512_and_si512(left_v, right_v);
        _mm512_storeu_si512(left_u8 + i, res);
    }

    if (size64 != size) {
        const uint64_t mask = get_mask(size - size64);
        const __m512i left_v = _mm512_maskz_loadu_epi8(mask, left_u8 + size64);
        const __m512i right_v = _mm512_maskz_loadu_epi8(mask, right_u8 + size64);
        const __m512i res = _mm512_and_si512(left_v, right_v);
        _mm512_mask_storeu_epi8(left_u8 + size64, mask, res);
    }
}

void
OrAVX512(void* const left, const void* const right, const size_t size) {
    uint8_t* const __restrict left_u8 = reinterpret_cast<uint8_t*>(left);
    const uint8_t* const __restrict right_u8 = reinterpret_cast<const uint8_t*>(right);

    const size_t size64 = (size / 64) * 64;
    for (size_t i = 0; i < size64; i += 64) {
        const __m512i left_v = _mm512_loadu_si512(left_u8 + i);
        const __m512i right_v = _mm512_loadu_si512(right_u8 + i);
        const __m512i res = _mm512_or_si512(left_v, right_v);
        _mm512_storeu_si512(left_u8 + i, res);
    }

    if (size64 != size) {
        const uint64_t mask = get_mask(size - size64);
        const __m512i left_v = _mm512_maskz_loadu_epi8(mask, left_u8 + size64);
        const __m512i right_v = _mm512_maskz_loadu_epi8(mask, right_u8 + size64);
        const __m512i res = _mm512_or_si512(left_v, right_v);
        _mm512_mask_storeu_epi8(left_u8 + size64, mask, res);
    }
}

enum class CompareType {
    GT = 1,
    GE = 2,
    LT = 3,
    LE = 4,
    EQ = 5,
    NEQ = 6,
};

template<typename T, CompareType type>
struct ComparePredicate {};

template<typename T>
struct ComparePredicate<T, CompareType::EQ> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_EQ_OQ : _MM_CMPINT_EQ; 
};

template<typename T>
struct ComparePredicate<T, CompareType::LT> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_LT_OQ : _MM_CMPINT_LT; 
};

template<typename T>
struct ComparePredicate<T, CompareType::LE> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_LE_OQ : _MM_CMPINT_LE; 
};

template<typename T>
struct ComparePredicate<T, CompareType::GT> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_GT_OQ : _MM_CMPINT_NLE; 
};

template<typename T>
struct ComparePredicate<T, CompareType::GE> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_GE_OQ : _MM_CMPINT_NLT; 
};

template<typename T>
struct ComparePredicate<T, CompareType::NEQ> {
    static inline constexpr int value = 
        std::is_floating_point_v<T> ? _CMP_NEQ_OQ : _MM_CMPINT_NE; 
};

template <typename T, CompareType type>
struct CompareValAVX512Impl {
    static void
    Compare(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                      "T must be integral or float/double type");
    }
};

template <CompareType type>
struct CompareValAVX512Impl<int8_t, type> {
        static void
    Compare(const int8_t* const __restrict src, const size_t size, const int8_t val, void* const __restrict res) {
        assert(size % 8 == 0);
        
        const __m512i target = _mm512_set1_epi8(val);
        uint64_t* const __restrict res_u64 = reinterpret_cast<uint64_t*>(res); 

        const size_t size64 = (size / 64) * 64;
        for (size_t i = 0; i < size64; i += 64) {
            const __m512i v = _mm512_loadu_si512(src + i);
            const __mmask64 cmp_mask = _mm512_cmp_epi8_mask(
                v, target, (ComparePredicate<int8_t, type>::value)
            );

            res_u64[i / 64] = cmp_mask;
        }

        if (size64 != size) {
            const uint64_t mask = get_mask(size - size64);
            const __m512i v = _mm512_maskz_loadu_epi8(mask, src + size64);
            const __mmask64 cmp_mask = _mm512_cmp_epi8_mask(
                v, target, (ComparePredicate<int8_t, type>::value)
            );

            const uint16_t store_mask = get_mask((size - size64) / 8);
            _mm_mask_storeu_epi8(
                res_u64 + size64 / 64, 
                store_mask, 
                _mm_setr_epi64(__m64(cmp_mask), __m64(0ULL)));
        }
    }
};

//
template <typename T>
void
EqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
    CompareValAVX512Impl<T, CompareType::EQ>::Compare(src, size, val, res);
}

template
void EqualValAVX512(const int8_t* const __restrict src, const size_t size, const int8_t val, void* const __restrict res);

//
template <typename T>
void
LessValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
    CompareValAVX512Impl<T, CompareType::LT>::Compare(src, size, val, res);
}

template
void LessValAVX512(const int8_t* const __restrict src, const size_t size, const int8_t val, void* const __restrict res);

//
template <typename T>
void
GreaterValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
    CompareValAVX512Impl<T, CompareType::GT>::Compare(src, size, val, res);
}

template
void GreaterValAVX512(const int8_t* const __restrict src, const size_t size, const int8_t val, void* const __restrict res);

//
template <typename T>
void
NotEqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
    CompareValAVX512Impl<T, CompareType::NEQ>::Compare(src, size, val, res);
}

template
void NotEqualValAVX512(const int8_t* const __restrict src, const size_t size, const int8_t val, void* const __restrict res);

//
template <typename T>
void
LessEqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
    CompareValAVX512Impl<T, CompareType::LE>::Compare(src, size, val, res);
}

template
void LessEqualValAVX512(const int8_t* const __restrict src, const size_t size, const int8_t val, void* const __restrict res);

//
template <typename T>
void
GreaterEqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
    CompareValAVX512Impl<T, CompareType::GE>::Compare(src, size, val, res);
}

template
void GreaterEqualValAVX512(const int8_t* const __restrict src, const size_t size, const int8_t val, void* const __restrict res);

}
}