#include "avx512.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <immintrin.h>

namespace milvus {
namespace bitset {
namespace detail {
namespace x86 {

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

template <typename T, CompareType Op>
struct CompareValAVX512Impl {
    static void
    Compare(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                      "T must be integral or float/double type");
    }
};

template <CompareType Op>
struct CompareValAVX512Impl<int8_t, Op> {
    static void Compare(
        const int8_t* const __restrict src, 
        const size_t size, 
        const int8_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        const __m512i target = _mm512_set1_epi8(val);
        uint64_t* const __restrict res_u64 = reinterpret_cast<uint64_t*>(res); 
        constexpr auto pred = ComparePredicate<int8_t, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size64 = (size / 64) * 64;
        for (size_t i = 0; i < size64; i += 64) {
            const __m512i v = _mm512_loadu_si512(src + i);
            const __mmask64 cmp_mask = _mm512_cmp_epi8_mask(v, target, pred);

            res_u64[i / 64] = cmp_mask;
        }

        // process leftovers
        if (size64 != size) {
            const uint64_t mask = get_mask(size - size64);
            const __m512i v = _mm512_maskz_loadu_epi8(mask, src + size64);
            const __mmask64 cmp_mask = _mm512_cmp_epi8_mask(v, target, pred);

            const uint16_t store_mask = get_mask((size - size64) / 8);
            _mm_mask_storeu_epi8(
                res_u64 + size64 / 64, 
                store_mask, 
                _mm_setr_epi64(__m64(cmp_mask), __m64(0ULL)));
        }
    }
};

template <CompareType Op>
struct CompareValAVX512Impl<float, Op> {
    static void Compare(
        const float* const __restrict src, 
        const size_t size, 
        const float val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res); 
        constexpr auto pred = ComparePredicate<float, Op>::value;

        const __m512 target = _mm512_set1_ps(val);

        // todo: aligned reads & writes

        // process big blocks
        const size_t size16 = (size / 16) * 16;
        for (size_t i = 0; i < size16; i += 16) {
            const __m512 v = _mm512_loadu_ps(src + i);
            const __mmask16 cmp_mask = _mm512_cmp_ps_mask(v, target, pred);

            res_u16[i / 16] = cmp_mask;
        }

        // process leftovers
        // todo: move to AVX512
        for (size_t i = size16; i < size; i += 8) {
            const __m256 target_sh = _mm256_set1_ps(val);

            const __m256 v0 = _mm256_loadu_ps(src + i);
            const __m256 cmp = _mm256_cmp_ps(v0, target_sh, pred);
            const uint8_t mmask = _mm256_movemask_ps(cmp);

            res_u8[i / 8] = mmask;            
        }     
    }
};

#define DECLARE_VAL_AVX512(NAME, OP) \
    template<typename T> \
    void NAME##ValAVX512( \
        const T* const __restrict src, \
        const size_t size, \
        const T val, \
        void* const __restrict res \
    ) { \
        CompareValAVX512Impl<T, CompareType::OP>::Compare(src, size, val, res); \
    }

DECLARE_VAL_AVX512(Equal, EQ);
DECLARE_VAL_AVX512(GreaterEqual, GE);
DECLARE_VAL_AVX512(Greater, GT);
DECLARE_VAL_AVX512(LessEqual, LE);
DECLARE_VAL_AVX512(Less, LT);
DECLARE_VAL_AVX512(NotEqual, NEQ);

#undef DECLARE_VAL_AVX512

#define INSTANTIATE_VAL_AVX512(NAME, TTYPE) \
    template void NAME##ValAVX512( \
        const TTYPE* const __restrict src, \
        const size_t size, \
        const TTYPE val, \
        void* const __restrict res \
    );

INSTANTIATE_VAL_AVX512(Equal, float);
INSTANTIATE_VAL_AVX512(GreaterEqual, float);
INSTANTIATE_VAL_AVX512(Greater, float);
INSTANTIATE_VAL_AVX512(LessEqual, float);
INSTANTIATE_VAL_AVX512(Less, float);
INSTANTIATE_VAL_AVX512(NotEqual, float);

INSTANTIATE_VAL_AVX512(Equal, int8_t);
INSTANTIATE_VAL_AVX512(GreaterEqual, int8_t);
INSTANTIATE_VAL_AVX512(Greater, int8_t);
INSTANTIATE_VAL_AVX512(LessEqual, int8_t);
INSTANTIATE_VAL_AVX512(Less, int8_t);
INSTANTIATE_VAL_AVX512(NotEqual, int8_t);

#undef INSTANTIATE_VAL_AVX512

//
template <typename T, CompareType Op>
struct CompareColumnAVX512Impl {
    static void
    Compare(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                      "T must be integral or float/double type");
    }
};

template <CompareType Op>
struct CompareColumnAVX512Impl<float, Op> {
    static inline void Compare(
        const float* const __restrict left, 
        const float* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        union {
            uint16_t u16[4];
            uint64_t u64;
        } foo;

//
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        uint64_t* const __restrict res_u64 = reinterpret_cast<uint64_t*>(res);
        constexpr auto pred = ComparePredicate<float, Op>::value;

        // todo: aligned reads & writes

        // todo: process in 64 elements

        const size_t size64 = (size / 64) * 64;
        for (size_t i = 0; i < size64; i += 64) {
            for (size_t j = 0; j < 64; j += 16) {
                const __m512 v0 = _mm512_loadu_ps(left + i + j);
                const __m512 v1 = _mm512_loadu_ps(right + i + j);
                const __mmask16 mmask = _mm512_cmp_ps_mask(v0, v1, pred);
                foo.u16[j / 16] = mmask;
            }

            res_u64[i / 64] = foo.u64;
        }

        // todo: move to AVX512
        for (size_t i = size64; i < size; i += 8) {
            const __m256 v0 = _mm256_loadu_ps(left + i);
            const __m256 v1 = _mm256_loadu_ps(right + i);
            const __m256 cmp = _mm256_cmp_ps(v0, v1, pred);
            const uint8_t mmask = _mm256_movemask_ps(cmp);

            res_u8[i / 8] = mmask;            
        }        
    }
};


//
template <typename T>
void EqualColumnAVX512(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX512Impl<T, CompareType::EQ>::Compare(left, right, size, res);   
}

template
void EqualColumnAVX512(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void GreaterEqualColumnAVX512(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX512Impl<T, CompareType::GE>::Compare(left, right, size, res);   
}

template
void GreaterEqualColumnAVX512(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void GreaterColumnAVX512(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX512Impl<T, CompareType::GT>::Compare(left, right, size, res);   
}

template
void GreaterColumnAVX512(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void LessEqualColumnAVX512(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX512Impl<T, CompareType::LE>::Compare(left, right, size, res);   
}

template
void LessEqualColumnAVX512(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void LessColumnAVX512(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX512Impl<T, CompareType::LT>::Compare(left, right, size, res);   
}

template
void LessColumnAVX512(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void NotEqualColumnAVX512(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX512Impl<T, CompareType::NEQ>::Compare(left, right, size, res);   
}

template
void NotEqualColumnAVX512(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

}
}
}
}
