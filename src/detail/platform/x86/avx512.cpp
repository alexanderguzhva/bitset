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

// a facility to run through all possible operations
#define ALL_OPS(FUNC,...) \
    FUNC(Equal,__VA_ARGS__); \
    FUNC(GreaterEqual,__VA_ARGS__); \
    FUNC(Greater,__VA_ARGS__); \
    FUNC(LessEqual,__VA_ARGS__); \
    FUNC(Less,__VA_ARGS__); \
    FUNC(NotEqual,__VA_ARGS__);

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

//
template <typename T, CompareType Op>
struct CompareValAVX512Impl {};

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
                _mm_setr_epi64(__m64(cmp_mask), __m64(0ULL))
            );
        }
    }
};

template <CompareType Op>
struct CompareValAVX512Impl<int16_t, Op> {
    static void Compare(
        const int16_t* const __restrict src, 
        const size_t size, 
        const int16_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        const __m512i target = _mm512_set1_epi16(val);
        uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res); 
        constexpr auto pred = ComparePredicate<int16_t, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size32 = (size / 32) * 32;
        for (size_t i = 0; i < size32; i += 32) {
            const __m512i v = _mm512_loadu_si512(src + i);
            const __mmask32 cmp_mask = _mm512_cmp_epi16_mask(v, target, pred);

            res_u32[i / 32] = cmp_mask;
        }

        // process leftovers
        if (size32 != size) {
            const uint32_t mask = get_mask(size - size32);
            const __m512i v = _mm512_maskz_loadu_epi16(mask, src + size32);
            const __mmask32 cmp_mask = _mm512_cmp_epi16_mask(v, target, pred);

            const uint16_t store_mask = get_mask((size - size32) / 8);
            _mm_mask_storeu_epi8(
                res_u32 + size32 / 32, 
                store_mask, 
                _mm_setr_epi32(cmp_mask, 0, 0, 0)
            );
        }
    }
};

template <CompareType Op>
struct CompareValAVX512Impl<int32_t, Op> {
    static void Compare(
        const int32_t* const __restrict src, 
        const size_t size, 
        const int32_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        const __m512i target = _mm512_set1_epi32(val);
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res); 
        constexpr auto pred = ComparePredicate<int32_t, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size16 = (size / 16) * 16;
        for (size_t i = 0; i < size16; i += 16) {
            const __m512i v = _mm512_loadu_si512(src + i);
            const __mmask16 cmp_mask = _mm512_cmp_epi32_mask(v, target, pred);

            res_u16[i / 16] = cmp_mask;
        }

        // process leftovers
        if (size16 != size) {
            const uint16_t mask = get_mask(size - size16);
            const __m512i v = _mm512_maskz_loadu_epi32(mask, src + size16);
            const __mmask16 cmp_mask = _mm512_cmp_epi32_mask(v, target, pred);

            const uint16_t store_mask = get_mask((size - size16) / 8);
            _mm_mask_storeu_epi8(
                res_u16 + size16 / 16, 
                store_mask, 
                _mm_setr_epi16(cmp_mask, 0, 0, 0, 0, 0, 0, 0)
            );
        }
    }
};

template <CompareType Op>
struct CompareValAVX512Impl<int64_t, Op> {
    static void Compare(
        const int64_t* const __restrict src, 
        const size_t size, 
        const int64_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        const __m512i target = _mm512_set1_epi64(val);
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res); 
        constexpr auto pred = ComparePredicate<int64_t, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m512i v = _mm512_loadu_si512(src + i);
            const __mmask8 cmp_mask = _mm512_cmp_epi64_mask(v, target, pred);

            res_u8[i / 8] = cmp_mask;
        }

        // process leftovers
        if (size8 != size) {
            const uint8_t mask = get_mask(size - size8);
            const __m512i v = _mm512_maskz_loadu_epi32(mask, src + size8);
            const __mmask8 cmp_mask = _mm512_cmp_epi32_mask(v, target, pred);

            res_u8[size8 / 8] = cmp_mask;
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
        if (size16 != size) {
            const uint16_t mask = get_mask(size - size16);
            const __m512 v = _mm512_maskz_loadu_ps(mask, src + size16);
            const __mmask16 cmp_mask = _mm512_cmp_ps_mask(v, target, pred);

            const uint16_t store_mask = get_mask((size - size16) / 8);
            _mm_mask_storeu_epi8(
                res_u16 + size16 / 16, 
                store_mask, 
                _mm_setr_epi16(cmp_mask, 0, 0, 0, 0, 0, 0, 0)
            );
        }
    }
};

template <CompareType Op>
struct CompareValAVX512Impl<double, Op> {
    static void Compare(
        const double* const __restrict src, 
        const size_t size, 
        const double val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        constexpr auto pred = ComparePredicate<double, Op>::value;

        const __m512d target = _mm512_set1_pd(val);

        // todo: aligned reads & writes

        // process big blocks
        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m512d v = _mm512_loadu_pd(src + i);
            const __mmask8 cmp_mask = _mm512_cmp_pd_mask(v, target, pred);

            res_u8[i / 8] = cmp_mask;
        }

        // process leftovers
        if (size8 != size) {
            const uint8_t mask = get_mask(size - size8);
            const __m512d v = _mm512_maskz_loadu_pd(mask, src + size8);
            const __mmask8 cmp_mask = _mm512_cmp_pd_mask(v, target, pred);

            res_u8[size8 / 8] = cmp_mask;
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

ALL_OPS(INSTANTIATE_VAL_AVX512, int8_t)
ALL_OPS(INSTANTIATE_VAL_AVX512, int16_t)
ALL_OPS(INSTANTIATE_VAL_AVX512, int32_t)
ALL_OPS(INSTANTIATE_VAL_AVX512, int64_t)
ALL_OPS(INSTANTIATE_VAL_AVX512, float)
ALL_OPS(INSTANTIATE_VAL_AVX512, double)

#undef INSTANTIATE_VAL_AVX512

//
template <typename T, CompareType Op>
struct CompareColumnAVX512Impl {};

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

#undef ALL_OPS

}
}
}
}
