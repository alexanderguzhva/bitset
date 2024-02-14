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

// a facility to run through all possible compare operations
#define ALL_COMPARE_OPS(FUNC,...) \
    FUNC(__VA_ARGS__,EQ); \
    FUNC(__VA_ARGS__,GE); \
    FUNC(__VA_ARGS__,GT); \
    FUNC(__VA_ARGS__,LE); \
    FUNC(__VA_ARGS__,LT); \
    FUNC(__VA_ARGS__,NEQ);

// a facility to run through all possible range operations
#define ALL_RANGE_OPS(FUNC,...) \
    FUNC(__VA_ARGS__,IncInc); \
    FUNC(__VA_ARGS__,IncExc); \
    FUNC(__VA_ARGS__,ExcInc); \
    FUNC(__VA_ARGS__,ExcExc);

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

///////////////////////////////////////////////////////////////////////////

//
template <typename T, CompareType Op>
struct CompareValAVX512Impl {};

template <CompareType Op>
struct CompareValAVX512Impl<int8_t, Op> {
    static void compare(
        const int8_t* const __restrict src, 
        const size_t size, 
        const int8_t val, 
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        const __m512i target = _mm512_set1_epi8(val);
        uint64_t* const __restrict res_u64 = reinterpret_cast<uint64_t*>(res_u8); 
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
            // 8, 16, 24, 32, 40, 48 or 56 elements to process
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
    static void compare(
        const int16_t* const __restrict src, 
        const size_t size, 
        const int16_t val, 
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        const __m512i target = _mm512_set1_epi16(val);
        uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res_u8); 
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
            // 8, 16 or 24 elements to process
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
    static void compare(
        const int32_t* const __restrict src, 
        const size_t size, 
        const int32_t val, 
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        const __m512i target = _mm512_set1_epi32(val);
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8); 
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
            // 8 elements to process
            const uint16_t mask = get_mask(size - size16);
            const __m512i v = _mm512_maskz_loadu_epi32(mask, src + size16);
            const __mmask16 cmp_mask = _mm512_cmp_epi32_mask(v, target, pred);

            res_u8[size16 / 8] = uint8_t(cmp_mask);
        }
    }
};

template <CompareType Op>
struct CompareValAVX512Impl<int64_t, Op> {
    static void compare(
        const int64_t* const __restrict src, 
        const size_t size, 
        const int64_t val, 
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        const __m512i target = _mm512_set1_epi64(val);
        constexpr auto pred = ComparePredicate<int64_t, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m512i v = _mm512_loadu_si512(src + i);
            const __mmask8 cmp_mask = _mm512_cmp_epi64_mask(v, target, pred);

            res_u8[i / 8] = cmp_mask;
        }
    }
};

template <CompareType Op>
struct CompareValAVX512Impl<float, Op> {
    static void compare(
        const float* const __restrict src, 
        const size_t size, 
        const float val, 
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8); 
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
            // 8 elements to process
            const uint8_t mask = get_mask(size - size16);
            const __m512 v = _mm512_maskz_loadu_ps(mask, src + size16);
            const __mmask16 cmp_mask = _mm512_cmp_ps_mask(v, target, pred);

            res_u8[size16 / 8] = uint8_t(cmp_mask);
        }
    }
};

template <CompareType Op>
struct CompareValAVX512Impl<double, Op> {
    static void compare(
        const double* const __restrict src, 
        const size_t size, 
        const double val, 
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
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
    }
};

//
template<typename T, CompareType Op>
void CompareValAVX512(const T* const __restrict src, const size_t size, const T val, uint8_t* const __restrict res) {
    CompareValAVX512Impl<T, Op>::compare(src, size, val, res);
}

#define INSTANTIATE_COMPARE_VAL_AVX512(TTYPE,OP) \
    template void CompareValAVX512<TTYPE, CompareType::OP>( \
        const TTYPE* const __restrict src, \
        const size_t size, \
        const TTYPE val, \
        uint8_t* const __restrict res \
    );

ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX512, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX512, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX512, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX512, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX512, float)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX512, double)

#undef INSTANTIATE_COMPARE_VAL_AVX512

///////////////////////////////////////////////////////////////////////////

//
template <typename T, CompareType Op>
struct CompareColumnAVX512Impl {};

template <CompareType Op>
struct CompareColumnAVX512Impl<int8_t, Op> {
    static inline void compare(
        const int8_t* const __restrict left, 
        const int8_t* const __restrict right, 
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint64_t* const __restrict res_u64 = reinterpret_cast<uint64_t*>(res_u8); 
        constexpr auto pred = ComparePredicate<int8_t, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size64 = (size / 64) * 64;
        for (size_t i = 0; i < size64; i += 64) {
            const __m512i vl = _mm512_loadu_si512(left + i);
            const __m512i vr = _mm512_loadu_si512(right + i);
            const __mmask64 cmp_mask = _mm512_cmp_epi8_mask(vl, vr, pred);

            res_u64[i / 64] = cmp_mask;
        }

        // process leftovers
        if (size64 != size) {
            // 8, 16, 24, 32, 40, 48 or 56 elements to process
            const uint64_t mask = get_mask(size - size64);
            const __m512i vl = _mm512_maskz_loadu_epi8(mask, left + size64);
            const __m512i vr = _mm512_maskz_loadu_epi8(mask, right + size64);
            const __mmask64 cmp_mask = _mm512_cmp_epi8_mask(vl, vr, pred);

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
struct CompareColumnAVX512Impl<int16_t, Op> {
    static inline void compare(
        const int16_t* const __restrict left, 
        const int16_t* const __restrict right, 
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res_u8); 
        constexpr auto pred = ComparePredicate<int16_t, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size32 = (size / 32) * 32;
        for (size_t i = 0; i < size32; i += 32) {
            const __m512i vl = _mm512_loadu_si512(left + i);
            const __m512i vr = _mm512_loadu_si512(right + i);
            const __mmask32 cmp_mask = _mm512_cmp_epi16_mask(vl, vr, pred);

            res_u32[i / 32] = cmp_mask;
        }

        // process leftovers
        if (size32 != size) {
            // 8, 16 or 24 elements to process
            const uint32_t mask = get_mask(size - size32);
            const __m512i vl = _mm512_maskz_loadu_epi16(mask, left + size32);
            const __m512i vr = _mm512_maskz_loadu_epi16(mask, right + size32);
            const __mmask32 cmp_mask = _mm512_cmp_epi16_mask(vl, vr, pred);

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
struct CompareColumnAVX512Impl<int32_t, Op> {
    static inline void compare(
        const int32_t* const __restrict left, 
        const int32_t* const __restrict right, 
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8); 
        constexpr auto pred = ComparePredicate<int32_t, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size16 = (size / 16) * 16;
        for (size_t i = 0; i < size16; i += 16) {
            const __m512i vl = _mm512_loadu_si512(left + i);
            const __m512i vr = _mm512_loadu_si512(right + i);
            const __mmask16 cmp_mask = _mm512_cmp_epi32_mask(vl, vr, pred);

            res_u16[i / 16] = cmp_mask;
        }

        // process leftovers
        if (size16 != size) {
            // 8 elements to process
            const uint16_t mask = get_mask(size - size16);
            const __m512i vl = _mm512_maskz_loadu_epi32(mask, left + size16);
            const __m512i vr = _mm512_maskz_loadu_epi32(mask, right + size16);
            const __mmask16 cmp_mask = _mm512_cmp_epi32_mask(vl, vr, pred);

            res_u8[size16 / 8] = uint8_t(cmp_mask);
        }
    }
};

template <CompareType Op>
struct CompareColumnAVX512Impl<int64_t, Op> {
    static inline void compare(
        const int64_t* const __restrict left, 
        const int64_t* const __restrict right, 
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        constexpr auto pred = ComparePredicate<int64_t, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m512i vl = _mm512_loadu_si512(left + i);
            const __m512i vr = _mm512_loadu_si512(right + i);
            const __mmask8 cmp_mask = _mm512_cmp_epi64_mask(vl, vr, pred);

            res_u8[i / 8] = cmp_mask;
        }
    }
};

template <CompareType Op>
struct CompareColumnAVX512Impl<float, Op> {
    static inline void compare(
        const float* const __restrict left, 
        const float* const __restrict right, 
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8); 
        constexpr auto pred = ComparePredicate<float, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size16 = (size / 16) * 16;
        for (size_t i = 0; i < size16; i += 16) {
            const __m512 vl = _mm512_loadu_ps(left + i);
            const __m512 vr = _mm512_loadu_ps(right + i);
            const __mmask16 cmp_mask = _mm512_cmp_ps_mask(vl, vr, pred);

            res_u16[i / 16] = cmp_mask;
        }

        // process leftovers
        if (size16 != size) {
            // process 8 elements
            const uint16_t mask = get_mask(size - size16);
            const __m512 vl = _mm512_maskz_loadu_ps(mask, left + size16);
            const __m512 vr = _mm512_maskz_loadu_ps(mask, right + size16);
            const __mmask16 cmp_mask = _mm512_cmp_ps_mask(vl, vr, pred);

            res_u8[size16 / 8] = uint8_t(cmp_mask);
        }
    }
};

template <CompareType Op>
struct CompareColumnAVX512Impl<double, Op> {
    static inline void compare(
        const double* const __restrict left, 
        const double* const __restrict right, 
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        constexpr auto pred = ComparePredicate<double, Op>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m512d vl = _mm512_loadu_pd(left + i);
            const __m512d vr = _mm512_loadu_pd(right + i);
            const __mmask8 cmp_mask = _mm512_cmp_pd_mask(vl, vr, pred);

            res_u8[i / 8] = cmp_mask;
        }
    }
};

//
template<typename T, CompareType Op>
void CompareColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, uint8_t* const __restrict res) {
    CompareColumnAVX512Impl<T, Op>::compare(left, right, size, res);
}

#define INSTANTIATE_COMPARE_COLUMN_AVX512(TTYPE,OP) \
    template void CompareColumnAVX512<TTYPE, CompareType::OP>( \
        const TTYPE* const __restrict left, \
        const TTYPE* const __restrict right, \
        const size_t size, \
        uint8_t* const __restrict res \
    );

ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX512, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX512, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX512, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX512, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX512, float)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX512, double)

#undef INSTANTIATE_COMPARE_COLUMN_AVX512

///////////////////////////////////////////////////////////////////////////

//
template <typename T, RangeType Op>
struct WithinRangeAVX512Impl {};

template<RangeType Op>
struct WithinRangeAVX512Impl<int8_t, Op> {
    static inline void within_range(
        const int8_t* const __restrict lower,
        const int8_t* const __restrict upper,
        const int8_t* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint64_t* const __restrict res_u64 = reinterpret_cast<uint64_t*>(res_u8); 
        constexpr auto pred_lower = ComparePredicate<int8_t, Range2Compare<Op>::lower>::value;
        constexpr auto pred_upper = ComparePredicate<int8_t, Range2Compare<Op>::upper>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size64 = (size / 64) * 64;
        for (size_t i = 0; i < size64; i += 64) {
            const __m512i vl = _mm512_loadu_si512(lower + i);
            const __m512i vu = _mm512_loadu_si512(upper + i);
            const __m512i vv = _mm512_loadu_si512(values + i);
            const __mmask64 cmpl_mask = _mm512_cmp_epi8_mask(vl, vv, pred_lower);
            const __mmask64 cmp_mask = _mm512_mask_cmp_epi8_mask(cmpl_mask, vv, vu, pred_upper);

            res_u64[i / 64] = cmp_mask;
        }

        // process leftovers
        if (size64 != size) {
            // 8, 16, 24, 32, 40, 48 or 56 elements to process
            const uint64_t mask = get_mask(size - size64);
            const __m512i vl = _mm512_maskz_loadu_epi8(mask, lower + size64);
            const __m512i vu = _mm512_maskz_loadu_epi8(mask, upper + size64);
            const __m512i vv = _mm512_maskz_loadu_epi8(mask, values + size64);
            const __mmask64 cmpl_mask = _mm512_cmp_epi8_mask(vl, vv, pred_lower);
            const __mmask64 cmp_mask = _mm512_mask_cmp_epi8_mask(cmpl_mask, vv, vu, pred_upper);

            const uint16_t store_mask = get_mask((size - size64) / 8);
            _mm_mask_storeu_epi8(
                res_u64 + size64 / 64, 
                store_mask, 
                _mm_setr_epi64(__m64(cmp_mask), __m64(0ULL))
            );
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX512Impl<int16_t, Op> {
    static inline void within_range(
        const int16_t* const __restrict lower,
        const int16_t* const __restrict upper,
        const int16_t* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {

        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res_u8); 
        constexpr auto pred_lower = ComparePredicate<int16_t, Range2Compare<Op>::lower>::value;
        constexpr auto pred_upper = ComparePredicate<int16_t, Range2Compare<Op>::upper>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size32 = (size / 32) * 32;
        for (size_t i = 0; i < size32; i += 32) {
            const __m512i vl = _mm512_loadu_si512(lower + i);
            const __m512i vu = _mm512_loadu_si512(upper + i);
            const __m512i vv = _mm512_loadu_si512(values + i);
            const __mmask32 cmpl_mask = _mm512_cmp_epi16_mask(vl, vv, pred_lower);
            const __mmask32 cmp_mask = _mm512_mask_cmp_epi16_mask(cmpl_mask, vv, vu, pred_upper);

            res_u32[i / 32] = cmp_mask;
        }

        // process leftovers
        if (size32 != size) {
            // 8, 16 or 24 elements to process
            const uint32_t mask = get_mask(size - size32);
            const __m512i vl = _mm512_maskz_loadu_epi16(mask, lower + size32);
            const __m512i vu = _mm512_maskz_loadu_epi16(mask, upper + size32);
            const __m512i vv = _mm512_maskz_loadu_epi16(mask, values + size32);
            const __mmask32 cmpl_mask = _mm512_cmp_epi16_mask(vl, vv, pred_lower);
            const __mmask32 cmp_mask = _mm512_mask_cmp_epi16_mask(cmpl_mask, vv, vu, pred_upper);

            const uint16_t store_mask = get_mask((size - size32) / 8);
            _mm_mask_storeu_epi8(
                res_u32 + size32 / 32, 
                store_mask, 
                _mm_setr_epi32(cmp_mask, 0, 0, 0)
            );
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX512Impl<int32_t, Op> {
    static inline void within_range(
        const int32_t* const __restrict lower,
        const int32_t* const __restrict upper,
        const int32_t* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8); 
        constexpr auto pred_lower = ComparePredicate<int32_t, Range2Compare<Op>::lower>::value;
        constexpr auto pred_upper = ComparePredicate<int32_t, Range2Compare<Op>::upper>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size16 = (size / 16) * 16;
        for (size_t i = 0; i < size16; i += 16) {
            const __m512i vl = _mm512_loadu_si512(lower + i);
            const __m512i vu = _mm512_loadu_si512(upper + i);
            const __m512i vv = _mm512_loadu_si512(values + i);
            const __mmask16 cmpl_mask = _mm512_cmp_epi32_mask(vl, vv, pred_lower);
            const __mmask16 cmp_mask = _mm512_mask_cmp_epi32_mask(cmpl_mask, vv, vu, pred_upper);

            res_u16[i / 16] = cmp_mask;
        }

        // process leftovers
        if (size16 != size) {
            // 8 elements to process
            const uint16_t mask = get_mask(size - size16);
            const __m512i vl = _mm512_maskz_loadu_epi32(mask, lower + size16);
            const __m512i vu = _mm512_maskz_loadu_epi32(mask, upper + size16);
            const __m512i vv = _mm512_maskz_loadu_epi32(mask, values + size16);
            const __mmask16 cmpl_mask = _mm512_cmp_epi32_mask(vl, vv, pred_lower);
            const __mmask16 cmp_mask = _mm512_mask_cmp_epi32_mask(cmpl_mask, vv, vu, pred_upper);

            res_u8[size16 / 8] = uint8_t(cmp_mask);
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX512Impl<int64_t, Op> {
    static inline void within_range(
        const int64_t* const __restrict lower,
        const int64_t* const __restrict upper,
        const int64_t* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        constexpr auto pred_lower = ComparePredicate<int64_t, Range2Compare<Op>::lower>::value;
        constexpr auto pred_upper = ComparePredicate<int64_t, Range2Compare<Op>::upper>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m512i vl = _mm512_loadu_si512(lower + i);
            const __m512i vu = _mm512_loadu_si512(upper + i);
            const __m512i vv = _mm512_loadu_si512(values + i);
            const __mmask8 cmpl_mask = _mm512_cmp_epi64_mask(vl, vv, pred_lower);
            const __mmask8 cmp_mask = _mm512_mask_cmp_epi64_mask(cmpl_mask, vv, vu, pred_upper);

            res_u8[i / 8] = cmp_mask;
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX512Impl<float, Op> {
    static inline void within_range(
        const float* const __restrict lower,
        const float* const __restrict upper,
        const float* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8); 
        constexpr auto pred_lower = ComparePredicate<float, Range2Compare<Op>::lower>::value;
        constexpr auto pred_upper = ComparePredicate<float, Range2Compare<Op>::upper>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size16 = (size / 16) * 16;
        for (size_t i = 0; i < size16; i += 16) {
            const __m512 vl = _mm512_loadu_ps(lower + i);
            const __m512 vu = _mm512_loadu_ps(upper + i);
            const __m512 vv = _mm512_loadu_ps(values + i);
            const __mmask16 cmpl_mask = _mm512_cmp_ps_mask(vl, vv, pred_lower);
            const __mmask16 cmp_mask = _mm512_mask_cmp_ps_mask(cmpl_mask, vv, vu, pred_upper);

            res_u16[i / 16] = cmp_mask;
        }

        // process leftovers
        if (size16 != size) {
            // process 8 elements
            const uint16_t mask = get_mask(size - size16);
            const __m512 vl = _mm512_maskz_loadu_ps(mask, lower + size16);
            const __m512 vu = _mm512_maskz_loadu_ps(mask, upper + size16);
            const __m512 vv = _mm512_maskz_loadu_ps(mask, values + size16);
            const __mmask16 cmpl_mask = _mm512_cmp_ps_mask(vl, vv, pred_lower);
            const __mmask16 cmp_mask = _mm512_mask_cmp_ps_mask(cmpl_mask, vv, vu, pred_upper);

            res_u8[size16 / 8] = uint8_t(cmp_mask);
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX512Impl<double, Op> {
    static inline void within_range(
        const double* const __restrict lower,
        const double* const __restrict upper,
        const double* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res_u8
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
        
        //
        constexpr auto pred_lower = ComparePredicate<double, Range2Compare<Op>::lower>::value;
        constexpr auto pred_upper = ComparePredicate<double, Range2Compare<Op>::upper>::value;

        // todo: aligned reads & writes

        // process big blocks
        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m512d vl = _mm512_loadu_pd(lower + i);
            const __m512d vu = _mm512_loadu_pd(upper + i);
            const __m512d vv = _mm512_loadu_pd(values + i);
            const __mmask8 cmpl_mask = _mm512_cmp_pd_mask(vl, vv, pred_lower);
            const __mmask8 cmp_mask = _mm512_mask_cmp_pd_mask(cmpl_mask, vv, vu, pred_upper);

            res_u8[i / 8] = cmp_mask;
        }
    }
};

template<typename T, RangeType Op>
void WithinRangeAVX512(const T* const __restrict lower, const T* const __restrict upper, const T* const __restrict values, const size_t size, uint8_t* const __restrict res) {
    WithinRangeAVX512Impl<T, Op>::within_range(lower, upper, values, size, res);
}

#define INSTANTIATE_WITHIN_RANGE_AVX512(TTYPE,OP) \
    template void WithinRangeAVX512<TTYPE, RangeType::OP>( \
        const TTYPE* const __restrict lower, \
        const TTYPE* const __restrict upper, \
        const TTYPE* const __restrict values, \
        const size_t size, \
        uint8_t* const __restrict res \
    );

ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX512, int8_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX512, int16_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX512, int32_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX512, int64_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX512, float)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX512, double)

#undef INSTANTIATE_WITHIN_RANGE_AVX512

///////////////////////////////////////////////////////////////////////////

#undef ALL_COMPARE_OPS
#undef ALL_RANGE_OPS

}
}
}
}
