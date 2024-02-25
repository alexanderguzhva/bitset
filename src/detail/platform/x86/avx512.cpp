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
namespace avx512 {

// a facility to run through all possible compare operations
#define ALL_COMPARE_OPS(FUNC,...) \
    FUNC(__VA_ARGS__,EQ); \
    FUNC(__VA_ARGS__,GE); \
    FUNC(__VA_ARGS__,GT); \
    FUNC(__VA_ARGS__,LE); \
    FUNC(__VA_ARGS__,LT); \
    FUNC(__VA_ARGS__,NE);

// a facility to run through all possible range operations
#define ALL_RANGE_OPS(FUNC,...) \
    FUNC(__VA_ARGS__,IncInc); \
    FUNC(__VA_ARGS__,IncExc); \
    FUNC(__VA_ARGS__,ExcInc); \
    FUNC(__VA_ARGS__,ExcExc);

// a facility to run through all possible arithmetic compare operations
#define ALL_ARITH_CMP_OPS(FUNC,...) \
    FUNC(__VA_ARGS__,Add,EQ); \
    FUNC(__VA_ARGS__,Add,NE); \
    FUNC(__VA_ARGS__,Sub,EQ); \
    FUNC(__VA_ARGS__,Sub,NE); \
    FUNC(__VA_ARGS__,Mul,EQ); \
    FUNC(__VA_ARGS__,Mul,NE); \
    FUNC(__VA_ARGS__,Div,EQ); \
    FUNC(__VA_ARGS__,Div,NE); \
    FUNC(__VA_ARGS__,Mod,EQ); \
    FUNC(__VA_ARGS__,Mod,NE);

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
template<CompareOpType Op>
bool OpCompareValImpl<int8_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int8_t* const __restrict src, 
    const size_t size, 
    const int8_t& val
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

    return true;
}

template<CompareOpType Op>
bool OpCompareValImpl<int16_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict src, 
    const size_t size, 
    const int16_t& val
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

    return true;
}

template<CompareOpType Op>
bool OpCompareValImpl<int32_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int32_t* const __restrict src, 
    const size_t size, 
    const int32_t& val 
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
        const __m256i v = _mm256_loadu_si256((const __m256i*)(src + size16));
        const __mmask8 cmp_mask = 
            _mm256_cmp_epi32_mask(v, _mm512_castsi512_si256(target), pred);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

template<CompareOpType Op>
bool OpCompareValImpl<int64_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int64_t* const __restrict src, 
    const size_t size, 
    const int64_t& val
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

    return true;
}

template<CompareOpType Op>
bool OpCompareValImpl<float, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const float* const __restrict src, 
    const size_t size, 
    const float& val
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
        const __m256 v = _mm256_loadu_ps(src + size16);
        const __mmask8 cmp_mask = 
            _mm256_cmp_ps_mask(v, _mm512_castps512_ps256(target), pred);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

template<CompareOpType Op>
bool OpCompareValImpl<double, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const double* const __restrict src, 
    const size_t size, 
    const double& val
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

    return true;
}

//
#define INSTANTIATE_COMPARE_VAL_AVX512(TTYPE,OP) \
    template bool OpCompareValImpl<TTYPE, CompareOpType::OP>::op_compare_val( \
        uint8_t* const __restrict bitmask, \
        const TTYPE* const __restrict src, \
        const size_t size, \
        const TTYPE& val \
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
template<CompareOpType Op>
bool OpCompareColumnImpl<int8_t, int8_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int8_t* const __restrict left, 
    const int8_t* const __restrict right, 
    const size_t size
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

    return true;
}

template<CompareOpType Op>
bool OpCompareColumnImpl<int16_t, int16_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict left, 
    const int16_t* const __restrict right, 
    const size_t size
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

    return true;
}

template<CompareOpType Op>
bool OpCompareColumnImpl<int32_t, int32_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int32_t* const __restrict left, 
    const int32_t* const __restrict right, 
    const size_t size
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
        const __m256i vl = _mm256_loadu_si256((const __m256i*)(left + size16));
        const __m256i vr = _mm256_loadu_si256((const __m256i*)(right + size16));
        const __mmask8 cmp_mask = _mm256_cmp_epi32_mask(vl, vr, pred);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

template<CompareOpType Op>
bool OpCompareColumnImpl<int64_t, int64_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int64_t* const __restrict left, 
    const int64_t* const __restrict right, 
    const size_t size
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

    return true;
}

template<CompareOpType Op>
bool OpCompareColumnImpl<float, float, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size
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
        const __m256 vl = _mm256_loadu_ps(left + size16);
        const __m256 vr = _mm256_loadu_ps(right + size16);
        const __mmask8 cmp_mask = _mm256_cmp_ps_mask(vl, vr, pred);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

template<CompareOpType Op>
bool OpCompareColumnImpl<double, double, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const double* const __restrict left, 
    const double* const __restrict right, 
    const size_t size
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

    return true;
}

//
#define INSTANTIATE_COMPARE_COLUMN_AVX2(TTYPE,OP) \
    template bool OpCompareColumnImpl<TTYPE, TTYPE, CompareOpType::OP>::op_compare_column( \
        uint8_t* const __restrict bitmask, \
        const TTYPE* const __restrict left, \
        const TTYPE* const __restrict right, \
        const size_t size \
    );

ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, float)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, double)

#undef INSTANTIATE_COMPARE_COLUMN_AVX2


///////////////////////////////////////////////////////////////////////////

//
template<RangeType Op>
bool OpWithinRangeColumnImpl<int8_t, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const int8_t* const __restrict lower,
    const int8_t* const __restrict upper,
    const int8_t* const __restrict values,
    const size_t size
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

    return true;
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<int16_t, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict lower,
    const int16_t* const __restrict upper,
    const int16_t* const __restrict values,
    const size_t size
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

    return true;
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<int32_t, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const int32_t* const __restrict lower,
    const int32_t* const __restrict upper,
    const int32_t* const __restrict values,
    const size_t size
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
        const __m256i vl = _mm256_loadu_si256((const __m256i*)(lower + size16));
        const __m256i vu = _mm256_loadu_si256((const __m256i*)(upper + size16));
        const __m256i vv = _mm256_loadu_si256((const __m256i*)(values + size16));
        const __mmask8 cmpl_mask = _mm256_cmp_epi32_mask(vl, vv, pred_lower);
        const __mmask8 cmp_mask = _mm256_mask_cmp_epi32_mask(cmpl_mask, vv, vu, pred_upper);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<int64_t, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const int64_t* const __restrict lower,
    const int64_t* const __restrict upper,
    const int64_t* const __restrict values,
    const size_t size
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

    return true;
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<float, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const float* const __restrict lower,
    const float* const __restrict upper,
    const float* const __restrict values,
    const size_t size
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
        const __m256 vl = _mm256_loadu_ps(lower + size16);
        const __m256 vu = _mm256_loadu_ps(upper + size16);
        const __m256 vv = _mm256_loadu_ps(values + size16);
        const __mmask8 cmpl_mask = _mm256_cmp_ps_mask(vl, vv, pred_lower);
        const __mmask8 cmp_mask = _mm256_mask_cmp_ps_mask(cmpl_mask, vv, vu, pred_upper);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<double, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const double* const __restrict lower,
    const double* const __restrict upper,
    const double* const __restrict values,
    const size_t size
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

    return true;
}

#define INSTANTIATE_WITHIN_RANGE_COLUMN_AVX512(TTYPE,OP) \
    template bool OpWithinRangeColumnImpl<TTYPE, RangeType::OP>::op_within_range_column( \
        uint8_t* const __restrict res_u8, \
        const TTYPE* const __restrict lower, \
        const TTYPE* const __restrict upper, \
        const TTYPE* const __restrict values, \
        const size_t size \
    );

ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_AVX512, int8_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_AVX512, int16_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_AVX512, int32_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_AVX512, int64_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_AVX512, float)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_AVX512, double)

#undef INSTANTIATE_WITHIN_RANGE_COLUMN_AVX512


///////////////////////////////////////////////////////////////////////////

//
template<RangeType Op>
bool OpWithinRangeValImpl<int8_t, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const int8_t& lower,
    const int8_t& upper,
    const int8_t* const __restrict values,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);
    
    //
    const __m512i lower_v = _mm512_set1_epi8(lower);
    const __m512i upper_v = _mm512_set1_epi8(upper);
    uint64_t* const __restrict res_u64 = reinterpret_cast<uint64_t*>(res_u8); 
    constexpr auto pred_lower = ComparePredicate<int8_t, Range2Compare<Op>::lower>::value;
    constexpr auto pred_upper = ComparePredicate<int8_t, Range2Compare<Op>::upper>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size64 = (size / 64) * 64;
    for (size_t i = 0; i < size64; i += 64) {
        const __m512i vv = _mm512_loadu_si512(values + i);
        const __mmask64 cmpl_mask = _mm512_cmp_epi8_mask(lower_v, vv, pred_lower);
        const __mmask64 cmp_mask = _mm512_mask_cmp_epi8_mask(cmpl_mask, vv, upper_v, pred_upper);

        res_u64[i / 64] = cmp_mask;
    }

    // process leftovers
    if (size64 != size) {
        // 8, 16, 24, 32, 40, 48 or 56 elements to process
        const uint64_t mask = get_mask(size - size64);
        const __m512i vv = _mm512_maskz_loadu_epi8(mask, values + size64);
        const __mmask64 cmpl_mask = _mm512_cmp_epi8_mask(lower_v, vv, pred_lower);
        const __mmask64 cmp_mask = _mm512_mask_cmp_epi8_mask(cmpl_mask, vv, upper_v, pred_upper);

        const uint16_t store_mask = get_mask((size - size64) / 8);
        _mm_mask_storeu_epi8(
            res_u64 + size64 / 64, 
            store_mask, 
            _mm_setr_epi64(__m64(cmp_mask), __m64(0ULL))
        );
    }

    return true;
}

template<RangeType Op>
bool OpWithinRangeValImpl<int16_t, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const int16_t& lower,
    const int16_t& upper,
    const int16_t* const __restrict values,
    const size_t size
) {

    // the restriction of the API
    assert((size % 8) == 0);
    
    //
    const __m512i lower_v = _mm512_set1_epi16(lower);
    const __m512i upper_v = _mm512_set1_epi16(upper);
    uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res_u8); 
    constexpr auto pred_lower = ComparePredicate<int16_t, Range2Compare<Op>::lower>::value;
    constexpr auto pred_upper = ComparePredicate<int16_t, Range2Compare<Op>::upper>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size32 = (size / 32) * 32;
    for (size_t i = 0; i < size32; i += 32) {
        const __m512i vv = _mm512_loadu_si512(values + i);
        const __mmask32 cmpl_mask = _mm512_cmp_epi16_mask(lower_v, vv, pred_lower);
        const __mmask32 cmp_mask = _mm512_mask_cmp_epi16_mask(cmpl_mask, vv, upper_v, pred_upper);

        res_u32[i / 32] = cmp_mask;
    }

    // process leftovers
    if (size32 != size) {
        // 8, 16 or 24 elements to process
        const uint32_t mask = get_mask(size - size32);
        const __m512i vv = _mm512_maskz_loadu_epi16(mask, values + size32);
        const __mmask32 cmpl_mask = _mm512_cmp_epi16_mask(lower_v, vv, pred_lower);
        const __mmask32 cmp_mask = _mm512_mask_cmp_epi16_mask(cmpl_mask, vv, upper_v, pred_upper);

        const uint16_t store_mask = get_mask((size - size32) / 8);
        _mm_mask_storeu_epi8(
            res_u32 + size32 / 32, 
            store_mask, 
            _mm_setr_epi32(cmp_mask, 0, 0, 0)
        );
    }

    return true;
}

template<RangeType Op>
bool OpWithinRangeValImpl<int32_t, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const int32_t& lower,
    const int32_t& upper,
    const int32_t* const __restrict values,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);
    
    //
    const __m512i lower_v = _mm512_set1_epi32(lower);
    const __m512i upper_v = _mm512_set1_epi32(upper);
    uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8); 
    constexpr auto pred_lower = ComparePredicate<int32_t, Range2Compare<Op>::lower>::value;
    constexpr auto pred_upper = ComparePredicate<int32_t, Range2Compare<Op>::upper>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const __m512i vv = _mm512_loadu_si512(values + i);
        const __mmask16 cmpl_mask = _mm512_cmp_epi32_mask(lower_v, vv, pred_lower);
        const __mmask16 cmp_mask = _mm512_mask_cmp_epi32_mask(cmpl_mask, vv, upper_v, pred_upper);

        res_u16[i / 16] = cmp_mask;
    }

    // process leftovers
    if (size16 != size) {
        // 8 elements to process
        const __m256i vv = _mm256_loadu_si256((const __m256i*)(values + size16));
        const __mmask8 cmpl_mask = _mm256_cmp_epi32_mask(_mm512_castsi512_si256(lower_v), vv, pred_lower);
        const __mmask8 cmp_mask = _mm256_mask_cmp_epi32_mask(cmpl_mask, vv, _mm512_castsi512_si256(upper_v), pred_upper);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

template<RangeType Op>
bool OpWithinRangeValImpl<int64_t, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const int64_t& lower,
    const int64_t& upper,
    const int64_t* const __restrict values,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);
    
    //
    const __m512i lower_v = _mm512_set1_epi64(lower);
    const __m512i upper_v = _mm512_set1_epi64(upper);
    constexpr auto pred_lower = ComparePredicate<int64_t, Range2Compare<Op>::lower>::value;
    constexpr auto pred_upper = ComparePredicate<int64_t, Range2Compare<Op>::upper>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const __m512i vv = _mm512_loadu_si512(values + i);
        const __mmask8 cmpl_mask = _mm512_cmp_epi64_mask(lower_v, vv, pred_lower);
        const __mmask8 cmp_mask = _mm512_mask_cmp_epi64_mask(cmpl_mask, vv, upper_v, pred_upper);

        res_u8[i / 8] = cmp_mask;
    }

    return true;
}

template<RangeType Op>
bool OpWithinRangeValImpl<float, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const float& lower,
    const float& upper,
    const float* const __restrict values,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);
    
    //
    const __m512 lower_v = _mm512_set1_ps(lower);
    const __m512 upper_v = _mm512_set1_ps(upper);
    uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8); 
    constexpr auto pred_lower = ComparePredicate<float, Range2Compare<Op>::lower>::value;
    constexpr auto pred_upper = ComparePredicate<float, Range2Compare<Op>::upper>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const __m512 vv = _mm512_loadu_ps(values + i);
        const __mmask16 cmpl_mask = _mm512_cmp_ps_mask(lower_v, vv, pred_lower);
        const __mmask16 cmp_mask = _mm512_mask_cmp_ps_mask(cmpl_mask, vv, upper_v, pred_upper);

        res_u16[i / 16] = cmp_mask;
    }

    // process leftovers
    if (size16 != size) {
        // process 8 elements
        const __m256 vv = _mm256_loadu_ps(values + size16);
        const __mmask8 cmpl_mask = _mm256_cmp_ps_mask(_mm512_castps512_ps256(lower_v), vv, pred_lower);
        const __mmask8 cmp_mask = _mm256_mask_cmp_ps_mask(cmpl_mask, vv, _mm512_castps512_ps256(upper_v), pred_upper);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

template<RangeType Op>
bool OpWithinRangeValImpl<double, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const double& lower,
    const double& upper,
    const double* const __restrict values,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);
    
    //
    const __m512d lower_v = _mm512_set1_pd(lower);
    const __m512d upper_v = _mm512_set1_pd(upper);
    constexpr auto pred_lower = ComparePredicate<double, Range2Compare<Op>::lower>::value;
    constexpr auto pred_upper = ComparePredicate<double, Range2Compare<Op>::upper>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const __m512d vv = _mm512_loadu_pd(values + i);
        const __mmask8 cmpl_mask = _mm512_cmp_pd_mask(lower_v, vv, pred_lower);
        const __mmask8 cmp_mask = _mm512_mask_cmp_pd_mask(cmpl_mask, vv, upper_v, pred_upper);

        res_u8[i / 8] = cmp_mask;
    }

    return true;
}

//
#define INSTANTIATE_WITHIN_RANGE_VAL_AVX512(TTYPE,OP) \
    template bool OpWithinRangeValImpl<TTYPE, RangeType::OP>::op_within_range_val( \
        uint8_t* const __restrict res_u8, \
        const TTYPE& lower, \
        const TTYPE& upper, \
        const TTYPE* const __restrict values, \
        const size_t size \
    );

ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_AVX512, int8_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_AVX512, int16_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_AVX512, int32_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_AVX512, int64_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_AVX512, float)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_AVX512, double)

#undef INSTANTIATE_WITHIN_RANGE_VAL_AVX512

///////////////////////////////////////////////////////////////////////////

//
template<ArithOpType AOp, CompareOpType CmpOp>
struct ArithHelperI64 {};

template<CompareOpType CmpOp>
struct ArithHelperI64<ArithOpType::Add, CmpOp> {
    static inline __mmask8 op(const __m512i left, const __m512i right, const __m512i value) {
        // left + right == value
        constexpr auto pred = ComparePredicate<int64_t, CmpOp>::value;
        return _mm512_cmp_epi64_mask(_mm512_add_epi64(left, right), value, pred);
    }
};

template<CompareOpType CmpOp>
struct ArithHelperI64<ArithOpType::Sub, CmpOp> {
    static inline __mmask8 op(const __m512i left, const __m512i right, const __m512i value) {
        // left - right == value
        constexpr auto pred = ComparePredicate<int64_t, CmpOp>::value;
        return _mm512_cmp_epi64_mask(_mm512_sub_epi64(left, right), value, pred);
    }
};

template<CompareOpType CmpOp>
struct ArithHelperI64<ArithOpType::Mul, CmpOp> {
    static inline __mmask8 op(const __m512i left, const __m512i right, const __m512i value) {
        // left * right == value
        constexpr auto pred = ComparePredicate<int64_t, CmpOp>::value;
        return _mm512_cmp_epi64_mask(_mm512_mullo_epi64(left, right), value, pred);
    }
};

//
template<ArithOpType AOp, CompareOpType CmpOp>
struct ArithHelperF32 {};

template<CompareOpType CmpOp>
struct ArithHelperF32<ArithOpType::Add, CmpOp> {
    static inline __mmask16 op(const __m512 left, const __m512 right, const __m512 value) {
        // left + right == value
        constexpr auto pred = ComparePredicate<float, CmpOp>::value;
        return _mm512_cmp_ps_mask(_mm512_add_ps(left, right), value, pred);
    }
};

template<CompareOpType CmpOp>
struct ArithHelperF32<ArithOpType::Sub, CmpOp> {
    static inline __mmask16 op(const __m512 left, const __m512 right, const __m512 value) {
        // left - right == value
        constexpr auto pred = ComparePredicate<float, CmpOp>::value;
        return _mm512_cmp_ps_mask(_mm512_sub_ps(left, right), value, pred);
    }
};

template<CompareOpType CmpOp>
struct ArithHelperF32<ArithOpType::Mul, CmpOp> {
    static inline __mmask16 op(const __m512 left, const __m512 right, const __m512 value) {
        // left * right == value
        constexpr auto pred = ComparePredicate<float, CmpOp>::value;
        return _mm512_cmp_ps_mask(_mm512_mul_ps(left, right), value, pred);
    }
};

template<CompareOpType CmpOp>
struct ArithHelperF32<ArithOpType::Div, CmpOp> {
    static inline __mmask16 op(const __m512 left, const __m512 right, const __m512 value) {
        // left == right * value
        constexpr auto pred = ComparePredicate<float, CmpOp>::value;
        return _mm512_cmp_ps_mask(left, _mm512_mul_ps(right, value), pred);
    }
};

//
template<ArithOpType AOp, CompareOpType CmpOp>
struct ArithHelperF64 {};

template<CompareOpType CmpOp>
struct ArithHelperF64<ArithOpType::Add, CmpOp> {
    static inline __mmask8 op(const __m512d left, const __m512d right, const __m512d value) {
        // left + right == value
        constexpr auto pred = ComparePredicate<double, CmpOp>::value;
        return _mm512_cmp_pd_mask(_mm512_add_pd(left, right), value, pred);
    }
};

template<CompareOpType CmpOp>
struct ArithHelperF64<ArithOpType::Sub, CmpOp> {
    static inline __mmask8 op(const __m512d left, const __m512d right, const __m512d value) {
        // left - right == value
        constexpr auto pred = ComparePredicate<double, CmpOp>::value;
        return _mm512_cmp_pd_mask(_mm512_sub_pd(left, right), value, pred);
    }
};

template<CompareOpType CmpOp>
struct ArithHelperF64<ArithOpType::Mul, CmpOp> {
    static inline __mmask8 op(const __m512d left, const __m512d right, const __m512d value) {
        // left * right == value
        constexpr auto pred = ComparePredicate<double, CmpOp>::value;
        return _mm512_cmp_pd_mask(_mm512_mul_pd(left, right), value, pred);
    }
};

template<CompareOpType CmpOp>
struct ArithHelperF64<ArithOpType::Div, CmpOp> {
    static inline __mmask8 op(const __m512d left, const __m512d right, const __m512d value) {
        // left == right * value
        constexpr auto pred = ComparePredicate<double, CmpOp>::value;
        return _mm512_cmp_pd_mask(left, _mm512_mul_pd(right, value), pred);
    }
};


// todo: Mul, Div, Mod

#define NOT_IMPLEMENTED_OP_ARITH_COMPARE(TTYPE, AOP, CMPOP) \
    template<> \
    bool OpArithCompareImpl<TTYPE, ArithOpType::AOP, CompareOpType::CMPOP>::op_arith_compare( \
        uint8_t* const __restrict res_u8, \
        const TTYPE* const __restrict src, \
        const ArithHighPrecisionType<TTYPE>& right_operand, \
        const ArithHighPrecisionType<TTYPE>& value, \
        const size_t size \
    ) { \
        return false; \
    }

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Div, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Div, NE)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Mod, NE)

template<ArithOpType AOp, CompareOpType CmpOp>
bool OpArithCompareImpl<int8_t, AOp, CmpOp>::op_arith_compare(
    uint8_t* const __restrict res_u8,
    const int8_t* const __restrict src,
    const ArithHighPrecisionType<int8_t>& right_operand,
    const ArithHighPrecisionType<int8_t>& value,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);
    static_assert(std::is_same_v<int64_t, ArithHighPrecisionType<int64_t>>);

    //
    const __m512i right_v = _mm512_set1_epi64(right_operand);
    const __m512i value_v = _mm512_set1_epi64(value);
    constexpr auto pred = ComparePredicate<int64_t, CmpOp>::value;

    // process big blocks
    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const __m128i vs = _mm_loadu_si128((const __m128i*)(src + i));
        const __m512i v0s = _mm512_cvtepi8_epi64(_mm_unpacklo_epi64(vs, _mm_setzero_si128()));
        const __m512i v1s = _mm512_cvtepi8_epi64(_mm_unpackhi_epi64(vs, _mm_setzero_si128()));
        const __mmask8 cmp_mask0 = ArithHelperI64<AOp, CmpOp>::op(v0s, right_v, value_v);
        const __mmask8 cmp_mask1 = ArithHelperI64<AOp, CmpOp>::op(v1s, right_v, value_v);

        res_u8[i / 8 + 0] = cmp_mask0;
        res_u8[i / 8 + 1] = cmp_mask1;
    }

    // process leftovers
    if (size16 != size) {
        // process 8 elements
        const int64_t* const __restrict src64 = (const int64_t*)(src + size16);
        const __m128i vs = _mm_set_epi64x(0, *src64);
        const __m512i v0s = _mm512_cvtepi16_epi64(vs);
        const __mmask8 cmp_mask = ArithHelperI64<AOp, CmpOp>::op(v0s, right_v, value_v);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Div, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Div, NE)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Mod, NE)

template<ArithOpType AOp, CompareOpType CmpOp>
bool OpArithCompareImpl<int16_t, AOp, CmpOp>::op_arith_compare(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict src,
    const ArithHighPrecisionType<int16_t>& right_operand,
    const ArithHighPrecisionType<int16_t>& value,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);
    static_assert(std::is_same_v<int64_t, ArithHighPrecisionType<int64_t>>);

    //
    const __m512i right_v = _mm512_set1_epi64(right_operand);
    const __m512i value_v = _mm512_set1_epi64(value);
    constexpr auto pred = ComparePredicate<int64_t, CmpOp>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const __m256i vs = _mm256_loadu_si256((const __m256i*)(src + i));
        const __m512i v0s = _mm512_cvtepi16_epi64(_mm256_extracti128_si256(vs, 0));
        const __m512i v1s = _mm512_cvtepi16_epi64(_mm256_extracti128_si256(vs, 1));
        const __mmask8 cmp_mask0 = ArithHelperI64<AOp, CmpOp>::op(v0s, right_v, value_v);
        const __mmask8 cmp_mask1 = ArithHelperI64<AOp, CmpOp>::op(v1s, right_v, value_v);

        res_u8[i / 8 + 0] = cmp_mask0;
        res_u8[i / 8 + 1] = cmp_mask1;
    }

    // process leftovers
    if (size16 != size) {
        // process 8 elements
        const __m128i vs = _mm_loadu_si128((const __m128i*)(src + size16));
        const __m512i v0s = _mm512_cvtepi16_epi64(vs);
        const __mmask8 cmp_mask = ArithHelperI64<AOp, CmpOp>::op(v0s, right_v, value_v);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Div, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Div, NE)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Mod, NE)

template<ArithOpType AOp, CompareOpType CmpOp>
bool OpArithCompareImpl<int32_t, AOp, CmpOp>::op_arith_compare(
    uint8_t* const __restrict res_u8,
    const int32_t* const __restrict src,
    const ArithHighPrecisionType<int32_t>& right_operand,
    const ArithHighPrecisionType<int32_t>& value,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);
    static_assert(std::is_same_v<int64_t, ArithHighPrecisionType<int64_t>>);

    //
    const __m512i right_v = _mm512_set1_epi64(right_operand);
    const __m512i value_v = _mm512_set1_epi64(value);
    constexpr auto pred = ComparePredicate<int64_t, CmpOp>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const __m512i vs = _mm512_loadu_si512((const __m512i*)(src + i));
        const __m512i v0s = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(vs, 0));
        const __m512i v1s = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(vs, 1));
        const __mmask8 cmp_mask0 = ArithHelperI64<AOp, CmpOp>::op(v0s, right_v, value_v);
        const __mmask8 cmp_mask1 = ArithHelperI64<AOp, CmpOp>::op(v1s, right_v, value_v);

        res_u8[i / 8 + 0] = cmp_mask0;
        res_u8[i / 8 + 1] = cmp_mask1;
    }

    // process leftovers
    if (size16 != size) {
        // process 8 elements
        const __m256i vs = _mm256_loadu_si256((const __m256i*)(src + size16));
        const __m512i v0s = _mm512_cvtepi32_epi64(vs);
        const __mmask8 cmp_mask = ArithHelperI64<AOp, CmpOp>::op(v0s, right_v, value_v);

        res_u8[size16 / 8] = cmp_mask;
    }

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Div, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Div, NE)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Mod, NE)

template<ArithOpType AOp, CompareOpType CmpOp>
bool OpArithCompareImpl<int64_t, AOp, CmpOp>::op_arith_compare(
    uint8_t* const __restrict res_u8,
    const int64_t* const __restrict src,
    const ArithHighPrecisionType<int64_t>& right_operand,
    const ArithHighPrecisionType<int64_t>& value,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);
    static_assert(std::is_same_v<int64_t, ArithHighPrecisionType<int64_t>>);

    //
    const __m512i right_v = _mm512_set1_epi64(right_operand);
    const __m512i value_v = _mm512_set1_epi64(value);
    constexpr auto pred = ComparePredicate<int64_t, CmpOp>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const __m512i v0s = _mm512_loadu_si512((const __m512i*)(src + i));
        const __mmask8 cmp_mask = ArithHelperI64<AOp, CmpOp>::op(v0s, right_v, value_v);

        res_u8[i / 8] = cmp_mask;
    }

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(float, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(float, Mod, NE)

template<ArithOpType AOp, CompareOpType CmpOp>
bool OpArithCompareImpl<float, AOp, CmpOp>::op_arith_compare(
    uint8_t* const __restrict res_u8,
    const float* const __restrict src,
    const ArithHighPrecisionType<float>& right_operand,
    const ArithHighPrecisionType<float>& value,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    const __m512 right_v = _mm512_set1_ps(right_operand);
    const __m512 value_v = _mm512_set1_ps(value);
    uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8);
    constexpr auto pred = ComparePredicate<float, CmpOp>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const __m512 v0s = _mm512_loadu_ps(src + i);
        const __mmask16 cmp_mask = ArithHelperF32<AOp, CmpOp>::op(v0s, right_v, value_v);
        res_u16[i / 16] = cmp_mask;
    }

    // process leftovers
    if (size16 != size) {
        // process 8 elements
        const __m256 vs = _mm256_loadu_ps(src + size16);
        const __m512 v0s = _mm512_castps256_ps512(vs);
        const __mmask16 cmp_mask = ArithHelperF32<AOp, CmpOp>::op(v0s, right_v, value_v);
        res_u8[size16 / 8] = uint8_t(cmp_mask);
    }

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(double, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(double, Mod, NE)

template<ArithOpType AOp, CompareOpType CmpOp>
bool OpArithCompareImpl<double, AOp, CmpOp>::op_arith_compare(
    uint8_t* const __restrict res_u8,
    const double* const __restrict src,
    const ArithHighPrecisionType<double>& right_operand,
    const ArithHighPrecisionType<double>& value,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    const __m512d right_v = _mm512_set1_pd(right_operand);
    const __m512d value_v = _mm512_set1_pd(value);
    constexpr auto pred = ComparePredicate<double, CmpOp>::value;

    // todo: aligned reads & writes

    // process big blocks
    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const __m512d v0s = _mm512_loadu_pd(src + i);
        const __mmask8 cmp_mask = ArithHelperF64<AOp, CmpOp>::op(v0s, right_v, value_v);

        res_u8[i / 8] = cmp_mask;
    }

    return true;
}

//
#undef NOT_IMPLEMENTED_OP_ARITH_COMPARE

//
#define INSTANTIATE_ARITH_COMPARE_AVX512(TTYPE,OP,CMP) \
    template bool OpArithCompareImpl<TTYPE, ArithOpType::OP, CompareOpType::CMP>::op_arith_compare( \
        uint8_t* const __restrict res_u8, \
        const TTYPE* const __restrict src, \
        const ArithHighPrecisionType<TTYPE>& right_operand, \
        const ArithHighPrecisionType<TTYPE>& value, \
        const size_t size \
    );

ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_AVX512, int8_t)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_AVX512, int16_t)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_AVX512, int32_t)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_AVX512, int64_t)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_AVX512, float)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_AVX512, double)

#undef INSTANTIATE_ARITH_COMPARE_AVX512


///////////////////////////////////////////////////////////////////////////

#undef ALL_COMPARE_OPS
#undef ALL_RANGE_OPS
#undef ALL_ARITH_CMP_OPS


}
}
}
}
}
