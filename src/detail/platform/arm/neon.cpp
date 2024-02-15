#include "neon.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <arm_neon.h>

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {
namespace neon {

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

// this function is missing somewhy
inline uint64x2_t vmvnq_u64(const uint64x2_t value) {
    const uint64x2_t m1 = vreinterpretq_u64_u32(vdupq_n_u32(0xFFFFFFFF));
    return veorq_u64(value, m1);
}

// draft: movemask functions from SSE2SIMD library.
// todo: can this be made better?

// todo: optimize
inline uint8_t movemask(const uint8x8_t cmp) {
    static const int8_t shifts[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    // shift right by 7, leaving 1 bit
    const uint8x8_t sh = vshr_n_u8(cmp, 7);
    // load shifts
    const int8x8_t shifts_v = vld1_s8(shifts);
    // shift each of 8 lanes with 1 bit values differently
    const uint8x8_t shifted_bits = vshl_u8(sh, shifts_v);
    // horizontal sum of bits on different positions
    return vaddv_u8(shifted_bits);
}

// todo: optimize
inline uint16_t movemask(const uint8x16_t cmp) {
    uint16x8_t high_bits = vreinterpretq_u16_u8(vshrq_n_u8(cmp, 7));
    uint32x4_t paired16 = vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 7));
    uint64x2_t paired32 = vreinterpretq_u64_u32(vsraq_n_u32(paired16, paired16, 14));
    uint8x16_t paired64 = vreinterpretq_u8_u64(vsraq_n_u64(paired32, paired32, 28));
    return vgetq_lane_u8(paired64, 0) | ((int)vgetq_lane_u8(paired64, 8) << 8);
}

// todo: optimize
inline uint32_t movemask(const uint8x16x2_t cmp) {
    return (uint32_t)(movemask(cmp.val[0])) | ((uint32_t)(movemask(cmp.val[1])) << 16);
}

// todo: optimize
inline uint8_t movemask(const uint16x8_t cmp) {
    static const int16_t shifts[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    // shift right by 15, leaving 1 bit
    const uint16x8_t sh = vshrq_n_u16(cmp, 15);
    // load shifts
    const int16x8_t shifts_v = vld1q_s16(shifts);
    // shift each of 8 lanes with 1 bit values differently
    const uint16x8_t shifted_bits = vshlq_u16(sh, shifts_v);
    // horizontal sum of bits on different positions
    return vaddvq_u16(shifted_bits);
}

// todo: optimize
inline uint16_t movemask(const uint16x8x2_t cmp) {
    return (uint16_t)(movemask(cmp.val[0])) | ((uint16_t)(movemask(cmp.val[1])) << 8);
}

// todo: optimize
inline uint32_t movemask(const uint32x4_t cmp) {
    static const int32_t shifts[4] = {0, 1, 2, 3};
    // shift right by 31, leaving 1 bit
    const uint32x4_t sh = vshrq_n_u32(cmp, 31);
    // load shifts
    const int32x4_t shifts_v = vld1q_s32(shifts);
    // shift each of 4 lanes with 1 bit values differently
    const uint32x4_t shifted_bits = vshlq_u32(sh, shifts_v);
    // horizontal sum of bits on different positions
    return vaddvq_u32(shifted_bits);
}

// todo: optimize
inline uint32_t movemask(const uint32x4x2_t cmp) {
    return movemask(cmp.val[0]) | (movemask(cmp.val[1]) << 4);
}

// todo: optimize
inline uint8_t movemask(const uint64x2_t cmp) {
    // shift right by 63, leaving 1 bit
    const uint64x2_t sh = vshrq_n_u64(cmp, 63);
    return vgetq_lane_u64(sh, 0) | (vgetq_lane_u64(sh, 1) << 1);
}

// todo: optimize
inline uint8_t movemask(const uint64x2x4_t cmp) {
    return movemask(cmp.val[0]) | (movemask(cmp.val[1]) << 2) | (movemask(cmp.val[2]) << 4) | (movemask(cmp.val[3]) << 6);
}

//
template<CompareType Op>
struct CmpHelper{};

template<>
struct CmpHelper<CompareType::EQ> {
    static inline uint8x8_t compare(const int8x8_t a, const int8x8_t b) {
        return vceq_s8(a, b);
    }

    static inline uint8x16x2_t compare(const int8x16x2_t a, const int8x16x2_t b) {
        return {vceqq_s8(a.val[0], b.val[0]), vceqq_s8(a.val[1], b.val[1])};
    }

    static inline uint16x8_t compare(const int16x8_t a, const int16x8_t b) {
        return vceqq_s16(a, b);
    }

    static inline uint16x8x2_t compare(const int16x8x2_t a, const int16x8x2_t b) {
        return {vceqq_s16(a.val[0], b.val[0]), vceqq_s16(a.val[1], b.val[1])};
    }

    static inline uint32x4x2_t compare(const int32x4x2_t a, const int32x4x2_t b) {
        return {vceqq_s32(a.val[0], b.val[0]), vceqq_s32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const int64x2x4_t a, const int64x2x4_t b) {
        return {vceqq_u64(a.val[0], b.val[0]), vceqq_u64(a.val[1], b.val[1]),
                vceqq_u64(a.val[2], b.val[2]), vceqq_u64(a.val[3], b.val[3])};
    }

    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vceqq_f32(a.val[0], b.val[0]), vceqq_f32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const float64x2x4_t a, const float64x2x4_t b) {
        return {vceqq_f64(a.val[0], b.val[0]), vceqq_f64(a.val[1], b.val[1]),
                vceqq_f64(a.val[2], b.val[2]), vceqq_f64(a.val[3], b.val[3])};
    }
};

template<>
struct CmpHelper<CompareType::GE> {
    static inline uint8x8_t compare(const int8x8_t a, const int8x8_t b) {
        return vcge_s8(a, b);
    }

    static inline uint8x16x2_t compare(const int8x16x2_t a, const int8x16x2_t b) {
        return {vcgeq_s8(a.val[0], b.val[0]), vcgeq_s8(a.val[1], b.val[1])};
    }

    static inline uint16x8_t compare(const int16x8_t a, const int16x8_t b) {
        return vcgeq_s16(a, b);
    }

    static inline uint16x8x2_t compare(const int16x8x2_t a, const int16x8x2_t b) {
        return {vcgeq_s16(a.val[0], b.val[0]), vcgeq_s16(a.val[1], b.val[1])};
    }

    static inline uint32x4x2_t compare(const int32x4x2_t a, const int32x4x2_t b) {
        return {vcgeq_s32(a.val[0], b.val[0]), vcgeq_s32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const int64x2x4_t a, const int64x2x4_t b) {
        return {vcgeq_u64(a.val[0], b.val[0]), vcgeq_u64(a.val[1], b.val[1]),
                vcgeq_u64(a.val[2], b.val[2]), vcgeq_u64(a.val[3], b.val[3])};
    }

    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vcgeq_f32(a.val[0], b.val[0]), vcgeq_f32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const float64x2x4_t a, const float64x2x4_t b) {
        return {vcgeq_f64(a.val[0], b.val[0]), vcgeq_f64(a.val[1], b.val[1]),
                vcgeq_f64(a.val[2], b.val[2]), vcgeq_f64(a.val[3], b.val[3])};
    }
};

template<>
struct CmpHelper<CompareType::GT> {
    static inline uint8x8_t compare(const int8x8_t a, const int8x8_t b) {
        return vcgt_s8(a, b);
    }

    static inline uint8x16x2_t compare(const int8x16x2_t a, const int8x16x2_t b) {
        return {vcgtq_s8(a.val[0], b.val[0]), vcgtq_s8(a.val[1], b.val[1])};
    }

    static inline uint16x8_t compare(const int16x8_t a, const int16x8_t b) {
        return vcgtq_s16(a, b);
    }

    static inline uint16x8x2_t compare(const int16x8x2_t a, const int16x8x2_t b) {
        return {vcgtq_s16(a.val[0], b.val[0]), vcgtq_s16(a.val[1], b.val[1])};
    }

    static inline uint32x4x2_t compare(const int32x4x2_t a, const int32x4x2_t b) {
        return {vcgtq_s32(a.val[0], b.val[0]), vcgtq_s32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const int64x2x4_t a, const int64x2x4_t b) {
        return {vcgtq_u64(a.val[0], b.val[0]), vcgtq_u64(a.val[1], b.val[1]),
                vcgtq_u64(a.val[2], b.val[2]), vcgtq_u64(a.val[3], b.val[3])};
    }

    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vcgtq_f32(a.val[0], b.val[0]), vcgtq_f32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const float64x2x4_t a, const float64x2x4_t b) {
        return {vcgtq_f64(a.val[0], b.val[0]), vcgtq_f64(a.val[1], b.val[1]),
                vcgtq_f64(a.val[2], b.val[2]), vcgtq_f64(a.val[3], b.val[3])};
    }
};

template<>
struct CmpHelper<CompareType::LE> {
    static inline uint8x8_t compare(const int8x8_t a, const int8x8_t b) {
        return vcle_s8(a, b);
    }

    static inline uint8x16x2_t compare(const int8x16x2_t a, const int8x16x2_t b) {
        return {vcleq_s8(a.val[0], b.val[0]), vcleq_s8(a.val[1], b.val[1])};
    }

    static inline uint16x8_t compare(const int16x8_t a, const int16x8_t b) {
        return vcleq_s16(a, b);
    }

    static inline uint16x8x2_t compare(const int16x8x2_t a, const int16x8x2_t b) {
        return {vcleq_s16(a.val[0], b.val[0]), vcleq_s16(a.val[1], b.val[1])};
    }

    static inline uint32x4x2_t compare(const int32x4x2_t a, const int32x4x2_t b) {
        return {vcleq_s32(a.val[0], b.val[0]), vcleq_s32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const int64x2x4_t a, const int64x2x4_t b) {
        return {vcleq_s64(a.val[0], b.val[0]), vcleq_s64(a.val[1], b.val[1]),
                vcleq_s64(a.val[2], b.val[2]), vcleq_s64(a.val[3], b.val[3])};
    }

    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vcleq_f32(a.val[0], b.val[0]), vcleq_f32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const float64x2x4_t a, const float64x2x4_t b) {
        return {vcleq_f64(a.val[0], b.val[0]), vcleq_f64(a.val[1], b.val[1]),
                vcleq_f64(a.val[2], b.val[2]), vcleq_f64(a.val[3], b.val[3])};
    }
};

template<>
struct CmpHelper<CompareType::LT> {
    static inline uint8x8_t compare(const int8x8_t a, const int8x8_t b) {
        return vclt_s8(a, b);
    }

    static inline uint8x16x2_t compare(const int8x16x2_t a, const int8x16x2_t b) {
        return {vcltq_s8(a.val[0], b.val[0]), vcltq_s8(a.val[1], b.val[1])};
    }

    static inline uint16x8_t compare(const int16x8_t a, const int16x8_t b) {
        return vcltq_s16(a, b);
    }

    static inline uint16x8x2_t compare(const int16x8x2_t a, const int16x8x2_t b) {
        return {vcltq_s16(a.val[0], b.val[0]), vcltq_s16(a.val[1], b.val[1])};
    }

    static inline uint32x4x2_t compare(const int32x4x2_t a, const int32x4x2_t b) {
        return {vcltq_s32(a.val[0], b.val[0]), vcltq_s32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const int64x2x4_t a, const int64x2x4_t b) {
        return {vcltq_u64(a.val[0], b.val[0]), vcltq_u64(a.val[1], b.val[1]),
                vcltq_u64(a.val[2], b.val[2]), vcltq_u64(a.val[3], b.val[3])};
    }

    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vcltq_f32(a.val[0], b.val[0]), vcltq_f32(a.val[1], b.val[1])};
    }

    static inline uint64x2x4_t compare(const float64x2x4_t a, const float64x2x4_t b) {
        return {vcltq_f64(a.val[0], b.val[0]), vcltq_f64(a.val[1], b.val[1]),
                vcltq_f64(a.val[2], b.val[2]), vcltq_f64(a.val[3], b.val[3])};
    }
};

template<>
struct CmpHelper<CompareType::NEQ> {
    static inline uint8x8_t compare(const int8x8_t a, const int8x8_t b) {
        return vmvn_u8(vceq_s8(a, b));
    }

    static inline uint8x16x2_t compare(const int8x16x2_t a, const int8x16x2_t b) {
        return {vmvnq_u8(vceqq_s8(a.val[0], b.val[0])), vmvnq_u8(vceqq_s8(a.val[1], b.val[1]))};
    }

    static inline uint16x8_t compare(const int16x8_t a, const int16x8_t b) {
        return vmvnq_u16(vceqq_s16(a, b));
    }

    static inline uint16x8x2_t compare(const int16x8x2_t a, const int16x8x2_t b) {
        return {vmvnq_u16(vceqq_s16(a.val[0], b.val[0])), vmvnq_u16(vceqq_s16(a.val[1], b.val[1]))};
    }

    static inline uint32x4x2_t compare(const int32x4x2_t a, const int32x4x2_t b) {
        return {vmvnq_u32(vceqq_s32(a.val[0], b.val[0])), vmvnq_u32(vceqq_s32(a.val[1], b.val[1]))};
    }

    static inline uint64x2x4_t compare(const int64x2x4_t a, const int64x2x4_t b) {
        return {vmvnq_u64(vceqq_s64(a.val[0], b.val[0])), vmvnq_u64(vceqq_s64(a.val[1], b.val[1])),
                vmvnq_u64(vceqq_s64(a.val[2], b.val[2])), vmvnq_u64(vceqq_s64(a.val[3], b.val[3]))};
    }

    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vmvnq_u32(vceqq_f32(a.val[0], b.val[0])), vmvnq_u32(vceqq_f32(a.val[1], b.val[1]))};
    }

    static inline uint64x2x4_t compare(const float64x2x4_t a, const float64x2x4_t b) {
        return {vmvnq_u64(vceqq_f64(a.val[0], b.val[0])), vmvnq_u64(vceqq_f64(a.val[1], b.val[1])),
                vmvnq_u64(vceqq_f64(a.val[2], b.val[2])), vmvnq_u64(vceqq_f64(a.val[3], b.val[3]))};
    }
};

///////////////////////////////////////////////////////////////////////////

//
template<CompareType Op>
bool OpCompareValImpl<int8_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int8_t* const __restrict src, 
    const size_t size, 
    const int8_t& val
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res_u8);
    const int8x16x2_t target = {vdupq_n_s8(val), vdupq_n_s8(val)};

    // todo: aligned reads & writes

    const size_t size32 = (size / 32) * 32;
    for (size_t i = 0; i < size32; i += 32) {
        const int8x16x2_t v0 = {vld1q_s8(src + i), vld1q_s8(src + i + 16)};
        const uint8x16x2_t cmp = CmpHelper<Op>::compare(v0, target);
        const uint32_t mmask = movemask(cmp);

        res_u32[i / 32] = mmask;
    }

    for (size_t i = size32; i < size; i += 8) {
        const int8x8_t v0 = vld1_s8(src + i);
        const uint8x8_t cmp = CmpHelper<Op>::compare(v0, vdup_n_s8(val));
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

template<CompareType Op>
bool OpCompareValImpl<int16_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict src, 
    const size_t size, 
    const int16_t& val
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8);
    const int16x8x2_t target = {vdupq_n_s16(val), vdupq_n_s16(val)};

    // todo: aligned reads & writes

    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const int16x8x2_t v0 = {vld1q_s16(src + i), vld1q_s16(src + i + 8)};
        const uint16x8x2_t cmp = CmpHelper<Op>::compare(v0, target);
        const uint16_t mmask = movemask(cmp);

        res_u16[i / 16] = mmask;
    }

    if (size16 != size) {
        // 8 elements to process
        const int16x8_t v0 = vld1q_s16(src + size16);
        const uint16x8_t cmp = CmpHelper<Op>::compare(v0, target.val[0]);
        const uint8_t mmask = movemask(cmp);

        res_u8[size16 / 8] = mmask;
    }

    return true;
}

template<CompareType Op>
bool OpCompareValImpl<int32_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int32_t* const __restrict src, 
    const size_t size, 
    const int32_t& val 
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    const int32x4x2_t target = {vdupq_n_s32(val), vdupq_n_s32(val)};

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const int32x4x2_t v0 = {vld1q_s32(src + i), vld1q_s32(src + i + 4)};
        const uint32x4x2_t cmp = CmpHelper<Op>::compare(v0, target);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

template<CompareType Op>
bool OpCompareValImpl<int64_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int64_t* const __restrict src, 
    const size_t size, 
    const int64_t& val
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    const int64x2x4_t target = {vdupq_n_s64(val), vdupq_n_s64(val), vdupq_n_s64(val), vdupq_n_s64(val)};

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const int64x2x4_t v0 = {vld1q_s64(src + i), vld1q_s64(src + i + 2), vld1q_s64(src + i + 4), vld1q_s64(src + i + 6)};
        const uint64x2x4_t cmp = CmpHelper<Op>::compare(v0, target);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

template<CompareType Op>
bool OpCompareValImpl<float, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const float* const __restrict src, 
    const size_t size, 
    const float& val
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    const float32x4x2_t target = {vdupq_n_f32(val), vdupq_n_f32(val)};

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const float32x4x2_t v0 = {vld1q_f32(src + i), vld1q_f32(src + i + 4)};
        const uint32x4x2_t cmp = CmpHelper<Op>::compare(v0, target);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

template<CompareType Op>
bool OpCompareValImpl<double, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const double* const __restrict src, 
    const size_t size, 
    const double& val 
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    const float64x2x4_t target = {vdupq_n_f64(val), vdupq_n_f64(val), vdupq_n_f64(val), vdupq_n_f64(val)};

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const float64x2x4_t v0 = {vld1q_f64(src + i), vld1q_f64(src + i + 2), vld1q_f64(src + i + 4), vld1q_f64(src + i + 6)};
        const uint64x2x4_t cmp = CmpHelper<Op>::compare(v0, target);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

//
#define INSTANTIATE_COMPARE_VAL_NEON(TTYPE,OP) \
    template bool OpCompareValImpl<TTYPE, CompareType::OP>::op_compare_val( \
        uint8_t* const __restrict bitmask, \
        const TTYPE* const __restrict src, \
        const size_t size, \
        const TTYPE& val \
    );

ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_NEON, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_NEON, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_NEON, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_NEON, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_NEON, float)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_NEON, double)

#undef INSTANTIATE_COMPARE_VAL_NEON


///////////////////////////////////////////////////////////////////////////

//
template<CompareType Op>
bool OpCompareColumnImpl<int8_t, int8_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int8_t* const __restrict left, 
    const int8_t* const __restrict right, 
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res_u8);

    // todo: aligned reads & writes

    const size_t size32 = (size / 32) * 32;
    for (size_t i = 0; i < size32; i += 32) {
        const int8x16x2_t v0l = {vld1q_s8(left + i), vld1q_s8(left + i + 16)};
        const int8x16x2_t v0r = {vld1q_s8(right + i), vld1q_s8(right + i + 16)};
        const uint8x16x2_t cmp = CmpHelper<Op>::compare(v0l, v0r);
        const uint32_t mmask = movemask(cmp);

        res_u32[i / 32] = mmask;
    }

    for (size_t i = size32; i < size; i += 8) {
        const int8x8_t v0l = vld1_s8(left + i);
        const int8x8_t v0r = vld1_s8(right + i);
        const uint8x8_t cmp = CmpHelper<Op>::compare(v0l, v0r);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }    

    return true;
}

template<CompareType Op>
bool OpCompareColumnImpl<int16_t, int16_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict left, 
    const int16_t* const __restrict right, 
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8);

    // todo: aligned reads & writes

    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const int16x8x2_t v0l = {vld1q_s16(left + i), vld1q_s16(left + i + 8)};
        const int16x8x2_t v0r = {vld1q_s16(right + i), vld1q_s16(right + i + 8)};
        const uint16x8x2_t cmp = CmpHelper<Op>::compare(v0l, v0r);
        const uint16_t mmask = movemask(cmp);

        res_u16[i / 16] = mmask;
    }

    if (size16 != size) {
        // 8 elements to process
        const int16x8_t v0l = vld1q_s16(left + size16);
        const int16x8_t v0r = vld1q_s16(right + size16);
        const uint16x8_t cmp = CmpHelper<Op>::compare(v0l, v0r);
        const uint8_t mmask = movemask(cmp);

        res_u8[size16 / 8] = mmask;
    }

    return true;
}

template<CompareType Op>
bool OpCompareColumnImpl<int32_t, int32_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int32_t* const __restrict left, 
    const int32_t* const __restrict right, 
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const int32x4x2_t v0l = {vld1q_s32(left + i), vld1q_s32(left + i + 4)};
        const int32x4x2_t v0r = {vld1q_s32(right + i), vld1q_s32(right + i + 4)};
        const uint32x4x2_t cmp = CmpHelper<Op>::compare(v0l, v0r);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

template<CompareType Op>
bool OpCompareColumnImpl<int64_t, int64_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int64_t* const __restrict left, 
    const int64_t* const __restrict right, 
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const int64x2x4_t v0l = {vld1q_s64(left + i), vld1q_s64(left + i + 2), vld1q_s64(left + i + 4), vld1q_s64(left + i + 6)};
        const int64x2x4_t v0r = {vld1q_s64(right + i), vld1q_s64(right + i + 2), vld1q_s64(right + i + 4), vld1q_s64(right + i + 6)};
        const uint64x2x4_t cmp = CmpHelper<Op>::compare(v0l, v0r);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

template<CompareType Op>
bool OpCompareColumnImpl<float, float, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const float32x4x2_t v0l = {vld1q_f32(left + i), vld1q_f32(left + i + 4)};
        const float32x4x2_t v0r = {vld1q_f32(right + i), vld1q_f32(right + i + 4)};
        const uint32x4x2_t cmp = CmpHelper<Op>::compare(v0l, v0r);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

template<CompareType Op>
bool OpCompareColumnImpl<double, double, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const double* const __restrict left, 
    const double* const __restrict right, 
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const float64x2x4_t v0l = {vld1q_f64(left + i), vld1q_f64(left + i + 2), vld1q_f64(left + i + 4), vld1q_f64(left + i + 6)};
        const float64x2x4_t v0r = {vld1q_f64(right + i), vld1q_f64(right + i + 2), vld1q_f64(right + i + 4), vld1q_f64(right + i + 6)};
        const uint64x2x4_t cmp = CmpHelper<Op>::compare(v0l, v0r);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

//
#define INSTANTIATE_COMPARE_COLUMN_NEON(TTYPE,OP) \
    template bool OpCompareColumnImpl<TTYPE, TTYPE, CompareType::OP>::op_compare_column( \
        uint8_t* const __restrict bitmask, \
        const TTYPE* const __restrict left, \
        const TTYPE* const __restrict right, \
        const size_t size \
    );

ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_NEON, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_NEON, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_NEON, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_NEON, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_NEON, float)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_NEON, double)

#undef INSTANTIATE_COMPARE_COLUMN_NEON


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
    uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res_u8);

    // todo: aligned reads & writes

    const size_t size32 = (size / 32) * 32;
    for (size_t i = 0; i < size32; i += 32) {
        const int8x16x2_t v0l = {vld1q_s8(lower + i), vld1q_s8(lower + i + 16)};
        const int8x16x2_t v0u = {vld1q_s8(upper + i), vld1q_s8(upper + i + 16)};
        const int8x16x2_t v0v = {vld1q_s8(values + i), vld1q_s8(values + i + 16)};
        const uint8x16x2_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(v0l, v0v);
        const uint8x16x2_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, v0u);
        const uint8x16x2_t cmp = {vandq_u8(cmp0l.val[0], cmp0u.val[0]), vandq_u8(cmp0l.val[1], cmp0u.val[1])};
        const uint32_t mmask = movemask(cmp);

        res_u32[i / 32] = mmask;
    }

    for (size_t i = size32; i < size; i += 8) {
        const int8x8_t v0l = vld1_s8(lower + i);
        const int8x8_t v0u = vld1_s8(upper + i);
        const int8x8_t v0v = vld1_s8(values + i);
        const uint8x8_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(v0l, v0v);
        const uint8x8_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, v0u);
        const uint8x8_t cmp = vand_u8(cmp0l, cmp0u);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
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
    uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8);

    // todo: aligned reads & writes

    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const int16x8x2_t v0l = {vld1q_s16(lower + i), vld1q_s16(lower + i + 8)};
        const int16x8x2_t v0u = {vld1q_s16(upper + i), vld1q_s16(upper + i + 8)};
        const int16x8x2_t v0v = {vld1q_s16(values + i), vld1q_s16(values + i + 8)};
        const uint16x8x2_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(v0l, v0v);
        const uint16x8x2_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, v0u);
        const uint16x8x2_t cmp = {vandq_u16(cmp0l.val[0], cmp0u.val[0]), vandq_u16(cmp0l.val[1], cmp0u.val[1])};
        const uint16_t mmask = movemask(cmp);

        res_u16[i / 16] = mmask;
    }

    if (size16 != size) {
        // 8 elements to process
        const int16x8_t v0l = vld1q_s16(lower + size16);
        const int16x8_t v0u = vld1q_s16(upper + size16);
        const int16x8_t v0v = vld1q_s16(values + size16);
        const uint16x8_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(v0l, v0v);
        const uint16x8_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, v0u);
        const uint16x8_t cmp = vandq_u16(cmp0l, cmp0u);
        const uint8_t mmask = movemask(cmp);

        res_u8[size16 / 8] = mmask;
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

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const int32x4x2_t v0l = {vld1q_s32(lower + i), vld1q_s32(lower + i + 4)};
        const int32x4x2_t v0u = {vld1q_s32(upper + i), vld1q_s32(upper + i + 4)};
        const int32x4x2_t v0v = {vld1q_s32(values + i), vld1q_s32(values + i + 4)};
        const uint32x4x2_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(v0l, v0v);
        const uint32x4x2_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, v0u);
        const uint32x4x2_t cmp = {vandq_u32(cmp0l.val[0], cmp0u.val[0]), vandq_u32(cmp0l.val[1], cmp0u.val[1])};
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
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

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const int64x2x4_t v0l = {vld1q_s64(lower + i), vld1q_s64(lower + i + 2), vld1q_s64(lower + i + 4), vld1q_s64(lower + i + 6)};
        const int64x2x4_t v0u = {vld1q_s64(upper + i), vld1q_s64(upper + i + 2), vld1q_s64(upper + i + 4), vld1q_s64(upper + i + 6)};
        const int64x2x4_t v0v = {vld1q_s64(values + i), vld1q_s64(values + i + 2), vld1q_s64(values + i + 4), vld1q_s64(values + i + 6)};
        const uint64x2x4_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(v0l, v0v);
        const uint64x2x4_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, v0u);
        const uint64x2x4_t cmp = {
            vandq_u64(cmp0l.val[0], cmp0u.val[0]), vandq_u64(cmp0l.val[1], cmp0u.val[1]),
            vandq_u64(cmp0l.val[2], cmp0u.val[2]), vandq_u64(cmp0l.val[3], cmp0u.val[3])
        };
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
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

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const float32x4x2_t v0l = {vld1q_f32(lower + i), vld1q_f32(lower + i + 4)};
        const float32x4x2_t v0u = {vld1q_f32(upper + i), vld1q_f32(upper + i + 4)};
        const float32x4x2_t v0v = {vld1q_f32(values + i), vld1q_f32(values + i + 4)};
        const uint32x4x2_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(v0l, v0v);
        const uint32x4x2_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, v0u);
        const uint32x4x2_t cmp = {vandq_u32(cmp0l.val[0], cmp0u.val[0]), vandq_u32(cmp0l.val[1], cmp0u.val[1])};
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
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

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const float64x2x4_t v0l = {vld1q_f64(lower + i), vld1q_f64(lower + i + 2), vld1q_f64(lower + i + 4), vld1q_f64(lower + i + 6)};
        const float64x2x4_t v0u = {vld1q_f64(upper + i), vld1q_f64(upper + i + 2), vld1q_f64(upper + i + 4), vld1q_f64(upper + i + 6)};
        const float64x2x4_t v0v = {vld1q_f64(values + i), vld1q_f64(values + i + 2), vld1q_f64(values + i + 4), vld1q_f64(values + i + 6)};
        const uint64x2x4_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(v0l, v0v);
        const uint64x2x4_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, v0u);
        const uint64x2x4_t cmp = {
            vandq_u64(cmp0l.val[0], cmp0u.val[0]), vandq_u64(cmp0l.val[1], cmp0u.val[1]),
            vandq_u64(cmp0l.val[2], cmp0u.val[2]), vandq_u64(cmp0l.val[3], cmp0u.val[3])
        };
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

#define INSTANTIATE_WITHIN_RANGE_COLUMN_NEON(TTYPE,OP) \
    template bool OpWithinRangeColumnImpl<TTYPE, RangeType::OP>::op_within_range_column( \
        uint8_t* const __restrict res_u8, \
        const TTYPE* const __restrict lower, \
        const TTYPE* const __restrict upper, \
        const TTYPE* const __restrict values, \
        const size_t size \
    );

ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_NEON, int8_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_NEON, int16_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_NEON, int32_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_NEON, int64_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_NEON, float)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_NEON, double)

#undef INSTANTIATE_WITHIN_RANGE_COLUMN_NEON


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
    const int8x16x2_t lower_v = {vdupq_n_s8(lower), vdupq_n_s8(lower)};
    const int8x16x2_t upper_v = {vdupq_n_s8(upper), vdupq_n_s8(upper)};
    uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res_u8);

    // todo: aligned reads & writes

    const size_t size32 = (size / 32) * 32;
    for (size_t i = 0; i < size32; i += 32) {
        const int8x16x2_t v0v = {vld1q_s8(values + i), vld1q_s8(values + i + 16)};
        const uint8x16x2_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(lower_v, v0v);
        const uint8x16x2_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, upper_v);
        const uint8x16x2_t cmp = {vandq_u8(cmp0l.val[0], cmp0u.val[0]), vandq_u8(cmp0l.val[1], cmp0u.val[1])};
        const uint32_t mmask = movemask(cmp);

        res_u32[i / 32] = mmask;
    }

    for (size_t i = size32; i < size; i += 8) {
        const int8x8_t lower_v1 = vdup_n_s8(lower);
        const int8x8_t upper_v1 = vdup_n_s8(upper);
        const int8x8_t v0v = vld1_s8(values + i);
        const uint8x8_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(lower_v1, v0v);
        const uint8x8_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, upper_v1);
        const uint8x8_t cmp = vand_u8(cmp0l, cmp0u);
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
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
    const int16x8x2_t lower_v = {vdupq_n_s16(lower), vdupq_n_s16(lower)};
    const int16x8x2_t upper_v = {vdupq_n_s16(upper), vdupq_n_s16(upper)};
    uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res_u8);

    // todo: aligned reads & writes

    const size_t size16 = (size / 16) * 16;
    for (size_t i = 0; i < size16; i += 16) {
        const int16x8x2_t v0v = {vld1q_s16(values + i), vld1q_s16(values + i + 8)};
        const uint16x8x2_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(lower_v, v0v);
        const uint16x8x2_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, upper_v);
        const uint16x8x2_t cmp = {vandq_u16(cmp0l.val[0], cmp0u.val[0]), vandq_u16(cmp0l.val[1], cmp0u.val[1])};
        const uint16_t mmask = movemask(cmp);

        res_u16[i / 16] = mmask;
    }

    if (size16 != size) {
        // 8 elements to process
        const int16x8_t v0v = vld1q_s16(values + size16);
        const uint16x8_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(lower_v.val[0], v0v);
        const uint16x8_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, upper_v.val[0]);
        const uint16x8_t cmp = vandq_u16(cmp0l, cmp0u);
        const uint8_t mmask = movemask(cmp);

        res_u8[size16 / 8] = mmask;
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
    const int32x4x2_t lower_v = {vdupq_n_s32(lower), vdupq_n_s32(lower)};
    const int32x4x2_t upper_v = {vdupq_n_s32(upper), vdupq_n_s32(upper)};

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const int32x4x2_t v0v = {vld1q_s32(values + i), vld1q_s32(values + i + 4)};
        const uint32x4x2_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(lower_v, v0v);
        const uint32x4x2_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, upper_v);
        const uint32x4x2_t cmp = {vandq_u32(cmp0l.val[0], cmp0u.val[0]), vandq_u32(cmp0l.val[1], cmp0u.val[1])};
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
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
    const int64x2x4_t lower_v = {vdupq_n_s64(lower), vdupq_n_s64(lower), vdupq_n_s64(lower), vdupq_n_s64(lower)};
    const int64x2x4_t upper_v = {vdupq_n_s64(upper), vdupq_n_s64(upper), vdupq_n_s64(upper), vdupq_n_s64(upper)};

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const int64x2x4_t v0v = {vld1q_s64(values + i), vld1q_s64(values + i + 2), vld1q_s64(values + i + 4), vld1q_s64(values + i + 6)};
        const uint64x2x4_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(lower_v, v0v);
        const uint64x2x4_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, upper_v);
        const uint64x2x4_t cmp = {
            vandq_u64(cmp0l.val[0], cmp0u.val[0]), vandq_u64(cmp0l.val[1], cmp0u.val[1]),
            vandq_u64(cmp0l.val[2], cmp0u.val[2]), vandq_u64(cmp0l.val[3], cmp0u.val[3])
        };
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
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
    const float32x4x2_t lower_v = {vdupq_n_f32(lower), vdupq_n_f32(lower)};
    const float32x4x2_t upper_v = {vdupq_n_f32(upper), vdupq_n_f32(upper)};

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const float32x4x2_t v0v = {vld1q_f32(values + i), vld1q_f32(values + i + 4)};
        const uint32x4x2_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(lower_v, v0v);
        const uint32x4x2_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, upper_v);
        const uint32x4x2_t cmp = {vandq_u32(cmp0l.val[0], cmp0u.val[0]), vandq_u32(cmp0l.val[1], cmp0u.val[1])};
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
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
    const float64x2x4_t lower_v = {vdupq_n_f64(lower), vdupq_n_f64(lower), vdupq_n_f64(lower), vdupq_n_f64(lower)};
    const float64x2x4_t upper_v = {vdupq_n_f64(upper), vdupq_n_f64(upper), vdupq_n_f64(upper), vdupq_n_f64(upper)};

    // todo: aligned reads & writes

    const size_t size8 = (size / 8) * 8;
    for (size_t i = 0; i < size8; i += 8) {
        const float64x2x4_t v0v = {vld1q_f64(values + i), vld1q_f64(values + i + 2), vld1q_f64(values + i + 4), vld1q_f64(values + i + 6)};
        const uint64x2x4_t cmp0l = CmpHelper<Range2Compare<Op>::lower>::compare(lower_v, v0v);
        const uint64x2x4_t cmp0u = CmpHelper<Range2Compare<Op>::upper>::compare(v0v, upper_v);
        const uint64x2x4_t cmp = {
            vandq_u64(cmp0l.val[0], cmp0u.val[0]), vandq_u64(cmp0l.val[1], cmp0u.val[1]),
            vandq_u64(cmp0l.val[2], cmp0u.val[2]), vandq_u64(cmp0l.val[3], cmp0u.val[3])
        };
        const uint8_t mmask = movemask(cmp);

        res_u8[i / 8] = mmask;
    }

    return true;
}

//
#define INSTANTIATE_WITHIN_RANGE_VAL_NEON(TTYPE,OP) \
    template bool OpWithinRangeValImpl<TTYPE, RangeType::OP>::op_within_range_val( \
        uint8_t* const __restrict res_u8, \
        const TTYPE& lower, \
        const TTYPE& upper, \
        const TTYPE* const __restrict values, \
        const size_t size \
    );

ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_NEON, int8_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_NEON, int16_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_NEON, int32_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_NEON, int64_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_NEON, float)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_NEON, double)

#undef INSTANTIATE_WITHIN_RANGE_VAL_NEON

///////////////////////////////////////////////////////////////////////////

//
#undef ALL_COMPARE_OPS
#undef ALL_RANGE_OPS

}
}
}
}
}

