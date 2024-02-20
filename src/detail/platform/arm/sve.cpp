#include "sve.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <stdio.h>

#include <arm_sve.h>

namespace milvus {
namespace bitset {
namespace detail {
namespace arm {
namespace sve {

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

// a facility to run through all possible arithmetic compare operations
#define ALL_ARITH_CMP_OPS(FUNC,...) \
    FUNC(__VA_ARGS__,Add,EQ); \
    FUNC(__VA_ARGS__,Add,NEQ); \
    FUNC(__VA_ARGS__,Sub,EQ); \
    FUNC(__VA_ARGS__,Sub,NEQ); \
    FUNC(__VA_ARGS__,Mul,EQ); \
    FUNC(__VA_ARGS__,Mul,NEQ); \
    FUNC(__VA_ARGS__,Div,EQ); \
    FUNC(__VA_ARGS__,Div,NEQ); \
    FUNC(__VA_ARGS__,Mod,EQ); \
    FUNC(__VA_ARGS__,Mod,NEQ);

//
constexpr size_t MAX_SVE_WIDTH = 2048; 

//
inline void write_bitmask_full(
    uint8_t* const bitmask, 
    const svbool_t pred
) {
    *((svbool_t*)bitmask) = pred;

    uint8_t pred_mask[512 / 64];
    *((svbool_t*)pred_mask) = pred; 
    uint64_t pred_m = *(uint64_t*)(pred_mask);
    for (size_t i = 0; i < 64; i++) {
        printf("%d ", int((pred_m >> i) & 1));
    }
    printf("\n");

/*
    // SVE width in bytes
    const size_t sve_width = svcntb();
    assert((sve_width % 8) == 0);
    assert(sve_width * 8 <= MAX_SVE_WIDTH);

    // 2048 bits, 256 bytes => 256 bits bitmask, 32 bytes
    uint8_t pred_mask[MAX_SVE_WIDTH / 64];
    // *((svbool_t)pred_mask) = pred; 
    std::memcpy(pred_mask, &pred, sve_width / 8);

    //
    for (size_t i = 0; i < sve_width / 8; i++) {
        bitmask[i] = pred_mask[i];
    }
*/
}

inline void write_bitmask_full_16(
    uint8_t* const bitmask,
    const svbool_t pred
) {
    uint8_t pred_mask[512 / 64];
    *((svbool_t*)pred_mask) = pred; 

    uint64_t pred_m = *(uint64_t*)(pred_mask);

    constexpr uint64_t mask = 0x5555555555555555ULL;
    constexpr uint64_t mask0 = 0xccccccccccccccccULL;
    constexpr uint64_t mask1 = 0xf0f0f0f0f0f0f0f0ULL;
    constexpr uint64_t mask2 = 0xff00ff00ff00ff00ULL;
    constexpr uint64_t mask3 = 0xffff0000ffff0000ULL;
    constexpr uint64_t mask4 = 0xffffffff00000000ULL;

    pred_m &= mask;
    pred_m = (pred_m & ~mask0) | ((pred_m & mask0) >> 1); 
    pred_m = (pred_m & ~mask1) | ((pred_m & mask1) >> 2); 
    pred_m = (pred_m & ~mask2) | ((pred_m & mask2) >> 4); 
    pred_m = (pred_m & ~mask3) | ((pred_m & mask3) >> 8); 
    pred_m = (pred_m & ~mask4) | ((pred_m & mask4) >> 16); 

    *(uint32_t*)(bitmask) = uint32_t(pred_m);
}


inline void write_bitmask_full_32(
    uint8_t* const bitmask,
    const svbool_t pred
) {
    uint8_t pred_mask[512 / 64];
    *((svbool_t*)pred_mask) = pred; 

    uint64_t pred_m = *(uint64_t*)(pred_mask);

    constexpr uint64_t mask = 0x1111111111111111ULL;
    constexpr uint64_t mask0 = 0xb4b4b4b4b4b4b4b4ULL;
    constexpr uint64_t mask1 = 0xc738c738c738c738ULL;
    constexpr uint64_t mask2 = 0xf83f07c0f83f07c0ULL;
    constexpr uint64_t mask3 = 0xffc007ff003ff800ULL;
    constexpr uint64_t mask4 = 0x000007ffffc00000ULL;
    constexpr uint64_t mask5 = 0xfffff80000000000ULL;

    pred_m &= mask;
    pred_m = (pred_m & ~mask0) | ((pred_m & mask0) >> 1); 
    pred_m = (pred_m & ~mask1) | ((pred_m & mask1) >> 2); 
    pred_m = (pred_m & ~mask2) | ((pred_m & mask2) >> 4); 
    pred_m = (pred_m & ~mask3) | ((pred_m & mask3) >> 8); 
    pred_m = (pred_m & ~mask4) | ((pred_m & mask4) >> 16); 
    pred_m = (pred_m & ~mask5) | ((pred_m & mask5) >> 32); 

    *(uint16_t*)(bitmask) = uint16_t(pred_m);
}

inline void write_bitmask_full_64(
    uint8_t* const bitmask,
    const svbool_t pred
) {
    uint8_t pred_mask[512 / 64];
    *((svbool_t*)pred_mask) = pred; 

    uint64_t pred_m = *(uint64_t*)(pred_mask);

    constexpr uint64_t mask = 0x0101010101010101ULL;
    constexpr uint64_t mask0 = 0xab54ab54ab54ab54ULL;
    constexpr uint64_t mask1 = 0xcc673398cc673398ULL;
    constexpr uint64_t mask2 = 0xf0783c1f0f87c3e0ULL;
    constexpr uint64_t mask3 = 0x007fc01ff007fc00ULL;
    constexpr uint64_t mask4 = 0xff80001ffff80000ULL;
    constexpr uint64_t mask5 = 0xffffffe000000000ULL;

    pred_m &= mask;
    pred_m = (pred_m & ~mask0) | ((pred_m & mask0) >> 1); 
    pred_m = (pred_m & ~mask1) | ((pred_m & mask1) >> 2); 
    pred_m = (pred_m & ~mask2) | ((pred_m & mask2) >> 4); 
    pred_m = (pred_m & ~mask3) | ((pred_m & mask3) >> 8); 
    pred_m = (pred_m & ~mask4) | ((pred_m & mask4) >> 16); 
    pred_m = (pred_m & ~mask5) | ((pred_m & mask5) >> 32); 

    *bitmask = uint8_t(pred_m);
}

//
inline void write_bitmask_partial(
    uint8_t* const bitmask, 
    const svbool_t pred,
    const svbool_t valid
) {
    // SVE width in bytes
    const size_t sve_width = svcntb();
    assert((sve_width % 8) == 0);
    assert(sve_width * 8 <= MAX_SVE_WIDTH);

    // 2048 bits, 256 bytes => 256 bits bitmask, 32 bytes
    uint8_t pred_mask[MAX_SVE_WIDTH / 64];
    *((svbool_t*)pred_mask) = pred; 
    //std::memcpy(pred_mask, &pred, sve_width / 8);

    uint8_t valid_mask[MAX_SVE_WIDTH / 64];
    *((svbool_t*)valid_mask) = pred; 
    //std::memcpy(valid_mask, &valid, sve_width / 8);

    for (size_t i = 0; i < sve_width / 8; i++) {
        if (valid_mask[i] == 0) {
            break;
        }
            
        bitmask[i] = pred_mask[i];
    }
}

///////////////////////////////////////////////////////////////////////////

//
template<CompareOpType Op>
struct CmpHelper{};

template<>
struct CmpHelper<CompareOpType::EQ> {
    static inline svbool_t compare(const svbool_t pred, const svint8_t a, const svint8_t b) {
        return svcmpeq_s8(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint16_t a, const svint16_t b) {
        return svcmpeq_s16(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint32_t a, const svint32_t b) {
        return svcmpeq_s32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint64_t a, const svint64_t b) {
        return svcmpeq_s64(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat32_t a, const svfloat32_t b) {
        return svcmpeq_f32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat64_t a, const svfloat64_t b) {
        return svcmpeq_f64(pred, a, b);
    }
};

template<>
struct CmpHelper<CompareOpType::GE> {
    static inline svbool_t compare(const svbool_t pred, const svint8_t a, const svint8_t b) {
        return svcmpge_s8(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint16_t a, const svint16_t b) {
        return svcmpge_s16(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint32_t a, const svint32_t b) {
        return svcmpge_s32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint64_t a, const svint64_t b) {
        return svcmpge_s64(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat32_t a, const svfloat32_t b) {
        return svcmpge_f32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat64_t a, const svfloat64_t b) {
        return svcmpge_f64(pred, a, b);
    }
};

template<>
struct CmpHelper<CompareOpType::GT> {
    static inline svbool_t compare(const svbool_t pred, const svint8_t a, const svint8_t b) {
        return svcmpgt_s8(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint16_t a, const svint16_t b) {
        return svcmpgt_s16(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint32_t a, const svint32_t b) {
        return svcmpgt_s32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint64_t a, const svint64_t b) {
        return svcmpgt_s64(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat32_t a, const svfloat32_t b) {
        return svcmpgt_f32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat64_t a, const svfloat64_t b) {
        return svcmpgt_f64(pred, a, b);
    }
};

template<>
struct CmpHelper<CompareOpType::LE> {
    static inline svbool_t compare(const svbool_t pred, const svint8_t a, const svint8_t b) {
        return svcmple_s8(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint16_t a, const svint16_t b) {
        return svcmple_s16(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint32_t a, const svint32_t b) {
        return svcmple_s32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint64_t a, const svint64_t b) {
        return svcmple_s64(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat32_t a, const svfloat32_t b) {
        return svcmple_f32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat64_t a, const svfloat64_t b) {
        return svcmple_f64(pred, a, b);
    }
};

template<>
struct CmpHelper<CompareOpType::LT> {
    static inline svbool_t compare(const svbool_t pred, const svint8_t a, const svint8_t b) {
        return svcmplt_s8(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint16_t a, const svint16_t b) {
        return svcmplt_s16(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint32_t a, const svint32_t b) {
        return svcmplt_s32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint64_t a, const svint64_t b) {
        return svcmplt_s64(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat32_t a, const svfloat32_t b) {
        return svcmplt_f32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat64_t a, const svfloat64_t b) {
        return svcmplt_f64(pred, a, b);
    }
};

template<>
struct CmpHelper<CompareOpType::NEQ> {
    static inline svbool_t compare(const svbool_t pred, const svint8_t a, const svint8_t b) {
        return svcmpne_s8(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint16_t a, const svint16_t b) {
        return svcmpne_s16(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint32_t a, const svint32_t b) {
        return svcmpne_s32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svint64_t a, const svint64_t b) {
        return svcmpne_s64(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat32_t a, const svfloat32_t b) {
        return svcmpne_f32(pred, a, b);
    }

    static inline svbool_t compare(const svbool_t pred, const svfloat64_t a, const svfloat64_t b) {
        return svcmpne_f64(pred, a, b);
    }
};

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

    // SVE width in bytes
    const size_t sve_width = svcntb();
    assert((sve_width % 8) == 0);

    const svbool_t pred_all = svptrue_b8();

    //
    const svint8_t target = svdup_n_s8(val);

    // process large blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const svint8_t v = svld1_s8(pred_all, src + i);
        const svbool_t cmp = CmpHelper<Op>::compare(pred_all, v, target);
        
        write_bitmask_full(res_u8 + i / 8, cmp);
    }

    // process leftovers
    if (size_sve != size) {
        const svint8_t v = svldff1_s8(pred_all, src + size_sve);
        const svbool_t valid_mask = svrdffr();
        const svbool_t cmp = CmpHelper<Op>::compare(valid_mask, v, target);

        write_bitmask_partial(res_u8 + size_sve / 8, cmp, valid_mask);
    }

    return true;
}

// https://stackoverflow.com/questions/77834169/what-is-a-fast-fallback-algorithm-which-emulates-pdep-and-pext-in-software
// SVE2 has bitperm
// https://github.com/zwegner/zp7/blob/master/zp7.c
template<CompareOpType Op>
bool OpCompareValImpl<int16_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict src, 
    const size_t size, 
    const int16_t& val
) {
    // the restriction of the API
    assert((size % 8) == 0);

    // SVE width in words
    const size_t sve_width = svcnth();
    assert((sve_width % 8) == 0);

    const svbool_t pred_all = svptrue_b16();

    //
    const svint16_t target = svdup_n_s16(val);

    // process large blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const svint16_t v = svld1_s16(pred_all, src + i);
        const svbool_t cmp = CmpHelper<Op>::compare(pred_all, v, target);
        
        write_bitmask_full_16(res_u8 + i / 8, cmp);
    }

/*
    // process leftovers
    if (size_sve != size) {
        const svint8_t v = svldff1_s8(pred_all, src + size_sve);
        const svbool_t valid_mask = svrdffr();
        const svbool_t cmp = CmpHelper<Op>::compare(valid_mask, v, target);

        write_bitmask_partial(res_u8 + size_sve / 8, cmp, valid_mask);
    }
*/

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

    // SVE width in words
    const size_t sve_width = svcntw();
    assert((sve_width % 8) == 0);

    const svbool_t pred_all = svptrue_b32();

    //
    const svint32_t target = svdup_n_s32(val);

    // process large blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const svint32_t v = svld1_s32(pred_all, src + i);
        const svbool_t cmp = CmpHelper<Op>::compare(pred_all, v, target);
        
        write_bitmask_full_32(res_u8 + i / 8, cmp);
    }

/*
    // process leftovers
    if (size_sve != size) {
        const svint8_t v = svldff1_s8(pred_all, src + size_sve);
        const svbool_t valid_mask = svrdffr();
        const svbool_t cmp = CmpHelper<Op>::compare(valid_mask, v, target);

        write_bitmask_partial(res_u8 + size_sve / 8, cmp, valid_mask);
    }
*/

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

    // SVE width in words
    const size_t sve_width = svcntd();
    assert((sve_width % 8) == 0);

    const svbool_t pred_all = svptrue_b64();

    //
    const svint64_t target = svdup_n_s64(val);

    // process large blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const svint64_t v = svld1_s64(pred_all, src + i);
        const svbool_t cmp = CmpHelper<Op>::compare(pred_all, v, target);
        
        write_bitmask_full_64(res_u8 + i / 8, cmp);
    }

/*
    // process leftovers
    if (size_sve != size) {
        const svint8_t v = svldff1_s8(pred_all, src + size_sve);
        const svbool_t valid_mask = svrdffr();
        const svbool_t cmp = CmpHelper<Op>::compare(valid_mask, v, target);

        write_bitmask_partial(res_u8 + size_sve / 8, cmp, valid_mask);
    }
*/

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

    // SVE width in words
    const size_t sve_width = svcntw();
    assert((sve_width % 8) == 0);

    const svbool_t pred_all = svptrue_b32();

    //
    const svfloat32_t target = svdup_n_f32(val);

    // process large blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const svfloat32_t v = svld1_f32(pred_all, src + i);
        const svbool_t cmp = CmpHelper<Op>::compare(pred_all, v, target);
        
        write_bitmask_full_32(res_u8 + i / 8, cmp);
    }

/*
    // process leftovers
    if (size_sve != size) {
        const svint8_t v = svldff1_s8(pred_all, src + size_sve);
        const svbool_t valid_mask = svrdffr();
        const svbool_t cmp = CmpHelper<Op>::compare(valid_mask, v, target);

        write_bitmask_partial(res_u8 + size_sve / 8, cmp, valid_mask);
    }
*/

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

    // SVE width in words
    const size_t sve_width = svcntd();
    assert((sve_width % 8) == 0);

    const svbool_t pred_all = svptrue_b64();

    //
    const svfloat64_t target = svdup_n_f64(val);

    // process large blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const svfloat64_t v = svld1_f64(pred_all, src + i);
        const svbool_t cmp = CmpHelper<Op>::compare(pred_all, v, target);
        
        write_bitmask_full_64(res_u8 + i / 8, cmp);
    }

/*
    // process leftovers
    if (size_sve != size) {
        const svint8_t v = svldff1_s8(pred_all, src + size_sve);
        const svbool_t valid_mask = svrdffr();
        const svbool_t cmp = CmpHelper<Op>::compare(valid_mask, v, target);

        write_bitmask_partial(res_u8 + size_sve / 8, cmp, valid_mask);
    }
*/

    return true;
}

//
#define INSTANTIATE_COMPARE_VAL_SVE(TTYPE,OP) \
    template bool OpCompareValImpl<TTYPE, CompareOpType::OP>::op_compare_val( \
        uint8_t* const __restrict bitmask, \
        const TTYPE* const __restrict src, \
        const size_t size, \
        const TTYPE& val \
    );

ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_SVE, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_SVE, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_SVE, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_SVE, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_SVE, float)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_SVE, double)

#undef INSTANTIATE_COMPARE_VAL_SVE


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

    return true;
}

//
#define INSTANTIATE_COMPARE_COLUMN_SVE(TTYPE,OP) \
    template bool OpCompareColumnImpl<TTYPE, TTYPE, CompareOpType::OP>::op_compare_column( \
        uint8_t* const __restrict bitmask, \
        const TTYPE* const __restrict left, \
        const TTYPE* const __restrict right, \
        const size_t size \
    );

ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_SVE, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_SVE, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_SVE, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_SVE, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_SVE, float)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_SVE, double)

#undef INSTANTIATE_COMPARE_COLUMN_SVE


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

    return true;
}

#define INSTANTIATE_WITHIN_RANGE_COLUMN_SVE(TTYPE,OP) \
    template bool OpWithinRangeColumnImpl<TTYPE, RangeType::OP>::op_within_range_column( \
        uint8_t* const __restrict res_u8, \
        const TTYPE* const __restrict lower, \
        const TTYPE* const __restrict upper, \
        const TTYPE* const __restrict values, \
        const size_t size \
    );

ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_SVE, int8_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_SVE, int16_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_SVE, int32_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_SVE, int64_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_SVE, float)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_COLUMN_SVE, double)

#undef INSTANTIATE_WITHIN_RANGE_COLUMN_SVE


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

    return true;
}

//
#define INSTANTIATE_WITHIN_RANGE_VAL_SVE(TTYPE,OP) \
    template bool OpWithinRangeValImpl<TTYPE, RangeType::OP>::op_within_range_val( \
        uint8_t* const __restrict res_u8, \
        const TTYPE& lower, \
        const TTYPE& upper, \
        const TTYPE* const __restrict values, \
        const size_t size \
    );

ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_SVE, int8_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_SVE, int16_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_SVE, int32_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_SVE, int64_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_SVE, float)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_VAL_SVE, double)

#undef INSTANTIATE_WITHIN_RANGE_VAL_SVE

///////////////////////////////////////////////////////////////////////////

// https://godbolt.org/z/CYipz7
// https://github.com/ridiculousfish/libdivide
// https://github.com/lemire/fastmod


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
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Mul, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Mul, NEQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Div, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Div, NEQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int8_t, Mod, NEQ)

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

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Mul, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Mul, NEQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Div, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Div, NEQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int16_t, Mod, NEQ)

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

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Mul, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Mul, NEQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Div, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Div, NEQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int32_t, Mod, NEQ)

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

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Mul, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Mul, NEQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Div, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Div, NEQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(int64_t, Mod, NEQ)

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

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(float, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(float, Mod, NEQ)

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

    return true;
}

//
NOT_IMPLEMENTED_OP_ARITH_COMPARE(double, Mod, EQ)
NOT_IMPLEMENTED_OP_ARITH_COMPARE(double, Mod, NEQ)

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

    return true;
}

//
#undef NOT_IMPLEMENTED_OP_ARITH_COMPARE

//
#define INSTANTIATE_ARITH_COMPARE_SVE(TTYPE,OP,CMP) \
    template bool OpArithCompareImpl<TTYPE, ArithOpType::OP, CompareOpType::CMP>::op_arith_compare( \
        uint8_t* const __restrict res_u8, \
        const TTYPE* const __restrict src, \
        const ArithHighPrecisionType<TTYPE>& right_operand, \
        const ArithHighPrecisionType<TTYPE>& value, \
        const size_t size \
    );

ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_SVE, int8_t)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_SVE, int16_t)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_SVE, int32_t)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_SVE, int64_t)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_SVE, float)
ALL_ARITH_CMP_OPS(INSTANTIATE_ARITH_COMPARE_SVE, double)

#undef INSTANTIATE_ARITH_COMPARE_SVE

///////////////////////////////////////////////////////////////////////////

//
#undef ALL_COMPARE_OPS
#undef ALL_RANGE_OPS
#undef ALL_ARITH_CMP_OPS

}
}
}
}
}
