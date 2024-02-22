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

constexpr uint8_t SVE_LANES_8[MAX_SVE_WIDTH / 8] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,

    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
    0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
    0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F,

    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F,
    0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF,
    0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
    0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF,

    0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
    0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF,
    0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
    0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF,
    0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,
    0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF,
    0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
    0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF
};

constexpr uint16_t SVE_LANES_16[MAX_SVE_WIDTH / 16] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,

    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
    0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
    0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F
};

constexpr uint32_t SVE_LANES_32[MAX_SVE_WIDTH / 32] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F
};

constexpr uint64_t SVE_LANES_64[MAX_SVE_WIDTH / 64] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F
};

/*
// debugging facilities

//
void print_svbool_t(const svbool_t value) {
    // 2048 bits, 256 bytes => 256 bits bitmask, 32 bytes
    uint8_t v[MAX_SVE_WIDTH / 64];
    *((svbool_t*)v) = value;

    const size_t sve_width = svcntb();
    for (size_t i = 0; i < sve_width / 8; i++) {
        printf("%d ", int(v[i]));
    }
    printf("\n");
}

//
void print_svuint8_t(const svuint8_t value) {
    uint8_t v[MAX_SVE_WIDTH / 8];
    *((svuint8_t*)v) = value;

    const size_t sve_width = svcntb();
    for (size_t i = 0; i < sve_width; i++) {
        printf("%d ", int(v[i]));
    }
    printf("\n");
}

*/

///////////////////////////////////////////////////////////////////////////

// an interesting duscission here:
// https://stackoverflow.com/questions/77834169/what-is-a-fast-fallback-algorithm-which-emulates-pdep-and-pext-in-software

// based on code from https://github.com/zwegner/zp7/blob/master/zp7.c

// SVE2 has bitperm, which contains the implementation of pext

//
inline uint32_t pext_u64_16b(const uint64_t pred_mask) {
    // pred_m = pext(pred_m, 0x55555555555555ULL);

    constexpr uint64_t mask = 0x5555555555555555ULL;
    constexpr uint64_t mask0 = 0xccccccccccccccccULL;
    constexpr uint64_t mask1 = 0xf0f0f0f0f0f0f0f0ULL;
    constexpr uint64_t mask2 = 0xff00ff00ff00ff00ULL;
    constexpr uint64_t mask3 = 0xffff0000ffff0000ULL;
    constexpr uint64_t mask4 = 0xffffffff00000000ULL;

    uint64_t pred_m = pred_mask & mask;
    pred_m = (pred_m & ~mask0) | ((pred_m & mask0) >> 1); 
    pred_m = (pred_m & ~mask1) | ((pred_m & mask1) >> 2); 
    pred_m = (pred_m & ~mask2) | ((pred_m & mask2) >> 4); 
    pred_m = (pred_m & ~mask3) | ((pred_m & mask3) >> 8); 
    pred_m = (pred_m & ~mask4) | ((pred_m & mask4) >> 16); 

    return uint32_t(pred_m);
}

//
inline uint16_t pext_u64_32b(const uint64_t pred_mask) {
    // pred_m = pext(pred_m, 0x1111111111111111ULL);

    constexpr uint64_t mask = 0x1111111111111111ULL;
    constexpr uint64_t mask0 = 0xb4b4b4b4b4b4b4b4ULL;
    constexpr uint64_t mask1 = 0xc738c738c738c738ULL;
    constexpr uint64_t mask2 = 0xf83f07c0f83f07c0ULL;
    constexpr uint64_t mask3 = 0xffc007ff003ff800ULL;
    constexpr uint64_t mask4 = 0x000007ffffc00000ULL;
    constexpr uint64_t mask5 = 0xfffff80000000000ULL;

    uint64_t pred_m = pred_mask & mask;
    pred_m = (pred_m & ~mask0) | ((pred_m & mask0) >> 1); 
    pred_m = (pred_m & ~mask1) | ((pred_m & mask1) >> 2); 
    pred_m = (pred_m & ~mask2) | ((pred_m & mask2) >> 4); 
    pred_m = (pred_m & ~mask3) | ((pred_m & mask3) >> 8); 
    pred_m = (pred_m & ~mask4) | ((pred_m & mask4) >> 16); 
    pred_m = (pred_m & ~mask5) | ((pred_m & mask5) >> 32); 

    return uint16_t(pred_m);
}

//
inline uint8_t pext_u64_64b(const uint64_t pred_mask) {
    // pred_m = pext(pred_m, 0x0101010101010101ULL);

    constexpr uint64_t mask = 0x0101010101010101ULL;
    constexpr uint64_t mask0 = 0xab54ab54ab54ab54ULL;
    constexpr uint64_t mask1 = 0xcc673398cc673398ULL;
    constexpr uint64_t mask2 = 0xf0783c1f0f87c3e0ULL;
    constexpr uint64_t mask3 = 0x007fc01ff007fc00ULL;
    constexpr uint64_t mask4 = 0xff80001ffff80000ULL;
    constexpr uint64_t mask5 = 0xffffffe000000000ULL;

    uint64_t pred_m = pred_mask & mask;
    pred_m = (pred_m & ~mask0) | ((pred_m & mask0) >> 1); 
    pred_m = (pred_m & ~mask1) | ((pred_m & mask1) >> 2); 
    pred_m = (pred_m & ~mask2) | ((pred_m & mask2) >> 4); 
    pred_m = (pred_m & ~mask3) | ((pred_m & mask3) >> 8); 
    pred_m = (pred_m & ~mask4) | ((pred_m & mask4) >> 16); 
    pred_m = (pred_m & ~mask5) | ((pred_m & mask5) >> 32); 

    return uint8_t(pred_m);
}

// 8 bit elements
inline void write_bitmask_full_8(
    uint8_t* const bitmask, 
    const svbool_t pred
) {
    *((svbool_t*)bitmask) = pred;
}

// 8 bit elements
inline void write_bitmask_partial_8(
    uint8_t* const bitmask, 
    const svbool_t pred,
    const svbool_t valid
) {
    // 2048 bits, 256 bytes => 256 bits bitmask, 32 bytes
    uint8_t pred_mask[MAX_SVE_WIDTH / 64];
    *((svbool_t*)pred_mask) = pred;

    const svuint8_t bits = svld1_u8(valid, pred_mask);
    svst1_u8(valid, bitmask, bits);
}

// 512 bit width, 16 bit elements
inline void write_bitmask_full_512_16(
    uint8_t* const bitmask,
    const svbool_t pred
) {
    uint8_t pred_mask[MAX_SVE_WIDTH / 64];
    *((svbool_t*)pred_mask) = pred;

    const uint64_t pred_m = *(const uint64_t*)(pred_mask);
    const uint32_t compressed_pred_m = pext_u64_16b(pred_m);

    *(uint32_t*)(bitmask) = compressed_pred_m;
}

// 512 bit width, 16 bit elements
// todo: remake
inline void write_bitmask_partial_512_16(
    uint8_t* const bitmask, 
    const svbool_t pred,
    const svbool_t valid
) {
    // 2048 bits, 256 bytes => 256 bits bitmask, 32 bytes
    uint8_t pred_mask[MAX_SVE_WIDTH / 64];
    *((svbool_t*)pred_mask) = pred;

    const uint64_t pred_m = *(const uint64_t*)(pred_mask);
    const uint32_t compressed_pred_m = pext_u64_16b(pred_m);

    const svuint32_t bits = svdup_n_u32(compressed_pred_m);
    svst1_u8(valid, bitmask, svreinterpret_u8_u32(bits));
}

// 512 bit width, 32 bit elements
inline void write_bitmask_full_512_32(
    uint8_t* const bitmask,
    const svbool_t pred
) {
    uint8_t pred_mask[MAX_SVE_WIDTH / 64];
    *((svbool_t*)pred_mask) = pred; 

    const uint64_t pred_m = *(const uint64_t*)(pred_mask);
    const uint16_t compressed_pred_m = pext_u64_32b(pred_m);

    *(uint16_t*)(bitmask) = compressed_pred_m;
}

// 512 bit width, 32 bit elements
// todo: remake
inline void write_bitmask_partial_512_32(
    uint8_t* const bitmask, 
    const svbool_t pred,
    const svbool_t valid
) {
    // 2048 bits, 256 bytes => 256 bits bitmask, 32 bytes
    uint8_t pred_mask[MAX_SVE_WIDTH / 64];
    *((svbool_t*)pred_mask) = pred;

    const uint64_t pred_m = *(const uint64_t*)(pred_mask);
    const uint16_t compressed_pred_m = pext_u64_32b(pred_m);

    const svuint16_t bits = svdup_n_u16(compressed_pred_m);
    svst1_u8(valid, bitmask, svreinterpret_u8_u16(bits));
}

// 512 bit width, 64 bit elements
inline void write_bitmask_full_512_64(
    uint8_t* const bitmask,
    const svbool_t pred
) {
    uint8_t pred_mask[MAX_SVE_WIDTH / 64];
    *((svbool_t*)pred_mask) = pred; 

    const uint64_t pred_m = *(uint64_t*)(pred_mask);
    const uint8_t compressed_pred_m = pext_u64_64b(pred_m);

    *bitmask = compressed_pred_m;
}

// 512 bit width, 64 bit elements
// todo: remake
inline void write_bitmask_partial_512_64(
    uint8_t* const bitmask, 
    const svbool_t pred,
    const svbool_t valid
) {
    // 2048 bits, 256 bytes => 256 bits bitmask, 32 bytes
    uint8_t pred_mask[MAX_SVE_WIDTH / 64];
    *((svbool_t*)pred_mask) = pred;

    const uint64_t pred_m = *(uint64_t*)(pred_mask);
    const uint8_t compressed_pred_m = pext_u64_64b(pred_m);

    const svuint8_t bits = svdup_n_u8(compressed_pred_m);
    svst1_u8(valid, bitmask, bits);
}

//
template<typename T, size_t width>
struct MaskWriter {};

template<size_t width>
struct MaskWriter<int8_t, width> {
    inline static void write_full(uint8_t* const bitmask, const svbool_t pred) {
        write_bitmask_full_8(bitmask, pred);
    }

    inline static void write_partial(uint8_t* const bitmask, const svbool_t pred, const svbool_t valid) {
        write_bitmask_partial_8(bitmask, pred, valid);
    }
};

template<>
struct MaskWriter<int16_t, 512> {
    inline static void write_full(uint8_t* const bitmask, const svbool_t pred) {
        write_bitmask_full_512_16(bitmask, pred);
    }

    inline static void write_partial(uint8_t* const bitmask, const svbool_t pred, const svbool_t valid) {
        write_bitmask_partial_512_16(bitmask, pred, valid);
    }
};

template<>
struct MaskWriter<int32_t, 512> {
    inline static void write_full(uint8_t* const bitmask, const svbool_t pred) {
        write_bitmask_full_512_32(bitmask, pred);
    }

    inline static void write_partial(uint8_t* const bitmask, const svbool_t pred, const svbool_t valid) {
        write_bitmask_partial_512_32(bitmask, pred, valid);
    }
};

template<>
struct MaskWriter<int64_t, 512> {
    inline static void write_full(uint8_t* const bitmask, const svbool_t pred) {
        write_bitmask_full_512_64(bitmask, pred);
    }

    inline static void write_partial(uint8_t* const bitmask, const svbool_t pred, const svbool_t valid) {
        write_bitmask_partial_512_64(bitmask, pred, valid);
    }
};

template<>
struct MaskWriter<float, 512> {
    inline static void write_full(uint8_t* const bitmask, const svbool_t pred) {
        write_bitmask_full_512_32(bitmask, pred);
    }

    inline static void write_partial(uint8_t* const bitmask, const svbool_t pred, const svbool_t valid) {
        write_bitmask_partial_512_32(bitmask, pred, valid);
    }
};

template<>
struct MaskWriter<double, 512> {
    inline static void write_full(uint8_t* const bitmask, const svbool_t pred) {
        write_bitmask_full_512_64(bitmask, pred);
    }

    inline static void write_partial(uint8_t* const bitmask, const svbool_t pred, const svbool_t valid) {
        write_bitmask_partial_512_64(bitmask, pred, valid);
    }
};

//
inline svbool_t get_pred_write(const size_t n_elements) {
    assert((n_elements % 8) == 0);

    const svbool_t pred_all_8 = svptrue_b8();
    const svuint8_t lanes_8 = svld1_u8(pred_all_8, SVE_LANES_8);
    const svuint8_t leftovers_w = svdup_n_u8(n_elements / 8);
    const svbool_t pred_write = svcmpgt_u8(pred_all_8, leftovers_w, lanes_8);
    return pred_write;
}

//
inline svbool_t get_pred_op_8(const size_t n_elements) {
    const svbool_t pred_all_8 = svptrue_b8();
    const svuint8_t lanes_8 = svld1_u8(pred_all_8, SVE_LANES_8);
    const svuint8_t leftovers_op = svdup_n_u8(n_elements);
    const svbool_t pred_op = svcmpgt_u8(pred_all_8, leftovers_op, lanes_8);
    return pred_op;
}

//
inline svbool_t get_pred_op_16(const size_t n_elements) {
    const svbool_t pred_all_16 = svptrue_b16();
    const svuint16_t lanes_16 = svld1_u16(pred_all_16, SVE_LANES_16);
    const svuint16_t leftovers_op = svdup_n_u16(n_elements);
    const svbool_t pred_op = svcmpgt_u16(pred_all_16, leftovers_op, lanes_16);
    return pred_op;
}

//
inline svbool_t get_pred_op_32(const size_t n_elements) {
    const svbool_t pred_all_32 = svptrue_b32();
    const svuint32_t lanes_32 = svld1_u32(pred_all_32, SVE_LANES_32);
    const svuint32_t leftovers_op = svdup_n_u32(n_elements);
    const svbool_t pred_op = svcmpgt_u32(pred_all_32, leftovers_op, lanes_32);
    return pred_op;
}

//
inline svbool_t get_pred_op_64(const size_t n_elements) {
    const svbool_t pred_all_64 = svptrue_b64();
    const svuint64_t lanes_64 = svld1_u64(pred_all_64, SVE_LANES_64);
    const svuint64_t leftovers_op = svdup_n_u64(n_elements);
    const svbool_t pred_op = svcmpgt_u64(pred_all_64, leftovers_op, lanes_64);
    return pred_op;
}

//
template<typename T>
struct GetPredHelper {};

template<>
struct GetPredHelper<int8_t> {
    inline static svbool_t get_pred_op(const size_t n_elements) {
        return get_pred_op_8(n_elements);
    }
};

template<>
struct GetPredHelper<int16_t> {
    inline static svbool_t get_pred_op(const size_t n_elements) {
        return get_pred_op_16(n_elements);
    }
};

template<>
struct GetPredHelper<int32_t> {
    inline static svbool_t get_pred_op(const size_t n_elements) {
        return get_pred_op_32(n_elements);
    }
};

template<>
struct GetPredHelper<int64_t> {
    inline static svbool_t get_pred_op(const size_t n_elements) {
        return get_pred_op_64(n_elements);
    }
};

template<>
struct GetPredHelper<float> {
    inline static svbool_t get_pred_op(const size_t n_elements) {
        return get_pred_op_32(n_elements);
    }
};

template<>
struct GetPredHelper<double> {
    inline static svbool_t get_pred_op(const size_t n_elements) {
        return get_pred_op_64(n_elements);
    }
};

template<typename T>
inline svbool_t get_pred_op(const size_t n_elements) {
    return GetPredHelper<T>::get_pred_op(n_elements);
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

template<typename T>
struct SVEVector {};

template<>
struct SVEVector<int8_t> {
    using data_type = int8_t;
    using sve_type = svint8_t;

    // measured in the number of elements that an SVE register can hold
    static inline size_t width() {
        return svcntb();
    }

    static inline svbool_t pred_all() {
        return svptrue_b8();
    }

    inline static sve_type set1(const data_type value) {
        return svdup_n_s8(value);
    }

    inline static sve_type load(const svbool_t pred, const data_type* value) {
        return svld1_s8(pred, value);
    }
};

template<>
struct SVEVector<int16_t> {
    using data_type = int16_t;
    using sve_type = svint16_t;

    // measured in the number of elements that an SVE register can hold
    static inline size_t width() {
        return svcnth();
    }

    static inline svbool_t pred_all() {
        return svptrue_b16();
    }

    inline static sve_type set1(const data_type value) {
        return svdup_n_s16(value);
    }

    inline static sve_type load(const svbool_t pred, const data_type* value) {
        return svld1_s16(pred, value);
    }
};

template<>
struct SVEVector<int32_t> {
    using data_type = int32_t;
    using sve_type = svint32_t;

    // measured in the number of elements that an SVE register can hold
    static inline size_t width() {
        return svcntw();
    }

    static inline svbool_t pred_all() {
        return svptrue_b32();
    }

    inline static sve_type set1(const data_type value) {
        return svdup_n_s32(value);
    }

    inline static sve_type load(const svbool_t pred, const data_type* value) {
        return svld1_s32(pred, value);
    }
};

template<>
struct SVEVector<int64_t> {
    using data_type = int64_t;
    using sve_type = svint64_t;

    // measured in the number of elements that an SVE register can hold
    static inline size_t width() {
        return svcntd();
    }

    static inline svbool_t pred_all() {
        return svptrue_b64();
    }

    inline static sve_type set1(const data_type value) {
        return svdup_n_s64(value);
    }

    inline static sve_type load(const svbool_t pred, const data_type* value) {
        return svld1_s64(pred, value);
    }
};

template<>
struct SVEVector<float> {
    using data_type = float;
    using sve_type = svfloat32_t;

    // measured in the number of elements that an SVE register can hold
    static inline size_t width() {
        return svcntw();
    }

    static inline svbool_t pred_all() {
        return svptrue_b32();
    }

    inline static sve_type set1(const data_type value) {
        return svdup_n_f32(value);
    }

    inline static sve_type load(const svbool_t pred, const data_type* value) {
        return svld1_f32(pred, value);
    }
};

template<>
struct SVEVector<double> {
    using data_type = double;
    using sve_type = svfloat64_t;

    // measured in the number of elements that an SVE register can hold
    static inline size_t width() {
        return svcntd();
    }

    static inline svbool_t pred_all() {
        return svptrue_b64();
    }

    inline static sve_type set1(const data_type value) {
        return svdup_n_f64(value);
    }

    inline static sve_type load(const svbool_t pred, const data_type* value) {
        return svld1_f64(pred, value);
    }
};

///////////////////////////////////////////////////////////////////////////
template<typename T, CompareOpType CmpOp>
bool op_compare_val_impl(
    uint8_t* const __restrict res_u8,
    const T* const __restrict src, 
    const size_t size, 
    const T& val
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    using sve_t = SVEVector<T>;

    // SVE width in bytes
    const size_t sve_width = sve_t::width();
    assert((sve_width % 8) == 0);

    // Only 512 bits are implemented for now. This is because 
    //   512 bits hold 8 64-bit values. If the width is lower, then
    //   a different code is needed, just like for AVX2.
    if (sve_width * 8 * sizeof(T) != 512) {
        return false;
    }

    //
    const svbool_t pred_all = sve_t::pred_all();
    const auto target = sve_t::set1(val);

    // process big blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const auto v = sve_t::load(pred_all, src + i);
        const svbool_t cmp = CmpHelper<CmpOp>::compare(pred_all, v, target);
        
        MaskWriter<T, 512>::write_full(res_u8 + i / 8, cmp);
    }

    // process leftovers
    if (size_sve != size) {
        const svbool_t pred_op = get_pred_op<T>(size - size_sve);
        const svbool_t pred_write = get_pred_write(size - size_sve);

        const auto v = sve_t::load(pred_op, src + size_sve);
        const svbool_t cmp = CmpHelper<CmpOp>::compare(pred_op, v, target);

        MaskWriter<T, 512>::write_partial(res_u8 + size_sve / 8, cmp, pred_write);
    }

    return true;
}

//
template<CompareOpType Op>
bool OpCompareValImpl<int8_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int8_t* const __restrict src, 
    const size_t size, 
    const int8_t& val
) {
    return op_compare_val_impl<int8_t, Op>(res_u8, src, size, val);
}

template<CompareOpType Op>
bool OpCompareValImpl<int16_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict src, 
    const size_t size, 
    const int16_t& val
) {
    return op_compare_val_impl<int16_t, Op>(res_u8, src, size, val);
}

template<CompareOpType Op>
bool OpCompareValImpl<int32_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int32_t* const __restrict src, 
    const size_t size, 
    const int32_t& val 
) {
    return op_compare_val_impl<int32_t, Op>(res_u8, src, size, val);
}

template<CompareOpType Op>
bool OpCompareValImpl<int64_t, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const int64_t* const __restrict src, 
    const size_t size, 
    const int64_t& val
) {
    return op_compare_val_impl<int64_t, Op>(res_u8, src, size, val);
}

template<CompareOpType Op>
bool OpCompareValImpl<float, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const float* const __restrict src, 
    const size_t size, 
    const float& val
) {
    return op_compare_val_impl<float, Op>(res_u8, src, size, val);
}

template<CompareOpType Op>
bool OpCompareValImpl<double, Op>::op_compare_val(
    uint8_t* const __restrict res_u8,
    const double* const __restrict src, 
    const size_t size, 
    const double& val 
) {
    return op_compare_val_impl<double, Op>(res_u8, src, size, val);
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
template<typename T, CompareOpType CmpOp>
bool op_compare_column_impl(
    uint8_t* const __restrict res_u8,
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    using sve_t = SVEVector<T>;

    // SVE width in bytes
    const size_t sve_width = sve_t::width();
    assert((sve_width % 8) == 0);

    // Only 512 bits are implemented for now. This is because 
    //   512 bits hold 8 64-bit values. If the width is lower, then
    //   a different code is needed, just like for AVX2.
    if (sve_width * 8 * sizeof(T) != 512) {
        return false;
    }

    //
    const svbool_t pred_all = sve_t::pred_all();

    // process big blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const auto left_v = sve_t::load(pred_all, left + i);
        const auto right_v = sve_t::load(pred_all, right + i);
        const svbool_t cmp = CmpHelper<CmpOp>::compare(pred_all, left_v, right_v);
        
        MaskWriter<T, 512>::write_full(res_u8 + i / 8, cmp);
    }

    // process leftovers
    if (size_sve != size) {
        const svbool_t pred_op = get_pred_op<T>(size - size_sve);
        const svbool_t pred_write = get_pred_write(size - size_sve);

        const auto left_v = sve_t::load(pred_op, left + size_sve);
        const auto right_v = sve_t::load(pred_op, right + size_sve);
        const svbool_t cmp = CmpHelper<CmpOp>::compare(pred_op, left_v, right_v);

        MaskWriter<T, 512>::write_partial(res_u8 + size_sve / 8, cmp, pred_write);
    }

    return true;
}

//
template<CompareOpType Op>
bool OpCompareColumnImpl<int8_t, int8_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int8_t* const __restrict left, 
    const int8_t* const __restrict right, 
    const size_t size
) {
    return op_compare_column_impl<int8_t, Op>(res_u8, left, right, size);
}

template<CompareOpType Op>
bool OpCompareColumnImpl<int16_t, int16_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict left, 
    const int16_t* const __restrict right, 
    const size_t size
) {
    return op_compare_column_impl<int16_t, Op>(res_u8, left, right, size);
}

template<CompareOpType Op>
bool OpCompareColumnImpl<int32_t, int32_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int32_t* const __restrict left, 
    const int32_t* const __restrict right, 
    const size_t size
) {
    return op_compare_column_impl<int32_t, Op>(res_u8, left, right, size);
}

template<CompareOpType Op>
bool OpCompareColumnImpl<int64_t, int64_t, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const int64_t* const __restrict left, 
    const int64_t* const __restrict right, 
    const size_t size
) {
    return op_compare_column_impl<int64_t, Op>(res_u8, left, right, size);
}

template<CompareOpType Op>
bool OpCompareColumnImpl<float, float, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size
) {
    return op_compare_column_impl<float, Op>(res_u8, left, right, size);
}

template<CompareOpType Op>
bool OpCompareColumnImpl<double, double, Op>::op_compare_column(
    uint8_t* const __restrict res_u8,
    const double* const __restrict left, 
    const double* const __restrict right, 
    const size_t size
) {
    return op_compare_column_impl<double, Op>(res_u8, left, right, size);
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

template<typename T, RangeType Op>
bool op_within_range_column_impl(
    uint8_t* const __restrict res_u8,
    const T* const __restrict lower,
    const T* const __restrict upper,
    const T* const __restrict values,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    using sve_t = SVEVector<T>;

    // SVE width in bytes
    const size_t sve_width = sve_t::width();
    assert((sve_width % 8) == 0);

    // Only 512 bits are implemented for now. This is because 
    //   512 bits hold 8 64-bit values. If the width is lower, then
    //   a different code is needed, just like for AVX2.
    if (sve_width * 8 * sizeof(T) != 512) {
        return false;
    }

    //
    const svbool_t pred_all = sve_t::pred_all();

    // process big blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const auto lower_v = sve_t::load(pred_all, lower + i);
        const auto upper_v = sve_t::load(pred_all, upper + i);
        const auto values_v = sve_t::load(pred_all, values + i);

        const svbool_t cmpl = CmpHelper<Range2Compare<Op>::lower>::compare(pred_all, lower_v, values_v);
        const svbool_t cmpu = CmpHelper<Range2Compare<Op>::upper>::compare(pred_all, values_v, upper_v);
        const svbool_t cmp = svand_b_z(pred_all, cmpl, cmpu);

        MaskWriter<T, 512>::write_full(res_u8 + i / 8, cmp);
    }

    // process leftovers
    if (size_sve != size) {
        const svbool_t pred_op = get_pred_op<T>(size - size_sve);
        const svbool_t pred_write = get_pred_write(size - size_sve);

        const auto lower_v = sve_t::load(pred_op, lower + size_sve);
        const auto upper_v = sve_t::load(pred_op, upper + size_sve);
        const auto values_v = sve_t::load(pred_op, values + size_sve);

        const svbool_t cmpl = CmpHelper<Range2Compare<Op>::lower>::compare(pred_op, lower_v, values_v);
        const svbool_t cmpu = CmpHelper<Range2Compare<Op>::upper>::compare(pred_op, values_v, upper_v);
        const svbool_t cmp = svand_b_z(pred_op, cmpl, cmpu);

        MaskWriter<T, 512>::write_partial(res_u8 + size_sve / 8, cmp, pred_write);
    }

    return true;
}

//
template<RangeType Op>
bool OpWithinRangeColumnImpl<int8_t, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const int8_t* const __restrict lower,
    const int8_t* const __restrict upper,
    const int8_t* const __restrict values,
    const size_t size
) {
    return op_within_range_column_impl<int8_t, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<int16_t, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const int16_t* const __restrict lower,
    const int16_t* const __restrict upper,
    const int16_t* const __restrict values,
    const size_t size
) {
    return op_within_range_column_impl<int16_t, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<int32_t, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const int32_t* const __restrict lower,
    const int32_t* const __restrict upper,
    const int32_t* const __restrict values,
    const size_t size
) {
    return op_within_range_column_impl<int32_t, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<int64_t, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const int64_t* const __restrict lower,
    const int64_t* const __restrict upper,
    const int64_t* const __restrict values,
    const size_t size
) {
    return op_within_range_column_impl<int64_t, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<float, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const float* const __restrict lower,
    const float* const __restrict upper,
    const float* const __restrict values,
    const size_t size
) {
    return op_within_range_column_impl<float, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeColumnImpl<double, Op>::op_within_range_column(
    uint8_t* const __restrict res_u8,
    const double* const __restrict lower,
    const double* const __restrict upper,
    const double* const __restrict values,
    const size_t size
) {
    return op_within_range_column_impl<double, Op>(res_u8, lower, upper, values, size);
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

template<typename T, RangeType Op>
bool op_within_range_val_impl(
    uint8_t* const __restrict res_u8,
    const T& lower,
    const T& upper,
    const T* const __restrict values,
    const size_t size
) {
    // the restriction of the API
    assert((size % 8) == 0);

    //
    using sve_t = SVEVector<T>;

    // SVE width in bytes
    const size_t sve_width = sve_t::width();
    assert((sve_width % 8) == 0);

    // Only 512 bits are implemented for now. This is because 
    //   512 bits hold 8 64-bit values. If the width is lower, then
    //   a different code is needed, just like for AVX2.
    if (sve_width * 8 * sizeof(T) != 512) {
        return false;
    }

    //
    const svbool_t pred_all = sve_t::pred_all();
    const auto lower_v = sve_t::set1(lower);
    const auto upper_v = sve_t::set1(upper);

    // process big blocks
    const size_t size_sve = (size / sve_width) * sve_width;
    for (size_t i = 0; i < size_sve; i += sve_width) {
        const auto values_v = sve_t::load(pred_all, values + i);

        const svbool_t cmpl = CmpHelper<Range2Compare<Op>::lower>::compare(pred_all, lower_v, values_v);
        const svbool_t cmpu = CmpHelper<Range2Compare<Op>::upper>::compare(pred_all, values_v, upper_v);
        const svbool_t cmp = svand_b_z(pred_all, cmpl, cmpu);

        MaskWriter<T, 512>::write_full(res_u8 + i / 8, cmp);
    }

    // process leftovers
    if (size_sve != size) {
        const svbool_t pred_op = get_pred_op<T>(size - size_sve);
        const svbool_t pred_write = get_pred_write(size - size_sve);

        const auto values_v = sve_t::load(pred_op, values + size_sve);

        const svbool_t cmpl = CmpHelper<Range2Compare<Op>::lower>::compare(pred_op, lower_v, values_v);
        const svbool_t cmpu = CmpHelper<Range2Compare<Op>::upper>::compare(pred_op, values_v, upper_v);
        const svbool_t cmp = svand_b_z(pred_op, cmpl, cmpu);

        MaskWriter<T, 512>::write_partial(res_u8 + size_sve / 8, cmp, pred_write);
    }

    return true;
}

//
template<RangeType Op>
bool OpWithinRangeValImpl<int8_t, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const int8_t& lower,
    const int8_t& upper,
    const int8_t* const __restrict values,
    const size_t size
) {
    return op_within_range_val_impl<int8_t, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeValImpl<int16_t, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const int16_t& lower,
    const int16_t& upper,
    const int16_t* const __restrict values,
    const size_t size
) {
    return op_within_range_val_impl<int16_t, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeValImpl<int32_t, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const int32_t& lower,
    const int32_t& upper,
    const int32_t* const __restrict values,
    const size_t size
) {
    return op_within_range_val_impl<int32_t, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeValImpl<int64_t, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const int64_t& lower,
    const int64_t& upper,
    const int64_t* const __restrict values,
    const size_t size
) {
    return op_within_range_val_impl<int64_t, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeValImpl<float, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const float& lower,
    const float& upper,
    const float* const __restrict values,
    const size_t size
) {
    return op_within_range_val_impl<float, Op>(res_u8, lower, upper, values, size);
}

template<RangeType Op>
bool OpWithinRangeValImpl<double, Op>::op_within_range_val(
    uint8_t* const __restrict res_u8,
    const double& lower,
    const double& upper,
    const double* const __restrict values,
    const size_t size
) {
    return op_within_range_val_impl<double, Op>(res_u8, lower, upper, values, size);
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
