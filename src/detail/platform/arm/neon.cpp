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

// a facility to run through all possible operations
#define ALL_OPS(FUNC,...) \
    FUNC(Equal,__VA_ARGS__); \
    FUNC(GreaterEqual,__VA_ARGS__); \
    FUNC(Greater,__VA_ARGS__); \
    FUNC(LessEqual,__VA_ARGS__); \
    FUNC(Less,__VA_ARGS__); \
    FUNC(NotEqual,__VA_ARGS__);

//
inline uint32_t movemask_32(const uint32x4_t cmp) {
    static const int32_t shifts[4] = {0, 1, 2, 3};
    // shift
    const uint32x4_t sh = vshrq_n_u32(cmp, 31);
    // load shifts
    const int32x4_t shifts_v = vld1q_s32(shifts);
    // shift values differently
    const uint32x4_t shifted_bits = vshlq_u32(sh, shifts_v);
    // horizontal sum of bits on different positions
    return vaddvq_u32(shifted_bits);
}

inline uint32_t movemask_32(const uint32x4x2_t cmp) {
    return movemask_32(cmp.val[0]) + 4 * movemask_32(cmp.val[1]);
}

//
template<CompareType Op>
struct CmpHelperF32{};

template<>
struct CmpHelperF32<CompareType::EQ> {
    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vceqq_f32(a.val[0], b.val[0]), vceqq_f32(a.val[1], b.val[1])};
    }
};

template<>
struct CmpHelperF32<CompareType::GE> {
    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vcgeq_f32(a.val[0], b.val[0]), vcgeq_f32(a.val[1], b.val[1])};
    }
};

template<>
struct CmpHelperF32<CompareType::GT> {
    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vcgtq_f32(a.val[0], b.val[0]), vcgtq_f32(a.val[1], b.val[1])};
    }
};

template<>
struct CmpHelperF32<CompareType::LE> {
    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vcleq_f32(a.val[0], b.val[0]), vcleq_f32(a.val[1], b.val[1])};
    }
};

template<>
struct CmpHelperF32<CompareType::LT> {
    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vcltq_f32(a.val[0], b.val[0]), vcltq_f32(a.val[1], b.val[1])};
    }
};

template<>
struct CmpHelperF32<CompareType::NEQ> {
    static inline uint32x4x2_t compare(const float32x4x2_t a, const float32x4x2_t b) {
        return {vmvnq_u32(vceqq_f32(a.val[0], b.val[0])), vmvnq_u32(vceqq_f32(a.val[1], b.val[1]))};
    }
};

//
template <typename T, CompareType Op>
struct CompareValNeonImpl {};

template<CompareType Op>
struct CompareValNeonImpl<int8_t, Op> {
    static void Compare(
        const int8_t* const __restrict src, 
        const size_t size, 
        const int8_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
    }
};

template<CompareType Op>
struct CompareValNeonImpl<int16_t, Op> {
    static void Compare(
        const int16_t* const __restrict src, 
        const size_t size, 
        const int16_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
    }
};

template<CompareType Op>
struct CompareValNeonImpl<int32_t, Op> {
    static void Compare(
        const int32_t* const __restrict src, 
        const size_t size, 
        const int32_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
    }
};

template<CompareType Op>
struct CompareValNeonImpl<int64_t, Op> {
    static void Compare(
        const int64_t* const __restrict src, 
        const size_t size, 
        const int64_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
    }
};

template<CompareType Op>
struct CompareValNeonImpl<float, Op> {
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
        const float32x4x2_t target = {vdupq_n_f32(val), vdupq_n_f32(val)};

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const float32x4x2_t v0 = {vld1q_f32(src + i), vld1q_f32(src + i + 4)};
            const uint32x4x2_t cmp = CmpHelperF32<Op>::compare(v0, target);
            const uint8_t mmask = movemask_32(cmp);

            res_u8[i / 8] = mmask;
        }
    }
};

template<CompareType Op>
struct CompareValNeonImpl<double, Op> {
    static void Compare(
        const double* const __restrict src, 
        const size_t size, 
        const double val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);
    }
};

//
#define DECLARE_VAL_NEON(NAME, OP) \
    template<typename T> \
    void NAME##ValNeon( \
        const T* const __restrict src, \
        const size_t size, \
        const T val, \
        void* const __restrict res \
    ) { \
        CompareValNeonImpl<T, CompareType::OP>::Compare(src, size, val, res); \
    }

DECLARE_VAL_NEON(Equal, EQ);    
DECLARE_VAL_NEON(GreaterEqual, GE);
DECLARE_VAL_NEON(Greater, GT);
DECLARE_VAL_NEON(LessEqual, LE);
DECLARE_VAL_NEON(Less, LT);
DECLARE_VAL_NEON(NotEqual, NEQ);

#undef DECLARE_VAL_NEON

#define INSTANTIATE_VAL_NEON(NAME, TTYPE) \
    template void NAME##ValNeon( \
        const TTYPE* const __restrict src, \
        const size_t size, \
        const TTYPE val, \
        void* const __restrict res \
    );

ALL_OPS(INSTANTIATE_VAL_NEON, int8_t)
ALL_OPS(INSTANTIATE_VAL_NEON, int16_t)
ALL_OPS(INSTANTIATE_VAL_NEON, int32_t)
ALL_OPS(INSTANTIATE_VAL_NEON, int64_t)
ALL_OPS(INSTANTIATE_VAL_NEON, float)
ALL_OPS(INSTANTIATE_VAL_NEON, double)

#undef INSTANTIATE_VAL_NEON

//
template <typename T, CompareType Op>
struct CompareColumnNeonImpl {};

template <CompareType Op>
struct CompareColumnNeonImpl<int8_t, Op> {
    static inline void Compare(
        const int8_t* const __restrict left, 
        const int8_t* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
    }
};

template <CompareType Op>
struct CompareColumnNeonImpl<int16_t, Op> {
    static inline void Compare(
        const int16_t* const __restrict left, 
        const int16_t* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
    }
};

template <CompareType Op>
struct CompareColumnNeonImpl<int32_t, Op> {
    static inline void Compare(
        const int32_t* const __restrict left, 
        const int32_t* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
    }
};

template <CompareType Op>
struct CompareColumnNeonImpl<int64_t, Op> {
    static inline void Compare(
        const int64_t* const __restrict left, 
        const int64_t* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
    }
};

template <CompareType Op>
struct CompareColumnNeonImpl<float, Op> {
    static inline void Compare(
        const float* const __restrict left, 
        const float* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
    }
};

template <CompareType Op>
struct CompareColumnNeonImpl<double, Op> {
    static inline void Compare(
        const double* const __restrict left, 
        const double* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
    }
};

#define DECLARE_COLUMN_NEON(NAME, OP) \
    template<typename T> \
    void NAME##ColumnNeon( \
        const T* const __restrict left, \
        const T* const __restrict right, \
        const size_t size, \
        void* const __restrict res \
    ) { \
        CompareColumnNeonImpl<T, CompareType::OP>::Compare(left, right, size, res); \
    }

DECLARE_COLUMN_NEON(Equal, EQ);
DECLARE_COLUMN_NEON(GreaterEqual, GE);
DECLARE_COLUMN_NEON(Greater, GT);
DECLARE_COLUMN_NEON(LessEqual, LE);
DECLARE_COLUMN_NEON(Less, LT);
DECLARE_COLUMN_NEON(NotEqual, NEQ);

#undef DECLARE_COLUMN_NEON

//
#define INSTANTIATE_COLUMN_NEON(NAME, TTYPE) \
    template void NAME##ColumnNeon( \
        const TTYPE* const __restrict left, \
        const TTYPE* const __restrict right, \
        const size_t size, \
        void* const __restrict res \
    );

ALL_OPS(INSTANTIATE_COLUMN_NEON, int8_t)
ALL_OPS(INSTANTIATE_COLUMN_NEON, int16_t)
ALL_OPS(INSTANTIATE_COLUMN_NEON, int32_t)
ALL_OPS(INSTANTIATE_COLUMN_NEON, int64_t)
ALL_OPS(INSTANTIATE_COLUMN_NEON, float)
ALL_OPS(INSTANTIATE_COLUMN_NEON, double)

#undef INSTANTIATE_COLUMN_NEON

//
#undef ALL_OPS

}
}
}
}
