#include "avx2.h"

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

// count is expected to be in range [0, 32)
inline uint32_t get_mask(const size_t count) {
    return (uint32_t(1) << count) - uint32_t(1);
}

//
template<CompareType Op>
struct CmpHelperI8{};

template<>
struct CmpHelperI8<CompareType::EQ> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpeq_epi8(a, b);
    }
};

template<>
struct CmpHelperI8<CompareType::GE> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpgt_epi8(b, a), _mm256_set1_epi32(-1));
    }
};

template<>
struct CmpHelperI8<CompareType::GT> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpgt_epi8(a, b);
    }
};

template<>
struct CmpHelperI8<CompareType::LE> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpgt_epi8(a, b), _mm256_set1_epi32(-1));
    }
};

template<>
struct CmpHelperI8<CompareType::LT> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpgt_epi8(b, a);
    }
};

template<>
struct CmpHelperI8<CompareType::NEQ> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpeq_epi8(a, b), _mm256_set1_epi32(-1));
    }
};

//
template<CompareType Op>
struct CmpHelperI16{};

template<>
struct CmpHelperI16<CompareType::EQ> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpeq_epi16(a, b);
    }

    static inline __m128i compare(const __m128i a, const __m128i b) {
        return _mm_cmpeq_epi16(a, b);
    }
};

template<>
struct CmpHelperI16<CompareType::GE> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpgt_epi16(b, a), _mm256_set1_epi32(-1));
    }

    static inline __m128i compare(const __m128i a, const __m128i b) {
        return _mm_xor_si128(_mm_cmpgt_epi16(b, a), _mm_set1_epi32(-1));
    }
};

template<>
struct CmpHelperI16<CompareType::GT> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpgt_epi16(a, b);
    }

    static inline __m128i compare(const __m128i a, const __m128i b) {
        return _mm_cmpgt_epi16(a, b);
    }
};

template<>
struct CmpHelperI16<CompareType::LE> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpgt_epi16(a, b), _mm256_set1_epi32(-1));
    }

    static inline __m128i compare(const __m128i a, const __m128i b) {
        return _mm_xor_si128(_mm_cmpgt_epi16(a, b), _mm_set1_epi32(-1));
    }
};

template<>
struct CmpHelperI16<CompareType::LT> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpgt_epi16(b, a);
    }

    static inline __m128i compare(const __m128i a, const __m128i b) {
        return _mm_cmpgt_epi16(b, a);
    }
};

template<>
struct CmpHelperI16<CompareType::NEQ> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpeq_epi16(a, b), _mm256_set1_epi32(-1));
    }

    static inline __m128i compare(const __m128i a, const __m128i b) {
        return _mm_xor_si128(_mm_cmpeq_epi16(a, b), _mm_set1_epi32(-1));
    }
};

//
template<CompareType Op>
struct CmpHelperI32{};

template<>
struct CmpHelperI32<CompareType::EQ> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpeq_epi32(a, b);
    }
};

template<>
struct CmpHelperI32<CompareType::GE> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpgt_epi32(b, a), _mm256_set1_epi32(-1));
    }
};

template<>
struct CmpHelperI32<CompareType::GT> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpgt_epi32(a, b);
    }
};

template<>
struct CmpHelperI32<CompareType::LE> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpgt_epi32(a, b), _mm256_set1_epi32(-1));
    }
};

template<>
struct CmpHelperI32<CompareType::LT> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpgt_epi32(b, a);
    }
};

template<>
struct CmpHelperI32<CompareType::NEQ> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpeq_epi32(a, b), _mm256_set1_epi32(-1));
    }
};


//
template<CompareType Op>
struct CmpHelperI64{};

template<>
struct CmpHelperI64<CompareType::EQ> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpeq_epi64(a, b);
    }
};

template<>
struct CmpHelperI64<CompareType::GE> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpgt_epi64(b, a), _mm256_set1_epi32(-1));
    }
};

template<>
struct CmpHelperI64<CompareType::GT> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpgt_epi64(a, b);
    }
};

template<>
struct CmpHelperI64<CompareType::LE> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpgt_epi64(a, b), _mm256_set1_epi32(-1));
    }
};

template<>
struct CmpHelperI64<CompareType::LT> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_cmpgt_epi64(b, a);
    }
};

template<>
struct CmpHelperI64<CompareType::NEQ> {
    static inline __m256i compare(const __m256i a, const __m256i b) {
        return _mm256_xor_si256(_mm256_cmpeq_epi64(a, b), _mm256_set1_epi32(-1));
    }
};

///////////////////////////////////////////////////////////////////////////

//
template <typename T, CompareType Op>
struct CompareValAVX2Impl {};

template<CompareType Op>
struct CompareValAVX2Impl<int8_t, Op> {
    static void compare(
        const int8_t* const __restrict src, 
        const size_t size, 
        const int8_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res);
        const __m256i target = _mm256_set1_epi8(val);

        // todo: aligned reads & writes

        const size_t size32 = (size / 32) * 32;
        for (size_t i = 0; i < size32; i += 32) {
            const __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + i));
            const __m256i cmp = CmpHelperI8<Op>::compare(v0, target);
            const uint32_t mmask = _mm256_movemask_epi8(cmp);

            res_u32[i / 32] = mmask;
        }

        if (size32 != size) {
            // 8, 16 or 24 elements to process
            const __m256i mask = _mm256_setr_epi64x(
                (size - size32 >= 8) ? (-1) : 0,
                (size - size32 >= 16) ? (-1) : 0,
                (size - size32 >= 24) ? (-1) : 0,
                0
            );

            const __m256i v0 = _mm256_maskload_epi64((const long long*)(src + size32), mask);
            const __m256i cmp = CmpHelperI8<Op>::compare(v0, target);
            const uint32_t mmask = _mm256_movemask_epi8(cmp);

            if (size - size32 >= 8) {
                res_u8[size32 / 8 + 0] = (mmask & 0xFF);
            }
            if (size - size32 >= 16) {
                res_u8[size32 / 8 + 1] = ((mmask >> 8) & 0xFF);
            }
            if (size - size32 >= 24) {
                res_u8[size32 / 8 + 2] = ((mmask >> 16) & 0xFF);
            }
        }
    }
};

template<CompareType Op>
struct CompareValAVX2Impl<int16_t, Op> {
    static void compare(
        const int16_t* const __restrict src, 
        const size_t size, 
        const int16_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res);
        const __m256i target = _mm256_set1_epi16(val);

        // todo: aligned reads & writes

        const size_t size16 = (size / 16) * 16;
        for (size_t i = 0; i < size16; i += 16) {
            const __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + i));
            const __m256i cmp = CmpHelperI16<Op>::compare(v0, target);
            const __m256i pcmp = _mm256_packs_epi16(cmp, cmp);
            const __m256i qcmp = _mm256_permute4x64_epi64(pcmp, _MM_SHUFFLE(3, 1, 2, 0));
            const uint16_t mmask = _mm256_movemask_epi8(qcmp);

            res_u16[i / 16] = mmask;
        }

        if (size16 != size) {
            // 8 elements to process
            const __m128i v0 = _mm_loadu_si128((const __m128i*)(src + size16));
            const __m128i target0 = _mm_set1_epi16(val);
            const __m128i cmp = CmpHelperI16<Op>::compare(v0, target0);
            const __m128i pcmp = _mm_packs_epi16(cmp, cmp);
            const uint32_t mmask = _mm_movemask_epi8(pcmp) & 0xFF;

            res_u8[size16 / 8] = mmask;
        }
    }
};

template<CompareType Op>
struct CompareValAVX2Impl<int32_t, Op> {
    static void compare(
        const int32_t* const __restrict src, 
        const size_t size, 
        const int32_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        const __m256i target = _mm256_set1_epi32(val);

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + i));
            const __m256i cmp = CmpHelperI32<Op>::compare(v0, target);
            const uint8_t mmask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

            res_u8[i / 8] = mmask;
        }
    }
};

template<CompareType Op>
struct CompareValAVX2Impl<int64_t, Op> {
    static void compare(
        const int64_t* const __restrict src, 
        const size_t size, 
        const int64_t val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        const __m256i target = _mm256_set1_epi64x(val);

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256i v0 = _mm256_loadu_si256((const __m256i*)(src + i));
            const __m256i v1 = _mm256_loadu_si256((const __m256i*)(src + i + 4));
            const __m256i cmp0 = CmpHelperI64<Op>::compare(v0, target);
            const __m256i cmp1 = CmpHelperI64<Op>::compare(v1, target);
            const uint8_t mmask0 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp0));
            const uint8_t mmask1 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp1));

            res_u8[i / 8] = mmask0 + mmask1 * 16;
        }
    }
};

template<CompareType Op>
struct CompareValAVX2Impl<float, Op> {
    static void compare(
        const float* const __restrict src, 
        const size_t size, 
        const float val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        constexpr auto pred = ComparePredicate<float, Op>::value;

        const __m256 target = _mm256_set1_ps(val);

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256 v0 = _mm256_loadu_ps(src + i);
            const __m256 cmp = _mm256_cmp_ps(v0, target, pred);
            const uint8_t mmask = _mm256_movemask_ps(cmp);

            res_u8[i / 8] = mmask;
        }
    }
};


template<CompareType Op>
struct CompareValAVX2Impl<double, Op> {
    static void compare(
        const double* const __restrict src, 
        const size_t size, 
        const double val, 
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        constexpr auto pred = ComparePredicate<float, Op>::value;

        const __m256d target = _mm256_set1_pd(val);

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256d v0 = _mm256_loadu_pd(src + i);
            const __m256d v1 = _mm256_loadu_pd(src + i + 4);
            const __m256d cmp0 = _mm256_cmp_pd(v0, target, pred);
            const __m256d cmp1 = _mm256_cmp_pd(v1, target, pred);
            const uint8_t mmask0 = _mm256_movemask_pd(cmp0);
            const uint8_t mmask1 = _mm256_movemask_pd(cmp1);

            res_u8[i / 8] = mmask0 + mmask1 * 16;
        }
    }
};

//
template<typename T, CompareType Op>
void CompareValAVX2(const T* const __restrict src, const size_t size, const T val, uint8_t* const __restrict res) {
    CompareValAVX2Impl<T, Op>::compare(src, size, val, res);
}

#define INSTANTIATE_COMPARE_VAL_AVX2(TTYPE,OP) \
    template void CompareValAVX2<TTYPE, CompareType::OP>( \
        const TTYPE* const __restrict src, \
        const size_t size, \
        const TTYPE val, \
        uint8_t* const __restrict res \
    );

ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX2, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX2, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX2, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX2, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX2, float)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_VAL_AVX2, double)

#undef INSTANTIATE_COMPARE_VAL_AVX2

///////////////////////////////////////////////////////////////////////////

//
template <typename T, CompareType Op>
struct CompareColumnAVX2Impl {};

template <CompareType Op>
struct CompareColumnAVX2Impl<int8_t, Op> {
    static inline void compare(
        const int8_t* const __restrict left, 
        const int8_t* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res);

        // todo: aligned reads & writes

        const size_t size32 = (size / 32) * 32;
        for (size_t i = 0; i < size32; i += 32) {
            const __m256i v0l = _mm256_loadu_si256((const __m256i*)(left + i));
            const __m256i v0r = _mm256_loadu_si256((const __m256i*)(right + i));
            const __m256i cmp = CmpHelperI8<Op>::compare(v0l, v0r);
            const uint32_t mmask = _mm256_movemask_epi8(cmp);

            res_u32[i / 32] = mmask;
        }

        if (size32 != size) {
            // 8, 16 or 24 elements to process
            const __m256i mask = _mm256_setr_epi64x(
                (size - size32 >= 8) ? (-1) : 0,
                (size - size32 >= 16) ? (-1) : 0,
                (size - size32 >= 24) ? (-1) : 0,
                0
            );

            const __m256i v0l = _mm256_maskload_epi64((const long long*)(left + size32), mask);
            const __m256i v0r = _mm256_maskload_epi64((const long long*)(right + size32), mask);
            const __m256i cmp = CmpHelperI8<Op>::compare(v0l, v0r);
            const uint32_t mmask = _mm256_movemask_epi8(cmp);

            if (size - size32 >= 8) {
                res_u8[size32 / 8 + 0] = (mmask & 0xFF);
            }
            if (size - size32 >= 16) {
                res_u8[size32 / 8 + 1] = ((mmask >> 8) & 0xFF);
            }
            if (size - size32 >= 24) {
                res_u8[size32 / 8 + 2] = ((mmask >> 16) & 0xFF);
            }
        }
    }
};

template <CompareType Op>
struct CompareColumnAVX2Impl<int16_t, Op> {
    static inline void compare(
        const int16_t* const __restrict left, 
        const int16_t* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res);

        // todo: aligned reads & writes

        const size_t size16 = (size / 16) * 16;
        for (size_t i = 0; i < size16; i += 16) {
            const __m256i v0l = _mm256_loadu_si256((const __m256i*)(left + i));
            const __m256i v0r = _mm256_loadu_si256((const __m256i*)(right + i));
            const __m256i cmp = CmpHelperI16<Op>::compare(v0l, v0r);
            const __m256i pcmp = _mm256_packs_epi16(cmp, cmp);
            const __m256i qcmp = _mm256_permute4x64_epi64(pcmp, _MM_SHUFFLE(3, 1, 2, 0));
            const uint16_t mmask = _mm256_movemask_epi8(qcmp);

            res_u16[i / 16] = mmask;
        }

        if (size16 != size) {
            // 8 elements to process
            const __m128i v0l = _mm_loadu_si128((const __m128i*)(left + size16));
            const __m128i v0r = _mm_loadu_si128((const __m128i*)(right + size16));
            const __m128i cmp = CmpHelperI16<Op>::compare(v0l, v0r);
            const __m128i pcmp = _mm_packs_epi16(cmp, cmp);
            const uint32_t mmask = _mm_movemask_epi8(pcmp) & 0xFF;

            res_u8[size16 / 8] = mmask;
        }
    }
};

template <CompareType Op>
struct CompareColumnAVX2Impl<int32_t, Op> {
    static inline void compare(
        const int32_t* const __restrict left, 
        const int32_t* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256i v0l = _mm256_loadu_si256((const __m256i*)(left + i));
            const __m256i v0r = _mm256_loadu_si256((const __m256i*)(right + i));
            const __m256i cmp = CmpHelperI32<Op>::compare(v0l, v0r);
            const uint8_t mmask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

            res_u8[i / 8] = mmask;
        }
    }
};

template <CompareType Op>
struct CompareColumnAVX2Impl<int64_t, Op> {
    static inline void compare(
        const int64_t* const __restrict left, 
        const int64_t* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256i v0l = _mm256_loadu_si256((const __m256i*)(left + i));
            const __m256i v1l = _mm256_loadu_si256((const __m256i*)(left + i + 4));
            const __m256i v0r = _mm256_loadu_si256((const __m256i*)(right + i));
            const __m256i v1r = _mm256_loadu_si256((const __m256i*)(right + i + 4));
            const __m256i cmp0 = CmpHelperI64<Op>::compare(v0l, v0r);
            const __m256i cmp1 = CmpHelperI64<Op>::compare(v1l, v1r);
            const uint8_t mmask0 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp0));
            const uint8_t mmask1 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp1));

            res_u8[i / 8] = mmask0 + mmask1 * 16;
        }
    }
};

template <CompareType Op>
struct CompareColumnAVX2Impl<float, Op> {
    static inline void compare(
        const float* const __restrict left, 
        const float* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        constexpr auto pred = ComparePredicate<float, Op>::value;

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256 v0l = _mm256_loadu_ps(left + i);
            const __m256 v0r = _mm256_loadu_ps(right + i);
            const __m256 cmp = _mm256_cmp_ps(v0l, v0r, pred);
            const uint8_t mmask = _mm256_movemask_ps(cmp);

            res_u8[i / 8] = mmask;
        }
    }
};

template <CompareType Op>
struct CompareColumnAVX2Impl<double, Op> {
    static inline void compare(
        const double* const __restrict left, 
        const double* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        constexpr auto pred = ComparePredicate<double, Op>::value;

        // todo: aligned reads & writes
        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256d v0l = _mm256_loadu_pd(left + i);
            const __m256d v1l = _mm256_loadu_pd(left + i + 4);
            const __m256d v0r = _mm256_loadu_pd(right + i);
            const __m256d v1r = _mm256_loadu_pd(right + i + 4);
            const __m256d cmp0 = _mm256_cmp_pd(v0l, v0r, pred);
            const __m256d cmp1 = _mm256_cmp_pd(v1l, v1r, pred);
            const uint8_t mmask0 = _mm256_movemask_pd(cmp0);
            const uint8_t mmask1 = _mm256_movemask_pd(cmp1);

            res_u8[i / 8] = mmask0 + mmask1 * 16;
        }
    }
};

//
template<typename T, CompareType Op>
void CompareColumnAVX2(const T* const __restrict left, const T* const __restrict right, const size_t size, uint8_t* const __restrict res) {
    CompareColumnAVX2Impl<T, Op>::compare(left, right, size, res);
}

#define INSTANTIATE_COMPARE_COLUMN_AVX2(TTYPE,OP) \
    template void CompareColumnAVX2<TTYPE, CompareType::OP>( \
        const TTYPE* const __restrict left, \
        const TTYPE* const __restrict right, \
        const size_t size, \
        uint8_t* const __restrict res \
    );

ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, float)
ALL_COMPARE_OPS(INSTANTIATE_COMPARE_COLUMN_AVX2, double)

#undef INSTANTIATE_COMPARE_COLUMN_AVX2

///////////////////////////////////////////////////////////////////////////

template<RangeType Op>
struct Range2Compare {};

template<>
struct Range2Compare<RangeType::IncInc> {
    static constexpr inline CompareType lower = CompareType::LE;  
    static constexpr inline CompareType upper = CompareType::LE;  
};

template<>
struct Range2Compare<RangeType::IncExc> {
    static constexpr inline CompareType lower = CompareType::LE;  
    static constexpr inline CompareType upper = CompareType::LT;  
};

template<>
struct Range2Compare<RangeType::ExcInc> {
    static constexpr inline CompareType lower = CompareType::LT;  
    static constexpr inline CompareType upper = CompareType::LE;  
};

template<>
struct Range2Compare<RangeType::ExcExc> {
    static constexpr inline CompareType lower = CompareType::LT;  
    static constexpr inline CompareType upper = CompareType::LT;  
};

//
template <typename T, RangeType Op>
struct WithinRangeAVX2Impl {};

template<RangeType Op>
struct WithinRangeAVX2Impl<int8_t, Op> {
    static inline void within_range(
        const int8_t* const __restrict lower,
        const int8_t* const __restrict upper,
        const int8_t* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        uint32_t* const __restrict res_u32 = reinterpret_cast<uint32_t*>(res);

        // todo: aligned reads & writes

        const size_t size32 = (size / 32) * 32;
        for (size_t i = 0; i < size32; i += 32) {
            const __m256i v0l = _mm256_loadu_si256((const __m256i*)(lower + i));
            const __m256i v0u = _mm256_loadu_si256((const __m256i*)(upper + i));
            const __m256i v0v = _mm256_loadu_si256((const __m256i*)(values + i));
            const __m256i cmpl = CmpHelperI8<Range2Compare<Op>::lower>::compare(v0l, v0v);
            const __m256i cmpu = CmpHelperI8<Range2Compare<Op>::upper>::compare(v0v, v0u);
            const __m256i cmp = _mm256_and_si256(cmpl, cmpu);
            const uint32_t mmask = _mm256_movemask_epi8(cmp);

            res_u32[i / 32] = mmask;
        }

        if (size32 != size) {
            // 8, 16 or 24 elements to process
            const __m256i mask = _mm256_setr_epi64x(
                (size - size32 >= 8) ? (-1) : 0,
                (size - size32 >= 16) ? (-1) : 0,
                (size - size32 >= 24) ? (-1) : 0,
                0
            );

            const __m256i v0l = _mm256_maskload_epi64((const long long*)(lower + size32), mask);
            const __m256i v0u = _mm256_maskload_epi64((const long long*)(upper + size32), mask);
            const __m256i v0v = _mm256_maskload_epi64((const long long*)(values + size32), mask);
            const __m256i cmpl = CmpHelperI8<Range2Compare<Op>::lower>::compare(v0l, v0v);
            const __m256i cmpu = CmpHelperI8<Range2Compare<Op>::upper>::compare(v0v, v0u);
            const __m256i cmp = _mm256_and_si256(cmpl, cmpu);
            const uint32_t mmask = _mm256_movemask_epi8(cmp);

            if (size - size32 >= 8) {
                res_u8[size32 / 8 + 0] = (mmask & 0xFF);
            }
            if (size - size32 >= 16) {
                res_u8[size32 / 8 + 1] = ((mmask >> 8) & 0xFF);
            }
            if (size - size32 >= 24) {
                res_u8[size32 / 8 + 2] = ((mmask >> 16) & 0xFF);
            }
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX2Impl<int16_t, Op> {
    static inline void within_range(
        const int16_t* const __restrict lower,
        const int16_t* const __restrict upper,
        const int16_t* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        uint16_t* const __restrict res_u16 = reinterpret_cast<uint16_t*>(res);

        // todo: aligned reads & writes

        const size_t size16 = (size / 16) * 16;
        for (size_t i = 0; i < size16; i += 16) {
            const __m256i v0l = _mm256_loadu_si256((const __m256i*)(lower + i));
            const __m256i v0u = _mm256_loadu_si256((const __m256i*)(upper + i));
            const __m256i v0v = _mm256_loadu_si256((const __m256i*)(values + i));
            const __m256i cmpl = CmpHelperI16<Range2Compare<Op>::lower>::compare(v0l, v0v);
            const __m256i cmpu = CmpHelperI16<Range2Compare<Op>::upper>::compare(v0v, v0u);
            const __m256i cmp = _mm256_and_si256(cmpl, cmpu);
            const __m256i pcmp = _mm256_packs_epi16(cmp, cmp);
            const __m256i qcmp = _mm256_permute4x64_epi64(pcmp, _MM_SHUFFLE(3, 1, 2, 0));
            const uint16_t mmask = _mm256_movemask_epi8(qcmp);

            res_u16[i / 16] = mmask;
        }

        if (size16 != size) {
            // 8 elements to process
            const __m128i v0l = _mm_loadu_si128((const __m128i*)(lower + size16));
            const __m128i v0u = _mm_loadu_si128((const __m128i*)(upper + size16));
            const __m128i v0v = _mm_loadu_si128((const __m128i*)(values + size16));
            const __m128i cmpl = CmpHelperI16<Range2Compare<Op>::lower>::compare(v0l, v0v);
            const __m128i cmpu = CmpHelperI16<Range2Compare<Op>::upper>::compare(v0v, v0u);
            const __m128i cmp = _mm_and_si128(cmpl, cmpu);
            const __m128i pcmp = _mm_packs_epi16(cmp, cmp);
            const uint32_t mmask = _mm_movemask_epi8(pcmp) & 0xFF;

            res_u8[size16 / 8] = mmask;
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX2Impl<int32_t, Op> {
    static inline void within_range(
        const int32_t* const __restrict lower,
        const int32_t* const __restrict upper,
        const int32_t* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256i v0l = _mm256_loadu_si256((const __m256i*)(lower + i));
            const __m256i v0u = _mm256_loadu_si256((const __m256i*)(upper + i));
            const __m256i v0v = _mm256_loadu_si256((const __m256i*)(values + i));
            const __m256i cmpl = CmpHelperI32<Range2Compare<Op>::lower>::compare(v0l, v0v);
            const __m256i cmpu = CmpHelperI32<Range2Compare<Op>::upper>::compare(v0v, v0u);
            const __m256i cmp = _mm256_and_si256(cmpl, cmpu);
            const uint8_t mmask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

            res_u8[i / 8] = mmask;
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX2Impl<int64_t, Op> {
    static inline void within_range(
        const int64_t* const __restrict lower,
        const int64_t* const __restrict upper,
        const int64_t* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256i v0l = _mm256_loadu_si256((const __m256i*)(lower + i));
            const __m256i v1l = _mm256_loadu_si256((const __m256i*)(lower + i + 4));
            const __m256i v0u = _mm256_loadu_si256((const __m256i*)(upper + i));
            const __m256i v1u = _mm256_loadu_si256((const __m256i*)(upper + i + 4));
            const __m256i v0v = _mm256_loadu_si256((const __m256i*)(values + i));
            const __m256i v1v = _mm256_loadu_si256((const __m256i*)(values + i + 4));
            const __m256i cmp0l = CmpHelperI64<Range2Compare<Op>::lower>::compare(v0l, v0v);
            const __m256i cmp0u = CmpHelperI64<Range2Compare<Op>::upper>::compare(v0v, v0u);
            const __m256i cmp1l = CmpHelperI64<Range2Compare<Op>::lower>::compare(v1l, v1v);
            const __m256i cmp1u = CmpHelperI64<Range2Compare<Op>::upper>::compare(v1v, v1u);
            const __m256i cmp0 = _mm256_and_si256(cmp0l, cmp0u);
            const __m256i cmp1 = _mm256_and_si256(cmp1l, cmp1u);
            const uint8_t mmask0 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp0));
            const uint8_t mmask1 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp1));

            res_u8[i / 8] = mmask0 + mmask1 * 16;
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX2Impl<float, Op> {
    static inline void within_range(
        const float* const __restrict lower,
        const float* const __restrict upper,
        const float* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        constexpr auto pred_lower = ComparePredicate<float, Range2Compare<Op>::lower>::value;
        constexpr auto pred_upper = ComparePredicate<float, Range2Compare<Op>::upper>::value;

        // todo: aligned reads & writes

        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256 v0l = _mm256_loadu_ps(lower + i);
            const __m256 v0u = _mm256_loadu_ps(upper + i);
            const __m256 v0v = _mm256_loadu_ps(values + i);
            const __m256 cmpl = _mm256_cmp_ps(v0l, v0v, pred_lower);
            const __m256 cmpu = _mm256_cmp_ps(v0v, v0u, pred_upper);
            const __m256 cmp = _mm256_and_ps(cmpl, cmpu);
            const uint8_t mmask = _mm256_movemask_ps(cmp);

            res_u8[i / 8] = mmask;
        }
    }
};

template<RangeType Op>
struct WithinRangeAVX2Impl<double, Op> {
    static inline void within_range(
        const double* const __restrict lower,
        const double* const __restrict upper,
        const double* const __restrict values,
        const size_t size,
        uint8_t* const __restrict res
    ) {
        // the restriction of the API
        assert((size % 8) == 0);

        //
        uint8_t* const __restrict res_u8 = reinterpret_cast<uint8_t*>(res);
        constexpr auto pred_lower = ComparePredicate<double, Range2Compare<Op>::lower>::value;
        constexpr auto pred_upper = ComparePredicate<double, Range2Compare<Op>::upper>::value;

        // todo: aligned reads & writes
        const size_t size8 = (size / 8) * 8;
        for (size_t i = 0; i < size8; i += 8) {
            const __m256d v0l = _mm256_loadu_pd(lower + i);
            const __m256d v1l = _mm256_loadu_pd(lower + i + 4);
            const __m256d v0u = _mm256_loadu_pd(upper + i);
            const __m256d v1u = _mm256_loadu_pd(upper + i + 4);
            const __m256d v0v = _mm256_loadu_pd(values + i);
            const __m256d v1v = _mm256_loadu_pd(values + i + 4);
            const __m256d cmp0l = _mm256_cmp_pd(v0l, v0v, pred_lower);
            const __m256d cmp0u = _mm256_cmp_pd(v0v, v0u, pred_upper);
            const __m256d cmp1l = _mm256_cmp_pd(v1l, v1v, pred_lower);
            const __m256d cmp1u = _mm256_cmp_pd(v1v, v1u, pred_upper);
            const __m256d cmp0 = _mm256_and_pd(cmp0l, cmp0u);
            const __m256d cmp1 = _mm256_and_pd(cmp1l, cmp1u);
            const uint8_t mmask0 = _mm256_movemask_pd(cmp0);
            const uint8_t mmask1 = _mm256_movemask_pd(cmp1);

            res_u8[i / 8] = mmask0 + mmask1 * 16;
        }
    }
};

//
template<typename T, RangeType Op>
void WithinRangeAVX2(const T* const __restrict lower, const T* const __restrict upper, const T* const __restrict values, const size_t size, uint8_t* const __restrict res) {
    WithinRangeAVX2Impl<T, Op>::within_range(lower, upper, values, size, res);
}

#define INSTANTIATE_WITHIN_RANGE_AVX2(TTYPE,OP) \
    template void WithinRangeAVX2<TTYPE, RangeType::OP>( \
        const TTYPE* const __restrict lower, \
        const TTYPE* const __restrict upper, \
        const TTYPE* const __restrict values, \
        const size_t size, \
        uint8_t* const __restrict res \
    );

ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX2, int8_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX2, int16_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX2, int32_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX2, int64_t)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX2, float)
ALL_RANGE_OPS(INSTANTIATE_WITHIN_RANGE_AVX2, double)

#undef INSTANTIATE_WITHIN_RANGE_AVX2

///////////////////////////////////////////////////////////////////////////

//
#undef ALL_COMPARE_OPS
#undef ALL_RANGE_OPS

}
}
}
}
