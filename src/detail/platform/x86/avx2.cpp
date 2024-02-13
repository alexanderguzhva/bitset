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

// a facility to run through all possible operations
#define ALL_OPS(FUNC,...) \
    FUNC(Equal,__VA_ARGS__); \
    FUNC(GreaterEqual,__VA_ARGS__); \
    FUNC(Greater,__VA_ARGS__); \
    FUNC(LessEqual,__VA_ARGS__); \
    FUNC(Less,__VA_ARGS__); \
    FUNC(NotEqual,__VA_ARGS__);

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

//
template <typename T, CompareType Op>
struct CompareValAVX2Impl {};

template<CompareType Op>
struct CompareValAVX2Impl<int8_t, Op> {
    static void Compare(
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
    static void Compare(
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
    static void Compare(
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
    static void Compare(
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

#define DECLARE_VAL_AVX2(NAME, OP) \
    template<typename T> \
    void NAME##ValAVX2( \
        const T* const __restrict src, \
        const size_t size, \
        const T val, \
        void* const __restrict res \
    ) { \
        CompareValAVX2Impl<T, CompareType::OP>::Compare(src, size, val, res); \
    }

DECLARE_VAL_AVX2(Equal, EQ);
DECLARE_VAL_AVX2(GreaterEqual, GE);
DECLARE_VAL_AVX2(Greater, GT);
DECLARE_VAL_AVX2(LessEqual, LE);
DECLARE_VAL_AVX2(Less, LT);
DECLARE_VAL_AVX2(NotEqual, NEQ);

#undef DECLARE_VAL_AVX2

#define INSTANTIATE_VAL_AVX2(NAME, TTYPE) \
    template void NAME##ValAVX2( \
        const TTYPE* const __restrict src, \
        const size_t size, \
        const TTYPE val, \
        void* const __restrict res \
    );

ALL_OPS(INSTANTIATE_VAL_AVX2, int8_t)
ALL_OPS(INSTANTIATE_VAL_AVX2, int16_t)
ALL_OPS(INSTANTIATE_VAL_AVX2, int32_t)
ALL_OPS(INSTANTIATE_VAL_AVX2, int64_t)
ALL_OPS(INSTANTIATE_VAL_AVX2, float)
ALL_OPS(INSTANTIATE_VAL_AVX2, double)

#undef INSTANTIATE_VAL_AVX2

//
template <typename T, CompareType Op>
struct CompareColumnAVX2Impl {};

template <CompareType Op>
struct CompareColumnAVX2Impl<int8_t, Op> {
    static inline void Compare(
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
    static inline void Compare(
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
    static inline void Compare(
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
    static inline void Compare(
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
    static inline void Compare(
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
    static inline void Compare(
        const double* const __restrict left, 
        const double* const __restrict right, 
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

#define DECLARE_COLUMN_AVX2(NAME, OP) \
    template<typename T> \
    void NAME##ColumnAVX2( \
        const T* const __restrict left, \
        const T* const __restrict right, \
        const size_t size, \
        void* const __restrict res \
    ) { \
        CompareColumnAVX2Impl<T, CompareType::OP>::Compare(left, right, size, res); \
    }

DECLARE_COLUMN_AVX2(Equal, EQ);
DECLARE_COLUMN_AVX2(GreaterEqual, GE);
DECLARE_COLUMN_AVX2(Greater, GT);
DECLARE_COLUMN_AVX2(LessEqual, LE);
DECLARE_COLUMN_AVX2(Less, LT);
DECLARE_COLUMN_AVX2(NotEqual, NEQ);

#undef DECLARE_COLUMN_AVX2

//
#define INSTANTIATE_COLUMN_AVX2(NAME, TTYPE) \
    template void NAME##ColumnAVX2( \
        const TTYPE* const __restrict left, \
        const TTYPE* const __restrict right, \
        const size_t size, \
        void* const __restrict res \
    );

ALL_OPS(INSTANTIATE_COLUMN_AVX2, int8_t)
ALL_OPS(INSTANTIATE_COLUMN_AVX2, int16_t)
ALL_OPS(INSTANTIATE_COLUMN_AVX2, int32_t)
ALL_OPS(INSTANTIATE_COLUMN_AVX2, int64_t)
ALL_OPS(INSTANTIATE_COLUMN_AVX2, float)
ALL_OPS(INSTANTIATE_COLUMN_AVX2, double)

#undef INSTANTIATE_COLUMN_AVX2

//
#undef ALL_OPS

}
}
}
}
