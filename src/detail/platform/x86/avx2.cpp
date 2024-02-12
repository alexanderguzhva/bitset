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
template <typename T, CompareType Op>
struct CompareValAVX2Impl {
    static void
    Compare(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                      "T must be integral or float/double type");
    }
};

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
            for (size_t i = size32; i < size; i += 8) {
                uint8_t mask = 0;
                for (size_t j = 0; j < 8; j++) {
                    bool bcmp = CompareOperator<Op>::compare(src[i + j], val);
                    mask |= (bcmp ? 1 : 0) << j; 
                }
                res_u8[i / 8] = mask;
            }
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

INSTANTIATE_VAL_AVX2(Equal, float);
INSTANTIATE_VAL_AVX2(GreaterEqual, float);
INSTANTIATE_VAL_AVX2(Greater, float);
INSTANTIATE_VAL_AVX2(LessEqual, float);
INSTANTIATE_VAL_AVX2(Less, float);
INSTANTIATE_VAL_AVX2(NotEqual, float);

INSTANTIATE_VAL_AVX2(Equal, int8_t);
INSTANTIATE_VAL_AVX2(GreaterEqual, int8_t);
INSTANTIATE_VAL_AVX2(Greater, int8_t);
INSTANTIATE_VAL_AVX2(LessEqual, int8_t);
INSTANTIATE_VAL_AVX2(Less, int8_t);
INSTANTIATE_VAL_AVX2(NotEqual, int8_t);

#undef INSTANTIATE_VAL_AVX2

//
template <typename T, CompareType Op>
struct CompareColumnAVX2Impl {
    static void
    Compare(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                      "T must be integral or float/double type");
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
void EqualColumnAVX2(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX2Impl<T, CompareType::EQ>::Compare(left, right, size, res);   
}

template
void EqualColumnAVX2(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void GreaterEqualColumnAVX2(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX2Impl<T, CompareType::GE>::Compare(left, right, size, res);   
}

template
void GreaterEqualColumnAVX2(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void GreaterColumnAVX2(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX2Impl<T, CompareType::GT>::Compare(left, right, size, res);   
}

template
void GreaterColumnAVX2(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void LessEqualColumnAVX2(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX2Impl<T, CompareType::LE>::Compare(left, right, size, res);   
}

template
void LessEqualColumnAVX2(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void LessColumnAVX2(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX2Impl<T, CompareType::LT>::Compare(left, right, size, res);   
}

template
void LessColumnAVX2(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

//
template <typename T>
void NotEqualColumnAVX2(
    const T* const __restrict left, 
    const T* const __restrict right, 
    const size_t size, 
    void* const __restrict res
) {
    CompareColumnAVX2Impl<T, CompareType::NEQ>::Compare(left, right, size, res);   
}

template
void NotEqualColumnAVX2(
    const float* const __restrict left, 
    const float* const __restrict right, 
    const size_t size, 
    void* const __restrict res
);

}
}
}
}
