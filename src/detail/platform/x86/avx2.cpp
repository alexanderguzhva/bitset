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

template <typename T, CompareType Op>
struct CompareColumnAVX2Impl {
    static void
    Compare(const T* const __restrict src, const size_t size, const T val, void* const __restrict res) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>,
                      "T must be integral or float/double type");
    }
};

template <CompareType Op>
struct CompareColumnAVX2Impl<float, Op> {
    static void Compare(
        const float* const __restrict left, 
        const float* const __restrict right, 
        const size_t size,
        void* const __restrict res
    ) {
        // the restriction of the API
        assert((uintptr_t(res) % 8) == 0);
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