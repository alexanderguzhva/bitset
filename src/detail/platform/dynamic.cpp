#include "dynamic.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__x86_64__)
#include "x86/instruction_set.h"
#include "x86/avx2.h"
#include "x86/avx512.h"

using namespace milvus::bitset::detail::x86;
#endif

#include "vectorized_ref.h"

namespace milvus {
namespace bitset {
namespace detail {

/////////////////////////////////////////////////////////////////////////////
// op_compare_column

// Define pointers for op_compare
template<typename T, typename U, CompareType Op>
using OpCompareColumnPtr = bool(*)(uint8_t* const __restrict output, const T* const __restrict t, const U* const __restrict u, const size_t size);

#define DECLARE_OP_COMPARE_COLUMN(TTYPE, UTYPE, OP) \
    OpCompareColumnPtr<TTYPE, UTYPE, CompareType::OP> op_compare_column_##TTYPE##_##UTYPE##_##OP = VectorizedRef::template op_compare_column<TTYPE, UTYPE, CompareType::OP>;

DECLARE_OP_COMPARE_COLUMN(float, float, EQ);
DECLARE_OP_COMPARE_COLUMN(float, float, GE);
DECLARE_OP_COMPARE_COLUMN(float, float, GT);
DECLARE_OP_COMPARE_COLUMN(float, float, LE);
DECLARE_OP_COMPARE_COLUMN(float, float, LT);
DECLARE_OP_COMPARE_COLUMN(float, float, NEQ);

#undef DECLARE_OP_COMPARE_COLUMN

// 
template<typename T, typename U, CompareType Op>
bool VectorizedDynamic::op_compare_column(
    uint8_t* const __restrict output, 
    const T* const __restrict t,
    const U* const __restrict u,
    const size_t size
) {
// define the comparator
#define DISPATCH_OP_COMPARE_COLUMN(TTYPE, UTYPE, OP) \
    if constexpr(std::is_same_v<T, TTYPE> && std::is_same_v<U, UTYPE> && Op == CompareType::OP) { \
        return op_compare_column_##TTYPE##_##UTYPE##_##OP(output, t, u, size); \
    }

    // find the appropriate function pointer
    DISPATCH_OP_COMPARE_COLUMN(float, float, EQ)
    DISPATCH_OP_COMPARE_COLUMN(float, float, GE)
    DISPATCH_OP_COMPARE_COLUMN(float, float, GT)
    DISPATCH_OP_COMPARE_COLUMN(float, float, LE)
    DISPATCH_OP_COMPARE_COLUMN(float, float, LT)
    DISPATCH_OP_COMPARE_COLUMN(float, float, NEQ)

#undef DISPATCH_OP_COMPARE_COLUMN

    // no vectorized implementation is available
    return false;
}

// Instantiate template methods.
#define INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN(TTYPE, UTYPE, OP) \
    template bool VectorizedDynamic::op_compare_column<TTYPE, UTYPE, CompareType::OP>( \
        uint8_t* const __restrict output, \
        const TTYPE* const __restrict t, \
        const UTYPE* const __restrict u, \
        const size_t size \
    );

INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN(float, float, EQ);
INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN(float, float, GE);
INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN(float, float, GT);
INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN(float, float, LE);
INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN(float, float, LT);
INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN(float, float, NEQ);

#undef INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN


/////////////////////////////////////////////////////////////////////////////
// op_compare_val
template<typename T, CompareType Op>
using OpCompareValPtr = bool(*)(uint8_t* const __restrict output, const T* const __restrict t, const size_t size, const T value);

#define DECLARE_OP_COMPARE_VAL(TTYPE, OP) \
    OpCompareValPtr<TTYPE, CompareType::OP> op_compare_val_##TTYPE##_##OP = VectorizedRef::template op_compare_val<TTYPE, CompareType::OP>;

DECLARE_OP_COMPARE_VAL(float, EQ);
DECLARE_OP_COMPARE_VAL(float, GE);
DECLARE_OP_COMPARE_VAL(float, GT);
DECLARE_OP_COMPARE_VAL(float, LE);
DECLARE_OP_COMPARE_VAL(float, LT);
DECLARE_OP_COMPARE_VAL(float, NEQ);

#undef DECLARE_OP_COMPARE_VAL

// 
template<typename T, CompareType Op>
bool VectorizedDynamic::op_compare_val(
    uint8_t* const __restrict output, 
    const T* const __restrict t,
    const size_t size,
    const T value
) {
// define the comparator
#define DISPATCH_OP_COMPARE_VAL(TTYPE, OP) \
    if constexpr(std::is_same_v<T, TTYPE> && Op == CompareType::OP) { \
        return op_compare_val_##TTYPE##_##OP(output, t, size, value); \
    }

    // find the appropriate function pointer
    DISPATCH_OP_COMPARE_VAL(float, EQ)
    DISPATCH_OP_COMPARE_VAL(float, GE)
    DISPATCH_OP_COMPARE_VAL(float, GT)
    DISPATCH_OP_COMPARE_VAL(float, LE)
    DISPATCH_OP_COMPARE_VAL(float, LT)
    DISPATCH_OP_COMPARE_VAL(float, NEQ)

#undef DISPATCH_OP_COMPARE_VAL

    // no vectorized implementation is available
    return false;
}

// Instantiate template methods.
#define INSTANTIATE_TEMPLATE_OP_COMPARE_VAL(TTYPE, OP) \
    template bool VectorizedDynamic::op_compare_val<TTYPE, CompareType::OP>( \
        uint8_t* const __restrict output, \
        const TTYPE* const __restrict t, \
        const size_t size, \
        const TTYPE value \
    );

INSTANTIATE_TEMPLATE_OP_COMPARE_VAL(float, EQ);
INSTANTIATE_TEMPLATE_OP_COMPARE_VAL(float, GE);
INSTANTIATE_TEMPLATE_OP_COMPARE_VAL(float, GT);
INSTANTIATE_TEMPLATE_OP_COMPARE_VAL(float, LE);
INSTANTIATE_TEMPLATE_OP_COMPARE_VAL(float, LT);
INSTANTIATE_TEMPLATE_OP_COMPARE_VAL(float, NEQ);

#undef INSTANTIATE_TEMPLATE_OP_COMPARE_VAL

}
}
}

//
static void init_dynamic_hook() {
    using namespace milvus::bitset;
    using namespace milvus::bitset::detail;

#if defined(__x86_64__)
    // AVX512 ?
    if (cpu_support_avx512()) {
#define SET_OP_COMPARE_COLUMN_AVX512(TTYPE, UTYPE, OP) \
    op_compare_column_##TTYPE##_##UTYPE##_##OP = VectorizedAvx512::template op_compare_column<TTYPE, UTYPE, CompareType::OP>;
#define SET_OP_COMPARE_VAL_AVX512(TTYPE, OP) \
    op_compare_val_##TTYPE##_##OP = VectorizedAvx512::template op_compare_val<TTYPE, CompareType::OP>;

        // assign AVX2-related pointers
        SET_OP_COMPARE_COLUMN_AVX512(float, float, EQ);
        SET_OP_COMPARE_COLUMN_AVX512(float, float, GE);
        SET_OP_COMPARE_COLUMN_AVX512(float, float, GT);
        SET_OP_COMPARE_COLUMN_AVX512(float, float, LE);
        SET_OP_COMPARE_COLUMN_AVX512(float, float, LT);
        SET_OP_COMPARE_COLUMN_AVX512(float, float, NEQ);

        SET_OP_COMPARE_VAL_AVX512(float, EQ);
        SET_OP_COMPARE_VAL_AVX512(float, GE);
        SET_OP_COMPARE_VAL_AVX512(float, GT);
        SET_OP_COMPARE_VAL_AVX512(float, LE);
        SET_OP_COMPARE_VAL_AVX512(float, LT);
        SET_OP_COMPARE_VAL_AVX512(float, NEQ);

#undef SET_OP_COMPARE_COLUMN_AVX512
#undef SET_OP_COMPARE_VAL_AVX512

        return;
    }

    // AVX2 ?
    if (cpu_support_avx2()) {
#define SET_OP_COMPARE_COLUMN_AVX2(TTYPE, UTYPE, OP) \
    op_compare_column_##TTYPE##_##UTYPE##_##OP = VectorizedAvx2::template op_compare_column<TTYPE, UTYPE, CompareType::OP>;
#define SET_OP_COMPARE_VAL_AVX2(TTYPE, OP) \
    op_compare_val_##TTYPE##_##OP = VectorizedAvx2::template op_compare_val<TTYPE, CompareType::OP>;

        // assign AVX2-related pointers
        SET_OP_COMPARE_COLUMN_AVX2(float, float, EQ);
        SET_OP_COMPARE_COLUMN_AVX2(float, float, GE);
        SET_OP_COMPARE_COLUMN_AVX2(float, float, GT);
        SET_OP_COMPARE_COLUMN_AVX2(float, float, LE);
        SET_OP_COMPARE_COLUMN_AVX2(float, float, LT);
        SET_OP_COMPARE_COLUMN_AVX2(float, float, NEQ);

        SET_OP_COMPARE_VAL_AVX2(float, EQ);
        SET_OP_COMPARE_VAL_AVX2(float, GE);
        SET_OP_COMPARE_VAL_AVX2(float, GT);
        SET_OP_COMPARE_VAL_AVX2(float, LE);
        SET_OP_COMPARE_VAL_AVX2(float, LT);
        SET_OP_COMPARE_VAL_AVX2(float, NEQ);

#undef SET_OP_COMPARE_COLUMN_AVX2
#undef SET_OP_COMPARE_VAL_AVX2

        return;
    }
#endif
}

//
static int init_dynamic_ = []() {
    init_dynamic_hook();

    return 0;
}();