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


// a facility to run through all possible operations
#define ALL_OPS(FUNC,...) \
    FUNC(__VA_ARGS__,EQ); \
    FUNC(__VA_ARGS__,GE); \
    FUNC(__VA_ARGS__,GT); \
    FUNC(__VA_ARGS__,LE); \
    FUNC(__VA_ARGS__,LT); \
    FUNC(__VA_ARGS__,NEQ);


//
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

ALL_OPS(DECLARE_OP_COMPARE_COLUMN, int8_t, int8_t)
ALL_OPS(DECLARE_OP_COMPARE_COLUMN, int16_t, int16_t)
ALL_OPS(DECLARE_OP_COMPARE_COLUMN, int32_t, int32_t)
ALL_OPS(DECLARE_OP_COMPARE_COLUMN, int64_t, int64_t)
ALL_OPS(DECLARE_OP_COMPARE_COLUMN, float, float)
ALL_OPS(DECLARE_OP_COMPARE_COLUMN, double, double)

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
    ALL_OPS(DISPATCH_OP_COMPARE_COLUMN, float, float)

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

ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, int8_t, int8_t)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, int16_t, int16_t)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, int32_t, int32_t)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, int64_t, int64_t)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, float, float)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, double, double)

#undef INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN


/////////////////////////////////////////////////////////////////////////////
// op_compare_val
template<typename T, CompareType Op>
using OpCompareValPtr = bool(*)(uint8_t* const __restrict output, const T* const __restrict t, const size_t size, const T value);

#define DECLARE_OP_COMPARE_VAL(TTYPE, OP) \
    OpCompareValPtr<TTYPE, CompareType::OP> op_compare_val_##TTYPE##_##OP = VectorizedRef::template op_compare_val<TTYPE, CompareType::OP>;

ALL_OPS(DECLARE_OP_COMPARE_VAL, int8_t)
ALL_OPS(DECLARE_OP_COMPARE_VAL, int16_t)
ALL_OPS(DECLARE_OP_COMPARE_VAL, int32_t)
ALL_OPS(DECLARE_OP_COMPARE_VAL, int64_t)
ALL_OPS(DECLARE_OP_COMPARE_VAL, float)
ALL_OPS(DECLARE_OP_COMPARE_VAL, double)

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
    ALL_OPS(DISPATCH_OP_COMPARE_VAL, int8_t)
    ALL_OPS(DISPATCH_OP_COMPARE_VAL, int16_t)
    ALL_OPS(DISPATCH_OP_COMPARE_VAL, int32_t)
    ALL_OPS(DISPATCH_OP_COMPARE_VAL, int64_t)
    ALL_OPS(DISPATCH_OP_COMPARE_VAL, float)
    ALL_OPS(DISPATCH_OP_COMPARE_VAL, double)

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

ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, int8_t)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, int16_t)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, int32_t)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, int64_t)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, float)
ALL_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, double)

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
        ALL_OPS(SET_OP_COMPARE_COLUMN_AVX512, float, float)

        ALL_OPS(SET_OP_COMPARE_VAL_AVX512, int8_t)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX512, int16_t)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX512, int32_t)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX512, int64_t)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX512, float)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX512, double)

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
        ALL_OPS(SET_OP_COMPARE_COLUMN_AVX2, float, float)

        ALL_OPS(SET_OP_COMPARE_VAL_AVX2, int8_t)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX2, int16_t)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX2, int32_t)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX2, int64_t)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX2, float)
        ALL_OPS(SET_OP_COMPARE_VAL_AVX2, double)

#undef SET_OP_COMPARE_COLUMN_AVX2
#undef SET_OP_COMPARE_VAL_AVX2

        return;
    }

#endif
}

// no longer needed
#undef ALL_OPS

//
static int init_dynamic_ = []() {
    init_dynamic_hook();

    return 0;
}();
