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

#if defined(__aarch64__)
#include "arm/neon.h"

using namespace milvus::bitset::detail::arm;
#endif

#include "vectorized_ref.h"


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

ALL_COMPARE_OPS(DECLARE_OP_COMPARE_COLUMN, int8_t, int8_t)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_COLUMN, int16_t, int16_t)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_COLUMN, int32_t, int32_t)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_COLUMN, int64_t, int64_t)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_COLUMN, float, float)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_COLUMN, double, double)

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
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_COLUMN, int8_t, int8_t)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_COLUMN, int16_t, int16_t)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_COLUMN, int32_t, int32_t)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_COLUMN, int64_t, int64_t)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_COLUMN, float, float)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_COLUMN, double, double)

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

ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, int8_t, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, int16_t, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, int32_t, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, int64_t, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, float, float)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN, double, double)

#undef INSTANTIATE_TEMPLATE_OP_COMPARE_COLUMN


/////////////////////////////////////////////////////////////////////////////
// op_compare_val
template<typename T, CompareType Op>
using OpCompareValPtr = bool(*)(uint8_t* const __restrict output, const T* const __restrict t, const size_t size, const T value);

#define DECLARE_OP_COMPARE_VAL(TTYPE, OP) \
    OpCompareValPtr<TTYPE, CompareType::OP> op_compare_val_##TTYPE##_##OP = VectorizedRef::template op_compare_val<TTYPE, CompareType::OP>;

ALL_COMPARE_OPS(DECLARE_OP_COMPARE_VAL, int8_t)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_VAL, int16_t)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_VAL, int32_t)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_VAL, int64_t)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_VAL, float)
ALL_COMPARE_OPS(DECLARE_OP_COMPARE_VAL, double)

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
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_VAL, int8_t)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_VAL, int16_t)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_VAL, int32_t)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_VAL, int64_t)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_VAL, float)
    ALL_COMPARE_OPS(DISPATCH_OP_COMPARE_VAL, double)

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

ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, int8_t)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, int16_t)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, int32_t)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, int64_t)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, float)
ALL_COMPARE_OPS(INSTANTIATE_TEMPLATE_OP_COMPARE_VAL, double)

#undef INSTANTIATE_TEMPLATE_OP_COMPARE_VAL


/////////////////////////////////////////////////////////////////////////////
// op_within_range
template<typename T, RangeType Op>
using OpWithinRangePtr = bool(*)(uint8_t* const __restrict output, const T* const __restrict lower, const T* const __restrict upper, const T* const __restrict values, const size_t size);

#define DECLARE_OP_WITHIN_RANGE(TTYPE, OP) \
    OpWithinRangePtr<TTYPE, RangeType::OP> op_within_range_##TTYPE##_##OP = VectorizedRef::template op_within_range<TTYPE, RangeType::OP>;

ALL_RANGE_OPS(DECLARE_OP_WITHIN_RANGE, int8_t)
ALL_RANGE_OPS(DECLARE_OP_WITHIN_RANGE, int16_t)
ALL_RANGE_OPS(DECLARE_OP_WITHIN_RANGE, int32_t)
ALL_RANGE_OPS(DECLARE_OP_WITHIN_RANGE, int64_t)
ALL_RANGE_OPS(DECLARE_OP_WITHIN_RANGE, float)
ALL_RANGE_OPS(DECLARE_OP_WITHIN_RANGE, double)

#undef DECLARE_OP_WITHIN_RANGE

// 
template<typename T, RangeType Op>
bool VectorizedDynamic::op_within_range(
    uint8_t* const __restrict output, 
    const T* const __restrict lower,
    const T* const __restrict upper,
    const T* const __restrict values,
    const size_t size
) {
// define the comparator
#define DISPATCH_OP_WITHIN_RANGE(TTYPE, OP) \
    if constexpr(std::is_same_v<T, TTYPE> && Op == RangeType::OP) { \
        return op_within_range_##TTYPE##_##OP(output, lower, upper, values, size); \
    }

    // find the appropriate function pointer
    ALL_RANGE_OPS(DISPATCH_OP_WITHIN_RANGE, int8_t)
    ALL_RANGE_OPS(DISPATCH_OP_WITHIN_RANGE, int16_t)
    ALL_RANGE_OPS(DISPATCH_OP_WITHIN_RANGE, int32_t)
    ALL_RANGE_OPS(DISPATCH_OP_WITHIN_RANGE, int64_t)
    ALL_RANGE_OPS(DISPATCH_OP_WITHIN_RANGE, float)
    ALL_RANGE_OPS(DISPATCH_OP_WITHIN_RANGE, double)

#undef DISPATCH_OP_WITHIN_RANGE

    // no vectorized implementation is available
    return false;
}

// Instantiate template methods.
#define INSTANTIATE_TEMPLATE_OP_WITHIN_RANGE(TTYPE, OP) \
    template bool VectorizedDynamic::op_within_range<TTYPE, RangeType::OP>( \
        uint8_t* const __restrict output, \
        const TTYPE* const __restrict lower, \
        const TTYPE* const __restrict upper, \
        const TTYPE* const __restrict values, \
        const size_t size \
    );

ALL_RANGE_OPS(INSTANTIATE_TEMPLATE_OP_WITHIN_RANGE, int8_t)
ALL_RANGE_OPS(INSTANTIATE_TEMPLATE_OP_WITHIN_RANGE, int16_t)
ALL_RANGE_OPS(INSTANTIATE_TEMPLATE_OP_WITHIN_RANGE, int32_t)
ALL_RANGE_OPS(INSTANTIATE_TEMPLATE_OP_WITHIN_RANGE, int64_t)
ALL_RANGE_OPS(INSTANTIATE_TEMPLATE_OP_WITHIN_RANGE, float)
ALL_RANGE_OPS(INSTANTIATE_TEMPLATE_OP_WITHIN_RANGE, double)

#undef INSTANTIATE_TEMPLATE_OP_WITHIN_RANGE


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
#define SET_OP_WITHIN_RANGE_AVX512(TTYPE, OP) \
    op_within_range_##TTYPE##_##OP = VectorizedAvx512::template op_within_range<TTYPE, RangeType::OP>;

        // assign AVX512-related pointers
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX512, int8_t, int8_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX512, int16_t, int16_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX512, int32_t, int32_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX512, int64_t, int64_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX512, float, float)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX512, double, double)

        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX512, int8_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX512, int16_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX512, int32_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX512, int64_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX512, float)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX512, double)

        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX512, int8_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX512, int16_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX512, int32_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX512, int64_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX512, float)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX512, double)

#undef SET_OP_COMPARE_COLUMN_AVX512
#undef SET_OP_COMPARE_VAL_AVX512
#undef SET_OP_WITHIN_RANGE_AVX512

        return;
    }

    // AVX2 ?
    if (cpu_support_avx2()) {
#define SET_OP_COMPARE_COLUMN_AVX2(TTYPE, UTYPE, OP) \
    op_compare_column_##TTYPE##_##UTYPE##_##OP = VectorizedAvx2::template op_compare_column<TTYPE, UTYPE, CompareType::OP>;
#define SET_OP_COMPARE_VAL_AVX2(TTYPE, OP) \
    op_compare_val_##TTYPE##_##OP = VectorizedAvx2::template op_compare_val<TTYPE, CompareType::OP>;
#define SET_OP_WITHIN_RANGE_AVX2(TTYPE, OP) \
    op_within_range_##TTYPE##_##OP = VectorizedAvx2::template op_within_range<TTYPE, RangeType::OP>;

        // assign AVX2-related pointers
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX2, int8_t, int8_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX2, int16_t, int16_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX2, int32_t, int32_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX2, int64_t, int64_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX2, float, float)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_AVX2, double, double)

        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX2, int8_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX2, int16_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX2, int32_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX2, int64_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX2, float)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_AVX2, double)

        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX2, int8_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX2, int16_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX2, int32_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX2, int64_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX2, float)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_AVX2, double)

#undef SET_OP_COMPARE_COLUMN_AVX2
#undef SET_OP_COMPARE_VAL_AVX2
#undef SET_OP_WITHIN_RANGE_AVX2

        return;
    }
#endif

#if defined(__aarch64__)
    // neon ?
    {
#define SET_OP_COMPARE_COLUMN_NEON(TTYPE, UTYPE, OP) \
    op_compare_column_##TTYPE##_##UTYPE##_##OP = VectorizedNeon::template op_compare_column<TTYPE, UTYPE, CompareType::OP>;
#define SET_OP_COMPARE_VAL_NEON(TTYPE, OP) \
    op_compare_val_##TTYPE##_##OP = VectorizedNeon::template op_compare_val<TTYPE, CompareType::OP>;
#define SET_OP_WITHIN_RANGE_NEON(TTYPE, OP) \
    op_within_range_##TTYPE##_##OP = VectorizedNeon::template op_within_range<TTYPE, RangeType::OP>;

        // assign AVX2-related pointers
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_NEON, int8_t, int8_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_NEON, int16_t, int16_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_NEON, int32_t, int32_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_NEON, int64_t, int64_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_NEON, float, float)
        ALL_COMPARE_OPS(SET_OP_COMPARE_COLUMN_NEON, double, double)

        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_NEON, int8_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_NEON, int16_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_NEON, int32_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_NEON, int64_t)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_NEON, float)
        ALL_COMPARE_OPS(SET_OP_COMPARE_VAL_NEON, double)

        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_NEON, int8_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_NEON, int16_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_NEON, int32_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_NEON, int64_t)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_NEON, float)
        ALL_RANGE_OPS(SET_OP_WITHIN_RANGE_NEON, double)

#undef SET_OP_COMPARE_COLUMN_NEON
#undef SET_OP_COMPARE_VAL_NEON
#undef SET_OP_WITHIN_RANGE_NEON

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
