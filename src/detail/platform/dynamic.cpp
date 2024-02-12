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

// Define pointers for op_compare
template<typename T, typename U, CompareType Op>
using OpComparePtr = bool(*)(uint8_t* const __restrict data, const T* const __restrict t, const U* const __restrict u, const size_t size);

#define DECLARE_OP_COMPARE(TTYPE, UTYPE, OP) \
    OpComparePtr<TTYPE, UTYPE, CompareType::OP> op_compare_##TTYPE##_##UTYPE##_##OP = VectorizedRef::template op_compare<TTYPE, UTYPE, CompareType::OP>;

DECLARE_OP_COMPARE(float, float, EQ);
DECLARE_OP_COMPARE(float, float, GE);
DECLARE_OP_COMPARE(float, float, GT);
DECLARE_OP_COMPARE(float, float, LE);
DECLARE_OP_COMPARE(float, float, LT);
DECLARE_OP_COMPARE(float, float, NEQ);

#undef DECLARE_OP_COMPARE

// size is in bytes
template<typename T, typename U, CompareType Op>
bool VectorizedDynamic::op_compare(
    uint8_t* const __restrict data, 
    const T* const __restrict t,
    const U* const __restrict u,
    const size_t size
) {
// define the comparator
#define DISPATCH_OP_COMPARE(TTYPE, UTYPE, OP) \
    if constexpr(std::is_same_v<T, TTYPE> && std::is_same_v<U, UTYPE> && Op == CompareType::OP) { \
        return op_compare_##TTYPE##_##UTYPE##_##OP(data, t, u, size); \
    }

    // find the appropriate function pointer
    DISPATCH_OP_COMPARE(float, float, EQ)
    DISPATCH_OP_COMPARE(float, float, GE)
    DISPATCH_OP_COMPARE(float, float, GT)
    DISPATCH_OP_COMPARE(float, float, LE)
    DISPATCH_OP_COMPARE(float, float, LT)
    DISPATCH_OP_COMPARE(float, float, NEQ)

#undef DISPATCH_OP_COMPARE

    return false;
}

// Instantiate template methods.
#define INSTANTIATE_TEMPLATE_OP_COMPARE(TTYPE, UTYPE, OP) \
    template bool VectorizedDynamic::op_compare<TTYPE, UTYPE, CompareType::OP>( \
        uint8_t* const __restrict data, \
        const TTYPE* const __restrict t, \
        const UTYPE* const __restrict u, \
        const size_t size \
    );

INSTANTIATE_TEMPLATE_OP_COMPARE(float, float, EQ);
INSTANTIATE_TEMPLATE_OP_COMPARE(float, float, GE);
INSTANTIATE_TEMPLATE_OP_COMPARE(float, float, GT);
INSTANTIATE_TEMPLATE_OP_COMPARE(float, float, LE);
INSTANTIATE_TEMPLATE_OP_COMPARE(float, float, LT);
INSTANTIATE_TEMPLATE_OP_COMPARE(float, float, NEQ);

#undef INSTANTIATE_TEMPLATE_OP_COMPARE

}
}
}

#if defined(__x86_64__)
bool
cpu_support_avx512() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.AVX512F() && instruction_set_inst.AVX512DQ() &&
            instruction_set_inst.AVX512BW() && instruction_set_inst.AVX512VL());
}

bool
cpu_support_avx2() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.AVX2());
}

bool
cpu_support_sse4_2() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.SSE42());
}

bool
cpu_support_sse2() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.SSE2());
}
#endif

//
static void init_dynamic_hook() {
    using namespace milvus::bitset;
    using namespace milvus::bitset::detail;

#if defined(__x86_64__)
    // AVX512 ?
    if (cpu_support_avx512()) {
// define a pointer assigner
#define SET_OP_COMPARE_AVX512(TTYPE, UTYPE, OP) \
    op_compare_##TTYPE##_##UTYPE##_##OP = VectorizedAvx512::template op_compare<TTYPE, UTYPE, CompareType::OP>;

        // assign AVX2-related pointers
        SET_OP_COMPARE_AVX512(float, float, EQ);
        SET_OP_COMPARE_AVX512(float, float, GE);
        SET_OP_COMPARE_AVX512(float, float, GT);
        SET_OP_COMPARE_AVX512(float, float, LE);
        SET_OP_COMPARE_AVX512(float, float, LT);
        SET_OP_COMPARE_AVX512(float, float, NEQ);

#undef SET_OP_COMPARE_AVX512

        return;

    }

    // AVX2 ?
    if (cpu_support_avx2()) {
// define a pointer assigner
#define SET_OP_COMPARE_AVX2(TTYPE, UTYPE, OP) \
    op_compare_##TTYPE##_##UTYPE##_##OP = VectorizedAvx2::template op_compare<TTYPE, UTYPE, CompareType::OP>;

        // assign AVX2-related pointers
        SET_OP_COMPARE_AVX2(float, float, EQ);
        SET_OP_COMPARE_AVX2(float, float, GE);
        SET_OP_COMPARE_AVX2(float, float, GT);
        SET_OP_COMPARE_AVX2(float, float, LE);
        SET_OP_COMPARE_AVX2(float, float, LT);
        SET_OP_COMPARE_AVX2(float, float, NEQ);

#undef SET_OP_COMPARE_AVX2

        return;
    }
#endif
}

//
static int init_dynamic_ = []() {
    init_dynamic_hook();

    return 0;
}();