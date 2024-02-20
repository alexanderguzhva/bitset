#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <bitset.h>
#include <detail/bit_wise.h>
#include <detail/element_wise.h>
#include <detail/element_vectorized.h>
#include <detail/platform/dynamic.h>

#if defined(__x86_64__)
#include <detail/platform/x86/avx2.h>
#include <detail/platform/x86/avx512.h>
#include <detail/platform/x86/instruction_set.h>
#endif

#if defined(__aarch64__)
#include <detail/platform/arm/neon.h>

#ifdef __ARM_FEATURE_SVE
#include <detail/platform/arm/sve.h>
#endif

#endif

#include "utils.h"

using namespace milvus::bitset;

//
namespace ref_u64_u8 {

using policy_type = milvus::bitset::detail::CustomBitsetPolicy<uint64_t>;
using container_type = std::vector<uint8_t>;
using bitset_type = milvus::bitset::CustomBitsetOwning<policy_type, container_type, false>;
using bitset_view = milvus::bitset::CustomBitsetNonOwning<policy_type, false>;

}

//
namespace element_u64_u8 {

using policy_type = milvus::bitset::detail::CustomBitsetPolicy2<uint64_t>;
using container_type = std::vector<uint8_t>;
using bitset_type = milvus::bitset::CustomBitsetOwning<policy_type, container_type, false>;
using bitset_view = milvus::bitset::CustomBitsetNonOwning<policy_type, false>;

}

//
#if defined(__x86_64__)
namespace avx2_u64_u8 {

using vectorized_type = milvus::bitset::detail::x86::VectorizedAvx2;
using policy_type = milvus::bitset::detail::CustomBitsetVectorizedPolicy<uint64_t, vectorized_type>;
using container_type = std::vector<uint8_t>;
using bitset_type = milvus::bitset::CustomBitsetOwning<policy_type, container_type, false>;
using bitset_view = milvus::bitset::CustomBitsetNonOwning<policy_type, false>;

}
#endif

//
#if defined(__x86_64__)
namespace avx512_u64_u8 {

using vectorized_type = milvus::bitset::detail::x86::VectorizedAvx512;
using policy_type = milvus::bitset::detail::CustomBitsetVectorizedPolicy<uint64_t, vectorized_type>;
using container_type = std::vector<uint8_t>;
using bitset_type = milvus::bitset::CustomBitsetOwning<policy_type, container_type, false>;
using bitset_view = milvus::bitset::CustomBitsetNonOwning<policy_type, false>;

}
#endif

//
#if defined(__aarch64__)
namespace neon_u64_u8 {

using vectorized_type = milvus::bitset::detail::arm::VectorizedNeon;
using policy_type = milvus::bitset::detail::CustomBitsetVectorizedPolicy<uint64_t, vectorized_type>;
using container_type = std::vector<uint8_t>;
using bitset_type = milvus::bitset::CustomBitsetOwning<policy_type, container_type, false>;
using bitset_view = milvus::bitset::CustomBitsetNonOwning<policy_type, false>;

}

#ifdef __ARM_FEATURE_SVE
namespace sve_u64_u8 {

using vectorized_type = milvus::bitset::detail::arm::VectorizedSve;
using policy_type = milvus::bitset::detail::CustomBitsetVectorizedPolicy<uint64_t, vectorized_type>;
using container_type = std::vector<uint8_t>;
using bitset_type = milvus::bitset::CustomBitsetOwning<policy_type, container_type, false>;
using bitset_view = milvus::bitset::CustomBitsetNonOwning<policy_type, false>;

}
#endif

#endif

//
namespace dynamic_u64_u8 {

using vectorized_type = milvus::bitset::detail::VectorizedDynamic;
using policy_type = milvus::bitset::detail::CustomBitsetVectorizedPolicy<uint64_t, vectorized_type>;
using container_type = std::vector<uint8_t>;
using bitset_type = milvus::bitset::CustomBitsetOwning<policy_type, container_type, false>;
using bitset_view = milvus::bitset::CustomBitsetNonOwning<policy_type, false>;

}

//////////////////////////////////////////////////////////////////////////////////////////

//
static constexpr bool print_log = false;
static constexpr bool print_timing = false;

//////////////////////////////////////////////////////////////////////////////////////////

//
template<typename T>
void FillRandom(
    std::vector<T>& t, 
    std::default_random_engine& rng,
    const size_t max_v
) {
    std::uniform_int_distribution<uint8_t> tt(0, max_v);
    for (size_t i = 0; i < t.size(); i++) {
        t[i] = tt(rng); 
    }
}

template<typename BitsetT>
void FillRandom(
    BitsetT& bitset,
    std::default_random_engine& rng
) {
    std::uniform_int_distribution<uint8_t> tt(0, 1);
    for (size_t i = 0; i < bitset.size(); i++) {
        bitset[i] = (tt(rng) == 0); 
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

//
template<typename BitsetT>
void TestFindImpl(BitsetT& bitset, const size_t max_v) {
    const size_t n = bitset.size();

    std::default_random_engine rng(123);
    std::uniform_int_distribution<int8_t> u(0, max_v);

    std::vector<size_t> one_pos;
    for (size_t i = 0; i < n; i++) {
        bool enabled = (u(rng) == 0);
        if (enabled) {
            one_pos.push_back(i);
            bitset[i] = true;
        }
    }

    StopWatch sw;

    auto bit_idx = bitset.find_first();
    if (!bit_idx.has_value()) {
        ASSERT_EQ(one_pos.size(), 0);
        return;
    }

    for (size_t i = 0; i < one_pos.size(); i++) {
        ASSERT_TRUE(bit_idx.has_value()) << n << ", " << max_v;
        ASSERT_EQ(bit_idx.value(), one_pos[i]) << n << ", " << max_v;
        bit_idx = bitset.find_next(bit_idx.value());
    }

    ASSERT_FALSE(bit_idx.has_value()) << n << ", " << max_v << ", " << bit_idx.value();

    if (print_timing) {
        printf("elapsed %f\n", sw.elapsed());
    }
}

template<typename BitsetT>
void TestFindImpl() {
    for (const size_t n : {0, 1, 10, 100, 1000, 10000}) {
        for (const size_t pr : {1, 100}) {
            BitsetT bitset(n);
            bitset.reset();

            if (print_log) {
                printf("Testing bitset, n=%zd, pr=%zd\n", n, pr);
            }
            
            TestFindImpl(bitset, pr);

            for (const size_t offset : {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 35, 45, 55, 63, 127, 703}) {
                if (offset >= n) {
                    continue;
                }

                bitset.reset();
                auto view = bitset.view(offset);

                if (print_log) {
                    printf("Testing bitset view, n=%zd, offset=%zd, pr=%zd\n", n, offset, pr);
                }
                
                TestFindImpl(view, pr);
            }
        }
    }
}

//
TEST(FindRef, f) {
    TestFindImpl<ref_u64_u8::bitset_type>();
}

//
TEST(FindElement, f) {
    TestFindImpl<element_u64_u8::bitset_type>();
}

// //
// TEST(FindVectorizedAvx2, f) {
//     TestFindImpl<avx2_u64_u8::bitset_type>();
// }


//////////////////////////////////////////////////////////////////////////////////////////

//
template<typename BitsetT, typename T, typename U>
void TestInplaceCompareColumnImpl(
    BitsetT& bitset, CompareOpType op
) {
    const size_t n = bitset.size();
    constexpr size_t max_v = 2;

    std::vector<T> t(n, 0);
    std::vector<U> u(n, 0);

    std::default_random_engine rng(123);
    FillRandom(t, rng, max_v);
    FillRandom(u, rng, max_v);

    StopWatch sw;
    bitset.inplace_compare_column(t.data(), u.data(), n, op);
    
    if (print_timing) {
        printf("elapsed %f\n", sw.elapsed());
    }

    for (size_t i = 0; i < n; i++) {
        if (op == CompareOpType::EQ) {
            ASSERT_EQ(t[i] == u[i], bitset[i]) << i;
        } else if (op == CompareOpType::GE) {
            ASSERT_EQ(t[i] >= u[i], bitset[i]) << i;
        } else if (op == CompareOpType::GT) {
            ASSERT_EQ(t[i] > u[i], bitset[i]) << i;
        } else if (op == CompareOpType::LE) {
            ASSERT_EQ(t[i] <= u[i], bitset[i]) << i;            
        } else if (op == CompareOpType::LT) {
            ASSERT_EQ(t[i] < u[i], bitset[i]) << i;            
        } else if (op == CompareOpType::NEQ) {
            ASSERT_EQ(t[i] != u[i], bitset[i]) << i;            
        } else {
            ASSERT_TRUE(false) << "Not implemented";
        }
    }
}

template<typename BitsetT, typename T, typename U>
void TestInplaceCompareColumnImpl() {
    for (const size_t n : {0, 1, 10, 100, 1000, 10000}) {
        for (const auto op : {CompareOpType::EQ, CompareOpType::GE, CompareOpType::GT, CompareOpType::LE, CompareOpType::LT, CompareOpType::NEQ}) {
            BitsetT bitset(n);
            bitset.reset();

            if (print_log) {
                printf("Testing bitset, n=%zd, op=%zd\n", n, (size_t)op);
            }
            
            TestInplaceCompareColumnImpl<BitsetT, T, U>(bitset, op);

            for (const size_t offset : {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 35, 45, 55, 63, 127, 703}) {
                if (offset >= n) {
                    continue;
                }

                bitset.reset();
                auto view = bitset.view(offset);

                if (print_log) {
                    printf("Testing bitset view, n=%zd, offset=%zd, op=%zd\n", n, offset, (size_t)op);
                }
                
                TestInplaceCompareColumnImpl<decltype(view), T, U>(view, op);
            }
        }
    }
}

//
template<typename T>
class InplaceCompareColumnSuite : public ::testing::Test {};

TYPED_TEST_SUITE_P(InplaceCompareColumnSuite);

TYPED_TEST_P(InplaceCompareColumnSuite, BitWise) {
    TestInplaceCompareColumnImpl<
        ref_u64_u8::bitset_type, 
        std::tuple_element_t<0, TypeParam>,
        std::tuple_element_t<1, TypeParam>>();
}

TYPED_TEST_P(InplaceCompareColumnSuite, ElementWise) {
    TestInplaceCompareColumnImpl<
        element_u64_u8::bitset_type, 
        std::tuple_element_t<0, TypeParam>,
        std::tuple_element_t<1, TypeParam>>();
}

TYPED_TEST_P(InplaceCompareColumnSuite, Avx2) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx2()) {
        TestInplaceCompareColumnImpl<
            avx2_u64_u8::bitset_type, 
            std::tuple_element_t<0, TypeParam>,
            std::tuple_element_t<1, TypeParam>>();
    }
#endif
}

TYPED_TEST_P(InplaceCompareColumnSuite, Avx512) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx512()) {
        TestInplaceCompareColumnImpl<
            avx512_u64_u8::bitset_type, 
            std::tuple_element_t<0, TypeParam>,
            std::tuple_element_t<1, TypeParam>>();
    }
#endif
}

TYPED_TEST_P(InplaceCompareColumnSuite, Neon) {
#if defined(__aarch64__)
    using namespace milvus::bitset::detail::arm;

    TestInplaceCompareColumnImpl<
        neon_u64_u8::bitset_type, 
        std::tuple_element_t<0, TypeParam>,
        std::tuple_element_t<1, TypeParam>>();
#endif
}

TYPED_TEST_P(InplaceCompareColumnSuite, Sve) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
    using namespace milvus::bitset::detail::arm;

    TestInplaceCompareColumnImpl<
        sve_u64_u8::bitset_type, 
        std::tuple_element_t<0, TypeParam>,
        std::tuple_element_t<1, TypeParam>>();
#endif
}

TYPED_TEST_P(InplaceCompareColumnSuite, Dynamic) {
    TestInplaceCompareColumnImpl<
        dynamic_u64_u8::bitset_type, 
        std::tuple_element_t<0, TypeParam>,
        std::tuple_element_t<1, TypeParam>>();
}


using InplaceCompareColumnTtypes = ::testing::Types<
    std::tuple<int8_t, int8_t>,
    std::tuple<int16_t, int16_t>,
    std::tuple<int32_t, int32_t>, 
    std::tuple<int64_t, int64_t>, 
    std::tuple<float, float>,
    std::tuple<double, double>
>;

REGISTER_TYPED_TEST_SUITE_P(InplaceCompareColumnSuite, BitWise, ElementWise, Avx2, Avx512, Neon, Sve, Dynamic);

INSTANTIATE_TYPED_TEST_SUITE_P(InplaceCompareColumnTest, InplaceCompareColumnSuite, InplaceCompareColumnTtypes);

//////////////////////////////////////////////////////////////////////////////////////////

//
template<typename BitsetT, typename T>
void TestInplaceCompareValImpl(
    BitsetT& bitset, CompareOpType op
) {
    const size_t n = bitset.size();
    constexpr size_t max_v = 3;
    constexpr T value = 1;

    std::vector<T> t(n, 0);

    std::default_random_engine rng(123);
    FillRandom(t, rng, max_v);

    StopWatch sw;
    bitset.inplace_compare_val(t.data(), n, value, op);
    
    if (print_timing) {
        printf("elapsed %f\n", sw.elapsed());
    }

    for (size_t i = 0; i < n; i++) {
        if (op == CompareOpType::EQ) {
            ASSERT_EQ(t[i] == value, bitset[i]) << i;
        } else if (op == CompareOpType::GE) {
            ASSERT_EQ(t[i] >= value, bitset[i]) << i;
        } else if (op == CompareOpType::GT) {
            ASSERT_EQ(t[i] > value, bitset[i]) << i;
        } else if (op == CompareOpType::LE) {
            ASSERT_EQ(t[i] <= value, bitset[i]) << i;            
        } else if (op == CompareOpType::LT) {
            ASSERT_EQ(t[i] < value, bitset[i]) << i;            
        } else if (op == CompareOpType::NEQ) {
            ASSERT_EQ(t[i] != value, bitset[i]) << i;            
        } else {
            ASSERT_TRUE(false) << "Not implemented";
        }
    }
}

template<typename BitsetT, typename T>
void TestInplaceCompareValImpl() {
    for (const size_t n : {0, 1, 10, 100, 1000, 10000}) {
        for (const auto op : {CompareOpType::EQ, CompareOpType::GE, CompareOpType::GT, CompareOpType::LE, CompareOpType::LT, CompareOpType::NEQ}) {
            BitsetT bitset(n);
            bitset.reset();

            if (print_log) {
                printf("Testing bitset, n=%zd, op=%zd\n", n, (size_t)op);
            }
            
            TestInplaceCompareValImpl<BitsetT, T>(bitset, op);

            for (const size_t offset : {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 35, 45, 55, 63, 127, 703}) {
                if (offset >= n) {
                    continue;
                }

                bitset.reset();
                auto view = bitset.view(offset);

                if (print_log) {
                    printf("Testing bitset view, n=%zd, offset=%zd, op=%zd\n", n, offset, (size_t)op);
                }
                
                TestInplaceCompareValImpl<decltype(view), T>(view, op);
            }
        }
    }
}

//
template<typename T>
class InplaceCompareValSuite : public ::testing::Test {};

TYPED_TEST_SUITE_P(InplaceCompareValSuite);

TYPED_TEST_P(InplaceCompareValSuite, BitWise) {
    TestInplaceCompareValImpl<
        ref_u64_u8::bitset_type, 
        TypeParam>();
}

TYPED_TEST_P(InplaceCompareValSuite, ElementWise) {
    TestInplaceCompareValImpl<
        element_u64_u8::bitset_type, 
        TypeParam>();
}

TYPED_TEST_P(InplaceCompareValSuite, Avx2) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx2()) {
        TestInplaceCompareValImpl<
            avx2_u64_u8::bitset_type, 
            TypeParam>();
    }
#endif
}

TYPED_TEST_P(InplaceCompareValSuite, Avx512) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx512()) {
        TestInplaceCompareValImpl<
            avx512_u64_u8::bitset_type, 
            TypeParam>();
    }
#endif
}

TYPED_TEST_P(InplaceCompareValSuite, Neon) {
#if defined(__aarch64__)
    using namespace milvus::bitset::detail::arm;

    TestInplaceCompareValImpl<
        neon_u64_u8::bitset_type, 
        TypeParam>();
#endif
}

TYPED_TEST_P(InplaceCompareValSuite, Sve) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
    using namespace milvus::bitset::detail::arm;

    TestInplaceCompareValImpl<
        sve_u64_u8::bitset_type, 
        TypeParam>();
#endif
}

TYPED_TEST_P(InplaceCompareValSuite, Dynamic) {
    TestInplaceCompareValImpl<
        dynamic_u64_u8::bitset_type, 
        TypeParam>();
}

using InplaceCompareValTtypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

REGISTER_TYPED_TEST_SUITE_P(InplaceCompareValSuite, BitWise, ElementWise, Avx2, Avx512, Neon, Sve, Dynamic);

INSTANTIATE_TYPED_TEST_SUITE_P(InplaceCompareValTest, InplaceCompareValSuite, InplaceCompareValTtypes);


//////////////////////////////////////////////////////////////////////////////////////////

//
template<typename BitsetT, typename T>
void TestInplaceWithinRangeColumnImpl(
    BitsetT& bitset, RangeType op
) {
    const size_t n = bitset.size();
    constexpr size_t max_v = 3;

    std::vector<T> range(n, 0);
    std::vector<T> values(n, 0);

    std::vector<T> lower(n, 0);
    std::vector<T> upper(n, 0);

    std::default_random_engine rng(123);
    FillRandom(lower, rng, max_v);
    FillRandom(range, rng, max_v);
    FillRandom(values, rng, 2 * max_v);

    for (size_t i = 0; i < n; i++) {
        upper[i] = lower[i] + range[i];
    }

    StopWatch sw;
    bitset.inplace_within_range_column(lower.data(), upper.data(), values.data(), n, op);
    
    if (print_timing) {
        printf("elapsed %f\n", sw.elapsed());
    }

    for (size_t i = 0; i < n; i++) {
        if (op == RangeType::IncInc) {
            ASSERT_EQ(lower[i] <= values[i] && values[i] <= upper[i], bitset[i]) << i << " " << lower[i] << " " << values[i] << " " << upper[i];
        } else if (op == RangeType::IncExc) {
            ASSERT_EQ(lower[i] <= values[i] && values[i] < upper[i], bitset[i]) << i << " " << lower[i] << " " << values[i] << " " << upper[i];
        } else if (op == RangeType::ExcInc) {
            ASSERT_EQ(lower[i] < values[i] && values[i] <= upper[i], bitset[i]) << i << " " << lower[i] << " " << values[i] << " " << upper[i];
        } else if (op == RangeType::ExcExc) {
            ASSERT_EQ(lower[i] < values[i] && values[i] < upper[i], bitset[i]) << i << " " << lower[i] << " " << values[i] << " " << upper[i];
        } else {
            ASSERT_TRUE(false) << "Not implemented";
        }
    }
}

template<typename BitsetT, typename T>
void TestInplaceWithinRangeColumnImpl() {
    for (const size_t n : {0, 1, 10, 100, 1000, 10000}) {
        for (const auto op : {RangeType::IncInc, RangeType::IncExc, RangeType::ExcInc, RangeType::ExcExc}) {
            BitsetT bitset(n);
            bitset.reset();

            if (print_log) {
                printf("Testing bitset, n=%zd, op=%zd\n", n, (size_t)op);
            }
            
            TestInplaceWithinRangeColumnImpl<BitsetT, T>(bitset, op);

            for (const size_t offset : {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 35, 45, 55, 63, 127, 703}) {
                if (offset >= n) {
                    continue;
                }

                bitset.reset();
                auto view = bitset.view(offset);

                if (print_log) {
                    printf("Testing bitset view, n=%zd, offset=%zd, op=%zd\n", n, offset, (size_t)op);
                }
                
                TestInplaceWithinRangeColumnImpl<decltype(view), T>(view, op);
            }
        }
    }
}


//
template<typename T>
class InplaceWithinRangeColumnSuite : public ::testing::Test {};

TYPED_TEST_SUITE_P(InplaceWithinRangeColumnSuite);

TYPED_TEST_P(InplaceWithinRangeColumnSuite, BitWise) {
    TestInplaceWithinRangeColumnImpl<
        ref_u64_u8::bitset_type, 
        TypeParam>();
}

TYPED_TEST_P(InplaceWithinRangeColumnSuite, ElementWise) {
    TestInplaceWithinRangeColumnImpl<
        element_u64_u8::bitset_type, 
        TypeParam>();
}

TYPED_TEST_P(InplaceWithinRangeColumnSuite, Avx2) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx2()) {
        TestInplaceWithinRangeColumnImpl<
            avx2_u64_u8::bitset_type, 
            TypeParam>();
    }
#endif
}

TYPED_TEST_P(InplaceWithinRangeColumnSuite, Avx512) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx512()) {
        TestInplaceWithinRangeColumnImpl<
            avx512_u64_u8::bitset_type, 
            TypeParam>();
    }
#endif
}

TYPED_TEST_P(InplaceWithinRangeColumnSuite, Neon) {
#if defined(__aarch64__)
    using namespace milvus::bitset::detail::arm;

    TestInplaceWithinRangeColumnImpl<
        neon_u64_u8::bitset_type, 
        TypeParam>();
#endif
}

TYPED_TEST_P(InplaceWithinRangeColumnSuite, Dynamic) {
    TestInplaceWithinRangeColumnImpl<
        dynamic_u64_u8::bitset_type, 
        TypeParam>();
}

using InplaceWithinRangeColumnTtypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

REGISTER_TYPED_TEST_SUITE_P(InplaceWithinRangeColumnSuite, BitWise, ElementWise, Avx2, Avx512, Neon, Dynamic);

INSTANTIATE_TYPED_TEST_SUITE_P(InplaceWithinRangeColumnTest, InplaceWithinRangeColumnSuite, InplaceWithinRangeColumnTtypes);


//////////////////////////////////////////////////////////////////////////////////////////

//
template<typename BitsetT, typename T>
void TestInplaceWithinRangeValImpl(
    BitsetT& bitset, RangeType op
) {
    const size_t n = bitset.size();
    constexpr size_t max_v = 10;
    constexpr T lower_v = 3;
    constexpr T upper_v = 7;

    std::vector<T> values(n, 0);

    std::default_random_engine rng(123);
    FillRandom(values, rng, max_v);


    StopWatch sw;
    bitset.inplace_within_range_val(lower_v, upper_v, values.data(), n, op);
    
    if (print_timing) {
        printf("elapsed %f\n", sw.elapsed());
    }

    for (size_t i = 0; i < n; i++) {
        if (op == RangeType::IncInc) {
            ASSERT_EQ(lower_v <= values[i] && values[i] <= upper_v, bitset[i]) << i << " " << lower_v << " " << values[i] << " " << upper_v;
        } else if (op == RangeType::IncExc) {
            ASSERT_EQ(lower_v <= values[i] && values[i] < upper_v, bitset[i]) << i << " " << lower_v << " " << values[i] << " " << upper_v;
        } else if (op == RangeType::ExcInc) {
            ASSERT_EQ(lower_v < values[i] && values[i] <= upper_v, bitset[i]) << i << " " << lower_v << " " << values[i] << " " << upper_v;
        } else if (op == RangeType::ExcExc) {
            ASSERT_EQ(lower_v < values[i] && values[i] < upper_v, bitset[i]) << i << " " << lower_v << " " << values[i] << " " << upper_v;
        } else {
            ASSERT_TRUE(false) << "Not implemented";
        }
    }
}

template<typename BitsetT, typename T>
void TestInplaceWithinRangeValImpl() {
    for (const size_t n : {0, 1, 10, 100, 1000, 10000}) {
        for (const auto op : {RangeType::IncInc, RangeType::IncExc, RangeType::ExcInc, RangeType::ExcExc}) {
            BitsetT bitset(n);
            bitset.reset();

            if (print_log) {
                printf("Testing bitset, n=%zd, op=%zd\n", n, (size_t)op);
            }
            
            TestInplaceWithinRangeValImpl<BitsetT, T>(bitset, op);

            for (const size_t offset : {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 35, 45, 55, 63, 127, 703}) {
                if (offset >= n) {
                    continue;
                }

                bitset.reset();
                auto view = bitset.view(offset);

                if (print_log) {
                    printf("Testing bitset view, n=%zd, offset=%zd, op=%zd\n", n, offset, (size_t)op);
                }
                
                TestInplaceWithinRangeValImpl<decltype(view), T>(view, op);
            }
        }
    }
}


//
template<typename T>
class InplaceWithinRangeValSuite : public ::testing::Test {};

TYPED_TEST_SUITE_P(InplaceWithinRangeValSuite);

TYPED_TEST_P(InplaceWithinRangeValSuite, BitWise) {
    TestInplaceWithinRangeValImpl<
        ref_u64_u8::bitset_type, 
        TypeParam>();
}

TYPED_TEST_P(InplaceWithinRangeValSuite, ElementWise) {
    TestInplaceWithinRangeValImpl<
        element_u64_u8::bitset_type, 
        TypeParam>();
}

TYPED_TEST_P(InplaceWithinRangeValSuite, Avx2) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx2()) {
        TestInplaceWithinRangeValImpl<
            avx2_u64_u8::bitset_type, 
            TypeParam>();
    }
#endif
}

TYPED_TEST_P(InplaceWithinRangeValSuite, Avx512) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx512()) {
        TestInplaceWithinRangeValImpl<
            avx512_u64_u8::bitset_type, 
            TypeParam>();
    }
#endif
}

TYPED_TEST_P(InplaceWithinRangeValSuite, Neon) {
#if defined(__aarch64__)
    using namespace milvus::bitset::detail::arm;

    TestInplaceWithinRangeValImpl<
        neon_u64_u8::bitset_type, 
        TypeParam>();
#endif
}

TYPED_TEST_P(InplaceWithinRangeValSuite, Dynamic) {
    TestInplaceWithinRangeValImpl<
        dynamic_u64_u8::bitset_type, 
        TypeParam>();
}

using InplaceWithinRangeValTtypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

REGISTER_TYPED_TEST_SUITE_P(InplaceWithinRangeValSuite, BitWise, ElementWise, Avx2, Avx512, Neon, Dynamic);

INSTANTIATE_TYPED_TEST_SUITE_P(InplaceWithinRangeValTest, InplaceWithinRangeValSuite, InplaceWithinRangeValTtypes);


//////////////////////////////////////////////////////////////////////////////////////////

//
template<typename BitsetT, typename T>
void TestInplaceArithCompareImpl(
    BitsetT& bitset, ArithOpType a_op, CompareOpType cmp_op
) {
    using HT = ArithHighPrecisionType<T>;

    const size_t n = bitset.size();
    constexpr size_t max_v = 10;

    std::vector<T> left(n, 0);
    HT right_operand = 2;
    HT value = 5;

    std::default_random_engine rng(123);
    FillRandom(left, rng, max_v);

    StopWatch sw;
    bitset.inplace_arith_compare(left.data(), right_operand, value, n, a_op, cmp_op);
    
    if (print_timing) {
        printf("elapsed %f\n", sw.elapsed());
    }

    for (size_t i = 0; i < n; i++) {
        if (a_op == ArithOpType::Add && cmp_op == CompareOpType::EQ) {
            ASSERT_EQ((left[i] + right_operand) == value, bitset[i]) << i;
        } else if (a_op == ArithOpType::Add && cmp_op == CompareOpType::NEQ) {
            ASSERT_EQ((left[i] + right_operand) != value, bitset[i]) << i;
        } else if (a_op == ArithOpType::Sub && cmp_op == CompareOpType::EQ) {
            ASSERT_EQ((left[i] - right_operand) == value, bitset[i]) << i;
        } else if (a_op == ArithOpType::Sub && cmp_op == CompareOpType::NEQ) {
            ASSERT_EQ((left[i] - right_operand) != value, bitset[i]) << i;
        } else if (a_op == ArithOpType::Mul && cmp_op == CompareOpType::EQ) {
            ASSERT_EQ((left[i] * right_operand) == value, bitset[i]) << i;
        } else if (a_op == ArithOpType::Mul && cmp_op == CompareOpType::NEQ) {
            ASSERT_EQ((left[i] * right_operand) != value, bitset[i]) << i;
        } else if (a_op == ArithOpType::Div && cmp_op == CompareOpType::EQ) {
            ASSERT_EQ((left[i] / right_operand) == value, bitset[i]) << i;
        } else if (a_op == ArithOpType::Div && cmp_op == CompareOpType::NEQ) {
            ASSERT_EQ((left[i] / right_operand) != value, bitset[i]) << i;
        } else if (a_op == ArithOpType::Mod && cmp_op == CompareOpType::EQ) {
            ASSERT_EQ(fmod(left[i], right_operand) == value, bitset[i]) << i;
        } else if (a_op == ArithOpType::Mod && cmp_op == CompareOpType::NEQ) {
            ASSERT_EQ(fmod(left[i], right_operand) != value, bitset[i]) << i;
        } else {
            ASSERT_TRUE(false) << "Not implemented";
        }
    }
}

template<typename BitsetT, typename T>
void TestInplaceArithCompareImpl() {
    for (const size_t n : {0, 1, 10, 100, 1000, 10000}) {
        for (const auto a_op : {ArithOpType::Add, ArithOpType::Sub, ArithOpType::Mul, ArithOpType::Div, ArithOpType::Mod}) {
            for (const auto cmp_op : {CompareOpType::EQ, CompareOpType::NEQ}) {
                BitsetT bitset(n);
                bitset.reset();

                if (print_log) {
                    printf("Testing bitset, n=%zd, a_op=%zd\n", n, (size_t)a_op);
                }
                
                TestInplaceArithCompareImpl<BitsetT, T>(bitset, a_op, cmp_op);

                for (const size_t offset : {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 35, 45, 55, 63, 127, 703}) {
                    if (offset >= n) {
                        continue;
                    }

                    bitset.reset();
                    auto view = bitset.view(offset);

                    if (print_log) {
                        printf("Testing bitset view, n=%zd, offset=%zd, a_op=%zd, cmp_op=%zd\n", n, offset, (size_t)a_op, (size_t)cmp_op);
                    }
                    
                    TestInplaceArithCompareImpl<decltype(view), T>(view, a_op, cmp_op);
                }
            }
        }
    }
}

//
template<typename T>
class InplaceArithCompareSuite : public ::testing::Test {};

TYPED_TEST_SUITE_P(InplaceArithCompareSuite);


TYPED_TEST_P(InplaceArithCompareSuite, BitWise) {
    TestInplaceArithCompareImpl<
        ref_u64_u8::bitset_type, 
        TypeParam>();
}

TYPED_TEST_P(InplaceArithCompareSuite, ElementWise) {
    TestInplaceArithCompareImpl<
        element_u64_u8::bitset_type, 
        TypeParam>();
}

TYPED_TEST_P(InplaceArithCompareSuite, Avx2) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx2()) {
        TestInplaceArithCompareImpl<
            avx2_u64_u8::bitset_type, 
            TypeParam>();
    }
#endif
}


TYPED_TEST_P(InplaceArithCompareSuite, Avx512) {
#if defined(__x86_64__)
    using namespace milvus::bitset::detail::x86;

    if (cpu_support_avx512()) {
        TestInplaceArithCompareImpl<
            avx512_u64_u8::bitset_type, 
            TypeParam>();
    }
#endif
}

TYPED_TEST_P(InplaceArithCompareSuite, Neon) {
#if defined(__aarch64__)
    using namespace milvus::bitset::detail::arm;

    TestInplaceArithCompareImpl<
        neon_u64_u8::bitset_type, 
        TypeParam>();
#endif
}


TYPED_TEST_P(InplaceArithCompareSuite, Dynamic) {
    TestInplaceArithCompareImpl<
        dynamic_u64_u8::bitset_type, 
        TypeParam>();
}

using InplaceArithCompareTtypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

REGISTER_TYPED_TEST_SUITE_P(InplaceArithCompareSuite, BitWise, ElementWise, Avx2, Avx512, Neon, Dynamic);

INSTANTIATE_TYPED_TEST_SUITE_P(InplaceArithCompareTest, InplaceArithCompareSuite, InplaceArithCompareTtypes);

//////////////////////////////////////////////////////////////////////////////////////////

template<typename BitsetT, typename BitsetU>
void TestAppendImpl(
    BitsetT& bitset_dst, const BitsetU& bitset_src
) {
    std::vector<bool> b_dst;
    b_dst.reserve(bitset_src.size() + bitset_dst.size());

    for (size_t i = 0; i < bitset_dst.size(); i++) {
        b_dst.push_back(bitset_dst[i]);
    }
    for (size_t i = 0; i < bitset_src.size(); i++) {
        b_dst.push_back(bitset_src[i]);
    }

    StopWatch sw;
    bitset_dst.append(bitset_src);
    
    if (print_timing) {
        printf("elapsed %f\n", sw.elapsed());
    }

    //
    ASSERT_EQ(b_dst.size(), bitset_dst.size());
    for (size_t i = 0; i < bitset_dst.size(); i++) {
        ASSERT_EQ(b_dst[i], bitset_dst[i]) << i;
    }
}

template<typename BitsetT>
void TestAppendImpl() {
    std::default_random_engine rng(345);

    const auto sizes = {0, 1, 10, 100, 1000, 10000};

    std::vector<BitsetT> bt0;
    for (const size_t n : sizes) {
        BitsetT bitset(n);
        FillRandom(bitset, rng);
        bt0.push_back(std::move(bitset));
    }

    std::vector<BitsetT> bt1;
    for (const size_t n : sizes) {
        BitsetT bitset(n);
        FillRandom(bitset, rng);
        bt1.push_back(std::move(bitset));
    }

    for (const auto& bt_a : bt0) {
        for (const auto& bt_b : bt1) {
            auto bt = bt_a.clone();

            if (print_log) {
                printf("Testing bitset, n=%zd, m=%zd\n", bt_a.size(), bt_b.size());
            }

            TestAppendImpl(bt, bt_b);

            for (const size_t offset : {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 35, 45, 55, 63, 127, 703}) {
                if (offset >= bt_b.size()) {
                    continue;
                }
                
                bt = bt_a.clone();
                auto view = bt_b.view(offset);

                if (print_log) {
                    printf("Testing bitset view, n=%zd, m=%zd, offset=%zd\n", bt_a.size(), bt_b.size(), offset);
                }

                TestAppendImpl(bt, view);
            }
        }
    }
}

TEST(Append, BitWise) {
    TestAppendImpl<ref_u64_u8::bitset_type>();
}

TEST(Append, ElementWise) {
    TestAppendImpl<element_u64_u8::bitset_type>();
}

//////////////////////////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////////////////////////
/*
TEST(FOO,Boo) {
    constexpr size_t n = 10000;
    using bitset_t = avx2_u64_u8::bitset_type;
    bitset_t bitset(n);
    bitset.reset();

    std::vector<std::string> values(n);
    std::string lower = "lower";
    std::string upper = "upper";

    std::default_random_engine rng(123);

    bitset.inplace_within_range_val(lower, upper, values.data(), n, RangeType::IncInc);

}
*/
