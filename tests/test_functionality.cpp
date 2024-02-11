#include <gtest/gtest.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <bitset.h>
#include <detail/bit_wise.h>
#include <detail/element_wise.h>
#include <detail/element_vectorized.h>

#include <detail/platform/x86/avx2.h>

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
namespace avx2_u64_u8 {

using vectorized_type = milvus::bitset::detail::x86::VectorizedAvx2;
using policy_type = milvus::bitset::detail::CustomBitsetVectorizedPolicy<uint64_t, vectorized_type>;
using container_type = std::vector<uint8_t>;
using bitset_type = milvus::bitset::CustomBitsetOwning<policy_type, container_type, false>;
using bitset_view = milvus::bitset::CustomBitsetNonOwning<policy_type, false>;

}


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

    ASSERT_FALSE(bit_idx.has_value());
}

template<typename BitsetT>
void TestFindImpl() {
    for (const size_t n : {0, 1, 10, 100, 1000, 10000}) {
        for (const size_t pr : {1, 100}) {
            BitsetT bitset(n);
            bitset.reset();

            printf("Testing bitset, n=%zd, pr=%zd\n", n, pr);
            TestFindImpl(bitset, pr);

            for (const size_t offset : {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 35, 45, 55, 63}) {
                if (offset >= n) {
                    continue;
                }

                bitset.reset();
                auto view = bitset.view(offset);

                printf("Testing bitset view, n=%zd, offset=%zd, pr=%zd\n", n, offset, pr);
                TestFindImpl(view, pr);
            }
        }
    }
}

// //
// TEST(FindRef, f) {
//     TestFindImpl<ref_u64_u8::bitset_type>();
// }

// //
// TEST(FindElement, f) {
//     TestFindImpl<element_u64_u8::bitset_type>();
// }

// //
// TEST(FindVectorizedAvx2, f) {
//     TestFindImpl<avx2_u64_u8::bitset_type>();
// }

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

//
template<typename BitsetT, typename T, typename U>
void TestInplaceCompareImpl(
    BitsetT& bitset, CompareType op
) {
    const size_t n = bitset.size();
    constexpr size_t max_v = 2;

    std::vector<T> t(n, 0);
    std::vector<U> u(n, 0);

    std::default_random_engine rng(123);
    FillRandom(t, rng, max_v);
    FillRandom(u, rng, max_v);

    bitset.inplace_compare(t.data(), u.data(), n, op);

    for (size_t i = 0; i < n; i++) {
        if (op == CompareType::EQ) {
            ASSERT_EQ(t[i] == u[i], bitset[i]) << i;
        } else if (op == CompareType::GE) {
            ASSERT_EQ(t[i] >= u[i], bitset[i]) << i;
        } else if (op == CompareType::GT) {
            ASSERT_EQ(t[i] > u[i], bitset[i]) << i;
        } else if (op == CompareType::LE) {
            ASSERT_EQ(t[i] <= u[i], bitset[i]) << i;            
        } else if (op == CompareType::LT) {
            ASSERT_EQ(t[i] < u[i], bitset[i]) << i;            
        } else if (op == CompareType::NEQ) {
            ASSERT_EQ(t[i] != u[i], bitset[i]) << i;            
        } else {
            ASSERT_TRUE(false) << "Not implemented";
        }
    }
}

template<typename BitsetT>
void TestInplaceCompareImpl() {
    for (const size_t n : {0, 1, 10, 100, 1000, 10000}) {
        for (const auto op : {CompareType::EQ, CompareType::GE, CompareType::GT, CompareType::LE, CompareType::LT, CompareType::NEQ}) {
            BitsetT bitset(n);
            bitset.reset();

            printf("Testing bitset, n=%zd, op=%zd\n", n, (size_t)op);
            TestInplaceCompareImpl<BitsetT, float, float>(bitset, op);

            for (const size_t offset : {0, 1, 2, 3, 4, 5, 6, 7, 11, 21, 35, 45, 55, 63}) {
                if (offset >= n) {
                    continue;
                }

                bitset.reset();
                auto view = bitset.view(offset);

                printf("Testing bitset view, n=%zd, offset=%zd, op=%zd\n", n, offset, (size_t)op);
                TestInplaceCompareImpl<decltype(view), float, float>(view, op);
            }
        }
    }
}

//
TEST(InplaceCompareRef, f) {
    TestInplaceCompareImpl<ref_u64_u8::bitset_type>();
}

//
TEST(InplaceCompareAvx2, f) {
    TestInplaceCompareImpl<avx2_u64_u8::bitset_type>();
}
