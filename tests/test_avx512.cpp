#include <gtest/gtest.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <avx512.h>
#include <bitset.h>
#include <detail/bit_wise.h>

using namespace milvus::bitset;
using namespace milvus::bitset::detail;

using policy_type = CustomBitsetPolicy<uint64_t>;
using container_type = std::vector<uint8_t>;
using bitset_type = CustomBitsetOwning<policy_type, container_type, false>;
using bitset_view = CustomBitsetNonOwning<policy_type, false>;

//
void TestEqualValAVX512(const size_t n) {
    // this is the constraint of the API
    ASSERT_TRUE(n % 8 == 0);

    bitset_type bitset(n);
    bitset.reset();

    std::default_random_engine rng(123);
    std::uniform_int_distribution<int8_t> u(0, 4);

    std::vector<int8_t> values(n, 0);
    for (size_t i = 0; i < n; i++) {
        values[i] = u(rng);
    }

    EqualValAVX512(values.data(), n, int8_t(1), bitset.data());

    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(bitset[i], values[i] == 1);
    }
}

TEST(EqualValAVX512, f) {
    TestEqualValAVX512(0x00);
    TestEqualValAVX512(0x08);
    TestEqualValAVX512(0x10);
    TestEqualValAVX512(0x18);
    TestEqualValAVX512(0x20);
    TestEqualValAVX512(0x28);
    TestEqualValAVX512(0x30);
    TestEqualValAVX512(0x38);
    TestEqualValAVX512(0x40);

    TestEqualValAVX512(0x1000);
    TestEqualValAVX512(0x1008);
    TestEqualValAVX512(0x1010);
    TestEqualValAVX512(0x1018);
    TestEqualValAVX512(0x1020);
    TestEqualValAVX512(0x1028);
    TestEqualValAVX512(0x1030);
    TestEqualValAVX512(0x1038);
    TestEqualValAVX512(0x1040);
}

//
void TestAndAVX512(const size_t n) {
    // this is the constraint of the API
    ASSERT_TRUE(n % 8 == 0);

    bitset_type bitset0(n);
    bitset0.reset();
    bitset_type bitset1(n);
    bitset1.reset();

    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    for (size_t i = 0; i < n; i++) {
        bitset0[i] = (u(rng) < 0.25);
    }

    for (size_t i = 0; i < n; i++) {
        bitset1[i] = (u(rng) < 0.25);
    }

    bitset_type bitset_r = bitset0.clone();
    AndAVX512(bitset_r.data(), bitset1.data(), n / 8);

    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(bitset_r[i], (bitset0[i] & bitset1[i]));
    }
}

TEST(AndAVX512, f) {
    TestAndAVX512(0x00);
    TestAndAVX512(0x08);
    TestAndAVX512(0x10);
    TestAndAVX512(0x18);
    TestAndAVX512(0x20);
    TestAndAVX512(0x28);
    TestAndAVX512(0x30);
    TestAndAVX512(0x38);
    TestAndAVX512(0x40);

    TestAndAVX512(0x1000);
    TestAndAVX512(0x1008);
    TestAndAVX512(0x1010);
    TestAndAVX512(0x1018);
    TestAndAVX512(0x1020);
    TestAndAVX512(0x1028);
    TestAndAVX512(0x1030);
    TestAndAVX512(0x1038);
    TestAndAVX512(0x1040);
}

//
void TestOrAVX512(const size_t n) {
    // this is the constraint of the API
    ASSERT_TRUE(n % 8 == 0);

    bitset_type bitset0(n);
    bitset0.reset();
    bitset_type bitset1(n);
    bitset1.reset();

    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    for (size_t i = 0; i < n; i++) {
        bitset0[i] = (u(rng) < 0.25);
    }

    for (size_t i = 0; i < n; i++) {
        bitset1[i] = (u(rng) < 0.25);
    }

    bitset_type bitset_r = bitset0.clone();
    OrAVX512(bitset_r.data(), bitset1.data(), n / 8);

    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(bitset_r[i], (bitset0[i] | bitset1[i]));
    }
}

TEST(OrAVX512, f) {
    TestOrAVX512(0x00);
    TestOrAVX512(0x08);
    TestOrAVX512(0x10);
    TestOrAVX512(0x18);
    TestOrAVX512(0x20);
    TestOrAVX512(0x28);
    TestOrAVX512(0x30);
    TestOrAVX512(0x38);
    TestOrAVX512(0x40);

    TestOrAVX512(0x1000);
    TestOrAVX512(0x1008);
    TestOrAVX512(0x1010);
    TestOrAVX512(0x1018);
    TestOrAVX512(0x1020);
    TestOrAVX512(0x1028);
    TestOrAVX512(0x1030);
    TestOrAVX512(0x1038);
    TestOrAVX512(0x1040);
}
