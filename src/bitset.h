#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "common.h"

namespace milvus {
namespace bitset {

namespace {

// A supporting facility for checking out of range.
// It is needed to add a capability to verify that we won't go out of 
//   range even for the Release build.
template<bool RangeCheck>
struct RangeChecker{};

// disabled.
template<>
struct RangeChecker<false> {
    // Check if a < max
    template<typename SizeT>
    static inline void lt(const SizeT a, const SizeT max) {}

    // Check if a <= max
    template<typename SizeT>
    static inline void le(const SizeT a, const SizeT max) {}

    // Check if a == b
    template<typename SizeT>
    static inline void eq(const SizeT a, const SizeT b) {}
};

// enabled.
template<>
struct RangeChecker<true> {
    // Check if a < max
    template<typename SizeT>
    static inline void lt(const SizeT a, const SizeT max) {
        // todo: replace
        assert(a < max);
    }

    // Check if a <= max
    template<typename SizeT>
    static inline void le(const SizeT a, const SizeT max) {
        // todo: replace
        assert(a <= max);
    }

    // Check if a == b
    template<typename SizeT>
    static inline void eq(const SizeT a, const SizeT b) {
        // todo: replace
        assert(a == b);
    }
};

}

// CRTP

// Bitset view, which does not own the data.
template<typename PolicyT, bool IsRangeCheckEnabled>
class CustomBitsetNonOwning;

// Bitset, which owns the data.
template<typename PolicyT, typename ContainerT, bool IsRangeCheckEnabled>
class CustomBitsetOwning;

// This is the base CRTP class.
template<typename PolicyT, typename ImplT, bool IsRangeCheckEnabled>
class CustomBitsetBase {
   template<typename, bool>
   friend class CustomBitsetNonOwning;

   template<typename, typename, bool>
   friend class CustomBitsetOwning;

public:
    using policy_type = PolicyT;
    using data_type = typename policy_type::data_type;
    using size_type = typename policy_type::size_type;
    using proxy_type = typename policy_type::proxy_type;
    using const_proxy_type = typename policy_type::const_proxy_type;

    using range_checker = RangeChecker<IsRangeCheckEnabled>;

    //
    inline data_type* data() { 
        return as_derived().data_impl(); 
    }

    //
    inline const data_type* data() const {
        return as_derived().data_impl();
    }

    // Return the number of bits we're working with.
    inline size_type size() const {
        return as_derived().size_impl();
    }

    // Return the number of bytes which is needed to 
    //   contain all our bits.
    inline size_type size_in_bytes() const { 
        return policy_type::get_required_size_in_bytes(this->size());
    }

    // Return the number of elements which is needed to 
    //   contain all our bits.
    inline size_type size_in_elements() const { 
        return policy_type::get_required_size_in_elements(this->size());
    }

    //
    inline bool empty() const {
        return (this->size() == 0);
    }

    //
    inline proxy_type operator[](const size_type bit_idx) {
        range_checker::lt(bit_idx, this->size());

        const size_type idx_v = bit_idx + this->offset();
        return policy_type::get_proxy(this->data(), idx_v);
    }

    //
    inline bool operator[](const size_type bit_idx) const {
        range_checker::lt(bit_idx, this->size());

        const size_type idx_v = bit_idx + this->offset();
        const auto proxy = policy_type::get_proxy(this->data(), idx_v);
        return proxy.operator bool();
    }

    // Set all bits to true.
    inline void set() {
        policy_type::set(this->data(), this->offset(), this->size());
    }

    // Set a given bit to true.
    inline void set(const size_type bit_idx) {
        this->operator[](bit_idx) = true;
    }

    // Set a given bit to a given value.
    inline void set(const size_type bit_idx, const bool value = true) {
        this->operator[](bit_idx) = value;
    }

    // Set all bits to false.
    inline void reset() {
        policy_type::reset(this->data(), this->offset(), this->size());
    }

    // Set a given bit to false.
    inline void reset(const size_type bit_idx) {
        this->operator[](bit_idx) = false;
    }

    // Return whether all bits are set to true.
    inline bool all() const {
        return policy_type::all(this->data(), this->offset(), this->size());
    }

    // Return whether any of the bits is set to true.
    inline bool any() const {
        return (!this->none());
    }

    // Return whether all bits are set to false.
    inline bool none() const {
        return policy_type::none(this->data(), this->offset(), this->size());
    }

    // Inplace and.
    template<typename I, bool R>
    inline void inplace_and(const CustomBitsetBase<PolicyT, I, R>& other, const size_type size) {
        range_checker::le(size, this->size());
        range_checker::le(size, other.size());

        policy_type::op_and(
            this->data(),
            other.data(),
            this->offset(),
            other.offset(),
            size
        );
    }

    // Inplace and. A given bitset / bitset view is expected to have the same size.
    template<typename I, bool R>
    inline ImplT& operator&=(const CustomBitsetBase<PolicyT, I, R>& other) {
        range_checker::eq(other.size(), this->size());

        this->inplace_and(other, this->size());
        return as_derived();
    }

    // Inplace or.
    template<typename I, bool R>
    inline void inplace_or(const CustomBitsetBase<PolicyT, I, R>& other, const size_type size) {
        range_checker::le(size, this->size());
        range_checker::le(size, other.size());

        policy_type::op_or(
            this->data(),
            other.data(),
            this->offset(),
            other.offset(),
            size
        );
    }

    // Inplace or. A given bitset / bitset view is expected to have the same size.
    template<typename I, bool R>
    inline ImplT& operator|=(const CustomBitsetBase<PolicyT, I, R>& other) {
        range_checker::eq(other.size(), this->size());

        this->inplace_or(other, this->size());
        return as_derived();
    }

    // Revert all bits.
    inline void flip() {
        policy_type::op_flip(this->data(), this->offset(), this->size());
    }

    //
    inline CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> operator+(const size_type offset) {
        return this->view(offset);
    }

    // Create a view of a given size from the given position.
    inline CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view(const size_type offset, const size_type size) {
        range_checker::le(offset, this->size());
        range_checker::le(offset + size, this->size());

        return CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled>(
            this->data(),
            this->offset() + offset,
            size
        );
    }

    // Create a const view of a given size from the given position.
    inline CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view(const size_type offset, const size_type size) const {
        range_checker::le(offset, this->size());
        range_checker::le(offset + size, this->size());

        return CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled>(
            const_cast<data_type*>(this->data()),
            this->offset() + offset,
            size
        );
    }

    // Create a view from the given position, which uses all available size.
    inline CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view(const size_type offset) {
        range_checker::le(offset, this->size());

        return CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled>(
            this->data(),
            this->offset() + offset,
            this->size() - offset
        );
    }

    // Create a const view from the given position, which uses all available size.
    inline const CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view(const size_type offset) const {
        range_checker::le(offset, this->size());

        return CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled>(
            const_cast<data_type*>(this->data()),
            this->offset() + offset,
            this->size() - offset
        );
    }

    // Create a view.
    inline CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view() {
        return this->view(0);
    }

    // Create a const view.
    inline const CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view() const {
        return this->view(0);
    }

    // Return the number of bits which are set to true.
    inline size_type count() const {
        return policy_type::op_count(this->data(), this->offset(), this->size());
    }

    // Compare the current bitset with another bitset / bitset view.
    template<typename I, bool R>
    inline bool operator==(const CustomBitsetBase<PolicyT, I, R>& other) {
        if (this->size() != other.size()) {
            return false;
        }

        return policy_type::op_eq(
            this->data(),
            other.data(),
            this->offset(),
            other.offset(),
            this->size()
        );
    }

    // Compare the current bitset with another bitset / bitset view.
    template<typename I, bool R>
    inline bool operator!=(const CustomBitsetBase<PolicyT, I, R>& other) {
        return (!(*this == other));
    }

    // Inplace xor.
    template<typename I, bool R>
    inline void inplace_xor(const CustomBitsetBase<PolicyT, I, R>& other, const size_type size) {
        range_checker::le(offset, this->size());
        range_checker::le(offset + size, this->size());
    
        policy_type::op_xor(
            this->data(),
            other.data(),
            this->offset(),
            other.offset(),
            size
        );
    }

    // Inplace xor. A given bitset / bitset view is expected to have the same size.
    template<typename I, bool R>
    inline ImplT& operator^=(const CustomBitsetBase<PolicyT, I, R>& other) {
        range_checker::eq(other.size(), this->size());

        this->inplace_xor(other, this->size());
        return as_derived();
    }

    // Inplace sub.
    template<typename I, bool R>
    inline void inplace_sub(const CustomBitsetBase<PolicyT, I, R>& other, const size_type size) {
        range_checker::le(offset, this->size());
        range_checker::le(offset + size, this->size());
    
        policy_type::op_sub(
            this->data(),
            other.data(),
            this->offset(),
            other.offset(),
            size
        );
    }

    // Inplace sub. A given bitset / bitset view is expected to have the same size.
    template<typename I, bool R>
    inline ImplT& operator-=(const CustomBitsetBase<PolicyT, I, R>& other) {
        range_checker::eq(other.size(), this->size());

        this->inplace_sub(other, this->size());
        return as_derived();
    }

    // Find the index of the first bit set to true.
    inline std::optional<size_type> find_first() const {
        return policy_type::find(this->data(), this->offset(), this->size(), 0);
    }

    // Find the index of the first bit set to true, starting from a given bit index.
    inline std::optional<size_type> find_next(const size_type starting_bit_idx) const {
        const size_type size_v = this->size();
        if (starting_bit_idx + 1 >= size_v) {
            return std::nullopt;
        }

        return policy_type::find(this->data(), this->offset(), this->size(), starting_bit_idx + 1);
    }

    // Read multiple bits starting from a given bit index.
    inline data_type read(
        const size_type starting_bit_idx,
        const size_type nbits
    ) {
        range_checker::le(nbits, sizeof(data_type));

        return policy_type::read(
            this->data(),
            this->offset() + starting_bit_idx,
            nbits
        );
    }

    // Write multiple bits starting from a given bit index.
    inline void write(
        const size_type starting_bit_idx,
        const data_type value,
        const size_type nbits
    ) {
        range_checker::le(nbits, sizeof(data_type));

        policy_type::write(
            this->data(),
            this->offset() + starting_bit_idx,
            nbits,
            value
        );
    }

    // Compare two arrays element-wise
    template<typename T, typename U>
    void inplace_compare_column(
        const T* const __restrict t,
        const U* const __restrict u,
        const size_type size,
        CompareType op
    ) {
        if (op == CompareType::EQ) {
            this->inplace_compare_column<T, U, CompareType::EQ>(t, u, size);
        }
        else if (op == CompareType::GE) {
            this->inplace_compare_column<T, U, CompareType::GE>(t, u, size);
        }
        else if (op == CompareType::GT) {
            this->inplace_compare_column<T, U, CompareType::GT>(t, u, size);
        }
        else if (op == CompareType::LE) {
            this->inplace_compare_column<T, U, CompareType::LE>(t, u, size);
        }
        else if (op == CompareType::LT) {
            this->inplace_compare_column<T, U, CompareType::LT>(t, u, size);
        }
        else if (op == CompareType::NEQ) {
            this->inplace_compare_column<T, U, CompareType::NEQ>(t, u, size);
        }
        else {
            // unimplemented
        }
    }

    template<typename T, typename U, CompareType Op>
    void inplace_compare_column(
        const T* const __restrict t,
        const U* const __restrict u,
        const size_type size
    ) {
        range_checker::le(size, this->size());

        policy_type::template op_compare_column<T, U, Op>(
            this->data(),
            this->offset(),
            t,
            u,
            size
        );
    }

    // Compare elements of an given array with a given value
    template<typename T>
    void inplace_compare_val(
        const T* const __restrict t,
        const size_type size,
        const T value,
        CompareType op
    ) {
        if (op == CompareType::EQ) {
            this->inplace_compare_val<T, CompareType::EQ>(t, size, value);
        }
        else if (op == CompareType::GE) {
            this->inplace_compare_val<T, CompareType::GE>(t, size, value);
        }
        else if (op == CompareType::GT) {
            this->inplace_compare_val<T, CompareType::GT>(t, size, value);
        }
        else if (op == CompareType::LE) {
            this->inplace_compare_val<T, CompareType::LE>(t, size, value);
        }
        else if (op == CompareType::LT) {
            this->inplace_compare_val<T, CompareType::LT>(t, size, value);
        }
        else if (op == CompareType::NEQ) {
            this->inplace_compare_val<T, CompareType::NEQ>(t, size, value);
        }
        else {
            // unimplemented
        }
    }

    template<typename T, CompareType Op>
    void inplace_compare_val(
        const T* const __restrict t,
        const size_type size,
        const T value
    ) {
        range_checker::le(size, this->size());

        policy_type::template op_compare_val<T, Op>(
            this->data(),
            this->offset(),
            t,
            size,
            value
        );
    }

private:
    // Return the starting bit offset in our container.
    inline size_type offset() const {
        return as_derived().offset_impl();
    }

    // CRTP
    inline ImplT& as_derived() {
        return static_cast<ImplT&>(*this);
    }

    // CRTP
    inline const ImplT& as_derived() const {
        return static_cast<const ImplT&>(*this);
    }
};

// Bitset view
template<typename PolicyT, bool IsRangeCheckEnabled>
class CustomBitsetNonOwning : public CustomBitsetBase<PolicyT, CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled>, IsRangeCheckEnabled> {
    friend class CustomBitsetBase<PolicyT, CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled>, IsRangeCheckEnabled>;

public:
    using policy_type = PolicyT;
    using data_type = typename policy_type::data_type;
    using size_type = typename policy_type::size_type;
    using proxy_type = typename policy_type::proxy_type;
    using const_proxy_type = typename policy_type::const_proxy_type;

    using range_checker = RangeChecker<IsRangeCheckEnabled>;

    CustomBitsetNonOwning() {}
    CustomBitsetNonOwning(const CustomBitsetNonOwning &) = default;
    CustomBitsetNonOwning(CustomBitsetNonOwning&&) = default;
    CustomBitsetNonOwning& operator =(const CustomBitsetNonOwning&) = default;
    CustomBitsetNonOwning& operator =(CustomBitsetNonOwning&&) = default;

    template<typename ImplT, bool R>
    CustomBitsetNonOwning(CustomBitsetBase<PolicyT, ImplT, R>& bitset) :
        Data{bitset.data()}, Size{bitset.size()}, Offset{bitset.offset()} {}

    CustomBitsetNonOwning(void* data, const size_type size) :
        Data{reinterpret_cast<data_type*>(data)}, Size{size}, Offset{0} {}

    CustomBitsetNonOwning(void* data, const size_type offset, const size_type size) :
        Data{reinterpret_cast<data_type*>(data)}, Size{size}, Offset{offset} {}

private:
    // the referenced bits are [Offset, Offset + Size)
    data_type* Data = nullptr;
    // measured in bits
    size_type Size = 0;
    // measured in bits
    size_type Offset = 0;

    inline data_type* data_impl() { return Data; }
    inline const data_type* data_impl() const { return Data; }
    inline size_type size_impl() const { return Size; }
    inline size_type offset_impl() const { return Offset; }
};

// Bitset
template<typename PolicyT, typename ContainerT, bool IsRangeCheckEnabled>
class CustomBitsetOwning : public CustomBitsetBase<PolicyT, CustomBitsetOwning<PolicyT, ContainerT, IsRangeCheckEnabled>, IsRangeCheckEnabled> {
    friend class CustomBitsetBase<PolicyT, CustomBitsetOwning<PolicyT, ContainerT, IsRangeCheckEnabled>, IsRangeCheckEnabled>;

public:
    using policy_type = PolicyT;
    using data_type = typename policy_type::data_type;
    using size_type = typename policy_type::size_type;
    using proxy_type = typename policy_type::proxy_type;
    using const_proxy_type = typename policy_type::const_proxy_type;

    // This is the container type.
    using container_type = ContainerT;
    // This is how the data is stored. For example, we may operate using
    //   uint64_t values, but store the data in std::vector<uint8_t> container.
    //   This is useful if we need to convert a bitset into a container
    //   using move operator.
    using container_data_type = typename container_type::value_type;

    using range_checker = RangeChecker<IsRangeCheckEnabled>;

    // Allocate an empty one.
    CustomBitsetOwning() {}
    // Allocate the given number of bits.
    CustomBitsetOwning(const size_type size) : 
        Data(get_required_size_in_container_elements(size)), Size{size} {}
    // Allocate the given number of bits, initialize with a given value.
    CustomBitsetOwning(const size_type size, const bool init) : 
        Data(
            get_required_size_in_container_elements(size), 
            init ? data_type(-1) : 0),
        Size{size} {}
    // Do not allow implicit copies (Rust style).
    CustomBitsetOwning(const CustomBitsetOwning &) = delete;
    // Allow default move.
    CustomBitsetOwning(CustomBitsetOwning&&) = default;
    // Do not allow implicit copies (Rust style).
    CustomBitsetOwning& operator =(const CustomBitsetOwning&) = delete;
    // Allow default move.
    CustomBitsetOwning& operator =(CustomBitsetOwning&&) = default;

    template<typename C, bool R>
    CustomBitsetOwning(const CustomBitsetBase<PolicyT, C, R>& other) {
        Data = container_type(get_required_size_in_container_elements(other.size()));
        Size = other.size();

        policy_type::copy(
            other.data(),
            other.offset(),
            this->data(),
            this->offset(),
            other.size()
        );
    }

    // Clone a current bitset (Rust style).
    CustomBitsetOwning clone() const {
        CustomBitsetOwning cloned;
        cloned.Data = Data;
        cloned.Size = Size;
        return cloned;
    }

    // Rust style.
    inline container_type into() && {
        return std::move(this->Data);
    }

    // Resize.
    void resize(const size_type new_size) {
        const size_type new_size_in_container_elements = 
            get_required_size_in_container_elements(new_size);
        Data.resize(new_size_in_container_elements);
        Size = new_size;
    }

    // Resize and initialize new bits with a given value if grown. 
    void resize(const size_type new_size, const bool init) {
        const size_type old_size = this->size();
        this->resize(new_size);

        if (new_size > old_size) {
            policy_type::fill(this->data(), old_size, new_size - old_size, init);
        }
    }

    // Append data from another bitset / bitset view in 
    //   [starting_bit_idx, starting_bit_idx + count) range
    //   to the end of this bitset.
    template<typename I, bool R>
    void append(const CustomBitsetBase<PolicyT, I, R>& other, const size_type starting_bit_idx, const size_type count) {
        range_checker::le(starting_bit_idx, other.size());
        
        const size_type old_size = this->size();
        this->resize(this->size() + count);

        policy_type::copy(
            other.data(),
            other.offset() + starting_bit_idx,
            this->data(),
            this->offset() + old_size,
            count
        );
    }

    // Append data from another bitset / bitset view
    //   to the end of this bitset.
    template<typename I, bool R>
    void append(const CustomBitsetBase<PolicyT, I, R>& other) {
        this->append(
            other,
            0,
            other.size()
        );
    }

    // Make bitset empty.
    inline void clear() {
        Data.clear();
        Size = 0;
    }

    // Reserve
    inline void reserve(const size_type capacity) {
        const size_type capacity_in_container_elements = 
            get_required_size_in_container_elements(capacity);
        Data.reserve(capacity_in_container_elements);
    }

    // Return a new bitset, equal to a | b
    template<typename I1, bool R1, typename I2, bool R2>
    friend CustomBitsetOwning operator|(
        const CustomBitsetBase<PolicyT, I1, R1>& a, 
        const CustomBitsetBase<PolicyT, I2, R2>& b
    ) {
        CustomBitsetOwning clone(a);
        return std::move(clone |= b);
    }

    // Return a new bitset, equal to a - b
    template<typename I1, bool R1, typename I2, bool R2>
    friend CustomBitsetOwning operator-(
        const CustomBitsetBase<PolicyT, I1, R1>& a, 
        const CustomBitsetBase<PolicyT, I2, R2>& b
    ) {
        CustomBitsetOwning clone(a);
        return std::move(clone -= b);
    }

protected:
    // the container
    container_type Data;
    // the actual number of bits
    size_type Size = 0;

    inline data_type* data_impl() { return reinterpret_cast<data_type*>(Data.data()); }
    inline const data_type* data_impl() const { return reinterpret_cast<const data_type*>(Data.data()); }
    inline size_type size_impl() const { return Size; }
    inline size_type offset_impl() const { return 0; }

    //
    static inline size_type get_required_size_in_container_elements(const size_t size) {
        const size_type size_in_bytes = policy_type::get_required_size_in_bytes(size);
        return (size_in_bytes + sizeof(container_data_type) - 1) / sizeof(container_data_type);
    }
};

}
}
