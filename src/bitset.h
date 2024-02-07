#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace milvus {
namespace bitset {

//
template<bool RangeCheck>
struct RangeChecker{};

template<>
struct RangeChecker<false> {
    template<typename SizeT>
    static inline void lt(const SizeT idx, const SizeT max) {}

    template<typename SizeT>
    static inline void le(const SizeT idx, const SizeT max) {}

    template<typename SizeT>
    static inline void eq(const SizeT a, const SizeT b) {}
};

template<>
struct RangeChecker<true> {
    template<typename SizeT>
    static inline void lt(const SizeT idx, const SizeT max) {
        assert(idx < max);
    }

    template<typename SizeT>
    static inline void le(const SizeT idx, const SizeT max) {
        assert(idx <= max);
    }

    template<typename SizeT>
    static inline void eq(const SizeT a, const SizeT b) {
        assert(a == b);
    }
};

//
template<typename ElementT>
struct CustomBitsetPolicy {
    using data_type = ElementT;
    constexpr static auto data_bits = sizeof(data_type) * 8;

    using size_type = size_t;

    struct ConstProxy {
        using parent_type = CustomBitsetPolicy;
        using size_type = parent_type::size_type;
        using data_type = parent_type::data_type;
        using self_type = ConstProxy;

        const data_type& element;
        data_type mask;

        inline ConstProxy(const data_type& _element, const size_type _shift) : 
            element{_element}
        {
            mask = (data_type(1) << _shift);
        } 

        inline operator bool() const { return ((element & mask) != 0); }
        inline bool operator~() const { return ((element & mask) == 0); }
    };

    struct Proxy {
        using parent_type = CustomBitsetPolicy;
        using size_type = parent_type::size_type;
        using data_type = parent_type::data_type;
        using self_type = Proxy;

        data_type& element;
        data_type mask;

        inline Proxy(data_type& _element, const size_type _shift) : 
            element{_element}
        {
            mask = (data_type(1) << _shift);
        } 

        inline operator bool() const { return ((element & mask) != 0); }
        inline bool operator~() const { return ((element & mask) == 0); }

        inline self_type& operator=(const bool value) {
            if (value) { set(); } else { reset(); }
            return *this;
        }

        inline self_type& operator=(const self_type& other) {
            bool value = other.operator bool();
            if (value) { set(); } else { reset(); }
            return *this;
        }

        inline self_type& operator|=(const bool value) {
            if (value) { set(); }
            return *this;
        }

        inline self_type& operator&=(const bool value) {
            if (!value) { reset(); }
            return *this;
        }

        inline self_type& operator^=(const bool value) {
            if (value) { flip(); }
            return *this;
        }

        inline void set() {
            element |= mask;
        }

        inline void reset() {
            element &= ~mask;
        }

        inline void flip() {
            element ^= mask;
        }
    };

    using proxy_type = Proxy;
    using const_proxy_type = ConstProxy;

    static inline size_type get_element(const size_t idx) {
        return idx / data_bits;
    }

    static inline size_type get_shift(const size_t idx) {
        return idx % data_bits;
    }

    static inline size_type get_required_size_in_elements(const size_t size) {
        return (size + data_bits - 1) / data_bits;
    }

    static inline size_type get_required_size_in_bytes(const size_t size) {
        return get_required_size_in_elements(size) * sizeof(data_type);
    }

    static inline proxy_type get_proxy(
        data_type* const __restrict data, 
        const size_type idx
    ) {
        data_type& element = data[get_element(idx)];
        const size_type shift = get_shift(idx);
        return proxy_type{element, shift};
    }

    static inline const_proxy_type get_proxy(
        const data_type* const __restrict data, 
        const size_type idx
    ) {
        const data_type& element = data[get_element(idx)];
        const size_type shift = get_shift(idx);
        return const_proxy_type{element, shift};
    }

    static inline data_type read(
        const data_type* const data,
        const size_type start,
        const size_type nbits
    ) {
        data_type value = 0;
        for (size_type i = 0; i < nbits; i++) {
            const auto proxy = get_proxy(data, start + i);
            value += proxy ? (data_type(1) << i) : 0;
        }

        return value;
    }

    static void write(
        data_type* const data,
        const size_type start,
        const size_type nbits,
        const data_type value
    ) {
        for (size_type i = 0; i < nbits; i++) {
            auto proxy = get_proxy(data, start + i);
            data_type mask = data_type(1) << i;
            if ((value & mask) == mask) {
                proxy = true;
            }
            else {
                proxy = false;
            }
        }
    }

    static inline void op_flip(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        for (size_type i = 0; i < size; i++) {
            auto proxy = get_proxy(data, start + i);
            proxy.flip();
        }
    }

    static inline void op_and(
        data_type* left, 
        const data_type* right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        // todo: check if intersect

        for (size_type i = 0; i < size; i++) {
            auto proxy_left = get_proxy(left, start_left + i);
            auto proxy_right = get_proxy(right, start_right + i);

            proxy_left &= proxy_right;
        }
    }

    static inline void op_or(
        data_type* left, 
        const data_type* right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        // todo: check if intersect

        for (size_type i = 0; i < size; i++) {
            auto proxy_left = get_proxy(left, start_left + i);
            auto proxy_right = get_proxy(right, start_right + i);

            proxy_left |= proxy_right;
        }
    }

    static inline void set(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        for (size_type i = 0; i < size; i++) {
            get_proxy(data, start + i) = true;
        }
    }

    static inline void reset(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        for (size_type i = 0; i < size; i++) {
            get_proxy(data, start + i) = false;
        }
    }

    static inline bool all(
        const data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        for (size_type i = 0; i < size; i++) {
            if (!get_proxy(data, start + i)) {
                return false;
            }
        }

        return true;
    }

    static inline bool none(
        const data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        for (size_type i = 0; i < size; i++) {
            if (get_proxy(data, start + i)) {
                return false;
            }
        }

        return true;
    }

    static void copy(
        const data_type* const src,
        const size_type start_src,
        data_type* const dst,
        const size_type start_dst,
        const size_type size
    ) {
        for (size_type i = 0; i < size; i++) {
            const auto src_p = get_proxy(src, start_src + i);
            auto dst_p = get_proxy(dst, start_dst + i);
            dst_p = src_p.operator bool();
        }
    }

    static void fill(
        data_type* const dst,
        const size_type start_dst,
        const size_type size,
        const bool value 
    ) {
        for (size_type i = 0; i < size; i++) {
            auto dst_p = get_proxy(dst, start_dst + i);
            dst_p = value;
        }
    }

    static inline size_type op_count(
        const data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        size_type count = 0;

        for (size_type i = 0; i < size; i++) {
            auto proxy = get_proxy(data, start + i);
            count += (proxy) ? 1 : 0;
        }

        return count;
    }

    static inline bool op_eq(
        const data_type* left, 
        const data_type* right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        for (size_type i = 0; i < size; i++) {
            const auto proxy_left = get_proxy(left, start_left + i);
            const auto proxy_right = get_proxy(right, start_right + i);

            if (proxy_left != proxy_right) {
                return false;
            }
        }

        return true;
    }

    static inline void op_xor(
        data_type* left, 
        const data_type* right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        // todo: check if intersect

        for (size_type i = 0; i < size; i++) {
            auto proxy_left = get_proxy(left, start_left + i);
            const auto proxy_right = get_proxy(right, start_right + i);

            proxy_left ^= proxy_right;
        }
    }

    static inline void op_sub(
        data_type* left, 
        const data_type* right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        // todo: check if intersect

        for (size_type i = 0; i < size; i++) {
            auto proxy_left = get_proxy(left, start_left + i);
            const auto proxy_right = get_proxy(right, start_right + i);

            proxy_left &= ~proxy_right;
        }
    }

    //
    static constexpr size_type npos = size_type(-1);

    static inline size_type find(
        const data_type* const data, 
        const size_type start, 
        const size_type size,
        const size_type starting_idx
    ) {
        for (size_type i = starting_idx; i < size; i++) {
            const auto proxy = get_proxy(data, start + i);
            if (proxy) {
                return start + i;
            }
        }

        return npos;
    }
};

// CRTP
template<typename PolicyT, bool IsRangeCheckEnabled>
class CustomBitsetNonOwning;

template<typename PolicyT, typename ContainerT, bool IsRangeCheckEnabled>
class CustomBitsetOwning;

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

    //
    inline size_type size() const {
        return as_derived().size_impl();
    }

    //
    inline size_type size_in_bytes() const { 
        return policy_type::get_required_size_in_bytes(this->size());
    }

    //
    inline size_type size_in_elements() const { 
        return policy_type::get_required_size_in_elements(this->size());
    }

    //
    inline bool empty() const {
        return (this->size() == 0);
    }

    //
    inline proxy_type operator[](const size_type idx) {
        range_checker::lt(idx, this->size());

        const size_type idx_v = idx + this->offset();
        return policy_type::get_proxy(this->data(), idx_v);
    }

    //
    inline bool operator[](const size_type idx) const {
        range_checker::lt(idx, this->size());

        const size_type idx_v = idx + this->offset();
        const auto proxy = policy_type::get_proxy(this->data(), idx_v);
        return proxy.operator bool();
    }

    //
    inline void set() {
        policy_type::set(this->data(), this->offset(), this->size());
    }

    //
    inline void set(const size_type idx, const bool value = true) {
        range_checker::lt(idx, this->size());

        const size_type idx_v = idx + this->offset();
        auto proxy = policy_type::get_proxy(this->data(), idx_v);
        proxy = value;
    }

    //
    inline void reset() {
        policy_type::reset(this->data(), this->offset(), this->size());
    }

    //
    inline void reset(const size_type idx) {
        range_checker::lt(idx, this->size());

        const size_type idx_v = idx + this->offset();
        auto proxy = policy_type::get_proxy(this->data(), idx_v);
        proxy = false;
    }

    //
    inline bool all() const {
        return policy_type::all(this->data(), this->offset(), this->size());
    }

    //
    inline bool any() const {
        return (!this->none());
    }

    //
    inline bool none() const {
        return policy_type::none(this->data(), this->offset(), this->size());
    }

    //
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

    //
    template<typename I, bool R>
    inline ImplT& operator&=(const CustomBitsetBase<PolicyT, I, R>& other) {
        this->inplace_and(other, this->size());
        return as_derived();
    }

    //
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

    //
    template<typename I, bool R>
    inline ImplT& operator|=(const CustomBitsetBase<PolicyT, I, R>& other) {
        this->inplace_or(other, this->size());
        return as_derived();
    }

    //
    inline void flip() {
        policy_type::op_flip(this->data(), this->offset(), this->size());
    }

    //
    inline CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> operator+(const size_type offset) {
        return this->view(offset);
    }

    //
    inline CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view(const size_type offset, const size_type size) {
        range_checker::le(offset, this->size());
        range_checker::le(offset + size, this->size());

        return CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled>(
            this->data(),
            this->offset() + offset,
            size
        );
    }

    //
    inline CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view(const size_type offset) {
        range_checker::le(offset, this->size());

        return CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled>(
            this->data(),
            this->offset() + offset,
            this->size() - offset
        );
    }

    inline const CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view(const size_type offset) const {
        range_checker::le(offset, this->size());

        return CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled>(
            const_cast<data_type*>(this->data()),
            this->offset() + offset,
            this->size() - offset
        );
    }

    //
    inline CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view() {
        return this->view(0);
    }

    //
    inline const CustomBitsetNonOwning<PolicyT, IsRangeCheckEnabled> view() const {
        return this->view(0);
    }

    //
    inline size_type count() const {
        return policy_type::op_count(this->data(), this->offset(), this->size());
    }

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

    template<typename I, bool R>
    inline bool operator!=(const CustomBitsetBase<PolicyT, I, R>& other) {
        return (!(*this == other));
    }

    template<typename I, bool R>
    inline void inplace_xor(const CustomBitsetBase<PolicyT, I, R>& other) {
        range_checker::eq(other.size(), this->size());
    
        policy_type::op_xor(
            this->data(),
            other.data(),
            this->offset(),
            other.offset(),
            this->size()
        );
    }

    template<typename I, bool R>
    inline ImplT& operator^=(const CustomBitsetBase<PolicyT, I, R>& other) {
        this->inplace_xor(other);
        return as_derived();
    }

    template<typename I, bool R>
    inline void inplace_sub(const CustomBitsetBase<PolicyT, I, R>& other) {
        range_checker::eq(other.size(), this->size());
    
        policy_type::op_sub(
            this->data(),
            other.data(),
            this->offset(),
            other.offset(),
            this->size()
        );
    }

    template<typename I, bool R>
    inline ImplT& operator-=(const CustomBitsetBase<PolicyT, I, R>& other) {
        this->inplace_sub(other);
        return as_derived();
    }

    //
    static constexpr size_type npos = policy_type::npos;

    inline size_type find_first() const {
        return policy_type::find(this->data(), this->offset(), this->size(), 0);
    }

    inline size_type find_next(const size_type starting_index) const {
        const size_type size_v = this->size();
        if (starting_index >= size_v) {
            return npos;
        }

        return policy_type::find(this->data(), this->offset(), this->size(), starting_index + 1);
    }

    //
    inline data_type read(
        const size_type idx,
        const size_type nbits
    ) {
        return policy_type::read(
            this->data(),
            this->offset() + idx,
            nbits
        );
    }

    inline void write(
        const size_type idx,
        const size_type value,
        const size_type nbits
    ) {
        policy_type::write(
            this->data(),
            this->offset() + idx,
            nbits,
            value
        );
    }

private:
    //
    inline size_type offset() const {
        return as_derived().offset_impl();
    }

private:
    //
    inline ImplT& as_derived() {
        return static_cast<ImplT&>(*this);
    }

    //
    inline const ImplT& as_derived() const {
        return static_cast<const ImplT&>(*this);
    }
};

//
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

//
template<typename PolicyT, typename ContainerT, bool IsRangeCheckEnabled>
class CustomBitsetOwning : public CustomBitsetBase<PolicyT, CustomBitsetOwning<PolicyT, ContainerT, IsRangeCheckEnabled>, IsRangeCheckEnabled> {
    friend class CustomBitsetBase<PolicyT, CustomBitsetOwning<PolicyT, ContainerT, IsRangeCheckEnabled>, IsRangeCheckEnabled>;

public:
    using policy_type = PolicyT;
    using data_type = typename policy_type::data_type;
    using size_type = typename policy_type::size_type;
    using proxy_type = typename policy_type::proxy_type;
    using const_proxy_type = typename policy_type::const_proxy_type;

    // this is the container
    using container_type = ContainerT;
    // this is how we store the data. For example, we may operate using
    //   uint64_t values, but store the data in std::vector<uint8_t> container.
    using container_data_type = typename container_type::value_type;

    using range_checker = RangeChecker<IsRangeCheckEnabled>;

    // allocate an empty one
    CustomBitsetOwning() {}
    // allocate the given number of bits
    CustomBitsetOwning(const size_type size) : 
        Data(get_required_size_in_container_elements(size)), Size{size} {}
    // allocate the given number of bits being filled with data
    CustomBitsetOwning(const size_type size, const bool init) : 
        Data(
            get_required_size_in_container_elements(size), 
            init ? data_type(-1) : 0),
        Size{size} {}
    CustomBitsetOwning(const CustomBitsetOwning &) = delete;
    CustomBitsetOwning(CustomBitsetOwning&&) = default;
    CustomBitsetOwning& operator =(const CustomBitsetOwning&) = delete;
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

    // rust style
    CustomBitsetOwning clone() const {
        CustomBitsetOwning cloned;
        cloned.Data = Data;
        cloned.Size = Size;
        return cloned;
    }

    //
    inline container_type into() && {
        return std::move(this->Data);
    }

    //
    void resize(const size_type new_size) {
        const size_type new_size_in_container_elements = 
            get_required_size_in_container_elements(new_size);
        Data.resize(new_size_in_container_elements);
        Size = new_size;
    }

    //
    void resize(const size_type new_size, const bool init) {
        const size_type old_size = this->size();
        this->resize(new_size);

        if (new_size > old_size) {
            policy_type::fill(this->data(), old_size, new_size - old_size, init);
        }
    }

    //
    template<typename I, bool R>
    void append(const CustomBitsetBase<PolicyT, I, R>& other, const size_type start, const size_type count) {
        // This function appends data from other in [start, start + count) range
        // to the end of Data.
        range_checker::le(start, other.size());
        
        const size_type old_size = this->size();
        this->resize(this->size() + count);

        policy_type::copy(
            other.data(),
            other.offset() + start,
            this->data(),
            this->offset() + old_size,
            count
        );
    }

    template<typename I, bool R>
    void append(const CustomBitsetBase<PolicyT, I, R>& other) {
        this->append(
            other,
            0,
            other.size()
        );
    }

    //
    inline void clear() {
        Data.clear();
        Size = 0;
    }

    //
    inline void reserve(const size_type capacity) {
        const size_type capacity_in_container_elements = 
            get_required_size_in_container_elements(capacity);
        Data.reserve(capacity_in_container_elements);
    }

    //
    template<typename I1, bool R1, typename I2, bool R2>
    friend CustomBitsetOwning operator|(
        const CustomBitsetBase<PolicyT, I1, R1>& a, 
        const CustomBitsetBase<PolicyT, I2, R2>& b
    ) {
        CustomBitsetOwning clone(a);
        return std::move(clone |= b);
    }

    //
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
