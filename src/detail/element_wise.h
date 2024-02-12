#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "proxy.h"

#include "ctz.h"
#include "popcount.h"

namespace milvus {
namespace bitset {
namespace detail {

//
template<typename ElementT>
struct CustomBitsetPolicy2 {
    using data_type = ElementT;
    constexpr static auto data_bits = sizeof(data_type) * 8;

    using size_type = size_t;

    using self_type = CustomBitsetPolicy<ElementT>;

    using proxy_type = Proxy<self_type>;
    using const_proxy_type = ConstProxy<self_type>;

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
        if (nbits == 0) [[unlikely]] {
            return 0;
        }

        const auto start_element = get_element(start);
        const auto end_element = get_element(start + nbits - 1);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + nbits - 1);

        if (start_element == end_element) {
            // read from 1 element only
            const data_type m1 = get_shift_mask_end(start_shift);
            const data_type m2 = get_shift_mask_begin(end_shift + 1);
            const data_type mask = 
                get_shift_mask_end(start_shift) & get_shift_mask_begin(end_shift + 1);

            // read and shift
            const data_type element = data[start_element];
            const data_type value = (element & mask) >> start_shift;
            return value;
        }
        else {
            // read from 2 elements
            const data_type first_v = data[start_element];
            const data_type second_v = data[start_element + 1];

            const data_type first_mask = get_shift_mask_end(start_shift);
            const data_type second_mask = get_shift_mask_begin(end_shift + 1);
            
            const data_type value1 = (first_v & first_mask) >> start_shift;
            const data_type value2 = (second_v & second_mask);
            const data_type value = value1 | (value2 << (data_bits - start_shift));
            
            return value;
        }
    }

    static inline void write(
        data_type* const data,
        const size_type start,
        const size_type nbits,
        const data_type value
    ) {
        if (nbits == 0) [[unlikely]] {
            return;
        }

        const auto start_element = get_element(start);
        const auto end_element = get_element(start + nbits - 1);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + nbits - 1);

        if (start_element == end_element) {
            // write into a single element

            const data_type m1 = get_shift_mask_end(start_shift);
            const data_type m2 = get_shift_mask_begin(end_shift + 1);
            const data_type mask = 
                get_shift_mask_end(start_shift) & get_shift_mask_begin(end_shift + 1);

            // read an existing value
            const data_type element = data[start_element];
            // combine a new value
            const data_type new_value = 
                (element & (~mask)) |
                ((value << start_shift) & mask);
            // write it back
            data[start_element] = new_value;
        }
        else {
            // write into two elements
            const data_type first_v = data[start_element];
            const data_type second_v = data[start_element + 1];

            const data_type first_mask = get_shift_mask_end(start_shift);
            const data_type second_mask = get_shift_mask_begin(end_shift + 1);

            const data_type value1 = 
                (first_v & (~first_mask)) |
                ((value << start_shift) & first_mask);
            const data_type value2 = 
                (second_v & (~second_mask)) |
                ((value >> (data_bits - start_shift)) & second_mask);

            data[start_element] = value1;
            data[start_element + 1] = value2;
        }
    }

    static inline void op_flip(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        if (size == 0) [[unlikely]] {
            return;
        }

        //
        auto start_element = get_element(start);
        const auto end_element = get_element(start + size);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + size);

        // same element to modify?
        if (start_element == end_element) [[unlikely]] {
            const data_type existing_v = data[start_element];
            const data_type new_v = ~existing_v;

            const data_type existing_mask = 
                get_shift_mask_begin(start_shift) | get_shift_mask_end(end_shift);
            const data_type new_mask =
                get_shift_mask_end(start_shift) & get_shift_mask_begin(end_shift);

            data[start_element] = (existing_v & existing_mask) | (new_v & new_mask);
            return;
        }

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            const data_type existing_v = data[start_element];
            const data_type new_v = ~existing_v;

            const data_type existing_mask = get_shift_mask_begin(start_shift);
            const data_type new_mask = get_shift_mask_end(start_shift);

            data[start_element] = (existing_v & existing_mask) | (new_v & new_mask);
            start_element += 1;
        }

        // process the middle
        for (size_type i = start_element; i < end_element; i++) {
            data[i] = ~data[i];
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            const data_type existing_v = data[end_element];
            const data_type new_v = ~existing_v;

            const data_type existing_mask = get_shift_mask_end(end_shift);
            const data_type new_mask = get_shift_mask_begin(end_shift);

            data[end_element] = (existing_v & existing_mask) | (new_v & new_mask);
        }
    }

    static inline void op_and(
        data_type* left, 
        const data_type* right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        if (size == 0) [[unlikely]] {
            return;
        }

        // process big blocks
        const size_type size_b = (size / data_bits) * data_bits;
        for (size_type i = 0; i < size_b; i += data_bits) {
            data_type left_v = read(left, start_left + i, data_bits);
            const data_type right_v = read(right, start_right + i, data_bits);

            left_v &= right_v;
            
            write(left, start_left + i, data_bits, left_v);
        }

        // process leftovers
        if (size_b != size) {
            data_type left_v = read(left, start_left + size_b, size - size_b);
            const data_type right_v = read(right, start_right + size_b, size - size_b);

            left_v &= right_v;

            write(left, start_left + size_b, size - size_b, left_v);
        }
    }

    static inline void op_or(
        data_type* left, 
        const data_type* right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        if (size == 0) [[unlikely]] {
            return;
        }

        // process big blocks
        const size_type size_b = (size / data_bits) * data_bits;
        for (size_type i = 0; i < size_b; i += data_bits) {
            data_type left_v = read(left, start_left + i, data_bits);
            const data_type right_v = read(right, start_right + i, data_bits);

            left_v |= right_v;
            
            write(left, start_left + i, data_bits, left_v);
        }

        // process leftovers
        if (size_b != size) {
            data_type left_v = read(left, start_left + size_b, size - size_b);
            const data_type right_v = read(right, start_right + size_b, size - size_b);

            left_v |= right_v;

            write(left, start_left + size_b, size - size_b, left_v);
        }
    }

    static inline data_type get_shift_mask_begin(const size_type shift) {
        // 0 -> 0b00000000
        // 1 -> 0b00000001
        // 2 -> 0b00000011
        if (shift == data_bits) {
            return data_type(-1);
        }

        return (data_type(1) << shift) - data_type(1);
    }

    static inline data_type get_shift_mask_end(const size_type shift) {
        // 0 -> 0b11111111
        // 1 -> 0b11111110
        // 2 -> 0b11111100
        return ~(get_shift_mask_begin(shift));
    }

    static inline void set(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        fill(data, start, size, true);
    }

    static inline void reset(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        fill(data, start, size, false);
    }

    static inline bool all(
        const data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        if (size == 0) [[unlikely]] {
            return true;
        }

        auto start_element = get_element(start);
        const auto end_element = get_element(start + size);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + size);

        // same element?
        if (start_element == end_element) [[unlikely]] {
            const data_type existing_v = data[start_element];

            const data_type existing_mask =
                get_shift_mask_end(start_shift) & get_shift_mask_begin(end_shift);

            return ((existing_v & existing_mask) == existing_mask);
        }

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            const data_type existing_v = data[start_element];

            const data_type existing_mask = get_shift_mask_end(start_shift);
            if ((existing_v & existing_mask) != existing_mask) {
                return false;
            }

            start_element += 1;
        }

        // process the middle
        for (size_type i = start_element; i < end_element; i++) {
            if (data[i] != data_type(-1)) {
                return false;
            }
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            const data_type existing_v = data[end_element];

            const data_type existing_mask = get_shift_mask_begin(end_shift);

            if ((existing_v & existing_mask) != existing_mask) {
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
        if (size == 0) [[unlikely]] {
            return true;
        }

        auto start_element = get_element(start);
        const auto end_element = get_element(start + size);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + size);

        // same element?
        if (start_element == end_element) [[unlikely]] {
            const data_type existing_v = data[start_element];

            const data_type existing_mask =
                get_shift_mask_end(start_shift) & get_shift_mask_begin(end_shift);

            return ((existing_v & existing_mask) == data_type(0));
        }

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            const data_type existing_v = data[start_element];

            const data_type existing_mask = get_shift_mask_end(start_shift);
            if ((existing_v & existing_mask) != data_type(0)) {
                return false;
            }

            start_element += 1;
        }

        // process the middle
        for (size_type i = start_element; i < end_element; i++) {
            if (data[i] != data_type(0)) {
                return false;
            }
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            const data_type existing_v = data[end_element];

            const data_type existing_mask = get_shift_mask_begin(end_shift);

            if ((existing_v & existing_mask) != data_type(0)) {
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
        if (size == 0) [[unlikely]] {
            return;
        }

        // process big blocks
        const size_type size_b = (size / data_bits) * data_bits;
        for (size_type i = 0; i < size_b; i += data_bits) {
            const data_type src_v = read(src, start_src + i, data_bits);
            write(dst, start_dst + i, data_bits, src_v);
        }

        // process leftovers
        if (size_b != size) {
            const data_type src_v = read(src, start_src + size_b, size - size_b);
            write(dst, start_dst + size_b, size - size_b, src_v);
        }
    }

    static void fill(
        data_type* const data,
        const size_type start,
        const size_type size,
        const bool value 
    ) {
        if (size == 0) [[unlikely]] {
            return;
        }

        const data_type new_v = (value) ? data_type(-1) : data_type(0); 

        //
        auto start_element = get_element(start);
        const auto end_element = get_element(start + size);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + size);

        // same element to modify?
        if (start_element == end_element) [[unlikely]] {
            const data_type existing_v = data[start_element];

            const data_type existing_mask = 
                get_shift_mask_begin(start_shift) | get_shift_mask_end(end_shift);
            const data_type new_mask =
                get_shift_mask_end(start_shift) & get_shift_mask_begin(end_shift);

            data[start_element] = (existing_v & existing_mask) | (new_v & new_mask);
            return;
        }

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            const data_type existing_v = data[start_element];

            const data_type existing_mask = get_shift_mask_begin(start_shift);
            const data_type new_mask = get_shift_mask_end(start_shift);

            data[start_element] = (existing_v & existing_mask) | (new_v & new_mask);
            start_element += 1;
        }

        // process the middle
        for (size_type i = start_element; i < end_element; i++) {
            data[i] = new_v;
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            const data_type existing_v = data[end_element];

            const data_type existing_mask = get_shift_mask_end(end_shift);
            const data_type new_mask = get_shift_mask_begin(end_shift);

            data[end_element] = (existing_v & existing_mask) | (new_v & new_mask);
        }
    }

    static inline size_type op_count(
        const data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        if (size == 0) [[unlikely]] {
            return 0;
        }

        size_type count = 0;

        auto start_element = get_element(start);
        const auto end_element = get_element(start + size);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + size);

        // same element?
        if (start_element == end_element) [[unlikely]] {
            const data_type existing_v = data[start_element];

            const data_type existing_mask =
                get_shift_mask_end(start_shift) & get_shift_mask_begin(end_shift);

            return PopCountHelper<size_type>::count(existing_v & existing_mask);
        }

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            const data_type existing_v = data[start_element];
            const data_type existing_mask = get_shift_mask_end(start_shift);

            count = PopCountHelper<size_type>::count(existing_v & existing_mask);

            start_element += 1;
        }

        // process the middle
        for (size_type i = start_element; i < end_element; i++) {
            count += PopCountHelper<size_type>::count(data[i]);
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            const data_type existing_v = data[end_element];
            const data_type existing_mask = get_shift_mask_begin(end_shift);

            count += PopCountHelper<size_type>::count(existing_v & existing_mask);
        }

        return count;
    }

    static inline bool op_eq(
        const data_type* left, 
        const data_type* right, 
        const size_type start_left,
        const size_type start_right, 
        const size_type size
    ) {
        if (size == 0) [[unlikely]] {
            return true;
        }

        //
        const size_type size_b = (size / data_bits) * data_bits;
        for (size_type i = 0; i < size_b; i += data_bits) {
            const data_type left_v = read(left, start_left + i, data_bits);
            const data_type right_v = read(right, start_right + i, data_bits);
            if (left_v != right_v) {
                return false;
            }
        }

        if (size_b != size) {
            const data_type left_v = read(left, start_left + size_b, size - size_b);
            const data_type right_v = read(right, start_right + size_b, size - size_b);            
            if (left_v != right_v) {
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
        if (size == 0) [[unlikely]] {
            return;
        }

        // process big blocks
        const size_type size_b = (size / data_bits) * data_bits;
        for (size_type i = 0; i < size_b; i += data_bits) {
            data_type left_v = read(left, start_left + i, data_bits);
            const data_type right_v = read(right, start_right + i, data_bits);

            left_v ^= right_v;
            
            write(left, start_left + i, data_bits, left_v);
        }

        // process leftovers
        if (size_b != size) {
            data_type left_v = read(left, start_left + size_b, size - size_b);
            const data_type right_v = read(right, start_right + size_b, size - size_b);

            left_v ^= right_v;

            write(left, start_left + size_b, size - size_b, left_v);
        }
    }

    static inline void op_sub(
        data_type* left, 
        const data_type* right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        if (size == 0) [[unlikely]] {
            return;
        }

        // process big blocks
        const size_type size_b = (size / data_bits) * data_bits;
        for (size_type i = 0; i < size_b; i += data_bits) {
            data_type left_v = read(left, start_left + i, data_bits);
            const data_type right_v = read(right, start_right + i, data_bits);

            left_v &= ~right_v;
            
            write(left, start_left + i, data_bits, left_v);
        }

        // process leftovers
        if (size_b != size) {
            data_type left_v = read(left, start_left + size_b, size - size_b);
            const data_type right_v = read(right, start_right + size_b, size - size_b);

            left_v &= ~right_v;

            write(left, start_left + size_b, size - size_b, left_v);
        }
    }

    //
    static inline std::optional<size_type> find(
        const data_type* const data, 
        const size_type start, 
        const size_type size,
        const size_type starting_idx
    ) {
        if (size == 0) [[unlikely]] {
            return std::nullopt;
        }

        //
        auto start_element = get_element(start + starting_idx);
        const auto end_element = get_element(start + size);

        const auto start_shift = get_shift(start + starting_idx);
        const auto end_shift = get_shift(start + size);

        size_type extra_offset = 0;

        // same element?
        if (start_element == end_element) {
            const data_type existing_v = data[start_element];

            const data_type existing_mask =
                get_shift_mask_end(start_shift) & get_shift_mask_begin(end_shift);

            const data_type value = existing_v & existing_mask;
            if (value != 0) {
                const auto ctz = CtzHelper<data_type>::ctz(value);
                return size_type(ctz) + start_element * data_bits - start;
            } 
            else {
                return std::nullopt;
            }
        }

        // process the first element
        if (start_shift != 0) [[likely]] {
            const data_type existing_v = data[start_element];
            const data_type existing_mask = get_shift_mask_end(start_shift);

            const data_type value = existing_v & existing_mask;
            if (value != 0) {
                const auto ctz = CtzHelper<data_type>::ctz(value) + start_element * data_bits - start;
                return size_type(ctz);
            }

            start_element += 1;
            extra_offset += data_bits - start_shift;
        }

        // process the middle
        for (size_type i = start_element; i < end_element; i++) {
            const data_type value = data[i];
            if (value != 0) {
                const auto ctz = CtzHelper<data_type>::ctz(value);
                return size_type(ctz) + i * data_bits - start;
            }
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            const data_type existing_v = data[end_element];
            const data_type existing_mask = get_shift_mask_begin(end_shift);

            const data_type value = existing_v & existing_mask;
            if (value != 0) {
                const auto ctz = CtzHelper<data_type>::ctz(value);
                return size_type(ctz) + end_element * data_bits - start;
            }
        }

        return std::nullopt;
    }

    //
    template<typename T, typename U, CompareType Op>
    static inline void op_compare_column(
        data_type* const __restrict data, 
        const size_type start,
        const T* const __restrict t,
        const U* const __restrict u,
        const size_type size
    ) {
        for (size_type i = 0; i < size; i++) {
            get_proxy(data, start + i) = CompareOperator<Op>::compare(t[i], u[i]);
        }
    }

    //
    template<typename T, CompareType Op>
    static inline void op_compare_val(
        data_type* const __restrict data, 
        const size_type start,
        const T* const __restrict t,
        const size_type size,
        const T value
    ) {
        for (size_type i = 0; i < size; i++) {
            get_proxy(data, start + i) = CompareOperator<Op>::compare(t[i], value);
        }
    }
};

}
}
}
