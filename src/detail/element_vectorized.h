#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "proxy.h"
#include "element_wise.h"

namespace milvus {
namespace bitset {
namespace detail {

//
template<typename ElementT, typename VectorizedT>
struct CustomBitsetVectorizedPolicy {
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

    static inline void set(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        CustomBitsetPolicy2<ElementT>::set(data, start, size);
    }

    static inline void reset(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        CustomBitsetPolicy2<ElementT>::reset(data, start, size);
    }

    static void fill(
        data_type* const data,
        const size_type start,
        const size_type size,
        const bool value 
    ) {
        CustomBitsetPolicy2<ElementT>::fill(data, start, size, value);
    }

    //
    static inline std::optional<size_type> find(
        const data_type* const data, 
        const size_type start, 
        const size_type size,
        const size_type starting_idx
    ) {
        return CustomBitsetPolicy2<ElementT>::find(data, start, size, starting_idx);
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
        if (size == 0) {
            return;
        }

        auto start_element = get_element(start);
        const auto end_element = get_element(start + size);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + size);

        // same element?
        if (start_element == end_element) {
            CustomBitsetPolicy2<ElementT>::template op_compare_column<T, U, Op>(
                data, start, t, u, size
            );

            return;
        }

        //
        uintptr_t ptr_offset = 0;

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            CustomBitsetPolicy2<ElementT>::template op_compare_column<T, U, Op>(
                data, start, t, u, data_bits - start_shift
            );

            //
            start_element += 1;
            ptr_offset += data_bits - start_shift;
        }

        // process the middle
        {
            const size_t starting_bit_idx = start_element * data_bits;
            const size_t nbits = (end_element - start_element) * data_bits;

            if (!VectorizedT::template op_compare_column<T, U, Op>(
                    reinterpret_cast<uint8_t*>(data + start_element),
                    t + ptr_offset,
                    u + ptr_offset,
                    nbits)
            ) {
                // vectorized implementation is not available, invoke the default one
                CustomBitsetPolicy2<ElementT>::template op_compare_column<T, U, Op>(
                    data, 
                    start_element * data_bits, 
                    t + ptr_offset,
                    u + ptr_offset,
                    nbits 
                );
            }
        
            //
            ptr_offset += nbits;
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            const size_t starting_bit_idx = end_element * data_bits; 

            CustomBitsetPolicy2<ElementT>::template op_compare_column<T, U, Op>(
                data, 
                starting_bit_idx, 
                t + ptr_offset, 
                u + ptr_offset, 
                end_shift
            );
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
        if (size == 0) {
            return;
        }

        auto start_element = get_element(start);
        const auto end_element = get_element(start + size);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + size);

        // same element?
        if (start_element == end_element) {
            CustomBitsetPolicy2<ElementT>::template op_compare_val<T, Op>(
                data, start, t, size, value
            );

            return;
        }

        //
        uintptr_t ptr_offset = 0;

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            CustomBitsetPolicy2<ElementT>::template op_compare_val<T, Op>(
                data, start, t, data_bits - start_shift, value
            );

            //
            start_element += 1;
            ptr_offset += data_bits - start_shift;
        }

        // process the middle
        {
            const size_t starting_bit_idx = start_element * data_bits;
            const size_t nbits = (end_element - start_element) * data_bits;

            if (!VectorizedT::template op_compare_val<T, Op>(
                    reinterpret_cast<uint8_t*>(data + start_element),
                    t + ptr_offset,
                    nbits,
                    value)
            ) {
                // vectorized implementation is not available, invoke the default one
                CustomBitsetPolicy2<ElementT>::template op_compare_val<T, Op>(
                    data, 
                    start_element * data_bits, 
                    t + ptr_offset,
                    nbits,
                    value
                );
            }
        
            //
            ptr_offset += nbits;
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            const size_t starting_bit_idx = end_element * data_bits; 

            CustomBitsetPolicy2<ElementT>::template op_compare_val<T, Op>(
                data, 
                starting_bit_idx, 
                t + ptr_offset,
                end_shift,
                value
            );
        }
    }

    //
    template<typename T, RangeType Op>
    static inline void op_within_range(
        data_type* const __restrict data, 
        const size_type start,
        const T* const __restrict lower,
        const T* const __restrict upper,
        const T* const __restrict values,
        const size_type size
    ) {
        if (size == 0) {
            return;
        }

        auto start_element = get_element(start);
        const auto end_element = get_element(start + size);

        const auto start_shift = get_shift(start);
        const auto end_shift = get_shift(start + size);

        // same element?
        if (start_element == end_element) {
            CustomBitsetPolicy2<ElementT>::template op_within_range<T, Op>(
                data, start, lower, upper, values, size
            );

            return;
        }

        //
        uintptr_t ptr_offset = 0;

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            CustomBitsetPolicy2<ElementT>::template op_within_range<T, Op>(
                data, start, lower, upper, values, size
            );

            //
            start_element += 1;
            ptr_offset += data_bits - start_shift;
        }

        // process the middle
        {
            const size_t starting_bit_idx = start_element * data_bits;
            const size_t nbits = (end_element - start_element) * data_bits;

            if (!VectorizedT::template op_within_range<T, Op>(
                    reinterpret_cast<uint8_t*>(data + start_element),
                    lower + ptr_offset,
                    upper + ptr_offset,
                    values + ptr_offset,
                    nbits)
            ) {
                // vectorized implementation is not available, invoke the default one
                CustomBitsetPolicy2<ElementT>::template op_within_range<T, Op>(
                    data, 
                    start,
                    lower + ptr_offset,
                    upper + ptr_offset,
                    values + ptr_offset,
                    nbits
                );
            }
        
            //
            ptr_offset += nbits;
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            const size_t starting_bit_idx = end_element * data_bits; 

            CustomBitsetPolicy2<ElementT>::template op_within_range<T, Op>(
                data, 
                starting_bit_idx,
                lower + ptr_offset,
                upper + ptr_offset,
                values + ptr_offset,
                end_shift
            );
        }
    }
};


}
}
}
