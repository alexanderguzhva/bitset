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

// SIMD applied on top of ElementWiseBitsetPolicy
template<typename ElementT, typename VectorizedT>
struct VectorizedElementWiseBitsetPolicy {
    using data_type = ElementT;
    constexpr static auto data_bits = sizeof(data_type) * 8;

    using size_type = size_t;

    using self_type = VectorizedElementWiseBitsetPolicy<ElementT, VectorizedT>;

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

    static inline void op_flip(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        ElementWiseBitsetPolicy<ElementT>::op_flip(data, start, size);
    }

    static inline void op_and(
        data_type* const left, 
        const data_type* const right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        ElementWiseBitsetPolicy<ElementT>::op_and(left, right, start_left, start_right, size);
    }

    static inline void op_or(
        data_type* const left, 
        const data_type* const right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        ElementWiseBitsetPolicy<ElementT>::op_or(left, right, start_left, start_right, size);
    }

    static inline void set(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        ElementWiseBitsetPolicy<ElementT>::set(data, start, size);
    }

    static inline void reset(
        data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        ElementWiseBitsetPolicy<ElementT>::reset(data, start, size);
    }

    static inline bool all(
        const data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        return ElementWiseBitsetPolicy<ElementT>::all(data, start, size);
    }

    static inline bool none(
        const data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        return ElementWiseBitsetPolicy<ElementT>::none(data, start, size);
    }

    static void copy(
        const data_type* const src,
        const size_type start_src,
        data_type* const dst,
        const size_type start_dst,
        const size_type size
    ) {
        ElementWiseBitsetPolicy<ElementT>::copy(src, start_src, dst, start_dst, size);
    }

    static inline size_type op_count(
        const data_type* const data, 
        const size_type start, 
        const size_type size
    ) {
        return ElementWiseBitsetPolicy<ElementT>::op_count(data, start, size);
    }

    static inline bool op_eq(
        const data_type* const left, 
        const data_type* const right, 
        const size_type start_left,
        const size_type start_right, 
        const size_type size
    ) {
        return ElementWiseBitsetPolicy<ElementT>::op_eq(left, right, start_left, start_right, size);
    }

    static inline void op_xor(
        data_type* const left, 
        const data_type* const right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        ElementWiseBitsetPolicy<ElementT>::op_xor(left, right, start_left, start_right, size);
    }

    static inline void op_sub(
        data_type* const left, 
        const data_type* const right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        ElementWiseBitsetPolicy<ElementT>::op_sub(left, right, start_left, start_right, size);
    }

    static void fill(
        data_type* const data,
        const size_type start,
        const size_type size,
        const bool value 
    ) {
        ElementWiseBitsetPolicy<ElementT>::fill(data, start, size, value);
    }

    //
    static inline std::optional<size_type> find(
        const data_type* const data, 
        const size_type start, 
        const size_type size,
        const size_type starting_idx
    ) {
        return ElementWiseBitsetPolicy<ElementT>::find(data, start, size, starting_idx);
    }

    //
    template<typename T, typename U, CompareOpType Op>
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
            ElementWiseBitsetPolicy<ElementT>::template op_compare_column<T, U, Op>(
                data, start, t, u, size
            );

            return;
        }

        //
        uintptr_t ptr_offset = 0;

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            ElementWiseBitsetPolicy<ElementT>::template op_compare_column<T, U, Op>(
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
                ElementWiseBitsetPolicy<ElementT>::template op_compare_column<T, U, Op>(
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

            ElementWiseBitsetPolicy<ElementT>::template op_compare_column<T, U, Op>(
                data, 
                starting_bit_idx, 
                t + ptr_offset, 
                u + ptr_offset, 
                end_shift
            );
        }
    }

    //
    template<typename T, CompareOpType Op>
    static inline void op_compare_val(
        data_type* const __restrict data, 
        const size_type start,
        const T* const __restrict t,
        const size_type size,
        const T& value
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
            ElementWiseBitsetPolicy<ElementT>::template op_compare_val<T, Op>(
                data, start, t, size, value
            );

            return;
        }

        //
        uintptr_t ptr_offset = 0;

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            ElementWiseBitsetPolicy<ElementT>::template op_compare_val<T, Op>(
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
                ElementWiseBitsetPolicy<ElementT>::template op_compare_val<T, Op>(
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

            ElementWiseBitsetPolicy<ElementT>::template op_compare_val<T, Op>(
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
    static inline void op_within_range_column(
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
            ElementWiseBitsetPolicy<ElementT>::template op_within_range_column<T, Op>(
                data, start, lower, upper, values, size
            );

            return;
        }

        //
        uintptr_t ptr_offset = 0;

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            ElementWiseBitsetPolicy<ElementT>::template op_within_range_column<T, Op>(
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

            if (!VectorizedT::template op_within_range_column<T, Op>(
                    reinterpret_cast<uint8_t*>(data + start_element),
                    lower + ptr_offset,
                    upper + ptr_offset,
                    values + ptr_offset,
                    nbits)
            ) {
                // vectorized implementation is not available, invoke the default one
                ElementWiseBitsetPolicy<ElementT>::template op_within_range_column<T, Op>(
                    data, 
                    starting_bit_idx,
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

            ElementWiseBitsetPolicy<ElementT>::template op_within_range_column<T, Op>(
                data, 
                starting_bit_idx,
                lower + ptr_offset,
                upper + ptr_offset,
                values + ptr_offset,
                end_shift
            );
        }
    }

    //
    template<typename T, RangeType Op>
    static inline void op_within_range_val(
        data_type* const __restrict data, 
        const size_type start,
        const T& lower,
        const T& upper,
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
            ElementWiseBitsetPolicy<ElementT>::template op_within_range_val<T, Op>(
                data, start, lower, upper, values, size
            );

            return;
        }

        //
        uintptr_t ptr_offset = 0;

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            ElementWiseBitsetPolicy<ElementT>::template op_within_range_val<T, Op>(
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

            if (!VectorizedT::template op_within_range_val<T, Op>(
                    reinterpret_cast<uint8_t*>(data + start_element),
                    lower,
                    upper,
                    values + ptr_offset,
                    nbits)
            ) {
                // vectorized implementation is not available, invoke the default one
                ElementWiseBitsetPolicy<ElementT>::template op_within_range_val<T, Op>(
                    data, 
                    starting_bit_idx,
                    lower,
                    upper,
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

            ElementWiseBitsetPolicy<ElementT>::template op_within_range_val<T, Op>(
                data, 
                starting_bit_idx,
                lower,
                upper,
                values + ptr_offset,
                end_shift
            );
        }
    }

    //
    template<typename T, ArithOpType AOp, CompareOpType CmpOp>
    static inline void op_arith_compare(
        data_type* const __restrict data, 
        const size_type start,
        const T* const __restrict src,
        const ArithHighPrecisionType<T>& right_operand,
        const ArithHighPrecisionType<T>& value,
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
            ElementWiseBitsetPolicy<ElementT>::template op_arith_compare<T, AOp, CmpOp>(
                data, start, src, right_operand, value, size
            );

            return;
        }

        //
        uintptr_t ptr_offset = 0;

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            ElementWiseBitsetPolicy<ElementT>::template op_arith_compare<T, AOp, CmpOp>(
                data, start, src, right_operand, value, size
            );

            //
            start_element += 1;
            ptr_offset += data_bits - start_shift;
        }

        // process the middle
        {
            const size_t starting_bit_idx = start_element * data_bits;
            const size_t nbits = (end_element - start_element) * data_bits;

            if (!VectorizedT::template op_arith_compare<T, AOp, CmpOp>(
                    reinterpret_cast<uint8_t*>(data + start_element),
                    src + ptr_offset,
                    right_operand,
                    value,
                    nbits)
            ) {
                // vectorized implementation is not available, invoke the default one
                ElementWiseBitsetPolicy<ElementT>::template op_arith_compare<T, AOp, CmpOp>(
                    data, 
                    starting_bit_idx,
                    src + ptr_offset,
                    right_operand,
                    value,
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

            ElementWiseBitsetPolicy<ElementT>::template op_arith_compare<T, AOp, CmpOp>(
                data, 
                starting_bit_idx,
                src + ptr_offset,
                right_operand,
                value,
                end_shift
            );
        }
    }

    //
    static inline size_t op_and_with_count(
        data_type* const left, 
        const data_type* const right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        return ElementWiseBitsetPolicy<ElementT>::op_and_with_count(
            left, right, start_left, start_right, size
        );
    }

    static inline size_t op_or_with_count(
        data_type* const left, 
        const data_type* const right, 
        const size_t start_left,
        const size_t start_right, 
        const size_t size
    ) {
        return ElementWiseBitsetPolicy<ElementT>::op_or_with_count(
            left, right, start_left, start_right, size
        );
    }

    //
    template<typename FuncBaseline, FuncVectorized>
    static inline void op_func(
        const size_type start,
        const size_type size,
        FuncBaseline func_baseline,
        FuncVectorized func_vectorized
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
            func_baseline(start, size);
            return;
        }

        //
        uintptr_t ptr_offset = 0;

        // process the first element
        if (start_shift != 0) [[unlikely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            func_baseline(start, 0, size);

            // start from the next element
            start_element += 1;
            ptr_offset += data_bits - start_shift;
        }

        // process the middle
        {
            const size_t starting_bit_idx = start_element * data_bits;
            const size_t nbits = (end_element - start_element) * data_bits;

            // check if vectorized implementation is available
            if (!func_vectorized(start_element, ptr_offset, nbits)) {
                // vectorized implementation is not available, invoke the default one
                func_baseline(starting_bit_idx, ptr_offset, nbits);
            }
        
            //
            ptr_offset += nbits;
        }

        // process the last element
        if (end_shift != 0) [[likely]] {
            // it is possible to do vectorized masking here, but it is not worth it
            const size_t starting_bit_idx = end_element * data_bits; 

            func_baseline(starting_bit_idx, ptr_offset, end_shift);
        }
    }
};


}
}
}
