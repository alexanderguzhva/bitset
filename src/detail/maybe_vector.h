#pragma once

#include <array>
#include <memory>
#include <type_traits>

namespace milvus {
namespace bitset {
namespace detail {

// A structure that allocates an array of elements.
// No ownership is implied.
// If the number of elements is small, 
//     then an allocation will be done on the stack.
// If the number of elements is large, 
//     then an allocation will be done on the heap.
template<typename T>
struct MaybeVector {
public:
    static_assert(std::is_scalar_v<T>);
    
    MaybeVector(const size_t n_elements) {
        m_size = n_elements;

        if (n_elements < num_array_elements) {
            m_data = maybe_array.data();
        } else {
            maybe_memory = std::make_unique<T[]>(m_size);
            m_data = maybe_memory.get();
        }
    }

    MaybeVector(const MaybeVector&) = delete;
    MaybeVector(MaybeVector&&) = delete;
    MaybeVector& operator =(const MaybeVector&) = delete;
    MaybeVector& operator =(MaybeVector&&) = delete;

    inline size_t size() const { return m_size; }
    inline T* data() { return m_data; }
    inline const T* data() const { return m_data; }

    inline T& operator[](const size_t idx) { return m_data[idx]; }
    inline const T& operator[](const size_t idx) const { return m_data[idx]; }

private:
    size_t m_size = 0;

    T* m_data = nullptr;

    //
    static constexpr size_t num_array_elements = 64;

    std::unique_ptr<T[]> maybe_memory;
    std::array<T, num_array_elements> maybe_array;
};

}
}
}