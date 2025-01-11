#pragma once

#include <cstddef>

namespace milvus {
namespace bitset {
namespace detail {

template<typename PolicyT>
struct ConstProxy {
    using policy_type = PolicyT;
    using data_type = typename policy_type::data_type;
    using self_type = ConstProxy;

    const data_type& element;
    data_type mask;

    inline ConstProxy(const data_type& _element, const size_t _shift) : 
        element{_element}
    {
        mask = (data_type(1) << _shift);
    } 

    inline operator bool() const { return ((element & mask) != 0); }
    inline bool operator~() const { return ((element & mask) == 0); }
};

template<typename PolicyT>
struct Proxy {
    using policy_type = PolicyT;
    using data_type = typename policy_type::data_type;
    using self_type = Proxy;

    data_type& element;
    data_type mask;

    inline Proxy(data_type& _element, const size_t _shift) : 
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

}
}
}
