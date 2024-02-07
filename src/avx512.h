#pragma once

#include <cstddef>
#include <cstdint>

namespace milvus {
namespace bitset {

// all values for 'size' are counted in bytes.

void
AndAVX512(void* const left, const void* const right, const size_t size);

void
OrAVX512(void* const left, const void* const right, const size_t size);

template <typename T>
void
EqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
LessValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
GreaterValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
NotEqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
LessEqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
GreaterEqualValAVX512(const T* const __restrict src, const size_t size, const T val, void* const __restrict res);

template <typename T>
void
EqualColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
LessColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
LessEqualColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
GreaterColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
GreaterEqualColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

template <typename T>
void
NotEqualColumnAVX512(const T* const __restrict left, const T* const __restrict right, const size_t size, void* const __restrict res);

}
}
