Basically, a clone of a `boost::dynamic_bitset` library with views and SIMD (AVX2/AVX512/NEON/SVE).

This is an alpha version of library. The work is still in progress.

This library was written specifically for the Milvus project (https://github.com/milvus-io/milvus) and expected to be placed at https://github.com/milvus-io/milvus/tree/master/internal/core/src/bitset in the future release. 
Although, this library may be used as a general-purpose ibe. The version in this repo contains additional attributes like `[[likely]]` and may be used as a header-only library.

Tested on:
* Intel Xeon 4th Gen
* Intel i7-1250U laptop 
* AWS Graviton3 (SVE width 256)
* docker-qemu which emulates ARM (SVE width 512)

Clang produces faster code for SIMD than GCC for x86, but may produce slower code for ARM.

===========================================================================

Ideas for the code are borrowed from the following sources:
* Zach Wegner's https://github.com/zwegner/zp7
* Agner Fog's https://github.com/vectorclass/version2
* sse2neon, https://github.com/DLTcollab/sse2neon

Some possible future things to read / use:
* https://godbolt.org/z/CYipz7
* https://github.com/ridiculousfish/libdivide
* https://github.com/lemire/fastmod
