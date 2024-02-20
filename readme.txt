An incomplete bitset library for Milvus (https://github.com/milvus-io/milvus).
The work is still in progress.

Clang-17 produces faster code for SIMD than GCC-9.

===========================================================================

Ideas for the code are borrowed from the following sources:
* Zach Wegner's https://github.com/zwegner/zp7
* Agner Fog's https://github.com/vectorclass/version2
* sse2neon, https://github.com/DLTcollab/sse2neon

Some possible future things to read / use:
* https://godbolt.org/z/CYipz7
* https://github.com/ridiculousfish/libdivide
* https://github.com/lemire/fastmod
