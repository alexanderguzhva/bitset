Basically, a clone of boost::dynamic_bitset with views and SIMD (AVX2/AVX512/NEON/SVE).

An early-alpha bitset library for Milvus (https://github.com/milvus-io/milvus).
The work is still in progress.

Tested on AWS Graviton3 (SVE width 256) and on docker-qemu (SVE width 512).

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
