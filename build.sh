CC=clang-16 CXX=clang++-16 cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release .
cd build
ninja
./bitset
